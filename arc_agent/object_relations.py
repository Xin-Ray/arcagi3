"""Object-relations layer for the v3 perception pipeline.

Per user feedback after the v3 ar25 5x80 run (2026-05-14): the Action Agent
needs to see object groupings + spatial relationships, not just individual
bboxes. Adds a deterministic, no-LLM layer between `extract_objects /
temporal_classifier` and the prompt builder.

Computes for each frame:
  - same_color_groups   color_name -> [obj.id, ...]   (>=2 members)
  - same_shape_groups   shape_key  -> [obj.id, ...]   (>=2 members)
  - closest_pairs       top N (by center distance) (obj_a, obj_b, dist)
  - edge_distances      obj.id -> dict of {top, bottom, left, right} cell counts
  - quadrant            obj.id -> "top-left" / "top-right" / "bottom-left" / "bottom-right" / "center"

`render_relations_block` turns the relations dict into a `[OBJECT RELATIONS]`
text block consumed by `prompts_v3.build_play_user_prompt`. Pure function;
no I/O, no LLM.

This module is intentionally separate from `object_extractor` so it can be
swapped or skipped (e.g. on Kaggle latency-constrained runs).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from arc_agent.object_extractor import ObjectRecord
from arc_agent.temporal_classifier import Layer


# Cap how many groups / pairs we list in the prompt to keep token budget bounded.
_MAX_GROUPS_PER_KIND = 5
_MAX_CLOSEST_PAIRS = 5
_MIN_GROUP_SIZE = 2


@dataclass
class ObjectRelations:
    """Pre-computed relations among objects in one frame. Read by the
    prompt builder; never mutated."""

    grid_shape: tuple[int, int] = (64, 64)
    same_color_groups: dict[str, list[int]] = field(default_factory=dict)
    same_shape_groups: dict[str, list[int]] = field(default_factory=dict)
    closest_pairs: list[tuple[int, int, float]] = field(default_factory=list)
    edge_distances: dict[int, dict[str, int]] = field(default_factory=dict)
    quadrant: dict[int, str] = field(default_factory=dict)

    # Quick "is this object x interesting" lookup
    active_ids: set[int] = field(default_factory=set)


def _shape_key(obj: ObjectRecord) -> str:
    """Human-readable shape key for grouping. Uses bbox dims + cell count
    so '2x2 solid' and '2x3 hollow' don't collide."""
    r0, c0, r1, c1 = obj.bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return f"{h}x{w}_size{obj.size}"


def _quadrant_of(center: tuple[float, float],
                 grid_shape: tuple[int, int]) -> str:
    """Bin a center into a 3x3 region label (top-left ... center ... bottom-right).

    Border bands ±20% from center are "center" / "top" / "bottom" / "left" /
    "right" only. The 4 corners are full 2-word labels.
    """
    H, W = grid_shape
    r, c = center
    # third boundaries (1/3 and 2/3)
    r_band = "top" if r < H / 3 else ("bottom" if r > 2 * H / 3 else "middle")
    c_band = "left" if c < W / 3 else ("right" if c > 2 * W / 3 else "middle")
    if r_band == "middle" and c_band == "middle":
        return "center"
    if r_band == "middle":
        return c_band
    if c_band == "middle":
        return r_band
    return f"{r_band}-{c_band}"


def _edge_distances(obj: ObjectRecord,
                    grid_shape: tuple[int, int]) -> dict[str, int]:
    """How many cells of clearance from each grid edge."""
    H, W = grid_shape
    r0, c0, r1, c1 = obj.bbox
    return {
        "top": r0,
        "bottom": H - 1 - r1,
        "left": c0,
        "right": W - 1 - c1,
    }


def _center_distance(a: ObjectRecord, b: ObjectRecord) -> float:
    """Euclidean distance between object centers."""
    dr = a.center[0] - b.center[0]
    dc = a.center[1] - b.center[1]
    return math.sqrt(dr * dr + dc * dc)


def compute_relations(
    objects: Iterable[ObjectRecord],
    *,
    grid_shape: tuple[int, int] = (64, 64),
    layer_by_id: Optional[dict[int, Layer]] = None,
    skip_texture: bool = True,
) -> ObjectRelations:
    """Build an `ObjectRelations` snapshot for one frame.

    Args:
        objects: extracted ObjectRecords (typically from `extract_objects`).
        grid_shape: (H, W) of the source grid.
        layer_by_id: optional STATIC/ACTIVE/TEXTURE classification; if given
            and skip_texture=True, TEXTURE objects are dropped before grouping.
        skip_texture: whether to drop TEXTURE-classified objects (default True).
    """
    objs = [o for o in objects]
    if layer_by_id is not None and skip_texture:
        objs = [o for o in objs if layer_by_id.get(o.id) is not Layer.TEXTURE]

    same_color: dict[str, list[int]] = {}
    same_shape: dict[str, list[int]] = {}
    edge_dist: dict[int, dict[str, int]] = {}
    quadrants: dict[int, str] = {}
    active_ids: set[int] = set()

    for obj in objs:
        same_color.setdefault(obj.color_name, []).append(obj.id)
        same_shape.setdefault(_shape_key(obj), []).append(obj.id)
        edge_dist[obj.id] = _edge_distances(obj, grid_shape)
        quadrants[obj.id] = _quadrant_of(obj.center, grid_shape)
        if layer_by_id is not None and layer_by_id.get(obj.id) is Layer.ACTIVE:
            active_ids.add(obj.id)

    # Keep only multi-member groups
    same_color = {k: ids for k, ids in same_color.items()
                  if len(ids) >= _MIN_GROUP_SIZE}
    same_shape = {k: ids for k, ids in same_shape.items()
                  if len(ids) >= _MIN_GROUP_SIZE}

    # Closest pairs by center distance. O(N^2) -- frame typically has <50 obj.
    closest: list[tuple[int, int, float]] = []
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            closest.append((a.id, b.id, _center_distance(a, b)))
    closest.sort(key=lambda t: t[2])
    closest = closest[:_MAX_CLOSEST_PAIRS]

    return ObjectRelations(
        grid_shape=grid_shape,
        same_color_groups=same_color,
        same_shape_groups=same_shape,
        closest_pairs=closest,
        edge_distances=edge_dist,
        quadrant=quadrants,
        active_ids=active_ids,
    )


# ── rendering ────────────────────────────────────────────────────────────


def render_relations_block(
    relations: Optional[ObjectRelations],
    *,
    focus_ids: Optional[set[int]] = None,
) -> str:
    """Turn ObjectRelations into a `[OBJECT RELATIONS]` prompt block.

    If `focus_ids` is provided, only those object ids are included in the
    edge_distances / quadrant sections (typically the ACTIVE set, to keep
    the prompt small).
    """
    if relations is None:
        return "[OBJECT RELATIONS]\n  (no data)"

    lines: list[str] = []

    # Same color groups
    if relations.same_color_groups:
        lines.append("  same-color groups (color: object ids):")
        for color, ids in list(relations.same_color_groups.items())[:_MAX_GROUPS_PER_KIND]:
            ids_str = ", ".join(f"#{i}" for i in sorted(ids))
            lines.append(f"    {color}: {ids_str}")
    else:
        lines.append("  same-color groups: (none -- every visible color is unique)")

    # Same shape groups
    if relations.same_shape_groups:
        lines.append("  same-shape groups (shape: object ids):")
        for shape_key, ids in list(relations.same_shape_groups.items())[:_MAX_GROUPS_PER_KIND]:
            ids_str = ", ".join(f"#{i}" for i in sorted(ids))
            lines.append(f"    {shape_key}: {ids_str}")

    # Closest pairs
    if relations.closest_pairs:
        lines.append("  closest object pairs (by center distance):")
        for a_id, b_id, dist in relations.closest_pairs:
            lines.append(f"    #{a_id} <-> #{b_id}: {dist:.1f} cells")

    # Edge distances + quadrant for focus objects (or all if focus_ids None)
    targets = relations.active_ids if focus_ids is None else focus_ids
    targets = targets & set(relations.edge_distances.keys()) if targets else set()
    if targets:
        lines.append("  active object placement:")
        for obj_id in sorted(targets):
            ed = relations.edge_distances.get(obj_id, {})
            q = relations.quadrant.get(obj_id, "?")
            lines.append(
                f"    #{obj_id} in {q}; clearance "
                f"top={ed.get('top', '?')} bottom={ed.get('bottom', '?')} "
                f"left={ed.get('left', '?')} right={ed.get('right', '?')}"
            )

    if not lines:
        return "[OBJECT RELATIONS]\n  (no relations -- only one tracked object)"
    return "[OBJECT RELATIONS]\n" + "\n".join(lines)


__all__ = [
    "ObjectRelations",
    "compute_relations",
    "render_relations_block",
]
