"""ACTION6 click-target candidate generator.

Per `V3_PROMPT_REFERENCE_zh.md` §10 P1: when ACTION6 is legal, list
plausible (x, y) targets so the model picks something object-specific
instead of a hard-coded constant like (12,30) or grid-center (32,32).

Ranking heuristic (priority):
  1. Small unique objects (size <= 5)  -> likely buttons / switches
  2. ACTIVE objects whose color matches a STATIC object's color
                                         -> likely target slot match
  3. Centers of ACTIVE objects           -> general click target
  4. Centers of STATIC objects of small/medium size (< 100 cells)

Texture / background regions are EXCLUDED from candidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from arc_agent.object_extractor import ObjectRecord
from arc_agent.temporal_classifier import Layer


@dataclass
class ClickCandidate:
    x: int
    y: int
    why: str            # short rationale (1 line)
    source_uid: str = ""  # if from an active object

    def as_line(self) -> str:
        suid = f" ({self.source_uid})" if self.source_uid else ""
        return f"({self.x}, {self.y}){suid} — {self.why}"


def _center(o: ObjectRecord) -> tuple[int, int]:
    """Centroid rounded to integer cell coordinates.

    Note: the ARC scorecard expects (x, y) where x is the COLUMN and y
    is the ROW. ObjectRecord.center = (row, col), so we swap.
    """
    row, col = o.center
    return int(round(col)), int(round(row))


def list_click_candidates(
    frame_objects: list[ObjectRecord],
    layer_by_id: dict[int, Layer],
    *,
    max_candidates: int = 8,
) -> list[ClickCandidate]:
    """Return ranked candidates for ACTION6 (x, y) targets.

    `frame_objects` is the full extracted list this frame; `layer_by_id`
    is the temporal classifier output. TEXTURE objects are skipped.
    """
    active = [o for o in frame_objects if layer_by_id.get(o.id) == Layer.ACTIVE]
    static = [o for o in frame_objects if layer_by_id.get(o.id) == Layer.STATIC]
    candidates: list[ClickCandidate] = []

    # Track which (x, y) we've already added so we don't dup
    seen: set[tuple[int, int]] = set()

    def _add(c: ClickCandidate) -> None:
        key = (c.x, c.y)
        if key in seen:
            return
        seen.add(key)
        candidates.append(c)

    # 1) Small ACTIVE objects (likely interactive buttons)
    small_active = sorted([o for o in active if o.size <= 5],
                          key=lambda o: o.size)
    for o in small_active:
        x, y = _center(o)
        _add(ClickCandidate(
            x=x, y=y,
            why=f"small {o.color_name} object (size={o.size}) — likely button",
            source_uid=f"id={o.id}",
        ))

    # 2) Color-matching ACTIVE↔STATIC pairs (likely target slots)
    static_colors = {o.color for o in static if o.size < 200}
    for o in active:
        if o.color in static_colors and o.size <= 50:
            # Find matching static target
            matches = [s for s in static if s.color == o.color and s.size < 200]
            if matches:
                target = min(matches,
                             key=lambda s: abs(s.center[0] - o.center[0])
                                          + abs(s.center[1] - o.center[1]))
                tx, ty = _center(target)
                _add(ClickCandidate(
                    x=tx, y=ty,
                    why=(f"{target.color_name} target matching active "
                         f"object id={o.id}"),
                    source_uid=f"id={target.id}",
                ))

    # 3) Remaining ACTIVE centers
    for o in active:
        x, y = _center(o)
        _add(ClickCandidate(
            x=x, y=y,
            why=f"active {o.color_name} object id={o.id} center",
            source_uid=f"id={o.id}",
        ))

    # 4) Small/medium STATIC centers (not too big — skip walls/backgrounds)
    small_static = sorted([o for o in static if o.size <= 100],
                          key=lambda o: o.size)
    for o in small_static[:5]:
        x, y = _center(o)
        _add(ClickCandidate(
            x=x, y=y,
            why=f"small static {o.color_name} object (size={o.size})",
            source_uid=f"id={o.id}",
        ))

    return candidates[:max_candidates]


def pick_default_action6_coords(
    frame_objects: list[ObjectRecord],
    layer_by_id: dict[int, Layer],
    tried_coords: Iterable[tuple[int, int]] = (),
) -> tuple[int, int] | None:
    """When the model says 'ACTION6' without coords, pick a sensible default.

    Returns the first untried candidate (x, y). Returns None if no
    candidates are available — the caller should then random-fallback.
    """
    tried_set = set(tried_coords)
    for c in list_click_candidates(frame_objects, layer_by_id):
        if (c.x, c.y) not in tried_set:
            return (c.x, c.y)
    return None


def render_click_candidates_block(candidates: list[ClickCandidate]) -> str:
    """[CLICK CANDIDATES] block for the prompt."""
    if not candidates:
        return ("[CLICK CANDIDATES — ACTION6 targets]\n"
                "  (no recommended click targets — only large STATIC regions "
                "or empty cells visible)")
    lines = ["[CLICK CANDIDATES — ranked ACTION6 targets]"]
    for i, c in enumerate(candidates, 1):
        lines.append(f"  ({i}) {c.as_line()}")
    lines.append("  Prefer one of these when using ACTION6 instead of "
                 "guessing coordinates.")
    return "\n".join(lines)
