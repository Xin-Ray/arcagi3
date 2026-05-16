"""Classify candidate objects into STATIC / ACTIVE / TEXTURE / CANDIDATE
based on temporal evidence across multiple frames.

This is the per-frame filter that the user proposed (ARCHITECTURE_v3 §1):
"don't pre-define what is an object — instead, use cross-frame statistics
to find what is stable background vs what genuinely changes."

Decisions per candidate object (rules in priority order):

  TEXTURE   — many same-color same-shape tiny candidates (size <= TEXTURE_SIZE,
              count >= TEXTURE_COUNT). Treat as background pattern.
  STATIC    — observed in last N frames with same position, color, shape.
  ACTIVE    — at least one frame where position OR shape OR color changed.
  CANDIDATE — fewer than N frames of observation, not yet classified.

Pure function over frame snapshots; no LLM dependency.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any

from arc_agent.object_extractor import ObjectRecord

MIN_FRAMES_TO_CLASSIFY = 3
TEXTURE_SIZE_MAX = 2
TEXTURE_COUNT_MIN = 20


class Layer(str, Enum):
    STATIC = "STATIC"
    ACTIVE = "ACTIVE"
    TEXTURE = "TEXTURE"
    CANDIDATE = "CANDIDATE"


@dataclass(frozen=True)
class ObjectSignature:
    """A lightweight identity for an object across frames.

    Two ObjectRecord with the same signature AND same bbox-anchor cell
    are treated as the same object across frames. Signature = (color,
    shape_signature). Bbox anchor handles two objects with same shape +
    color but different positions.
    """
    color: int
    shape_sig: tuple[tuple[int, int], ...]
    anchor: tuple[int, int]   # bbox top-left

    @classmethod
    def of(cls, o: ObjectRecord) -> "ObjectSignature":
        return cls(color=o.color, shape_sig=o.shape_signature,
                   anchor=(o.bbox[0], o.bbox[1]))


def is_likely_texture(candidate: ObjectRecord,
                      all_candidates: list[ObjectRecord]) -> bool:
    """Mark tiny same-color same-shape repetitions as texture.

    Example: bp35 has 191 small green cells — none of them are gameplay
    objects, they're patterned background.
    """
    if candidate.size > TEXTURE_SIZE_MAX:
        return False
    siblings = [
        c for c in all_candidates
        if c.color == candidate.color
        and c.size == candidate.size
        and c.shape_signature == candidate.shape_signature
    ]
    return len(siblings) >= TEXTURE_COUNT_MIN


def classify_frame(
    frame_objects: list[ObjectRecord],
    history_per_signature: dict[tuple[int, tuple[tuple[int, int], ...]], list[tuple[int, int]]],
    *,
    min_frames: int = MIN_FRAMES_TO_CLASSIFY,
) -> dict[int, Layer]:
    """Return {object_id: Layer} for every object in `frame_objects`.

    `history_per_signature` is a dict that the caller maintains across
    frames: key = (color, shape_signature), value = list of (anchor_row,
    anchor_col) observed across past frames. The classifier reads this
    history but does not mutate it (caller does the mutation).

    Returns one Layer label per object id.
    """
    # Step 1: texture detection over the current frame
    texture_ids: set[int] = set()
    for o in frame_objects:
        if is_likely_texture(o, frame_objects):
            texture_ids.add(o.id)

    out: dict[int, Layer] = {}
    for o in frame_objects:
        if o.id in texture_ids:
            out[o.id] = Layer.TEXTURE
            continue
        key = (o.color, o.shape_signature)
        anchors_seen = history_per_signature.get(key, [])
        # `anchors_seen` includes the CURRENT frame's anchor at the end
        # (the caller is expected to append before calling classify_frame).
        if len(anchors_seen) < min_frames:
            out[o.id] = Layer.CANDIDATE
            continue
        # Last min_frames anchors all same as current?
        recent = anchors_seen[-min_frames:]
        current_anchor = (o.bbox[0], o.bbox[1])
        if all(a == current_anchor for a in recent):
            out[o.id] = Layer.STATIC
        else:
            out[o.id] = Layer.ACTIVE
    return out


def update_history(
    history_per_signature: dict[tuple[int, tuple[tuple[int, int], ...]], list[tuple[int, int]]],
    frame_objects: list[ObjectRecord],
) -> None:
    """Append the current frame's anchor positions to history.

    Caller invokes this each step (BEFORE classify_frame) so the
    classifier sees the current frame's data. Mutation in place.

    For each (color, shape_signature) seen this frame, we append all
    anchors. Multiple instances of the same signature → multiple
    appends. This is intentional: it lets us count how often an "L-shape
    red" appears overall, which matters for texture detection across
    frames as well.
    """
    for o in frame_objects:
        key = (o.color, o.shape_signature)
        history_per_signature.setdefault(key, []).append((o.bbox[0], o.bbox[1]))


def filter_active(frame_objects: list[ObjectRecord],
                  layer_by_id: dict[int, Layer]) -> list[ObjectRecord]:
    """Convenience: pluck the objects classified as ACTIVE."""
    return [o for o in frame_objects if layer_by_id.get(o.id) == Layer.ACTIVE]


def filter_non_texture(frame_objects: list[ObjectRecord],
                       layer_by_id: dict[int, Layer]) -> list[ObjectRecord]:
    """Plus CANDIDATE / STATIC — everything except TEXTURE."""
    return [o for o in frame_objects
            if layer_by_id.get(o.id) != Layer.TEXTURE]


def texture_summary(frame_objects: list[ObjectRecord],
                    layer_by_id: dict[int, Layer]) -> dict[str, Any]:
    """Aggregate the TEXTURE-tagged cells into a one-line summary.

    Returns:
      {"texture_cells_total": int, "by_color": {color_name: count, ...}}
    """
    by_color: Counter = Counter()
    total = 0
    for o in frame_objects:
        if layer_by_id.get(o.id) == Layer.TEXTURE:
            by_color[o.color_name] += o.size
            total += o.size
    return {"texture_cells_total": total, "by_color": dict(by_color)}
