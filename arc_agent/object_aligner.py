"""Cross-frame object alignment via Hungarian algorithm (replaces Qwen-VL align).

Per `docs/OBJECT_PIPELINE_DESIGN_zh.md` §9.1:

  cost(a, b) = SHAPE_W * shape_mismatch
             + COLOR_W * color_mismatch
             + DIST_W  * centroid_distance
             + SIZE_W  * abs(size_diff)

`scipy.optimize.linear_sum_assignment` finds the min-cost 1-1 matching.
Pairs with cost above `NO_MATCH_COST` are split into `disappeared`
(BEFORE-side leftovers) and `appeared` (AFTER-side leftovers).

Output schema matches the Qwen-align JSON spec so downstream code can
swap freely.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

from arc_agent.object_extractor import ObjectRecord

# Cost weights — tuned so SHAPE > COLOR > DISTANCE > SIZE.
# A different shape is a much stronger signal than a moved blob.
SHAPE_MISMATCH_COST = 50.0
COLOR_MISMATCH_COST = 10.0
DISTANCE_WEIGHT = 1.0
SIZE_DIFF_WEIGHT = 5.0
NO_MATCH_COST = 500.0   # pairs above this are treated as "no match"


@dataclass
class Match:
    """One match in the alignment list."""

    before_id: Optional[int]
    after_id: Optional[int]
    type: str
    color: Optional[int] = None
    delta: Optional[dict] = None

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _pair_cost(a: ObjectRecord, b: ObjectRecord) -> float:
    """Cost of treating a (BEFORE) and b (AFTER) as the same object."""
    cost = 0.0
    if a.shape_signature != b.shape_signature:
        cost += SHAPE_MISMATCH_COST
    if a.color != b.color:
        cost += COLOR_MISMATCH_COST
    dr = a.center[0] - b.center[0]
    dc = a.center[1] - b.center[1]
    cost += DISTANCE_WEIGHT * float(np.hypot(dr, dc))
    cost += SIZE_DIFF_WEIGHT * abs(a.size - b.size)
    return cost


def _classify(a: ObjectRecord, b: ObjectRecord) -> tuple[str, Optional[dict]]:
    """Decide match_type + delta for a paired (a, b)."""
    same_color = a.color == b.color
    same_shape = a.shape_signature == b.shape_signature
    dy = int(round(b.center[0] - a.center[0]))
    dx = int(round(b.center[1] - a.center[1]))
    same_position = (dy == 0 and dx == 0)

    if same_color and same_shape and same_position:
        return "unchanged", None
    if same_color and same_shape and not same_position:
        return "moved", {"dy": dy, "dx": dx}
    if same_shape and same_position and not same_color:
        return "recolored", {"from": a.color, "to": b.color}
    if same_color and not same_shape:
        cells_added = max(0, b.size - a.size)
        cells_removed = max(0, a.size - b.size)
        return "reshaped", {"cells_added": cells_added,
                            "cells_removed": cells_removed}
    # Fallback: treat as "moved" if mostly position-only, else reshaped
    if same_color:
        return "moved", {"dy": dy, "dx": dx}
    return "reshaped", {"cells_added": max(0, b.size - a.size),
                        "cells_removed": max(0, a.size - b.size)}


def align_objects(before: list[ObjectRecord],
                  after: list[ObjectRecord]) -> list[Match]:
    """Min-cost 1-1 matching + disappeared / appeared for leftovers.

    Returns matches in this order: paired matches first (sorted by
    before_id), then disappeared, then appeared.
    """
    from scipy.optimize import linear_sum_assignment

    nA, nB = len(before), len(after)
    matches: list[Match] = []

    if nA == 0 and nB == 0:
        return matches

    if nA == 0:
        return [Match(before_id=None, after_id=b.id, type="appeared",
                      color=b.color, delta=None) for b in after]
    if nB == 0:
        return [Match(before_id=a.id, after_id=None, type="disappeared",
                      color=a.color, delta=None) for a in before]

    cost = np.full((nA, nB), NO_MATCH_COST + 1.0, dtype=float)
    for i, a in enumerate(before):
        for j, b in enumerate(after):
            cost[i, j] = _pair_cost(a, b)

    # Pad to square if non-square; we'll filter NO_MATCH_COST pairs later.
    n = max(nA, nB)
    pad_cost = np.full((n, n), NO_MATCH_COST + 1.0, dtype=float)
    pad_cost[:nA, :nB] = cost
    row_idx, col_idx = linear_sum_assignment(pad_cost)

    matched_a: set[int] = set()
    matched_b: set[int] = set()
    paired: list[Match] = []

    for i, j in zip(row_idx, col_idx):
        if i >= nA or j >= nB:
            continue
        if cost[i, j] > NO_MATCH_COST:
            continue
        a = before[i]
        b = after[j]
        match_type, delta = _classify(a, b)
        # Color field: prefer common color, or the before color if changed
        color = a.color
        paired.append(Match(before_id=a.id, after_id=b.id,
                            type=match_type, color=color, delta=delta))
        matched_a.add(i)
        matched_b.add(j)

    paired.sort(key=lambda m: m.before_id if m.before_id is not None else -1)
    matches.extend(paired)

    for i, a in enumerate(before):
        if i not in matched_a:
            matches.append(Match(before_id=a.id, after_id=None,
                                 type="disappeared", color=a.color, delta=None))
    for j, b in enumerate(after):
        if j not in matched_b:
            matches.append(Match(before_id=None, after_id=b.id,
                                 type="appeared", color=b.color, delta=None))
    return matches


def matches_to_dict(matches: list[Match]) -> dict[str, Any]:
    """JSON-friendly wrapper matching Qwen align output shape."""
    return {"matches": [m.as_dict() for m in matches]}
