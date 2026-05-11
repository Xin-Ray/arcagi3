"""Verifier-based intrinsic reward primitives for RL training.

Core idea (see vlm_test/README.md §3): each step the agent predicts which
cells will change. We compare the prediction set against the real change set
extracted from (s_t, s_{t+1}) and score it with F1. That F1 becomes a dense
reward signal — every step produces feedback, no need to wait for WIN.

Three primitives:
- changes_to_set:   normalise Qwen's JSON output to a comparison set
- real_changes:     extract the ground-truth change set from two consecutive grids
- verify_prediction_f1: F1 between predicted and real change sets ∈ [0, 1]
"""
from __future__ import annotations

from typing import Any

import numpy as np

ChangeSet = set[tuple[int, int, int]]
"""(row, col, new_color) — set so order-insensitive comparison is trivial."""


def changes_to_set(changes: Any) -> ChangeSet:
    """Convert Qwen's `predicted_changes` list to a ChangeSet.

    Accepts anything; silently drops malformed entries. Designed to never raise
    on bad model output — reward should reflect quality, not crash the loop.

    Valid entry shape: {"row": int, "col": int, "to_color": int}.
    """
    out: ChangeSet = set()
    if not isinstance(changes, list):
        return out
    for c in changes:
        if not isinstance(c, dict):
            continue
        try:
            out.add((int(c["row"]), int(c["col"]), int(c["to_color"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def real_changes(s_t: np.ndarray, s_tp1: np.ndarray) -> ChangeSet:
    """Extract the ground-truth (row, col, new_color) set from two frames.

    Two grids of identical shape are required; cells that differ contribute one
    entry with `new_color = s_tp1[r, c]`.
    """
    if s_t.shape != s_tp1.shape:
        raise ValueError(f"shape mismatch: {s_t.shape} vs {s_tp1.shape}")
    rows, cols = np.where(s_t != s_tp1)
    return {(int(r), int(c), int(s_tp1[r, c])) for r, c in zip(rows.tolist(), cols.tolist())}


def verify_prediction_f1(predicted: ChangeSet, real: ChangeSet) -> float:
    """F1 score between predicted and real change sets, in [0, 1].

    Edge cases follow the convention:
    - Both empty (correctly predicted "nothing changes") → 1.0
    - One empty, the other not                         → 0.0
    - No intersection                                  → 0.0
    - Otherwise standard F1: 2·P·R / (P+R)
    """
    if not predicted and not real:
        return 1.0
    if not predicted or not real:
        return 0.0
    tp = len(predicted & real)
    if tp == 0:
        return 0.0
    p = tp / len(predicted)
    r = tp / len(real)
    return 2 * p * r / (p + r)
