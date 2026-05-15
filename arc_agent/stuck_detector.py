"""Stuck detection with progress-bar / counter masking.

Background. The v3 / v3.2 stuck signals use `hash(grid.tobytes())` over the
raw 64x64 grid. That breaks down when the game has a deterministically-
animating UI element (a step counter, a progress bar, a timer) -- two
semantically-identical states get different hashes because the counter
changed, so `state_revisit_count` stays at 1 forever even when the agent
is in a loop.

This module:
  - `compute_noise_mask(frames, min_unique)`       deterministic mask of
       pixels that take too many distinct values across the window.
  - `masked_grid_hash(grid, mask)`                  hash with masked pixels
       zeroed -- safe to compare across frames.
  - `detect_repeat_stuck(hashes, window, min_repeats)` flag when a masked
       hash appears N+ times in the recent window.

All pure functions; no I/O, no globals. The orchestrator (run_v3_multi_round)
owns the rolling history and feeds it in.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

# Need at least this many frames in the window before computing a mask.
# With fewer than 4 frames almost every pixel looks "stable" and we'd mask
# nothing (or noise nothing); either way the mask isn't useful yet.
MIN_FRAMES_FOR_MASK = 4

# Default: a pixel is "noisy" if it takes >= 3 distinct values in the
# history window. Real game state in ARC-AGI-3 rarely flips a single cell
# through 3+ colors in 10 moves; a step counter or progress bar will.
DEFAULT_MIN_UNIQUE_THRESHOLD = 3

# Default stuck window + repeat thresholds. Per user spec (2026-05-14):
# "if the past 10 frames have a repeating frame appearing several times,
# it's stuck." Three+ visits in the last 10 is the trigger.
DEFAULT_STUCK_WINDOW = 10
DEFAULT_STUCK_MIN_REPEATS = 3


def compute_noise_mask(
    frames: list[np.ndarray],
    min_unique_threshold: int = DEFAULT_MIN_UNIQUE_THRESHOLD,
) -> np.ndarray:
    """Return a boolean (H, W) mask: True = pixel changes too often to be
    meaningful game state (likely a progress bar / counter).

    Algorithm: for each pixel, count the number of distinct values taken
    across `frames`. Pixels with >= `min_unique_threshold` distinct values
    are flagged. A static cell yields unique_count = 1. A cell that flips
    between two game states yields 2 (kept). A counter that cycles through
    3+ values is masked.

    Returns an all-False mask when fewer than MIN_FRAMES_FOR_MASK frames
    are provided -- callers should treat that as "no masking yet".
    """
    if not frames:
        raise ValueError("compute_noise_mask: frames is empty")
    h, w = frames[0].shape
    if len(frames) < MIN_FRAMES_FOR_MASK:
        return np.zeros((h, w), dtype=bool)
    arr = np.stack(frames)               # (N, H, W)
    # Vectorized unique-count per pixel: sort along the frame axis, count
    # positions where adjacent sorted values differ -> distinct-values - 1.
    sorted_arr = np.sort(arr, axis=0)
    diffs = sorted_arr[1:] != sorted_arr[:-1]
    unique_count = 1 + diffs.sum(axis=0)
    return unique_count >= min_unique_threshold


def masked_grid_hash(
    grid: np.ndarray,
    noise_mask: Optional[np.ndarray] = None,
) -> int:
    """Hash a grid after zeroing out noise pixels.

    If `noise_mask` is None or all-False, this is equivalent to
    `hash(grid.tobytes())` -- so callers can use it unconditionally.
    """
    if noise_mask is None or not noise_mask.any():
        return hash(grid.tobytes())
    masked = grid.copy()
    masked[noise_mask] = 0
    return hash(masked.tobytes())


def detect_repeat_stuck(
    masked_hashes: list[int],
    *,
    window: int = DEFAULT_STUCK_WINDOW,
    min_repeats: int = DEFAULT_STUCK_MIN_REPEATS,
) -> tuple[bool, str, int]:
    """True if some masked hash appears >= min_repeats times in the last
    `window` hashes.

    Returns `(is_stuck, reason, repeat_count)`. `reason` is empty when
    not stuck; otherwise it's a one-line message suitable for the
    Reflection alert or the orchestrator stuck-alert block.
    """
    if len(masked_hashes) < min_repeats:
        return False, "", 0
    recent = masked_hashes[-window:]
    counts = Counter(recent)
    _hash, count = counts.most_common(1)[0]
    if count >= min_repeats:
        return (
            True,
            (f"Same state (progress-bar / counter excluded) seen {count}x "
             f"in the last {len(recent)} steps -- you are in a loop. "
             f"Try an untried action or interact with an unexplored object."),
            count,
        )
    return False, "", count


# Default thresholds for the orchestrator-side stuck alert.
DEFAULT_NO_OP_STREAK_THRESHOLD = 5
DEFAULT_STATE_REVISIT_THRESHOLD = 5


def compute_orchestrator_alert(
    *,
    matches_reasoning: Optional[str] = None,
    masked_stuck_reason: Optional[str] = None,
    no_op_streak: int = 0,
    state_revisit: int = 0,
    last_picks: Optional[list[str]] = None,
    no_op_streak_threshold: int = DEFAULT_NO_OP_STREAK_THRESHOLD,
    state_revisit_threshold: int = DEFAULT_STATE_REVISIT_THRESHOLD,
) -> str:
    """Single alert channel: returns ONE alert string for the next-step
    Action prompt, chosen by priority. Empty string when nothing fires.

    Priority (highest -> lowest):
      1. matches_reasoning == "NO"         (Action Agent's mental model was wrong)
      2. masked-hash repeat                (deterministic loop with counter-aware hash)
      3. no_op_streak >= threshold         (game ignoring the agent)
      4. state_revisit >= threshold        (raw-hash loop, fallback)

    This deliberately replaces the ad-hoc `_build_stuck_alert` in
    run_v3_multi_round + Reflection's current_alert field. Reflection can
    still propose an alert via delta, but the orchestrator overwrites
    when ANY of the above fires, so the agent always gets the strongest
    deterministic signal.
    """
    last_picks = last_picks or []

    # P1: reasoning mismatch
    if matches_reasoning == "NO":
        culprit = last_picks[-1] if last_picks else "the last action"
        return (
            f"Your reasoning predicted an effect but the outcome contradicted "
            f"it ({culprit}). Revise your model of what this action does -- "
            f"do NOT repeat the same prediction next step."
        )

    # P2: masked-hash loop -- strongest deterministic loop signal
    if masked_stuck_reason:
        return (
            f"{masked_stuck_reason} "
            "Abandon the current goal_hypothesis -- it isn't working. "
            "Try a different action category OR a different obj_id."
        )

    # P3: raw-signal stuck (no_op_streak / state_revisit)
    if (no_op_streak >= no_op_streak_threshold
            or state_revisit >= state_revisit_threshold):
        if last_picks:
            from collections import Counter
            counts = Counter(last_picks)
            culprit, n = counts.most_common(1)[0]
            others = ", ".join(sorted(set(f"ACTION{i}" for i in range(1, 8))
                                      - {culprit}))
            return (
                f"Stuck for {max(no_op_streak, state_revisit)} steps; "
                f"last picks were mostly {culprit} ({n} times). "
                f"Pick a non-{culprit} action like one of: {others}."
            )
        return (
            f"Stuck: state_revisit={state_revisit} no_op_streak={no_op_streak}. "
            "Try a different action category."
        )

    return ""


__all__ = [
    "DEFAULT_MIN_UNIQUE_THRESHOLD",
    "DEFAULT_NO_OP_STREAK_THRESHOLD",
    "DEFAULT_STATE_REVISIT_THRESHOLD",
    "DEFAULT_STUCK_MIN_REPEATS",
    "DEFAULT_STUCK_WINDOW",
    "MIN_FRAMES_FOR_MASK",
    "compute_noise_mask",
    "compute_orchestrator_alert",
    "detect_repeat_stuck",
    "masked_grid_hash",
]
