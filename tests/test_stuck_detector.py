"""Tests for arc_agent.stuck_detector.

Covers:
  - compute_noise_mask cold start / single-row noise / fully-static
  - masked_grid_hash equals raw hash when no noise
  - masked_grid_hash collides for grids that differ only in noisy pixels
  - detect_repeat_stuck threshold semantics
"""
from __future__ import annotations

import numpy as np
import pytest

from arc_agent.stuck_detector import (
    MIN_FRAMES_FOR_MASK,
    compute_noise_mask,
    detect_repeat_stuck,
    masked_grid_hash,
)


def _grid(values: int | list[list[int]], shape: tuple[int, int] = (8, 8)) -> np.ndarray:
    if isinstance(values, int):
        return np.full(shape, values, dtype=np.int32)
    return np.array(values, dtype=np.int32)


# ── compute_noise_mask ──────────────────────────────────────────────────


def test_noise_mask_cold_start_returns_all_false() -> None:
    """Fewer than MIN_FRAMES_FOR_MASK frames -> no masking yet."""
    frames = [_grid(0) for _ in range(MIN_FRAMES_FOR_MASK - 1)]
    mask = compute_noise_mask(frames)
    assert mask.shape == frames[0].shape
    assert not mask.any()


def test_noise_mask_all_static_returns_all_false() -> None:
    """All identical frames -> nothing is noisy."""
    frames = [_grid(5) for _ in range(10)]
    mask = compute_noise_mask(frames)
    assert not mask.any()


def test_noise_mask_catches_a_noisy_row() -> None:
    """One row that cycles through 4 colors should be masked; others not."""
    frames = []
    for i in range(8):
        g = _grid(0)
        g[3, :] = (i % 4) + 1   # row 3 cycles through colors 1..4
        frames.append(g)
    mask = compute_noise_mask(frames)
    # Row 3 fully masked
    assert mask[3, :].all()
    # Other rows untouched
    other = np.delete(mask, 3, axis=0)
    assert not other.any()


def test_noise_mask_keeps_two_state_pixels() -> None:
    """A pixel that flips between exactly 2 values is NOT noise (default
    threshold = 3 distinct values)."""
    frames = []
    for i in range(10):
        g = _grid(0)
        g[2, 2] = 1 if i % 2 == 0 else 2     # 2-state cell
        g[5, 5] = (i % 5) + 1                # 5-state cell (counter)
        frames.append(g)
    mask = compute_noise_mask(frames)
    assert mask[2, 2] is np.bool_(False) or not mask[2, 2]   # NOT masked
    assert mask[5, 5]                                         # IS masked


def test_noise_mask_threshold_is_tunable() -> None:
    """Lower the threshold; the 2-state cell now gets masked too."""
    frames = []
    for i in range(10):
        g = _grid(0)
        g[2, 2] = 1 if i % 2 == 0 else 2
        frames.append(g)
    mask = compute_noise_mask(frames, min_unique_threshold=2)
    assert mask[2, 2]


def test_noise_mask_raises_on_empty_input() -> None:
    with pytest.raises(ValueError):
        compute_noise_mask([])


# ── masked_grid_hash ────────────────────────────────────────────────────


def test_masked_hash_equals_raw_hash_when_no_mask() -> None:
    g = _grid(0)
    g[1, 1] = 7
    assert masked_grid_hash(g, None) == hash(g.tobytes())
    assert masked_grid_hash(g, np.zeros_like(g, dtype=bool)) == hash(g.tobytes())


def test_masked_hash_collides_when_only_noisy_pixels_differ() -> None:
    """Two grids that differ ONLY in pixels covered by the mask should
    produce the same masked hash."""
    g1 = _grid(0)
    g2 = _grid(0)
    g1[7, 7] = 3   # only this pixel differs
    g2[7, 7] = 5
    mask = np.zeros_like(g1, dtype=bool)
    mask[7, 7] = True
    assert masked_grid_hash(g1, mask) == masked_grid_hash(g2, mask)
    # Sanity: without mask they'd differ
    assert hash(g1.tobytes()) != hash(g2.tobytes())


def test_masked_hash_distinguishes_unmasked_differences() -> None:
    g1 = _grid(0)
    g2 = _grid(0)
    g1[7, 7] = 3
    g2[7, 7] = 5
    mask = np.zeros_like(g1, dtype=bool)
    mask[0, 0] = True   # mask irrelevant pixel
    assert masked_grid_hash(g1, mask) != masked_grid_hash(g2, mask)


# ── detect_repeat_stuck ─────────────────────────────────────────────────


def test_repeat_stuck_empty_history() -> None:
    is_stuck, reason, count = detect_repeat_stuck([])
    assert is_stuck is False
    assert reason == ""
    assert count == 0


def test_repeat_stuck_below_threshold() -> None:
    # 10 distinct hashes -> max repeat count = 1
    hashes = list(range(10))
    is_stuck, _, count = detect_repeat_stuck(hashes)
    assert is_stuck is False
    assert count == 1


def test_repeat_stuck_threshold_3_hits() -> None:
    # A,B,A,B,C,A,B,C,A,D -> hash A appears 4 times
    hashes = [1, 2, 1, 2, 3, 1, 2, 3, 1, 4]
    is_stuck, reason, count = detect_repeat_stuck(hashes)
    assert is_stuck is True
    assert count == 4
    assert "loop" in reason.lower() or "stuck" in reason.lower()


def test_repeat_stuck_only_looks_at_window() -> None:
    """Earlier-history repeats should NOT trigger stuck once they fall out
    of the window."""
    # 5 repeats of hash=1 EARLY, then 10 distinct hashes
    hashes = [1]*5 + list(range(100, 115))   # window=10 -> recent = last 10
    is_stuck, _, count = detect_repeat_stuck(hashes, window=10)
    assert is_stuck is False
    assert count == 1


def test_repeat_stuck_min_repeats_tunable() -> None:
    # 2 repeats only -> default threshold (3) misses, threshold=2 catches
    hashes = [1, 1, 2, 3, 4]
    assert detect_repeat_stuck(hashes, min_repeats=3)[0] is False
    is_stuck, _, count = detect_repeat_stuck(hashes, min_repeats=2)
    assert is_stuck is True
    assert count == 2


# ── compute_orchestrator_alert ─────────────────────────────────────────


from arc_agent.stuck_detector import compute_orchestrator_alert


def test_orchestrator_alert_empty_when_no_signal() -> None:
    assert compute_orchestrator_alert(
        matches_reasoning="YES",
        masked_stuck_reason="",
        no_op_streak=0,
        state_revisit=0,
    ) == ""


def test_orchestrator_alert_reasoning_mismatch_wins() -> None:
    """matches_reasoning=NO is the highest-priority alert -- even when
    other stuck signals fire, the mismatch message takes precedence."""
    out = compute_orchestrator_alert(
        matches_reasoning="NO",
        masked_stuck_reason="loop detected",
        no_op_streak=20,
        state_revisit=20,
        last_picks=["ACTION1", "ACTION1", "ACTION1"],
    )
    low = out.lower()
    assert "reasoning" in low
    assert "contradict" in low or "wrong" in low
    # Should NOT also include the masked-stuck phrasing
    assert "loop detected" not in out


def test_orchestrator_alert_masked_stuck_beats_raw_signals() -> None:
    """Masked-hash repeat (counter-aware) ranks above no_op_streak so the
    LLM sees the goal-abandonment hint before the action-rotation hint."""
    out = compute_orchestrator_alert(
        matches_reasoning="N/A",
        masked_stuck_reason="Same state seen 4x in last 8 steps",
        no_op_streak=7,
        state_revisit=7,
    )
    assert "Same state seen 4x" in out
    assert "goal_hypothesis" in out.lower() or "abandon" in out.lower()


def test_orchestrator_alert_no_op_streak_fires_above_threshold() -> None:
    out = compute_orchestrator_alert(
        matches_reasoning="N/A",
        masked_stuck_reason="",
        no_op_streak=5,
        state_revisit=0,
        last_picks=["ACTION6", "ACTION6", "ACTION6", "ACTION1", "ACTION6"],
    )
    assert "Stuck" in out
    # culprit (most common) should be named
    assert "ACTION6" in out


def test_orchestrator_alert_state_revisit_fires_above_threshold() -> None:
    out = compute_orchestrator_alert(
        matches_reasoning="N/A",
        masked_stuck_reason="",
        no_op_streak=0,
        state_revisit=6,
        last_picks=["ACTION1"],
    )
    assert "Stuck" in out


def test_orchestrator_alert_below_threshold_silent() -> None:
    """A single no_op step is not stuck -- no alert."""
    out = compute_orchestrator_alert(
        matches_reasoning="N/A",
        masked_stuck_reason="",
        no_op_streak=2,
        state_revisit=2,
        last_picks=["ACTION1"],
    )
    assert out == ""
