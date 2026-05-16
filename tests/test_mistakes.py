"""Unit tests for arc_agent.mistakes — deterministic detectors (A4)."""
from __future__ import annotations

import numpy as np
import pytest

from arc_agent.mistakes import (
    LOOP_COUNT,
    LOOP_WINDOW,
    MAX_MISTAKES,
    MistakeBuffer,
    NO_OP_STREAK,
    StepRecord,
    detect_illegal_action,
    detect_loop,
    detect_no_op_streak,
    detect_regression,
    frame_hash,
    update_mistakes,
)


def _rec(step: int, action: str = "ACTION1", *, legal: bool = True,
         changed: bool = True, hash_: int = 0, lvl: int = 0) -> StepRecord:
    return StepRecord(step=step, action=action, legal=legal,
                      frame_changed=changed, frame_hash=hash_,
                      levels_completed=lvl)


# ── illegal action ────────────────────────────────────────────────────────


def test_illegal_action_fires_on_illegal() -> None:
    msg = detect_illegal_action(_rec(0, "ACTION5", legal=False))
    assert msg is not None
    assert "ACTION5" in msg and "not legal" in msg


def test_illegal_action_silent_on_legal() -> None:
    assert detect_illegal_action(_rec(0, "ACTION3", legal=True)) is None


# ── no-op streak ──────────────────────────────────────────────────────────


def test_no_op_streak_fires_after_N() -> None:
    recs = [_rec(i, changed=False) for i in range(NO_OP_STREAK)]
    msg = detect_no_op_streak(recs)
    assert msg is not None
    assert "did not change" in msg


def test_no_op_streak_silent_below_N() -> None:
    recs = [_rec(i, changed=False) for i in range(NO_OP_STREAK - 1)]
    assert detect_no_op_streak(recs) is None


def test_no_op_streak_silent_when_recent_step_changed() -> None:
    recs = [_rec(0, changed=False), _rec(1, changed=False), _rec(2, changed=True)]
    assert detect_no_op_streak(recs) is None


# ── regression ────────────────────────────────────────────────────────────


def test_regression_fires_when_levels_drop() -> None:
    recs = [_rec(0, "ACTION1", lvl=2), _rec(1, "ACTION7", lvl=1)]
    msg = detect_regression(recs)
    assert msg is not None and "undid a level" in msg


def test_regression_silent_when_levels_stable() -> None:
    recs = [_rec(0, lvl=2), _rec(1, lvl=2)]
    assert detect_regression(recs) is None


def test_regression_silent_when_levels_advance() -> None:
    recs = [_rec(0, lvl=1), _rec(1, lvl=2)]
    assert detect_regression(recs) is None


# ── loop detector ─────────────────────────────────────────────────────────


def test_loop_fires_when_same_hash_action_repeats() -> None:
    recs = [_rec(i, "ACTION1", hash_=99) for i in range(LOOP_COUNT)]
    msg = detect_loop(recs)
    assert msg is not None and "ACTION1" in msg and "looping" in msg


def test_loop_silent_below_threshold() -> None:
    recs = [_rec(i, "ACTION1", hash_=99) for i in range(LOOP_COUNT - 1)]
    assert detect_loop(recs) is None


def test_loop_silent_when_hashes_differ() -> None:
    recs = [_rec(i, "ACTION1", hash_=i) for i in range(LOOP_COUNT)]
    assert detect_loop(recs) is None


def test_loop_only_considers_recent_window() -> None:
    # Old hits beyond window should not count
    old = [_rec(i, "ACTION1", hash_=99) for i in range(LOOP_COUNT)]
    fresh = [_rec(LOOP_COUNT + i, "ACTION2", hash_=i + 100) for i in range(LOOP_WINDOW)]
    assert detect_loop(old + fresh) is None


# ── buffer + cap ──────────────────────────────────────────────────────────


def test_mistake_buffer_caps_at_max() -> None:
    buf = MistakeBuffer()
    for i in range(MAX_MISTAKES + 3):
        buf.add_mistake(f"mistake {i}")
    assert len(buf.mistakes) == MAX_MISTAKES
    # Oldest should be evicted; newest should remain.
    assert buf.mistakes[-1] == f"mistake {MAX_MISTAKES + 2}"


def test_mistake_buffer_dedups() -> None:
    buf = MistakeBuffer()
    buf.add_mistake("same")
    buf.add_mistake("same")
    assert buf.mistakes == ["same"]


def test_buffer_reset_clears_everything() -> None:
    buf = MistakeBuffer()
    buf.append_record(_rec(0))
    buf.add_mistake("x")
    buf.reset()
    assert buf.records == []
    assert buf.mistakes == []


# ── update_mistakes orchestration ─────────────────────────────────────────


def test_update_mistakes_fires_illegal_immediately() -> None:
    buf = MistakeBuffer()
    msgs = update_mistakes(buf, _rec(0, "ACTION5", legal=False))
    assert any("ACTION5" in m for m in msgs)


def test_update_mistakes_accumulates_no_op_then_regression() -> None:
    buf = MistakeBuffer()
    for i in range(NO_OP_STREAK):
        update_mistakes(buf, _rec(i, "ACTION1", changed=False, lvl=2))
    assert any("did not change" in m for m in buf.mistakes)
    msgs = update_mistakes(buf, _rec(NO_OP_STREAK, "ACTION7", changed=True, lvl=1))
    assert any("undid a level" in m for m in msgs)


def test_update_mistakes_loop_capped_by_max_mistakes() -> None:
    buf = MistakeBuffer()
    # Pump 10 distinct illegal actions; expect cap.
    for i in range(MAX_MISTAKES + 3):
        update_mistakes(buf, _rec(i, f"ACTION{(i % 7) + 1}", legal=False))
    assert len(buf.mistakes) <= MAX_MISTAKES


# ── frame_hash ────────────────────────────────────────────────────────────


def test_frame_hash_stable_for_equal_grids() -> None:
    a = np.zeros((4, 4), dtype=int)
    b = np.zeros((4, 4), dtype=int)
    assert frame_hash(a) == frame_hash(b)


def test_frame_hash_differs_for_different_grids() -> None:
    a = np.zeros((4, 4), dtype=int)
    b = np.zeros((4, 4), dtype=int)
    b[0, 0] = 1
    assert frame_hash(a) != frame_hash(b)
