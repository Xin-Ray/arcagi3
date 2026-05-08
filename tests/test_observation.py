"""Tests for arc_agent.observation."""
from __future__ import annotations

import numpy as np
import pytest
from arcengine import FrameDataRaw, GameState

from arc_agent.observation import (
    analyze_animation,
    animation_to_text,
    available_action_names,
    grid_diff,
    grid_to_text,
    latest_grid,
    summarize_frame,
)


def _frame(grids: list[np.ndarray] | None = None, **kwargs) -> FrameDataRaw:
    defaults = dict(
        game_id="test",
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        win_levels=3,
        available_actions=[1, 2, 3, 4],
    )
    defaults.update(kwargs)
    f = FrameDataRaw(**defaults)
    if grids is not None:
        f.frame = grids  # set after construction; field has default_factory=list
    return f


def test_grid_to_text_small() -> None:
    g = np.array([[0, 1, 15], [10, 7, 0]])
    assert grid_to_text(g) == "01F\nA70"


def test_grid_to_text_clips_out_of_range() -> None:
    g = np.array([[20, -3], [0, 16]])
    # 20 -> clip 15 -> F; -3 -> clip 0 -> 0; 16 -> clip 15 -> F
    assert grid_to_text(g) == "F0\n0F"


def test_grid_to_text_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        grid_to_text(np.zeros((3, 3, 3)))


def test_grid_diff_lists_changed_cells() -> None:
    prev = np.array([[0, 1], [2, 3]])
    curr = np.array([[0, 9], [2, 7]])
    assert grid_diff(prev, curr) == [(0, 1, 1, 9), (1, 1, 3, 7)]


def test_grid_diff_empty_when_identical() -> None:
    g = np.array([[1, 2], [3, 4]])
    assert grid_diff(g, g) == []


def test_grid_diff_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        grid_diff(np.zeros((3, 3)), np.zeros((3, 4)))


def test_latest_grid_returns_last_animation_frame() -> None:
    a = np.zeros((4, 4), dtype=int)
    b = np.ones((4, 4), dtype=int)
    f = _frame(grids=[a, b])
    assert latest_grid(f) is b


def test_latest_grid_raises_on_empty_animation() -> None:
    f = _frame()  # default frame.frame=[]
    with pytest.raises(ValueError):
        latest_grid(f)


def test_available_action_names_maps_ids() -> None:
    f = _frame(available_actions=[1, 2, 6])
    names = available_action_names(f)
    assert names == ["ACTION1", "ACTION2", "ACTION6"]


def test_summarize_frame_includes_state_and_grid() -> None:
    g = np.array([[0, 1], [2, 3]])
    f = _frame(grids=[g])
    out = summarize_frame(f)
    assert "state: NOT_FINISHED" in out
    assert "levels_completed: 0/3" in out
    assert "ACTION1" in out
    assert "0123" not in out  # grid is rendered as multi-line, not single row
    assert "01\n23" in out  # actual hex content


def test_summarize_frame_can_omit_grid() -> None:
    g = np.array([[0, 1], [2, 3]])
    f = _frame(grids=[g])
    out = summarize_frame(f, include_grid=False)
    assert "01\n23" not in out


def test_summarize_frame_with_diff() -> None:
    prev = np.array([[0, 0], [0, 0]])
    curr = np.array([[0, 5], [0, 0]])
    f = _frame(grids=[curr])
    out = summarize_frame(f, diff_with=prev)
    assert "changed_cells_since_prev_step: 1" in out
    assert "(0, 1, 0, 5)" in out


# ── analyze_animation tests ────────────────────────────────────────────────────

def _make_frames(n: int, color: int, start_col: int, axis: str = "col") -> list[np.ndarray]:
    """Helper: slide `color` one cell per frame along axis."""
    frames = []
    for i in range(n):
        g = np.zeros((16, 16), dtype=int)
        if axis == "col":
            g[8, start_col + i] = color
        else:
            g[start_col + i, 8] = color
        frames.append(g)
    return frames


def test_analyze_animation_linear_right() -> None:
    frames = _make_frames(6, color=1, start_col=2, axis="col")
    records = analyze_animation(frames)
    assert len(records) == 1
    r = records[0]
    assert r["color"] == 1
    assert r["event"] == "moved"
    assert r["segments"][0][0] == "RIGHT"
    assert r["segments"][0][1] >= 4   # moved ~5 cells rightward


def test_analyze_animation_linear_down() -> None:
    frames = _make_frames(5, color=2, start_col=3, axis="row")
    records = analyze_animation(frames)
    assert len(records) == 1
    assert records[0]["segments"][0][0] == "DOWN"


def test_analyze_animation_appear() -> None:
    empty = np.zeros((8, 8), dtype=int)
    appeared = np.zeros((8, 8), dtype=int)
    appeared[4, 4] = 3
    records = analyze_animation([empty, appeared])
    assert any(r["event"] == "appeared" and r["color"] == 3 for r in records)


def test_analyze_animation_disappear() -> None:
    gone = np.zeros((8, 8), dtype=int)
    present = np.zeros((8, 8), dtype=int)
    present[2, 2] = 4
    records = analyze_animation([present, gone])
    assert any(r["event"] == "disappeared" and r["color"] == 4 for r in records)


def test_analyze_animation_stationary_ignored() -> None:
    g = np.zeros((8, 8), dtype=int)
    g[3, 3] = 5
    records = analyze_animation([g, g.copy()])
    assert not any(r["event"] == "moved" for r in records)


def test_analyze_animation_single_frame_returns_empty() -> None:
    g = np.zeros((8, 8), dtype=int)
    assert analyze_animation([g]) == []


def test_animation_to_text_includes_direction() -> None:
    frames = _make_frames(8, color=1, start_col=1, axis="col")
    text = animation_to_text(frames)
    assert "RIGHT" in text
    assert "8 frames" in text


def test_summarize_frame_includes_animation() -> None:
    frames = _make_frames(5, color=2, start_col=0, axis="col")
    f = _frame(grids=frames)
    out = summarize_frame(f, include_grid=False)
    assert "animation" in out
    assert "RIGHT" in out


def test_summarize_frame_no_animation_if_single_frame() -> None:
    g = np.zeros((4, 4), dtype=int)
    f = _frame(grids=[g])
    out = summarize_frame(f, include_grid=False)
    assert "animation" not in out
