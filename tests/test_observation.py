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
    grid_to_image,
    grid_to_text,
    latest_grid,
    serialize_step,
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


# ---- grid_to_image -----------------------------------------------------


def test_grid_to_image_default_scale_8() -> None:
    g = np.zeros((64, 64), dtype=int)
    img = grid_to_image(g)
    assert img.size == (512, 512)  # PIL is (width, height)
    assert img.mode == "RGB"


def test_grid_to_image_custom_scale() -> None:
    g = np.zeros((4, 8), dtype=int)
    img = grid_to_image(g, scale=3)
    assert img.size == (24, 12)  # (W*scale, H*scale)


def test_grid_to_image_color_mapping() -> None:
    # cell value 1 should map to ARC blue (0, 116, 217)
    g = np.array([[1]], dtype=int)
    img = grid_to_image(g, scale=1)
    assert img.getpixel((0, 0)) == (0, 116, 217)


def test_grid_to_image_each_color_distinct() -> None:
    g = np.arange(16, dtype=int).reshape(4, 4)
    img = grid_to_image(g, scale=1)
    pixels = {img.getpixel((c, r)) for r in range(4) for c in range(4)}
    assert len(pixels) == 16  # all 16 ARC colors are distinct


def test_grid_to_image_clip_high_equals_color_15() -> None:
    g_high = np.array([[99]], dtype=int)
    g_15 = np.array([[15]], dtype=int)
    assert grid_to_image(g_high, scale=1).getpixel((0, 0)) == grid_to_image(g_15, scale=1).getpixel((0, 0))


def test_grid_to_image_clip_low_equals_color_0() -> None:
    g_low = np.array([[-7]], dtype=int)
    g_0 = np.array([[0]], dtype=int)
    assert grid_to_image(g_low, scale=1).getpixel((0, 0)) == grid_to_image(g_0, scale=1).getpixel((0, 0))


def test_grid_to_image_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        grid_to_image(np.zeros((3, 3, 3), dtype=int))


def test_grid_to_image_rejects_zero_scale() -> None:
    with pytest.raises(ValueError):
        grid_to_image(np.zeros((4, 4), dtype=int), scale=0)


def test_grid_to_image_rejects_negative_scale() -> None:
    with pytest.raises(ValueError):
        grid_to_image(np.zeros((4, 4), dtype=int), scale=-1)


# ── serialize_step (Stage 0 trace schema) ────────────────────────────────────

_STEP_KEYS = {
    "step", "game_id", "level", "state", "image_path",
    "prompt", "response_raw", "parse_ok",
    "predicted_diff", "chosen_action", "real_diff", "f1",
}


def _full_step_kwargs() -> dict:
    return dict(
        step=5,
        game_id="ls20-9607627b",
        level=1,
        state="NOT_FINISHED",
        image_path="step_005.png",
        prompt="State: NOT_FINISHED ...",
        response_raw='{"chosen_action":"ACTION3"}',
        parse_ok=True,
        predicted_diff={(10, 4, 2), (10, 3, 2)},
        chosen_action="ACTION3",
        real_diff={(10, 4, 2)},
        f1=0.667,
    )


def test_serialize_step_has_all_schema_keys() -> None:
    row = serialize_step(**_full_step_kwargs())
    assert set(row.keys()) == _STEP_KEYS


def test_serialize_step_diffs_are_sorted_lists_of_lists() -> None:
    row = serialize_step(**_full_step_kwargs())
    assert row["predicted_diff"] == [[10, 3, 2], [10, 4, 2]]
    assert row["real_diff"] == [[10, 4, 2]]


def test_serialize_step_diff_none_passes_through() -> None:
    kw = _full_step_kwargs()
    kw["predicted_diff"] = None
    kw["real_diff"] = None
    row = serialize_step(**kw)
    assert row["predicted_diff"] is None
    assert row["real_diff"] is None


def test_serialize_step_parse_failure_shape() -> None:
    """Failed parse: parse_ok=False, predicted_diff/chosen_action/f1 all None."""
    row = serialize_step(
        step=0, game_id="g", level=1, state="NOT_FINISHED",
        image_path=None, prompt="...", response_raw="garbage", parse_ok=False,
        predicted_diff=None, chosen_action=None, real_diff=None, f1=None,
    )
    assert set(row.keys()) == _STEP_KEYS
    assert row["parse_ok"] is False
    assert row["predicted_diff"] is None
    assert row["chosen_action"] is None
    assert row["f1"] is None
    assert row["image_path"] is None


def test_serialize_step_round_trips_through_json() -> None:
    import json
    row = serialize_step(**_full_step_kwargs())
    on_disk = json.loads(json.dumps(row))
    assert on_disk == row


def test_serialize_step_coerces_numeric_strings() -> None:
    """str/int leniency for upstream callers, but type-coerced in output."""
    row = serialize_step(
        step="3",  # type: ignore[arg-type]
        game_id="g", level="2", state="WIN",  # type: ignore[arg-type]
        image_path="p.png", prompt="x", response_raw="y", parse_ok=True,
        predicted_diff=set(), chosen_action="ACTION1",
        real_diff=set(), f1="1.0",  # type: ignore[arg-type]
    )
    assert row["step"] == 3
    assert row["level"] == 2
    assert row["f1"] == 1.0
    assert row["predicted_diff"] == []
    assert row["real_diff"] == []
