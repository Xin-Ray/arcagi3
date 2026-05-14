"""Unit tests for arc_agent.viz_v3_2.compose_step_image_v32.

Strategy: build a small grid, render, then sanity-check the resulting
PIL image's size + check pixel signatures (e.g. red alert bar present
when alert non-empty).
"""
from __future__ import annotations

import numpy as np
import pytest

from arc_agent.viz_v3_2 import (
    ALERT_BG,
    HEADER_H,
    PANEL_H,
    PANEL_W,
    TOTAL_H,
    TOTAL_W,
    _summarize_reflection_delta,
    _wrap,
    compose_step_image_v32,
    write_gif_v3_2,
)


def _grid(shape=(64, 64), filled=None) -> np.ndarray:
    g = np.zeros(shape, dtype=int)
    if filled:
        for (r, c), v in filled.items():
            g[r, c] = v
    return g


# ── basic shape ───────────────────────────────────────────────────────────


def test_returns_pil_image_with_correct_size() -> None:
    img = compose_step_image_v32(
        _grid(),
        action="ACTION1",
        reasoning="try going up",
        reflection_delta={"action_semantics_update": {"ACTION1": "up 1"}},
        alert="",
        header="ar25 step=0",
        matches_reasoning="YES",
    )
    assert img.size == (TOTAL_W, TOTAL_H)
    assert TOTAL_W == 512
    assert TOTAL_H == HEADER_H + PANEL_H


def test_handles_small_grid_via_letterbox() -> None:
    img = compose_step_image_v32(
        _grid(shape=(8, 8)),
        action="ACTION1",
        reasoning="",
        reflection_delta=None,
        alert="",
        header="x",
    )
    assert img.size == (TOTAL_W, TOTAL_H)


def test_rejects_non_2d_grid() -> None:
    with pytest.raises(ValueError):
        compose_step_image_v32(
            np.zeros((4, 4, 3), dtype=int),
            action="ACTION1",
        )


# ── alert visibility (pixel signature) ────────────────────────────────────


def _pixel(img, x: int, y: int) -> tuple[int, int, int]:
    return img.getpixel((x, y))[:3]


def test_alert_bar_paints_red_when_alert_present() -> None:
    img = compose_step_image_v32(
        _grid(),
        action="ACTION1",
        reasoning="",
        reflection_delta=None,
        alert="STOP spamming ACTION1",
    )
    # Top-left of right panel should be red-ish (ALERT_BG)
    # right panel x range: [PANEL_W, 2*PANEL_W); y range: [HEADER_H, ...)
    px = _pixel(img, PANEL_W + 3, HEADER_H + 3)
    assert px == ALERT_BG


def test_no_alert_bar_when_alert_empty() -> None:
    img = compose_step_image_v32(
        _grid(),
        action="ACTION1",
        reasoning="",
        reflection_delta=None,
        alert="",
    )
    # Same pixel location should be the dark panel bg, not red
    px = _pixel(img, PANEL_W + 3, HEADER_H + 3)
    assert px != ALERT_BG


# ── helpers ───────────────────────────────────────────────────────────────


def test_wrap_breaks_long_lines() -> None:
    lines = _wrap("a" * 100, width=20)
    assert len(lines) == 5
    assert all(len(l) <= 20 for l in lines)


def test_wrap_preserves_short_lines() -> None:
    assert _wrap("short", width=20) == ["short"]


def test_wrap_handles_explicit_newlines() -> None:
    out = _wrap("line1\nline2", width=20)
    assert out == ["line1", "line2"]


def test_wrap_empty_returns_empty() -> None:
    assert _wrap("") == []


def test_summarize_delta_action_semantics() -> None:
    out = _summarize_reflection_delta({
        "action_semantics_update": {"ACTION1": "moves red up"},
    })
    assert any("ACTION1" in ln for ln in out)


def test_summarize_delta_goal() -> None:
    out = _summarize_reflection_delta({
        "goal_hypothesis_update": "reach top",
    })
    assert any("reach top" in ln for ln in out)


def test_summarize_delta_rules_failed() -> None:
    out = _summarize_reflection_delta({
        "rules_append": ["red goes first"],
        "failed_strategies_append": ["spam ACTION6"],
    })
    joined = " ".join(out)
    assert "red goes first" in joined
    assert "spam ACTION6" in joined


def test_summarize_delta_empty_falls_back() -> None:
    out = _summarize_reflection_delta({})
    assert out == ["(none)"] or out == ["(empty delta)"]
    out = _summarize_reflection_delta(None)
    assert out == ["(none)"]


# ── GIF round-trip ────────────────────────────────────────────────────────


def test_write_gif_round_trip(tmp_path) -> None:
    frames = [
        compose_step_image_v32(_grid(), action=f"ACTION{i}",
                               reasoning=f"step {i}", reflection_delta=None,
                               alert="", header=f"frame {i}")
        for i in range(3)
    ]
    out = write_gif_v3_2(frames, tmp_path / "play.gif", fps=2)
    assert out.exists()
    assert out.stat().st_size > 0
