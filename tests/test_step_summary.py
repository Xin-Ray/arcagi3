"""Unit tests for arc_agent.step_summary.

Covers:
- compute_matches_reasoning across YES/PARTIAL/NO/N/A branches
- StepSummary.render produces a self-contained text panel
- object_delta_lines summarizes duck-typed matches
- grid_changed
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from arc_agent.step_summary import (
    StepSummary,
    compute_matches_reasoning,
    grid_changed,
    object_delta_lines,
)


# ── compute_matches_reasoning ─────────────────────────────────────────────


def test_matches_yes_direction() -> None:
    out = compute_matches_reasoning(
        "I expect the red block to move up by 3 cells",
        frame_changed=True,
        primary_direction="UP",
    )
    assert out == "YES"


def test_matches_no_direction_opposite() -> None:
    out = compute_matches_reasoning(
        "object will move up",
        frame_changed=True,
        primary_direction="DOWN",
    )
    assert out == "NO"


def test_matches_partial_when_direction_orthogonal() -> None:
    out = compute_matches_reasoning(
        "I expect movement up",
        frame_changed=True,
        primary_direction="LEFT",
    )
    assert out == "PARTIAL"


def test_matches_no_when_predicted_movement_but_no_op() -> None:
    out = compute_matches_reasoning(
        "the red block will move right",
        frame_changed=False,
        primary_direction=None,
    )
    assert out == "NO"


def test_matches_yes_when_predicted_noop_and_no_change() -> None:
    out = compute_matches_reasoning(
        "this should be a no-op since ACTION7 is undo without history",
        frame_changed=False,
        primary_direction=None,
    )
    assert out == "YES"


def test_matches_no_when_predicted_noop_and_changed() -> None:
    out = compute_matches_reasoning(
        "nothing will change",
        frame_changed=True,
        primary_direction="UP",
    )
    assert out == "NO"


def test_matches_partial_when_predicted_noop_but_direction_also_mentioned() -> None:
    out = compute_matches_reasoning(
        "no effect on shape, just move down a bit",
        frame_changed=True,
        primary_direction="DOWN",
    )
    # noop hint conflicts with frame_changed -> miss; direction mentioned + matches -> partial
    assert out == "PARTIAL"


def test_matches_na_when_reasoning_makes_no_claim() -> None:
    out = compute_matches_reasoning(
        "exploring untried action per knowledge advice",
        frame_changed=True,
        primary_direction="UP",
    )
    assert out == "N/A"


def test_matches_yes_compound_direction() -> None:
    out = compute_matches_reasoning(
        "move up and to the left",
        frame_changed=True,
        primary_direction="UP+LEFT",
    )
    assert out == "YES"


def test_matches_partial_compound_subset() -> None:
    out = compute_matches_reasoning(
        "should move up",
        frame_changed=True,
        primary_direction="UP+LEFT",
    )
    # claim ⊂ outcome but not equal — partial
    assert out == "PARTIAL"


def test_matches_word_boundary_avoids_substring_false_positive() -> None:
    # "uphold" contains "up" but not as a word
    out = compute_matches_reasoning(
        "uphold the strategy",
        frame_changed=True,
        primary_direction="LEFT",
    )
    assert out == "N/A"


# ── StepSummary.render ────────────────────────────────────────────────────


def _make_summary(**overrides: Any) -> StepSummary:
    base = dict(
        step=12,
        action="ACTION6",
        action_coords=(12, 30),
        reasoning="click red marker; knowledge says it advances level",
        frame_changed=True,
        primary_direction="UP",
        primary_distance=3,
        object_deltas=["obj#5 (color 2) APPEARED"],
        no_op_streak=0,
        state_revisit_count=1,
        matches_reasoning="PARTIAL",
        recent_steps=[("ACTION1", True, "UP"), ("ACTION3", False, None)],
    )
    base.update(overrides)
    return StepSummary(**base)


def test_render_contains_all_labels() -> None:
    text = _make_summary().render()
    for needle in [
        "[ACTION AGENT'S REASONING]",
        "[ACTION AGENT'S CHOICE]",
        "[ACTUAL OUTCOME]",
        "frame_changed: True",
        "primary_direction: UP",
        "no_op_streak: 0",
        "state_revisit_count: 1",
        "matches_reasoning: PARTIAL",
        "ACTION6 (12, 30)",
    ]:
        assert needle in text, f"missing: {needle!r}"


def test_render_includes_recent_steps_block() -> None:
    text = _make_summary().render()
    assert "[LAST STEPS" in text
    assert "ACTION1 -> CHANGED (UP)" in text
    assert "ACTION3 -> no-op" in text


def test_render_truncates_long_reasoning() -> None:
    long = "a" * 500
    text = _make_summary(reasoning=long).render()
    assert "..." in text
    assert len(text) < 1500


def test_render_no_coords_for_non_action6() -> None:
    text = _make_summary(action="ACTION1", action_coords=None).render()
    assert "ACTION1" in text
    assert "(12, 30)" not in text


def test_render_no_op_message_when_unchanged() -> None:
    text = _make_summary(
        frame_changed=False, primary_direction=None,
        primary_distance=0, object_deltas=[],
    ).render()
    assert "frame_changed: False" in text
    assert "(none)" in text


def test_render_handles_frame_changed_with_empty_deltas() -> None:
    text = _make_summary(object_deltas=[]).render()
    assert "frame changed but no tracked-object movement" in text


# ── object_delta_lines ────────────────────────────────────────────────────


@dataclass
class _FakeMatch:
    type: str
    before_id: int = 0
    after_id: int = 0
    color: int = 0
    delta: dict | None = None


def test_object_delta_lines_moved() -> None:
    matches = [_FakeMatch(type="moved", before_id=1, color=2,
                          delta={"dy": -3, "dx": 0})]
    lines = object_delta_lines(matches)
    assert len(lines) == 1
    assert "UP" in lines[0]
    assert "3 cell" in lines[0]


def test_object_delta_lines_skips_unchanged() -> None:
    matches = [_FakeMatch(type="unchanged"), _FakeMatch(type="moved",
               delta={"dy": 0, "dx": 1}, color=4)]
    lines = object_delta_lines(matches)
    assert len(lines) == 1
    assert "RIGHT" in lines[0]


def test_object_delta_lines_appeared_disappeared() -> None:
    matches = [
        _FakeMatch(type="appeared", after_id=7, color=3),
        _FakeMatch(type="disappeared", before_id=2, color=5),
    ]
    lines = object_delta_lines(matches)
    assert any("APPEARED" in ln for ln in lines)
    assert any("DISAPPEARED" in ln for ln in lines)


def test_object_delta_lines_caps_at_max() -> None:
    matches = [_FakeMatch(type="moved", before_id=i, color=1,
                          delta={"dy": 1, "dx": 0}) for i in range(10)]
    lines = object_delta_lines(matches, max_lines=3)
    assert len(lines) == 3


# ── grid_changed ──────────────────────────────────────────────────────────


def test_grid_changed_true_on_diff() -> None:
    a = np.zeros((4, 4), dtype=int)
    b = a.copy()
    b[0, 0] = 1
    assert grid_changed(a, b) is True


def test_grid_changed_false_on_equal() -> None:
    a = np.zeros((4, 4), dtype=int)
    assert grid_changed(a, a.copy()) is False


def test_grid_changed_false_on_missing_input() -> None:
    a = np.zeros((4, 4), dtype=int)
    assert grid_changed(None, a) is False
    assert grid_changed(a, None) is False
