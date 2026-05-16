"""Smoke tests for arc_agent.prompts_v3 — does the prompt builder produce
sensible blocks and respect the v3 hard constraints (no action semantics
leaked into SYSTEM, dynamic info goes only to USER)."""
from __future__ import annotations

import re

import numpy as np

from arc_agent.action_inference import OutcomeLog, StepOutcome
from arc_agent.object_extractor import extract_objects
from arc_agent.object_tracker import ObjectMemory
from arc_agent.prompts_v3 import (
    PLAY_SYSTEM,
    REFLECT_SYSTEM,
    build_play_user_prompt,
    build_reflect_user_prompt,
)
from arc_agent.temporal_classifier import Layer


def _grid(spec, shape=(10, 10)) -> np.ndarray:
    g = np.zeros(shape, dtype=int)
    for (r, c), v in spec.items():
        g[r, c] = v
    return g


# ── System-prompt invariants (v3 hard constraint #1) ──────────────────────


def test_play_system_does_not_leak_action_semantics() -> None:
    """No 'ACTION1=up' style hardcoded mapping."""
    forbidden = ["ACTION1=", "ACTION2=", "ACTION3=", "ACTION4=",
                 "ACTION5=", "ACTION6=", "ACTION7=",
                 "up,", "down,", "left,", "right,",
                 "interact ", "undo "]
    for f in forbidden:
        assert f.lower() not in PLAY_SYSTEM.lower(), (
            f"PLAY_SYSTEM leaked action semantics with substring '{f}'")


def test_reflect_system_does_not_leak_action_semantics() -> None:
    forbidden = ["ACTION1=", "up=", "down=", "left=", "right="]
    for f in forbidden:
        assert f.lower() not in REFLECT_SYSTEM.lower()


def test_play_system_is_pure_ascii() -> None:
    PLAY_SYSTEM.encode("ascii")   # raises if non-ASCII


def test_play_system_clarifies_coord_format() -> None:
    """v3.1 P0-A: SYSTEM must explicitly tell model that only ACTION6 takes coords."""
    s = PLAY_SYSTEM.lower()
    # Positive: must mention the ACTION6 exclusivity
    assert "only" in s and "action6" in s
    # Positive: must contain the invalid examples
    assert "invalid" in s or "do not output" in s.replace(" ", "")
    # Negative: must NOT have action1 with trailing coords as a positive example
    assert "action1 10" not in s.replace(" ", "")   # rejected pattern


def test_reflect_system_is_pure_ascii() -> None:
    REFLECT_SYSTEM.encode("ascii")


# ── User prompt structure ─────────────────────────────────────────────────


def _build_minimal(*, outcome_log=None, object_memory=None,
                   frame_objects=None, layer_by_id=None,
                   diversification_hint=None,
                   goal_hypothesis=""):
    g = _grid({(2, 2): 5})
    objs = frame_objects if frame_objects is not None else extract_objects(g)
    layer = layer_by_id if layer_by_id is not None else {o.id: Layer.CANDIDATE for o in objs}
    om = object_memory if object_memory is not None else ObjectMemory()
    log = outcome_log if outcome_log is not None else OutcomeLog()
    return build_play_user_prompt(
        step=3, max_steps=80, level=1, total_levels=8,
        state="NOT_FINISHED",
        legal_actions=["ACTION1", "ACTION2", "ACTION3"],
        frame_objects=objs,
        layer_by_id=layer,
        object_memory=om,
        outcome_log=log,
        goal_hypothesis=goal_hypothesis,
        diversification_hint=diversification_hint,
    )


def test_play_prompt_has_all_required_blocks() -> None:
    p = _build_minimal()
    for marker in ("[STATUS]", "[ACTIVE]", "[TEXTURE]", "[ACTION effects",
                   "[UNTRIED", "[HISTORY", "[GOAL", "[ASK]"):
        assert marker in p, f"missing block: {marker}"


def test_play_prompt_lists_legal_actions() -> None:
    p = _build_minimal()
    assert "ACTION1" in p and "ACTION2" in p and "ACTION3" in p


def test_play_prompt_status_annotates_params() -> None:
    """v3.1 P0-A: STATUS legal-action line must mark which actions take params."""
    p = _build_minimal()
    status_block = p.split("[STATUS]")[1].split("[ACTIVE]")[0]
    assert "(no params)" in status_block
    # The action set passed in (_build_minimal) is [ACTION1, ACTION2, ACTION3]
    # so we expect "no params" annotation; ACTION6 isn't in this case


def test_play_prompt_marks_untried() -> None:
    """All actions untried -> [UNTRIED] should list all of them."""
    p = _build_minimal()
    untried_section = p.split("[UNTRIED")[1]
    assert "ACTION1" in untried_section
    assert "ACTION2" in untried_section
    assert "ACTION3" in untried_section


def test_play_prompt_includes_step_and_level() -> None:
    p = _build_minimal()
    assert "step: 3 / 80" in p
    assert "level: 1 / 8" in p


def test_play_prompt_includes_diversification_alert_when_provided() -> None:
    p = _build_minimal(diversification_hint="you have repeated ACTION1 3 times")
    assert "[ALERT]" in p
    assert "repeated ACTION1" in p


def test_play_prompt_omits_diversification_when_absent() -> None:
    p = _build_minimal()
    assert "[ALERT]" not in p


def test_play_prompt_renders_goal_when_provided() -> None:
    p = _build_minimal(goal_hypothesis="push the red L to the green corner")
    assert "push the red L to the green corner" in p


def test_play_prompt_renders_unknown_goal_initially() -> None:
    p = _build_minimal()
    assert "unknown" in p.lower()


def test_play_prompt_under_4000_chars_for_small_scene() -> None:
    """Sanity: a small grid should not produce a huge prompt."""
    p = _build_minimal()
    assert len(p) < 4000


def test_active_block_shows_uid_and_color() -> None:
    """Drop a single object into memory; [ACTIVE] should describe it."""
    g = _grid({(2, 3): 5})
    objs = extract_objects(g)
    om = ObjectMemory()
    om.update(step=0, current_active=objs, matches=[])
    layer = {o.id: Layer.ACTIVE for o in objs}
    p = build_play_user_prompt(
        step=1, max_steps=80, level=1, total_levels=3,
        state="NOT_FINISHED",
        legal_actions=["ACTION1", "ACTION2"],
        frame_objects=objs,
        layer_by_id=layer,
        object_memory=om,
        outcome_log=OutcomeLog(),
    )
    active_block = p.split("[ACTIVE]")[1].split("[TEXTURE]")[0]
    assert "obj_000" in active_block
    assert "gray" in active_block


def test_action_block_summarizes_tried_actions() -> None:
    log = OutcomeLog()
    log.record(StepOutcome(step=0, action="ACTION1", legal=True,
                           frame_changed=True, n_active_changed=1,
                           primary_direction="UP", primary_distance=3))
    p = _build_minimal(outcome_log=log)
    action_block = p.split("[ACTION")[1].split("[UNTRIED")[0]
    assert "ACTION1" in action_block
    assert "UP" in action_block


def test_reflect_prompt_asks_for_json() -> None:
    p = build_reflect_user_prompt(
        legal_actions=["ACTION1", "ACTION2"],
        outcome_log=OutcomeLog(),
        object_memory=ObjectMemory(),
        current_goal="explore",
    )
    assert '"goal"' in p
    assert '"confidence"' in p
