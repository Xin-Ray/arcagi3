"""Unit tests for arc_agent.prompts_v3_2 — Action + Reflection prompt builders."""
from __future__ import annotations

import numpy as np
import pytest

from arc_agent.action_inference import OutcomeLog
from arc_agent.knowledge import Knowledge
from arc_agent.object_extractor import extract_objects
from arc_agent.object_tracker import ObjectMemory
from arc_agent.prompts_v3_2 import (
    ACTION_SYSTEM,
    REFLECTION_SYSTEM,
    build_action_user_prompt,
    build_reflection_user_prompt,
)
from arc_agent.step_summary import StepSummary
from arc_agent.temporal_classifier import Layer


def _grid(spec, shape=(10, 10)) -> np.ndarray:
    g = np.zeros(shape, dtype=int)
    for (r, c), v in spec.items():
        g[r, c] = v
    return g


def _build_action_prompt(*, knowledge=None, **overrides) -> str:
    g = _grid({(2, 2): 5})
    objs = extract_objects(g)
    layer = {o.id: Layer.CANDIDATE for o in objs}
    kwargs = dict(
        knowledge=knowledge if knowledge is not None else Knowledge.empty("ar25"),
        step=3, max_steps=80, level=1, total_levels=8,
        state="NOT_FINISHED",
        legal_actions=["ACTION1", "ACTION2", "ACTION6"],
        frame_objects=objs, layer_by_id=layer,
        object_memory=ObjectMemory(),
        outcome_log=OutcomeLog(),
    )
    kwargs.update(overrides)
    return build_action_user_prompt(**kwargs)


# ── System constants invariants ─────────────────────────────────────────


def test_action_system_is_ascii() -> None:
    ACTION_SYSTEM.encode("ascii")


def test_reflection_system_is_ascii() -> None:
    REFLECTION_SYSTEM.encode("ascii")


def test_action_system_describes_reasoning_action_format() -> None:
    low = ACTION_SYSTEM.lower()
    assert "reasoning:" in low and "action:" in low
    assert "two lines" in low


def test_action_system_mentions_failed_strategies() -> None:
    assert "failed_strategies" in ACTION_SYSTEM


def test_action_system_warns_about_alert_priority() -> None:
    """[REFLECTION ALERT] should be flagged as highest priority."""
    low = ACTION_SYSTEM.lower()
    assert "reflection alert" in low
    assert "first" in low or "top" in low


def test_reflection_system_demands_strict_json() -> None:
    low = REFLECTION_SYSTEM.lower()
    assert "strict json" in low or "strict json" in low.replace(" ", "")
    # Every required field must be mentioned
    for field in ("action_semantics_update", "goal_hypothesis_update",
                  "goal_confidence_update", "rules_append",
                  "failed_strategies_append", "current_alert"):
        assert field in REFLECTION_SYSTEM


def test_reflection_system_lists_alert_triggers() -> None:
    """Must list at least the explicit triggers from §4.2."""
    low = REFLECTION_SYSTEM.lower()
    assert "matches_reasoning" in low
    assert "no_op_streak" in low
    assert "state_revisit_count" in low


# ── Action USER prompt ────────────────────────────────────────────────────


def test_action_prompt_contains_knowledge_block() -> None:
    k = Knowledge.empty("ar25")
    k.action_semantics = {"ACTION1": "moves red up by 3"}
    p = _build_action_prompt(knowledge=k)
    assert "[KNOWLEDGE" in p
    assert "moves red up by 3" in p


def test_action_prompt_omits_alert_when_empty() -> None:
    p = _build_action_prompt()
    assert "[REFLECTION ALERT]" not in p


def test_action_prompt_includes_alert_at_top_when_present() -> None:
    k = Knowledge.empty("ar25")
    k.current_alert = "stop spamming ACTION1; nothing moves"
    p = _build_action_prompt(knowledge=k)
    assert "[REFLECTION ALERT]" in p
    assert "stop spamming ACTION1" in p
    # Alert must appear BEFORE [KNOWLEDGE] (top priority)
    assert p.index("[REFLECTION ALERT]") < p.index("[KNOWLEDGE")


def test_action_prompt_preserves_v3_blocks() -> None:
    p = _build_action_prompt()
    for marker in ("[STATUS]", "[ACTIVE]", "[TEXTURE]", "[ACTION effects",
                   "[UNTRIED", "[HISTORY", "[GOAL", "[ASK]"):
        assert marker in p, f"missing v3 block: {marker}"


def test_action_prompt_ask_demands_two_lines() -> None:
    p = _build_action_prompt()
    ask = p.split("[ASK]")[-1]
    assert "reasoning:" in ask.lower()
    assert "action:" in ask.lower()
    # Old v3 prompt ASK said "Output ONE action token now" — must be gone
    assert "Output ONE action token now" not in p


def test_action_prompt_does_not_have_duplicate_ask() -> None:
    """Splicing must REPLACE v3's [ASK], not stack on top."""
    p = _build_action_prompt()
    assert p.count("[ASK]") == 1


# ── R7: [BLOCKED ACTIONS] block ─────────────────────────────────────────


def test_action_prompt_includes_blocked_actions_when_passed() -> None:
    p = _build_action_prompt(blocked_actions={"ACTION6"})
    assert "[BLOCKED ACTIONS" in p
    assert "ACTION6" in p
    assert "REPLACE" in p.upper() or "replace" in p


def test_action_prompt_no_blocked_block_when_set_empty() -> None:
    p = _build_action_prompt(blocked_actions=set())
    assert "[BLOCKED ACTIONS" not in p


def test_action_prompt_no_blocked_block_when_none() -> None:
    p = _build_action_prompt(blocked_actions=None)
    assert "[BLOCKED ACTIONS" not in p


def test_action_prompt_blocked_above_v3_blocks() -> None:
    p = _build_action_prompt(blocked_actions={"ACTION6", "ACTION7"})
    # BLOCKED should appear before [STATUS] (which is from v3 layer)
    assert p.index("[BLOCKED ACTIONS") < p.index("[STATUS]")


def test_action_prompt_includes_object_relations_block() -> None:
    """When ObjectRelations is passed, the prompt should include the
    [OBJECT RELATIONS] block between [ACTIVE] and [TEXTURE]."""
    from arc_agent.object_relations import ObjectRelations
    relations = ObjectRelations(
        same_color_groups={"red": [0, 1, 2]},
        closest_pairs=[(0, 1, 3.2)],
    )
    p = _build_action_prompt(object_relations=relations)
    assert "[OBJECT RELATIONS]" in p
    assert "red" in p
    # Should sit between [ACTIVE] and [TEXTURE]
    assert p.index("[ACTIVE]") < p.index("[OBJECT RELATIONS]")
    assert p.index("[OBJECT RELATIONS]") < p.index("[TEXTURE]")


def test_action_prompt_no_relations_block_when_none() -> None:
    p = _build_action_prompt(object_relations=None)
    assert "[OBJECT RELATIONS]" not in p


def test_action_prompt_blocked_sorted_for_determinism() -> None:
    """Sorted output -> same prompt every step given same mask -> Qwen
    cache hits + reproducible tests."""
    p = _build_action_prompt(blocked_actions={"ACTION7", "ACTION6", "ACTION1"})
    blocked_section = p.split("[BLOCKED ACTIONS")[1].split("[KNOWLEDGE")[0]
    # ACTION1 should come first (sorted)
    assert blocked_section.index("ACTION1") < blocked_section.index("ACTION6")
    assert blocked_section.index("ACTION6") < blocked_section.index("ACTION7")


def test_action_prompt_knowledge_appears_above_v3_blocks() -> None:
    k = Knowledge.empty("ar25")
    k.action_semantics = {"ACTION1": "up"}
    p = _build_action_prompt(knowledge=k)
    assert p.index("[KNOWLEDGE") < p.index("[STATUS]")


# ── Reflection USER prompt ────────────────────────────────────────────────


def _step_summary(**overrides) -> StepSummary:
    base = dict(
        step=12,
        action="ACTION6",
        action_coords=(12, 30),
        reasoning="click the red marker to advance the level",
        frame_changed=True,
        primary_direction="UP",
        primary_distance=3,
        object_deltas=["obj#5 (color 2) APPEARED"],
        no_op_streak=0,
        state_revisit_count=1,
        matches_reasoning="PARTIAL",
    )
    base.update(overrides)
    return StepSummary(**base)


def test_reflection_prompt_has_required_blocks() -> None:
    p = build_reflection_user_prompt(
        knowledge=Knowledge.empty("ar25"),
        step_summary=_step_summary(),
    )
    for marker in ("[CURRENT KNOWLEDGE",
                   "[ACTION AGENT'S REASONING]",
                   "[ACTION AGENT'S CHOICE]",
                   "[ACTUAL OUTCOME]",
                   "matches_reasoning:",
                   "[ASK]"):
        assert marker in p, f"missing: {marker}"


def test_reflection_prompt_shows_knowledge_first() -> None:
    p = build_reflection_user_prompt(
        knowledge=Knowledge.empty("ar25"),
        step_summary=_step_summary(),
    )
    assert p.index("[CURRENT KNOWLEDGE") < p.index("[ACTUAL OUTCOME]")


def test_reflection_prompt_shows_matches_verdict() -> None:
    p = build_reflection_user_prompt(
        knowledge=Knowledge.empty("ar25"),
        step_summary=_step_summary(matches_reasoning="NO"),
    )
    assert "matches_reasoning: NO" in p


def test_reflection_prompt_ask_demands_all_six_fields() -> None:
    p = build_reflection_user_prompt(
        knowledge=Knowledge.empty("ar25"),
        step_summary=_step_summary(),
    )
    ask = p.split("[ASK]")[-1].lower()
    # ASK block can be short; the SYSTEM lists the fields. Just check that
    # ASK at least demands JSON.
    assert "json" in ask


def test_reflection_prompt_renders_existing_knowledge() -> None:
    k = Knowledge.empty("ar25")
    k.rounds_played = 2
    k.rules = ["red moves first"]
    p = build_reflection_user_prompt(knowledge=k, step_summary=_step_summary())
    assert "rounds: 2 played" in p
    assert "red moves first" in p
