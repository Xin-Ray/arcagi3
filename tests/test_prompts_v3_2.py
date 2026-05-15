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
    """The alert block (now [ALERT], previously [REFLECTION ALERT]) should
    be flagged as top priority."""
    low = ACTION_SYSTEM.lower()
    assert "alert" in low
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
    """Reflection only owns the matches_reasoning trigger now.
    no_op_streak / state_revisit are orchestrator-side (compute_orchestrator_alert).
    """
    low = REFLECTION_SYSTEM.lower()
    assert "matches_reasoning" in low
    # Reflection should explicitly STATE that the orchestrator handles
    # stuck/no_op/revisit so the LLM stops trying to write those alerts
    assert "orchestrator" in low


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
    assert "[ALERT]" in p
    assert "stop spamming ACTION1" in p
    # Alert must appear BEFORE [KNOWLEDGE] (top priority)
    assert p.index("[ALERT]") < p.index("[KNOWLEDGE")


def test_action_prompt_has_required_blocks() -> None:
    """v3.2 consolidated block list (was 17 blocks in v3, now 7)."""
    p = _build_action_prompt()
    for marker in ("[KNOWLEDGE", "[STATE]", "[ACTION stats", "[ASK]"):
        assert marker in p, f"missing v3.2 block: {marker}"
    # Dropped v3 blocks must NOT appear in the v3.2 path
    for dropped in ("[STATUS]", "[ACTIVE]", "[TEXTURE]", "[UNTRIED",
                    "[HISTORY", "[GOAL hyp", "[CLICK CANDIDATES",
                    "[STUCK SIGNALS]", "[LOW-PRIORITY ACTIONS"):
        assert dropped not in p, f"v3 block {dropped} should be dropped from v3.2"


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


# ── Object relations folded into [STATE] ────────────────────────────────


def test_action_prompt_state_block_includes_relations_when_passed() -> None:
    """v3.2 folds [OBJECT RELATIONS] inline under [STATE]. The relations
    content (e.g. same-color group color names) must still reach the
    prompt -- the contract is the FACTS, not the block name."""
    from arc_agent.object_relations import ObjectRelations
    relations = ObjectRelations(
        same_color_groups={"red": [0, 1, 2]},
        closest_pairs=[(0, 1, 3.2)],
    )
    p = _build_action_prompt(object_relations=relations)
    assert "[STATE]" in p
    assert "red" in p


def test_action_prompt_no_relations_text_when_none() -> None:
    p = _build_action_prompt(object_relations=None)
    # No top-level OBJECT RELATIONS block and no inline relations sub-section
    assert "[OBJECT RELATIONS]" not in p
    assert "\n  relations:" not in p


def test_action_prompt_knowledge_appears_above_state() -> None:
    """Order: ALERT -> KNOWLEDGE -> EXPLORATION HINT -> CLICK TARGETS -> STATE."""
    k = Knowledge.empty("ar25")
    k.action_semantics = {"ACTION1": "moves the red 1x1 up"}
    p = _build_action_prompt(knowledge=k)
    assert p.index("[KNOWLEDGE") < p.index("[STATE]")


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


def test_reflection_prompt_renders_status_when_passed() -> None:
    """A (2026-05-14): Reflection USER prompt should include [STATUS]
    when env-context kwargs are supplied."""
    p = build_reflection_user_prompt(
        knowledge=Knowledge.empty("ar25"),
        step_summary=_step_summary(),
        step=12, max_steps=80, level=1, total_levels=4,
        state_name="NOT_FINISHED",
        legal_actions=["ACTION1", "ACTION2"],
    )
    assert "[STATUS]" in p
    assert "12 / 80" in p
    assert "1 / 4" in p
    assert "NOT_FINISHED" in p


def test_reflection_prompt_renders_object_relations_when_passed() -> None:
    """A: Reflection now sees [OBJECT RELATIONS] so it can infer goals
    from object configuration (same-color groups etc)."""
    from arc_agent.object_relations import ObjectRelations
    relations = ObjectRelations(
        same_color_groups={"red": [0, 1]},
        same_shape_groups={"2x2_size4": [0, 1]},
    )
    p = build_reflection_user_prompt(
        knowledge=Knowledge.empty("ar25"),
        step_summary=_step_summary(),
        object_relations=relations,
    )
    assert "[OBJECT RELATIONS]" in p
    assert "red" in p
    assert "2x2_size4" in p


def test_reflection_prompt_renders_action_effects_when_outcome_log_passed() -> None:
    """A: Reflection now sees per-action OutcomeLog stats so its rules
    are grounded in empirical truth."""
    from arc_agent.action_inference import OutcomeLog, StepOutcome
    log = OutcomeLog()
    for i in range(3):
        log.record(StepOutcome(step=i, action="ACTION1", legal=True,
                               frame_changed=True, n_active_changed=1,
                               primary_direction="UP", primary_distance=3))
    p = build_reflection_user_prompt(
        knowledge=Knowledge.empty("ar25"),
        step_summary=_step_summary(),
        outcome_log=log,
        legal_actions=["ACTION1", "ACTION2"],
    )
    assert "[ACTION effects observed]" in p
    assert "ACTION1" in p


def test_reflection_prompt_ask_block_guides_goal_inference() -> None:
    """ASK block should give concrete examples of state-described goals
    and discourage 'unknown' / 'ACTION_X should ...' style outputs."""
    p = build_reflection_user_prompt(
        knowledge=Knowledge.empty("ar25"),
        step_summary=_step_summary(),
    )
    ask = p.split("[ASK]")[-1].lower()
    assert "win state" in ask
    assert "same-color" in ask or "color groups" in ask
    # Forbids the bad outputs we've seen
    assert "action_x should" in ask or "action_x" in ask


def test_reflection_prompt_backward_compatible_without_state_kwargs() -> None:
    """A: legacy callers passing only knowledge+step_summary still work."""
    p = build_reflection_user_prompt(
        knowledge=Knowledge.empty("ar25"),
        step_summary=_step_summary(),
    )
    assert "[CURRENT KNOWLEDGE" in p
    assert "[STATUS]" not in p          # no env context, no STATUS block
    assert "[OBJECT RELATIONS]" not in p
    assert "[ACTION effects observed]" not in p


def test_reflection_prompt_renders_existing_knowledge() -> None:
    k = Knowledge.empty("ar25")
    k.rounds_played = 2
    k.rules = ["red moves first"]
    p = build_reflection_user_prompt(knowledge=k, step_summary=_step_summary())
    assert "rounds: 2 played" in p
    assert "red moves first" in p


# ── [EXPLORATION HINT] block (idea 1) ──────────────────────────────────


def test_action_prompt_includes_exploration_hint_when_passed() -> None:
    p = _build_action_prompt(
        exploration_hint="[EXPLORATION HINT]\n  Actions you have NOT tried: ACTION3, ACTION5"
    )
    assert "[EXPLORATION HINT]" in p
    assert "ACTION3" in p and "ACTION5" in p


def test_action_prompt_no_exploration_block_when_none() -> None:
    p = _build_action_prompt(exploration_hint=None)
    assert "[EXPLORATION HINT]" not in p


def test_action_prompt_exploration_hint_below_knowledge_above_state() -> None:
    """Order: KNOWLEDGE -> EXPLORATION HINT -> [STATE]."""
    p = _build_action_prompt(
        exploration_hint="[EXPLORATION HINT]\n  Actions you have NOT tried: ACTION5"
    )
    assert p.index("[KNOWLEDGE") < p.index("[EXPLORATION HINT]")
    assert p.index("[EXPLORATION HINT]") < p.index("[STATE]")


def test_reflection_prompt_includes_exploration_hint_when_passed() -> None:
    p = build_reflection_user_prompt(
        knowledge=Knowledge.empty("ar25"),
        step_summary=_step_summary(),
        exploration_hint=(
            "[EXPLORATION HINT]\n  Actions you have NOT tried: ACTION5\n"
            "  Objects that have NEVER reacted: obj_007 (cyan ...)"
        ),
    )
    assert "[EXPLORATION HINT]" in p
    assert "ACTION5" in p
    assert "obj_007" in p


def test_reflection_prompt_no_exploration_block_when_none() -> None:
    p = build_reflection_user_prompt(
        knowledge=Knowledge.empty("ar25"),
        step_summary=_step_summary(),
        exploration_hint=None,
    )
    assert "[EXPLORATION HINT]" not in p


def test_reflection_system_mentions_exploration_hint() -> None:
    """Reflection SYSTEM should tell the agent it MAY write an alert based
    on the EXPLORATION HINT."""
    assert "EXPLORATION HINT" in REFLECTION_SYSTEM


# ── BUG-10: [CLICK TARGETS] block ──────────────────────────────────────


def _make_click_target(**overrides):
    from arc_agent.click_targets import ClickTarget
    defaults = dict(
        obj_id="obj_001", signature="cyan_1x1", coords=(12, 30),
        color_name="cyan", bbox=(12, 30, 12, 30),
        confidence=1.0, tries=0, successes=0, last_seen_step=0, alive=True,
    )
    defaults.update(overrides)
    return ClickTarget(**defaults)


def test_action_prompt_includes_click_targets_when_passed() -> None:
    p = _build_action_prompt(click_targets=[
        _make_click_target(obj_id="obj_002", color_name="red",
                           bbox=(10, 20, 11, 21), coords=(10, 20)),
    ])
    assert "[CLICK TARGETS" in p
    assert "obj_002" in p
    assert "(10,20)" in p


def test_action_prompt_no_click_targets_block_when_empty() -> None:
    p = _build_action_prompt(click_targets=[])
    assert "[CLICK TARGETS" not in p


def test_action_prompt_no_click_targets_block_when_none() -> None:
    p = _build_action_prompt(click_targets=None)
    assert "[CLICK TARGETS" not in p


def test_action_prompt_click_targets_ordered_by_priority() -> None:
    """Untried target (priority 1.0) must appear before decayed one."""
    p = _build_action_prompt(click_targets=[
        _make_click_target(obj_id="obj_decayed", confidence=0.9, tries=8),
        _make_click_target(obj_id="obj_fresh", confidence=1.0, tries=0),
    ])
    assert p.index("obj_fresh") < p.index("obj_decayed")


def test_action_prompt_click_targets_above_state() -> None:
    """[CLICK TARGETS] should appear before [STATE] so it's not buried."""
    p = _build_action_prompt(click_targets=[_make_click_target()])
    assert p.index("[CLICK TARGETS") < p.index("[STATE]")


# ── Core-info-preservation contract (must survive any refactor) ──────────
#
# These tests pin down what facts MUST be visible to the Action Agent
# regardless of which blocks we add / remove / merge. The block names can
# change; the *facts* cannot. If a refactor accidentally drops one of these,
# the test fails and we know we have an information leak.


class _RichFixture:
    """Builds an action prompt with a fully-populated state so every
    contract-required fact has a non-default value to look for."""

    def __init__(self) -> None:
        from arc_agent.click_targets import ClickTarget
        self.knowledge = Knowledge.empty("ar25")
        self.knowledge.action_semantics = {
            "ACTION1": "moves the maroon 1x1 UP by 3 cells",
            "ACTION7": "rotates the cyan square",
        }
        self.knowledge.goal_hypothesis = "match every red dot with a red target"
        self.knowledge.goal_confidence = "medium"
        self.knowledge.rules = ["ACTION6 effective only near object centers"]
        self.knowledge.failed_strategies = ["clicking near (32,32) never advances"]
        self.knowledge.rejected_goals = ["push the player to the top-left"]
        self.knowledge.rounds_played = 2
        self.knowledge.current_alert = (
            "matches_reasoning=NO: you expected UP but obj moved DOWN"
        )
        self.click_targets = [
            ClickTarget(obj_id="obj_007", signature="cyan_1x1",
                        coords=(12, 30), color_name="cyan",
                        bbox=(12, 30, 12, 30),
                        confidence=1.0, tries=0, successes=0),
            ClickTarget(obj_id="obj_009", signature="yellow_1x1",
                        coords=(5, 60), color_name="yellow",
                        bbox=(5, 60, 5, 60),
                        confidence=0.05, tries=10, successes=0),
        ]
        self.exploration_hint = (
            "[EXPLORATION HINT]\n"
            "  STUCK: same state seen 4x in last 8 steps\n"
            "  Actions you have NOT tried this round: ACTION3, ACTION5"
        )

    def build(self, **overrides):
        kwargs = dict(
            knowledge=self.knowledge,
            click_targets=self.click_targets,
            exploration_hint=self.exploration_hint,
        )
        kwargs.update(overrides)
        return _build_action_prompt(**kwargs)


def test_contract_goal_hypothesis_visible() -> None:
    """The current goal MUST reach the prompt (used to live in [GOAL]
    block, now lives in [KNOWLEDGE]; this test stays true either way)."""
    p = _RichFixture().build()
    assert "match every red dot with a red target" in p


def test_contract_goal_confidence_visible() -> None:
    p = _RichFixture().build()
    assert "medium" in p   # confidence label


def test_contract_rejected_goals_visible() -> None:
    """BUG-8 negative memory must reach the prompt."""
    p = _RichFixture().build()
    assert "push the player to the top-left" in p


def test_contract_action_semantics_visible() -> None:
    """Both keys' subject-having descriptions must reach the prompt."""
    p = _RichFixture().build()
    assert "moves the maroon 1x1 UP by 3 cells" in p
    assert "rotates the cyan square" in p


def test_contract_rules_visible() -> None:
    p = _RichFixture().build()
    assert "ACTION6 effective only near object centers" in p


def test_contract_failed_strategies_visible() -> None:
    p = _RichFixture().build()
    assert "clicking near (32,32) never advances" in p


def test_contract_current_alert_visible_and_prominent() -> None:
    """Alert must appear AND must be near the top so the LLM sees it."""
    p = _RichFixture().build()
    alert_text = "matches_reasoning=NO"
    assert alert_text in p
    # Must appear in the first third of the prompt for visibility
    assert p.index(alert_text) < len(p) // 3


def test_contract_legal_actions_visible() -> None:
    """The list of currently legal actions must reach the prompt — the LLM
    can't pick from an option it doesn't know exists."""
    p = _RichFixture().build()
    # _build_action_prompt defaults to ACTION1, ACTION2, ACTION6
    for a in ("ACTION1", "ACTION2", "ACTION6"):
        assert a in p


def test_contract_click_targets_visible_with_obj_id_and_coords() -> None:
    """obj_id + coords MUST both reach the prompt so the LLM has a concrete
    target to address."""
    p = _RichFixture().build()
    assert "obj_007" in p
    assert "(12,30)" in p


def test_contract_click_targets_confidence_visible() -> None:
    """Decayed target's low confidence must show so the LLM knows to skip."""
    p = _RichFixture().build()
    # obj_009 has confidence 0.05 -> should be flagged WRITTEN OFF or similar
    assert "obj_009" in p
    assert "0.05" in p or "WRITTEN OFF" in p


def test_contract_exploration_hint_text_visible() -> None:
    """Untried actions + stuck signal must reach the prompt."""
    p = _RichFixture().build()
    # untried actions
    assert "ACTION3" in p and "ACTION5" in p
    # stuck signal
    assert "STUCK" in p


def test_contract_active_objects_with_bbox_visible() -> None:
    """[ACTIVE] objects with their bbox/center must reach the prompt so the
    LLM can target them with ACTION6."""
    # Build with non-empty object_memory
    from arc_agent.object_extractor import extract_objects
    from arc_agent.object_tracker import ObjectMemory
    from arc_agent.temporal_classifier import Layer

    g = np.zeros((10, 10), dtype=int)
    g[5, 5] = 3
    objs = extract_objects(g)
    layer = {o.id: Layer.CANDIDATE for o in objs}
    # Simulate ObjectMemory containing a tracked object
    mem = ObjectMemory()
    from arc_agent.object_tracker import ObjectSnapshot, TrackedObject
    t = TrackedObject(uid="obj_005", color=3, last_step=0, alive=True,
                      history=[ObjectSnapshot(step=0, scipy_id=0, color=3,
                                              color_name="green",
                                              bbox=(5, 5, 5, 5),
                                              center=(5.0, 5.0), size=1)])
    mem._tracked["obj_005"] = t

    p = _build_action_prompt(
        frame_objects=objs, layer_by_id=layer, object_memory=mem,
    )
    # Either explicit bbox numbers OR obj_005 anchor must reach the prompt
    assert "obj_005" in p or "[5,5,5,5]" in p


def test_contract_step_and_level_visible() -> None:
    """[STATUS] facts (step, level) must reach the prompt."""
    p = _build_action_prompt(step=42, level=3, total_levels=8)
    # step / max_steps appears as "42 / 80" etc.
    assert "42" in p
    # level appears as "3 / 8"
    assert "3 / 8" in p or "level: 3" in p


def test_contract_ask_format_visible() -> None:
    """The output format directive must reach the prompt EVERY step."""
    p = _RichFixture().build()
    assert "[ASK]" in p
    ask = p.split("[ASK]")[-1].lower()
    assert "reasoning:" in ask
    assert "action:" in ask


def test_contract_no_information_lost_when_alerts_empty() -> None:
    """Even when most optional blocks are empty, baseline facts (goal,
    semantics, legal actions, ASK) must still reach the prompt."""
    f = _RichFixture()
    f.knowledge.current_alert = ""
    p = f.build(exploration_hint=None)   # no alert, no exploration hint
    # Baseline facts still present
    assert "match every red dot with a red target" in p
    assert "moves the maroon 1x1 UP by 3 cells" in p
    assert "ACTION1" in p
    assert "[ASK]" in p
