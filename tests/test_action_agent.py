"""Unit tests for arc_agent.agents.action_agent.ActionAgent.

Mirrors test_text_agent.py but adds Knowledge attachment + reasoning
parsing + alert visibility in prompt. Backbone is faked.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from arcengine import FrameDataRaw, GameAction, GameState

from arc_agent.agents.action_agent import ActionAgent, parse_reasoning_and_action
from arc_agent.knowledge import Knowledge


class _FakeBackbone:
    """Stub: returns canned response strings in order."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[dict] = []

    def generate(self, image, prompt, *, system="", **kw) -> str:
        self.calls.append({"image": image, "prompt": prompt, "system": system, "kw": kw})
        if not self._replies:
            return "reasoning: default\naction: ACTION1"
        return self._replies.pop(0)


def _frame(state=GameState.NOT_FINISHED, available=None, grid=None,
           lvl=0, win_levels=3) -> FrameDataRaw:
    f = FrameDataRaw(
        game_id="ar25",
        state=state,
        levels_completed=lvl,
        win_levels=win_levels,
        available_actions=available or [1, 2, 3, 4, 5, 7],
    )
    f.frame = [grid if grid is not None else np.zeros((8, 8), dtype=int)]
    return f


# ── parse_reasoning_and_action ─────────────────────────────────────────────


def test_parse_two_line_format() -> None:
    r, a = parse_reasoning_and_action(
        "reasoning: clicking center should advance level\naction: ACTION6 32 32"
    )
    assert "clicking center" in r
    assert "ACTION6 32 32" in a


def test_parse_case_insensitive_labels() -> None:
    r, a = parse_reasoning_and_action("REASONING: foo\nACTION: ACTION3")
    assert r == "foo"
    assert a == "ACTION3"


def test_parse_missing_reasoning_returns_empty() -> None:
    r, a = parse_reasoning_and_action("action: ACTION1")
    assert r == ""
    assert "ACTION1" in a


def test_parse_missing_action_label_falls_back_to_whole_text() -> None:
    # No `action:` label -- the regex in _coerce_action will still parse
    r, a = parse_reasoning_and_action("just ACTION5 do it")
    assert r == ""
    assert "ACTION5" in a


def test_parse_handles_non_string() -> None:
    r, a = parse_reasoning_and_action(None)  # type: ignore[arg-type]
    assert r == "" and a == ""


# ── ActionAgent: basic flow ────────────────────────────────────────────────


def test_construct_no_gpu() -> None:
    ActionAgent(backbone=_FakeBackbone([]))
    ActionAgent()


def test_choose_returns_tuple_of_action_and_reasoning() -> None:
    bb = _FakeBackbone(["reasoning: try untried\naction: ACTION3"])
    agent = ActionAgent(backbone=bb)
    action, reasoning = agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert action is GameAction.ACTION3
    assert reasoning == "try untried"


def test_choose_returns_reset_on_not_played() -> None:
    bb = _FakeBackbone([])
    a, r = ActionAgent(backbone=bb).choose(
        _frame(state=GameState.NOT_PLAYED), history=[])
    assert a is GameAction.RESET
    assert r == ""
    assert bb.calls == []


def test_garbage_response_falls_back_random() -> None:
    bb = _FakeBackbone(["nope, no idea"])
    agent = ActionAgent(backbone=bb, seed=0)
    action, reasoning = agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert action.value in [1, 2, 3]
    assert agent._state.parse_failures == 1
    assert reasoning == ""   # no `reasoning:` line in the garbage


def test_reasoning_captured_in_state_even_on_fallback() -> None:
    """If reasoning line is present but action is unparseable, reasoning
    is still captured for the trace."""
    bb = _FakeBackbone(["reasoning: trying ACTION99\naction: ACTION99"])
    agent = ActionAgent(backbone=bb, seed=0)
    _, reasoning = agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert agent._state.last_reasoning == "trying ACTION99"
    assert reasoning == "trying ACTION99"


# ── Knowledge attachment ───────────────────────────────────────────────────


def test_attach_knowledge_renders_into_prompt() -> None:
    bb = _FakeBackbone(["reasoning: x\naction: ACTION1"])
    agent = ActionAgent(backbone=bb)
    k = Knowledge.empty("ar25")
    k.action_semantics = {"ACTION1": "moves player up by 3"}
    agent.attach_knowledge(k)
    agent.choose(_frame(available=[1, 2, 3]), history=[])
    prompt = bb.calls[-1]["prompt"]
    assert "[KNOWLEDGE" in prompt
    assert "moves player up by 3" in prompt


def test_current_alert_shows_at_top_of_prompt() -> None:
    bb = _FakeBackbone(["reasoning: x\naction: ACTION1"])
    agent = ActionAgent(backbone=bb)
    k = Knowledge.empty("ar25")
    k.current_alert = "STOP spamming ACTION1"
    agent.attach_knowledge(k)
    agent.choose(_frame(available=[1, 2, 3]), history=[])
    prompt = bb.calls[-1]["prompt"]
    assert "[REFLECTION ALERT]" in prompt
    assert prompt.index("[REFLECTION ALERT]") < prompt.index("[KNOWLEDGE")


def test_no_alert_block_when_alert_empty() -> None:
    bb = _FakeBackbone(["reasoning: x\naction: ACTION1"])
    agent = ActionAgent(backbone=bb)
    agent.attach_knowledge(Knowledge.empty("ar25"))
    agent.choose(_frame(available=[1, 2, 3]), history=[])
    prompt = bb.calls[-1]["prompt"]
    assert "[REFLECTION ALERT]" not in prompt


def test_reset_episode_keeps_knowledge_by_default() -> None:
    bb = _FakeBackbone(["reasoning: x\naction: ACTION1",
                        "reasoning: x\naction: ACTION1"])
    agent = ActionAgent(backbone=bb)
    k = Knowledge.empty("ar25")
    k.action_semantics = {"ACTION1": "kept across episode"}
    agent.attach_knowledge(k)
    agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert agent._state.step_count == 1

    agent.reset_episode_state()
    assert agent._state.step_count == 0
    # Knowledge survives the reset
    agent.choose(_frame(available=[1, 2, 3]), history=[])
    prompt = bb.calls[-1]["prompt"]
    assert "kept across episode" in prompt


def test_reset_episode_can_swap_knowledge() -> None:
    bb = _FakeBackbone(["reasoning: x\naction: ACTION1"])
    agent = ActionAgent(backbone=bb)
    agent.attach_knowledge(Knowledge.empty("ar25"))
    new_k = Knowledge.empty("ar25")
    new_k.action_semantics = {"ACTION1": "new value"}
    agent.reset_episode_state(knowledge=new_k)
    agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert "new value" in bb.calls[-1]["prompt"]


# ── Orchestrator helpers ───────────────────────────────────────────────────


def test_no_op_streak_initially_zero() -> None:
    agent = ActionAgent(backbone=_FakeBackbone([]))
    assert agent.no_op_streak() == 0


def test_no_op_streak_counts_after_steps() -> None:
    """Three consecutive ACTION1 calls on an empty grid -> no frame change."""
    bb = _FakeBackbone(["reasoning: x\naction: ACTION1"] * 3)
    agent = ActionAgent(backbone=bb)
    same_grid = np.zeros((8, 8), dtype=int)
    for _ in range(3):
        agent.choose(_frame(available=[1, 2, 3], grid=same_grid), history=[])
    # After 3 steps, the previous-outcome recording has registered 2 no-ops
    # (the step at t records the outcome of t-1's action). So streak >= 2.
    assert agent.no_op_streak() >= 2


def test_recent_step_records_returns_tail() -> None:
    bb = _FakeBackbone(["reasoning: x\naction: ACTION1"] * 4)
    agent = ActionAgent(backbone=bb)
    same = np.zeros((8, 8), dtype=int)
    for _ in range(4):
        agent.choose(_frame(available=[1, 2, 3], grid=same), history=[])
    recent = agent.recent_step_records(n=3)
    assert len(recent) <= 3
    if recent:
        # Each entry is (action_name, changed_bool, direction_or_None)
        assert recent[-1][0] == "ACTION1"


def test_state_revisit_count_on_repeated_grid() -> None:
    bb = _FakeBackbone(["reasoning: x\naction: ACTION1"] * 3)
    agent = ActionAgent(backbone=bb)
    same = np.zeros((8, 8), dtype=int)
    for _ in range(3):
        agent.choose(_frame(available=[1, 2, 3], grid=same), history=[])
    assert agent.state_revisit_count(same) >= 2


# ── Anti-collapse still works ─────────────────────────────────────────────


def test_anti_collapse_overrides_repeated_action() -> None:
    """After 3 ACTION1s in a row on an unchanging grid, the agent should
    reject another ACTION1 in favor of an untried legal action."""
    bb = _FakeBackbone([
        "reasoning: try1\naction: ACTION1",
        "reasoning: try1\naction: ACTION1",
        "reasoning: try1\naction: ACTION1",
        "reasoning: try1\naction: ACTION1",   # 4th: should be overridden
    ])
    agent = ActionAgent(backbone=bb, seed=42)
    same = np.zeros((8, 8), dtype=int)
    for _ in range(3):
        agent.choose(_frame(available=[1, 2, 3], grid=same), history=[])
    a4, _ = agent.choose(_frame(available=[1, 2, 3], grid=same), history=[])
    # ACTION1 was repeated 3 times in a row; the 4th must NOT be ACTION1
    assert a4.name != "ACTION1"


# ── R3: stuck-state forced exploration ────────────────────────────────────


def test_R3_forces_untried_when_stuck() -> None:
    """When stuck (no-op streak OR state-revisit crosses threshold) and an
    untried legal action exists, R3 overrides the LLM choice.

    Setup: 5 same-grid calls all picking ACTION1 (or ACTION3 alternating).
    By call 5, state_revisit_count == 5 -> stuck. ACTION5 still untried ->
    R3 picks it.
    """
    same = np.zeros((8, 8), dtype=int)
    # Alternate to keep anti-collapse off; only ACTION1 / ACTION3 picked
    replies = ["reasoning: r\naction: ACTION1",
               "reasoning: r\naction: ACTION3"] * 4
    bb = _FakeBackbone(replies)
    agent = ActionAgent(backbone=bb, seed=0)

    legal = [1, 3, 5]
    # First 4 calls: revisit goes 1->2->3->4, all below threshold; ACTION1/A3 picked
    for _ in range(4):
        a, _ = agent.choose(_frame(available=legal, grid=same), history=[])
        # Sanity: not forced yet
        assert a.name in ("ACTION1", "ACTION3"), f"early force: {a.name}"

    # 5th call: revisit reaches 5 -> R3 fires -> ACTION5 forced
    action, _ = agent.choose(_frame(available=legal, grid=same), history=[])
    assert action.name == "ACTION5", (
        f"R3 should force ACTION5 on stuck state but got {action.name}"
    )


def test_R3_skipped_when_untried_already_includes_chosen() -> None:
    """If LLM happens to pick an untried action while stuck, R3 leaves it
    alone (no need to override what's already untried)."""
    same = np.zeros((8, 8), dtype=int)
    # 5 calls of ACTION1 -> stuck.
    # 6th call: LLM picks ACTION5 (untried at this point). R3 must pass through.
    replies = ["reasoning: r\naction: ACTION1"] * 5 + [
        "reasoning: r\naction: ACTION5",
    ]
    bb = _FakeBackbone(replies)
    agent = ActionAgent(backbone=bb, seed=0)
    legal = [1, 5]

    for _ in range(5):
        agent.choose(_frame(available=legal, grid=same), history=[])
    action, _ = agent.choose(_frame(available=legal, grid=same), history=[])
    assert action.name == "ACTION5"
    # last_response_raw should NOT contain the override marker
    assert "R3 forced_explore" not in agent._state.last_response_raw


def test_R3_does_not_fire_when_no_untried() -> None:
    """If every legal action has been tried, R3 should NOT force anything --
    just let the LLM's choice through (orchestrator R2 mask is the fallback)."""
    same = np.zeros((8, 8), dtype=int)
    bb = _FakeBackbone(["reasoning: r\naction: ACTION1"] * 10)
    agent = ActionAgent(backbone=bb, seed=0)
    legal = [1]   # only one legal action ever

    for _ in range(7):
        agent.choose(_frame(available=legal, grid=same), history=[])
    # Streak is high but there's nothing to force-pick -> action stays ACTION1
    action, _ = agent.choose(_frame(available=legal, grid=same), history=[])
    assert action.name == "ACTION1"


def test_R3_records_override_in_last_response_raw() -> None:
    """Override should be auditable from the trace (last_response_raw)."""
    same = np.zeros((8, 8), dtype=int)
    replies = ["reasoning: r\naction: ACTION1",
               "reasoning: r\naction: ACTION3"] * 4
    bb = _FakeBackbone(replies)
    agent = ActionAgent(backbone=bb, seed=0)
    legal = [1, 3, 5]

    # Run until R3 fires (revisit threshold = 5 reached on call 5)
    for _ in range(5):
        agent.choose(_frame(available=legal, grid=same), history=[])
    assert "R3 forced_explore" in agent._state.last_response_raw, (
        f"R3 marker missing; last_response_raw={agent._state.last_response_raw!r}"
    )


def test_R3_skipped_when_streak_below_threshold() -> None:
    """One no-op should NOT trigger R3."""
    same = np.zeros((8, 8), dtype=int)
    bb = _FakeBackbone(["reasoning: r\naction: ACTION1",
                        "reasoning: r\naction: ACTION1"])
    agent = ActionAgent(backbone=bb, seed=0)
    agent.choose(_frame(available=[1, 2], grid=same), history=[])
    action, _ = agent.choose(_frame(available=[1, 2], grid=same), history=[])
    # Streak is at most 1 here -- under threshold -- ACTION1 should pass through
    assert action.name == "ACTION1"
    assert "R3 forced_explore" not in agent._state.last_response_raw


# ── max_new_tokens / temperature passthrough ──────────────────────────────


def test_max_new_tokens_passed_to_backbone() -> None:
    bb = _FakeBackbone(["reasoning: x\naction: ACTION1"])
    agent = ActionAgent(backbone=bb, max_new_tokens=128)
    agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert bb.calls[0]["kw"]["max_new_tokens"] == 128
