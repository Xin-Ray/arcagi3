"""Unit tests for arc_agent.agents.reflection_agent.

Backbone is faked so this runs without GPU/network. Covers:
- happy path: valid JSON -> delta passed through, parse_ok True
- ```json fence stripping
- garbage prose -> first balanced {...} extracted
- truly broken response -> empty delta, parse_ok False
- unknown keys filtered out
- backbone exception -> empty delta, parse_failures bumped
- delta + merged_with_delta integration round-trip
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from arc_agent.agents.reflection_agent import (
    ReflectionAgent,
    parse_reflection_output,
)
from arc_agent.knowledge import Knowledge
from arc_agent.step_summary import StepSummary


# ── parse_reflection_output ──────────────────────────────────────────────


def _valid_delta_json() -> str:
    return json.dumps({
        "action_semantics_update": {"ACTION1": "moves red up 1"},
        "goal_hypothesis_update": "reach top",
        "goal_confidence_update": "medium",
        "rules_append": ["red goes first"],
        "failed_strategies_append": [],
        "current_alert": "",
    })


def test_parse_valid_json() -> None:
    delta, ok = parse_reflection_output(_valid_delta_json())
    assert ok is True
    assert delta["action_semantics_update"] == {"ACTION1": "moves red up 1"}
    assert delta["goal_hypothesis_update"] == "reach top"
    assert delta["goal_confidence_update"] == "medium"
    assert delta["rules_append"] == ["red goes first"]
    assert delta["current_alert"] == ""


def test_parse_strips_json_fence() -> None:
    raw = "```json\n" + _valid_delta_json() + "\n```"
    delta, ok = parse_reflection_output(raw)
    assert ok is True
    assert "action_semantics_update" in delta


def test_parse_strips_generic_fence() -> None:
    raw = "```\n" + _valid_delta_json() + "\n```"
    delta, ok = parse_reflection_output(raw)
    assert ok is True


def test_parse_with_prose_prefix() -> None:
    raw = ("Here is my delta:\n" + _valid_delta_json()
           + "\n\nLet me know if that's right.")
    delta, ok = parse_reflection_output(raw)
    assert ok is True
    assert delta["goal_hypothesis_update"] == "reach top"


def test_parse_unknown_keys_filtered() -> None:
    raw = json.dumps({
        "action_semantics_update": {"ACTION1": "up"},
        "random_extra_field": "ignored",
        "current_alert": "ok",
    })
    delta, ok = parse_reflection_output(raw)
    assert ok is True
    assert "random_extra_field" not in delta
    assert delta["current_alert"] == "ok"


def test_parse_empty_response() -> None:
    delta, ok = parse_reflection_output("")
    assert ok is False
    assert delta == {}


def test_parse_garbage_response() -> None:
    delta, ok = parse_reflection_output("uhh I don't know what to say")
    assert ok is False
    assert delta == {}


def test_parse_malformed_json() -> None:
    delta, ok = parse_reflection_output("{unclosed")
    assert ok is False


def test_parse_non_string_input() -> None:
    delta, ok = parse_reflection_output(None)  # type: ignore[arg-type]
    assert ok is False
    delta, ok = parse_reflection_output(42)  # type: ignore[arg-type]
    assert ok is False


def test_parse_json_without_any_known_key_is_failure() -> None:
    raw = json.dumps({"only_unknown": "x"})
    delta, ok = parse_reflection_output(raw)
    assert ok is False
    assert delta == {}


def test_parse_array_not_dict() -> None:
    delta, ok = parse_reflection_output("[1, 2, 3]")
    assert ok is False
    assert delta == {}


# ── ReflectionAgent integration ──────────────────────────────────────────


class _FakeBackbone:
    """Stub backbone returning canned text. Records all calls."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[dict] = []

    def generate(self, image, prompt, *, system="", **kw) -> str:
        self.calls.append({
            "image": image, "prompt": prompt, "system": system, "kw": kw,
        })
        if not self._replies:
            return ""
        return self._replies.pop(0)


def _step_summary() -> StepSummary:
    return StepSummary(
        step=5,
        action="ACTION1",
        reasoning="I expect the red block to move up",
        frame_changed=True,
        primary_direction="UP",
        primary_distance=2,
        object_deltas=["obj#1 (color 2) moved UP 2 cell(s)"],
        no_op_streak=0,
        state_revisit_count=1,
        matches_reasoning="YES",
    )


def test_construct_without_backbone_then_lazy_error() -> None:
    agent = ReflectionAgent()
    with pytest.raises(RuntimeError, match="no backbone"):
        agent.reflect_after_step(
            knowledge=Knowledge.empty("ar25"),
            step_summary=_step_summary(),
        )


def test_happy_path_returns_delta_and_raw() -> None:
    bb = _FakeBackbone([_valid_delta_json()])
    agent = ReflectionAgent(backbone=bb)
    delta, raw = agent.reflect_after_step(
        knowledge=Knowledge.empty("ar25"),
        step_summary=_step_summary(),
    )
    assert "action_semantics_update" in delta
    assert raw == _valid_delta_json()
    assert agent._state.last_parse_ok is True
    assert agent._state.call_count == 1


def test_records_prompt_with_system_for_trace() -> None:
    bb = _FakeBackbone(["{\"current_alert\": \"hi\"}"])
    agent = ReflectionAgent(backbone=bb)
    agent.reflect_after_step(
        knowledge=Knowledge.empty("ar25"),
        step_summary=_step_summary(),
    )
    assert "[CURRENT KNOWLEDGE" in agent._state.last_prompt
    # SYSTEM must be in the captured prompt for full trace fidelity
    assert "Reflection Agent" in agent._state.last_prompt


def test_parse_failure_returns_empty_delta_and_keeps_raw() -> None:
    bb = _FakeBackbone(["I don't know, sorry."])
    agent = ReflectionAgent(backbone=bb)
    delta, raw = agent.reflect_after_step(
        knowledge=Knowledge.empty("ar25"),
        step_summary=_step_summary(),
    )
    assert delta == {}
    assert raw == "I don't know, sorry."
    assert agent._state.last_parse_ok is False
    assert agent._state.parse_failures == 1


def test_backbone_exception_returns_empty_delta() -> None:
    class _Boom:
        def generate(self, *a, **kw):
            raise RuntimeError("CUDA OOM")
    agent = ReflectionAgent(backbone=_Boom())
    delta, raw = agent.reflect_after_step(
        knowledge=Knowledge.empty("ar25"),
        step_summary=_step_summary(),
    )
    assert delta == {}
    assert raw == ""
    assert agent._state.last_parse_ok is False
    assert agent._state.parse_failures == 1


def test_delta_merges_into_knowledge() -> None:
    """End-to-end: a fake reflection output should produce a Knowledge
    that reflects the new action_semantics + alert."""
    raw = json.dumps({
        "action_semantics_update": {"ACTION1": "moves red up 2 cells"},
        "current_alert": "ACTION1 confirmed",
    })
    bb = _FakeBackbone([raw])
    agent = ReflectionAgent(backbone=bb)
    k = Knowledge.empty("ar25")
    delta, _ = agent.reflect_after_step(knowledge=k, step_summary=_step_summary())
    k2 = k.merged_with_delta(delta)
    assert k2.action_semantics["ACTION1"] == "moves red up 2 cells"
    assert k2.current_alert == "ACTION1 confirmed"


def test_reset_clears_state() -> None:
    bb = _FakeBackbone([_valid_delta_json()])
    agent = ReflectionAgent(backbone=bb)
    agent.reflect_after_step(knowledge=Knowledge.empty("ar25"),
                             step_summary=_step_summary())
    assert agent._state.call_count == 1
    agent.reset()
    assert agent._state.call_count == 0
    assert agent._state.last_parse_ok is False


def test_max_new_tokens_passed_to_backbone() -> None:
    bb = _FakeBackbone([_valid_delta_json()])
    agent = ReflectionAgent(backbone=bb, max_new_tokens=64)
    agent.reflect_after_step(knowledge=Knowledge.empty("ar25"),
                             step_summary=_step_summary())
    assert bb.calls[0]["kw"]["max_new_tokens"] == 64


def test_temperature_passed_to_backbone() -> None:
    bb = _FakeBackbone([_valid_delta_json()])
    agent = ReflectionAgent(backbone=bb, temperature=0.7)
    agent.reflect_after_step(knowledge=Knowledge.empty("ar25"),
                             step_summary=_step_summary())
    assert bb.calls[0]["kw"]["temperature"] == 0.7
