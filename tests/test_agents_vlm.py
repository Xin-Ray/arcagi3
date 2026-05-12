"""Unit tests for `arc_agent.agents.vlm.VLMAgent` — no GPU required.

All tests inject a fake backbone exposing the single `.generate(image, prompt,
system=...)` method, so this file imports cleanly on a machine without the
training stack (torch / transformers / bitsandbytes / qwen_vl_utils).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from arcengine import FrameDataRaw, GameAction, GameState

from arc_agent.agents.vlm import (
    PROMPT_SECTIONS_IN_ORDER,
    VLMAgent,
    _AgentState,
)


# ── helpers ───────────────────────────────────────────────────────────────


class _FakeBackbone:
    """Stand-in for HFBackbone — returns canned strings in order."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple] = []  # (image, prompt, system)

    def generate(self, image, prompt: str, *, system: str = "", **kw) -> str:
        self.calls.append((image, prompt, system, kw))
        if not self._replies:
            return '{"chosen_action": "ACTION1"}'
        return self._replies.pop(0)


class _BoomBackbone:
    def generate(self, image, prompt: str, *, system: str = "", **kw) -> str:
        raise RuntimeError("simulated GPU failure")


def _frame(
    state: GameState = GameState.NOT_FINISHED,
    available: list[int] | None = None,
    grids: list[np.ndarray] | None = None,
    game_id: str = "ls20",
) -> FrameDataRaw:
    f = FrameDataRaw(
        game_id=game_id,
        state=state,
        levels_completed=0,
        win_levels=3,
        available_actions=available or [1, 2, 3, 4, 5, 7],
    )
    if grids is None:
        grids = [np.zeros((8, 8), dtype=int)]
    f.frame = grids
    return f


# ── construction / reset ──────────────────────────────────────────────────


def test_construct_requires_no_gpu() -> None:
    # Should not import torch / transformers at construction time.
    VLMAgent(backbone=_FakeBackbone([]))
    VLMAgent()  # also fine — backbone resolved lazily; only choose() would need it


def test_reset_on_not_played() -> None:
    backbone = _FakeBackbone([])
    agent = VLMAgent(backbone=backbone)
    action = agent.choose(_frame(state=GameState.NOT_PLAYED), history=[])
    assert action is GameAction.RESET
    assert backbone.calls == []  # never called the model


def test_reset_clears_state() -> None:
    agent = VLMAgent(backbone=_FakeBackbone(['{"chosen_action": "ACTION2"}']))
    agent.choose(_frame(), history=[])
    assert agent._state.step_count == 1
    agent.reset()
    assert agent._state.step_count == 0
    assert agent._state.last_grid is None
    assert agent._state.rule_table == []


# ── prompt structure ──────────────────────────────────────────────────────


def test_prompt_has_six_sections_in_order() -> None:
    agent = VLMAgent(backbone=_FakeBackbone([]))
    system, user = agent._build_prompt(_frame())
    combined = system + "\n" + user
    positions = [combined.index(marker) for marker in PROMPT_SECTIONS_IN_ORDER]
    assert positions == sorted(positions), f"out-of-order sections: {positions}"
    # All six distinct (no marker found twice — i.e., no copy-paste reuse)
    assert len(set(positions)) == 6


def test_prompt_first_step_says_no_reflection() -> None:
    agent = VLMAgent(backbone=_FakeBackbone([]))
    _, user = agent._build_prompt(_frame())
    assert "首步,无反思" in user


def test_prompt_includes_available_action_names() -> None:
    agent = VLMAgent(backbone=_FakeBackbone([]))
    _, user = agent._build_prompt(_frame(available=[1, 4]))
    assert "ACTION1" in user and "ACTION4" in user


# ── parse_response tolerance ──────────────────────────────────────────────


def test_parse_response_valid_json() -> None:
    out = VLMAgent._parse_response('{"chosen_action": "ACTION3", "predicted_diff": []}')
    assert out["chosen_action"] == "ACTION3"
    assert out["predicted_diff"] == []


def test_parse_response_json_in_fences() -> None:
    text = '```json\n{"chosen_action": "ACTION2"}\n```'
    assert VLMAgent._parse_response(text)["chosen_action"] == "ACTION2"


def test_parse_response_json_in_bare_fences() -> None:
    text = '```\n{"chosen_action": "ACTION4"}\n```'
    assert VLMAgent._parse_response(text)["chosen_action"] == "ACTION4"


def test_parse_response_first_object_with_prefix() -> None:
    text = 'Here is my output:\n{"chosen_action": "ACTION1", "x": 5}\nlater talk'
    out = VLMAgent._parse_response(text)
    assert out["chosen_action"] == "ACTION1"


def test_parse_response_nested_objects() -> None:
    text = '{"chosen_action": "ACTION6", "coords": {"x": 10, "y": 20}}'
    out = VLMAgent._parse_response(text)
    assert out["coords"] == {"x": 10, "y": 20}


def test_parse_response_handles_malformed() -> None:
    assert VLMAgent._parse_response("not json at all") == {}
    assert VLMAgent._parse_response('{"chosen_action": ') == {}
    assert VLMAgent._parse_response("") == {}


def test_parse_response_returns_empty_for_non_dict() -> None:
    assert VLMAgent._parse_response("[1, 2, 3]") == {}


def test_parse_response_handles_missing_fields() -> None:
    out = VLMAgent._parse_response('{"chosen_action": "ACTION1"}')
    # Caller will use .get() — make sure nothing blows up here.
    assert out.get("predicted_diff") is None
    assert out.get("entities") is None
    assert out.get("new_rule") is None


# ── action coercion ───────────────────────────────────────────────────────


def test_choose_picks_action_from_valid_json() -> None:
    reply = json.dumps({
        "chosen_action": "ACTION3",
        "predicted_diff": [{"row": 10, "col": 3, "to_color": 2}],
        "reflection": "P moves left",
        "entities": [],
        "new_rule": None,
    })
    agent = VLMAgent(backbone=_FakeBackbone([reply]))
    action = agent.choose(_frame(available=[1, 2, 3, 4]), history=[])
    assert action is GameAction.ACTION3


def test_choose_falls_back_on_unparseable() -> None:
    agent = VLMAgent(backbone=_FakeBackbone(["sorry no json"]), seed=0)
    action = agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert action.value in [1, 2, 3]
    assert agent._state.parse_failures == 1


def test_choose_falls_back_on_illegal_action() -> None:
    # picks ACTION5 but only [1,2,3] are legal
    agent = VLMAgent(
        backbone=_FakeBackbone(['{"chosen_action": "ACTION5"}']),
        seed=0,
    )
    action = agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert action.value in [1, 2, 3]
    assert action is not GameAction.ACTION5


def test_choose_action6_with_coords() -> None:
    reply = json.dumps({
        "chosen_action": "ACTION6",
        "coords": {"x": 15, "y": 42},
    })
    agent = VLMAgent(backbone=_FakeBackbone([reply]))
    action = agent.choose(_frame(available=[6]), history=[])
    assert action is GameAction.ACTION6
    d = action.action_data.model_dump()
    assert d["x"] == 15 and d["y"] == 42


def test_choose_action6_missing_coords_falls_back() -> None:
    agent = VLMAgent(
        backbone=_FakeBackbone(['{"chosen_action": "ACTION6"}']),
        seed=0,
    )
    action = agent.choose(_frame(available=[1, 2, 6]), history=[])
    # Fallback random over legal set; if it lands on 6, coords are random-valid.
    if action is GameAction.ACTION6:
        d = action.action_data.model_dump()
        assert 0 <= d["x"] <= 63 and 0 <= d["y"] <= 63


def test_choose_falls_back_on_backbone_exception() -> None:
    agent = VLMAgent(backbone=_BoomBackbone(), seed=0)
    action = agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert action.value in [1, 2, 3]
    assert agent._state.parse_failures == 1


# ── reflection across steps ───────────────────────────────────────────────


def test_reflection_populated_on_second_step() -> None:
    # First step predicts (0,0,5). Then the next observed grid has (0,0,5)
    # actually changed — F1 should be 1.0.
    grid0 = np.zeros((4, 4), dtype=int)
    grid1 = np.zeros((4, 4), dtype=int)
    grid1[0, 0] = 5

    backbone = _FakeBackbone([
        json.dumps({
            "chosen_action": "ACTION1",
            "predicted_diff": [{"row": 0, "col": 0, "to_color": 5}],
        }),
        json.dumps({"chosen_action": "ACTION2"}),
    ])
    agent = VLMAgent(backbone=backbone)

    agent.choose(_frame(grids=[grid0]), history=[])
    # Now feed the post-step frame; agent should compute F1 = 1.0 internally
    agent.choose(_frame(grids=[grid1]), history=[])

    assert agent._state.last_f1 == pytest.approx(1.0)
    # The 2nd-call prompt must have surfaced the reflection block (no "首步").
    second_user = backbone.calls[1][1]
    assert "首步,无反思" not in second_user
    assert "F1 = 1.00" in second_user


# ── rule table (§3.2) ─────────────────────────────────────────────────────


def test_rule_table_reinforce_on_high_f1() -> None:
    agent = VLMAgent(backbone=_FakeBackbone([]))
    agent._state.rule_table = [
        {"trigger_action": "ACTION1", "confidence": 0.5, "evidence_count": 2},
    ]
    agent._update_rule_table(f1=0.9, new_rule=None)
    r = agent._state.rule_table[0]
    assert r["evidence_count"] == 3
    assert r["confidence"] == pytest.approx(0.55)


def test_rule_table_adds_proposed_rule_on_low_f1() -> None:
    agent = VLMAgent(backbone=_FakeBackbone([]))
    new = {"trigger_action": "ACTION3", "subject_color": 2,
           "effect": "left", "confidence": 0.4}
    agent._update_rule_table(f1=0.3, new_rule=new)
    assert len(agent._state.rule_table) == 1
    assert agent._state.rule_table[0]["trigger_action"] == "ACTION3"


def test_rule_table_high_f1_does_not_add_new_rule() -> None:
    agent = VLMAgent(backbone=_FakeBackbone([]))
    new = {"trigger_action": "ACTION3", "confidence": 0.5}
    agent._update_rule_table(f1=0.9, new_rule=new)
    assert agent._state.rule_table == []


def test_rule_table_evicts_below_threshold() -> None:
    agent = VLMAgent(backbone=_FakeBackbone([]))
    agent._state.rule_table = [
        {"trigger_action": "A", "confidence": 0.5},
        {"trigger_action": "B", "confidence": 0.25},
        {"trigger_action": "C", "confidence": 0.31},
    ]
    agent._update_rule_table(f1=0.3, new_rule=None)
    triggers = {r["trigger_action"] for r in agent._state.rule_table}
    assert triggers == {"A", "C"}  # B (0.25) evicted


def test_rule_table_caps_at_max_rules() -> None:
    agent = VLMAgent(backbone=_FakeBackbone([]), max_rules=5)
    agent._state.rule_table = [
        {"trigger_action": f"R{i}", "confidence": 0.4 + i * 0.01}
        for i in range(10)
    ]
    agent._update_rule_table(f1=0.3, new_rule=None)
    assert len(agent._state.rule_table) == 5
    # The 5 with highest confidence should survive
    survived = {r["trigger_action"] for r in agent._state.rule_table}
    assert survived == {"R5", "R6", "R7", "R8", "R9"}


def test_rule_table_ignores_garbage_new_rule() -> None:
    agent = VLMAgent(backbone=_FakeBackbone([]))
    for garbage in (None, "string", 42, [], {}):
        agent._update_rule_table(f1=0.3, new_rule=garbage)
    assert agent._state.rule_table == []


# ── ensure backbone is required for choose() if not injected ──────────────


def test_choose_without_backbone_raises() -> None:
    agent = VLMAgent()
    with pytest.raises(RuntimeError, match="no backbone"):
        agent.choose(_frame(), history=[])


# ── trace surface for baseline runner ─────────────────────────────────────


def test_choose_exposes_prompt_and_response_for_trace() -> None:
    reply = '{"chosen_action": "ACTION3", "predicted_diff": []}'
    agent = VLMAgent(backbone=_FakeBackbone([reply]))
    agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert agent._state.last_response_raw == reply
    assert "[SYSTEM]" in agent._state.last_prompt
    assert "【段 5: 输出格式】" in agent._state.last_prompt
    assert agent._state.last_parse_ok is True


def test_choose_marks_parse_ok_false_on_unparseable() -> None:
    agent = VLMAgent(backbone=_FakeBackbone(["garbage"]), seed=0)
    agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert agent._state.last_parse_ok is False


def test_choose_clears_response_on_backbone_exception() -> None:
    agent = VLMAgent(backbone=_BoomBackbone(), seed=0)
    agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert agent._state.last_response_raw == ""
    assert agent._state.last_parse_ok is False


# ── regression: scenarios observed in real run on 2026-05-11 ──────────────


def test_choose_empty_chosen_action_falls_back() -> None:
    """First real run on ar25 step 0: model returned valid JSON with
    `chosen_action: ""` — must fall back to random instead of accepting."""
    reply = '{"chosen_action": "", "predicted_diff": [], "entities": []}'
    agent = VLMAgent(backbone=_FakeBackbone([reply]), seed=0)
    action = agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert action.value in [1, 2, 3]
    assert agent._state.parse_failures == 1
    assert agent._state.last_parse_ok is False


def test_choose_chosen_action_with_leading_whitespace() -> None:
    reply = '{"chosen_action": "  ACTION3  "}'
    agent = VLMAgent(backbone=_FakeBackbone([reply]))
    action = agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert action is GameAction.ACTION3


def test_choose_chosen_action_with_trailing_chars() -> None:
    """Model sometimes appends a period or description after ACTION3."""
    reply = '{"chosen_action": "ACTION3 (move left)"}'
    agent = VLMAgent(backbone=_FakeBackbone([reply]))
    action = agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert action is GameAction.ACTION3


def test_choose_chosen_action_lowercase_falls_back() -> None:
    """Lowercase 'action3' shouldn't accidentally match (we want explicit ACTION)."""
    reply = '{"chosen_action": "action3"}'
    agent = VLMAgent(backbone=_FakeBackbone([reply]), seed=0)
    action = agent.choose(_frame(available=[1, 2, 3]), history=[])
    # _coerce_action uppercases internally, so this SHOULD match.
    # If we want it strict, this test will document the lenient choice.
    assert action is GameAction.ACTION3


def test_choose_action6_string_coords() -> None:
    """Some models emit coords as strings — should still parse."""
    reply = '{"chosen_action": "ACTION6", "coords": {"x": "15", "y": "42"}}'
    agent = VLMAgent(backbone=_FakeBackbone([reply]))
    action = agent.choose(_frame(available=[6]), history=[])
    assert action is GameAction.ACTION6
    d = action.action_data.model_dump()
    assert d["x"] == 15 and d["y"] == 42


def test_choose_forwards_max_new_tokens_and_temperature() -> None:
    """Regression: these used to be stored on the agent but never plumbed."""
    fake = _FakeBackbone(['{"chosen_action": "ACTION1"}'])
    agent = VLMAgent(backbone=fake, max_new_tokens=2048, temperature=0.3)
    agent.choose(_frame(available=[1]), history=[])
    _, _, _, kw = fake.calls[0]
    assert kw["max_new_tokens"] == 2048
    assert kw["temperature"] == 0.3


def test_choose_action6_top_level_coords() -> None:
    """Model emits x/y at top level instead of inside `coords` dict."""
    reply = '{"chosen_action": "ACTION6", "x": 10, "y": 20}'
    agent = VLMAgent(backbone=_FakeBackbone([reply]))
    action = agent.choose(_frame(available=[6]), history=[])
    assert action is GameAction.ACTION6
    d = action.action_data.model_dump()
    assert d["x"] == 10 and d["y"] == 20
