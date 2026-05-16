"""Unit tests for `arc_agent.agents.vlm_lite.VLMAgentLite` (A1 + A1+h).

No GPU needed — backbone is a stub returning canned strings.
"""
from __future__ import annotations

import numpy as np
import pytest
from arcengine import FrameDataRaw, GameAction, GameState

from arc_agent.agents.vlm_lite import VLMAgentLite


class _FakeBackbone:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple] = []

    def generate(self, image, prompt: str, *, system: str = "", **kw) -> str:
        self.calls.append((image, prompt, system, kw))
        if not self._replies:
            return "ACTION1"
        return self._replies.pop(0)


class _BoomBackbone:
    def generate(self, image, prompt, *, system="", **kw):
        raise RuntimeError("boom")


def _frame(
    state: GameState = GameState.NOT_FINISHED,
    available: list[int] | None = None,
    grids: list[np.ndarray] | None = None,
) -> FrameDataRaw:
    f = FrameDataRaw(
        game_id="ls20",
        state=state,
        levels_completed=0,
        win_levels=3,
        available_actions=available or [1, 2, 3, 4, 5, 7],
    )
    f.frame = grids or [np.zeros((8, 8), dtype=int)]
    return f


# ── construction / reset ──────────────────────────────────────────────────


def test_construct_no_gpu_imports() -> None:
    VLMAgentLite(backbone=_FakeBackbone([]))
    VLMAgentLite()  # lazy


def test_negative_history_rejected() -> None:
    with pytest.raises(ValueError):
        VLMAgentLite(backbone=_FakeBackbone([]), history=-1)


def test_reset_clears_state() -> None:
    agent = VLMAgentLite(backbone=_FakeBackbone(["ACTION2"]))
    agent.choose(_frame(), history=[])
    assert agent._state.step_count == 1
    agent.reset()
    assert agent._state.step_count == 0
    assert agent._state.action_history == []
    assert agent._state.last_chosen_action is None


# ── basic action picking ──────────────────────────────────────────────────


def test_returns_reset_on_not_played() -> None:
    bb = _FakeBackbone([])
    agent = VLMAgentLite(backbone=bb)
    a = agent.choose(_frame(state=GameState.NOT_PLAYED), history=[])
    assert a is GameAction.RESET
    assert bb.calls == []


def test_returns_reset_on_game_over() -> None:
    bb = _FakeBackbone([])
    agent = VLMAgentLite(backbone=bb)
    a = agent.choose(_frame(state=GameState.GAME_OVER), history=[])
    assert a is GameAction.RESET


def test_parses_plain_action_token() -> None:
    agent = VLMAgentLite(backbone=_FakeBackbone(["ACTION3"]))
    a = agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert a is GameAction.ACTION3
    assert agent._state.last_parse_ok is True
    assert agent._state.last_chosen_action == "ACTION3"


def test_parses_lowercase() -> None:
    agent = VLMAgentLite(backbone=_FakeBackbone(["action2"]))
    a = agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert a is GameAction.ACTION2


def test_parses_action_with_trailing_text() -> None:
    agent = VLMAgentLite(backbone=_FakeBackbone(["ACTION1 (go up)"]))
    a = agent.choose(_frame(available=[1, 2]), history=[])
    assert a is GameAction.ACTION1


def test_garbage_response_falls_back_random() -> None:
    agent = VLMAgentLite(backbone=_FakeBackbone(["zzz"]), seed=0)
    a = agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert a.value in [1, 2, 3]
    assert agent._state.parse_failures == 1
    assert agent._state.last_parse_ok is False


def test_illegal_action_falls_back_random() -> None:
    agent = VLMAgentLite(backbone=_FakeBackbone(["ACTION5"]), seed=0)
    a = agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert a.value in [1, 2, 3]
    assert a is not GameAction.ACTION5


def test_backbone_exception_falls_back() -> None:
    agent = VLMAgentLite(backbone=_BoomBackbone(), seed=0)
    a = agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert a.value in [1, 2, 3]
    assert agent._state.parse_failures == 1


# ── ACTION6 coordinate handling ───────────────────────────────────────────


def test_action6_with_coords_parses() -> None:
    agent = VLMAgentLite(backbone=_FakeBackbone(["ACTION6 12 30"]))
    a = agent.choose(_frame(available=[6]), history=[])
    assert a is GameAction.ACTION6
    d = a.action_data.model_dump()
    assert d["x"] == 12 and d["y"] == 30


def test_action6_with_comma_separator() -> None:
    agent = VLMAgentLite(backbone=_FakeBackbone(["ACTION6, x=5, y=7"]))
    a = agent.choose(_frame(available=[6]), history=[])
    assert a is GameAction.ACTION6
    d = a.action_data.model_dump()
    assert d["x"] == 5 and d["y"] == 7


def test_action6_without_coords_falls_back() -> None:
    agent = VLMAgentLite(backbone=_FakeBackbone(["ACTION6"]), seed=0)
    a = agent.choose(_frame(available=[1, 6]), history=[])
    # Either fallback returns ACTION1 (legal non-complex) or ACTION6 with random
    # coordinates from the fallback path.
    if a is GameAction.ACTION6:
        d = a.action_data.model_dump()
        assert 0 <= d["x"] <= 63 and 0 <= d["y"] <= 63
    else:
        assert a.value == 1


def test_action6_coord_out_of_range_falls_back() -> None:
    agent = VLMAgentLite(backbone=_FakeBackbone(["ACTION6 99 200"]), seed=0)
    a = agent.choose(_frame(available=[1, 6]), history=[])
    # 99 / 200 are illegal → coerce_action returns None → fallback random.
    assert agent._state.parse_failures == 1


# ── prompt / generation budget ────────────────────────────────────────────


def test_default_max_new_tokens_is_tiny() -> None:
    bb = _FakeBackbone(["ACTION1"])
    agent = VLMAgentLite(backbone=bb)
    agent.choose(_frame(), history=[])
    _, _, _, kw = bb.calls[0]
    assert kw["max_new_tokens"] == 8
    assert kw["temperature"] == 0.0


def test_prompt_does_not_contain_json_keywords() -> None:
    """A1's contract is no JSON, no entities, no predicted_diff."""
    bb = _FakeBackbone(["ACTION1"])
    agent = VLMAgentLite(backbone=bb)
    agent.choose(_frame(available=[1, 2]), history=[])
    _, prompt, _, _ = bb.calls[0]
    for forbidden in ("predicted_diff", "entities", "JSON", "json"):
        assert forbidden not in prompt, f"A1 prompt leaked '{forbidden}'"


def test_prompt_lists_legal_actions() -> None:
    bb = _FakeBackbone(["ACTION1"])
    agent = VLMAgentLite(backbone=bb)
    agent.choose(_frame(available=[1, 4]), history=[])
    _, prompt, _, _ = bb.calls[0]
    assert "ACTION1" in prompt
    assert "ACTION4" in prompt


# ── A1+h history flag ─────────────────────────────────────────────────────


def test_history_zero_does_not_inject_actions() -> None:
    bb = _FakeBackbone(["ACTION1", "ACTION2"])
    agent = VLMAgentLite(backbone=bb, history=0)
    agent.choose(_frame(available=[1, 2]), history=[])
    agent.choose(_frame(available=[1, 2]), history=[])
    second_prompt = bb.calls[1][1]
    assert "Last" not in second_prompt or "actions" not in second_prompt
    assert "ACTION1" in second_prompt  # only as a legal action, not history


def test_history_five_injects_last_actions() -> None:
    bb = _FakeBackbone(["ACTION1", "ACTION2", "ACTION3"])
    agent = VLMAgentLite(backbone=bb, history=5)
    agent.choose(_frame(available=[1, 2, 3]), history=[])
    agent.choose(_frame(available=[1, 2, 3]), history=[])
    agent.choose(_frame(available=[1, 2, 3]), history=[])
    third_prompt = bb.calls[2][1]
    assert "Last 2 actions" in third_prompt
    assert "ACTION1, ACTION2" in third_prompt


def test_history_caps_at_n() -> None:
    replies = ["ACTION1"] * 8
    bb = _FakeBackbone(replies)
    agent = VLMAgentLite(backbone=bb, history=3)
    for _ in range(7):
        agent.choose(_frame(available=[1, 2, 3]), history=[])
    last_prompt = bb.calls[-1][1]
    # Already had 6 prior actions; we ask for last 3.
    assert "Last 3 actions" in last_prompt


# ── trace surface ─────────────────────────────────────────────────────────


def test_exposes_prompt_and_response_for_trace() -> None:
    bb = _FakeBackbone(["ACTION3"])
    agent = VLMAgentLite(backbone=bb)
    agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert agent._state.last_response_raw == "ACTION3"
    assert "Legal actions" in agent._state.last_prompt
    assert agent._state.last_parse_ok is True
    # A1 never produces a predicted_diff — must stay None for the runner.
    assert agent._state.last_predicted_diff is None


def test_choose_without_backbone_raises() -> None:
    agent = VLMAgentLite()
    with pytest.raises(RuntimeError, match="no backbone"):
        agent.choose(_frame(), history=[])
