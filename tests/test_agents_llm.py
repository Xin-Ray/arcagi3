"""Tests for arc_agent.agents.llm.LLMAgent — uses an injected mock LLMClient."""
from __future__ import annotations

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from arc_agent.agents.llm import LLMAgent
from arc_agent.llm import LLMResponse


class _FakeLLM:
    """Stand-in for LLMClient that returns canned responses in order."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[str, str]] = []  # (system, user) tuples

    def complete(self, *, system: str, user: str) -> LLMResponse:
        self.calls.append((system, user))
        if not self._replies:
            return LLMResponse(text="ACTION: ACTION1")
        return LLMResponse(text=self._replies.pop(0))


def _frame(
    state: GameState = GameState.NOT_FINISHED,
    available: list[int] | None = None,
    grids: list[np.ndarray] | None = None,
) -> FrameDataRaw:
    f = FrameDataRaw(
        game_id="test_game",
        state=state,
        levels_completed=0,
        win_levels=2,
        available_actions=available or [1, 2, 3, 4, 5, 7],
    )
    if grids is not None:
        f.frame = grids
    return f


def test_llm_emits_reset_on_not_played() -> None:
    agent = LLMAgent(llm=_FakeLLM([]))
    action = agent.choose(_frame(GameState.NOT_PLAYED), history=[])
    assert action is GameAction.RESET


def test_llm_picks_action_from_reply() -> None:
    fake = _FakeLLM(["HYPOTHESIS: ...\nEXECUTE: ...\nACTION: ACTION3"])
    agent = LLMAgent(llm=fake)
    action = agent.choose(_frame(grids=[np.zeros((4, 4), dtype=int)]), history=[])
    assert action is GameAction.ACTION3
    assert len(fake.calls) == 1


def test_llm_respects_available_actions() -> None:
    # LLM picks ACTION5 but only [1, 2, 3, 4] are legal — fall back, not crash
    fake = _FakeLLM(["ACTION: ACTION5"])
    agent = LLMAgent(llm=fake, seed=0)
    action = agent.choose(
        _frame(available=[1, 2, 3, 4], grids=[np.zeros((4, 4), dtype=int)]),
        history=[],
    )
    assert action.value in [1, 2, 3, 4]
    assert action is not GameAction.ACTION5


def test_llm_parses_action6_coords() -> None:
    fake = _FakeLLM(["...\nACTION: ACTION6 x=15 y=42"])
    agent = LLMAgent(llm=fake)
    action = agent.choose(
        _frame(available=[6], grids=[np.zeros((4, 4), dtype=int)]),
        history=[],
    )
    assert action is GameAction.ACTION6
    data = action.action_data.model_dump()
    assert data["x"] == 15
    assert data["y"] == 42


def test_llm_falls_back_when_action6_missing_coords() -> None:
    # ACTION6 without x/y → invalid, fallback to random
    fake = _FakeLLM(["ACTION: ACTION6"])
    agent = LLMAgent(llm=fake, seed=0)
    action = agent.choose(
        _frame(available=[1, 2, 6], grids=[np.zeros((4, 4), dtype=int)]),
        history=[],
    )
    # could be 1, 2, or 6 (random) — but if 6, must have valid coords
    if action is GameAction.ACTION6:
        d = action.action_data.model_dump()
        assert 0 <= d["x"] <= 63 and 0 <= d["y"] <= 63


def test_llm_falls_back_on_unparseable_reply() -> None:
    fake = _FakeLLM(["I'm not going to follow your format. Bye."])
    agent = LLMAgent(llm=fake, seed=0)
    action = agent.choose(
        _frame(available=[1, 2, 3], grids=[np.zeros((4, 4), dtype=int)]),
        history=[],
    )
    assert action.value in [1, 2, 3]
    assert agent._state.parse_failures == 1


def test_llm_takes_last_action_directive_when_word_appears_earlier() -> None:
    # The word "ACTION" appears in narrative BEFORE the directive; only the trailing
    # ACTION: line should bind.
    reply = (
        "Looking at the grid, the ACTION1 key seems to move things up.\n"
        "I'll try ACTION3 next.\n"
        "ACTION: ACTION3"
    )
    fake = _FakeLLM([reply])
    agent = LLMAgent(llm=fake)
    action = agent.choose(
        _frame(available=[1, 2, 3, 4], grids=[np.zeros((4, 4), dtype=int)]),
        history=[],
    )
    assert action is GameAction.ACTION3


def test_llm_falls_back_on_api_failure() -> None:
    class _BoomLLM:
        def complete(self, *, system, user):
            raise RuntimeError("simulated 500")

    agent = LLMAgent(llm=_BoomLLM(), seed=0)
    action = agent.choose(
        _frame(available=[1, 2, 3], grids=[np.zeros((4, 4), dtype=int)]),
        history=[],
    )
    assert action.value in [1, 2, 3]
