"""Tests for arc_agent.runner."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from arcengine import FrameDataRaw, GameAction, GameState

from arc_agent.runner import Agent, play_one


class _CountingAgent:
    """Agent that always picks ACTION1 and counts how many times it was called."""

    def __init__(self) -> None:
        self.calls = 0

    def choose(
        self, latest: FrameDataRaw, history: list[FrameDataRaw]
    ) -> GameAction:
        self.calls += 1
        return GameAction.ACTION1


def _frame(state: GameState, levels: int = 0) -> FrameDataRaw:
    return FrameDataRaw(
        game_id="test",
        state=state,
        levels_completed=levels,
        win_levels=3,
    )


def test_play_one_returns_error_on_make_none() -> None:
    arc = MagicMock()
    arc.make.return_value = None
    result = play_one(arc, _CountingAgent(), "ls20", "card_x", max_actions=10)
    assert result["error"].startswith("make()")


def test_play_one_stops_at_max_actions() -> None:
    env = MagicMock()
    env.reset.return_value = _frame(GameState.NOT_FINISHED)
    env.step.return_value = _frame(GameState.NOT_FINISHED)

    arc = MagicMock()
    arc.make.return_value = env

    agent = _CountingAgent()
    result = play_one(arc, agent, "ls20", "card_x", max_actions=5)
    assert result["actions"] == 5
    assert agent.calls == 5
    assert result["final_state"] == "NOT_FINISHED"


def test_play_one_stops_on_win() -> None:
    env = MagicMock()
    env.reset.return_value = _frame(GameState.NOT_FINISHED)
    # 2 NOT_FINISHED then WIN
    env.step.side_effect = [
        _frame(GameState.NOT_FINISHED),
        _frame(GameState.NOT_FINISHED),
        _frame(GameState.WIN, levels=3),
    ]

    arc = MagicMock()
    arc.make.return_value = env

    agent = _CountingAgent()
    result = play_one(arc, agent, "ls20", "card_x", max_actions=99)
    assert result["actions"] == 3
    assert result["final_state"] == "WIN"
    assert result["levels_completed"] == 3


def test_counting_agent_satisfies_protocol() -> None:
    assert isinstance(_CountingAgent(), Agent)
