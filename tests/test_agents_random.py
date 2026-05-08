"""Tests for arc_agent.agents.random.RandomAgent."""
from __future__ import annotations

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from arc_agent.agents.random import RandomAgent
from arc_agent.runner import Agent


def _frame(state: GameState, available: list[int] | None = None) -> FrameDataRaw:
    return FrameDataRaw(
        game_id="test",
        state=state,
        levels_completed=0,
        win_levels=1,
        available_actions=available or [1, 2, 3, 4, 5, 6, 7],
    )


def test_random_emits_reset_on_not_played() -> None:
    agent = RandomAgent(seed=0)
    action = agent.choose(_frame(GameState.NOT_PLAYED), history=[])
    assert action is GameAction.RESET


def test_random_emits_reset_on_game_over() -> None:
    agent = RandomAgent(seed=0)
    action = agent.choose(_frame(GameState.GAME_OVER), history=[])
    assert action is GameAction.RESET


def test_random_avoids_reset_when_playing() -> None:
    agent = RandomAgent(seed=0)
    # Sample many to be confident RESET is never picked when state=NOT_FINISHED
    for _ in range(200):
        action = agent.choose(_frame(GameState.NOT_FINISHED), history=[])
        assert action is not GameAction.RESET


def test_random_action6_carries_xy_coords() -> None:
    agent = RandomAgent(seed=42)
    # Force ACTION6 by sampling until it fires; with 7 simple + 1 complex it'll come quickly
    for _ in range(200):
        action = agent.choose(_frame(GameState.NOT_FINISHED), history=[])
        if action is GameAction.ACTION6:
            data = action.action_data.model_dump()
            assert 0 <= data["x"] <= 63
            assert 0 <= data["y"] <= 63
            return
    raise AssertionError("RandomAgent never emitted ACTION6 in 200 trials (RNG bug?)")


def test_random_satisfies_agent_protocol() -> None:
    assert isinstance(RandomAgent(), Agent)
