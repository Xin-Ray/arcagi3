"""Single-episode runner + Agent protocol.

Pure library code: takes an already-constructed `Arcade` and `EnvironmentWrapper`.
No env-var reads, no file I/O — those live in `eval.py` / `agent_starter.py`.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from arc_agi import Arcade
from arcengine import FrameDataRaw, GameAction, GameState


@runtime_checkable
class Agent(Protocol):
    """An agent that picks the next GameAction given the latest frame + history."""

    def choose(
        self, latest: FrameDataRaw, history: list[FrameDataRaw]
    ) -> GameAction: ...


def play_one(
    arc: Arcade,
    agent: Agent,
    game_id: str,
    card_id: str,
    max_actions: int = 80,
    history_limit: int = 8,
) -> dict:
    """Play one episode of `game_id`. Return per-episode metrics.

    Stops when state == WIN or `max_actions` is reached. Does NOT close the
    scorecard (caller owns the lifecycle so a single card can span many games).
    """
    env = arc.make(game_id, scorecard_id=card_id)
    if env is None:
        return {"error": "make() returned None", "actions": 0}

    history: list[FrameDataRaw] = []
    latest = env.reset()
    history.append(latest)

    actions = 0
    while latest.state is not GameState.WIN and actions < max_actions:
        action = agent.choose(latest, history[-history_limit:])
        latest = env.step(
            action,
            data=action.action_data.model_dump(),
            reasoning=getattr(action, "reasoning", None),
        )
        history.append(latest)
        actions += 1

    return {
        "actions": actions,
        "final_state": latest.state.name,
        "levels_completed": latest.levels_completed,
        "total_levels": latest.win_levels,  # SDK quirk: this field is the level count
        "guid": latest.guid,
    }
