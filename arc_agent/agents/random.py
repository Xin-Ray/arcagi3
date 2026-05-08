"""Uniform-random baseline agent. Absolute lower bound for RHAE."""
from __future__ import annotations

import random

from arcengine import FrameDataRaw, GameAction, GameState


class RandomAgent:
    """Picks `RESET` on NOT_PLAYED/GAME_OVER, otherwise uniform over the rest.

    For ACTION6 (coordinate), draws random (x, y) ∈ [0, 63]².
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose(
        self, latest: FrameDataRaw, history: list[FrameDataRaw]
    ) -> GameAction:
        if latest.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        action = self._rng.choice(
            [a for a in GameAction if a is not GameAction.RESET]
        )
        if action.is_complex():  # ACTION6 needs (x, y) on the 64×64 grid
            action.set_data(
                {"x": self._rng.randint(0, 63), "y": self._rng.randint(0, 63)}
            )
        action.reasoning = f"random pick: {action.name}"
        return action
