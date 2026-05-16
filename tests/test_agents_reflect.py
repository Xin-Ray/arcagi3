"""Unit tests for arc_agent.agents.reflect — A3 and A4 (no GPU)."""
from __future__ import annotations

import json

import numpy as np
import pytest
from arcengine import FrameDataRaw, GameAction, GameState

from arc_agent.agents.reflect import (
    PlayReflectAgent,
    PlayReflectMistakesAgent,
    REFLECT_EVERY,
)


class _FakeBackbone:
    """Stub backbone — returns next reply from a list."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[dict] = []

    def generate(self, image, prompt: str, *, system: str = "", **kw) -> str:
        self.calls.append({"prompt": prompt, "system": system, "kw": kw})
        if not self._replies:
            return "ACTION1"
        return self._replies.pop(0)


def _frame(
    state: GameState = GameState.NOT_FINISHED,
    available: list[int] | None = None,
    grid: np.ndarray | None = None,
    lvl: int = 0,
) -> FrameDataRaw:
    f = FrameDataRaw(
        game_id="ls20",
        state=state,
        levels_completed=lvl,
        win_levels=3,
        available_actions=available or [1, 2, 3, 4, 5, 7],
    )
    f.frame = [grid if grid is not None else np.zeros((4, 4), dtype=int)]
    return f


# ── basic construction / reset ─────────────────────────────────────────────


def test_construct_no_gpu() -> None:
    PlayReflectAgent(backbone=_FakeBackbone([]))
    PlayReflectAgent()  # lazy


def test_invalid_reflect_every() -> None:
    with pytest.raises(ValueError):
        PlayReflectAgent(backbone=_FakeBackbone([]), reflect_every=0)


def test_reset_clears_state() -> None:
    agent = PlayReflectAgent(backbone=_FakeBackbone(["ACTION1"] * 3))
    agent.choose(_frame(), history=[])
    assert agent._state.step_count == 1
    agent.reset()
    assert agent._state.step_count == 0
    assert agent._state.world_model.rules == []


# ── reflection cadence ────────────────────────────────────────────────────


def test_reflection_not_triggered_on_first_step() -> None:
    bb = _FakeBackbone(["ACTION1"])
    agent = PlayReflectAgent(backbone=bb)
    agent.choose(_frame(), history=[])
    assert agent._state.reflections_run == 0
    assert len(bb.calls) == 1   # only the play call


def test_reflection_triggers_every_K_steps() -> None:
    """Cadence-only test — uses distinct grids per step so the no-op
    stuck-trigger doesn't add extra reflections."""
    K_small = 2
    reflect_reply = json.dumps({"rules": ["r"], "entities": ["e"], "goal": "g"})
    backbone = _FakeBackbone([
        "ACTION1",        # step 0 play
        "ACTION1",        # step 1 play
        reflect_reply,    # step 2 reflect
        "ACTION1",        # step 2 play
        "ACTION1",        # step 3 play
        reflect_reply,    # step 4 reflect
        "ACTION1",        # step 4 play
        "ACTION1",        # step 5 play
    ])
    agent = PlayReflectAgent(backbone=backbone, reflect_every=K_small)
    # Distinct grids so frame_changed=True every step.
    for i in range(6):
        g = np.zeros((4, 4), dtype=int)
        g[0, 0] = (i % 15) + 1
        agent.choose(_frame(grid=g), history=[])
    assert agent._state.reflections_run == 2
    assert agent._state.world_model.rules == ["r"]


def test_reflection_triggers_on_stuck_frame_streak() -> None:
    """3 consecutive unchanged frames must force a reflection regardless of K."""
    bb = _FakeBackbone([
        "ACTION1",
        "ACTION1",
        "ACTION1",
        # Reflection after the 3rd no-op (step 3, since we use the same grid each time)
        json.dumps({"rules": ["stuck"], "entities": [], "goal": "find exit"}),
        "ACTION1",
    ])
    agent = PlayReflectAgent(backbone=bb, reflect_every=100)
    same_grid = np.zeros((4, 4), dtype=int)
    for _ in range(4):
        agent.choose(_frame(grid=same_grid), history=[])
    assert agent._state.reflections_run >= 1
    assert agent._state.world_model.goal == "find exit"


def test_reflection_failure_keeps_old_world_model() -> None:
    bb = _FakeBackbone([
        "ACTION1",
        "ACTION1",
        "not json",       # reflection returns garbage
        "ACTION1",
    ])
    agent = PlayReflectAgent(backbone=bb, reflect_every=2)
    agent._state.world_model.goal = "preexisting"
    for _ in range(3):
        agent.choose(_frame(), history=[])
    # Reflection ran but failed to parse — WM should not be overwritten.
    assert agent._state.world_model.goal == "preexisting"


# ── play parsing ──────────────────────────────────────────────────────────


def test_play_parses_action_token() -> None:
    bb = _FakeBackbone(["ACTION3"])
    agent = PlayReflectAgent(backbone=bb)
    a = agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert a is GameAction.ACTION3


def test_play_falls_back_on_garbage() -> None:
    bb = _FakeBackbone(["zzzz"])
    agent = PlayReflectAgent(backbone=bb, seed=0)
    a = agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert a.value in [1, 2, 3]


def test_world_model_in_play_prompt() -> None:
    bb = _FakeBackbone(["ACTION1"])
    agent = PlayReflectAgent(backbone=bb)
    agent._state.world_model.goal = "reach green corner"
    agent.choose(_frame(), history=[])
    play_prompt = bb.calls[0]["prompt"]
    assert "reach green corner" in play_prompt
    assert "Legal actions" in play_prompt


# ── A4: mistakes integration ──────────────────────────────────────────────


def test_a4_mistakes_appear_in_play_prompt() -> None:
    """An illegal action on step 0 triggers a reflection before step 1's
    play call. The play prompt that *follows* the reflection must surface
    the auto-detected mistake."""
    bb = _FakeBackbone([
        "ACTION5",          # step 0 play — illegal in [1,2]
        json.dumps({"rules": [], "entities": [], "goal": "x"}),  # step 1 reflect
        "ACTION1",          # step 1 play
    ])
    agent = PlayReflectMistakesAgent(backbone=bb, seed=0, reflect_every=999)
    agent.choose(_frame(available=[1, 2]), history=[])
    agent.choose(_frame(available=[1, 2]), history=[])
    # Last call is the post-reflection play. It should have the mistake block.
    play_prompt = bb.calls[-1]["prompt"]
    assert "RECENT MISTAKES" in play_prompt
    assert " - " in play_prompt.split("RECENT MISTAKES", 1)[1]


def test_a3_no_mistakes_block_in_prompt() -> None:
    bb = _FakeBackbone(["ACTION1", "ACTION1"])
    agent = PlayReflectAgent(backbone=bb, seed=0)
    agent.choose(_frame(available=[1, 2]), history=[])
    agent.choose(_frame(available=[1, 2]), history=[])
    second_prompt = bb.calls[1]["prompt"]
    assert "RECENT MISTAKES" not in second_prompt


# ── trace surface for baseline runner ─────────────────────────────────────


def test_trace_surface_after_play() -> None:
    bb = _FakeBackbone(["ACTION1"])
    agent = PlayReflectAgent(backbone=bb)
    agent.choose(_frame(), history=[])
    assert agent._state.last_response_raw == "ACTION1"
    assert agent._state.last_parse_ok is True
    assert agent._state.last_predicted_diff is None
