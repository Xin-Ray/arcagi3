"""Unit tests for arc_agent.agents.text_agent.TextAgent.

Backbone is faked so this runs without GPU/network.
"""
from __future__ import annotations

import numpy as np
import pytest
from arcengine import FrameDataRaw, GameAction, GameState

from arc_agent.agents.text_agent import TextAgent


class _FakeBackbone:
    """Stub: returns canned action tokens in order."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[dict] = []

    def generate(self, image, prompt, *, system="", **kw) -> str:
        self.calls.append({"image": image, "prompt": prompt, "system": system, "kw": kw})
        if not self._replies:
            return "ACTION1"
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


# ── construction ──────────────────────────────────────────────────────────


def test_construct_no_gpu() -> None:
    TextAgent(backbone=_FakeBackbone([]))
    TextAgent()   # lazy


def test_reset_clears_state() -> None:
    agent = TextAgent(backbone=_FakeBackbone(["ACTION1"]))
    agent.choose(_frame(), history=[])
    assert agent._state.step_count == 1
    agent.reset()
    assert agent._state.step_count == 0


# ── basic action picking ──────────────────────────────────────────────────


def test_picks_action_from_response() -> None:
    bb = _FakeBackbone(["ACTION3"])
    agent = TextAgent(backbone=bb)
    a = agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert a is GameAction.ACTION3


def test_returns_reset_on_not_played() -> None:
    bb = _FakeBackbone([])
    a = TextAgent(backbone=bb).choose(
        _frame(state=GameState.NOT_PLAYED), history=[])
    assert a is GameAction.RESET
    assert bb.calls == []


def test_garbage_response_falls_back() -> None:
    agent = TextAgent(backbone=_FakeBackbone(["nope"]), seed=0)
    a = agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert a.value in [1, 2, 3]
    assert agent._state.parse_failures == 1


def test_illegal_action_falls_back() -> None:
    agent = TextAgent(backbone=_FakeBackbone(["ACTION5"]), seed=0)
    a = agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert a.value in [1, 2, 3]
    assert a is not GameAction.ACTION5


# ── prompt content (v3 hard constraints) ──────────────────────────────────


def test_user_prompt_lists_v3_blocks() -> None:
    bb = _FakeBackbone(["ACTION1"])
    agent = TextAgent(backbone=bb)
    agent.choose(_frame(available=[1, 2, 3]), history=[])
    prompt = bb.calls[0]["prompt"]
    for marker in ("[STATUS]", "[ACTIVE]", "[TEXTURE]", "[ACTION", "[UNTRIED",
                   "[HISTORY", "[GOAL", "[ASK]"):
        assert marker in prompt, f"missing block: {marker}"


def test_system_does_not_leak_action_semantics() -> None:
    bb = _FakeBackbone(["ACTION1"])
    agent = TextAgent(backbone=bb)
    agent.choose(_frame(), history=[])
    system = bb.calls[0]["system"]
    forbidden = ["ACTION1=up", "ACTION2=down", "ACTION3=left", "ACTION4=right"]
    for f in forbidden:
        assert f.lower() not in system.lower()


def test_image_arg_is_none_in_text_only_mode() -> None:
    bb = _FakeBackbone(["ACTION1"])
    TextAgent(backbone=bb).choose(_frame(), history=[])
    assert bb.calls[0]["image"] is None


# ── anti-collapse ─────────────────────────────────────────────────────────


def test_collapse_triggers_diversification() -> None:
    """After 3 ACTION1 in a row, agent should try something else."""
    # 4 replies all ACTION1. The diversification should override the 4th.
    bb = _FakeBackbone(["ACTION1", "ACTION1", "ACTION1", "ACTION1"])
    agent = TextAgent(backbone=bb, seed=0)
    for _ in range(4):
        agent.choose(_frame(available=[1, 2, 3, 4]), history=[])
    # The diversification only kicks in when the model picks the SAME
    # action again. Confirm the 4th-step prompt has the [ALERT] block.
    assert "[ALERT]" in bb.calls[3]["prompt"]
    assert "ACTION1" in bb.calls[3]["prompt"].split("[ALERT]")[1]


def test_no_diversification_when_actions_diverse() -> None:
    bb = _FakeBackbone(["ACTION1", "ACTION2", "ACTION3"])
    agent = TextAgent(backbone=bb)
    for _ in range(3):
        agent.choose(_frame(available=[1, 2, 3]), history=[])
    # No collapse → no [ALERT] in the 3rd prompt
    assert "[ALERT]" not in bb.calls[2]["prompt"]


# ── outcome logging ──────────────────────────────────────────────────────


def test_outcome_log_records_changes_across_steps() -> None:
    grid0 = np.zeros((8, 8), dtype=int)
    grid1 = np.zeros((8, 8), dtype=int)
    grid1[0, 0] = 5    # something changed
    bb = _FakeBackbone(["ACTION1", "ACTION2"])
    agent = TextAgent(backbone=bb)
    agent.choose(_frame(grid=grid0), history=[])
    agent.choose(_frame(grid=grid1), history=[])
    # OutcomeLog should have at least one entry for ACTION1 with frame_changed
    log = agent._state.outcome_log
    assert log.n_tried("ACTION1") == 1
    assert log.by_action["ACTION1"][0].frame_changed is True


def test_outcome_log_records_no_op() -> None:
    same_grid = np.zeros((8, 8), dtype=int)
    bb = _FakeBackbone(["ACTION1", "ACTION1"])
    agent = TextAgent(backbone=bb)
    agent.choose(_frame(grid=same_grid), history=[])
    agent.choose(_frame(grid=same_grid), history=[])
    log = agent._state.outcome_log
    assert log.n_tried("ACTION1") == 1
    assert log.by_action["ACTION1"][0].frame_changed is False


# ── trace surface ─────────────────────────────────────────────────────────


def test_trace_surface_after_choose() -> None:
    bb = _FakeBackbone(["ACTION3"])
    agent = TextAgent(backbone=bb)
    agent.choose(_frame(available=[1, 2, 3]), history=[])
    assert agent._state.last_response_raw == "ACTION3"
    assert "[STATUS]" in agent._state.last_prompt
    assert agent._state.last_parse_ok is True
    assert agent._state.last_predicted_diff is None


def test_choose_without_backbone_raises() -> None:
    agent = TextAgent()
    with pytest.raises(RuntimeError, match="no backbone"):
        agent.choose(_frame(), history=[])
