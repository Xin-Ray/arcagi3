"""Tests for `arc_agent.baseline.play_one_with_trace`.

Uses a hand-rolled FakeEnv (no `arc_agi`/`arcengine` real server) plus a
VLMAgent powered by a deterministic FakeBackbone. The combination exercises
the full trace + image + GIF pipeline without GPU or network.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from arcengine import FrameDataRaw, GameAction, GameState

from arc_agent.agents.vlm import VLMAgent
from arc_agent.baseline import GameMetrics, play_one_with_trace


# ── fixtures ──────────────────────────────────────────────────────────────


class _FakeBackbone:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple] = []

    def generate(self, image, prompt: str, *, system: str = "", **kw) -> str:
        self.calls.append((image, prompt, system, kw))
        if not self._replies:
            return '{"chosen_action": "ACTION1", "predicted_diff": []}'
        return self._replies.pop(0)


class _FakeEnv:
    """Tiny env mock: scripted sequence of FrameDataRaw on step()."""

    def __init__(
        self,
        initial: FrameDataRaw,
        sequence: list[FrameDataRaw],
    ) -> None:
        self._initial = initial
        self._sequence = list(sequence)
        self.step_calls: list[tuple] = []

    def reset(self) -> FrameDataRaw:
        return self._initial

    def step(self, action: GameAction, *, data=None, reasoning=None) -> FrameDataRaw:
        self.step_calls.append((action, data, reasoning))
        if not self._sequence:
            # default: return WIN to terminate
            return _make_frame(GameState.WIN, np.zeros((4, 4), dtype=int))
        return self._sequence.pop(0)


def _make_frame(
    state: GameState,
    grid: np.ndarray,
    *,
    levels_completed: int = 0,
    win_levels: int = 3,
    available: list[int] | None = None,
) -> FrameDataRaw:
    f = FrameDataRaw(
        game_id="test",
        state=state,
        levels_completed=levels_completed,
        win_levels=win_levels,
        available_actions=available or [1, 2, 3, 4, 5, 7],
    )
    f.frame = [grid]
    return f


# ── tests ─────────────────────────────────────────────────────────────────


def test_play_one_with_trace_produces_all_artifacts(tmp_path) -> None:
    g0 = np.zeros((4, 4), dtype=int)
    g1 = np.zeros((4, 4), dtype=int); g1[0, 0] = 5
    g2 = np.zeros((4, 4), dtype=int); g2[0, 0] = 5; g2[1, 1] = 7

    backbone = _FakeBackbone([
        '{"chosen_action": "ACTION1", "predicted_diff": [{"row": 0, "col": 0, "to_color": 5}]}',
        '{"chosen_action": "ACTION2", "predicted_diff": [{"row": 1, "col": 1, "to_color": 7}]}',
    ])
    agent = VLMAgent(backbone=backbone)
    env = _FakeEnv(
        initial=_make_frame(GameState.NOT_FINISHED, g0),
        sequence=[
            _make_frame(GameState.NOT_FINISHED, g1),
            _make_frame(GameState.WIN, g2, levels_completed=3),
        ],
    )

    out = tmp_path / "ar25"
    metrics = play_one_with_trace(env, agent, run_dir=out, game_id="ar25", fps=2)

    assert isinstance(metrics, GameMetrics)
    assert metrics.actions == 2
    assert metrics.final_state == "WIN"
    assert metrics.levels_completed == 3
    assert metrics.parse_rate == 1.0
    assert metrics.mean_f1 == pytest.approx(1.0)  # both predictions were exactly right

    # Files on disk
    assert (out / "trace.jsonl").exists()
    assert (out / "step_0000.png").exists()
    assert (out / "step_0001.png").exists()
    assert (out / "play.gif").exists()

    # Trace.jsonl has 2 rows with required keys + correct types
    lines = (out / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for raw in lines:
        row = json.loads(raw)
        assert row["game_id"] == "ar25"
        assert row["parse_ok"] is True
        assert row["chosen_action"] in {"ACTION1", "ACTION2"}
        assert row["f1"] == pytest.approx(1.0)
        assert isinstance(row["real_diff"], list) and len(row["real_diff"]) == 1


def test_play_one_with_trace_counts_parse_failures(tmp_path) -> None:
    g0 = np.zeros((4, 4), dtype=int)
    g1 = np.zeros((4, 4), dtype=int); g1[0, 0] = 5

    backbone = _FakeBackbone([
        "not json — model failed",
        '{"chosen_action": "ACTION1", "predicted_diff": [{"row": 0, "col": 0, "to_color": 5}]}',
    ])
    agent = VLMAgent(backbone=backbone, seed=0)
    env = _FakeEnv(
        initial=_make_frame(GameState.NOT_FINISHED, g0),
        sequence=[
            _make_frame(GameState.NOT_FINISHED, g0),  # fallback acted; nothing changed
            _make_frame(GameState.WIN, g1, levels_completed=3),
        ],
    )

    out = tmp_path / "x"
    metrics = play_one_with_trace(env, agent, run_dir=out, game_id="x", fps=2)

    assert metrics.actions == 2
    assert metrics.parse_rate == 0.5  # one parse fail, one ok
    # mean_f1: parse-fail counted as 0 → (0 + 1.0) / 2
    assert metrics.mean_f1 == pytest.approx(0.5)
    # mean_f1_when_parsed: only the one good step
    assert metrics.mean_f1_when_parsed == pytest.approx(1.0)

    # trace.jsonl: row 0 has parse_ok=False, predicted_diff=null, f1=null
    rows = [
        json.loads(ln)
        for ln in (out / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    assert rows[0]["parse_ok"] is False
    assert rows[0]["predicted_diff"] is None
    assert rows[0]["f1"] is None
    assert rows[1]["parse_ok"] is True


def test_play_one_with_trace_respects_max_actions(tmp_path) -> None:
    g = np.zeros((4, 4), dtype=int)
    backbone = _FakeBackbone(['{"chosen_action": "ACTION1"}'] * 20)
    agent = VLMAgent(backbone=backbone)
    env = _FakeEnv(
        initial=_make_frame(GameState.NOT_FINISHED, g),
        sequence=[_make_frame(GameState.NOT_FINISHED, g)] * 20,
    )

    out = tmp_path / "y"
    metrics = play_one_with_trace(
        env, agent, run_dir=out, game_id="y", max_actions=3, fps=2,
    )
    assert metrics.actions == 3
    assert metrics.final_state == "NOT_FINISHED"
    lines = (out / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_play_one_with_trace_skips_images_when_disabled(tmp_path) -> None:
    g = np.zeros((4, 4), dtype=int)
    backbone = _FakeBackbone(['{"chosen_action": "ACTION1"}'])
    agent = VLMAgent(backbone=backbone)
    env = _FakeEnv(
        initial=_make_frame(GameState.NOT_FINISHED, g),
        sequence=[_make_frame(GameState.WIN, g)],
    )
    out = tmp_path / "z"
    metrics = play_one_with_trace(
        env, agent, run_dir=out, game_id="z", write_images=False,
    )
    assert metrics.actions == 1
    assert not (out / "step_0000.png").exists()
    assert not (out / "play.gif").exists()
    assert metrics.gif_path is None
    # trace.jsonl is still written
    assert (out / "trace.jsonl").exists()
    # And the trace row's image_path is None
    row = json.loads((out / "trace.jsonl").read_text(encoding="utf-8").strip())
    assert row["image_path"] is None


def test_play_one_with_trace_works_with_non_vlm_agent(tmp_path) -> None:
    """Plumbing safety — should not crash if agent doesn't expose _state."""
    from arc_agent.agents.random import RandomAgent

    g = np.zeros((4, 4), dtype=int)
    env = _FakeEnv(
        initial=_make_frame(GameState.NOT_FINISHED, g),
        sequence=[_make_frame(GameState.WIN, g)],
    )
    out = tmp_path / "r"
    metrics = play_one_with_trace(
        env, RandomAgent(seed=0), run_dir=out, game_id="r", write_images=False,
    )
    assert metrics.actions == 1
    row = json.loads((out / "trace.jsonl").read_text(encoding="utf-8").strip())
    assert row["parse_ok"] is False
    assert row["predicted_diff"] is None
    assert row["prompt"] == ""
    assert row["response_raw"] == ""
