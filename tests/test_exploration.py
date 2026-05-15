"""Tests for arc_agent.exploration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from arc_agent.exploration import (
    UninteractedObject,
    compute_uninteracted_objects,
    compute_untried_actions,
    render_exploration_hint,
)


# ── compute_untried_actions ────────────────────────────────────────────


class _FakeLog:
    def __init__(self, tried: list[str]) -> None:
        self._tried = set(tried)

    def untried(self, legal: list[str]) -> list[str]:
        return [a for a in legal if a not in self._tried]


def test_untried_passthrough() -> None:
    log = _FakeLog(["ACTION1", "ACTION6"])
    out = compute_untried_actions(log, ["ACTION1", "ACTION2", "ACTION6", "ACTION7"])
    assert out == ["ACTION2", "ACTION7"]


def test_untried_handles_none_log() -> None:
    """When the log is None (e.g. cold start), every legal action is untried."""
    out = compute_untried_actions(None, ["ACTION1", "ACTION2"])
    assert out == ["ACTION1", "ACTION2"]


# ── compute_uninteracted_objects ───────────────────────────────────────


@dataclass
class _Snap:
    step: int
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    color_name: str = "red"


@dataclass
class _Tracked:
    uid: str
    history: list[_Snap] = field(default_factory=list)


class _FakeObjMemory:
    def __init__(self, tracked: list[_Tracked]) -> None:
        self._t = tracked

    def alive_tracked(self) -> list[_Tracked]:
        return self._t


def _make_static_obj(uid: str, n_steps: int, color: str = "cyan") -> _Tracked:
    return _Tracked(uid=uid, history=[
        _Snap(step=i, center=(10.0, 20.0), bbox=(10, 20, 12, 22), color_name=color)
        for i in range(n_steps)
    ])


def _make_moving_obj(uid: str, n_steps: int) -> _Tracked:
    return _Tracked(uid=uid, history=[
        _Snap(step=i, center=(10.0 + i, 20.0), bbox=(10+i, 20, 12+i, 22))
        for i in range(n_steps)
    ])


def test_uninteracted_includes_static_after_min_steps() -> None:
    mem = _FakeObjMemory([_make_static_obj("obj_001", n_steps=7)])
    out = compute_uninteracted_objects(mem, min_seen_steps=5)
    assert len(out) == 1
    assert out[0].uid == "obj_001"
    assert out[0].seen_steps == 7


def test_uninteracted_excludes_movers() -> None:
    mem = _FakeObjMemory([_make_moving_obj("obj_001", n_steps=7)])
    out = compute_uninteracted_objects(mem, min_seen_steps=5)
    assert out == []


def test_uninteracted_requires_min_seen_steps() -> None:
    """An object alive for only 3 steps is still 'new' — don't surface it
    until it's been around for a while."""
    mem = _FakeObjMemory([_make_static_obj("obj_001", n_steps=3)])
    out = compute_uninteracted_objects(mem, min_seen_steps=5)
    assert out == []


def test_uninteracted_mix() -> None:
    """Static + moving + new objects: only the matured static one passes."""
    mem = _FakeObjMemory([
        _make_static_obj("obj_001", n_steps=10),    # static, mature  -> KEEP
        _make_moving_obj("obj_002", n_steps=10),    # moved
        _make_static_obj("obj_003", n_steps=2),     # new
    ])
    out = compute_uninteracted_objects(mem, min_seen_steps=5)
    assert [u.uid for u in out] == ["obj_001"]


def test_uninteracted_handles_none_memory() -> None:
    assert compute_uninteracted_objects(None) == []


# ── render_exploration_hint ────────────────────────────────────────────


def test_render_empty_inputs_returns_empty_string() -> None:
    assert render_exploration_hint([], [], None) == ""


def test_render_only_untried() -> None:
    text = render_exploration_hint(["ACTION3", "ACTION5"], [], None)
    assert "[EXPLORATION HINT]" in text
    assert "ACTION3" in text and "ACTION5" in text
    assert "Objects that have NEVER" not in text
    assert "STUCK" not in text


def test_render_only_uninteracted() -> None:
    obj = UninteractedObject(
        uid="obj_007", color_name="cyan", bbox=(10, 20, 12, 22),
        center=(11, 21), seen_steps=12,
    )
    text = render_exploration_hint([], [obj], None)
    assert "obj_007" in text
    assert "cyan" in text
    assert "Actions you have NOT" not in text


def test_render_with_stuck_signal_first() -> None:
    """STUCK line must appear above untried/uninteracted blocks."""
    text = render_exploration_hint(
        ["ACTION3"], [], stuck_reason="masked hash repeated 3x",
    )
    assert text.index("STUCK") < text.index("ACTION3")


def test_render_truncates_object_list() -> None:
    """More than max_objects -> shows first max + 'and N more'."""
    objs = [UninteractedObject(
        uid=f"obj_{i:03d}", color_name="red", bbox=(0, 0, 1, 1),
        center=(0, 0), seen_steps=10,
    ) for i in range(8)]
    text = render_exploration_hint([], objs, None, max_objects=3)
    assert "obj_000" in text and "obj_002" in text
    assert "obj_005" not in text
    assert "5 more" in text   # 8 - 3 = 5 hidden


def test_render_includes_closing_nudge() -> None:
    """The closing line tells the LLM what to do with the info."""
    text = render_exploration_hint(["ACTION5"], [], None)
    low = text.lower()
    assert "prefer" in low or "unexplored" in low
