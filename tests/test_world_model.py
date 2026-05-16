"""Tests for arc_agent.world_model.WorldModel."""
from __future__ import annotations

from arc_agent.world_model import (
    MAX_ENTITIES,
    MAX_GOAL_CHARS,
    MAX_RULES,
    WorldModel,
)


def test_empty_world_model() -> None:
    wm = WorldModel()
    assert wm.rules == []
    assert wm.entities == []
    assert wm.goal == ""


def test_render_handles_unknown_goal() -> None:
    out = WorldModel().render()
    assert "goal: (unknown)" in out


def test_caps_enforced_on_construction() -> None:
    wm = WorldModel(
        rules=[f"r{i}" for i in range(20)],
        entities=[f"e{i}" for i in range(20)],
        goal="x" * (MAX_GOAL_CHARS + 50),
    )
    assert len(wm.rules) == MAX_RULES
    assert len(wm.entities) == MAX_ENTITIES
    assert len(wm.goal) == MAX_GOAL_CHARS


def test_from_dict_tolerant() -> None:
    wm = WorldModel.from_dict({"rules": ["a"], "entities": ["b"], "goal": "c"})
    assert wm.rules == ["a"]
    assert wm.entities == ["b"]
    assert wm.goal == "c"


def test_from_dict_drops_garbage() -> None:
    wm = WorldModel.from_dict({"rules": [None, {"x": 1}, "ok"]})
    # Non-string/number entries are dropped; "ok" survives.
    assert "ok" in wm.rules
    assert not any(r in ("None", "{'x': 1}") for r in wm.rules)


def test_from_dict_non_dict_returns_empty() -> None:
    assert WorldModel.from_dict("not a dict").to_dict() == {
        "rules": [], "entities": [], "goal": ""
    }
    assert WorldModel.from_dict(None).rules == []


def test_to_dict_roundtrip() -> None:
    wm1 = WorldModel(rules=["r"], entities=["e"], goal="g")
    wm2 = WorldModel.from_dict(wm1.to_dict())
    assert wm2.rules == ["r"] and wm2.entities == ["e"] and wm2.goal == "g"


def test_render_lists_rules_and_entities() -> None:
    wm = WorldModel(rules=["push left to move"], entities=["red 1x1"], goal="reach green")
    block = wm.render()
    assert "push left to move" in block
    assert "red 1x1" in block
    assert "reach green" in block
