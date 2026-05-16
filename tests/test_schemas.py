"""Validate ARCHITECTURE_AGENTS §1 A2 schemas with `jsonschema`."""
from __future__ import annotations

import pytest

jsonschema = pytest.importorskip("jsonschema")

from arc_agent.schemas import (
    A2_OUTPUT_SCHEMA,
    ACTION_ENUM,
    ENTITY_TYPE_ENUM,
    SHAPE_ENUM,
    WORLD_MODEL_SCHEMA,
)


def test_a2_minimal_output_validates() -> None:
    jsonschema.validate({"chosen_action": "ACTION3"}, A2_OUTPUT_SCHEMA)


def test_a2_full_output_validates() -> None:
    payload = {
        "entities": [
            {"shape": "rectangle", "color": 2, "count": 1,
             "type": "player", "position": [10, 3]},
        ],
        "reflection": "moved left",
        "predicted_diff": [{"row": 10, "col": 2, "to_color": 2}],
        "chosen_action": "ACTION3",
        "coords": {"x": 5, "y": 7},
        "new_rule": {"trigger_action": "ACTION3", "subject_color": 2,
                     "effect": "left", "confidence": 0.5},
    }
    jsonschema.validate(payload, A2_OUTPUT_SCHEMA)


def test_a2_caps_entities_at_8() -> None:
    payload = {
        "chosen_action": "ACTION1",
        "entities": [
            {"shape": "single", "color": 1, "count": 1,
             "type": "unknown", "position": [0, 0]}
            for _ in range(9)
        ],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, A2_OUTPUT_SCHEMA)


def test_a2_shape_is_enum() -> None:
    payload = {
        "chosen_action": "ACTION1",
        "entities": [
            {"shape": "L-shaped object that looks like a snake",
             "color": 1, "count": 1, "type": "player", "position": [0, 0]},
        ],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, A2_OUTPUT_SCHEMA)


def test_a2_action_enum_blocks_typos() -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"chosen_action": "action3"}, A2_OUTPUT_SCHEMA)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"chosen_action": "MOVE_LEFT"}, A2_OUTPUT_SCHEMA)


def test_a2_coord_range_enforced() -> None:
    payload = {"chosen_action": "ACTION6", "coords": {"x": 99, "y": 5}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, A2_OUTPUT_SCHEMA)


def test_a2_new_rule_can_be_null() -> None:
    jsonschema.validate(
        {"chosen_action": "ACTION1", "new_rule": None},
        A2_OUTPUT_SCHEMA,
    )


# ── WorldModel schema ─────────────────────────────────────────────────────


def test_world_model_minimal() -> None:
    jsonschema.validate(
        {"rules": [], "entities": [], "goal": "explore"},
        WORLD_MODEL_SCHEMA,
    )


def test_world_model_caps() -> None:
    bad_rules = {
        "rules": ["r"] * 11,
        "entities": [],
        "goal": "x",
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad_rules, WORLD_MODEL_SCHEMA)
    bad_entities = {
        "rules": [],
        "entities": ["e"] * 9,
        "goal": "x",
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad_entities, WORLD_MODEL_SCHEMA)


def test_enums_are_non_empty() -> None:
    assert SHAPE_ENUM and ENTITY_TYPE_ENUM and ACTION_ENUM
    assert len(ACTION_ENUM) == 7
