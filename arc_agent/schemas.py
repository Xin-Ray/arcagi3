"""JSON schemas for constrained-decoding (`docs/ARCHITECTURE_AGENTS.md` §1 A2).

The bp35 incident on 2026-05-12 (76+ unparseable rows) was caused by the
backbone running away on free-form `entities` and `shape` fields. Two fixes,
both encoded as JSON Schema here:

- `entities` is `max_items: 8` — caps the worst-case token blowup.
- `shape` is an enum so the model can't burn budget on prose ("an
  L-shaped object that sits in the top-left and may be the player").

`A2_OUTPUT_SCHEMA` is the full A2 turn schema (entities + reflection +
predicted_diff + chosen_action + new_rule). `WORLD_MODEL_SCHEMA` is the
typed dict produced by the Reflection agent in A3 (§1.A3 — `rules`,
`entities`, `goal`).

These schemas are consumed by `lm-format-enforcer` / `outlines` inside
`arc_agent.vlm_backbone.generate` when `constrained_schema=` is passed.
The schemas are also useful as a checklist for hand-written prompts —
keep them in sync with the prompt templates.
"""
from __future__ import annotations

from typing import Any

# ── enums (kept as module constants so prompts can stringify them) ────────

SHAPE_ENUM: tuple[str, ...] = (
    "rectangle",
    "L",
    "T",
    "blob",
    "single",
)

ENTITY_TYPE_ENUM: tuple[str, ...] = (
    "player",
    "wall",
    "goal",
    "enemy",
    "movable_obj",
    "decoration",
    "unknown",
)

ACTION_ENUM: tuple[str, ...] = tuple(f"ACTION{i}" for i in range(1, 8))


# ── A2 schema (full VLMAgent JSON output) ────────────────────────────────

_A2_ENTITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "shape":    {"type": "string", "enum": list(SHAPE_ENUM)},
        "color":    {"type": "integer", "minimum": 0, "maximum": 15},
        "count":    {"type": "integer", "minimum": 0, "maximum": 64},
        "type":     {"type": "string", "enum": list(ENTITY_TYPE_ENUM)},
        "function": {"type": "string", "maxLength": 40},
        "position": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0, "maximum": 63},
            "minItems": 2,
            "maxItems": 2,
        },
    },
    "required": ["shape", "color", "count", "type", "position"],
    "additionalProperties": False,
}


_A2_DIFF_CELL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "row":      {"type": "integer", "minimum": 0, "maximum": 63},
        "col":      {"type": "integer", "minimum": 0, "maximum": 63},
        "to_color": {"type": "integer", "minimum": 0, "maximum": 15},
    },
    "required": ["row", "col", "to_color"],
    "additionalProperties": False,
}


_A2_NEW_RULE_SCHEMA: dict[str, Any] = {
    "type": ["object", "null"],
    "properties": {
        "trigger_action": {"type": "string", "enum": list(ACTION_ENUM)},
        "subject_color":  {"type": "integer", "minimum": 0, "maximum": 15},
        "effect":         {"type": "string", "maxLength": 40},
        "confidence":     {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


A2_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": _A2_ENTITY_SCHEMA,
            "maxItems": 8,   # §1 A2: hard cap to kill the bp35 blowup
        },
        "reflection": {"type": "string", "maxLength": 80},
        "predicted_diff": {
            "type": "array",
            "items": _A2_DIFF_CELL_SCHEMA,
            "maxItems": 20,
        },
        "chosen_action": {"type": "string", "enum": list(ACTION_ENUM)},
        "coords": {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "minimum": 0, "maximum": 63},
                "y": {"type": "integer", "minimum": 0, "maximum": 63},
            },
            "required": ["x", "y"],
            "additionalProperties": False,
        },
        "new_rule": _A2_NEW_RULE_SCHEMA,
    },
    "required": ["chosen_action"],
    "additionalProperties": False,
}


# ── WorldModel schema (A3 reflection output) ──────────────────────────────

WORLD_MODEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rules": {
            "type": "array",
            "items": {"type": "string", "maxLength": 80},
            "maxItems": 10,
        },
        "entities": {
            "type": "array",
            "items": {"type": "string", "maxLength": 60},
            "maxItems": 8,
        },
        "goal": {"type": "string", "maxLength": 120},
    },
    "required": ["rules", "entities", "goal"],
    "additionalProperties": False,
}
