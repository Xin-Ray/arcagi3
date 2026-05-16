"""WorldModel — persistent A3 state carried across steps (per episode).

Spec from `docs/ARCHITECTURE_AGENTS.md` §1 A3:

    rules:    list[str]   # ≤ 10 short bullets
    entities: list[str]   # ≤ 8 short bullets
    goal:     str         # one sentence

The reflection agent rewrites this dict every K steps. Caps are enforced
on construction *and* on every merge so a runaway reflection can't blow
the prompt budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MAX_RULES = 10
MAX_ENTITIES = 8
MAX_GOAL_CHARS = 120


@dataclass
class WorldModel:
    rules: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    goal: str = ""

    def __post_init__(self) -> None:
        self._clamp()

    def _clamp(self) -> None:
        self.rules = [str(r)[:80] for r in self.rules[:MAX_RULES]]
        self.entities = [str(e)[:60] for e in self.entities[:MAX_ENTITIES]]
        self.goal = str(self.goal)[:MAX_GOAL_CHARS]

    def to_dict(self) -> dict[str, Any]:
        return {"rules": list(self.rules),
                "entities": list(self.entities),
                "goal": self.goal}

    @classmethod
    def from_dict(cls, payload: Any) -> "WorldModel":
        """Tolerant parse — drops anything that doesn't fit the schema."""
        if not isinstance(payload, dict):
            return cls()
        rules = payload.get("rules") or []
        entities = payload.get("entities") or []
        goal = payload.get("goal") or ""
        rules = [str(r) for r in rules if isinstance(r, (str, int, float))]
        entities = [str(e) for e in entities if isinstance(e, (str, int, float))]
        if not isinstance(goal, str):
            goal = str(goal)
        return cls(rules=rules, entities=entities, goal=goal)

    def render(self) -> str:
        """One short block for Play-agent prompts."""
        out = ["[WORLD MODEL]"]
        out.append(f"goal: {self.goal or '(unknown)'}")
        if self.entities:
            out.append("entities:")
            for e in self.entities:
                out.append(f"  - {e}")
        if self.rules:
            out.append("rules:")
            for r in self.rules:
                out.append(f"  - {r}")
        return "\n".join(out)
