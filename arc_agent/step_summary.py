"""Per-step bundle consumed by the Reflection Agent (v3.2).

Per `docs/arch_v3_2_zh.md` §4.3, after each `env.step` the orchestrator
packages one step's worth of observations into a `StepSummary` and feeds
its text rendering to the Reflection Agent. The rendering is meant to be
short (<1k tokens) and SELF-CONTAINED — Reflection should not need to
re-derive anything from OutcomeLog.

Also exports `compute_matches_reasoning(reasoning, prev_grid, curr_grid,
primary_direction, frame_changed)` returning one of:

    "YES"      reasoning's claim matches outcome
    "PARTIAL"  some agreement, some disagreement
    "NO"       reasoning predicted a specific effect that did NOT happen
    "N/A"      reasoning made no falsifiable claim

The orchestrator stuffs this back into the Reflection prompt so the
Reflection model doesn't have to re-judge alignment from raw text.

This module is pure (no env, no network, no PIL). It is the seam between
the v3 perception/memory layer and the v3.2 Reflection layer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

# ── direction lexicon for matches_reasoning ───────────────────────────────

_DIRECTION_WORDS: dict[str, str] = {
    "up": "UP", "upward": "UP", "north": "UP", "top": "UP",
    "down": "DOWN", "downward": "DOWN", "south": "DOWN", "bottom": "DOWN",
    "left": "LEFT", "west": "LEFT",
    "right": "RIGHT", "east": "RIGHT",
}

# Words suggesting "I expect nothing to happen / no change"
_NOOP_HINT_WORDS = ("no-op", "no op", "no effect", "nothing", "no change",
                    "shouldn't change", "won't change")


@dataclass
class StepSummary:
    """A self-contained view of one step for the Reflection Agent.

    Built by orchestrator AFTER env.step but BEFORE calling Reflection.
    All values are precomputed scalars/strings/lists; rendering is plain
    text with no LLM-side derivation expected.
    """

    step: int
    action: str                              # e.g. "ACTION3" or "ACTION6"
    action_coords: Optional[tuple[int, int]] = None   # only for ACTION6
    reasoning: str = ""                      # action agent's pre-step reasoning

    # Outcome flags
    frame_changed: bool = False
    primary_direction: Optional[str] = None  # "UP" / "DOWN" / "LEFT" / "RIGHT" / "UP+LEFT" / None
    primary_distance: int = 0

    # Object-level deltas (short human strings, max 5 lines)
    object_deltas: list[str] = field(default_factory=list)

    # Cumulative signals
    no_op_streak: int = 0
    state_revisit_count: int = 1   # how many times current frame_hash has appeared

    # Auto-judged by orchestrator
    matches_reasoning: str = "N/A"   # YES / PARTIAL / NO / N/A

    # Last K (action, changed) tuples for short context
    recent_steps: list[tuple[str, bool, Optional[str]]] = field(default_factory=list)

    def render(self) -> str:
        """Produce the [STEP] / [OUTCOME] / [JUDGMENT] block bundle that
        goes into the Reflection USER prompt.

        Layout (matches §4.3 prompt example):

            [ACTION AGENT'S REASONING]
              "<reasoning>"

            [ACTION AGENT'S CHOICE]
              ACTION6 (12, 30)

            [ACTUAL OUTCOME]
              frame_changed: True
              primary_direction: UP (distance=3)
              object delta:
                obj_005 (red 1x1) APPEARED at (12, 30)
              no_op_streak: 0
              state_revisit_count: 1
              matches_reasoning: YES
        """
        lines: list[str] = []

        # Reasoning
        r = (self.reasoning or "(none)").strip().replace("\n", " ")
        if len(r) > 200:
            r = r[:200] + "..."
        lines.append("[ACTION AGENT'S REASONING]")
        lines.append(f"  \"{r}\"")
        lines.append("")

        # Action chosen
        lines.append("[ACTION AGENT'S CHOICE]")
        if self.action_coords is not None:
            lines.append(
                f"  {self.action} ({self.action_coords[0]}, {self.action_coords[1]})"
            )
        else:
            lines.append(f"  {self.action}")
        lines.append("")

        # Outcome
        lines.append("[ACTUAL OUTCOME]")
        lines.append(f"  frame_changed: {self.frame_changed}")
        if self.primary_direction:
            lines.append(
                f"  primary_direction: {self.primary_direction} "
                f"(distance={self.primary_distance})"
            )
        if self.object_deltas:
            lines.append("  object delta:")
            for ln in self.object_deltas[:5]:
                lines.append(f"    {ln}")
        elif self.frame_changed:
            lines.append("  object delta: (frame changed but no tracked-object movement)")
        else:
            lines.append("  object delta: (none)")
        lines.append(f"  no_op_streak: {self.no_op_streak}")
        lines.append(f"  state_revisit_count: {self.state_revisit_count}")
        lines.append(f"  matches_reasoning: {self.matches_reasoning}")

        # Short recent context
        if self.recent_steps:
            lines.append("")
            lines.append("[LAST STEPS — short context]")
            for (act, ch, dir_) in self.recent_steps[-3:]:
                tag = "CHANGED" if ch else "no-op"
                if ch and dir_:
                    lines.append(f"  {act} -> {tag} ({dir_})")
                else:
                    lines.append(f"  {act} -> {tag}")

        return "\n".join(lines)


# ── matches_reasoning judge ───────────────────────────────────────────────


def _extract_direction_claims(text: str) -> set[str]:
    """Return the set of canonical direction tags found in `text`."""
    if not text:
        return set()
    lower = text.lower()
    claims: set[str] = set()
    # Word-boundary search to avoid "uphold" -> UP
    for word, tag in _DIRECTION_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", lower):
            claims.add(tag)
    return claims


def _claims_noop(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(hint in lower for hint in _NOOP_HINT_WORDS)


def _outcome_directions(primary_direction: Optional[str]) -> set[str]:
    if not primary_direction:
        return set()
    # primary_direction may be a single tag or "UP+LEFT"
    return set(primary_direction.split("+"))


_OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def compute_matches_reasoning(
    reasoning: str,
    *,
    frame_changed: bool,
    primary_direction: Optional[str],
) -> str:
    """Judge reasoning-vs-outcome alignment.

    Falsifiable claims are EITHER a direction word OR a no-op hint.
    Otherwise we return "N/A" — Reflection can still pick up on vibes
    in its own prompt.
    """
    claim_dirs = _extract_direction_claims(reasoning)
    claims_noop = _claims_noop(reasoning)

    # Nothing falsifiable
    if not claim_dirs and not claims_noop:
        return "N/A"

    out_dirs = _outcome_directions(primary_direction)

    # no-op claims
    if claims_noop and not claim_dirs:
        return "YES" if not frame_changed else "NO"
    if claims_noop and frame_changed:
        # Predicted no-op but something happened. Even if a direction was
        # also mentioned, the no-op claim is wrong.
        if claim_dirs & out_dirs:
            return "PARTIAL"
        return "NO"

    # Direction claims only
    if claim_dirs and not frame_changed:
        # Predicted movement, got nothing — clear miss
        return "NO"

    # Both predicted and observed direction — compare sets
    matched = claim_dirs & out_dirs
    if matched == claim_dirs and matched == out_dirs:
        return "YES"
    if matched:
        return "PARTIAL"
    # Predicted direction, observed a different / opposite one
    if any(_OPPOSITE.get(d) in out_dirs for d in claim_dirs):
        return "NO"
    return "PARTIAL"   # changed somewhere unexpected


# ── ObjectMemory delta → human strings ───────────────────────────────────


def object_delta_lines(matches: Iterable, max_lines: int = 5) -> list[str]:
    """Convert `align_objects` matches into short human lines.

    Lazy import so this module stays free of arcengine. Tolerates duck-typed
    inputs (anything with .type / .delta / .color attributes).
    """
    out: list[str] = []
    for m in matches:
        kind = getattr(m, "type", None)
        if kind == "unchanged" or kind is None:
            continue
        color = getattr(m, "color", None)
        if kind == "moved":
            delta = getattr(m, "delta", None) or {}
            dy = int(delta.get("dy", 0))
            dx = int(delta.get("dx", 0))
            tags = []
            if dy < 0: tags.append("UP")
            elif dy > 0: tags.append("DOWN")
            if dx < 0: tags.append("LEFT")
            elif dx > 0: tags.append("RIGHT")
            tag = "+".join(tags) or "STILL"
            dist = max(abs(dy), abs(dx))
            out.append(f"obj#{getattr(m, 'before_id', '?')} (color {color}) "
                       f"moved {tag} {dist} cell(s)")
        elif kind == "appeared":
            out.append(f"obj#{getattr(m, 'after_id', '?')} (color {color}) APPEARED")
        elif kind == "disappeared":
            out.append(f"obj#{getattr(m, 'before_id', '?')} (color {color}) DISAPPEARED")
        elif kind == "recolored":
            out.append(f"obj#{getattr(m, 'before_id', '?')} RECOLORED -> color {color}")
        else:
            out.append(f"obj#? {kind}")
        if len(out) >= max_lines:
            break
    return out


def grid_changed(prev_grid: Optional[np.ndarray],
                 curr_grid: Optional[np.ndarray]) -> bool:
    """True if prev != curr (any cell)."""
    if prev_grid is None or curr_grid is None:
        return False
    if prev_grid.shape != curr_grid.shape:
        return True
    return not np.array_equal(prev_grid, curr_grid)


__all__ = [
    "StepSummary",
    "compute_matches_reasoning",
    "object_delta_lines",
    "grid_changed",
]
