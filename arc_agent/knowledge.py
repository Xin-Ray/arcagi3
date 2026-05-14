"""Knowledge — the shared message passed between Action Agent and
Reflection Agent in the v3.2 dual-agent loop.

Per `docs/arch_v3_2_zh.md` §3, a `Knowledge` instance persists across
rounds within one game (reset only when a new game_id starts). The
Action Agent reads it; the Reflection Agent mutates it via deltas
produced after every env.step.

Fields:
  rounds_played / rounds_won  — round counters
  action_semantics            — {"ACTION1": "moves the red 1x1 up 1 cell", ...}
  goal_hypothesis             — one-sentence goal guess
  goal_confidence             — "low" | "medium" | "high"
  rules                       — short patterns observed (cap 10, dedup)
  failed_strategies           — high-level strategies that didn't work (cap 5)
  round_history               — one line per finished round
  current_alert               — short message shown at the TOP of the next
                                Action Agent prompt; mostly empty.

Delta shape (produced by Reflection):
  {
    "action_semantics_update": {"ACTION3": "..."},
    "goal_hypothesis_update": "..." | None,
    "goal_confidence_update": "low|medium|high" | None,
    "rules_append": ["..."],
    "failed_strategies_append": ["..."],
    "current_alert": "...",
  }
Missing keys are treated as "no change". `merged_with_delta` returns a
NEW Knowledge instance so callers can keep snapshots.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

_VALID_CONFIDENCE = ("low", "medium", "high")
_RULES_CAP = 10
_FAILED_CAP = 5
_ROUND_HISTORY_CAP = 20
_SHORT_TEXT_CAP = 200  # per-field char cap to keep prompts bounded

# Reject these literal strings as goal_hypothesis_update. Reflection
# sometimes writes them despite the SYSTEM prompt forbidding it -- orchestrator
# enforces here so the bad value never reaches knowledge.goal_hypothesis.
# Case-insensitive, stripped, exact match. R1 in docs/ref_v3_2_dataflow_zh.md.
_GOAL_SENTINELS: frozenset[str] = frozenset({
    "unknown", "none", "n/a", "na", "tbd", "exploring", "?", "",
    "no idea", "uncertain", "still learning", "to be determined",
})


def _is_goal_sentinel(value: Any) -> bool:
    """True if `value` is a sentinel like 'unknown' that should be REJECTED
    as a goal_hypothesis update (kept the existing hypothesis instead)."""
    if value is None:
        return True
    return str(value).strip().lower() in _GOAL_SENTINELS


# R6: action-described "goals" are misclassified Reflection outputs.
# A goal MUST describe a target state ("reach the top edge", "match red dots
# to red targets"), NOT an action's effect ("ACTION1 should move up").
# Pattern caught: starts with "ACTION" OR contains "should move/advance/click".
import re as _re   # local alias so module top doesn't reorder imports
_ACTION_PREFIX_RE = _re.compile(r"^\s*action[1-7]\b", _re.IGNORECASE)
_ACTION_VERB_PATTERNS = (
    "should move", "should advance", "should click", "should push",
    "should be tried", "should be used",
)


def _is_action_described_goal(value: Any) -> bool:
    """True if the candidate goal looks like an action description (R6)."""
    if value is None:
        return False
    s = str(value).strip()
    if not s:
        return False
    if _ACTION_PREFIX_RE.match(s):
        return True
    low = s.lower()
    return any(pat in low for pat in _ACTION_VERB_PATTERNS)


# R4: detect rules / failed_strategies that contradict an existing positive
# action_semantics entry. Reflection sometimes hallucinates "ACTION_X has no
# effect" right after confirming ACTION_X moves things; without this filter,
# the action_mask later blocks the working action.
_ACTION_TOKEN_RE = _re.compile(r"\bACTION([1-7])\b", _re.IGNORECASE)
_NEGATION_PHRASES = (
    "no effect", "no observable effect", "ineffective",
    "never changes", "didn't work", "doesn't work",
    "did not work", "does not work", "no-op", "no op",
    # "anywhere in / at" is Reflection's go-to failed_strategy phrasing
    # (e.g. "ACTION6 anywhere in the right half") -- treat as negation
    # for R4 consistency with action_mask's regex.
    "anywhere in", "anywhere at", "anywhere on",
)
_POSITIVE_SEMANTIC_HINTS = (
    "moves", "move", "shifts", "shifted",
    "advances", "advanced", "rotates", "rotated",
    "places", "placed", "drops", "dropped",
    "up", "down", "left", "right",
    "cells", "cell",
)


def _is_negative_about_action(text: str) -> Optional[str]:
    """If `text` says some ACTION_X is ineffective, return that action name
    (e.g. 'ACTION1'). Returns None otherwise."""
    if not text:
        return None
    low = text.lower()
    if not any(neg in low for neg in _NEGATION_PHRASES):
        return None
    m = _ACTION_TOKEN_RE.search(text)
    if not m:
        return None
    return f"ACTION{m.group(1)}"


def _has_positive_semantic(action_semantics: dict[str, str], action: str) -> bool:
    """True if action_semantics[action] looks POSITIVE (mentions a direction
    or 'moves'). Used by R4 to detect contradictions."""
    sem = action_semantics.get(action)
    if not sem:
        return False
    low = sem.lower()
    # If the semantic itself says "no effect", that's NOT positive
    if any(neg in low for neg in _NEGATION_PHRASES):
        return False
    return any(hint in low for hint in _POSITIVE_SEMANTIC_HINTS)


def _clip(s: Any, cap: int = _SHORT_TEXT_CAP) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    return s[:cap]


@dataclass
class Knowledge:
    """Cross-round persistent knowledge for one game_id."""

    game_id: str = ""
    rounds_played: int = 0
    rounds_won: int = 0

    action_semantics: dict[str, str] = field(default_factory=dict)

    goal_hypothesis: str = ""
    goal_confidence: str = "low"

    rules: list[str] = field(default_factory=list)
    failed_strategies: list[str] = field(default_factory=list)
    round_history: list[str] = field(default_factory=list)

    current_alert: str = ""

    # ── factories ────────────────────────────────────────────────────────

    @classmethod
    def empty(cls, game_id: str = "") -> "Knowledge":
        return cls(game_id=game_id)

    # ── serialization ────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Knowledge":
        # Be tolerant: ignore unknown keys, fill in defaults.
        return cls(
            game_id=str(d.get("game_id", "")),
            rounds_played=int(d.get("rounds_played", 0)),
            rounds_won=int(d.get("rounds_won", 0)),
            action_semantics=dict(d.get("action_semantics", {})),
            goal_hypothesis=str(d.get("goal_hypothesis", "")),
            goal_confidence=_coerce_confidence(d.get("goal_confidence", "low")),
            rules=list(d.get("rules", [])),
            failed_strategies=list(d.get("failed_strategies", [])),
            round_history=list(d.get("round_history", [])),
            current_alert=str(d.get("current_alert", "")),
        )

    # ── rendering for Action Agent prompt ────────────────────────────────

    def render(self) -> str:
        """Multi-block text used inside the [KNOWLEDGE] section of the
        Action Agent USER prompt. See `prompts_v3_2.build_action_user_prompt`.

        The alert is NOT rendered here — the orchestrator places it in its
        own [REFLECTION ALERT] block when non-empty (see §5.3).
        """
        lines: list[str] = []
        lines.append(
            f"  rounds: {self.rounds_played} played, {self.rounds_won} won"
        )
        if self.action_semantics:
            lines.append("  action_semantics:")
            for name in sorted(self.action_semantics):
                lines.append(f"    {name}: {self.action_semantics[name]}")
        else:
            lines.append("  action_semantics: (nothing learned yet)")
        if self.goal_hypothesis:
            lines.append(
                f"  goal_hypothesis ({self.goal_confidence}): {self.goal_hypothesis}"
            )
        else:
            lines.append("  goal_hypothesis: (unknown - still exploring)")
        if self.rules:
            lines.append("  rules:")
            for r in self.rules:
                lines.append(f"    - {r}")
        if self.failed_strategies:
            lines.append("  failed_strategies (do NOT repeat):")
            for s in self.failed_strategies:
                lines.append(f"    - {s}")
        if self.round_history:
            lines.append("  round_history:")
            for h in self.round_history[-5:]:
                lines.append(f"    - {h}")
        return "\n".join(lines)

    def render_alert(self) -> str:
        """Render the [REFLECTION ALERT] block (caller checks non-empty)."""
        return f"  {self.current_alert}"

    # ── merge a Reflection delta ─────────────────────────────────────────

    def merged_with_delta(self, delta: Optional[dict[str, Any]]) -> "Knowledge":
        """Return a NEW Knowledge with delta applied.

        Tolerant of missing keys, None values, and bad types — bad inputs
        are silently dropped so a single garbled Reflection output cannot
        wipe out accumulated knowledge.
        """
        if not isinstance(delta, dict):
            return self._copy()

        new = self._copy()

        # action_semantics_update — per-key overwrite
        sem_upd = delta.get("action_semantics_update") or {}
        if isinstance(sem_upd, dict):
            for k, v in sem_upd.items():
                if isinstance(k, str) and v is not None:
                    new.action_semantics[k] = _clip(v)

        # R5: prospective failed_strategies set (existing + to-be-appended)
        # used to cross-check goal_hypothesis_update below. Reflection
        # sometimes writes a failed_strategies-style string into the
        # goal_hypothesis field; this guard drops it instead of letting
        # the wrong-direction hypothesis pollute downstream Action prompts.
        prospective_failed_lower: set[str] = {
            s.strip().lower() for s in new.failed_strategies
        }
        for s in delta.get("failed_strategies_append") or []:
            s_str = _clip(s).strip()
            if s_str:
                prospective_failed_lower.add(s_str.lower())

        # goal_hypothesis_update — replace only when it passes all three
        # quality gates. A goal MUST describe a target state, not a sentinel
        # ("unknown"), not a failed strategy, and not an action description.
        # R1: reject sentinel placeholders.
        # R5: reject failed_strategies cross-pollution.
        # R6: reject action-described goals ("ACTION_X should ...").
        goal_upd = delta.get("goal_hypothesis_update")
        if (goal_upd is not None
                and not _is_goal_sentinel(goal_upd)
                and not _is_action_described_goal(goal_upd)):
            candidate = str(goal_upd).strip()
            if candidate.lower() not in prospective_failed_lower:
                new.goal_hypothesis = _clip(goal_upd)

        # goal_confidence_update — replace if valid
        conf_upd = delta.get("goal_confidence_update")
        if conf_upd is not None:
            new.goal_confidence = _coerce_confidence(conf_upd, fallback=new.goal_confidence)

        # rules_append — append + dedup + cap. R4: drop rules that
        # contradict an existing positive action_semantics entry. Reflection
        # sometimes hallucinates "ACTION_X has no effect" right after
        # confirming X works -- this filter prevents the bad rule from
        # later masking the working action.
        for r in delta.get("rules_append") or []:
            r_clip = _clip(r)
            if not r_clip or r_clip in new.rules:
                continue
            contradicted_action = _is_negative_about_action(r_clip)
            if (contradicted_action
                    and _has_positive_semantic(new.action_semantics, contradicted_action)):
                continue   # R4 drop
            new.rules.append(r_clip)
        if len(new.rules) > _RULES_CAP:
            new.rules = new.rules[-_RULES_CAP:]

        # failed_strategies_append — same pattern, smaller cap. R4 also
        # applies here (a failed_strategy mentioning ACTION_X with negation
        # phrasing while we have positive action_semantics for X).
        for s in delta.get("failed_strategies_append") or []:
            s_clip = _clip(s)
            if not s_clip or s_clip in new.failed_strategies:
                continue
            contradicted_action = _is_negative_about_action(s_clip)
            if (contradicted_action
                    and _has_positive_semantic(new.action_semantics, contradicted_action)):
                continue   # R4 drop
            new.failed_strategies.append(s_clip)
        if len(new.failed_strategies) > _FAILED_CAP:
            new.failed_strategies = new.failed_strategies[-_FAILED_CAP:]

        # current_alert — OVERWRITE (the next step's Action Agent sees this)
        # Note: empty string means "clear the alert"; only overwrite when
        # the key is explicitly present so missing key keeps existing alert.
        if "current_alert" in delta:
            alert = delta.get("current_alert")
            new.current_alert = _clip(alert) if alert is not None else ""

        return new

    def append_round_summary(self, line: str) -> None:
        """Append a one-line round summary; in-place (orchestrator calls
        this at end-of-round)."""
        line = _clip(line, cap=300)
        if line:
            self.round_history.append(line)
            if len(self.round_history) > _ROUND_HISTORY_CAP:
                self.round_history = self.round_history[-_ROUND_HISTORY_CAP:]

    def _copy(self) -> "Knowledge":
        return Knowledge(
            game_id=self.game_id,
            rounds_played=self.rounds_played,
            rounds_won=self.rounds_won,
            action_semantics=dict(self.action_semantics),
            goal_hypothesis=self.goal_hypothesis,
            goal_confidence=self.goal_confidence,
            rules=list(self.rules),
            failed_strategies=list(self.failed_strategies),
            round_history=list(self.round_history),
            current_alert=self.current_alert,
        )


def _coerce_confidence(value: Any, fallback: str = "low") -> str:
    if isinstance(value, str) and value.lower() in _VALID_CONFIDENCE:
        return value.lower()
    return fallback


__all__ = ["Knowledge"]
