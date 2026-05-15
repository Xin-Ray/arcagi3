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
_REJECTED_GOALS_CAP = 10
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


# BUG-9 fix: action_semantics MUST identify which object is acted on,
# otherwise the entry is uninformative when multiple active objects exist.
# A "subject" is any of:
#   - a color name (red / blue / yellow / ...)
#   - an obj_id pattern (obj_NNN or obj_NN)
#   - a shape descriptor (1x1, 2x2, NxM in general, "square", "rectangle", "L-shape", "pixel")
# A positive movement claim ("moves" / "shifts" / "rotates" / etc.) WITHOUT
# any of these is rejected. No-op claims ("no observable effect") pass.
_SUBJECT_COLORS = (
    "red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta",
    "pink", "brown", "white", "black", "gray", "grey", "teal",
)
_SUBJECT_SHAPE_WORDS = (
    "square", "rectangle", "line", "l-shape", "l shape", "pixel", "block",
    "dot", "row", "column",
)
_SUBJECT_OBJ_ID_RE = _re.compile(r"\bobj[_-]?\d+\b", _re.IGNORECASE)
_SUBJECT_SIZE_RE = _re.compile(r"\b\d+\s*[xX]\s*\d+\b")  # "1x1", "2 x 3"
_POSITIVE_MOTION_HINTS = (
    "moves", "move", "moved", "shifts", "shifted", "shift",
    "rotates", "rotated", "rotate",
    "advances", "advanced", "advance",
    "places", "placed", "place",
    "drops", "dropped", "drop",
)


def _has_subject(text: str) -> bool:
    """True if `text` names a concrete subject (color / obj_id / shape)."""
    if not text:
        return False
    low = text.lower()
    if any(c in low for c in _SUBJECT_COLORS):
        return True
    if _SUBJECT_OBJ_ID_RE.search(text):
        return True
    if _SUBJECT_SIZE_RE.search(text):
        return True
    if any(w in low for w in _SUBJECT_SHAPE_WORDS):
        return True
    return False


def _is_positive_movement_claim(text: str) -> bool:
    """True if `text` claims an action moves/shifts/rotates something."""
    if not text:
        return False
    low = text.lower()
    # negation phrases override — "no observable effect" should not be
    # treated as a movement claim even though "effect" is in there
    if any(neg in low for neg in _NEGATION_PHRASES):
        return False
    return any(hint in low for hint in _POSITIVE_MOTION_HINTS)


def _action_semantic_passes_subject_check(value: str) -> bool:
    """True iff `value` is either (a) not a positive movement claim, or
    (b) a positive movement claim that names a concrete subject. BUG-9 gate."""
    if not _is_positive_movement_claim(value):
        return True
    return _has_subject(value)


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

    # BUG-8 fix: goals that were proposed and later overwritten by a
    # different hypothesis. Kept so Reflection sees them and does NOT
    # re-propose. Append-only (dedup, case-insensitive); capped at
    # _REJECTED_GOALS_CAP.
    rejected_goals: list[str] = field(default_factory=list)

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
            rejected_goals=list(d.get("rejected_goals", [])),
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
        if self.rejected_goals:
            lines.append("  rejected_goals (tried and disproved, do NOT re-propose):")
            for g in self.rejected_goals:
                lines.append(f"    - {g}")
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

        # action_semantics_update — per-key overwrite.
        # BUG-9: drop entries that claim positive movement without naming a
        # concrete subject (color / obj_id / shape). "moves an active object
        # UP" is uninformative when multiple active objects exist; we keep
        # the existing entry instead of letting it be clobbered.
        sem_upd = delta.get("action_semantics_update") or {}
        if isinstance(sem_upd, dict):
            for k, v in sem_upd.items():
                if isinstance(k, str) and v is not None:
                    clipped = _clip(v)
                    if not _action_semantic_passes_subject_check(clipped):
                        continue
                    new.action_semantics[k] = clipped

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

        # goal_hypothesis_update — replace only when it passes all four
        # quality gates. A goal MUST describe a target state, not a sentinel
        # ("unknown"), not a failed strategy, not an action description, and
        # not a goal we've already rejected.
        # R1: reject sentinel placeholders.
        # R5: reject failed_strategies cross-pollution.
        # R6: reject action-described goals ("ACTION_X should ...").
        # BUG-8: reject goals already in rejected_goals (negative memory).
        rejected_lower = {g.strip().lower() for g in new.rejected_goals}
        goal_upd = delta.get("goal_hypothesis_update")
        if (goal_upd is not None
                and not _is_goal_sentinel(goal_upd)
                and not _is_action_described_goal(goal_upd)):
            candidate = str(goal_upd).strip()
            candidate_low = candidate.lower()
            if (candidate_low not in prospective_failed_lower
                    and candidate_low not in rejected_lower):
                # BUG-8: when overwriting a different, non-empty existing
                # goal, archive the old one so it isn't re-proposed later.
                clipped_new = _clip(goal_upd)
                old_goal = new.goal_hypothesis.strip()
                if (old_goal
                        and old_goal.lower() != clipped_new.lower()
                        and old_goal.lower() not in rejected_lower):
                    new.rejected_goals.append(old_goal)
                    if len(new.rejected_goals) > _REJECTED_GOALS_CAP:
                        new.rejected_goals = new.rejected_goals[-_REJECTED_GOALS_CAP:]
                new.goal_hypothesis = clipped_new

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
            rejected_goals=list(self.rejected_goals),
            current_alert=self.current_alert,
        )


def _coerce_confidence(value: Any, fallback: str = "low") -> str:
    if isinstance(value, str) and value.lower() in _VALID_CONFIDENCE:
        return value.lower()
    return fallback


__all__ = ["Knowledge"]
