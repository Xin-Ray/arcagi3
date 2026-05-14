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

        # goal_hypothesis_update — replace if non-null
        goal_upd = delta.get("goal_hypothesis_update")
        if goal_upd is not None and str(goal_upd).strip():
            new.goal_hypothesis = _clip(goal_upd)

        # goal_confidence_update — replace if valid
        conf_upd = delta.get("goal_confidence_update")
        if conf_upd is not None:
            new.goal_confidence = _coerce_confidence(conf_upd, fallback=new.goal_confidence)

        # rules_append — append + dedup + cap
        for r in delta.get("rules_append") or []:
            r_clip = _clip(r)
            if r_clip and r_clip not in new.rules:
                new.rules.append(r_clip)
        if len(new.rules) > _RULES_CAP:
            new.rules = new.rules[-_RULES_CAP:]

        # failed_strategies_append — same pattern, smaller cap
        for s in delta.get("failed_strategies_append") or []:
            s_clip = _clip(s)
            if s_clip and s_clip not in new.failed_strategies:
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
