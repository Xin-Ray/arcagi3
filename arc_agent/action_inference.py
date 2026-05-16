"""Per-episode bookkeeping of `(action, outcome)` pairs and human-readable
summary of what each ACTION does in the current game.

Architecture v3 §4: the LLM should not have to grind through "ACTION1
was tried at step 5 with outcome A, step 12 with outcome B, ...".
Instead, we pre-aggregate and feed a single line like:

  "ACTION1: tried 3 times, 2 moved obj_000 UP by 3 cells, 1 no-op"

This module does the aggregation and renders it as text.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

# Action enum names — kept in module so we can iterate without
# importing arcengine (lets tests run without the SDK installed).
ALL_ACTIONS = tuple(f"ACTION{i}" for i in range(1, 8))


@dataclass
class StepOutcome:
    """One row in the OutcomeLog. Recorded by the agent after env.step."""
    step: int
    action: str
    legal: bool
    frame_changed: bool
    n_active_changed: int   # how many ACTIVE objects had non-unchanged matches
    primary_direction: Optional[str] = None   # "UP" / "DOWN" / "LEFT" / "RIGHT" / "UP+LEFT"
    primary_distance: int = 0
    notes: str = ""


@dataclass
class OutcomeLog:
    """Per-episode log of (action -> list of StepOutcome)."""
    by_action: dict[str, list[StepOutcome]] = field(default_factory=lambda: defaultdict(list))
    all_steps: list[StepOutcome] = field(default_factory=list)

    def record(self, outcome: StepOutcome) -> None:
        self.by_action.setdefault(outcome.action, []).append(outcome)
        self.all_steps.append(outcome)

    def n_tried(self, action: str) -> int:
        return len(self.by_action.get(action, []))

    def n_changed(self, action: str) -> int:
        return sum(1 for o in self.by_action.get(action, []) if o.frame_changed)

    def untried(self, legal_actions: list[str]) -> list[str]:
        return [a for a in legal_actions if a not in self.by_action]

    def reset(self) -> None:
        self.by_action.clear()
        self.all_steps.clear()


def summarize_action(action: str, outcomes: list[StepOutcome]) -> str:
    """One human-friendly line summarizing what `action` did.

    Examples:
        "tried 3x: 2x moved UP 3 cells, 1x no-op"
        "tried 1x: illegal in this state"
        "tried 5x: all no-op"
    """
    if not outcomes:
        return "(untried)"
    n = len(outcomes)
    legal_count = sum(1 for o in outcomes if o.legal)
    changed_count = sum(1 for o in outcomes if o.frame_changed)
    if legal_count == 0:
        return f"tried {n}x: always illegal"
    if changed_count == 0:
        return f"tried {n}x: all no-op"
    # Group by (direction, distance)
    bucket: Counter = Counter()
    for o in outcomes:
        if not o.frame_changed:
            bucket[("no-op", 0)] += 1
            continue
        bucket[(o.primary_direction or "?", o.primary_distance)] += 1
    parts = []
    for (dir_, dist), count in bucket.most_common():
        if dir_ == "no-op":
            parts.append(f"{count}x no-op")
        elif dir_ == "?":
            parts.append(f"{count}x changed (no clear direction)")
        else:
            parts.append(f"{count}x moved {dir_} {dist} cell(s)")
    return f"tried {n}x: " + ", ".join(parts)


def render_action_block(log: OutcomeLog, legal_actions: list[str]) -> str:
    """Build the [ACTION] block of the user prompt.

    Per legal action lists total + recent-window stats so stuck patterns
    are visible:  "ACTION3: total 10x (8 RIGHT, 2 no-op); recent 5x: 2 no-op"
    """
    lines = []
    for action in legal_actions:
        outcomes = log.by_action.get(action, [])
        if not outcomes:
            lines.append(f"  {action}: UNTRIED")
            continue
        total_summary = summarize_action(action, outcomes)
        recent = recent_no_op_summary(log, action, window=5)
        if recent["n_recent"] >= 3:
            line = (f"  {action}: total {total_summary}; "
                    f"recent {recent['n_recent']}x: "
                    f"{recent['n_no_op']} no-op "
                    f"({recent['no_op_rate']:.0%})")
        else:
            line = f"  {action}: {total_summary}"
        lines.append(line)
    return "\n".join(lines)


def render_untried_block(log: OutcomeLog, legal_actions: list[str]) -> str:
    """Short block: just the untried list."""
    untried = log.untried(legal_actions)
    if not untried:
        return "(none — every legal action has been tried)"
    return ", ".join(untried)


def render_history_tail(log: OutcomeLog, n: int = 5) -> str:
    """Last N steps in compact form for [HISTORY] block."""
    tail = log.all_steps[-n:]
    if not tail:
        return "(none)"
    return "\n".join(
        f"  step {o.step}: {o.action} -> "
        f"{'CHANGED' if o.frame_changed else 'no-op'}"
        + (f" ({o.primary_direction} {o.primary_distance})" if o.primary_direction else "")
        for o in tail
    )


def detect_collapse(log: OutcomeLog, window: int = 3) -> bool:
    """True if the last `window` actions were all the same.

    Used by `TextAgent` to trigger the anti-collapse fallback.
    """
    tail = log.all_steps[-window:]
    if len(tail) < window:
        return False
    return len({o.action for o in tail}) == 1


def detect_stuck(log: OutcomeLog,
                 frame_hashes: list[int],
                 *,
                 no_op_streak_threshold: int = 5,
                 state_revisit_window: int = 20,
                 state_revisit_threshold: int = 3,
                 alternating_min_repeats: int = 3,
                 recent_window: int = 5,
                 recent_no_op_threshold: float = 0.6,
                 ) -> tuple[bool, str]:
    """Multi-condition stuck detector. Returns (stuck, reason).

    Each condition fires independently; returns first match.

    Conditions (in priority order):
      1. 3 consecutive same action  (existing detect_collapse behaviour)
      2. no-op streak >= no_op_streak_threshold
      3. same frame_hash visited >= N times in last K steps
      4. alternating pattern A,B,A,B,A,B for >= 6 steps
      5. any action's recent N occurrences are >= recent_no_op_threshold no-op
    """
    if detect_collapse(log, window=3):
        last = log.all_steps[-1].action
        return True, f"{last} picked 3 times in a row"

    # No-op streak
    streak = 0
    for o in reversed(log.all_steps):
        if not o.frame_changed:
            streak += 1
        else:
            break
    if streak >= no_op_streak_threshold:
        return True, f"no-op streak: {streak} consecutive steps with no frame change"

    # State revisit
    if frame_hashes:
        recent = frame_hashes[-state_revisit_window:]
        counts = Counter(recent)
        h, c = counts.most_common(1)[0]
        if c >= state_revisit_threshold:
            return True, (f"current state hash visited {c} times in last "
                          f"{len(recent)} steps — you are in a loop")

    # Alternating 2-action pattern (A,B,A,B,A,B,...)
    if len(log.all_steps) >= alternating_min_repeats * 2:
        tail = log.all_steps[-alternating_min_repeats * 2:]
        actions = [o.action for o in tail]
        a, b = actions[0], actions[1]
        if a != b:
            ok = all(actions[i] == (a if i % 2 == 0 else b)
                     for i in range(len(actions)))
            if ok:
                return True, (f"alternating between {a} and {b} for "
                              f"{alternating_min_repeats} cycles — also stuck")

    # Recent-window no-op rate per action
    for action, outcomes in log.by_action.items():
        recent = outcomes[-recent_window:]
        if len(recent) < 3:
            continue
        no_ops = sum(1 for o in recent if not o.frame_changed)
        rate = no_ops / len(recent)
        if rate > recent_no_op_threshold:
            return True, (f"{action} mostly no-op in last {len(recent)} tries "
                          f"({no_ops}/{len(recent)} no-op) — try a different action")

    return False, ""


def recent_no_op_summary(log: OutcomeLog, action: str,
                         window: int = 5) -> dict:
    """For an action, return recent-window stats. Used by prompt builders."""
    outcomes = log.by_action.get(action, [])
    recent = outcomes[-window:]
    if not recent:
        return {"n_recent": 0, "n_no_op": 0, "no_op_rate": 0.0}
    n_no_op = sum(1 for o in recent if not o.frame_changed)
    return {
        "n_recent": len(recent),
        "n_no_op": n_no_op,
        "no_op_rate": n_no_op / len(recent),
    }
