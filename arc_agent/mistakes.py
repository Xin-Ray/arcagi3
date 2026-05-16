"""Deterministic mistake detectors for A4 (`docs/ARCHITECTURE_AGENTS.md` §1).

Rule of thumb from §1 A4: do NOT ask a 3B model to self-judge. Each detector
is a pure function over recent step records and emits an auto-message — the
Play agent's reflection sees these strings verbatim, the reflection agent
only dedups when the list grows past `MAX_MISTAKES`.

Step record schema (a single dict, immutable per step):

    {
        "step":      int,
        "action":    str,                 # "ACTION3" / "ACTION6:12,30"
        "legal":     bool,                # was action ∈ available_actions
        "frame_changed":  bool,
        "frame_hash":     int,
        "levels_completed": int,
    }

Detectors return a string when a mistake fires, else None. The orchestrator
(`update_mistakes`) gathers them, dedups against the running list, and caps
the list at `MAX_MISTAKES` (oldest evicted first — A4's "Reflection dedups
only" rule means we never let it grow unboundedly).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

# Knobs from §1 A4 (tunable in the pilot run before the full ablation; §4
# limitation 6 calls them out as "guesses" — keep them adjustable here).
NO_OP_STREAK = 3
LOOP_WINDOW = 20
LOOP_COUNT = 5
MAX_MISTAKES = 5


@dataclass
class StepRecord:
    """Immutable snapshot of one (action, outcome) pair. See module docstring."""

    step: int
    action: str
    legal: bool
    frame_changed: bool
    frame_hash: int
    levels_completed: int


@dataclass
class MistakeBuffer:
    """Per-episode running buffer the orchestrator feeds + reads.

    `records` is the truth source (append-only). `mistakes` is the cap-5
    string list exposed to the Play prompt.
    """

    records: list[StepRecord] = field(default_factory=list)
    mistakes: list[str] = field(default_factory=list)

    def append_record(self, rec: StepRecord) -> None:
        self.records.append(rec)

    def add_mistake(self, msg: str) -> None:
        if msg in self.mistakes:
            # Already known — move to most-recent slot so eviction stays FIFO
            # on truly-stale mistakes.
            self.mistakes.remove(msg)
        self.mistakes.append(msg)
        if len(self.mistakes) > MAX_MISTAKES:
            # Drop oldest (FIFO). §1 A4: reflection MAY dedup further; that's
            # additive, not a substitute.
            self.mistakes = self.mistakes[-MAX_MISTAKES:]

    def reset(self) -> None:
        self.records.clear()
        self.mistakes.clear()


# ── detectors ──────────────────────────────────────────────────────────────


def detect_illegal_action(rec: StepRecord) -> Optional[str]:
    """Action was not in available_actions when chosen."""
    if rec.legal:
        return None
    return f"{rec.action} not legal in this state"


def detect_no_op_streak(records: list[StepRecord]) -> Optional[str]:
    """Frame unchanged for `NO_OP_STREAK` consecutive steps."""
    if len(records) < NO_OP_STREAK:
        return None
    tail = records[-NO_OP_STREAK:]
    if all(not r.frame_changed for r in tail):
        names = [r.action for r in tail]
        return f"{', '.join(names)} did not change the grid"
    return None


def detect_regression(records: list[StepRecord]) -> Optional[str]:
    """`levels_completed` dropped between the last two records."""
    if len(records) < 2:
        return None
    prev, curr = records[-2], records[-1]
    if curr.levels_completed < prev.levels_completed:
        return f"{curr.action} undid a level"
    return None


def detect_loop(records: list[StepRecord]) -> Optional[str]:
    """Same (frame_hash, action) pair seen ≥ LOOP_COUNT times in last LOOP_WINDOW steps."""
    if len(records) < LOOP_COUNT:
        return None
    window = records[-LOOP_WINDOW:]
    pairs = Counter((r.frame_hash, r.action) for r in window)
    (hash_act, count), = pairs.most_common(1)
    if count >= LOOP_COUNT:
        _, action = hash_act
        return f"looping on {action} from this state"
    return None


def update_mistakes(buf: MistakeBuffer, rec: StepRecord) -> list[str]:
    """Append `rec`, run all detectors, return the current mistakes list.

    Order matters: legality is per-record, the rest scan the running list.
    Each detector that fires contributes at most one auto-message per call.
    """
    buf.append_record(rec)
    for detector in (
        detect_illegal_action,
        detect_no_op_streak,
        detect_regression,
        detect_loop,
    ):
        if detector is detect_illegal_action:
            msg = detector(rec)
        else:
            msg = detector(buf.records)
        if msg:
            buf.add_mistake(msg)
    return list(buf.mistakes)


# ── helpers used by Play agents to build records cheaply ───────────────────


def frame_hash(grid) -> int:
    """Stable hash of a numpy grid. Used as `StepRecord.frame_hash`.

    Falls back to `hash(bytes)` so we don't depend on numpy's array.tobytes
    quirks across dtypes — caller can pass int8 / int64 / object grids and
    still get a deterministic value.
    """
    try:
        return int(hash(bytes(grid.astype("int8").tobytes())))
    except Exception:
        return int(hash(repr(grid)))
