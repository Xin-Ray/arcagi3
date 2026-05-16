"""Code-side action proposer (v0).

Per docs/project/2026-05-16-v0-action_proposer/architecture.md.

Given the current state + Knowledge + OutcomeLog + click_targets, propose
K=3 candidate actions for the LLM to pick from. Each candidate has a
short text reason that goes into the prompt.

Strategy (priority order):
  1. Always include 1 untried legal action if any exists this round.
  2. Always include 1 known-good action (positive action_semantics in Knowledge).
  3. Fill remaining slot with:
     - high-confidence click_target if ACTION6 legal and any target alive
     - else: random non-recent legal action

If known-good and untried collide (action both untried AND in semantics —
which shouldn't happen), de-dup. If we can't fill 3 slots, return what we
have (caller falls back to mask logic or free-text).

The output `letter` field assigns A/B/C labels in shuffled order so the
LLM can't always pick "A" by position bias.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from arc_agent.action_inference import ALL_ACTIONS, OutcomeLog
from arc_agent.knowledge import Knowledge, _has_positive_semantic


@dataclass
class Candidate:
    """One proposed action with text reason for the LLM prompt."""
    letter: str               # "A" / "B" / "C"
    action_name: str          # "ACTION1" ... "ACTION7"
    coords: Optional[tuple[int, int]]  # (x, y) for ACTION6, else None
    reason: str               # short, human-readable; goes into prompt


_NEGATIVE_PHRASES = (
    "no effect", "no observable", "ineffective", "never changes",
    "didn't work", "no-op", "no op", "avoid",
)


def _is_negative_semantic(s: str) -> bool:
    s = (s or "").lower()
    return any(w in s for w in _NEGATIVE_PHRASES)


def _untried_actions(
    outcome_log: OutcomeLog, legal_actions: list[str],
) -> list[str]:
    return [a for a in outcome_log.untried(legal_actions)]


def _known_good_actions(
    knowledge: Knowledge, legal_actions: list[str],
) -> list[str]:
    """Actions whose action_semantics is non-empty and not negative."""
    out = []
    for a in legal_actions:
        sem = knowledge.action_semantics.get(a, "")
        if sem and not _is_negative_semantic(sem):
            out.append(a)
    return out


def _ct_get(target, attr, default=None):
    """Read attribute from a click_target — dataclass or dict (compat)."""
    if hasattr(target, attr):
        return getattr(target, attr)
    if isinstance(target, dict):
        return target.get(attr, default)
    return default


def _best_click_target(knowledge: Knowledge):
    """Pick highest-confidence alive click_target with low tries.

    Priority: untried (tries=0) high-conf first, else high-conf.
    """
    if not getattr(knowledge, "click_targets", None):
        return None
    alive = [t for t in knowledge.click_targets if _ct_get(t, "alive", True)]
    if not alive:
        return None
    untried = [t for t in alive if _ct_get(t, "tries", 0) == 0]
    pool = untried if untried else alive
    return max(pool, key=lambda t: _ct_get(t, "confidence", 0.0))


def propose(
    knowledge: Knowledge,
    outcome_log: OutcomeLog,
    legal_actions: list[str],
    *,
    recent_action_names: Optional[list[str]] = None,
    rng: Optional[random.Random] = None,
    k: int = 3,
) -> list[Candidate]:
    """Build K candidates with letter labels A..C.

    `recent_action_names` (most recent N picks) is used to penalise
    repeating; defaults to empty.
    """
    rng = rng or random.Random(42)
    recent = set(recent_action_names or [])

    untried = _untried_actions(outcome_log, legal_actions)
    known_good = _known_good_actions(knowledge, legal_actions)
    # Exclude over-committed from known_good if user just picked it 3+
    # times in a row (defensive against ACTION1 lock-in)
    if recent_action_names:
        from collections import Counter
        counts = Counter(recent_action_names[-5:])
        for over_action, c in counts.items():
            if c >= 3:
                known_good = [a for a in known_good if a != over_action]

    candidates: list[tuple[str, Optional[tuple[int, int]], str]] = []  # (action, coords, reason)

    # Slot 1: untried (highest priority)
    if untried:
        a = untried[0]
        coords = None
        # ACTION6 needs coords; use click_target if any, else random
        if a == "ACTION6":
            ct = _best_click_target(knowledge)
            if ct is not None:
                coords = tuple(_ct_get(ct, "coords"))
                reason = f"untried this round; click target {_ct_get(ct, 'signature', 'obj')}"
            else:
                coords = (rng.randint(0, 63), rng.randint(0, 63))
                reason = f"untried this round (random click coords)"
        else:
            reason = "untried this round"
        candidates.append((a, coords, reason))

    # Slot 2: known-good (different from slot 1)
    used = {c[0] for c in candidates}
    for a in known_good:
        if a in used:
            continue
        sem = knowledge.action_semantics.get(a, "")
        short_sem = sem[:60] + "..." if len(sem) > 60 else sem
        coords = None
        if a == "ACTION6":
            ct = _best_click_target(knowledge)
            coords = tuple(_ct_get(ct, "coords")) if ct else (rng.randint(0, 63), rng.randint(0, 63))
        candidates.append((a, coords, f"known-good: {short_sem}"))
        break

    # Slot 3+: keep filling until we have K
    # First try click_target ACTION6
    used = {c[0] for c in candidates}
    if len(candidates) < k and "ACTION6" in legal_actions and "ACTION6" not in used:
        ct = _best_click_target(knowledge)
        if ct is not None:
            candidates.append((
                "ACTION6", tuple(_ct_get(ct, "coords")),
                f"click target {_ct_get(ct, 'signature', 'obj')} (conf={_ct_get(ct, 'confidence', 0.0):.2f})",
            ))

    # Then more untried (prefer)
    while len(candidates) < k:
        used = {c[0] for c in candidates}
        more_untried = [a for a in untried if a not in used]
        if more_untried:
            a = more_untried[0]
            coords = (rng.randint(0, 63), rng.randint(0, 63)) if a == "ACTION6" else None
            candidates.append((a, coords, "another untried option"))
            continue
        # Then random non-recent legal
        pool = [a for a in legal_actions if a not in used and a not in recent]
        if not pool:
            pool = [a for a in legal_actions if a not in used]
        if not pool:
            break  # truly out of options
        a = rng.choice(pool)
        coords = (rng.randint(0, 63), rng.randint(0, 63)) if a == "ACTION6" else None
        candidates.append((a, coords, "diverse pick"))

    # Truncate to k
    candidates = candidates[:k]

    # Shuffle letter assignment (so position doesn't bias)
    indices = list(range(len(candidates)))
    rng.shuffle(indices)
    letters = "ABCDEFG"[:len(candidates)]
    out = [
        Candidate(
            letter=letters[pos], action_name=candidates[orig_idx][0],
            coords=candidates[orig_idx][1], reason=candidates[orig_idx][2],
        )
        for pos, orig_idx in enumerate(indices)
    ]
    # Sort by letter so [A][B][C] in order
    out.sort(key=lambda c: c.letter)
    return out


def candidates_to_prompt_block(candidates: list[Candidate]) -> str:
    """Render candidates as a single prompt block."""
    if not candidates:
        return ""
    lines = ["[CANDIDATES — pick exactly one letter]"]
    for c in candidates:
        coord_str = f" ({c.coords[0]},{c.coords[1]})" if c.coords is not None else ""
        lines.append(f"  [{c.letter}] {c.action_name}{coord_str}   reason: {c.reason}")
    return "\n".join(lines)


def resolve_letter(
    letter: str, candidates: list[Candidate],
) -> Optional[Candidate]:
    """Map LLM's "A"/"B"/"C" reply back to the Candidate."""
    letter = letter.strip().upper()[:1] if letter else ""
    for c in candidates:
        if c.letter == letter:
            return c
    return None


__all__ = ["Candidate", "propose", "candidates_to_prompt_block", "resolve_letter"]
