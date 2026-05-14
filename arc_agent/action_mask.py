"""Knowledge-driven action mask (R2 hard rule).

Per `docs/ref_v3_2_dataflow_zh.md` (the 2026-05-14 hardrules thread): the
Action Agent sometimes ignores `Knowledge.failed_strategies` / `rules` and
keeps spamming actions Reflection has already marked ineffective. Prompt
language is advisory; the orchestrator enforces it here.

Two functions:

  `compute_action_mask(outcome_log, knowledge, legal_actions)` -> set[str]
      The set of action NAMES to BLOCK. An action is blocked when EITHER:
        (a) it has been tried `no_op_threshold` times this round with
            ZERO frame changes, OR
        (b) Knowledge.rules / Knowledge.failed_strategies explicitly
            mention it as ineffective (regex pattern match).
      Safety: never returns a mask that would block ALL legal actions --
      if it would, returns the empty set.

  `apply_action_mask(chosen, mask, legal_actions, outcome_log,
                     knowledge, rng)` -> (final_action, was_replaced)
      Given the LLM's chosen GameAction and the mask, returns either the
      original (if unblocked) or a preferred replacement. Replacement
      preference, in order:
        1. an untried legal action (excluding masked)
        2. an action that has a positive `action_semantics` entry
        3. a random non-blocked legal action

This module imports `arcengine.GameAction` lazily so unit tests on a
GPU-less machine still work.
"""
from __future__ import annotations

import logging
import random
import re
from typing import Any, Optional

from arc_agent.action_inference import ALL_ACTIONS, OutcomeLog
from arc_agent.knowledge import Knowledge

logger = logging.getLogger(__name__)

DEFAULT_NO_OP_THRESHOLD = 5  # number of all-no-op tries before hard-block

# Words / phrases in Knowledge.rules or failed_strategies that mean
# "this action category has been shown to fail".
_NEGATION_PATTERNS = (
    r"has no effect",
    r"no observable effect",
    r"is ineffective",
    r"never changes",
    r"didn'?t work",
    r"do(?:es)? not work",
    r"failed",
    r"anywhere",   # "ACTION6 anywhere in the right half" -> high-level region fail
)


def _knowledge_text_blob(knowledge: Knowledge) -> str:
    """Lower-cased concatenation of rules + failed_strategies. Cached
    per-call (Knowledge is mutable but cheap to re-scan)."""
    return " | ".join(knowledge.rules + knowledge.failed_strategies).lower()


def _action_flagged_in_knowledge(action: str, blob: str) -> bool:
    """True if `action` (e.g. 'ACTION6') co-occurs with a negation pattern
    in the knowledge text blob."""
    name = action.lower()
    if name not in blob:
        return False
    for pat in _NEGATION_PATTERNS:
        # Require the negation pattern within ~80 chars of the action mention
        for m in re.finditer(re.escape(name), blob):
            window = blob[max(0, m.start() - 80): m.end() + 80]
            if re.search(pat, window):
                return True
    return False


def compute_action_mask(
    outcome_log: OutcomeLog,
    knowledge: Knowledge,
    legal_actions: list[str],
    *,
    no_op_threshold: int = DEFAULT_NO_OP_THRESHOLD,
) -> set[str]:
    """Return the set of action names the orchestrator should block.

    Args:
        outcome_log: this round's observed (action -> outcome) stats.
        knowledge: cross-round Knowledge with rules / failed_strategies.
        legal_actions: the action names legal in the current frame.
        no_op_threshold: how many all-no-op tries before hard-blocking on
            empirical evidence alone (default 5).

    Returns:
        A subset of `legal_actions` (by name) to block. Never returns a
        set that would mask all of `legal_actions` -- if every legal
        action is blockable, returns the empty set (let the LLM choose).
    """
    blocked: set[str] = set()
    blob = _knowledge_text_blob(knowledge)

    for action in legal_actions:
        # Rule (a): empirical no-op streak
        n_tried = outcome_log.n_tried(action)
        n_changed = outcome_log.n_changed(action)
        if n_tried >= no_op_threshold and n_changed == 0:
            blocked.add(action)
            continue
        # Rule (b): explicit Knowledge flag
        if _action_flagged_in_knowledge(action, blob):
            blocked.add(action)

    # Safety: never mask all legal actions. If masking every legal action
    # would leave nothing, drop the mask entirely (orchestrator falls back
    # to letting the LLM choose).
    remaining = set(legal_actions) - blocked
    if not remaining:
        logger.debug("compute_action_mask: would block all legal -- dropping mask")
        return set()

    return blocked


def _positive_action_from_knowledge(
    knowledge: Knowledge,
    legal_actions: list[str],
    mask: set[str],
) -> Optional[str]:
    """Pick an action whose action_semantics is positive (mentions a
    direction or actual change), is legal, and is not masked."""
    _negative_words = ("no effect", "no observable", "ineffective",
                       "never changes", "didn't work", "no-op", "no op")
    for action, sem in knowledge.action_semantics.items():
        if action not in legal_actions or action in mask:
            continue
        sem_lower = sem.lower()
        if any(w in sem_lower for w in _negative_words):
            continue
        return action
    return None


def apply_action_mask(
    chosen_action_name: str,
    mask: set[str],
    legal_actions: list[str],
    outcome_log: OutcomeLog,
    knowledge: Knowledge,
    rng: random.Random,
) -> tuple[str, bool, str]:
    """If `chosen_action_name` is in `mask`, return a preferred replacement.

    Args:
        chosen_action_name: the action name the LLM picked (e.g. "ACTION6").
        mask: set returned by `compute_action_mask`.
        legal_actions: action names legal in the current frame.
        outcome_log: this round's stats.
        knowledge: cross-round Knowledge.
        rng: orchestrator's random (seeded).

    Returns:
        (final_action_name, was_replaced, reason)
        - `was_replaced=False, reason=""` when the chosen action is not
          masked OR the mask is empty.
        - `was_replaced=True, reason="..."` when replacement happened.
    """
    if not mask or chosen_action_name not in mask:
        return chosen_action_name, False, ""

    # Preference 1: untried legal action (and not masked)
    untried = [a for a in outcome_log.untried(legal_actions) if a not in mask]
    if untried:
        return untried[0], True, f"untried {untried[0]} over masked {chosen_action_name}"

    # Preference 2: action with positive action_semantics
    positive = _positive_action_from_knowledge(knowledge, legal_actions, mask)
    if positive is not None:
        return positive, True, (
            f"known-good {positive} over masked {chosen_action_name}"
        )

    # Preference 3: random non-masked legal action
    non_blocked = [a for a in legal_actions if a not in mask]
    if non_blocked:
        pick = rng.choice(non_blocked)
        return pick, True, f"random non-blocked {pick} over masked {chosen_action_name}"

    # Should not happen because compute_action_mask drops full masks.
    return chosen_action_name, False, ""


__all__ = [
    "DEFAULT_NO_OP_THRESHOLD",
    "compute_action_mask",
    "apply_action_mask",
]
