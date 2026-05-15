"""Deterministic rule + failed_strategy generation from OutcomeLog.

Why this module exists. Reflection's `rules_append` and
`failed_strategies_append` were the main source of token waste in the v3.2
smoke runs: at every step the LLM re-emitted the same line
("ACTION6 has no effect on any tested coord.") that the orchestrator
already knows deterministically from `OutcomeLog`. The dedup in
`Knowledge.merged_with_delta` swallowed the duplicates but the LLM still
spent tokens generating them.

This module moves those two list fields back to the orchestrator. Pure
functions over `OutcomeLog`; no LLM. The orchestrator calls these each
step and merges the output into `knowledge.rules` / `failed_strategies`
via the same R4 contradiction filter the Reflection deltas used.

Reflection no longer needs to write these fields (handled in the schema
simplification of P2). `merged_with_delta` keeps tolerating legacy
`rules_append` / `failed_strategies_append` keys silently so old traces
load.

Scope: deterministic patterns ONLY. Anything that needs creative
inference (goal_hypothesis, conditional action_semantics) stays with
Reflection.
"""
from __future__ import annotations

from typing import Any, Optional


# Thresholds for what counts as a "rule worth writing".
RULE_MIN_TRIES = 5            # need this many tries before declaring no-effect
RULE_NO_OP_RATE = 0.9          # fraction of tries that must be no-op
FAILED_STRATEGY_MIN_TRIES = 5  # same but for failed_strategies (higher bar)


def auto_rules_from_outcome_log(
    outcome_log: Any,
    legal_actions: list[str],
    *,
    min_tries: int = RULE_MIN_TRIES,
    no_op_rate: float = RULE_NO_OP_RATE,
) -> list[str]:
    """Return rules deterministically derivable from OutcomeLog.

    Current rule patterns:
      - "ACTION_X: tried Nx, 0% changed -- ineffective on tested coords"
        (when >= min_tries and >= no_op_rate fraction are no-op)

    Other patterns are kept as Reflection's responsibility (anything
    requiring creative pattern inference, e.g. "ACTION1 followed by ACTION2
    always reaches the top edge").
    """
    if outcome_log is None:
        return []
    rules: list[str] = []
    for action in legal_actions:
        outcomes = outcome_log.by_action.get(action, [])
        n_tries = len(outcomes)
        if n_tries < min_tries:
            continue
        n_no_op = sum(1 for o in outcomes if not o.frame_changed)
        rate = n_no_op / n_tries
        if rate >= no_op_rate:
            rules.append(
                f"{action}: tried {n_tries}x, {int(rate * 100)}% no-op "
                f"-- ineffective on tested coords"
            )
    return rules


def auto_failed_strategies_from_outcome_log(
    outcome_log: Any,
    legal_actions: list[str],
    *,
    min_tries: int = FAILED_STRATEGY_MIN_TRIES,
    no_op_rate: float = RULE_NO_OP_RATE,
) -> list[str]:
    """Return failed_strategies deterministically derivable from OutcomeLog.

    Currently the same surface as rules but phrased as a strategy. We keep
    the two functions separate so we can specialize later (e.g. add
    region-level patterns for ACTION6 by analysing tried (x,y) clusters).
    """
    if outcome_log is None:
        return []
    out: list[str] = []
    for action in legal_actions:
        outcomes = outcome_log.by_action.get(action, [])
        n_tries = len(outcomes)
        if n_tries < min_tries:
            continue
        n_no_op = sum(1 for o in outcomes if not o.frame_changed)
        rate = n_no_op / n_tries
        if rate >= no_op_rate:
            out.append(f"{action}: confirmed ineffective after {n_tries} tries")
    return out


def infer_goal_confidence_from_log(
    outcome_log: Any,
    *,
    win_seen: bool = False,
    n_observations_floor: int = 5,
) -> Optional[str]:
    """Return a confidence label inferred from the OutcomeLog signal density.

    Returns None when there isn't enough data to say -- caller should leave
    the existing confidence untouched. Reflection can still override.

    Heuristic:
      - WIN seen this episode                  -> "high"
      - >= 25% of recent steps frame_changed   -> "medium"
      - >= n_observations_floor steps recorded -> "low"
      - otherwise                               -> None (no opinion)
    """
    if outcome_log is None:
        return None
    if win_seen:
        return "high"
    all_steps = getattr(outcome_log, "all_steps", []) or []
    if len(all_steps) < n_observations_floor:
        return None
    recent = all_steps[-20:]
    if not recent:
        return None
    rate = sum(1 for o in recent if o.frame_changed) / len(recent)
    if rate >= 0.25:
        return "medium"
    return "low"


__all__ = [
    "FAILED_STRATEGY_MIN_TRIES",
    "RULE_MIN_TRIES",
    "RULE_NO_OP_RATE",
    "auto_failed_strategies_from_outcome_log",
    "auto_rules_from_outcome_log",
    "infer_goal_confidence_from_log",
]
