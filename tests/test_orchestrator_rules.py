"""Tests for arc_agent.orchestrator_rules."""
from __future__ import annotations

from arc_agent.action_inference import OutcomeLog, StepOutcome
from arc_agent.orchestrator_rules import (
    auto_failed_strategies_from_outcome_log,
    auto_rules_from_outcome_log,
    infer_goal_confidence_from_log,
)


def _log_with(action: str, n_total: int, n_no_op: int) -> OutcomeLog:
    log = OutcomeLog()
    for i in range(n_total):
        changed = i >= n_no_op   # first n_no_op are no-ops, rest changed
        log.record(StepOutcome(
            step=i, action=action, legal=True,
            frame_changed=changed, n_active_changed=int(changed),
            primary_direction="UP" if changed else None,
            primary_distance=3 if changed else 0,
        ))
    return log


# ── auto_rules_from_outcome_log ───────────────────────────────────────


def test_rule_fires_when_action_all_noop() -> None:
    log = _log_with("ACTION6", n_total=8, n_no_op=8)
    rules = auto_rules_from_outcome_log(log, ["ACTION1", "ACTION6"])
    assert any("ACTION6" in r and "no-op" in r for r in rules)
    assert all("ACTION1" not in r for r in rules)  # ACTION1 untried


def test_rule_silent_below_min_tries() -> None:
    """Default min_tries=5; 4 tries shouldn't be enough."""
    log = _log_with("ACTION6", n_total=4, n_no_op=4)
    rules = auto_rules_from_outcome_log(log, ["ACTION6"])
    assert rules == []


def test_rule_silent_when_action_works() -> None:
    """100% changed -> no failure rule."""
    log = _log_with("ACTION1", n_total=10, n_no_op=0)
    rules = auto_rules_from_outcome_log(log, ["ACTION1"])
    assert rules == []


def test_rule_fires_with_90pct_threshold() -> None:
    """9/10 no-op meets default 90% threshold."""
    log = _log_with("ACTION6", n_total=10, n_no_op=9)
    rules = auto_rules_from_outcome_log(log, ["ACTION6"])
    assert any("ACTION6" in r for r in rules)


def test_rule_silent_below_90pct() -> None:
    """8/10 no-op (80%) misses the default."""
    log = _log_with("ACTION6", n_total=10, n_no_op=8)
    rules = auto_rules_from_outcome_log(log, ["ACTION6"])
    assert rules == []


def test_rule_handles_none_log() -> None:
    assert auto_rules_from_outcome_log(None, ["ACTION1"]) == []


# ── auto_failed_strategies_from_outcome_log ───────────────────────────


def test_failed_strategy_fires_when_action_confirmed_dead() -> None:
    log = _log_with("ACTION6", n_total=8, n_no_op=8)
    out = auto_failed_strategies_from_outcome_log(log, ["ACTION6"])
    assert any("ACTION6" in s and "ineffective" in s for s in out)


def test_failed_strategy_handles_none_log() -> None:
    assert auto_failed_strategies_from_outcome_log(None, ["ACTION6"]) == []


# ── infer_goal_confidence_from_log ────────────────────────────────────


def test_confidence_high_when_win_seen() -> None:
    log = OutcomeLog()
    assert infer_goal_confidence_from_log(log, win_seen=True) == "high"


def test_confidence_medium_when_changing() -> None:
    log = _log_with("ACTION1", n_total=20, n_no_op=10)   # 50% changed
    assert infer_goal_confidence_from_log(log) == "medium"


def test_confidence_low_when_few_changes() -> None:
    log = _log_with("ACTION1", n_total=20, n_no_op=19)   # 5% changed
    assert infer_goal_confidence_from_log(log) == "low"


def test_confidence_none_when_too_few_observations() -> None:
    """Below n_observations_floor: leave existing confidence alone."""
    log = _log_with("ACTION1", n_total=3, n_no_op=0)
    assert infer_goal_confidence_from_log(log) is None


def test_confidence_handles_none_log() -> None:
    assert infer_goal_confidence_from_log(None) is None
