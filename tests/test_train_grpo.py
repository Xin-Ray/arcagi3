"""Tests for `arc_agent.train_grpo.reward_fn` — one branch per test.

The formula in `docs/ARCHITECTURE_RL.md` §3 has four independent terms; each
is covered alone so a regression points at exactly which coefficient drifted.
Combined cases verify the terms truly are additive.
"""
from __future__ import annotations

import pytest

from arc_agent.train_grpo import (
    R_ENTITY_BONUS,
    R_F1_COEFF,
    R_ILLEGAL_ACTION,
    R_PARSE_FAIL,
    R_WIN,
    StepRecord,
    _action_name_to_id,
    reward_fn,
)


# ── single-term cases ────────────────────────────────────────────────────


def test_reward_win_alone() -> None:
    s = StepRecord(state="WIN", parsed_json_ok=True, f1=0.0,
                   action="ACTION1", available_actions=[1])
    # WIN(+1) + F1(0) = +1.0
    assert reward_fn(s) == pytest.approx(R_WIN)


def test_reward_parse_fail_alone() -> None:
    s = StepRecord(state="NOT_FINISHED", parsed_json_ok=False,
                   action=None, available_actions=[1, 2])
    # parse_fail(-0.5); no action so no illegal penalty
    assert reward_fn(s) == pytest.approx(R_PARSE_FAIL)


def test_reward_perfect_f1_alone() -> None:
    s = StepRecord(state="NOT_FINISHED", parsed_json_ok=True, f1=1.0,
                   action="ACTION1", available_actions=[1])
    # F1 * 0.2 only
    assert reward_fn(s) == pytest.approx(R_F1_COEFF)


def test_reward_partial_f1() -> None:
    s = StepRecord(state="NOT_FINISHED", parsed_json_ok=True, f1=0.5,
                   action="ACTION1", available_actions=[1])
    assert reward_fn(s) == pytest.approx(R_F1_COEFF * 0.5)


def test_reward_illegal_action_alone() -> None:
    s = StepRecord(state="NOT_FINISHED", parsed_json_ok=True, f1=0.0,
                   action="ACTION5", available_actions=[1, 2, 3])
    # F1(0) + illegal(-0.3)
    assert reward_fn(s) == pytest.approx(R_ILLEGAL_ACTION)


def test_reward_entity_bonus_alone() -> None:
    s = StepRecord(state="NOT_FINISHED", parsed_json_ok=True, f1=0.0,
                   action="ACTION1", available_actions=[1],
                   entity_recognition_consistent=True)
    assert reward_fn(s) == pytest.approx(R_ENTITY_BONUS)


# ── combined: terms must be additive ─────────────────────────────────────


def test_reward_max_case() -> None:
    s = StepRecord(state="WIN", parsed_json_ok=True, f1=1.0,
                   action="ACTION1", available_actions=[1],
                   entity_recognition_consistent=True)
    # WIN + max-F1 + entity = 1 + 0.2 + 0.05
    assert reward_fn(s) == pytest.approx(R_WIN + R_F1_COEFF + R_ENTITY_BONUS)


def test_reward_min_case() -> None:
    s = StepRecord(state="NOT_FINISHED", parsed_json_ok=False,
                   action="ACTION5", available_actions=[1, 2, 3])
    # parse_fail(-0.5) + illegal(-0.3) = -0.8
    assert reward_fn(s) == pytest.approx(R_PARSE_FAIL + R_ILLEGAL_ACTION)


def test_reward_parse_fail_overrides_f1_contribution() -> None:
    """parse_fail=True means the f1 term is replaced by the penalty, not added on."""
    s = StepRecord(state="NOT_FINISHED", parsed_json_ok=False, f1=1.0,
                   action="ACTION1", available_actions=[1])
    # F1 field should be IGNORED when parse_ok is False
    assert reward_fn(s) == pytest.approx(R_PARSE_FAIL)


def test_reward_legal_action_no_penalty() -> None:
    s = StepRecord(state="NOT_FINISHED", parsed_json_ok=True, f1=0.0,
                   action="ACTION3", available_actions=[1, 2, 3, 4])
    assert reward_fn(s) == pytest.approx(0.0)


def test_reward_no_action_no_penalty() -> None:
    s = StepRecord(state="NOT_FINISHED", parsed_json_ok=True, f1=0.0,
                   action=None, available_actions=[1])
    assert reward_fn(s) == pytest.approx(0.0)


# ── action name parsing ──────────────────────────────────────────────────


@pytest.mark.parametrize("name,expected", [
    ("ACTION1", 1), ("ACTION7", 7), ("ACTION3", 3),
    ("ACTION0", None), ("ACTION8", None),
    ("RESET", None), ("action1", None),  # case-sensitive
    ("ACTIONx", None), ("", None), (None, None),
])
def test_action_name_to_id(name, expected) -> None:
    assert _action_name_to_id(name) == expected


# ── coefficient drift sentinel ───────────────────────────────────────────


def test_coefficients_match_spec() -> None:
    """If anyone edits these, the spec in ARCHITECTURE_RL.md §3 must update too."""
    assert R_WIN == 1.0
    assert R_F1_COEFF == 0.2
    assert R_PARSE_FAIL == -0.5
    assert R_ILLEGAL_ACTION == -0.3
    assert R_ENTITY_BONUS == 0.05
