"""Sanity tests for arc_agent.goal_evaluator (2026-05-18 v0)."""
from __future__ import annotations

import pytest

from arc_agent.goal_evaluator import (
    GoalPredicate,
    evaluate_goal,
    evaluate_predicate,
    parse_goal_hypothesis,
)
from arc_agent.object_extractor import ObjectRecord


def _obj(id_: int, color_name: str, row: int, col: int, size: int = 1) -> ObjectRecord:
    return ObjectRecord(
        id=id_, color=0, color_name=color_name,
        cells=[(row, col)],
        bbox=(row, col, row, col),
        center=(float(row), float(col)),
        size=size,
    )


# ── parse tests ────────────────────────────────────────────────────────

def test_parse_align_col_with_left_column():
    pred = parse_goal_hypothesis(
        "align the two yellow squares vertically in the left column"
    )
    assert pred is not None
    assert pred.kind == "align_col"
    assert pred.target == 0
    assert pred.colors == ("yellow",)


def test_parse_align_col_with_col_eq_n():
    pred = parse_goal_hypothesis(
        "align yellow and red objects vertically in column 5"
    )
    assert pred is not None
    assert pred.kind == "align_col"
    assert pred.target == 5
    assert set(pred.colors) >= {"yellow", "red"}


def test_parse_align_row_horizontal():
    pred = parse_goal_hypothesis(
        "align all yellow squares horizontally in the top row"
    )
    assert pred is not None
    assert pred.kind == "align_row"
    assert pred.target == 0


def test_parse_stack_on_top_of():
    pred = parse_goal_hypothesis("stack the blue box on top of the green block")
    assert pred is not None
    assert pred.kind == "stack"
    assert set(pred.colors) >= {"blue", "green"}


def test_parse_move_to_col():
    pred = parse_goal_hypothesis("move the red object to column 7")
    assert pred is not None
    assert pred.kind == "move_to_col"
    assert pred.target == 7


def test_parse_unknown_returns_none():
    assert parse_goal_hypothesis("") is None
    assert parse_goal_hypothesis("explore the level") is None
    assert parse_goal_hypothesis("solve the puzzle by thinking") is None


# ── evaluate tests ─────────────────────────────────────────────────────

def test_evaluate_align_col_true_at_target():
    pred = GoalPredicate(kind="align_col", colors=("yellow",), target=0, min_count=2)
    objects = [
        _obj(1, "yellow", 5, 0),
        _obj(2, "yellow", 30, 0),
    ]
    assert evaluate_predicate(pred, objects) is True


def test_evaluate_align_col_false_not_at_target():
    pred = GoalPredicate(kind="align_col", colors=("yellow",), target=0, min_count=2)
    objects = [
        _obj(1, "yellow", 5, 0),
        _obj(2, "yellow", 30, 7),  # col 7, not 0
    ]
    assert evaluate_predicate(pred, objects) is False


def test_evaluate_align_col_none_not_enough_objects():
    pred = GoalPredicate(kind="align_col", colors=("yellow",), target=0, min_count=2)
    objects = [_obj(1, "yellow", 5, 0)]  # only 1 yellow
    assert evaluate_predicate(pred, objects) is None


def test_evaluate_align_col_no_target_same_col():
    pred = GoalPredicate(kind="align_col", colors=("yellow",), target=None, min_count=2)
    objects = [
        _obj(1, "yellow", 5, 12),
        _obj(2, "yellow", 30, 12),
    ]
    assert evaluate_predicate(pred, objects) is True


def test_evaluate_stack_true():
    pred = GoalPredicate(kind="stack", colors=("blue", "green"), min_count=2)
    objects = [
        _obj(1, "blue", 10, 5),
        _obj(2, "green", 11, 5),  # blue is at row 10, green at 11 -> blue on top
    ]
    assert evaluate_predicate(pred, objects) is True


def test_evaluate_stack_false():
    pred = GoalPredicate(kind="stack", colors=("blue", "green"), min_count=2)
    objects = [
        _obj(1, "blue", 10, 5),
        _obj(2, "green", 10, 6),  # adjacent but not stacked
    ]
    assert evaluate_predicate(pred, objects) is False


def test_evaluate_adjacent_true():
    pred = GoalPredicate(kind="adjacent", colors=("blue", "green"), min_count=2)
    objects = [
        _obj(1, "blue", 10, 5),
        _obj(2, "green", 10, 6),
    ]
    assert evaluate_predicate(pred, objects) is True


# ── end-to-end ─────────────────────────────────────────────────────────

def test_e2e_ar25_like_success():
    """Mirrors the T-GOAL probe success case."""
    objects = [
        _obj(1, "yellow", 12, 0),
        _obj(2, "yellow", 32, 0),
    ]
    achieved, pred = evaluate_goal(
        "align the two yellow squares vertically in the left column",
        objects,
    )
    assert achieved is True
    assert pred is not None
    assert pred.kind == "align_col"


def test_e2e_ar25_like_failure():
    """Mirrors the T-GOAL probe failure case."""
    objects = [
        _obj(1, "yellow", 12, 0),
        _obj(2, "yellow", 32, 5),  # off by 5 cols
    ]
    achieved, _ = evaluate_goal(
        "align the two yellow squares vertically in the left column",
        objects,
    )
    assert achieved is False


def test_e2e_unparseable_hypothesis_returns_none():
    achieved, pred = evaluate_goal("just play around", [_obj(1, "red", 0, 0)])
    assert achieved is None
    assert pred is None


def test_e2e_match_count_too_low_returns_none():
    """If only 1 yellow object present, can't evaluate 'align two'."""
    objects = [_obj(1, "yellow", 5, 0)]
    achieved, _ = evaluate_goal(
        "align two yellow squares vertically in left column", objects)
    assert achieved is None
