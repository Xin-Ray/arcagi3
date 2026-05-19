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


# ── v1 (2026-05-18): production vocab extension ────────────────────────

def test_parse_top_edge():
    pred = parse_goal_hypothesis(
        "move the red 1x1 (obj_0) and the yellow 1x1 (obj_1) to the top edge of the board"
    )
    assert pred is not None
    assert pred.kind == "move_to_row"
    assert pred.target == 0


def test_parse_left_edge():
    pred = parse_goal_hypothesis("move red and yellow to the left edge of the board")
    assert pred is not None
    assert pred.kind == "move_to_col"
    assert pred.target == 0


def test_parse_right_edge():
    pred = parse_goal_hypothesis("push the gray block to the right edge")
    assert pred is not None
    assert pred.kind == "move_to_col"
    assert pred.target == 63


def test_parse_to_center():
    pred = parse_goal_hypothesis(
        "move the tan objects #7 and #8 to the center of the board")
    assert pred is not None
    assert pred.kind == "move_to_center"
    assert "tan" in pred.colors


def test_parse_towards_center():
    pred = parse_goal_hypothesis(
        "move the purple objects towards the center of the board to align them")
    assert pred is not None
    assert pred.kind == "move_to_center"


def test_parse_align_no_axis_falls_back_to_align_any():
    """'align the tan objects #7 and #8' without axis -> align_any."""
    pred = parse_goal_hypothesis("align the tan objects #7 and #8")
    assert pred is not None
    assert pred.kind == "align_any"
    assert "tan" in pred.colors


def test_evaluate_top_edge_true():
    pred = GoalPredicate(kind="move_to_row", colors=("red",), target=0, min_count=1)
    objs = [_obj(1, "red", 0, 30)]  # at top edge
    assert evaluate_predicate(pred, objs) is True


def test_evaluate_top_edge_false():
    pred = GoalPredicate(kind="move_to_row", colors=("red",), target=0, min_count=1)
    objs = [_obj(1, "red", 25, 30)]  # not at top
    assert evaluate_predicate(pred, objs) is False


def test_evaluate_center_true():
    pred = GoalPredicate(kind="move_to_center", colors=("tan",), min_count=1)
    objs = [_obj(1, "tan", 31, 31), _obj(2, "tan", 30, 32)]  # both ~center
    assert evaluate_predicate(pred, objs) is True


def test_evaluate_center_false():
    pred = GoalPredicate(kind="move_to_center", colors=("tan",), min_count=1)
    objs = [_obj(1, "tan", 0, 0)]  # corner, far from center
    assert evaluate_predicate(pred, objs) is False


def test_evaluate_align_any_same_col_true():
    """align_any succeeds when objects share column (even no axis specified)."""
    pred = GoalPredicate(kind="align_any", colors=("tan",), min_count=2)
    objs = [_obj(1, "tan", 5, 10), _obj(2, "tan", 25, 10)]
    assert evaluate_predicate(pred, objs) is True


def test_evaluate_align_any_same_row_true():
    pred = GoalPredicate(kind="align_any", colors=("tan",), min_count=2)
    objs = [_obj(1, "tan", 30, 5), _obj(2, "tan", 30, 25)]
    assert evaluate_predicate(pred, objs) is True


def test_evaluate_align_any_false():
    pred = GoalPredicate(kind="align_any", colors=("tan",), min_count=2)
    objs = [_obj(1, "tan", 5, 10), _obj(2, "tan", 25, 30)]  # neither col nor row match
    assert evaluate_predicate(pred, objs) is False


# ── coverage on v2 round 0 production hypotheses ───────────────────────

V2_HYPOTHESES = [
    "move the red 1x1 (obj_0) and the yellow 1x1 (obj_1) to the top edge of the board",
    "move the red 1x1 (obj_0) and the yellow 1x1 (obj_1) to the left edge of the board",
    "align the tan objects #7 and #8",
    "move the tan objects #7 and #8 towards each other to align them",
    "move the tan objects #7 and #8 to the center of the board",
    "move the tan objects #11 and #12 to the center of the board",
    "move the tan objects #10 and #11 to the center of the board",
    "align the purple objects #6, #7, #8, #9 to the center of the board",
    "move the purple objects towards the center of the board to align them",
]


def test_v2_production_hypothesis_coverage():
    """Parser must handle all 9 distinct v2 round 0 hypotheses."""
    unparsed = [h for h in V2_HYPOTHESES if parse_goal_hypothesis(h) is None]
    assert unparsed == [], f"Unparsed: {unparsed}"


# ── v2 (smoke 3 finding 2026-05-18): "reach the X edge" verb ───────────

SMOKE3_HYPOTHESES = [
    "all yellow objects must reach the bottom edge of the grid",
    "reach the left edge",
    "reaching the right edge",
    "yellow must reach the center",
    "must reach the bottom edge",
]


def test_smoke3_reach_the_edge_patterns():
    for h in SMOKE3_HYPOTHESES:
        pred = parse_goal_hypothesis(h)
        assert pred is not None, f"failed to parse {h!r}"
        assert pred.kind in ("move_to_row", "move_to_col",
                             "move_to_center"), \
            f"unexpected kind {pred.kind} for {h!r}"


# ── v3 (Phase 1B 2026-05-19): "match X with Y" pattern ─────────────────

PHASE1B_HYPOTHESES = [
    "match every yellow 1x1 with a yellow target square",
    "match the moving blue square to the static blue target",
    "match every red dot with a red target square",
    "match the moving yellow 1x1 (obj_000) with the static yellow 1x1 on the bottom edge",
]


def test_phase1b_match_pattern():
    for h in PHASE1B_HYPOTHESES:
        pred = parse_goal_hypothesis(h)
        assert pred is not None, f"failed to parse {h!r}"
        assert pred.kind == "align_any", \
            f"expected align_any for {h!r}, got {pred.kind}"
        # Must extract at least one color
        assert len(pred.colors) >= 1, f"no color extracted from {h!r}"
