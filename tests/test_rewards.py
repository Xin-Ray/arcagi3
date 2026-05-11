"""Tests for arc_agent.rewards (verifier primitives)."""
from __future__ import annotations

import numpy as np
import pytest

from arc_agent.rewards import changes_to_set, real_changes, verify_prediction_f1


# ---- changes_to_set ----------------------------------------------------


def test_changes_to_set_well_formed_list() -> None:
    payload = [
        {"row": 10, "col": 3, "to_color": 2},
        {"row": 11, "col": 3, "to_color": 5},
    ]
    assert changes_to_set(payload) == {(10, 3, 2), (11, 3, 5)}


def test_changes_to_set_empty_list() -> None:
    assert changes_to_set([]) == set()


def test_changes_to_set_drops_malformed_entries() -> None:
    payload = [
        {"row": 0, "col": 0, "to_color": 1},     # valid
        {"row": 1, "col": 1},                    # missing key
        {"row": "x", "col": 2, "to_color": 3},   # non-int row
        "not a dict",                            # wrong type
        {"row": 5, "col": 5, "to_color": 7},     # valid
    ]
    assert changes_to_set(payload) == {(0, 0, 1), (5, 5, 7)}


def test_changes_to_set_non_list_input_returns_empty() -> None:
    assert changes_to_set(None) == set()
    assert changes_to_set("string") == set()
    assert changes_to_set({"row": 0, "col": 0, "to_color": 1}) == set()


def test_changes_to_set_deduplicates() -> None:
    payload = [
        {"row": 0, "col": 0, "to_color": 1},
        {"row": 0, "col": 0, "to_color": 1},  # duplicate
    ]
    assert changes_to_set(payload) == {(0, 0, 1)}


# ---- real_changes ------------------------------------------------------


def test_real_changes_identical_grids_empty() -> None:
    g = np.array([[1, 2], [3, 4]])
    assert real_changes(g, g) == set()


def test_real_changes_single_cell_changed() -> None:
    s_t = np.array([[0, 0], [0, 0]])
    s_tp1 = np.array([[0, 5], [0, 0]])
    assert real_changes(s_t, s_tp1) == {(0, 1, 5)}


def test_real_changes_multiple_cells() -> None:
    s_t = np.array([[1, 2], [3, 4]])
    s_tp1 = np.array([[1, 7], [9, 4]])
    assert real_changes(s_t, s_tp1) == {(0, 1, 7), (1, 0, 9)}


def test_real_changes_shape_mismatch_raises() -> None:
    a = np.zeros((3, 3))
    b = np.zeros((3, 4))
    with pytest.raises(ValueError):
        real_changes(a, b)


# ---- verify_prediction_f1 ---------------------------------------------


def test_f1_perfect_match() -> None:
    pred = {(0, 1, 5), (1, 0, 9)}
    real = {(0, 1, 5), (1, 0, 9)}
    assert verify_prediction_f1(pred, real) == 1.0


def test_f1_both_empty_means_perfect() -> None:
    """Agent correctly predicted "nothing changes" — full credit."""
    assert verify_prediction_f1(set(), set()) == 1.0


def test_f1_predicted_empty_but_real_nonempty() -> None:
    """Agent missed all changes."""
    assert verify_prediction_f1(set(), {(0, 0, 1)}) == 0.0


def test_f1_predicted_nonempty_but_real_empty() -> None:
    """Agent hallucinated changes that did not happen."""
    assert verify_prediction_f1({(0, 0, 1)}, set()) == 0.0


def test_f1_no_intersection_is_zero() -> None:
    pred = {(0, 0, 1)}
    real = {(1, 1, 2)}
    assert verify_prediction_f1(pred, real) == 0.0


def test_f1_partial_overlap_known_value() -> None:
    # Predicted: {a, b, c}, Real: {a, b}
    # TP=2, FP=1, FN=0 → P = 2/3, R = 2/2 = 1 → F1 = 2 * (2/3) * 1 / (2/3 + 1) = (4/3) / (5/3) = 0.8
    pred = {(0, 0, 1), (1, 0, 2), (2, 0, 3)}
    real = {(0, 0, 1), (1, 0, 2)}
    assert verify_prediction_f1(pred, real) == pytest.approx(0.8)


def test_f1_wrong_color_at_right_position_counts_as_miss() -> None:
    """Position right, color wrong → set elements differ → no credit for that cell."""
    pred = {(0, 0, 1)}
    real = {(0, 0, 2)}
    assert verify_prediction_f1(pred, real) == 0.0


def test_f1_symmetric() -> None:
    pred = {(0, 0, 1), (1, 0, 2), (2, 0, 3)}
    real = {(0, 0, 1), (1, 0, 2)}
    assert verify_prediction_f1(pred, real) == verify_prediction_f1(real, pred)


# ---- end-to-end: combine primitives ------------------------------------


def test_end_to_end_full_pipeline() -> None:
    """Simulate one verifier step: Qwen JSON → set → compare with real grid diff."""
    s_t = np.array([[0, 0, 0], [0, 0, 0]])
    s_tp1 = np.array([[0, 5, 0], [0, 0, 7]])

    qwen_output = [
        {"row": 0, "col": 1, "to_color": 5},   # correct prediction
        {"row": 1, "col": 2, "to_color": 9},   # wrong color (real is 7)
    ]

    predicted = changes_to_set(qwen_output)
    real = real_changes(s_t, s_tp1)

    # predicted: {(0,1,5), (1,2,9)}
    # real:      {(0,1,5), (1,2,7)}
    # intersection: {(0,1,5)} → TP=1, P=1/2, R=1/2, F1 = 0.5
    assert verify_prediction_f1(predicted, real) == pytest.approx(0.5)
