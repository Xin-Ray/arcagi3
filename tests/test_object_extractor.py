"""Unit tests for arc_agent.object_extractor."""
from __future__ import annotations

import numpy as np
import pytest

from arc_agent.object_extractor import (
    ObjectRecord,
    _shape_signature,
    extract_objects,
    objects_to_dict,
)


def test_empty_grid_returns_no_objects() -> None:
    g = np.zeros((4, 4), dtype=int)
    assert extract_objects(g) == []


def test_single_cell_object() -> None:
    g = np.zeros((4, 4), dtype=int)
    g[1, 2] = 2
    objs = extract_objects(g)
    assert len(objs) == 1
    o = objs[0]
    assert o.color == 2
    assert o.cells == [(1, 2)]
    assert o.bbox == (1, 2, 1, 2)
    assert o.size == 1
    assert o.description == "single cell"


def test_L_shape_extraction() -> None:
    g = np.zeros((5, 5), dtype=int)
    # An L:
    #   . . . . .
    #   . X . . .
    #   . X . . .
    #   . X X . .
    #   . . . . .
    g[1, 1] = 3
    g[2, 1] = 3
    g[3, 1] = 3
    g[3, 2] = 3
    objs = extract_objects(g)
    assert len(objs) == 1
    o = objs[0]
    assert o.size == 4
    assert o.color == 3
    assert set(o.cells) == {(1, 1), (2, 1), (3, 1), (3, 2)}


def test_two_same_color_disjoint_blobs_split() -> None:
    g = np.zeros((6, 6), dtype=int)
    g[1, 1] = 5
    g[4, 4] = 5
    objs = extract_objects(g)
    assert len(objs) == 2
    assert {o.size for o in objs} == {1}


def test_two_colors_separate() -> None:
    g = np.zeros((4, 4), dtype=int)
    g[0, 0] = 1
    g[3, 3] = 2
    objs = extract_objects(g)
    assert len(objs) == 2
    colors = sorted(o.color for o in objs)
    assert colors == [1, 2]


def test_diagonal_not_4_connected() -> None:
    """Two cells touching only diagonally must split into 2 objects."""
    g = np.zeros((3, 3), dtype=int)
    g[0, 0] = 1
    g[1, 1] = 1
    objs = extract_objects(g)
    assert len(objs) == 2


def test_2d_required() -> None:
    with pytest.raises(ValueError):
        extract_objects(np.zeros((4, 4, 4), dtype=int))


def test_shape_signature_translation_invariant() -> None:
    cells_a = [(2, 3), (2, 4), (3, 3)]
    cells_b = [(10, 1), (10, 2), (11, 1)]
    assert _shape_signature(cells_a) == _shape_signature(cells_b)


def test_shape_signature_differs_on_different_shapes() -> None:
    sig_l = _shape_signature([(0, 0), (1, 0), (1, 1)])
    sig_line = _shape_signature([(0, 0), (0, 1), (0, 2)])
    assert sig_l != sig_line


def test_objects_to_dict_round_trip_shape() -> None:
    g = np.zeros((3, 3), dtype=int)
    g[0, 0] = 4
    payload = objects_to_dict(extract_objects(g))
    assert "objects" in payload
    assert len(payload["objects"]) == 1
    o = payload["objects"][0]
    # All JSON-friendly types
    assert isinstance(o["cells"], list)
    assert isinstance(o["bbox"], list)
    assert isinstance(o["size"], int)


def test_full_64x64_no_crash() -> None:
    """Sanity check on a realistic ARC grid size."""
    rng = np.random.default_rng(0)
    g = rng.integers(0, 4, size=(64, 64), dtype=int)
    objs = extract_objects(g)
    # No assertions on count — random grids have many small blobs.
    assert all(o.color != 0 for o in objs)
    assert all(o.size >= 1 for o in objs)
