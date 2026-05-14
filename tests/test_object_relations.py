"""Unit tests for arc_agent.object_relations."""
from __future__ import annotations

import pytest

from arc_agent.object_extractor import ObjectRecord
from arc_agent.object_relations import (
    ObjectRelations,
    compute_relations,
    render_relations_block,
)
from arc_agent.temporal_classifier import Layer


def _obj(id_: int, color: int, color_name: str,
         bbox: tuple[int, int, int, int], size: int = 1,
         center: tuple[float, float] | None = None) -> ObjectRecord:
    r0, c0, r1, c1 = bbox
    cells = [(r0, c0)]
    return ObjectRecord(
        id=id_, color=color, color_name=color_name,
        cells=cells, bbox=bbox, size=size,
        center=center if center is not None else ((r0 + r1) / 2, (c0 + c1) / 2),
    )


# ── same-color groups ────────────────────────────────────────────────────


def test_same_color_groups_2_reds() -> None:
    objs = [
        _obj(0, 2, "red", (0, 0, 0, 0)),
        _obj(1, 2, "red", (10, 10, 10, 10)),
        _obj(2, 1, "blue", (5, 5, 5, 5)),
    ]
    r = compute_relations(objs)
    assert r.same_color_groups == {"red": [0, 1]}


def test_same_color_groups_skipped_when_unique() -> None:
    """Objects with unique colors aren't grouped."""
    objs = [
        _obj(0, 2, "red", (0, 0, 0, 0)),
        _obj(1, 1, "blue", (5, 5, 5, 5)),
    ]
    r = compute_relations(objs)
    assert r.same_color_groups == {}


def test_same_color_three_or_more() -> None:
    objs = [_obj(i, 2, "red", (i, i, i, i)) for i in range(4)]
    r = compute_relations(objs)
    assert r.same_color_groups["red"] == [0, 1, 2, 3]


# ── same-shape groups ────────────────────────────────────────────────────


def test_same_shape_groups_two_2x2() -> None:
    """Two 2x2 objects with size 4 each get grouped by shape_key."""
    objs = [
        _obj(0, 2, "red", (0, 0, 1, 1), size=4),
        _obj(1, 1, "blue", (10, 10, 11, 11), size=4),
        _obj(2, 4, "yellow", (20, 20, 20, 20), size=1),
    ]
    r = compute_relations(objs)
    # both 2x2 size 4 -> grouped under "2x2_size4"
    assert any("2x2_size4" == k for k in r.same_shape_groups)
    assert sorted(r.same_shape_groups["2x2_size4"]) == [0, 1]


def test_same_shape_different_sizes_not_grouped() -> None:
    """2x2 hollow (size=4) and 3x3 hollow (size=8) are different keys."""
    objs = [
        _obj(0, 2, "red", (0, 0, 1, 1), size=4),
        _obj(1, 2, "red", (5, 5, 7, 7), size=8),
    ]
    r = compute_relations(objs)
    # color group still kicks in
    assert "red" in r.same_color_groups
    # but shape keys differ -- no shape group with both
    assert all(len(ids) < 2 or 0 not in ids or 1 not in ids
               for ids in r.same_shape_groups.values())


# ── closest pairs ────────────────────────────────────────────────────────


def test_closest_pairs_sorted_by_distance() -> None:
    objs = [
        _obj(0, 2, "red", (0, 0, 0, 0)),
        _obj(1, 2, "red", (0, 1, 0, 1)),   # dist 1.0
        _obj(2, 2, "red", (0, 10, 0, 10)),  # dist 10 from #0
    ]
    r = compute_relations(objs)
    assert r.closest_pairs[0] == (0, 1, pytest.approx(1.0))
    # the pair (0, 2) and (1, 2) should follow
    assert r.closest_pairs[1][2] < r.closest_pairs[2][2]


def test_closest_pairs_capped() -> None:
    """More than 5 objects -> at most 5 pairs returned."""
    objs = [_obj(i, 2, "red", (0, i, 0, i)) for i in range(8)]
    r = compute_relations(objs)
    assert len(r.closest_pairs) <= 5


def test_closest_pairs_empty_for_singleton() -> None:
    r = compute_relations([_obj(0, 2, "red", (0, 0, 0, 0))])
    assert r.closest_pairs == []


# ── edge distances ───────────────────────────────────────────────────────


def test_edge_distance_top_left_corner() -> None:
    obj = _obj(0, 2, "red", (0, 0, 0, 0))
    r = compute_relations([obj], grid_shape=(64, 64))
    ed = r.edge_distances[0]
    assert ed == {"top": 0, "bottom": 63, "left": 0, "right": 63}


def test_edge_distance_center() -> None:
    obj = _obj(0, 2, "red", (32, 32, 33, 33))
    r = compute_relations([obj], grid_shape=(64, 64))
    ed = r.edge_distances[0]
    assert ed == {"top": 32, "bottom": 30, "left": 32, "right": 30}


# ── quadrant ─────────────────────────────────────────────────────────────


def test_quadrant_corners() -> None:
    objs = [
        _obj(0, 2, "red", (1, 1, 1, 1), center=(1, 1)),       # top-left
        _obj(1, 2, "red", (1, 62, 1, 62), center=(1, 62)),    # top-right
        _obj(2, 2, "red", (62, 1, 62, 1), center=(62, 1)),    # bottom-left
        _obj(3, 2, "red", (62, 62, 62, 62), center=(62, 62)),  # bottom-right
    ]
    r = compute_relations(objs, grid_shape=(64, 64))
    assert r.quadrant[0] == "top-left"
    assert r.quadrant[1] == "top-right"
    assert r.quadrant[2] == "bottom-left"
    assert r.quadrant[3] == "bottom-right"


def test_quadrant_center() -> None:
    obj = _obj(0, 2, "red", (32, 32, 32, 32), center=(32, 32))
    r = compute_relations([obj], grid_shape=(64, 64))
    assert r.quadrant[0] == "center"


# ── layer_by_id filtering (skip texture) ─────────────────────────────────


def test_texture_objects_excluded_by_default() -> None:
    objs = [
        _obj(0, 2, "red", (0, 0, 0, 0)),
        _obj(1, 2, "red", (10, 10, 10, 10)),
    ]
    layer = {0: Layer.ACTIVE, 1: Layer.TEXTURE}
    r = compute_relations(objs, layer_by_id=layer)
    # only obj 0 considered (texture skipped) -> no group of 2
    assert r.same_color_groups == {}
    assert r.active_ids == {0}


def test_texture_objects_kept_when_skip_disabled() -> None:
    objs = [
        _obj(0, 2, "red", (0, 0, 0, 0)),
        _obj(1, 2, "red", (10, 10, 10, 10)),
    ]
    layer = {0: Layer.ACTIVE, 1: Layer.TEXTURE}
    r = compute_relations(objs, layer_by_id=layer, skip_texture=False)
    assert r.same_color_groups == {"red": [0, 1]}


# ── render block ─────────────────────────────────────────────────────────


def test_render_includes_color_groups() -> None:
    r = ObjectRelations(
        same_color_groups={"red": [0, 1, 2]},
    )
    out = render_relations_block(r)
    assert "[OBJECT RELATIONS]" in out
    assert "red" in out
    assert "#0" in out and "#1" in out and "#2" in out


def test_render_includes_shape_groups() -> None:
    r = ObjectRelations(
        same_shape_groups={"2x2_size4": [0, 5]},
    )
    out = render_relations_block(r)
    assert "2x2_size4" in out


def test_render_includes_closest_pairs() -> None:
    r = ObjectRelations(closest_pairs=[(0, 1, 3.2), (2, 4, 7.0)])
    out = render_relations_block(r)
    assert "#0 <-> #1" in out
    assert "3.2" in out


def test_render_includes_edge_clearance_for_active() -> None:
    r = ObjectRelations(
        active_ids={5},
        edge_distances={5: {"top": 10, "bottom": 50, "left": 5, "right": 58}},
        quadrant={5: "top-left"},
    )
    out = render_relations_block(r)
    assert "#5" in out
    assert "top-left" in out
    assert "top=10" in out


def test_render_focus_ids_overrides_active() -> None:
    r = ObjectRelations(
        active_ids={1, 2, 3},
        edge_distances={1: {"top": 0, "bottom": 0, "left": 0, "right": 0},
                        2: {"top": 0, "bottom": 0, "left": 0, "right": 0},
                        3: {"top": 0, "bottom": 0, "left": 0, "right": 0}},
        quadrant={1: "x", 2: "y", 3: "z"},
    )
    out = render_relations_block(r, focus_ids={2})
    # Only object 2 in the placement section
    placement = out.split("active object placement:")
    if len(placement) > 1:
        body = placement[1]
        assert "#2" in body
        # Other ids not in placement (could appear elsewhere though)
        assert "#1" not in body
        assert "#3" not in body


def test_render_empty_relations_returns_marker() -> None:
    out = render_relations_block(None)
    assert "[OBJECT RELATIONS]" in out
    assert "no data" in out
