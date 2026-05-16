"""Tests for arc_agent.click_candidates."""
from __future__ import annotations

import numpy as np

from arc_agent.click_candidates import (
    list_click_candidates,
    pick_default_action6_coords,
    render_click_candidates_block,
)
from arc_agent.object_extractor import extract_objects
from arc_agent.temporal_classifier import Layer


def _grid(spec, shape=(20, 20)) -> np.ndarray:
    g = np.zeros(shape, dtype=int)
    for (r, c), v in spec.items():
        g[r, c] = v
    return g


def test_empty_frame_no_candidates() -> None:
    g = np.zeros((10, 10), dtype=int)
    objs = extract_objects(g)
    cands = list_click_candidates(objs, {})
    assert cands == []


def test_small_active_object_becomes_candidate() -> None:
    g = _grid({(5, 7): 3})
    objs = extract_objects(g)
    layer = {o.id: Layer.ACTIVE for o in objs}
    cands = list_click_candidates(objs, layer)
    assert len(cands) >= 1
    # ObjectRecord.center = (row=5, col=7) -> ClickCandidate (x=7, y=5)
    assert cands[0].x == 7
    assert cands[0].y == 5
    assert "button" in cands[0].why.lower() or "active" in cands[0].why.lower()


def test_color_match_pair_gets_priority() -> None:
    g = _grid({(2, 2): 3, (12, 12): 3})
    objs = extract_objects(g)
    layer = {objs[0].id: Layer.ACTIVE, objs[1].id: Layer.STATIC}
    cands = list_click_candidates(objs, layer)
    descriptions = "\n".join(c.why.lower() for c in cands)
    assert "matching" in descriptions or "target" in descriptions


def test_pick_default_returns_some_coord() -> None:
    g = _grid({(5, 7): 3})
    objs = extract_objects(g)
    layer = {o.id: Layer.ACTIVE for o in objs}
    coords = pick_default_action6_coords(objs, layer)
    assert coords is not None
    assert 0 <= coords[0] <= 63
    assert 0 <= coords[1] <= 63


def test_pick_default_skips_tried_coords() -> None:
    """When the obvious candidate has been tried, return a different one."""
    g = _grid({(5, 7): 3, (10, 12): 4})
    objs = extract_objects(g)
    layer = {o.id: Layer.ACTIVE for o in objs}
    first = pick_default_action6_coords(objs, layer)
    second = pick_default_action6_coords(objs, layer, tried_coords=[first])
    assert second != first


def test_render_block_empty() -> None:
    out = render_click_candidates_block([])
    assert "no recommended click targets" in out.lower()


def test_render_block_non_empty() -> None:
    from arc_agent.click_candidates import ClickCandidate
    out = render_click_candidates_block([
        ClickCandidate(x=10, y=20, why="test1", source_uid="id=0"),
        ClickCandidate(x=30, y=40, why="test2", source_uid="id=1"),
    ])
    assert "(10, 20)" in out
    assert "(30, 40)" in out
    assert "test1" in out


def test_texture_layer_objects_excluded() -> None:
    """Objects tagged TEXTURE should never be candidates."""
    g = _grid({(5, 5): 3, (10, 10): 3})
    objs = extract_objects(g)
    layer = {o.id: Layer.TEXTURE for o in objs}
    cands = list_click_candidates(objs, layer)
    # Both are texture -> no candidates
    assert cands == []
