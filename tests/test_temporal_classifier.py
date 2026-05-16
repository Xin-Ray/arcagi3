"""Tests for arc_agent.temporal_classifier."""
from __future__ import annotations

import numpy as np
import pytest

from arc_agent.object_extractor import extract_objects
from arc_agent.temporal_classifier import (
    Layer,
    MIN_FRAMES_TO_CLASSIFY,
    TEXTURE_COUNT_MIN,
    classify_frame,
    filter_active,
    filter_non_texture,
    is_likely_texture,
    texture_summary,
    update_history,
)


def _grid(spec: dict, shape=(10, 10)) -> np.ndarray:
    g = np.zeros(shape, dtype=int)
    for (r, c), v in spec.items():
        g[r, c] = v
    return g


# ── is_likely_texture ─────────────────────────────────────────────────────


def test_is_likely_texture_fires_on_many_single_cells() -> None:
    # 25 same-color single cells -> texture
    g = np.zeros((20, 20), dtype=int)
    for r in range(0, 10, 2):
        for c in range(0, 10, 2):
            g[r, c] = 3
    objs = extract_objects(g)
    assert len(objs) >= TEXTURE_COUNT_MIN
    # All of them should be flagged
    for o in objs:
        assert is_likely_texture(o, objs)


def test_is_likely_texture_silent_on_few_objects() -> None:
    g = _grid({(0, 0): 3, (5, 5): 3})
    objs = extract_objects(g)
    for o in objs:
        assert not is_likely_texture(o, objs)


def test_is_likely_texture_silent_on_large_object() -> None:
    g = np.zeros((10, 10), dtype=int)
    g[0:3, 0:3] = 3   # 9-cell solid
    objs = extract_objects(g)
    assert not is_likely_texture(objs[0], objs)


# ── classify_frame ────────────────────────────────────────────────────────


def test_candidate_when_history_short() -> None:
    g = _grid({(2, 2): 5})
    objs = extract_objects(g)
    history: dict = {}
    update_history(history, objs)
    layer = classify_frame(objs, history)
    assert layer[objs[0].id] == Layer.CANDIDATE


def test_static_after_min_frames_unchanged() -> None:
    g = _grid({(2, 2): 5})
    objs = extract_objects(g)
    history: dict = {}
    layer = {}
    for _ in range(MIN_FRAMES_TO_CLASSIFY):
        update_history(history, objs)
        layer = classify_frame(objs, history)
    assert layer[objs[0].id] == Layer.STATIC


def test_active_when_position_changed() -> None:
    """Object's signature is (color, shape_sig) — moving keeps the
    signature but the anchor changes, so over N frames with different
    anchors it becomes ACTIVE."""
    history: dict = {}
    # 3 frames with the same shape (1x1) at 3 different positions
    for r in (1, 2, 3):
        g = _grid({(r, 4): 5})
        objs = extract_objects(g)
        update_history(history, objs)
    layer = classify_frame(objs, history)
    assert layer[objs[0].id] == Layer.ACTIVE


def test_classify_marks_texture_first() -> None:
    """Big patterned background gets TEXTURE, not STATIC, even if
    it also hasn't moved across frames."""
    g = np.zeros((20, 20), dtype=int)
    for r in range(0, 10, 2):
        for c in range(0, 10, 2):
            g[r, c] = 3
    objs = extract_objects(g)
    history: dict = {}
    for _ in range(MIN_FRAMES_TO_CLASSIFY):
        update_history(history, objs)
    layer = classify_frame(objs, history)
    assert all(v == Layer.TEXTURE for v in layer.values())


# ── filter helpers ────────────────────────────────────────────────────────


def test_filter_active_returns_only_active() -> None:
    objs = [
        type("O", (), {"id": 0})(),
        type("O", (), {"id": 1})(),
        type("O", (), {"id": 2})(),
    ]
    layer = {0: Layer.ACTIVE, 1: Layer.STATIC, 2: Layer.ACTIVE}
    kept = filter_active(objs, layer)
    assert {o.id for o in kept} == {0, 2}


def test_filter_non_texture() -> None:
    objs = [
        type("O", (), {"id": 0})(),
        type("O", (), {"id": 1})(),
        type("O", (), {"id": 2})(),
    ]
    layer = {0: Layer.TEXTURE, 1: Layer.STATIC, 2: Layer.ACTIVE}
    kept = filter_non_texture(objs, layer)
    assert {o.id for o in kept} == {1, 2}


def test_texture_summary_aggregates_by_color() -> None:
    g = np.zeros((20, 20), dtype=int)
    for r in range(0, 10, 2):
        for c in range(0, 10, 2):
            g[r, c] = 3
    objs = extract_objects(g)
    history: dict = {}
    for _ in range(MIN_FRAMES_TO_CLASSIFY):
        update_history(history, objs)
    layer = classify_frame(objs, history)
    summary = texture_summary(objs, layer)
    assert summary["texture_cells_total"] >= TEXTURE_COUNT_MIN
    assert "green" in summary["by_color"]


# ── update_history mutation ───────────────────────────────────────────────


def test_update_history_appends_anchors() -> None:
    history: dict = {}
    g1 = _grid({(2, 2): 5})
    g2 = _grid({(3, 2): 5})  # moved down
    update_history(history, extract_objects(g1))
    update_history(history, extract_objects(g2))
    # key = (color=5, shape_sig=((0,0),))
    key = list(history.keys())[0]
    assert history[key] == [(2, 2), (3, 2)]
