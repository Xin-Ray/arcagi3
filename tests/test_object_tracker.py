"""Tests for arc_agent.object_tracker.ObjectMemory."""
from __future__ import annotations

import numpy as np

from arc_agent.object_aligner import align_objects
from arc_agent.object_extractor import extract_objects
from arc_agent.object_tracker import ObjectMemory, ObjectSnapshot, TrackedObject


def _grid(spec: dict, shape=(10, 10)) -> np.ndarray:
    g = np.zeros(shape, dtype=int)
    for (r, c), v in spec.items():
        g[r, c] = v
    return g


def test_empty_memory_has_no_alive_objects() -> None:
    mem = ObjectMemory()
    assert mem.alive_tracked() == []


def test_first_step_assigns_uid() -> None:
    objs = extract_objects(_grid({(2, 2): 5}))
    mem = ObjectMemory()
    # No prior step, so use empty matches; current_active alone seeds memory.
    mem.update(step=0, current_active=objs, matches=[])
    alive = mem.alive_tracked()
    assert len(alive) == 1
    assert alive[0].uid == "obj_000"
    assert alive[0].color == 5
    assert len(alive[0].history) == 1
    assert alive[0].history[0].step == 0


def test_uid_persists_when_object_moves() -> None:
    g1 = _grid({(2, 2): 5})
    g2 = _grid({(3, 2): 5})   # moved down
    objs1 = extract_objects(g1)
    objs2 = extract_objects(g2)
    mem = ObjectMemory()
    mem.update(step=0, current_active=objs1, matches=[])
    matches = align_objects(objs1, objs2)
    mem.update(step=1, current_active=objs2, matches=matches)
    alive = mem.alive_tracked()
    assert len(alive) == 1
    assert alive[0].uid == "obj_000"
    assert len(alive[0].history) == 2
    assert alive[0].history[1].bbox == (3, 2, 3, 2)


def test_two_separate_objects_get_two_uids() -> None:
    g = _grid({(1, 1): 2, (5, 5): 3})
    objs = extract_objects(g)
    mem = ObjectMemory()
    mem.update(step=0, current_active=objs, matches=[])
    alive = mem.alive_tracked()
    assert len(alive) == 2
    uids = {t.uid for t in alive}
    assert uids == {"obj_000", "obj_001"}


def test_disappeared_marks_object_dead() -> None:
    g1 = _grid({(1, 1): 5})
    g2 = _grid({})   # empty
    objs1 = extract_objects(g1)
    objs2 = extract_objects(g2)
    mem = ObjectMemory()
    mem.update(step=0, current_active=objs1, matches=[])
    mem.update(step=1, current_active=objs2,
               matches=align_objects(objs1, objs2))
    alive = mem.alive_tracked()
    assert alive == []
    # but the tracked object still in dict, marked dead
    assert mem.get("obj_000") is not None
    assert mem.get("obj_000").alive is False


def test_appeared_creates_new_uid() -> None:
    g1 = _grid({(1, 1): 5})
    g2 = _grid({(1, 1): 5, (8, 8): 7})   # new object appeared
    objs1 = extract_objects(g1)
    objs2 = extract_objects(g2)
    mem = ObjectMemory()
    mem.update(step=0, current_active=objs1, matches=[])
    mem.update(step=1, current_active=objs2,
               matches=align_objects(objs1, objs2))
    alive = mem.alive_tracked()
    assert len(alive) == 2
    uids = sorted(t.uid for t in alive)
    assert uids == ["obj_000", "obj_001"]


def test_descriptor_format() -> None:
    objs = extract_objects(_grid({(2, 3): 5}))
    mem = ObjectMemory()
    mem.update(step=0, current_active=objs, matches=[])
    t = mem.alive_tracked()[0]
    d = t.descriptor
    assert "obj_000" in d
    assert "gray" in d
    assert "size=1" in d


def test_reset() -> None:
    mem = ObjectMemory()
    mem.update(step=0, current_active=extract_objects(_grid({(0, 0): 5})),
               matches=[])
    assert len(mem.alive_tracked()) == 1
    mem.reset()
    assert mem.alive_tracked() == []
    # uid counter also reset
    mem.update(step=0, current_active=extract_objects(_grid({(0, 0): 5})),
               matches=[])
    assert mem.alive_tracked()[0].uid == "obj_000"


def test_multistep_tracks_history_chronologically() -> None:
    """Move an object through 4 steps; history should reflect every step."""
    mem = ObjectMemory()
    grids = [
        _grid({(2, 2): 5}),
        _grid({(2, 3): 5}),
        _grid({(2, 4): 5}),
        _grid({(2, 5): 5}),
    ]
    prev_objs = None
    for step, g in enumerate(grids):
        objs = extract_objects(g)
        if prev_objs is None:
            mem.update(step=step, current_active=objs, matches=[])
        else:
            mem.update(step=step, current_active=objs,
                       matches=align_objects(prev_objs, objs))
        prev_objs = objs
    alive = mem.alive_tracked()
    assert len(alive) == 1
    t = alive[0]
    assert len(t.history) == 4
    # Centroids progress right by 1 each step
    cols = [snap.center[1] for snap in t.history]
    assert cols == [2.0, 3.0, 4.0, 5.0]
