"""Sanity tests for subtask_probes."""
from __future__ import annotations

import random

from arc_agent.subtask_probes import (
    GENERATORS, gen_T_NAV_1, gen_T_NAV_2, gen_T_NAV_3,
    gen_T_SEL_1, gen_T_GOAL,
)


def test_T_NAV_1_basic():
    probes = gen_T_NAV_1(n=10, seed=42)
    assert len(probes) == 10
    for p in probes:
        assert p["cat"] == "T-NAV-1"
        assert set(p["options"].keys()) == {"A", "B", "C", "D"}
        assert p["correct"] in p["options"]
        # Verify the correct answer matches metadata
        ca = p["_meta"]["correct_action"]
        assert ca in p["options"][p["correct"]]


def test_T_NAV_1_direction_sanity():
    """When target is to the RIGHT, correct_action must be ACTION4."""
    probes = gen_T_NAV_1(n=200, seed=42)
    for p in probes:
        obj = p["_meta"]["object"]
        tgt = p["_meta"]["target"]
        ca = p["_meta"]["correct_action"]
        if tgt[0] < obj[0]:
            assert ca == "ACTION1"
        elif tgt[0] > obj[0]:
            assert ca == "ACTION2"
        elif tgt[1] < obj[1]:
            assert ca == "ACTION3"
        elif tgt[1] > obj[1]:
            assert ca == "ACTION4"


def test_T_NAV_2_count_sanity():
    probes = gen_T_NAV_2(n=50, seed=42)
    for p in probes:
        n = p["_meta"]["n_steps"]
        correct_option = p["options"][p["correct"]]
        assert f"{n} times" in correct_option


def test_T_NAV_3_total_sanity():
    probes = gen_T_NAV_3(n=50, seed=42)
    for p in probes:
        total = p["_meta"]["n_total"]
        correct_option = p["options"][p["correct"]]
        # Correct option contains total count
        assert f"{total} total" in correct_option


def test_T_SEL_1_bbox_contains_correct():
    probes = gen_T_SEL_1(n=20, seed=42)
    for p in probes:
        bbox = p["_meta"]["target_bbox"]
        r1, c1, r2, c2 = bbox
        # Correct option must contain coords inside bbox
        correct_str = p["options"][p["correct"]]
        # Parse "ACTION6 x=N y=M"
        import re
        m = re.search(r"x=(\d+) y=(\d+)", correct_str)
        assert m, f"Couldn't parse correct option: {correct_str}"
        x, y = int(m.group(1)), int(m.group(2))
        assert c1 <= x <= c2, f"x={x} not in [{c1}, {c2}]"
        assert r1 <= y <= r2, f"y={y} not in [{r1}, {r2}]"


def test_T_GOAL_success_vs_failure():
    probes = gen_T_GOAL(n=100, seed=42)
    successes = [p for p in probes if p["_meta"]["success"]]
    failures = [p for p in probes if not p["_meta"]["success"]]
    assert len(successes) >= 5  # at least some
    assert len(failures) >= 5
    # All successes' correct answer must be the YES option
    for p in successes:
        assert "YES" in p["options"][p["correct"]]
    for p in failures:
        assert "NO" in p["options"][p["correct"]]


def test_all_generators_in_registry():
    assert "T-NAV-1" in GENERATORS
    assert "T-NAV-2" in GENERATORS
    assert "T-NAV-3" in GENERATORS
    assert "T-SEL-1" in GENERATORS
    assert "T-GOAL" in GENERATORS


def test_seeded_reproducible():
    p1 = gen_T_NAV_1(n=20, seed=99)
    p2 = gen_T_NAV_1(n=20, seed=99)
    assert [x["q"] for x in p1] == [x["q"] for x in p2]
