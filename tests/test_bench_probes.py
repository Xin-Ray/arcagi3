"""Sanity checks for the spatial-reasoning probe suite."""
from __future__ import annotations

import pytest

from arc_agent.bench_probes import get_probes, N_PROBES, CATEGORIES


def test_probe_count():
    assert N_PROBES >= 20


def test_categories_complete():
    """All 8 categories T1..T8 are present."""
    assert "T1" in CATEGORIES
    assert "T2" in CATEGORIES
    assert "T3" in CATEGORIES
    assert "T4" in CATEGORIES
    assert "T5" in CATEGORIES
    assert "T6" in CATEGORIES
    assert "T7" in CATEGORIES
    assert "T8" in CATEGORIES


def test_each_probe_has_required_fields():
    for p in get_probes():
        assert "id" in p
        assert "cat" in p
        assert "q" in p
        assert "options" in p
        assert "correct" in p
        # Options must be A-D
        assert set(p["options"].keys()) == {"A", "B", "C", "D"}
        # Correct answer must be a valid option letter
        assert p["correct"] in p["options"]


def test_no_duplicate_ids():
    probes = get_probes()
    ids = [p["id"] for p in probes]
    assert len(ids) == len(set(ids)), "Probe IDs must be unique"


def test_returned_list_is_a_copy():
    """Mutating the returned list must not affect future calls."""
    probes1 = get_probes()
    probes1[0]["q"] = "MUTATED"
    probes2 = get_probes()
    assert probes2[0]["q"] != "MUTATED"
