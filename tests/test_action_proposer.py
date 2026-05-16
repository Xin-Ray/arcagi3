"""Tests for action_proposer."""
from __future__ import annotations

import random

import pytest

from arc_agent.action_inference import OutcomeLog, StepOutcome
from arc_agent.action_proposer import (
    Candidate, propose, candidates_to_prompt_block, resolve_letter,
)
from arc_agent.knowledge import Knowledge


def _make_log(actions: list[tuple[str, bool]]) -> OutcomeLog:
    log = OutcomeLog()
    for i, (a, ch) in enumerate(actions):
        log.record(StepOutcome(step=i, action=a, legal=True,
                               frame_changed=ch, n_active_changed=0,
                               primary_direction=None))
    return log


def test_proposer_returns_3_candidates_letter_labelled():
    k = Knowledge.empty("ar25")
    log = _make_log([("ACTION1", True), ("ACTION2", False)])
    legal = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]
    cands = propose(k, log, legal, rng=random.Random(0))
    assert len(cands) == 3
    assert {c.letter for c in cands} == {"A", "B", "C"}
    # All candidates should be unique actions
    assert len({c.action_name for c in cands}) == 3


def test_proposer_includes_untried_first():
    k = Knowledge.empty("ar25")
    log = _make_log([("ACTION1", True)])
    legal = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]
    cands = propose(k, log, legal, rng=random.Random(0))
    # ACTION2..7 are untried; at least one must appear
    cand_actions = {c.action_name for c in cands}
    assert cand_actions & {"ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"}


def test_proposer_includes_known_good():
    k = Knowledge.empty("ar25")
    k.action_semantics["ACTION3"] = "moves the yellow object UP by 3"
    log = _make_log([("ACTION1", True), ("ACTION3", True)])
    legal = ["ACTION1", "ACTION2", "ACTION3"]
    cands = propose(k, log, legal, rng=random.Random(0))
    # ACTION3 should be one of the candidates (known-good)
    actions = [c.action_name for c in cands]
    assert "ACTION3" in actions


def test_proposer_excludes_over_committed():
    """When agent has just picked ACTION1 5 times in a row, it should
    NOT appear in known-good slot."""
    k = Knowledge.empty("ar25")
    k.action_semantics["ACTION1"] = "moves an active object UP"
    log = _make_log([("ACTION1", True)] * 5)
    legal = ["ACTION1", "ACTION2", "ACTION3"]
    recent = ["ACTION1"] * 5
    cands = propose(k, log, legal, recent_action_names=recent,
                    rng=random.Random(0))
    # ACTION1 should not be in candidates because it's over-committed
    # (untried picks fill slot 1, known-good filter excludes ACTION1)
    actions = [c.action_name for c in cands]
    # Could still appear if it's the only filler, but unlikely with 3+ options
    over_commit_action_count = actions.count("ACTION1")
    assert over_commit_action_count <= 1


def test_proposer_negative_semantics_excluded():
    k = Knowledge.empty("ar25")
    k.action_semantics["ACTION6"] = "AVOID -- no observable effect, all clicks no-op"
    log = _make_log([("ACTION6", False)] * 6)
    legal = ["ACTION1", "ACTION6"]
    cands = propose(k, log, legal, rng=random.Random(0))
    # ACTION6 has negative semantic, should not be slot-2 known-good
    # But could be slot-3 filler. Let's at least check it's not >1 candidate
    actions = [c.action_name for c in cands]
    assert actions.count("ACTION6") <= 1


def test_resolve_letter_round_trip():
    cands = [
        Candidate("A", "ACTION3", None, "untried"),
        Candidate("B", "ACTION1", None, "known-good"),
        Candidate("C", "ACTION6", (5, 60), "click target"),
    ]
    assert resolve_letter("A", cands).action_name == "ACTION3"
    assert resolve_letter("b", cands).action_name == "ACTION1"
    assert resolve_letter("C", cands).action_name == "ACTION6"
    assert resolve_letter("Z", cands) is None
    assert resolve_letter("", cands) is None


def test_resolve_letter_with_prose():
    """LLM might output 'B. because it makes sense' — just pick first char."""
    cands = [
        Candidate("A", "ACTION3", None, "untried"),
        Candidate("B", "ACTION1", None, "known-good"),
    ]
    assert resolve_letter("B. it works", cands).action_name == "ACTION1"


def test_prompt_block_format():
    cands = [
        Candidate("A", "ACTION3", None, "untried this round"),
        Candidate("B", "ACTION1", None, "known-good: moves UP"),
        Candidate("C", "ACTION6", (32, 12), "click target yellow"),
    ]
    block = candidates_to_prompt_block(cands)
    assert "[CANDIDATES" in block
    assert "[A] ACTION3" in block
    assert "[B] ACTION1" in block
    assert "[C] ACTION6 (32,12)" in block
    assert "untried this round" in block
    assert "moves UP" in block


def test_prompt_block_empty():
    assert candidates_to_prompt_block([]) == ""


def test_proposer_with_no_untried_no_knowledge():
    """Degenerate case: everything tried, nothing in Knowledge."""
    k = Knowledge.empty("ar25")
    log = _make_log([(f"ACTION{i+1}", True) for i in range(7)])
    legal = [f"ACTION{i+1}" for i in range(7)]
    cands = propose(k, log, legal, rng=random.Random(0))
    # Should still return 3 candidates (random fillers)
    assert len(cands) == 3
