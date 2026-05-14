"""Unit tests for arc_agent.action_mask (R2 hard rule)."""
from __future__ import annotations

import random
from typing import Optional

import pytest

from arc_agent.action_inference import OutcomeLog, StepOutcome
from arc_agent.action_mask import (
    DEFAULT_NO_OP_THRESHOLD,
    apply_action_mask,
    compute_action_mask,
)
from arc_agent.knowledge import Knowledge


# ── helpers ───────────────────────────────────────────────────────────────


def _log_with(action: str, n: int, *, changed: bool = False) -> OutcomeLog:
    log = OutcomeLog()
    for i in range(n):
        log.record(StepOutcome(
            step=i, action=action, legal=True,
            frame_changed=changed, n_active_changed=0,
        ))
    return log


def _mixed_log(specs: list[tuple[str, int, bool]]) -> OutcomeLog:
    """Build an OutcomeLog from [(action, n, all_changed)] tuples."""
    log = OutcomeLog()
    step = 0
    for (action, n, changed) in specs:
        for _ in range(n):
            log.record(StepOutcome(
                step=step, action=action, legal=True,
                frame_changed=changed, n_active_changed=0,
            ))
            step += 1
    return log


# ── compute_action_mask -- empirical rule ─────────────────────────────────


def test_mask_empty_when_no_evidence() -> None:
    log = OutcomeLog()
    k = Knowledge.empty("ar25")
    mask = compute_action_mask(log, k, ["ACTION1", "ACTION2"])
    assert mask == set()


def test_mask_includes_action_with_5_noops() -> None:
    log = _log_with("ACTION6", 5, changed=False)
    k = Knowledge.empty("ar25")
    mask = compute_action_mask(log, k, ["ACTION1", "ACTION6"])
    assert "ACTION6" in mask
    assert "ACTION1" not in mask


def test_mask_excludes_action_with_some_changes() -> None:
    """Threshold is 5 all-no-op. Even 4 no-op + 1 change should NOT mask."""
    log = _mixed_log([("ACTION1", 4, False), ("ACTION1", 1, True)])
    k = Knowledge.empty("ar25")
    mask = compute_action_mask(log, k, ["ACTION1"])
    assert mask == set()


def test_mask_under_threshold_not_blocked() -> None:
    log = _log_with("ACTION6", 4, changed=False)
    mask = compute_action_mask(log, Knowledge.empty("ar25"), ["ACTION1", "ACTION6"])
    assert mask == set()


def test_mask_custom_threshold() -> None:
    log = _log_with("ACTION6", 3, changed=False)
    mask = compute_action_mask(
        log, Knowledge.empty("ar25"), ["ACTION1", "ACTION6"],
        no_op_threshold=3,
    )
    assert "ACTION6" in mask


# ── compute_action_mask -- Knowledge rule ─────────────────────────────────


def test_mask_includes_action_flagged_in_knowledge_rules() -> None:
    log = OutcomeLog()  # no empirical evidence
    k = Knowledge.empty("ar25")
    k.rules = ["ACTION6 has no effect on any tested coord."]
    mask = compute_action_mask(log, k, ["ACTION1", "ACTION6"])
    assert "ACTION6" in mask
    assert "ACTION1" not in mask


def test_mask_includes_action_flagged_in_failed_strategies() -> None:
    log = OutcomeLog()
    k = Knowledge.empty("ar25")
    k.failed_strategies = ["ACTION7 anywhere in the right half."]
    mask = compute_action_mask(log, k, ["ACTION1", "ACTION7"])
    assert "ACTION7" in mask


def test_mask_ignores_action_mentioned_without_negation() -> None:
    """Positive mention of an action (e.g. action_semantics) must NOT mask."""
    log = OutcomeLog()
    k = Knowledge.empty("ar25")
    k.rules = ["ACTION1 moves the player UP by 3 cells"]
    mask = compute_action_mask(log, k, ["ACTION1", "ACTION6"])
    assert "ACTION1" not in mask
    assert "ACTION6" not in mask


# ── safety: never mask all legal ──────────────────────────────────────────


def test_mask_dropped_when_it_would_block_everything() -> None:
    """If every legal action has 5+ all-no-ops, the mask is dropped so the
    agent can still produce SOMETHING."""
    log = _mixed_log([("ACTION1", 5, False), ("ACTION6", 5, False)])
    mask = compute_action_mask(log, Knowledge.empty("ar25"), ["ACTION1", "ACTION6"])
    assert mask == set()


def test_mask_dropped_when_knowledge_flags_all_legal() -> None:
    k = Knowledge.empty("ar25")
    k.rules = [
        "ACTION1 has no effect on any tested coord.",
        "ACTION6 has no effect on any tested coord.",
    ]
    mask = compute_action_mask(OutcomeLog(), k, ["ACTION1", "ACTION6"])
    assert mask == set()


# ── apply_action_mask: preference order ──────────────────────────────────


def _rng() -> random.Random:
    return random.Random(42)


def test_apply_mask_passthrough_when_chosen_not_masked() -> None:
    final, replaced, _ = apply_action_mask(
        "ACTION1", mask={"ACTION6"},
        legal_actions=["ACTION1", "ACTION6"],
        outcome_log=OutcomeLog(),
        knowledge=Knowledge.empty("ar25"),
        rng=_rng(),
    )
    assert final == "ACTION1"
    assert replaced is False


def test_apply_mask_replaces_with_untried_first() -> None:
    """ACTION6 is masked; ACTION1 has been tried, ACTION7 hasn't.
    Replacement should prefer ACTION7 (untried)."""
    log = _log_with("ACTION1", 1, changed=True)   # ACTION1 tried
    final, replaced, reason = apply_action_mask(
        "ACTION6", mask={"ACTION6"},
        legal_actions=["ACTION1", "ACTION6", "ACTION7"],
        outcome_log=log,
        knowledge=Knowledge.empty("ar25"),
        rng=_rng(),
    )
    assert replaced is True
    assert final == "ACTION7"
    assert "untried" in reason


def test_apply_mask_falls_back_to_positive_action_semantics() -> None:
    """All legal actions tried; ACTION1 has positive semantics.
    Replacement picks ACTION1."""
    log = _mixed_log([("ACTION1", 2, True), ("ACTION6", 5, False)])
    k = Knowledge.empty("ar25")
    k.action_semantics = {"ACTION1": "moves the player UP 3 cells"}
    final, replaced, reason = apply_action_mask(
        "ACTION6", mask={"ACTION6"},
        legal_actions=["ACTION1", "ACTION6"],
        outcome_log=log,
        knowledge=k,
        rng=_rng(),
    )
    assert replaced is True
    assert final == "ACTION1"
    assert "known-good" in reason


def test_apply_mask_skips_negative_action_semantics() -> None:
    """An action_semantics entry containing 'no effect' should NOT be used
    as a positive replacement."""
    log = _mixed_log([("ACTION1", 2, False), ("ACTION6", 5, False)])
    k = Knowledge.empty("ar25")
    k.action_semantics = {"ACTION1": "ACTION1 has no effect"}
    final, replaced, _ = apply_action_mask(
        "ACTION6", mask={"ACTION6"},
        legal_actions=["ACTION1", "ACTION6"],
        outcome_log=log,
        knowledge=k,
        rng=_rng(),
    )
    # Should fall through to "random non-blocked" -> ACTION1 (only non-masked left)
    assert final == "ACTION1"
    assert replaced is True


def test_apply_mask_falls_back_to_random_when_nothing_known() -> None:
    """All legal tried, no positive semantics -> random non-blocked."""
    log = _mixed_log([("ACTION1", 2, True), ("ACTION6", 5, False)])
    final, replaced, reason = apply_action_mask(
        "ACTION6", mask={"ACTION6"},
        legal_actions=["ACTION1", "ACTION6"],
        outcome_log=log,
        knowledge=Knowledge.empty("ar25"),
        rng=_rng(),
    )
    assert replaced is True
    assert final == "ACTION1"


def test_apply_mask_empty_mask_passthrough() -> None:
    final, replaced, _ = apply_action_mask(
        "ACTION6", mask=set(),
        legal_actions=["ACTION1", "ACTION6"],
        outcome_log=OutcomeLog(),
        knowledge=Knowledge.empty("ar25"),
        rng=_rng(),
    )
    assert final == "ACTION6"
    assert replaced is False


# ── integration: real-world ar25 trace pattern ────────────────────────────


def test_ar25_trace_pattern_masks_action6_after_5_noops() -> None:
    """Reproduces the ar25 round-1 stuck pattern: ACTION6 tried 5+ times,
    all no-op; ACTION1 known to move things UP."""
    log = _mixed_log([
        ("ACTION6", 5, False),
        ("ACTION1", 2, True),
    ])
    k = Knowledge.empty("ar25")
    k.action_semantics = {"ACTION1": "moves an active object UP by 3 cells"}
    k.rules = ["ACTION6 has no effect on any tested coord."]
    k.failed_strategies = ["ACTION6 anywhere in the right half."]

    mask = compute_action_mask(log, k, ["ACTION1", "ACTION6", "ACTION7"])
    assert "ACTION6" in mask
    assert "ACTION1" not in mask   # has positive semantics

    final, replaced, _ = apply_action_mask(
        "ACTION6", mask=mask,
        legal_actions=["ACTION1", "ACTION6", "ACTION7"],
        outcome_log=log,
        knowledge=k,
        rng=_rng(),
    )
    # ACTION7 untried -> should be picked
    assert final == "ACTION7"
    assert replaced is True
