"""Tests for arc_agent.action_inference."""
from __future__ import annotations

from arc_agent.action_inference import (
    ALL_ACTIONS,
    OutcomeLog,
    StepOutcome,
    detect_collapse,
    render_action_block,
    render_history_tail,
    render_untried_block,
    summarize_action,
)


def _o(step, action, *, changed=True, direction="UP", distance=3,
       legal=True) -> StepOutcome:
    return StepOutcome(step=step, action=action, legal=legal,
                       frame_changed=changed, n_active_changed=1 if changed else 0,
                       primary_direction=direction if changed else None,
                       primary_distance=distance if changed else 0)


def test_empty_log_says_untried() -> None:
    assert summarize_action("ACTION1", []) == "(untried)"


def test_all_no_op() -> None:
    out = summarize_action("ACTION2",
                           [_o(0, "ACTION2", changed=False),
                            _o(1, "ACTION2", changed=False)])
    assert "all no-op" in out


def test_all_illegal() -> None:
    out = summarize_action("ACTION5",
                           [_o(0, "ACTION5", legal=False, changed=False)])
    assert "illegal" in out


def test_mixed_outcomes() -> None:
    outs = [
        _o(0, "ACTION1", changed=True, direction="UP", distance=3),
        _o(1, "ACTION1", changed=True, direction="UP", distance=3),
        _o(2, "ACTION1", changed=False),
    ]
    s = summarize_action("ACTION1", outs)
    assert "2x" in s and "UP" in s and "no-op" in s


def test_outcome_log_record_and_untried() -> None:
    log = OutcomeLog()
    log.record(_o(0, "ACTION1"))
    log.record(_o(1, "ACTION3"))
    assert log.n_tried("ACTION1") == 1
    assert log.n_tried("ACTION2") == 0
    untried = log.untried(list(ALL_ACTIONS))
    assert "ACTION2" in untried and "ACTION1" not in untried


def test_render_action_block_includes_untried_label() -> None:
    log = OutcomeLog()
    log.record(_o(0, "ACTION1"))
    out = render_action_block(log, ["ACTION1", "ACTION2"])
    assert "ACTION1:" in out
    assert "ACTION2: UNTRIED" in out


def test_render_untried_block() -> None:
    log = OutcomeLog()
    log.record(_o(0, "ACTION1"))
    assert "ACTION1" not in render_untried_block(log, ["ACTION1", "ACTION2"])
    assert "ACTION2" in render_untried_block(log, ["ACTION1", "ACTION2"])


def test_render_history_tail_limits_n() -> None:
    log = OutcomeLog()
    for i in range(10):
        log.record(_o(i, "ACTION1"))
    out = render_history_tail(log, n=3)
    assert out.count("step ") == 3
    assert "step 7" in out and "step 9" in out


def test_detect_collapse_three_in_a_row() -> None:
    log = OutcomeLog()
    for i in range(3):
        log.record(_o(i, "ACTION1"))
    assert detect_collapse(log, window=3) is True


def test_detect_collapse_diverse() -> None:
    log = OutcomeLog()
    log.record(_o(0, "ACTION1"))
    log.record(_o(1, "ACTION2"))
    log.record(_o(2, "ACTION1"))
    assert detect_collapse(log, window=3) is False


# ─── detect_stuck (P0-B) ───────────────────────────────────────────────────


def test_detect_stuck_collapse_first() -> None:
    """3 same in a row should fire even before other conditions."""
    from arc_agent.action_inference import detect_stuck
    log = OutcomeLog()
    for i in range(3):
        log.record(_o(i, "ACTION1"))
    stuck, reason = detect_stuck(log, frame_hashes=[1, 2, 3])
    assert stuck
    assert "ACTION1" in reason


def test_detect_stuck_no_op_streak() -> None:
    from arc_agent.action_inference import detect_stuck
    log = OutcomeLog()
    # 5 different actions but all no-op
    for i, a in enumerate(["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"]):
        log.record(_o(i, a, changed=False))
    stuck, reason = detect_stuck(log, frame_hashes=[1, 2, 3, 4, 5])
    assert stuck
    assert "no-op streak" in reason


def test_detect_stuck_state_revisit() -> None:
    from arc_agent.action_inference import detect_stuck
    log = OutcomeLog()
    log.record(_o(0, "ACTION1"))
    # Same hash 7 times -> definitely a loop
    hashes = [42] * 7
    stuck, reason = detect_stuck(log, frame_hashes=hashes)
    assert stuck
    assert "visited" in reason.lower()


def test_detect_stuck_alternating_pattern() -> None:
    from arc_agent.action_inference import detect_stuck
    log = OutcomeLog()
    for i in range(6):
        log.record(_o(i, "ACTION1" if i % 2 == 0 else "ACTION2", changed=True))
    # No-op streak 0, no collapse (3-in-a-row), no state revisit, but
    # alternating A,B,A,B,A,B for 6 steps -> should fire
    stuck, reason = detect_stuck(log, frame_hashes=list(range(6)))
    assert stuck
    assert "alternating" in reason.lower()


def test_detect_stuck_recent_no_op_rate() -> None:
    """Action tried 5 times, 4 of last 5 are no-op -> stuck."""
    from arc_agent.action_inference import detect_stuck
    log = OutcomeLog()
    # Make these spread out so no consecutive same-action collapse:
    log.record(_o(0, "ACTION1", changed=False))
    log.record(_o(1, "ACTION2", changed=True))
    log.record(_o(2, "ACTION1", changed=False))
    log.record(_o(3, "ACTION2", changed=True))
    log.record(_o(4, "ACTION1", changed=False))
    log.record(_o(5, "ACTION2", changed=True))
    log.record(_o(6, "ACTION1", changed=False))
    # Some change so no no-op streak; alternating IS the pattern.
    stuck, reason = detect_stuck(log, frame_hashes=list(range(7)))
    assert stuck   # either alternating or recent no-op rate triggers


def test_detect_stuck_clean_episode_not_stuck() -> None:
    from arc_agent.action_inference import detect_stuck
    log = OutcomeLog()
    for i, a in enumerate(["ACTION1", "ACTION3", "ACTION5", "ACTION2"]):
        log.record(_o(i, a, changed=True, direction="UP"))
    stuck, _ = detect_stuck(log, frame_hashes=[1, 2, 3, 4])
    assert not stuck


def test_recent_no_op_summary() -> None:
    from arc_agent.action_inference import recent_no_op_summary
    log = OutcomeLog()
    log.record(_o(0, "ACTION1", changed=True))
    log.record(_o(1, "ACTION1", changed=False))
    log.record(_o(2, "ACTION1", changed=False))
    s = recent_no_op_summary(log, "ACTION1", window=5)
    assert s["n_recent"] == 3
    assert s["n_no_op"] == 2
    assert abs(s["no_op_rate"] - 2/3) < 1e-6


def test_reset_empties_log() -> None:
    log = OutcomeLog()
    log.record(_o(0, "ACTION1"))
    log.reset()
    assert log.all_steps == []
    assert log.n_tried("ACTION1") == 0
