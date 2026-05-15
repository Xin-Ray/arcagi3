"""Tests for arc_agent.click_targets — the ACTION6 bandit module."""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from arc_agent.click_targets import (
    DEFAULT_BOOST,
    DEFAULT_DECAY,
    ClickTarget,
    render_click_targets_block,
    signature_of,
    update_click_targets,
)


# ── lightweight TrackedObject double ───────────────────────────────────


@dataclass
class _Snap:
    step: int
    color_name: str
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]


@dataclass
class _Tracked:
    uid: str
    history: list[_Snap] = field(default_factory=list)


def _alive(uid: str, color: str, bbox: tuple[int, int, int, int],
           center: tuple[float, float]) -> _Tracked:
    return _Tracked(uid=uid, history=[_Snap(0, color, bbox, center)])


# ── signature_of ───────────────────────────────────────────────────────


def test_signature_combines_color_and_shape() -> None:
    assert signature_of("cyan", (10, 20, 10, 20)) == "cyan_1x1"
    assert signature_of("red", (3, 5, 4, 7)) == "red_2x3"


def test_signature_is_independent_of_position() -> None:
    """Two cyan 1x1 objects at different positions share signature."""
    assert signature_of("cyan", (0, 0, 0, 0)) == signature_of("cyan", (40, 40, 40, 40))


# ── ClickTarget serialization ──────────────────────────────────────────


def test_click_target_roundtrip() -> None:
    t = ClickTarget(
        obj_id="obj_005", signature="red_2x2",
        coords=(12, 30), color_name="red", bbox=(11, 29, 12, 30),
        confidence=0.42, tries=3, successes=1, last_seen_step=7, alive=True,
    )
    t2 = ClickTarget.from_dict(t.to_dict())
    assert t2 == t


def test_click_target_priority_untried_wins_over_decayed() -> None:
    """Fresh untried target outranks heavily-decayed one regardless of confidence."""
    fresh = ClickTarget(
        obj_id="a", signature="red_1x1", coords=(0, 0), color_name="red",
        bbox=(0, 0, 0, 0), confidence=1.0, tries=0,
    )
    decayed = ClickTarget(
        obj_id="b", signature="blue_1x1", coords=(0, 0), color_name="blue",
        bbox=(0, 0, 0, 0), confidence=0.9, tries=8,
    )
    assert fresh.priority > decayed.priority


# ── update_click_targets: alive set reconciliation ─────────────────────


def test_update_adds_new_obj_with_fresh_confidence() -> None:
    obj = _alive("obj_001", "cyan", (10, 20, 10, 20), (10.0, 20.0))
    out = update_click_targets(
        [], [obj],
        last_action="ACTION1", last_coords=None,
        frame_changed=True, step=0,
    )
    assert len(out) == 1
    assert out[0].obj_id == "obj_001"
    assert out[0].confidence == 1.0
    assert out[0].tries == 0


def test_update_preserves_history_by_uid() -> None:
    """When the same obj_id appears next step, its confidence carries forward."""
    prior = [ClickTarget(
        obj_id="obj_001", signature="cyan_1x1", coords=(10, 20),
        color_name="cyan", bbox=(10, 20, 10, 20),
        confidence=0.49, tries=3, successes=0, last_seen_step=5,
    )]
    obj = _alive("obj_001", "cyan", (10, 20, 10, 20), (10.0, 20.0))
    out = update_click_targets(
        prior, [obj],
        last_action="ACTION1", last_coords=None,
        frame_changed=False, step=6,
    )
    assert out[0].confidence == pytest.approx(0.49)
    assert out[0].tries == 3
    assert out[0].last_seen_step == 6


def test_update_revives_by_signature_across_uid_change() -> None:
    """Round 1 had obj_001 cyan_1x1 with conf=0.3, tries=4. Round 2 starts
    fresh ObjectMemory -> obj_009 cyan_1x1. Confidence transfers."""
    prior = [ClickTarget(
        obj_id="obj_001", signature="cyan_1x1", coords=(10, 20),
        color_name="cyan", bbox=(10, 20, 10, 20),
        confidence=0.30, tries=4, successes=0, last_seen_step=12,
    )]
    new_obj = _alive("obj_009", "cyan", (10, 20, 10, 20), (10.0, 20.0))
    out = update_click_targets(
        prior, [new_obj],
        last_action="ACTION1", last_coords=None,
        frame_changed=False, step=0,
    )
    assert out[0].obj_id == "obj_009"   # re-linked to new uid
    assert out[0].confidence == pytest.approx(0.30)
    assert out[0].tries == 4


# ── update_click_targets: ACTION6 outcome ──────────────────────────────


def test_action6_noop_within_radius_decays_target() -> None:
    obj = _alive("obj_001", "cyan", (10, 20, 10, 20), (10.0, 20.0))
    out = update_click_targets(
        [], [obj],
        last_action="ACTION6", last_coords=(10, 20),  # exact hit
        frame_changed=False, step=0,
    )
    assert out[0].tries == 1
    assert out[0].confidence == pytest.approx(DEFAULT_DECAY)


def test_action6_success_boosts_target() -> None:
    obj = _alive("obj_001", "cyan", (10, 20, 10, 20), (10.0, 20.0))
    prior = [ClickTarget(
        obj_id="obj_001", signature="cyan_1x1", coords=(10, 20),
        color_name="cyan", bbox=(10, 20, 10, 20),
        confidence=0.4, tries=2,
    )]
    out = update_click_targets(
        prior, [obj],
        last_action="ACTION6", last_coords=(10, 20),
        frame_changed=True, step=1,
    )
    assert out[0].successes == 1
    assert out[0].confidence == pytest.approx(min(1.0, 0.4 * DEFAULT_BOOST))


def test_action6_success_caps_at_1() -> None:
    """Boost x2.0 starting from 0.8 must NOT exceed 1.0."""
    obj = _alive("obj_001", "cyan", (10, 20, 10, 20), (10.0, 20.0))
    prior = [ClickTarget(
        obj_id="obj_001", signature="cyan_1x1", coords=(10, 20),
        color_name="cyan", bbox=(10, 20, 10, 20),
        confidence=0.8, tries=1,
    )]
    out = update_click_targets(
        prior, [obj],
        last_action="ACTION6", last_coords=(10, 20),
        frame_changed=True, step=2,
    )
    assert out[0].confidence == pytest.approx(1.0)


def test_action6_outside_radius_is_wild_click_no_update() -> None:
    """Click far from any tracked object -> ignored entirely."""
    obj = _alive("obj_001", "cyan", (10, 20, 10, 20), (10.0, 20.0))
    out = update_click_targets(
        [], [obj],
        last_action="ACTION6", last_coords=(50, 50),  # far away
        frame_changed=False, step=0,
    )
    assert out[0].tries == 0
    assert out[0].confidence == 1.0


def test_action6_credits_nearest_when_multiple_targets() -> None:
    a = _alive("obj_a", "red", (5, 5, 5, 5), (5.0, 5.0))
    b = _alive("obj_b", "blue", (40, 40, 40, 40), (40.0, 40.0))
    out = update_click_targets(
        [], [a, b],
        last_action="ACTION6", last_coords=(6, 5),  # nearer to obj_a
        frame_changed=False, step=0,
    )
    by_uid = {t.obj_id: t for t in out}
    assert by_uid["obj_a"].tries == 1
    assert by_uid["obj_b"].tries == 0


def test_non_action6_step_does_not_touch_confidence() -> None:
    prior = [ClickTarget(
        obj_id="obj_001", signature="cyan_1x1", coords=(10, 20),
        color_name="cyan", bbox=(10, 20, 10, 20),
        confidence=0.55, tries=2,
    )]
    obj = _alive("obj_001", "cyan", (10, 20, 10, 20), (10.0, 20.0))
    out = update_click_targets(
        prior, [obj],
        last_action="ACTION1", last_coords=None,
        frame_changed=True, step=3,
    )
    assert out[0].confidence == pytest.approx(0.55)
    assert out[0].tries == 2


# ── cap + priority sort ───────────────────────────────────────────────


def test_cap_keeps_highest_priority() -> None:
    """15 alive objects, cap=10 -> keep top 10 by priority."""
    objs = [_alive(f"obj_{i:03d}", "red", (i, 0, i, 0), (float(i), 0.0))
            for i in range(15)]
    prior = []
    # Pre-decay obj_010..obj_014 so they're below the fresh ones
    for i in range(10, 15):
        prior.append(ClickTarget(
            obj_id=f"obj_{i:03d}", signature=f"red_1x1",
            coords=(i, 0), color_name="red", bbox=(i, 0, i, 0),
            confidence=0.05, tries=10,
        ))
    out = update_click_targets(
        prior, objs,
        last_action="ACTION1", last_coords=None,
        frame_changed=False, step=1,
        max_targets=10,
    )
    assert len(out) == 10
    uids = {t.obj_id for t in out}
    # Heavily decayed should be evicted
    for evicted in ("obj_010", "obj_011", "obj_012", "obj_013", "obj_014"):
        assert evicted not in uids


# ── render_click_targets_block ─────────────────────────────────────────


def test_render_empty_returns_empty_string() -> None:
    assert render_click_targets_block([]) == ""


def test_render_includes_block_header_and_each_target() -> None:
    targets = [ClickTarget(
        obj_id="obj_005", signature="cyan_1x1", coords=(12, 30),
        color_name="cyan", bbox=(12, 30, 12, 30),
        confidence=1.0, tries=0,
    )]
    out = render_click_targets_block(targets)
    assert "[CLICK TARGETS" in out
    assert "obj_005" in out
    assert "cyan" in out
    assert "(12,30)" in out
    assert "UNTRIED" in out   # because tries=0


def test_render_marks_written_off_when_low_confidence() -> None:
    targets = [ClickTarget(
        obj_id="obj_009", signature="yellow_1x1", coords=(5, 60),
        color_name="yellow", bbox=(5, 60, 5, 60),
        confidence=0.05, tries=10,
    )]
    out = render_click_targets_block(targets)
    assert "WRITTEN OFF" in out


def test_render_marks_known_interactive_after_success() -> None:
    targets = [ClickTarget(
        obj_id="obj_002", signature="red_2x2", coords=(20, 20),
        color_name="red", bbox=(20, 20, 21, 21),
        confidence=0.8, tries=2, successes=1,
    )]
    out = render_click_targets_block(targets)
    assert "known interactive" in out


def test_render_instructs_not_to_invent_coords() -> None:
    targets = [ClickTarget(
        obj_id="obj_001", signature="red_1x1", coords=(0, 0),
        color_name="red", bbox=(0, 0, 0, 0),
    )]
    out = render_click_targets_block(targets)
    low = out.lower()
    assert "do not invent" in low or "do not invent x,y" in low


def test_render_suggests_other_action_when_all_low_confidence() -> None:
    targets = [ClickTarget(
        obj_id="obj_001", signature="red_1x1", coords=(0, 0),
        color_name="red", bbox=(0, 0, 0, 0),
        confidence=0.05, tries=10,
    )]
    out = render_click_targets_block(targets)
    assert "different ACTION" in out or "not the right tool" in out.lower()
