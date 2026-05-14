"""Unit tests for arc_agent.knowledge.Knowledge.

Covers:
- empty / construction
- to_dict / from_dict round-trip
- render() block content
- merged_with_delta (per-key update, append+dedup+cap, alert overwrite)
- append_round_summary cap
- tolerant parsing of garbled delta
"""
from __future__ import annotations

import json

import pytest

from arc_agent.knowledge import Knowledge


# ── construction / factory ────────────────────────────────────────────────


def test_empty_defaults() -> None:
    k = Knowledge.empty("ar25")
    assert k.game_id == "ar25"
    assert k.rounds_played == 0 and k.rounds_won == 0
    assert k.action_semantics == {}
    assert k.goal_hypothesis == ""
    assert k.goal_confidence == "low"
    assert k.rules == []
    assert k.failed_strategies == []
    assert k.round_history == []
    assert k.current_alert == ""


# ── serialization round-trip ──────────────────────────────────────────────


def test_to_dict_from_dict_roundtrip() -> None:
    k = Knowledge.empty("ar25")
    k.action_semantics = {"ACTION1": "moves red up"}
    k.goal_hypothesis = "reach the top"
    k.goal_confidence = "medium"
    k.rules = ["red moves first"]
    k.failed_strategies = ["spamming ACTION6"]
    k.round_history = ["round 0: no progress"]
    k.current_alert = "stop repeating ACTION1"
    k.rounds_played = 2
    k.rounds_won = 1

    d = k.to_dict()
    # Must be JSON-serializable
    s = json.dumps(d)
    k2 = Knowledge.from_dict(json.loads(s))
    assert k2 == k


def test_from_dict_tolerates_missing_keys() -> None:
    k = Knowledge.from_dict({"game_id": "x"})
    assert k.game_id == "x"
    assert k.action_semantics == {}
    assert k.goal_confidence == "low"


def test_from_dict_clamps_bad_confidence() -> None:
    k = Knowledge.from_dict({"goal_confidence": "ULTRA-CONFIDENT"})
    assert k.goal_confidence == "low"


# ── render() ──────────────────────────────────────────────────────────────


def test_render_empty_knowledge_mentions_unknown_goal() -> None:
    text = Knowledge.empty("ar25").render()
    assert "rounds: 0 played" in text
    assert "nothing learned yet" in text
    assert "unknown" in text.lower()


def test_render_includes_action_semantics_sorted() -> None:
    k = Knowledge.empty("ar25")
    k.action_semantics = {"ACTION3": "left", "ACTION1": "up"}
    text = k.render()
    # ACTION1 appears before ACTION3 (sorted)
    assert text.index("ACTION1") < text.index("ACTION3")
    assert "up" in text and "left" in text


def test_render_includes_goal_with_confidence() -> None:
    k = Knowledge.empty("ar25")
    k.goal_hypothesis = "reach the red dot"
    k.goal_confidence = "high"
    text = k.render()
    assert "reach the red dot" in text
    assert "high" in text


def test_render_includes_failed_strategies_warning() -> None:
    k = Knowledge.empty("ar25")
    k.failed_strategies = ["click center"]
    text = k.render()
    assert "do NOT repeat" in text
    assert "click center" in text


def test_render_caps_round_history_at_5() -> None:
    k = Knowledge.empty("ar25")
    k.round_history = [f"round {i}: nothing" for i in range(10)]
    text = k.render()
    assert "round 9: nothing" in text   # last one always present
    assert "round 4: nothing" not in text  # outside last 5


def test_render_alert_block() -> None:
    k = Knowledge.empty("ar25")
    k.current_alert = "ACTION1 has no effect — try another"
    assert "ACTION1 has no effect" in k.render_alert()


# ── merged_with_delta — happy path ────────────────────────────────────────


def test_merge_action_semantics_update_per_key_overwrite() -> None:
    k = Knowledge.empty("ar25")
    k.action_semantics = {"ACTION1": "moves up"}
    delta = {"action_semantics_update": {"ACTION1": "moves up 1 cell",
                                         "ACTION3": "moves left"}}
    k2 = k.merged_with_delta(delta)
    assert k2.action_semantics == {
        "ACTION1": "moves up 1 cell",
        "ACTION3": "moves left",
    }
    # Original is unchanged (immutability via copy)
    assert k.action_semantics == {"ACTION1": "moves up"}


def test_merge_goal_hypothesis_replaces() -> None:
    k = Knowledge.empty("ar25")
    delta = {"goal_hypothesis_update": "reach top-right",
             "goal_confidence_update": "medium"}
    k2 = k.merged_with_delta(delta)
    assert k2.goal_hypothesis == "reach top-right"
    assert k2.goal_confidence == "medium"


def test_merge_goal_confidence_invalid_keeps_prior() -> None:
    k = Knowledge.empty("ar25")
    k.goal_confidence = "medium"
    k2 = k.merged_with_delta({"goal_confidence_update": "OVER-9000"})
    assert k2.goal_confidence == "medium"


def test_merge_rules_dedup_and_cap() -> None:
    k = Knowledge.empty("ar25")
    # First merge: 5 rules
    k = k.merged_with_delta(
        {"rules_append": [f"rule {i}" for i in range(5)]}
    )
    assert len(k.rules) == 5
    # Second merge: 3 dupes + 8 new — should cap to 10, keeping newest
    k = k.merged_with_delta(
        {"rules_append": ["rule 0", "rule 1", "rule 2"]
                          + [f"new {i}" for i in range(8)]}
    )
    assert len(k.rules) == 10
    assert "new 7" in k.rules         # newest survives
    assert "rule 0" not in k.rules    # oldest evicted


def test_merge_failed_strategies_cap_5() -> None:
    k = Knowledge.empty("ar25")
    k = k.merged_with_delta(
        {"failed_strategies_append": [f"s{i}" for i in range(8)]}
    )
    assert len(k.failed_strategies) == 5
    assert "s7" in k.failed_strategies   # newest


def test_merge_current_alert_overwrites_when_key_present() -> None:
    k = Knowledge.empty("ar25")
    k.current_alert = "old alert"
    k2 = k.merged_with_delta({"current_alert": "new alert"})
    assert k2.current_alert == "new alert"


def test_merge_current_alert_cleared_with_empty_string() -> None:
    k = Knowledge.empty("ar25")
    k.current_alert = "old alert"
    k2 = k.merged_with_delta({"current_alert": ""})
    assert k2.current_alert == ""


def test_merge_current_alert_kept_when_key_absent() -> None:
    k = Knowledge.empty("ar25")
    k.current_alert = "old alert"
    k2 = k.merged_with_delta({})
    assert k2.current_alert == "old alert"


# ── merged_with_delta — defensive parsing ─────────────────────────────────


def test_merge_garbled_delta_keeps_knowledge() -> None:
    k = Knowledge.empty("ar25")
    k.action_semantics = {"ACTION1": "kept"}
    for bad in [None, "not a dict", 42, [], False]:
        k2 = k.merged_with_delta(bad)  # type: ignore[arg-type]
        assert k2.action_semantics == {"ACTION1": "kept"}


def test_merge_drops_bad_action_semantics_entries() -> None:
    k = Knowledge.empty("ar25")
    delta = {"action_semantics_update": {
        "ACTION1": "good",
        123: "bad-key",           # non-str key
        "ACTION2": None,          # None value
    }}
    k2 = k.merged_with_delta(delta)
    assert k2.action_semantics == {"ACTION1": "good"}


def test_merge_empty_strings_in_rules_dropped() -> None:
    k = Knowledge.empty("ar25")
    k2 = k.merged_with_delta(
        {"rules_append": ["", "   ", "real rule", None]}
    )
    assert k2.rules == ["real rule"]


def test_merge_does_not_mutate_input_delta() -> None:
    k = Knowledge.empty("ar25")
    delta = {"action_semantics_update": {"ACTION1": "moves"}}
    snapshot = dict(delta)
    k.merged_with_delta(delta)
    assert delta == snapshot


# ── append_round_summary ──────────────────────────────────────────────────


def test_append_round_summary_caps_history() -> None:
    k = Knowledge.empty("ar25")
    for i in range(25):
        k.append_round_summary(f"round {i}: meh")
    assert len(k.round_history) == 20
    assert k.round_history[-1] == "round 24: meh"


def test_append_round_summary_skips_blank() -> None:
    k = Knowledge.empty("ar25")
    k.append_round_summary("")
    k.append_round_summary("   ")
    assert k.round_history == []
