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
    k.action_semantics = {"ACTION1": "moves the red 1x1 up"}
    # BUG-9: positive-motion values MUST name a subject (color / obj_id /
    # shape word like "1x1"/"square") or they get dropped.
    delta = {"action_semantics_update": {"ACTION1": "moves the red 1x1 up 1 cell",
                                         "ACTION3": "moves the yellow square left"}}
    k2 = k.merged_with_delta(delta)
    assert k2.action_semantics == {
        "ACTION1": "moves the red 1x1 up 1 cell",
        "ACTION3": "moves the yellow square left",
    }
    # Original is unchanged (immutability via copy)
    assert k.action_semantics == {"ACTION1": "moves the red 1x1 up"}


def test_merge_goal_hypothesis_replaces() -> None:
    k = Knowledge.empty("ar25")
    delta = {"goal_hypothesis_update": "reach top-right",
             "goal_confidence_update": "medium"}
    k2 = k.merged_with_delta(delta)
    assert k2.goal_hypothesis == "reach top-right"
    assert k2.goal_confidence == "medium"


# ── R1: goal sentinel filter ──────────────────────────────────────────────


def test_merge_rejects_unknown_string_as_goal() -> None:
    """Reflection sometimes writes the literal string 'unknown' despite the
    SYSTEM prompt telling it to use null. Knowledge must filter it out."""
    k = Knowledge.empty("ar25")
    k.goal_hypothesis = "reach top"   # something real already there
    delta = {"goal_hypothesis_update": "unknown"}
    k2 = k.merged_with_delta(delta)
    assert k2.goal_hypothesis == "reach top"  # unchanged


def test_merge_rejects_various_sentinels() -> None:
    """All known sentinel strings get filtered."""
    k = Knowledge.empty("ar25")
    k.goal_hypothesis = "stable hypothesis"
    for sentinel in ("unknown", "None", "N/A", "n/a", "TBD", "exploring",
                     "?", "", "  unknown  ", "no idea", "uncertain"):
        k2 = k.merged_with_delta({"goal_hypothesis_update": sentinel})
        assert k2.goal_hypothesis == "stable hypothesis", (
            f"sentinel {sentinel!r} leaked through")


def test_merge_accepts_real_hypothesis() -> None:
    """Non-sentinel strings should still be accepted."""
    k = Knowledge.empty("ar25")
    k2 = k.merged_with_delta(
        {"goal_hypothesis_update": "reach the red dot at top"}
    )
    assert k2.goal_hypothesis == "reach the red dot at top"


def test_merge_filters_sentinel_when_no_prior_hypothesis() -> None:
    """Empty -> 'unknown' update should leave it empty (not become 'unknown')."""
    k = Knowledge.empty("ar25")
    assert k.goal_hypothesis == ""
    k2 = k.merged_with_delta({"goal_hypothesis_update": "unknown"})
    assert k2.goal_hypothesis == ""


# ── R5: failed_strategies cross-pollution filter ─────────────────────────


def test_R5_rejects_goal_matching_existing_failed_strategy() -> None:
    """Reproduces the v2 ar25 bug: Reflection wrote
    `goal_hypothesis_update: "ACTION6 anywhere in the right half."`
    while that exact string was already in failed_strategies. R5 drops it."""
    k = Knowledge.empty("ar25")
    k.failed_strategies = ["ACTION6 anywhere in the right half."]
    k.goal_hypothesis = "valid prior goal"

    delta = {
        "goal_hypothesis_update": "ACTION6 anywhere in the right half.",
    }
    k2 = k.merged_with_delta(delta)
    assert k2.goal_hypothesis == "valid prior goal", (
        f"R5 should reject cross-pollution; got {k2.goal_hypothesis!r}"
    )


def test_R5_rejects_goal_matching_appended_failed_strategy() -> None:
    """When the same delta both APPENDS a failed_strategies entry AND
    writes that same string into goal_hypothesis_update -- reject the
    goal write but still accept the failed_strategies append."""
    k = Knowledge.empty("ar25")
    k.goal_hypothesis = "prior goal"

    delta = {
        "goal_hypothesis_update": "ACTION6 anywhere in the right half.",
        "failed_strategies_append": ["ACTION6 anywhere in the right half."],
    }
    k2 = k.merged_with_delta(delta)

    # goal_hypothesis stays as prior; failed_strategies gets appended
    assert k2.goal_hypothesis == "prior goal"
    assert "ACTION6 anywhere in the right half." in k2.failed_strategies


def test_R5_case_insensitive_match() -> None:
    """Cross-pollution check is case-insensitive."""
    k = Knowledge.empty("ar25")
    k.failed_strategies = ["acTion6 ANYWHERE in the right half."]
    k.goal_hypothesis = "prior"

    delta = {"goal_hypothesis_update": "ACTION6 anywhere in the RIGHT HALF."}
    k2 = k.merged_with_delta(delta)
    assert k2.goal_hypothesis == "prior"


def test_R5_whitespace_tolerant() -> None:
    """Leading/trailing whitespace should not let a copy through."""
    k = Knowledge.empty("ar25")
    k.failed_strategies = ["ACTION6 anywhere in the right half."]
    k.goal_hypothesis = "prior"

    delta = {"goal_hypothesis_update": "  ACTION6 anywhere in the right half.  "}
    k2 = k.merged_with_delta(delta)
    assert k2.goal_hypothesis == "prior"


def test_R5_legitimate_goal_passes_through() -> None:
    """Goals that are NOT in failed_strategies must still be accepted --
    R5 should have zero false positives on real hypotheses."""
    k = Knowledge.empty("ar25")
    k.failed_strategies = ["ACTION6 anywhere in the right half."]

    delta = {"goal_hypothesis_update": "reach the red dot at the top of the grid"}
    k2 = k.merged_with_delta(delta)
    assert k2.goal_hypothesis == "reach the red dot at the top of the grid"


def test_R5_legitimate_goal_with_action_word_passes() -> None:
    """A goal mentioning an action by name (but not matching a failed
    strategy) should pass."""
    k = Knowledge.empty("ar25")
    k.failed_strategies = ["clicking center never advances"]

    delta = {
        "goal_hypothesis_update": "use ACTION1 to move the avatar to the top edge"
    }
    k2 = k.merged_with_delta(delta)
    assert k2.goal_hypothesis == "use ACTION1 to move the avatar to the top edge"


def test_R5_does_not_block_failed_append() -> None:
    """R5 must NOT break the failed_strategies append path -- it should
    only filter the goal write."""
    k = Knowledge.empty("ar25")
    delta = {
        "goal_hypothesis_update": "new bad goal",
        "failed_strategies_append": ["new bad goal"],
    }
    k2 = k.merged_with_delta(delta)
    assert "new bad goal" in k2.failed_strategies   # append succeeded
    assert k2.goal_hypothesis == ""   # but goal write was blocked


# ── R6: action-described goal filter ─────────────────────────────────────


def test_R6_rejects_goal_starting_with_action() -> None:
    """Observed in v3 ar25: Reflection wrote 'ACTION6 should move object
    obj_002 left by 3 cells' into goal_hypothesis_update. That's an
    action description, not a target state. R6 drops it."""
    k = Knowledge.empty("ar25")
    k.goal_hypothesis = "reach the top edge"

    delta = {"goal_hypothesis_update": "ACTION6 should move object obj_002 left by 3 cells"}
    k2 = k.merged_with_delta(delta)
    assert k2.goal_hypothesis == "reach the top edge"


def test_R6_rejects_lowercase_action_prefix() -> None:
    k = Knowledge.empty("ar25")
    k.goal_hypothesis = "valid prior"
    delta = {"goal_hypothesis_update": "action4 should move LEFT by 3 cells"}
    k2 = k.merged_with_delta(delta)
    assert k2.goal_hypothesis == "valid prior"


def test_R6_rejects_should_move_anywhere() -> None:
    """Even when the goal doesn't start with 'ACTION', 'should move/advance'
    is a strong signal of action-described intent."""
    k = Knowledge.empty("ar25")
    k.goal_hypothesis = "valid prior"
    delta = {"goal_hypothesis_update": "the player should move to the top-right"}
    k2 = k.merged_with_delta(delta)
    # 'should move' caught by pattern
    assert k2.goal_hypothesis == "valid prior"


def test_R6_accepts_state_described_goal() -> None:
    """Real goals describe target states -- they must pass."""
    k = Knowledge.empty("ar25")
    for good in [
        "reach the top edge of the grid",
        "match every red dot with a red target",
        "all yellow objects in the same row",
        "blue object at position (32, 32)",
    ]:
        k2 = k.merged_with_delta({"goal_hypothesis_update": good})
        assert k2.goal_hypothesis == good, f"R6 false positive on {good!r}"


# ── R4: drop rules contradicting positive action_semantics ──────────────


def test_R4_drops_negative_rule_when_action_has_positive_semantic() -> None:
    """Reproduces the smoke run bug: Reflection wrote 'ACTION1 has no
    effect' AFTER it had already confirmed 'ACTION1 moves UP by 3 cells'
    in action_semantics. R4 drops the contradicting rule."""
    k = Knowledge.empty("ar25")
    k.action_semantics["ACTION1"] = "moves an active object UP by 3 cells"

    delta = {"rules_append": ["ACTION1 has no effect on any tested coord."]}
    k2 = k.merged_with_delta(delta)
    assert "ACTION1 has no effect on any tested coord." not in k2.rules


def test_R4_drops_negative_failed_strategy_with_positive_semantic() -> None:
    """Same filter for failed_strategies_append."""
    k = Knowledge.empty("ar25")
    k.action_semantics["ACTION3"] = "moves an active object RIGHT by 2 cells"

    delta = {"failed_strategies_append": ["ACTION3 has no effect at any tested coord"]}
    k2 = k.merged_with_delta(delta)
    assert all("ACTION3" not in s.upper() or "no effect" not in s.lower()
               for s in k2.failed_strategies)


def test_R4_keeps_rule_when_no_positive_semantic_exists() -> None:
    """If action_semantics doesn't have ACTION_X, the negative rule is
    legitimate (no contradiction to detect)."""
    k = Knowledge.empty("ar25")
    # No entry for ACTION6
    delta = {"rules_append": ["ACTION6 has no effect on any tested coord."]}
    k2 = k.merged_with_delta(delta)
    assert "ACTION6 has no effect on any tested coord." in k2.rules


def test_R4_keeps_rule_about_action_unrelated_to_positive_semantic() -> None:
    """Positive semantic for ACTION1, negative rule about ACTION6 -> both kept."""
    k = Knowledge.empty("ar25")
    k.action_semantics["ACTION1"] = "moves an active object UP by 3 cells"

    delta = {"rules_append": ["ACTION6 has no effect on any tested coord."]}
    k2 = k.merged_with_delta(delta)
    assert "ACTION6 has no effect on any tested coord." in k2.rules


def test_R4_keeps_non_negative_rule_even_when_action_mentioned() -> None:
    """A rule that mentions ACTION1 but isn't negative (e.g. observation,
    co-occurrence pattern) should pass through."""
    k = Knowledge.empty("ar25")
    k.action_semantics["ACTION1"] = "moves an active object UP by 3 cells"

    delta = {"rules_append": ["ACTION1 followed by ACTION2 always reaches the top"]}
    k2 = k.merged_with_delta(delta)
    assert "ACTION1 followed by ACTION2 always reaches the top" in k2.rules


def test_R4_drops_only_when_semantic_is_truly_positive() -> None:
    """If action_semantics happens to say 'ACTION_X: no effect at (5,5)',
    that's not POSITIVE, so a 'no effect' rule should pass."""
    k = Knowledge.empty("ar25")
    k.action_semantics["ACTION6"] = "no effect at (5,5)"   # negative semantic

    delta = {"rules_append": ["ACTION6 has no effect on any tested coord."]}
    k2 = k.merged_with_delta(delta)
    assert "ACTION6 has no effect on any tested coord." in k2.rules


def test_R4_handles_same_delta_with_action_semantics_and_contradiction() -> None:
    """Same delta sets ACTION1 positive AND tries to add 'ACTION1 has no
    effect' rule. The positive sem applies first; the rule is then dropped."""
    k = Knowledge.empty("ar25")

    delta = {
        # BUG-9: must name a subject ('red 1x1') for positive-motion entry
        "action_semantics_update": {"ACTION1": "moves the red 1x1 UP 3 cells"},
        "rules_append": ["ACTION1 has no effect"],
    }
    k2 = k.merged_with_delta(delta)
    assert k2.action_semantics["ACTION1"] == "moves the red 1x1 UP 3 cells"
    assert "ACTION1 has no effect" not in k2.rules


def test_R6_does_not_block_action_mention_in_middle() -> None:
    """A goal that mentions an action name in the middle but isn't an
    action description should pass (e.g. describing what won't be needed)."""
    k = Knowledge.empty("ar25")
    delta = {
        "goal_hypothesis_update": "match colors without using ACTION6",
    }
    k2 = k.merged_with_delta(delta)
    assert k2.goal_hypothesis == "match colors without using ACTION6"


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


# ── BUG-8: rejected_goals (negative memory across overwrites) ─────────────


def test_BUG8_overwriting_goal_pushes_old_to_rejected() -> None:
    """When Reflection writes a new goal_hypothesis_update that differs
    from the existing non-empty hypothesis, the old one must be archived
    into rejected_goals — otherwise the agent re-proposes wrong goals
    forever (observed in ar25 2x80 trace: step 0 -> step 5 -> step 61
    each silently overwrote the previous hypothesis)."""
    k = Knowledge.empty("ar25")
    k.goal_hypothesis = "move the active object to the edge"

    k2 = k.merged_with_delta({"goal_hypothesis_update": "reshape objects"})

    assert k2.goal_hypothesis == "reshape objects"
    assert "move the active object to the edge" in k2.rejected_goals


def test_BUG8_rejected_goal_blocks_re_proposal() -> None:
    """Once a goal is in rejected_goals, a later delta that proposes the
    same string must be DROPPED (not re-installed as the active goal)."""
    k = Knowledge.empty("ar25")
    k.rejected_goals = ["reach the top edge"]
    k.goal_hypothesis = "match red dots"

    k2 = k.merged_with_delta({"goal_hypothesis_update": "reach the top edge"})

    assert k2.goal_hypothesis == "match red dots"   # not overwritten
    # And the rejected_goals list is not double-counted
    assert k2.rejected_goals == ["reach the top edge"]


def test_BUG8_rejected_match_is_case_insensitive() -> None:
    k = Knowledge.empty("ar25")
    k.rejected_goals = ["Reach The Top Edge"]
    k.goal_hypothesis = "valid prior"

    k2 = k.merged_with_delta({"goal_hypothesis_update": "reach the top edge"})

    assert k2.goal_hypothesis == "valid prior"


def test_BUG8_same_value_overwrite_does_not_archive() -> None:
    """If a delta repeats the existing hypothesis verbatim (very common —
    Reflection often re-confirms), rejected_goals should NOT grow."""
    k = Knowledge.empty("ar25")
    k.goal_hypothesis = "reach the top edge"

    k2 = k.merged_with_delta({"goal_hypothesis_update": "reach the top edge"})

    assert k2.goal_hypothesis == "reach the top edge"
    assert k2.rejected_goals == []


def test_BUG8_first_goal_does_not_create_rejected_entry() -> None:
    """Writing into an empty goal_hypothesis must not push the empty
    string into rejected_goals."""
    k = Knowledge.empty("ar25")

    k2 = k.merged_with_delta({"goal_hypothesis_update": "reach the top edge"})

    assert k2.goal_hypothesis == "reach the top edge"
    assert k2.rejected_goals == []


def test_BUG8_rejected_goals_capped_at_10() -> None:
    """rejected_goals must not grow unbounded across many overwrites."""
    k = Knowledge.empty("ar25")
    for i in range(15):
        k = k.merged_with_delta({"goal_hypothesis_update": f"goal {i:02d}"})
    assert len(k.rejected_goals) == 10
    # Newest evicted -> only the last few archived; current goal is the latest
    assert k.goal_hypothesis == "goal 14"


def test_BUG8_render_includes_rejected_goals() -> None:
    """Reflection must SEE the rejected list when it builds the next delta —
    so render() has to include it."""
    k = Knowledge.empty("ar25")
    k.rejected_goals = ["reach the bottom", "reshape objects"]

    text = k.render()
    assert "rejected_goals" in text
    assert "reach the bottom" in text
    assert "reshape objects" in text


def test_BUG8_rejected_goals_roundtrip_json() -> None:
    """from_dict/to_dict must preserve rejected_goals across persistence."""
    k = Knowledge.empty("ar25")
    k.rejected_goals = ["wrong A", "wrong B"]
    k.goal_hypothesis = "current"

    s = json.dumps(k.to_dict())
    k2 = Knowledge.from_dict(json.loads(s))
    assert k2.rejected_goals == ["wrong A", "wrong B"]


# ── BUG-9: action_semantics must name a subject ──────────────────────────


def test_BUG9_drops_movement_claim_without_subject() -> None:
    """'moves an active object UP by 3' is uninformative when multiple
    actives exist. Drop it — keep whatever was there before."""
    k = Knowledge.empty("ar25")
    k.action_semantics = {"ACTION1": "moves the red 1x1 UP by 3"}

    bad = {"action_semantics_update": {"ACTION1": "moves an active object UP by 3 cells"}}
    k2 = k.merged_with_delta(bad)

    assert k2.action_semantics == {"ACTION1": "moves the red 1x1 UP by 3"}


def test_BUG9_accepts_subject_by_color() -> None:
    k = Knowledge.empty("ar25")
    delta = {"action_semantics_update": {"ACTION1": "moves the yellow object UP by 3"}}
    k2 = k.merged_with_delta(delta)
    assert k2.action_semantics["ACTION1"] == "moves the yellow object UP by 3"


def test_BUG9_accepts_subject_by_obj_id() -> None:
    k = Knowledge.empty("ar25")
    delta = {"action_semantics_update": {"ACTION1": "moves obj_002 UP by 3 cells"}}
    k2 = k.merged_with_delta(delta)
    assert "obj_002" in k2.action_semantics["ACTION1"]


def test_BUG9_accepts_subject_by_shape_size() -> None:
    k = Knowledge.empty("ar25")
    delta = {"action_semantics_update": {"ACTION1": "moves the 2x2 block UP by 1"}}
    k2 = k.merged_with_delta(delta)
    assert "2x2" in k2.action_semantics["ACTION1"]


def test_BUG9_does_not_filter_negative_claims() -> None:
    """'no observable effect' is a valid no-op semantic and must pass even
    without a subject — there's no object being claimed to move."""
    k = Knowledge.empty("ar25")
    delta = {"action_semantics_update": {
        "ACTION6": "no observable effect at any tested coord",
    }}
    k2 = k.merged_with_delta(delta)
    assert "no observable effect" in k2.action_semantics["ACTION6"]


def test_BUG9_drops_only_offending_key_in_mixed_delta() -> None:
    """A delta with one good entry and one bad entry should apply the
    good one and silently drop the bad one."""
    k = Knowledge.empty("ar25")
    delta = {"action_semantics_update": {
        "ACTION1": "moves an active object UP",       # bad
        "ACTION2": "moves the blue 1x1 DOWN by 2",    # good
    }}
    k2 = k.merged_with_delta(delta)
    assert "ACTION1" not in k2.action_semantics
    assert k2.action_semantics["ACTION2"] == "moves the blue 1x1 DOWN by 2"


# ── BUG-10: Knowledge.click_targets persistence + Reflection hooks ────


def test_BUG10_click_targets_default_empty() -> None:
    k = Knowledge.empty("ar25")
    assert k.click_targets == []


def test_BUG10_click_targets_roundtrip_via_dict() -> None:
    from arc_agent.click_targets import ClickTarget
    k = Knowledge.empty("ar25")
    k.click_targets = [
        ClickTarget(
            obj_id="obj_005", signature="cyan_1x1", coords=(12, 30),
            color_name="cyan", bbox=(12, 30, 12, 30),
            confidence=0.49, tries=3, successes=0, last_seen_step=7,
        ),
    ]
    s = json.dumps(k.to_dict())
    k2 = Knowledge.from_dict(json.loads(s))
    assert len(k2.click_targets) == 1
    assert k2.click_targets[0].obj_id == "obj_005"
    assert k2.click_targets[0].coords == (12, 30)
    assert k2.click_targets[0].confidence == pytest.approx(0.49)


def test_BUG10_merge_marks_dead_zeroes_confidence() -> None:
    """Reflection can flag a target as confirmed-useless. The orchestrator
    zeroes confidence and sets alive=False so the prompt drops it."""
    from arc_agent.click_targets import ClickTarget
    k = Knowledge.empty("ar25")
    k.click_targets = [
        ClickTarget(obj_id="obj_005", signature="cyan_1x1", coords=(10, 20),
                    color_name="cyan", bbox=(10, 20, 10, 20),
                    confidence=0.7, tries=2),
        ClickTarget(obj_id="obj_009", signature="red_1x1", coords=(40, 40),
                    color_name="red", bbox=(40, 40, 40, 40),
                    confidence=0.5, tries=1),
    ]
    k2 = k.merged_with_delta({"click_targets_marked_dead": ["obj_005"]})

    by_uid = {t.obj_id: t for t in k2.click_targets}
    assert by_uid["obj_005"].confidence == 0.0
    assert by_uid["obj_005"].alive is False
    # The other target is untouched
    assert by_uid["obj_009"].confidence == pytest.approx(0.5)
    assert by_uid["obj_009"].alive is True


def test_BUG10_merge_promoted_resets_target() -> None:
    """Reflection can rescue a wrongly-decayed target — gives it a fresh
    confidence=1.0 and zero tries so it bubbles back to the top of the list."""
    from arc_agent.click_targets import ClickTarget
    k = Knowledge.empty("ar25")
    k.click_targets = [ClickTarget(
        obj_id="obj_005", signature="cyan_1x1", coords=(10, 20),
        color_name="cyan", bbox=(10, 20, 10, 20),
        confidence=0.05, tries=10, successes=0,
    )]
    k2 = k.merged_with_delta({"click_targets_promoted": ["obj_005"]})
    assert k2.click_targets[0].confidence == 1.0
    assert k2.click_targets[0].tries == 0


def test_BUG10_merge_ignores_unknown_obj_ids() -> None:
    """marked_dead / promoted with obj_ids not in the list = no-op."""
    from arc_agent.click_targets import ClickTarget
    k = Knowledge.empty("ar25")
    k.click_targets = [ClickTarget(
        obj_id="obj_005", signature="cyan_1x1", coords=(10, 20),
        color_name="cyan", bbox=(10, 20, 10, 20),
        confidence=0.7,
    )]
    k2 = k.merged_with_delta({
        "click_targets_marked_dead": ["obj_999"],
        "click_targets_promoted": ["obj_888"],
    })
    assert k2.click_targets[0].confidence == pytest.approx(0.7)
    assert k2.click_targets[0].alive is True
