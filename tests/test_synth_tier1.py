"""Tests for arc_agent.finetune.synth_tier1 — Tier 1 SFT data generators.

The most important test in this file is `test_T8_no_fixed_binding`: the
v3 architecture forbids hard-coding any ACTION->direction prior into the
LLM weights (action semantics are per-game and must be discovered via
OutcomeLog). T8 must therefore distribute every action token across
every direction with comparable frequency — if this test fails, the SFT
data is poisoning the model.
"""
from __future__ import annotations

import random
from collections import Counter

import pytest

from arc_agent.finetune.chat_format import to_chat
from arc_agent.finetune.synth_tier1 import (
    ACTION_TOKENS,
    DIRECTIONS_4,
    Sample,
    build_ood,
    build_tier1,
    gen_T1,
    gen_T2,
    gen_T2_multi,
    gen_T2_remainder,
    gen_T3,
    gen_T4,
    gen_T8,
)


def _required_keys(sample: Sample) -> None:
    """Every sample has the 4 keys, all non-empty strings."""
    for k in ("system", "user", "assistant", "task"):
        assert k in sample, f"missing key: {k}"
        assert isinstance(sample[k], str)
        assert sample[k], f"empty value for: {k}"


# ── T1 ────────────────────────────────────────────────────────────────────

def test_T1_screen_convention_label_correct() -> None:
    """Force screen convention; src_y < dst_y -> DOWN."""
    rng = random.Random(0)
    for _ in range(200):
        s = gen_T1(rng, "en")
        _required_keys(s)
        assert s["task"] == "T1"
        assert s["assistant"] in ("UP", "DOWN")
        # parse the convention + numbers out of the user prompt
        if "top to bottom" in s["user"]:
            # screen: larger dst_y -> DOWN
            src = int(s["user"].split("Object at y=")[1].split(".")[0])
            dst = int(s["user"].split("Target at y=")[1].split(".")[0])
            expected = "DOWN" if dst > src else "UP"
        else:
            src = int(s["user"].split("Object at y=")[1].split(".")[0])
            dst = int(s["user"].split("Target at y=")[1].split(".")[0])
            expected = "UP" if dst > src else "DOWN"
        assert s["assistant"] == expected


def test_T1_zh_label_consistent_with_en() -> None:
    """Chinese T1 should be the same task with a translated prompt."""
    rng = random.Random(1)
    s = gen_T1(rng, "zh")
    _required_keys(s)
    assert s["assistant"] in ("UP", "DOWN")
    assert "y" in s["user"]
    assert "y" in s["user"] and ("从上往下" in s["user"] or "从下往上" in s["user"])


def test_T1_convention_distribution_balanced() -> None:
    """50/50 screen vs math convention (the doc's anti-prior design)."""
    rng = random.Random(42)
    n = 4000
    screen = sum(
        1 for _ in range(n)
        if "top to bottom" in gen_T1(rng, "en")["user"]
    )
    # 95% CI for 4000 binomial(0.5) is roughly 2000 ± 60
    assert 1900 < screen < 2100, f"unbalanced T1 conventions: {screen}/{n}"


# ── T2 main + variants ───────────────────────────────────────────────────

def test_T2_main_arithmetic() -> None:
    rng = random.Random(0)
    for _ in range(200):
        s = gen_T2(rng, "en")
        _required_keys(s)
        assert s["task"] == "T2"
        n = int(s["assistant"])
        assert 1 <= n <= 20


def test_T2_remainder_format_strict() -> None:
    """Output is exactly `完整 <Q> 余 <R>`."""
    rng = random.Random(0)
    for _ in range(100):
        s = gen_T2_remainder(rng)
        _required_keys(s)
        assert s["task"] == "T2_remainder"
        parts = s["assistant"].split()
        assert len(parts) == 4
        assert parts[0] == "完整" and parts[2] == "余"
        int(parts[1])  # ensure integers
        int(parts[3])


def test_T2_multi_format_strict() -> None:
    """Output is `x:<n> y:<m>` (no spaces inside, colon separator)."""
    rng = random.Random(0)
    for _ in range(100):
        s = gen_T2_multi(rng)
        _required_keys(s)
        assert s["task"] == "T2_multi"
        parts = s["assistant"].split()
        assert len(parts) == 2
        assert parts[0].startswith("x:")
        assert parts[1].startswith("y:")
        int(parts[0][2:])
        int(parts[1][2:])


# ── T3 ────────────────────────────────────────────────────────────────────

def test_T3_relation_map_correct() -> None:
    """If A is at the top-left of B, A->B move is right-down (and so on)."""
    rng = random.Random(0)
    for _ in range(200):
        s = gen_T3(rng, "en")
        _required_keys(s)
        assert s["task"] == "T3"
        ans = s["assistant"]
        assert ans in {
            "up", "down", "left", "right",
            "left-up", "right-up", "left-down", "right-down",
        }


def test_T3_zh_uses_chinese_relation() -> None:
    rng = random.Random(7)
    s = gen_T3(rng, "zh")
    _required_keys(s)
    # at least one Chinese relation char should appear
    assert any(c in s["user"] for c in ("左", "右", "上", "下"))


# ── T4 ────────────────────────────────────────────────────────────────────

def test_T4_user_prompt_is_format_agnostic() -> None:
    """T4 deliberately decouples user prompt from system task; user is fixed."""
    rng = random.Random(0)
    seen_user = set()
    for _ in range(50):
        s = gen_T4(rng)
        _required_keys(s)
        seen_user.add(s["user"])
    assert seen_user == {"Now produce one valid output following the format."}


# ── T8: the load-bearing test ────────────────────────────────────────────

def test_T8_no_fixed_binding() -> None:
    """CRITICAL: every (token, direction) cell must have ≥ 20% within its
    token row. If any cell drops below, the SFT data is teaching the LLM
    a fixed prior — directly breaks v3's per-game OutcomeLog design.
    See docs/arch_sft_tier1_zh.md §3.6.
    """
    rng = random.Random(42)
    n = 40_000
    pairs = Counter()
    for _ in range(n):
        s = gen_T8(rng, "en")
        token = s["assistant"].split("action:")[1].strip()
        # extract direction from the reasoning line
        reasoning = s["assistant"].split("\n")[0]
        for d in ("up", "down", "left", "right"):
            if f" {d} " in reasoning or reasoning.endswith(d):
                direction = d.upper()
                break
        else:
            pytest.fail(f"no direction word found in: {reasoning!r}")
        pairs[(token, direction)] += 1

    for token in ACTION_TOKENS:
        row_total = sum(pairs[(token, d)] for d in DIRECTIONS_4)
        assert row_total > 0, f"token {token} never appeared"
        for d in DIRECTIONS_4:
            frac = pairs[(token, d)] / row_total
            assert frac >= 0.20, (
                f"{token} maps to {d} only {frac:.2%} of the time "
                f"(< 20% — binding too skewed; see §3.6)"
            )


def test_T8_action_and_reasoning_directions_agree() -> None:
    """The whole point of T8 is `reasoning` direction == `action` direction."""
    rng = random.Random(11)
    for _ in range(500):
        s = gen_T8(rng, "en")
        # parse direction declared in user prompt
        for d in DIRECTIONS_4:
            if f"moves objects {d}" in s["user"]:
                user_dir = d
                break
        else:
            pytest.fail("no direction in user prompt")
        # reasoning + action lines
        reasoning, action_line = s["assistant"].split("\n")
        assert user_dir.lower() in reasoning.lower()
        token = action_line.replace("action:", "").strip()
        assert token in ACTION_TOKENS
        assert token in s["user"]  # binding is in-context


# ── full mix + OOD ───────────────────────────────────────────────────────

def test_build_tier1_deterministic_with_seed() -> None:
    """Same seed -> identical samples (so we can reproduce training data)."""
    a = build_tier1(seed=99)[:200]
    b = build_tier1(seed=99)[:200]
    assert a == b


def test_build_tier1_total_size() -> None:
    """Spec sums to 200k."""
    samples = build_tier1(seed=42)
    assert 199_000 <= len(samples) <= 201_000
    # task tags non-empty, no malformed records
    tasks = Counter(s["task"] for s in samples)
    assert "T1" in tasks and tasks["T1"] >= 49_000
    assert "T8" in tasks and tasks["T8"] >= 39_000


def test_ood_has_no_overlap_with_train() -> None:
    """OOD must use phrasing/numbers outside the training distribution.

    Compare the (system, user) tuple, not just user — T4 by design uses a
    fixed content-free user prompt, so it's the system template that
    carries the novelty signal.
    """
    train = {
        (s["system"], s["user"]) for s in build_tier1(seed=42)[:5000]
    }
    ood = build_ood(seed=1337, n_per_task=200)
    for s in ood:
        assert (s["system"], s["user"]) not in train


def test_ood_uses_distinct_task_tags() -> None:
    """OOD samples are tagged `<T>_ood` so eval can split them out."""
    ood = build_ood(seed=1, n_per_task=20)
    for s in ood:
        assert s["task"].endswith("_ood")


# ── chat_format ──────────────────────────────────────────────────────────

def test_to_chat_produces_three_role_messages() -> None:
    rng = random.Random(0)
    chat = to_chat(gen_T1(rng, "en"))
    roles = [m["role"] for m in chat["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert chat["task"] == "T1"
