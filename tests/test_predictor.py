"""Predictor module tests — dataset + features + models.

Per docs/arch_predictor_v0_zh.md S11. No torch + sklearn? Skip the heavy
tests; the unit suite must run on a GPU-less box.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from arc_agent.predictor.dataset import (
    Sample, scan_traces, build_dataset, _parse_row, _short_game,
)
from arc_agent.predictor.features import (
    encode, encode_batch, labels_array, FEATURE_DIM,
)
from arc_agent.predictor.features_no_game import (
    encode_no_game, encode_batch_no_game, _GAME_SLOTS,
)


# ───────── dataset.py ──────────────────────────────────────────────────


def _v3_2_row(**overrides) -> dict:
    """Minimal v3.2 schema row."""
    row = {
        "step": 0,
        "game_id": "ar25-0c556536",
        "action": "ACTION1",
        "action_coords": None,
        "frame_changed": True,
        "no_op_streak": 0,
        "state_revisit_count": 1,
        "primary_direction": "UP",
        "primary_distance": 3,
    }
    row.update(overrides)
    return row


def _v3_legacy_row(**overrides) -> dict:
    """Older v3 / ablation schema row."""
    row = {
        "step": 0,
        "game_id": "bp35-0a0ad940",
        "chosen_action": "ACTION4",
        "real_diff": [[1, 2, 3]],  # one cell changed
    }
    row.update(overrides)
    return row


def test_parse_v3_2_changed():
    row = _v3_2_row(action="ACTION3", frame_changed=False, primary_direction="DOWN")
    s = _parse_row(row, Path("/tmp"))
    assert s is not None
    assert s.action == "ACTION3"
    assert s.label == 0
    assert s.primary_direction == "DOWN"
    assert s.game_id == "ar25-0c556536"


def test_parse_legacy_real_diff_nonempty():
    s = _parse_row(_v3_legacy_row(), Path("/tmp"))
    assert s is not None
    assert s.action == "ACTION4"
    assert s.label == 1


def test_parse_legacy_real_diff_empty_means_no_change():
    s = _parse_row(_v3_legacy_row(real_diff=[]), Path("/tmp"))
    assert s is not None
    assert s.label == 0


def test_parse_skips_reset_and_unknown():
    s = _parse_row(_v3_2_row(action="RESET"), Path("/tmp"))
    assert s is None
    s2 = _parse_row(_v3_legacy_row(chosen_action=None), Path("/tmp"))
    assert s2 is None


def test_scan_traces_finds_and_parses(tmp_path: Path):
    # Create a fake trace.jsonl tree
    rd = tmp_path / "exp_a" / "round_00"
    rd.mkdir(parents=True)
    (rd / "trace.jsonl").write_text(
        json.dumps(_v3_2_row()) + "\n" +
        json.dumps(_v3_2_row(step=1, action="ACTION6",
                              frame_changed=False, action_coords=[5, 60])) + "\n",
        encoding="utf-8",
    )
    samples = scan_traces(tmp_path)
    assert len(samples) == 2
    assert samples[0].action == "ACTION1"
    assert samples[1].action == "ACTION6"
    assert samples[1].action_coords == (5, 60)


def test_short_game():
    assert _short_game("ar25-0c556536") == "ar25"
    assert _short_game("bp35-1234") == "bp35"
    assert _short_game("nodash") == "nodash"


def test_build_dataset_per_action_balance_drops_degenerate():
    # 10 ACTION1 with 5 changed + 5 no-op = balanced ok
    # 10 ACTION2 with 10 changed + 0 no-op = degenerate, drop
    raw = []
    for i in range(5):
        raw.append(Sample("ar25", i, "ACTION1", None, 1, None))
        raw.append(Sample("ar25", i, "ACTION1", None, 0, None))
        raw.append(Sample("ar25", i, "ACTION2", None, 1, None))
        raw.append(Sample("ar25", i, "ACTION2", None, 1, None))
    out = build_dataset(raw, balance="per_action")
    actions = {s.action for s in out}
    assert "ACTION1" in actions
    assert "ACTION2" not in actions  # dropped as degenerate


def test_build_dataset_balance_equalises_classes():
    raw = []
    for i in range(20):
        raw.append(Sample("ar25", i, "ACTION1", None, 1, None))
    for i in range(60):
        raw.append(Sample("ar25", i, "ACTION1", None, 0, None))
    out = build_dataset(raw, balance="per_action")
    pos = sum(1 for s in out if s.label == 1)
    neg = sum(1 for s in out if s.label == 0)
    assert pos == neg == 20


# ───────── features.py ─────────────────────────────────────────────────


def test_feature_dim_constant():
    assert FEATURE_DIM == 35
    s = Sample("ar25", 0, "ACTION1", None, 1, None)
    vec = encode(s)
    assert vec.shape == (FEATURE_DIM,)


def test_encode_action_one_hot():
    s = Sample("ar25", 0, "ACTION3", None, 1, None)
    vec = encode(s)
    assert vec[2] == 1.0  # ACTION3 -> idx 2
    assert vec[0] == 0.0


def test_encode_action6_coords_normalised():
    s = Sample("ar25", 0, "ACTION6", (32, 16), 1, None)
    vec = encode(s)
    assert vec[5] == 1.0   # ACTION6 -> idx 5
    assert abs(vec[7] - 0.5) < 1e-6     # x=32/64
    assert abs(vec[8] - 0.25) < 1e-6    # y=16/64
    assert vec[31] == 1.0   # is_complex
    assert vec[32] == 1.0   # has_coords


def test_encode_non_action6_zero_coords():
    s = Sample("ar25", 0, "ACTION1", None, 1, None)
    vec = encode(s)
    assert vec[7] == 0.0
    assert vec[8] == 0.0
    assert vec[31] == 0.0
    assert vec[32] == 0.0


def test_encode_game_one_hot():
    for short, idx in [("ar25", 9), ("bp35", 10), ("cd82", 11),
                        ("cn04", 12), ("dc22", 13)]:
        s = Sample(short + "-xxxxx", 0, "ACTION1", None, 1, None)
        vec = encode(s)
        assert vec[idx] == 1.0
        assert vec[14] == 0.0


def test_encode_unknown_game_lands_in_slot_14():
    s = Sample("unknown-game", 0, "ACTION1", None, 1, None)
    vec = encode(s)
    assert vec[14] == 1.0
    for i in range(9, 14):
        assert vec[i] == 0.0


def test_encode_no_op_streak_buckets():
    # buckets: 0, 1-2, 3-4, 5-7, 8+
    cases = [(0, 15), (1, 16), (2, 16), (3, 17), (5, 18), (8, 19), (50, 19)]
    for streak, idx in cases:
        s = Sample("ar25", 0, "ACTION1", None, 1, None, no_op_streak=streak)
        vec = encode(s)
        # exactly that bucket should be on
        for i in range(15, 20):
            assert vec[i] == (1.0 if i == idx else 0.0), f"streak={streak} idx={idx}"


def test_encode_direction_one_hot():
    s = Sample("ar25", 0, "ACTION1", None, 1, None, primary_direction="LEFT")
    vec = encode(s)
    assert vec[27] == 1.0   # LEFT
    assert vec[25] == 0.0
    assert vec[29] == 0.0   # none-slot should be off


def test_encode_direction_none_fallback():
    s = Sample("ar25", 0, "ACTION1", None, 1, None, primary_direction=None)
    vec = encode(s)
    assert vec[29] == 1.0   # 'none' bucket


def test_encode_batch_shape():
    samples = [
        Sample("ar25", i, "ACTION1", None, i % 2, None) for i in range(7)
    ]
    X = encode_batch(samples)
    assert X.shape == (7, FEATURE_DIM)
    assert X.dtype == np.float32
    y = labels_array(samples)
    assert y.shape == (7,)
    assert y.dtype == np.float32


def test_encode_batch_empty():
    X = encode_batch([])
    assert X.shape == (0, FEATURE_DIM)


# ───────── features_no_game.py (ablation) ──────────────────────────────


def test_no_game_zeros_game_slots():
    s = Sample("ar25-xxxx", 0, "ACTION1", None, 1, None)
    vec_full = encode(s)
    vec_ng = encode_no_game(s)
    assert vec_full[9] == 1.0
    assert vec_ng[9] == 0.0
    for i in _GAME_SLOTS:
        assert vec_ng[i] == 0.0
    # Non-game slots unchanged
    for i in range(FEATURE_DIM):
        if i not in _GAME_SLOTS:
            assert vec_ng[i] == vec_full[i]


def test_encode_batch_no_game_shape():
    samples = [Sample("ar25", i, "ACTION1", None, i % 2, None) for i in range(4)]
    X = encode_batch_no_game(samples)
    assert X.shape == (4, FEATURE_DIM)


# ───────── models.py (light tests; skip if torch missing) ──────────────


def test_logreg_smoke():
    pytest.importorskip("sklearn")
    from arc_agent.predictor.models import LogRegPredictor
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, FEATURE_DIM)).astype(np.float32)
    y = ((X[:, 0] + X[:, 1] > 0).astype(np.float32))
    m = LogRegPredictor()
    hist = m.fit(X, y, X, y, max_epochs=100, seed=0)
    assert "train_loss" in hist
    p = m.predict_proba(X)
    assert p.shape == (100,)
    # synthetic: model should be near-perfect
    auc_pred = ((p > 0.5).astype(np.float32) == y).mean()
    assert auc_pred > 0.85


def test_mlp_s_smoke():
    pytest.importorskip("torch")
    from arc_agent.predictor.models import MLP_S
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, FEATURE_DIM)).astype(np.float32)
    y = ((X[:, 0] + X[:, 1] > 0).astype(np.float32))
    m = MLP_S(in_dim=FEATURE_DIM)
    hist = m.fit(X, y, X, y, max_epochs=10, seed=0)
    assert len(hist["train_loss"]) == 10
    p = m.predict_proba(X)
    assert p.shape == (80,)
    assert (p >= 0).all() and (p <= 1).all()
