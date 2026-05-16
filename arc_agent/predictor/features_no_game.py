"""Ablation: F1 hand features WITHOUT the game_id one-hot.

Zeros out the game-id slots in the F1 encoding so we can isolate
whether the predictor learned per-game priors vs in-state signal.

Per docs/arch_predictor_v0_zh.md S2.2 + per-action AUC discussion.
"""
from __future__ import annotations

import numpy as np

from arc_agent.predictor.dataset import Sample
from arc_agent.predictor.features import encode, FEATURE_DIM

# Game one-hot lives at indices [9, 10, 11, 12, 13, 14]
_GAME_SLOTS = list(range(9, 15))


def encode_no_game(sample: Sample) -> np.ndarray:
    """Same as `features.encode` but zero out the game-id slots."""
    vec = encode(sample)
    for i in _GAME_SLOTS:
        vec[i] = 0.0
    return vec


def encode_batch_no_game(samples: list[Sample]) -> np.ndarray:
    if not samples:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.stack([encode_no_game(s) for s in samples])
