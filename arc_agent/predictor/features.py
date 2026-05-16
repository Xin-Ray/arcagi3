"""F1 hand feature extractor for the frame-change predictor.

Output shape: float32 vector of length FEATURE_DIM (currently 35).

Layout (so unit tests can index):
  [ 0:7  ]  action one-hot (ACTION1..ACTION7)
  [ 7:9  ]  ACTION6 coords normalised (x/64, y/64); 0 for non-ACTION6
  [ 9:14 ]  game one-hot (ar25, bp35, cd82, cn04, dc22). One-hot in known
            games; the "unknown" slot 14 is set when game_id matches none
            of the above.
  [ 14   ]  unknown-game bit
  [ 15:20]  no_op_streak buckets (0, 1-2, 3-4, 5-7, 8+)
  [ 20:25]  state_revisit_count buckets (1, 2, 3-4, 5-9, 10+)
  [ 25:30]  primary_direction one-hot (UP, DOWN, LEFT, RIGHT, none)
  [ 30   ]  primary_distance / 10 (clipped to [0, 1])
  [ 31   ]  is_complex (1 if ACTION6 else 0)
  [ 32   ]  has_coords (1 if action_coords is not None else 0)
  [ 33   ]  step / 200 clipped to [0, 1]   (rough "into the round" signal)
  [ 34   ]  label-leak guard slot: always 0 (placeholder for future)

Stateless, pure numpy. Trivial to mock in tests.
"""
from __future__ import annotations

import numpy as np

from arc_agent.predictor.dataset import Sample, _short_game

# Action vocab (ordered)
_ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]
_ACTION_IDX = {a: i for i, a in enumerate(_ACTIONS)}

# Demo-set games we have data for. Anything else lands in "unknown".
_GAMES = ["ar25", "bp35", "cd82", "cn04", "dc22"]
_GAME_IDX = {g: i for i, g in enumerate(_GAMES)}

_DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
_DIR_IDX = {d: i for i, d in enumerate(_DIRECTIONS)}

FEATURE_DIM = 35


def _bucket(v: int, edges: list[int]) -> int:
    """Return the bucket index for v relative to right-open edges.
    edges=[0, 1, 3, 5, 8] means buckets are [0,0], [1,2], [3,4], [5,7], [8,+inf]."""
    for i in range(len(edges) - 1, -1, -1):
        if v >= edges[i]:
            return i
    return 0


def encode(sample: Sample) -> np.ndarray:
    """Encode one Sample to a float32 vector of length FEATURE_DIM."""
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    # Action one-hot
    aidx = _ACTION_IDX.get(sample.action)
    if aidx is not None:
        vec[aidx] = 1.0

    # ACTION6 coords
    if sample.action == "ACTION6" and sample.action_coords is not None:
        vec[7] = sample.action_coords[0] / 64.0
        vec[8] = sample.action_coords[1] / 64.0

    # Game one-hot
    short = _short_game(sample.game_id)
    gidx = _GAME_IDX.get(short)
    if gidx is not None:
        vec[9 + gidx] = 1.0
    else:
        vec[14] = 1.0  # unknown-game bit

    # no_op_streak bucket (offsets 15..19)
    nb = _bucket(sample.no_op_streak, [0, 1, 3, 5, 8])
    vec[15 + nb] = 1.0

    # state_revisit bucket (offsets 20..24)
    rb = _bucket(sample.state_revisit_count, [1, 2, 3, 5, 10])
    vec[20 + rb] = 1.0

    # primary_direction one-hot (25..29) - last slot is "none"
    didx = _DIR_IDX.get(sample.primary_direction) if sample.primary_direction else None
    if didx is not None:
        vec[25 + didx] = 1.0
    else:
        vec[29] = 1.0  # none

    # primary_distance / 10 clipped
    vec[30] = min(sample.primary_distance / 10.0, 1.0)

    # is_complex / has_coords / step
    vec[31] = 1.0 if sample.action == "ACTION6" else 0.0
    vec[32] = 1.0 if sample.action_coords is not None else 0.0
    vec[33] = min(sample.step / 200.0, 1.0)

    # vec[34] always 0 - reserved for future "label-leak guard" features
    return vec


def encode_batch(samples: list[Sample]) -> np.ndarray:
    """Encode many samples; returns shape (N, FEATURE_DIM)."""
    if not samples:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.stack([encode(s) for s in samples])


def labels_array(samples: list[Sample]) -> np.ndarray:
    """Labels as a float32 (N,) array."""
    return np.array([s.label for s in samples], dtype=np.float32)
