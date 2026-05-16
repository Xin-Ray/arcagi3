"""Reverse the viz_v3_2 rendering: PNG -> 64x64 grid.

`compose_step_image_v32` draws the 64x64 grid as a 4x-upscaled 256x256
block sitting on the LEFT half of a 512x286 composite PNG. The grid
region starts at (HEADER_H, 0) and has stride 4 per grid cell. We sample
the center of each 4x4 block, look up the nearest ARC palette entry, and
return a (64, 64) int8 numpy array.

If `viz_v3_2.compose_step_image_v32` is changed (header height, scale,
or layout), this decoder must change too. Keep in sync via the constants
re-imported here.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional

from arc_agent.observation import _ARC_PALETTE

# Mirrors arc_agent/viz_v3_2.py constants. If they drift, fix here.
_HEADER_H = 30        # top header band on the composite PNG
_GRID_SIDE_PX = 256   # 64 * 4
_SCALE = 4

# Precompute palette as numpy for fast nearest-neighbor lookup
_PALETTE = np.array(_ARC_PALETTE, dtype=np.int16)  # shape (16, 3)


def decode_grid(png_path: Path) -> Optional[np.ndarray]:
    """Decode a viz_v3_2 step PNG back to a 64x64 grid.

    Returns None if the file is missing or doesn't match the expected
    layout. Output dtype is int8 (color id in [0, 15]).
    """
    if not png_path.exists():
        return None
    try:
        from PIL import Image
        img = Image.open(png_path).convert("RGB")
    except (OSError, ImportError):
        return None
    arr = np.asarray(img, dtype=np.int16)  # (H, W, 3)
    if arr.shape[0] < _HEADER_H + _GRID_SIDE_PX or arr.shape[1] < _GRID_SIDE_PX:
        return None
    # Grid region is the upper-left _GRID_SIDE_PX x _GRID_SIDE_PX below header
    grid_region = arr[_HEADER_H: _HEADER_H + _GRID_SIDE_PX, : _GRID_SIDE_PX, :]
    # Sample centers of each 4x4 block
    centers_y = (np.arange(64) * _SCALE) + (_SCALE // 2)
    centers_x = centers_y.copy()
    sampled = grid_region[centers_y[:, None], centers_x[None, :], :]  # (64, 64, 3)

    # Nearest palette lookup
    # diff shape (64, 64, 16) — sum over RGB axis
    diff = sampled[:, :, None, :] - _PALETTE[None, None, :, :]
    dist = (diff * diff).sum(axis=-1)
    out = dist.argmin(axis=-1).astype(np.int8)
    return out


def encode_for_cnn(grid: np.ndarray, action_idx: int) -> np.ndarray:
    """Build the CNN input tensor for one (grid, action) pair.

    Returns shape (17, 64, 64): 16 one-hot color channels + 1 action
    channel (constant value = action_idx / 7 broadcast over the 64x64
    grid).
    """
    if grid.shape != (64, 64):
        raise ValueError(f"expected 64x64 grid, got {grid.shape}")
    one_hot = np.zeros((16, 64, 64), dtype=np.float32)
    for c in range(16):
        one_hot[c][grid == c] = 1.0
    action_plane = np.full((1, 64, 64), action_idx / 7.0, dtype=np.float32)
    return np.concatenate([one_hot, action_plane], axis=0)
