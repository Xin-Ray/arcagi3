"""Observation parsers — turn FrameDataRaw into LLM-ready text.

Each cell is one of 16 colors → encoded as a single hex char (0–F).
A 64×64 grid is therefore 64 lines × 64 chars = 4096 chars (~1500 BPE tokens).

Multi-frame animations: most agents care about the last frame (the resolved
post-step state). `latest_grid()` returns it; if you need the full animation,
read `frame.frame` directly.
"""
from __future__ import annotations

import numpy as np
from arcengine import FrameDataRaw, GameAction

_HEX = "0123456789ABCDEF"  # 16-color palette


def latest_grid(frame: FrameDataRaw) -> np.ndarray:
    """Return the LAST 2D grid of an animation (the post-step state).

    Raises ValueError if `frame.frame` is empty.
    """
    if not frame.frame:
        raise ValueError("FrameDataRaw.frame is empty — no grid to read")
    return frame.frame[-1]


def grid_to_text(grid: np.ndarray) -> str:
    """Render a (H, W) int grid as multi-line hex string. Values clipped to [0, 15]."""
    if grid.ndim != 2:
        raise ValueError(f"expected 2D grid, got shape {grid.shape}")
    clipped = np.clip(grid, 0, 15).astype(int)
    return "\n".join("".join(_HEX[v] for v in row) for row in clipped)


def grid_diff(prev: np.ndarray, curr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return [(row, col, old_value, new_value), ...] for cells that differ.

    Both grids must have the same shape.
    """
    if prev.shape != curr.shape:
        raise ValueError(f"shape mismatch: {prev.shape} vs {curr.shape}")
    rows, cols = np.where(prev != curr)
    return [
        (int(r), int(c), int(prev[r, c]), int(curr[r, c]))
        for r, c in zip(rows.tolist(), cols.tolist())
    ]


def available_action_names(frame: FrameDataRaw) -> list[str]:
    """Map `frame.available_actions` (ints) to GameAction names."""
    out: list[str] = []
    for v in frame.available_actions:
        try:
            out.append(GameAction.from_id(v).name)
        except (ValueError, KeyError):
            out.append(f"UNKNOWN_{v}")
    return out


def summarize_frame(
    frame: FrameDataRaw,
    *,
    include_grid: bool = True,
    diff_with: np.ndarray | None = None,
) -> str:
    """One-shot text summary: state line + available actions + (optional) ASCII grid + (optional) diff."""
    lines = [
        f"state: {frame.state.name}",
        f"levels_completed: {frame.levels_completed}/{frame.win_levels}",
        f"available_actions: {available_action_names(frame)}",
    ]
    if include_grid and frame.frame:
        grid = latest_grid(frame)
        lines.append(f"grid ({grid.shape[0]}x{grid.shape[1]}, hex 0..F):")
        lines.append(grid_to_text(grid))
    if diff_with is not None and frame.frame:
        diff = grid_diff(diff_with, latest_grid(frame))
        lines.append(f"changed_cells_since_prev_step: {len(diff)}")
        if 0 < len(diff) <= 32:  # show small diffs explicitly; suppress huge ones
            lines.append(f"  {diff}")
    return "\n".join(lines)
