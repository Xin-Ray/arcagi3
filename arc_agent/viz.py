"""Visualization helpers for baseline / validation runs.

Two pure functions, both consume PIL — no I/O outside the GIF writer:

- `compose_step_image(grid_now, predicted_diff, grid_next, json_text, header)`
  builds the 4-quadrant per-step composite documented in
  `docs/ARCHITECTURE_RL.md` §5.2 (current grid · predicted overlay · real
  next · model JSON text), with a header band on top.

- `write_gif(frames, out_path, fps)` saves a list of PIL Images as an
  animated GIF. The only side effect of this module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from arc_agent.observation import _ARC_PALETTE, grid_to_image


def _clip_palette(color: int) -> tuple[int, int, int]:
    return _ARC_PALETTE[max(0, min(15, int(color)))]


def compose_step_image(
    grid_now: np.ndarray,
    predicted_diff: set[tuple[int, int, int]] | None,
    grid_next: np.ndarray,
    json_text: str,
    *,
    header: str,
    cell_scale: int = 4,
):
    """Build the 4-quadrant composite for one trace step.

    Layout (each grid cell is `cell_scale × cell_scale` pixels):

        ┌──────────────── header band ───────────────┐
        │  <header>                                  │
        ├──────────────────┬─────────────────────────┤
        │  grid_now        │  grid_now overlaid with │
        │  (current state) │  predicted_diff cells   │
        ├──────────────────┼─────────────────────────┤
        │  grid_next       │  json_text panel        │
        │  (real next)     │  (model output, wrapped)│
        └──────────────────┴─────────────────────────┘

    Both grids must have the same shape. `predicted_diff` may be None or
    empty (then the prediction quadrant just shows `grid_now`).
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError as e:
        raise ImportError("Pillow is required for compose_step_image") from e

    if grid_now.shape != grid_next.shape:
        raise ValueError(
            f"grid_now {grid_now.shape} != grid_next {grid_next.shape}"
        )
    if cell_scale < 1:
        raise ValueError(f"cell_scale must be >= 1, got {cell_scale}")

    cell_w = grid_now.shape[1] * cell_scale
    cell_h = grid_now.shape[0] * cell_scale
    header_h = 40
    total_w = cell_w * 2
    total_h = header_h + cell_h * 2

    canvas = Image.new("RGB", (total_w, total_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), header, fill=(255, 255, 255))

    # Quadrant 1 (top-left): current grid
    canvas.paste(grid_to_image(grid_now, scale=cell_scale), (0, header_h))

    # Quadrant 2 (top-right): current grid + predicted-diff overlay
    pred_img = grid_to_image(grid_now, scale=cell_scale).copy()
    if predicted_diff:
        overlay = ImageDraw.Draw(pred_img)
        for r, c, new_color in predicted_diff:
            x0 = int(c) * cell_scale
            y0 = int(r) * cell_scale
            x1 = x0 + cell_scale - 1
            y1 = y0 + cell_scale - 1
            overlay.rectangle(
                [x0, y0, x1, y1],
                fill=_clip_palette(new_color),
                outline=(255, 255, 255),
                width=1,
            )
    canvas.paste(pred_img, (cell_w, header_h))

    # Quadrant 3 (bottom-left): real next grid
    canvas.paste(grid_to_image(grid_next, scale=cell_scale), (0, header_h + cell_h))

    # Quadrant 4 (bottom-right): JSON text panel
    text_box = Image.new("RGB", (cell_w, cell_h), color=(30, 30, 30))
    text_draw = ImageDraw.Draw(text_box)
    lines: list[str] = []
    for raw_line in (json_text or "").split("\n"):
        while len(raw_line) > 48:
            lines.append(raw_line[:48])
            raw_line = raw_line[48:]
        lines.append(raw_line)
    line_h = 12
    max_lines = max(1, (cell_h - 12) // line_h)
    for i, line in enumerate(lines[:max_lines]):
        text_draw.text((6, 6 + i * line_h), line, fill=(220, 220, 220))
    canvas.paste(text_box, (cell_w, header_h + cell_h))

    return canvas


def write_gif(
    frames: Iterable,
    out_path: Path | str,
    *,
    fps: int = 2,
) -> Path:
    """Save an iterable of PIL Images as an animated GIF and return the path.

    `frames` must contain at least one image; all frames should be the same
    size and mode (RGB). Returns the resolved Path of the written GIF.
    """
    seq = list(frames)
    if not seq:
        raise ValueError("write_gif: no frames provided")
    if fps < 1:
        raise ValueError(f"fps must be >= 1, got {fps}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(20, int(1000 / fps))
    seq[0].save(
        out,
        save_all=True,
        append_images=seq[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return out
