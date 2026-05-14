"""v3.2 step visualization -- a 2-pane PNG showing the grid + reasoning
+ reflection delta + alert.

Per `docs/arch_v3_2_zh.md` §5.5. Replaces v3's 4-quadrant
`compose_step_image` (predicted_diff doesn't exist in v3 / v3.2):

    HEADER (30 px tall, full width)
    +-----------------+------------------------+
    | grid (256x256)  | text panel (256x256)   |
    | 64x64 @ 4x scale|   [ACTION]             |
    |                 |   [REASONING] ...      |
    |                 |   [REFLECTION DELTA]   |
    |                 |   matches_reasoning:.. |
    +-----------------+------------------------+

If `alert` is non-empty, the top of the right panel gets a red
background bar with the alert text so a human scanning the GIF sees it
instantly.

The old `viz.compose_step_image` is kept for v3/baseline use.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from arc_agent.observation import grid_to_image


# ── layout constants ───────────────────────────────────────────────────────

GRID_CELL_SCALE = 4         # 64×64 grid -> 256×256 image
PANEL_W = 256
PANEL_H = 256               # match grid height
HEADER_H = 30
TOTAL_W = PANEL_W * 2       # 512
TOTAL_H = HEADER_H + PANEL_H   # 286

BG_COLOR = (20, 20, 20)
TEXT_COLOR = (220, 220, 220)
LABEL_COLOR = (140, 200, 255)
ALERT_BG = (140, 30, 30)
ALERT_TEXT = (255, 240, 240)
MATCH_COLORS = {
    "YES": (110, 220, 110),
    "PARTIAL": (220, 200, 110),
    "NO": (240, 110, 110),
    "N/A": (160, 160, 160),
}

_TEXT_WRAP_CHARS = 32       # right panel is narrow
_PANEL_PAD = 6
_LINE_H = 12


def _wrap(text: str, width: int = _TEXT_WRAP_CHARS) -> list[str]:
    """Greedy char-wrap for the narrow right panel. Returns a list of lines."""
    if not text:
        return []
    out: list[str] = []
    for raw_line in text.split("\n"):
        cur = raw_line
        while len(cur) > width:
            out.append(cur[:width])
            cur = cur[width:]
        out.append(cur)
    return out


def _summarize_reflection_delta(delta: Optional[dict[str, Any]]) -> list[str]:
    """Turn a Reflection delta dict into <=3 short bullet lines."""
    if not isinstance(delta, dict) or not delta:
        return ["(none)"]
    lines: list[str] = []
    sem = delta.get("action_semantics_update") or {}
    if isinstance(sem, dict) and sem:
        # Pick the first updated action to display compactly
        for k, v in sem.items():
            lines.append(f"{k}: {v}")
            if len(lines) >= 2:
                break
    goal = delta.get("goal_hypothesis_update")
    if goal:
        lines.append(f"goal -> {goal}")
    rules = delta.get("rules_append") or []
    if isinstance(rules, list) and rules:
        lines.append(f"+rule: {rules[0]}")
    failed = delta.get("failed_strategies_append") or []
    if isinstance(failed, list) and failed:
        lines.append(f"+failed: {failed[0]}")
    if not lines:
        lines.append("(empty delta)")
    return lines[:4]


def compose_step_image_v32(
    grid: np.ndarray,
    *,
    action: str,
    reasoning: str = "",
    reflection_delta: Optional[dict[str, Any]] = None,
    alert: str = "",
    header: str = "",
    matches_reasoning: str = "N/A",
):
    """Build the v3.2 single-step PNG.

    `grid` is the current 64×64 grid (any 2-D int array). All text args
    are plain strings (orchestrator pre-renders reasoning and reflection).
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError as e:
        raise ImportError("Pillow is required for compose_step_image_v32") from e

    if grid is None or grid.ndim != 2:
        raise ValueError("grid must be a 2-D ndarray")

    canvas = Image.new("RGB", (TOTAL_W, TOTAL_H), color=BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    if header:
        draw.text((10, 8), header[:80], fill=(255, 255, 255))

    # Left pane: grid -- scale 4× so a 64×64 grid is 256×256
    grid_img = grid_to_image(grid, scale=GRID_CELL_SCALE)
    # Letterbox if grid is smaller than 64×64 (e.g. 8×8 in tests)
    if grid_img.size != (PANEL_W, PANEL_H):
        framed = Image.new("RGB", (PANEL_W, PANEL_H), color=BG_COLOR)
        ox = (PANEL_W - grid_img.size[0]) // 2
        oy = (PANEL_H - grid_img.size[1]) // 2
        framed.paste(grid_img, (max(ox, 0), max(oy, 0)))
        grid_img = framed
    canvas.paste(grid_img, (0, HEADER_H))

    # Right pane: text panel
    panel = Image.new("RGB", (PANEL_W, PANEL_H), color=(30, 30, 30))
    panel_draw = ImageDraw.Draw(panel)
    y = _PANEL_PAD

    # Alert banner (if any)
    if alert:
        # Three-line red bar at the top
        alert_lines = _wrap(f"ALERT: {alert}", width=_TEXT_WRAP_CHARS)[:3]
        bar_h = _PANEL_PAD + _LINE_H * len(alert_lines) + _PANEL_PAD
        panel_draw.rectangle([0, 0, PANEL_W, bar_h], fill=ALERT_BG)
        for i, ln in enumerate(alert_lines):
            panel_draw.text(
                (_PANEL_PAD, _PANEL_PAD + i * _LINE_H),
                ln, fill=ALERT_TEXT,
            )
        y = bar_h + _PANEL_PAD

    # [ACTION]
    panel_draw.text((_PANEL_PAD, y), "[ACTION]", fill=LABEL_COLOR)
    y += _LINE_H
    panel_draw.text((_PANEL_PAD, y), action[:_TEXT_WRAP_CHARS] or "(unknown)",
                    fill=TEXT_COLOR)
    y += _LINE_H + 2

    # [REASONING]
    panel_draw.text((_PANEL_PAD, y), "[REASONING]", fill=LABEL_COLOR)
    y += _LINE_H
    reasoning_lines = _wrap(reasoning or "(none)", width=_TEXT_WRAP_CHARS)[:4]
    for ln in reasoning_lines:
        panel_draw.text((_PANEL_PAD, y), ln, fill=TEXT_COLOR)
        y += _LINE_H
    y += 2

    # [REFLECTION DELTA]
    if y < PANEL_H - 3 * _LINE_H:
        panel_draw.text((_PANEL_PAD, y), "[REFLECTION DELTA]", fill=LABEL_COLOR)
        y += _LINE_H
        for ln in _summarize_reflection_delta(reflection_delta):
            wrapped = _wrap(ln, width=_TEXT_WRAP_CHARS)
            for w in wrapped[:2]:
                if y >= PANEL_H - _LINE_H:
                    break
                panel_draw.text((_PANEL_PAD, y), w, fill=TEXT_COLOR)
                y += _LINE_H

    # matches_reasoning verdict (always shown at bottom-right corner)
    match_color = MATCH_COLORS.get(matches_reasoning, MATCH_COLORS["N/A"])
    verdict = f"matches: {matches_reasoning}"
    panel_draw.text(
        (_PANEL_PAD, PANEL_H - _LINE_H - _PANEL_PAD),
        verdict, fill=match_color,
    )

    canvas.paste(panel, (PANEL_W, HEADER_H))
    return canvas


def write_gif_v3_2(frames: Iterable, out_path: Path | str, *, fps: int = 2) -> Path:
    """Thin wrapper over `viz.write_gif` for v3.2 step PNGs. Lives here
    so the v3.2 orchestrator imports a single module for its visuals."""
    from arc_agent.viz import write_gif
    return write_gif(frames, out_path, fps=fps)


__all__ = ["compose_step_image_v32", "write_gif_v3_2"]
