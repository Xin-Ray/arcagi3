"""Observation parsers — turn FrameDataRaw into LLM-ready text or images.

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

_COLOR_NAMES: dict[int, str] = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "gray", 6: "magenta", 7: "orange", 8: "lightblue", 9: "maroon",
    10: "purple", 11: "tan", 12: "teal", 13: "lime", 14: "rose", 15: "navy",
}

# ARC standard 16-color palette (RGB tuples).
# Colors 0–9 match ARC-AGI-1/2 convention; 10–15 are ARC-AGI-3 extensions.
_ARC_PALETTE: list[tuple[int, int, int]] = [
    (0,   0,   0),   # 0 black
    (0,  116, 217),  # 1 blue
    (255,  65,  54), # 2 red
    (46,  204,  64), # 3 green
    (255, 220,   0), # 4 yellow
    (170, 170, 170), # 5 gray
    (240,  18, 190), # 6 magenta
    (255, 133,  27), # 7 orange
    (127, 219, 255), # 8 light blue
    (135,  12,  37), # 9 maroon
    ( 84,  13, 110), # 10 purple
    (230, 190, 120), # 11 tan
    ( 40, 160, 140), # 12 teal
    (200, 230,  50), # 13 lime
    (180,  80,  80), # 14 rose
    (  5, 100, 180), # 15 navy
]


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
    include_animation: bool = True,
) -> str:
    """One-shot text summary for LLM prompts.

    Includes: state / level / available actions / optional ASCII grid /
    optional cell diff / optional animation path analysis.
    """
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
        if 0 < len(diff) <= 32:
            lines.append(f"  {diff}")
    if include_animation and frame.frame and len(frame.frame) >= 2:
        lines.append(animation_to_text(frame.frame))
    return "\n".join(lines)


def analyze_animation(frames: list[np.ndarray]) -> list[dict]:
    """Track how each colored object moves across animation frames.

    Returns one record per object that changed, with direction, distance, and
    path description. Handles linear, multi-segment, appear, and disappear events.

    Input:  list of (H,W) grids from FrameDataRaw.frame  (must have >= 2 frames)
    Output: list of dicts — see _make_motion_record for field names
    """
    if len(frames) < 2:
        return []

    first, last = frames[0], frames[-1]
    colors_first = set(int(c) for c in np.unique(first) if c != 0)
    colors_last  = set(int(c) for c in np.unique(last)  if c != 0)
    all_colors   = colors_first | colors_last

    records: list[dict] = []
    for color in sorted(all_colors):
        mask_f = first == color
        mask_l = last  == color
        appeared    = (not mask_f.any()) and mask_l.any()
        disappeared = mask_f.any() and (not mask_l.any())
        name = _COLOR_NAMES.get(color, f"color-{color}")

        if appeared:
            rows, cols = np.where(mask_l)
            pos = (int(rows.mean()), int(cols.mean()))
            records.append({
                "color": color, "name": name, "event": "appeared",
                "to_pos": pos,
                "summary": f"color {color} ({name}): APPEARED at row~{pos[0]} col~{pos[1]}",
            })
            continue

        if disappeared:
            rows, cols = np.where(mask_f)
            pos = (int(rows.mean()), int(cols.mean()))
            records.append({
                "color": color, "name": name, "event": "disappeared",
                "from_pos": pos,
                "summary": f"color {color} ({name}): DISAPPEARED from row~{pos[0]} col~{pos[1]}",
            })
            continue

        # Present in both frames — compute centroid per frame to get path
        centroids: list[tuple[float, float] | None] = []
        for f in frames:
            m = f == color
            if m.any():
                r, c = np.where(m)
                centroids.append((float(r.mean()), float(c.mean())))
            else:
                centroids.append(None)

        valid = [(i, c) for i, c in enumerate(centroids) if c is not None]
        if len(valid) < 2:
            continue

        from_pos = valid[0][1]
        to_pos   = valid[-1][1]
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]

        if abs(dr) < 0.5 and abs(dc) < 0.5:
            continue  # stationary

        segments = _path_segments(valid)
        summary  = _format_motion(color, name, from_pos, to_pos, segments, len(frames))

        records.append({
            "color":    color,
            "name":     name,
            "event":    "moved",
            "from_pos": (round(from_pos[0]), round(from_pos[1])),
            "to_pos":   (round(to_pos[0]),   round(to_pos[1])),
            "segments": segments,   # list of (direction, distance) tuples
            "n_frames": len(frames),
            "summary":  summary,
        })

    return records


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    rows, cols = np.where(mask)
    return float(rows.mean()), float(cols.mean())


def _path_segments(valid: list[tuple[int, tuple[float, float]]]) -> list[tuple[str, int]]:
    """Decompose centroid sequence into (direction, distance) segments.

    A new segment starts whenever the dominant axis or sign changes.
    Example: RIGHT 5, DOWN 3  →  L-shaped path.
    """
    segments: list[tuple[str, int]] = []
    cur_dir: str | None = None
    cur_dist: float = 0.0

    for i in range(1, len(valid)):
        r0, c0 = valid[i - 1][1]
        r1, c1 = valid[i][1]
        dr, dc = r1 - r0, c1 - c0
        if abs(dr) < 0.2 and abs(dc) < 0.2:
            continue  # sub-pixel jitter
        new_dir = ("DOWN" if dr > 0 else "UP") if abs(dr) >= abs(dc) \
                  else ("RIGHT" if dc > 0 else "LEFT")
        step_dist = abs(dr) if abs(dr) >= abs(dc) else abs(dc)

        if new_dir == cur_dir:
            cur_dist += step_dist
        else:
            if cur_dir is not None and cur_dist >= 0.5:
                segments.append((cur_dir, round(cur_dist)))
            cur_dir  = new_dir
            cur_dist = step_dist

    if cur_dir is not None and cur_dist >= 0.5:
        segments.append((cur_dir, round(cur_dist)))

    return segments


def _format_motion(
    color: int, name: str,
    from_pos: tuple[float, float],
    to_pos:   tuple[float, float],
    segments: list[tuple[str, int]],
    n_frames: int,
) -> str:
    fp = f"(row~{round(from_pos[0])}, col~{round(from_pos[1])})"
    tp = f"(row~{round(to_pos[0])},  col~{round(to_pos[1])})"
    if not segments:
        return f"color {color} ({name}): moved  {fp} -> {tp}  [{n_frames} frames]"
    path_str = " then ".join(f"{d} {dist} cells" for d, dist in segments)
    return f"color {color} ({name}): {fp} -> {tp}  [{path_str}, {n_frames} frames]"


def animation_to_text(frames: list[np.ndarray]) -> str:
    """Summarize animation motion as a single text block for LLM prompts.

    Includes frame count, per-object direction + distance + path, appear/disappear events.
    Returns a short string even for 0 or 1 frames.
    """
    n = len(frames)
    if n < 2:
        return f"animation: {n} frame(s) — no motion"

    records = analyze_animation(frames)
    lines = [f"animation: {n} frames"]
    if not records:
        lines.append("  no object movement detected")
    else:
        for r in records:
            lines.append(f"  {r['summary']}")
    return "\n".join(lines)


def grid_to_image(grid: np.ndarray, scale: int = 8):
    """Render a (H, W) int grid as an RGB PIL Image for VLM input.

    Each cell becomes a scale×scale pixel square using the ARC 16-color palette.
    A 64×64 grid at scale=8 produces a 512×512 PNG — the input size for Qwen2.5-VL.
    Values outside [0, 15] are clipped.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required for grid_to_image — pip install Pillow")

    if grid.ndim != 2:
        raise ValueError(f"expected 2D grid, got shape {grid.shape}")
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")

    h, w = grid.shape
    clipped = np.clip(grid, 0, 15).astype(np.uint8)

    # Build RGB array directly: shape (H*scale, W*scale, 3)
    palette = np.array(_ARC_PALETTE, dtype=np.uint8)  # (16, 3)
    # Map each cell to its RGB color, then repeat scale times in both axes
    rgb = palette[clipped]                             # (H, W, 3)
    rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)  # (H*s, W*s, 3)

    return Image.fromarray(rgb, mode="RGB")


# ── trace serialization (Stage 0 schema, see README §6) ──────────────────────


def serialize_step(
    *,
    step: int,
    game_id: str,
    level: int,
    state: str,
    image_path: str | None,
    prompt: str,
    response_raw: str,
    parse_ok: bool,
    predicted_diff: set[tuple[int, int, int]] | None,
    chosen_action: str | None,
    real_diff: set[tuple[int, int, int]] | None,
    f1: float | None,
) -> dict:
    """Build one trace.jsonl row with the fixed Stage 0 schema.

    All 12 fields must always be present in the output (None / null when data
    is missing). The schema is the only contract between writers and readers
    of `outputs/baseline_<ts>/<game_id>/trace.jsonl` and friends — extend
    here, never inline.

    Schema (every field always present in returned dict):
      step:           int            — episode step index, 0-based
      game_id:        str            — full game id (e.g. "ls20-9607627b")
      level:          int            — frame.levels_completed + 1
      state:          str            — frame.state.name
      image_path:     str | None     — relative to the run folder; None if not saved
      prompt:         str            — full text shown to the model
      response_raw:   str            — model output verbatim, before parsing
      parse_ok:       bool           — JSON parse + action validation succeeded
      predicted_diff: list[[r,c,c]] | None  — None when parse_ok=False
      chosen_action:  str | None     — e.g. "ACTION3"; None when parse_ok=False
      real_diff:      list[[r,c,c]] | None  — None when no env step taken
      f1:             float | None   — None when either diff is None

    `predicted_diff` and `real_diff` are sets of (row, col, new_color) for
    in-memory work; serialized here as sorted list of [row, col, new_color]
    triples (deterministic; JSON-friendly).
    """
    def _diff_to_list(d: set[tuple[int, int, int]] | None) -> list[list[int]] | None:
        if d is None:
            return None
        return sorted([[int(r), int(c), int(v)] for r, c, v in d])

    return {
        "step":           int(step),
        "game_id":        str(game_id),
        "level":          int(level),
        "state":          str(state),
        "image_path":     image_path if image_path is None else str(image_path),
        "prompt":         str(prompt),
        "response_raw":   str(response_raw),
        "parse_ok":       bool(parse_ok),
        "predicted_diff": _diff_to_list(predicted_diff),
        "chosen_action":  None if chosen_action is None else str(chosen_action),
        "real_diff":      _diff_to_list(real_diff),
        "f1":             None if f1 is None else float(f1),
    }
