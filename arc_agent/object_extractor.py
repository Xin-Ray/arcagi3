"""Deterministic per-frame object extraction (replaces Qwen-VL extract).

An "object" is a maximal set of cells with the same color that are
4-connected (share an edge). Black (color 0) is the background and is
NOT extracted.

Designed per `docs/OBJECT_PIPELINE_DESIGN_zh.md` §9.1 — `scipy.ndimage.label`
applied per non-background color, with each connected component becoming
one `ObjectRecord`. Output schema matches the Qwen-extract JSON spec so
downstream code (alignment, prompt rendering) is interchangeable.

This module has no LLM dependency and runs in microseconds per frame.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# ARC palette — kept in sync with arc_agent.observation._COLOR_NAMES.
_COLOR_NAMES: dict[int, str] = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "gray", 6: "magenta", 7: "orange", 8: "light blue", 9: "maroon",
    10: "purple", 11: "tan", 12: "teal", 13: "lime", 14: "rose", 15: "navy",
}

BACKGROUND_COLOR = 0


@dataclass
class ObjectRecord:
    """One extracted object. Fields mirror Qwen's extract JSON spec."""

    id: int
    color: int
    color_name: str
    cells: list[tuple[int, int]] = field(default_factory=list)
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    center: tuple[float, float] = (0.0, 0.0)
    size: int = 0
    shape_signature: tuple[tuple[int, int], ...] = ()
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Tuples -> lists for JSON friendliness; shape_signature stays tuple-of-tuple
        d["cells"] = [list(c) for c in self.cells]
        d["bbox"] = list(self.bbox)
        d["center"] = [round(self.center[0], 2), round(self.center[1], 2)]
        d["shape_signature"] = [list(c) for c in self.shape_signature]
        return d


def _shape_signature(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Translate cells to origin (min row/col) and sort lex.

    Two objects with the same `shape_signature` have identical shape under
    pure translation (rotation / reflection still produce different
    signatures — that's the §6 v0 limitation).
    """
    min_r = min(c[0] for c in cells)
    min_c = min(c[1] for c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _describe(size: int, bbox: tuple[int, int, int, int]) -> str:
    """One-phrase human-readable summary of the object's shape."""
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if size == 1:
        return "single cell"
    if size == h * w:
        if h == 1 and w > 1:
            return f"{w}x1 horizontal bar"
        if w == 1 and h > 1:
            return f"{h}x1 vertical bar"
        if h == w:
            return f"{h}x{w} solid square"
        return f"{h}x{w} solid rectangle"
    return f"irregular ({size} cells in {h}x{w} bbox)"


def extract_objects(grid: np.ndarray, *,
                    background: int = BACKGROUND_COLOR) -> list[ObjectRecord]:
    """Return all 4-connected non-background objects in a 2D grid.

    Args:
        grid: shape (H, W), integer values in 0..15.
        background: color id treated as empty (default 0).

    Returns:
        List of ObjectRecord, ids assigned in extraction order
        (color-major, then label order from scipy).
    """
    from scipy.ndimage import label

    if grid.ndim != 2:
        raise ValueError(f"expected 2D grid, got shape {grid.shape}")

    # 4-connectivity structuring element
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)

    out: list[ObjectRecord] = []
    next_id = 0
    unique_colors = sorted(int(c) for c in np.unique(grid) if int(c) != background)

    for color in unique_colors:
        mask = (grid == color).astype(np.int32)
        labeled, n = label(mask, structure=struct)
        if n == 0:
            continue
        for comp in range(1, n + 1):
            ys, xs = np.where(labeled == comp)
            cells = [(int(r), int(c)) for r, c in zip(ys, xs)]
            r_min, r_max = int(ys.min()), int(ys.max())
            c_min, c_max = int(xs.min()), int(xs.max())
            size = len(cells)
            center = (float(ys.mean()), float(xs.mean()))
            bbox = (r_min, c_min, r_max, c_max)
            obj = ObjectRecord(
                id=next_id,
                color=color,
                color_name=_COLOR_NAMES.get(color, f"color-{color}"),
                cells=cells,
                bbox=bbox,
                center=center,
                size=size,
                shape_signature=_shape_signature(cells),
                description=_describe(size, bbox),
            )
            out.append(obj)
            next_id += 1
    return out


def objects_to_dict(objects: list[ObjectRecord]) -> dict[str, Any]:
    """JSON-friendly wrapper matching the Qwen extract output shape."""
    return {"objects": [o.as_dict() for o in objects]}
