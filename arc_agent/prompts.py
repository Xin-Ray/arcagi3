"""Prompt constants for the Qwen object-diagnosis pipeline.

These are the exact strings fed to Qwen2.5-VL during EXP-1 (extract) and
EXP-2 (align). Kept here (not inlined in scripts) so any retuning is
one-file edit, and the document `docs/OBJECT_PIPELINE_DESIGN_zh.md` §3
stays the single source of truth — change here, re-cite there.

Two pairs (system, user) are exported:
  EXTRACT_SYSTEM / EXTRACT_USER  -- per-frame object extraction
  ALIGN_SYSTEM   / ALIGN_USER    -- BEFORE/AFTER pair alignment
"""
from __future__ import annotations


EXTRACT_SYSTEM = """You are looking at a single frame from a turn-based grid game. The image
shows a 64x64 grid rendered with one of 16 colors per cell:

  0 = black (background)
  1 = blue
  2 = red
  3 = green
  4 = yellow
  5 = gray
  6 = magenta
  7 = orange
  8 = light blue
  9 = maroon
  10 = purple
  11 = tan
  12 = teal
  13 = lime
  14 = rose
  15 = navy

An "object" is a connected group of cells with the same color, where
"connected" means cells share an edge (up/down/left/right, NOT diagonal).
Black cells (color 0) are background -- do NOT include them as objects.

Your job: identify every object in the image and return strict JSON."""


EXTRACT_USER = """List every object in this frame. For each object, give:
  - id:          integer index starting at 0
  - color:       integer 1-15 (skip 0/black/background)
  - color_name:  one word from the palette above
  - bbox:        [row_min, col_min, row_max, col_max]
                 (row=y, col=x, 0-indexed, both endpoints inclusive)
  - size:        number of cells in the object
  - description: a short human-readable phrase, e.g. "L-shape",
                 "3x1 horizontal bar", "single cell"

Output strict JSON only -- no prose, no markdown fences:

{
  "objects": [
    {"id": 0, "color": 2, "color_name": "red",
     "bbox": [10,3,12,5], "size": 5, "description": "L-shape"}
  ]
}"""


ALIGN_SYSTEM = """You are looking at TWO consecutive frames (BEFORE and AFTER) from a
turn-based grid game. Both frames use the same 64x64 grid with the
16-color palette (black=background; same palette as the extraction task).

The first image is BEFORE, the second image is AFTER. Some objects may
have moved, changed color, changed shape, appeared, or disappeared
between the two frames.

Match types:
  - unchanged:    object exists in both frames at same position with
                  same color and shape
  - moved:        same shape and color, different position
  - recolored:    same shape and position, different color
  - reshaped:     overlapping position, same color, different shape
                  (a few cells added/removed)
  - disappeared:  object in BEFORE has no plausible match in AFTER
  - appeared:     object in AFTER did not exist in BEFORE

Objects are 4-connected same-color groups (same definition as the
per-frame extractor)."""


ALIGN_USER = """For every object in BEFORE find its match in AFTER (or mark it
disappeared). For every NEW object in AFTER that has no match in BEFORE,
mark it appeared.

Output strict JSON only -- no prose, no markdown fences:

{
  "matches": [
    {"before_id": 0, "after_id": 2, "type": "moved",
     "color": 2, "delta": {"dy": -1, "dx": 0}},
    {"before_id": 1, "after_id": null, "type": "disappeared",
     "color": 3, "delta": null},
    {"before_id": null, "after_id": 4, "type": "appeared",
     "color": 4, "delta": null}
  ]
}

Rules for delta:
  - moved:     {"dy": int, "dx": int}        row/col displacement of centroid
  - recolored: {"from": int, "to": int}      color change
  - reshaped:  {"cells_added": int, "cells_removed": int}
  - unchanged / disappeared / appeared: null

before_id and after_id refer to indices in each frame's extraction list.
Re-extract objects implicitly per frame -- you do NOT need to match the
exact ids of a previous extraction call."""


__all__ = [
    "EXTRACT_SYSTEM",
    "EXTRACT_USER",
    "ALIGN_SYSTEM",
    "ALIGN_USER",
]
