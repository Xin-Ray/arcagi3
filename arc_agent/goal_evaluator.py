"""Deterministic goal-achievement evaluator (2026-05-18 v0).

Background: subtask T-GOAL bench (`docs/project/2026-05-18-v0-force_cot/`)
showed SmolLM3-3B answers near-random (long_acc 33.7% with 98/100 CoT
activated). Conclusion: even when the model DOES think, it cannot
reliably recognize "is the goal hypothesis achieved?". This module
replaces that recognition step with a deterministic parser.

Approach:
  1. Parse `goal_hypothesis` text into a structural `GoalPredicate`.
     Patterns we handle:
       - "align ... vertically in {left|right|first|last|leftmost|rightmost} column"
       - "align ... horizontally in {top|bottom|first|last} row"
       - "align ... vertically"          (same col, any col)
       - "align ... horizontally"        (same row, any row)
       - "move ... to column N" / "to col N"
       - "move ... to row N"
       - "stack X on top of Y"           (X.row == Y.row - 1, same col)
       - "place ... next to ..."         (adjacent cells)
  2. Evaluate predicate against the current object set.
  3. If we can't parse, return None -> caller falls back to LLM judgment.

This is NOT task-specific (per ARC Prize rules); it's a generic
NL-pattern -> coord-predicate parser. The same code runs on any game
whose Reflection Agent writes one of the recognized hypothesis patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from arc_agent.object_extractor import ObjectRecord, _COLOR_NAMES


_COLOR_NAMES_LOWER: set[str] = {v for v in _COLOR_NAMES.values()}

_COLUMN_KEYWORDS: dict[str, int] = {
    "left": 0,
    "leftmost": 0,
    "first": 0,
    "right": 63,
    "rightmost": 63,
    "last": 63,
}
_ROW_KEYWORDS: dict[str, int] = {
    "top": 0,
    "topmost": 0,
    "first": 0,
    "bottom": 63,
    "last": 63,
}

# 2026-05-18 v1: production hypothesis vocab discovered from v2 round 0
# trace. Reflection writes "to the top edge" / "to the center" / "align X
# and Y" without specifying axis. Map these to structural predicates.
_EDGE_TARGETS: dict[str, tuple[str, int]] = {
    "top edge":    ("move_to_row", 0),
    "topmost":     ("move_to_row", 0),
    "bottom edge": ("move_to_row", 63),
    "bottommost":  ("move_to_row", 63),
    "left edge":   ("move_to_col", 0),
    "leftmost":    ("move_to_col", 0),
    "right edge":  ("move_to_col", 63),
    "rightmost":   ("move_to_col", 63),
}

_GRID_MAX = 63  # 64x64 grid, indices 0..63
_GRID_CENTER = 31  # ≈ center of 64x64 grid
_CENTER_TOLERANCE = 3  # within +/- 3 cells of center counts as "centered"


# ── predicate types ────────────────────────────────────────────────────

@dataclass
class GoalPredicate:
    """Parsed structural form of a natural-language goal hypothesis."""

    kind: str
    """One of:
      align_col      — N+ objects share a column (optionally pinned to specific col)
      align_row      — N+ objects share a row    (optionally pinned to specific row)
      align_any      — N+ objects share EITHER same col OR same row (no axis hint)
      move_to_col    — object reaches a specific column
      move_to_row    — object reaches a specific row
      move_to_center — object(s) within +/- _CENTER_TOLERANCE of grid center
      stack          — X is directly above Y (same col, X.row == Y.row - 1)
      adjacent       — X is adjacent to Y (any 4-neighbor)
    """

    # Selector for objects involved -- the most common case is a single
    # color filter ("yellow squares"). When `colors` is non-empty, only
    # objects whose color_name is in this set participate.
    colors: tuple[str, ...] = ()
    """Color name filter (lowercase). Empty = all objects."""

    target: Optional[int] = None
    """For move_to_col / move_to_row / pinned align_*: the target index 0..63."""

    min_count: int = 2
    """Minimum number of qualifying objects required for align_* to evaluate.
    If fewer match, evaluator returns None (can't tell)."""

    raw: str = ""
    """Original hypothesis text, kept for debugging."""


def _select_objects(
    objects: Sequence[ObjectRecord],
    colors: Sequence[str],
) -> list[ObjectRecord]:
    """Filter object list by color name(s). Empty colors = all."""
    if not colors:
        return list(objects)
    colors_set = {c.lower() for c in colors}
    return [o for o in objects if o.color_name.lower() in colors_set]


def evaluate_predicate(
    pred: GoalPredicate,
    objects: Sequence[ObjectRecord],
) -> Optional[bool]:
    """Evaluate a parsed predicate against the current objects.

    Returns True if the goal is achieved, False if not, None if we don't
    have enough information (e.g., fewer than min_count matching objects).
    """
    selected = _select_objects(objects, pred.colors)

    if pred.kind == "align_col":
        if len(selected) < pred.min_count:
            return None
        # Take the center col of each
        cols = [int(round(o.center[1])) for o in selected]
        all_same = len(set(cols)) == 1
        if pred.target is not None:
            return all_same and cols[0] == pred.target
        return all_same

    if pred.kind == "align_any":
        # No axis specified: succeeds if all selected share SAME col OR
        # SAME row. (Captures "align tan and red" style hypothesis.)
        if len(selected) < pred.min_count:
            return None
        cols = [int(round(o.center[1])) for o in selected]
        rows = [int(round(o.center[0])) for o in selected]
        same_col = len(set(cols)) == 1
        same_row = len(set(rows)) == 1
        return same_col or same_row

    if pred.kind == "move_to_center":
        if not selected:
            return None
        # All selected must be within tolerance of grid center
        for o in selected:
            r = int(round(o.center[0]))
            c = int(round(o.center[1]))
            if abs(r - _GRID_CENTER) > _CENTER_TOLERANCE \
                    or abs(c - _GRID_CENTER) > _CENTER_TOLERANCE:
                return False
        return True

    if pred.kind == "align_row":
        if len(selected) < pred.min_count:
            return None
        rows = [int(round(o.center[0])) for o in selected]
        all_same = len(set(rows)) == 1
        if pred.target is not None:
            return all_same and rows[0] == pred.target
        return all_same

    if pred.kind == "move_to_col":
        if not selected:
            return None
        # Any selected object at target col is success (single-object goal usually)
        target = pred.target
        if target is None:
            return None
        return any(int(round(o.center[1])) == target for o in selected)

    if pred.kind == "move_to_row":
        if not selected:
            return None
        target = pred.target
        if target is None:
            return None
        return any(int(round(o.center[0])) == target for o in selected)

    if pred.kind == "stack":
        # Need >= 2 objects to compare. Pair-wise check: any (X, Y) where
        # X.col == Y.col and X.row == Y.row - 1.
        if len(selected) < 2:
            return None
        for i, a in enumerate(selected):
            for j, b in enumerate(selected):
                if i == j:
                    continue
                ac = (int(round(a.center[0])), int(round(a.center[1])))
                bc = (int(round(b.center[0])), int(round(b.center[1])))
                if ac[1] == bc[1] and ac[0] == bc[0] - 1:
                    return True
        return False

    if pred.kind == "adjacent":
        if len(selected) < 2:
            return None
        # Check any pair of cells in different objects are 4-neighbors
        for i, a in enumerate(selected):
            cells_a = set(map(tuple, a.cells))
            for j, b in enumerate(selected):
                if i >= j:
                    continue
                for (r, c) in cells_a:
                    if (r - 1, c) in map(tuple, b.cells):
                        return True
                    if (r + 1, c) in map(tuple, b.cells):
                        return True
                    if (r, c - 1) in map(tuple, b.cells):
                        return True
                    if (r, c + 1) in map(tuple, b.cells):
                        return True
        return False

    return None  # unknown kind


# ── parser ─────────────────────────────────────────────────────────────

_COLOR_LIST_RE = re.compile(
    r"\b(" + "|".join(sorted(_COLOR_NAMES_LOWER, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_COL_TARGET_NUM_RE = re.compile(r"\b(?:col|column)\s*[=:]?\s*(\d{1,2})\b", re.IGNORECASE)
_ROW_TARGET_NUM_RE = re.compile(r"\brow\s*[=:]?\s*(\d{1,2})\b", re.IGNORECASE)


def _extract_colors(text: str) -> tuple[str, ...]:
    """Pull all color words out of `text`, lowercased, deduped, in order seen."""
    seen: list[str] = []
    for m in _COLOR_LIST_RE.finditer(text):
        c = m.group(1).lower()
        if c not in seen:
            seen.append(c)
    return tuple(seen)


def _resolve_column_target(text: str) -> Optional[int]:
    """Look for 'col=N' / 'column N' OR a directional keyword (left/right)."""
    m = _COL_TARGET_NUM_RE.search(text)
    if m:
        try:
            v = int(m.group(1))
            if 0 <= v <= _GRID_MAX:
                return v
        except ValueError:
            pass
    low = text.lower()
    for kw, val in _COLUMN_KEYWORDS.items():
        # Match "leftmost column" / "left column" / "in left col" patterns
        if re.search(rf"\b{kw}\b\s+(?:column|col)\b", low):
            return val
    return None


def _resolve_row_target(text: str) -> Optional[int]:
    m = _ROW_TARGET_NUM_RE.search(text)
    if m:
        try:
            v = int(m.group(1))
            if 0 <= v <= _GRID_MAX:
                return v
        except ValueError:
            pass
    low = text.lower()
    for kw, val in _ROW_KEYWORDS.items():
        if re.search(rf"\b{kw}\b\s+row\b", low):
            return val
    return None


def parse_goal_hypothesis(text: str) -> Optional[GoalPredicate]:
    """Parse a natural-language goal hypothesis into a GoalPredicate.

    Returns None if no pattern matches -- caller should fall back to
    LLM-based judgment.

    Pattern priority (most specific first):
      1. "to the {top|bottom|left|right} edge"  -> move_to_row/col
      2. "to the center"                         -> move_to_center
      3. stack / on top of
      4. adjacent / next to
      5. align ... vertically [in ... column]
      6. align ... horizontally [in ... row]
      7. align X and Y (no axis)                -> align_any (NEW)
      8. move to col=N / row=N
      9. generic "in left column" without align
    """
    if not isinstance(text, str) or not text.strip():
        return None

    low = text.lower()
    colors = _extract_colors(text)

    # 1) "to the {top|bottom|left|right} edge" - the most common
    # production pattern (v2 round 0 trace step 0, 15; smoke 3 step 2
    # extended to "reach the X edge" / "must X the X edge" forms).
    _EDGE_PREP_VERBS = (
        "to the", "to",
        "toward the", "toward", "towards the", "towards",
        "reach the", "reach", "reaches the", "reaches",
        "reaching the", "reaching",
        "at the", "against the",
        "into the", "onto the",
        "must reach the", "should reach the",
    )
    for keyword, (kind, target) in _EDGE_TARGETS.items():
        for prep in _EDGE_PREP_VERBS:
            if f"{prep} {keyword}" in low:
                return GoalPredicate(
                    kind=kind, colors=colors, target=target,
                    min_count=1, raw=text)

    # 2) "to the center" / "towards the center" - production pattern
    # (v2 round 0 trace step 47, 56, 60, 75, 79; also "reach the
    # center" / "in the center" / "at the center")
    if re.search(
        r"\b(?:to|toward|towards|reach|reaches|reaching|at|in|into|onto)"
        r"\s+(?:the\s+)?(?:center|middle)\b", low):
        return GoalPredicate(
            kind="move_to_center", colors=colors, target=None,
            min_count=1, raw=text)

    # 3) stack / on top of
    if re.search(r"\bstack(?:ed)?\b.*\bon\s+top\s+of\b", low) \
            or re.search(r"\bon\s+top\s+of\b.*", low) and "align" not in low:
        return GoalPredicate(
            kind="stack", colors=colors, min_count=2, raw=text)

    # 4) adjacent / next to
    if re.search(r"\b(?:next\s+to|adjacent|touching)\b", low) and "align" not in low:
        return GoalPredicate(
            kind="adjacent", colors=colors, min_count=2, raw=text)

    # 5) align ... vertically [in ... column]
    if re.search(r"\balign(?:ed|ing)?\b.*\bvertical(?:ly)?\b", low) \
            or re.search(r"\bvertical(?:ly)?\b.*\balign", low):
        target = _resolve_column_target(text)
        return GoalPredicate(
            kind="align_col", colors=colors, target=target,
            min_count=2, raw=text)

    # 6) align ... horizontally [in ... row]
    if re.search(r"\balign(?:ed|ing)?\b.*\bhorizontal(?:ly)?\b", low) \
            or re.search(r"\bhorizontal(?:ly)?\b.*\balign", low):
        target = _resolve_row_target(text)
        return GoalPredicate(
            kind="align_row", colors=colors, target=target,
            min_count=2, raw=text)

    # 7) NEW: "align X and Y" without axis - production writes
    # "align the tan objects #7 and #8" or "align the purple objects"
    # without specifying vertical/horizontal. v2 round 0 trace step 17, 75.
    # Fall back to align_any (same col OR same row).
    if re.search(r"\balign(?:ed|ing)?\b", low):
        return GoalPredicate(
            kind="align_any", colors=colors, target=None,
            min_count=2, raw=text)

    # 7b) NEW (2026-05-19 Phase 1B): "match X with/to Y" pattern.
    # Reflection in /no_think mode writes things like:
    #   "match every yellow 1x1 with a yellow target square"
    #   "match the moving X to the static Y"
    # Semantic: two same-color objects should co-locate (overlap or be
    # adjacent). We approximate as align_any -- if all named-color objs
    # share col OR row, treat as achieved.
    if re.search(r"\bmatch(?:es|ed|ing)?\b", low):
        return GoalPredicate(
            kind="align_any", colors=colors, target=None,
            min_count=2, raw=text)

    # 8) move to column N
    if re.search(r"\bmove\b.*\b(?:col|column)\b", low):
        target = _resolve_column_target(text)
        if target is not None:
            return GoalPredicate(
                kind="move_to_col", colors=colors, target=target,
                min_count=1, raw=text)

    # move to row N
    if re.search(r"\bmove\b.*\brow\b", low):
        target = _resolve_row_target(text)
        if target is not None:
            return GoalPredicate(
                kind="move_to_row", colors=colors, target=target,
                min_count=1, raw=text)

    # 9) generic "in left column" / "in col 0" without 'align' keyword
    target_col = _resolve_column_target(text)
    if target_col is not None and ("column" in low or "col" in low):
        return GoalPredicate(
            kind="align_col" if (colors and len(colors) >= 1) else "move_to_col",
            colors=colors, target=target_col, min_count=1, raw=text)

    return None


def evaluate_goal(
    goal_hypothesis: str,
    objects: Sequence[ObjectRecord],
) -> tuple[Optional[bool], Optional[GoalPredicate]]:
    """Top-level convenience: parse + evaluate.

    Returns (achieved, predicate). `achieved` is None if we couldn't
    parse OR couldn't evaluate (e.g. not enough matching objects).
    `predicate` is the parsed form (or None if parse failed) -- useful
    for logging.
    """
    pred = parse_goal_hypothesis(goal_hypothesis)
    if pred is None:
        return None, None
    return evaluate_predicate(pred, objects), pred


__all__ = [
    "GoalPredicate",
    "parse_goal_hypothesis",
    "evaluate_predicate",
    "evaluate_goal",
]
