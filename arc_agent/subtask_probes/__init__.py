"""Synthetic probe generators for the 7 ar25-completion subtasks.

Per `docs/project/2026-05-17-v0-subtask_decomp/architecture.md`.

Each generator:
  - Takes a seeded random for reproducibility
  - Returns list[dict] with same shape as bench_probes:
    {id, cat, q, options: {A,B,C,D}, correct: letter}
  - Calibrated to ar25 / G_base mechanics:
      1 cell per ACTION
      ACTION1=UP (row decreases), ACTION2=DOWN (row increases)
      ACTION3=LEFT (col decreases), ACTION4=RIGHT (col increases)
      ACTION5 cycles to next selectable object
      ACTION6 click x y
      ACTION7 undo

Probe IDs use "<SUBTASK>-<NNNN>" so they can be filtered.

This module only generates probes; running them is in
`scripts/bench_subtask.py`.
"""
from __future__ import annotations

import random
from typing import Optional


_ACTIONS_DIR = {
    "ACTION1": ("UP",    "row decreases"),
    "ACTION2": ("DOWN",  "row increases"),
    "ACTION3": ("LEFT",  "col decreases"),
    "ACTION4": ("RIGHT", "col increases"),
}


def gen_T_NAV_1(n: int = 100, seed: int = 42) -> list[dict]:
    """T-NAV-1 single-step navigation.

    Q: Object at (r1, c1). Target at (r2, c2) where exactly one of
       Δrow / Δcol is non-zero and small (1..6).
    Options: 4 directions (A/B/C/D = ACTION1/2/3/4 in some shuffled order).
    Correct: the ACTION whose direction reduces the active delta.
    """
    rng = random.Random(seed)
    probes: list[dict] = []
    for i in range(n):
        r1 = rng.randint(5, 58)
        c1 = rng.randint(5, 58)
        # Pick whether vertical or horizontal move
        axis = rng.choice(["vert", "horiz"])
        if axis == "vert":
            delta = rng.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
            r2, c2 = r1 + delta, c1
            correct_action = "ACTION1" if delta < 0 else "ACTION2"
        else:
            delta = rng.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
            r2, c2 = r1, c1 + delta
            correct_action = "ACTION3" if delta < 0 else "ACTION4"

        # Build 4 options, ordered ACTION1..4, then shuffle letters
        action_list = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
        rng.shuffle(action_list)
        letters = ["A", "B", "C", "D"]
        opts = {}
        correct_letter = None
        for L, act in zip(letters, action_list):
            name, descr = _ACTIONS_DIR[act]
            opts[L] = f"{act} ({name}, {descr})"
            if act == correct_action:
                correct_letter = L

        q = (
            f"Grid is 64x64 (rows 0-63, cols 0-63). "
            f"ACTION1=UP (row-1), ACTION2=DOWN (row+1), "
            f"ACTION3=LEFT (col-1), ACTION4=RIGHT (col+1). "
            f"Active object at (row={r1}, col={c1}). "
            f"Target at (row={r2}, col={c2}). "
            f"Which single ACTION reduces the distance most?"
        )
        probes.append({
            "id": f"T-NAV-1-{i:04d}",
            "cat": "T-NAV-1",
            "q": q,
            "options": opts,
            "correct": correct_letter,
            # Metadata for debugging
            "_meta": {
                "object": [r1, c1], "target": [r2, c2],
                "correct_action": correct_action,
            },
        })
    return probes


def gen_T_NAV_2(n: int = 100, seed: int = 42) -> list[dict]:
    """T-NAV-2 linear multi-step: how many ACTIONs to traverse N cells.

    Q: object at A, target at B on the same row OR col, distance N.
    Options: 4 candidate counts (one correct, three off by ±1..2 or wrong action).
    """
    rng = random.Random(seed + 1)
    probes: list[dict] = []
    for i in range(n):
        r1 = rng.randint(10, 53)
        c1 = rng.randint(10, 53)
        n_steps = rng.randint(2, 10)
        axis = rng.choice(["vert", "horiz"])
        if axis == "vert":
            direction = rng.choice([-1, 1])
            r2, c2 = r1 + direction * n_steps, c1
            correct_action = "ACTION1" if direction < 0 else "ACTION2"
        else:
            direction = rng.choice([-1, 1])
            r2, c2 = r1, c1 + direction * n_steps
            correct_action = "ACTION3" if direction < 0 else "ACTION4"

        # 4 options: correct count + 3 distractors
        distractors = {n_steps - 1, n_steps + 1, n_steps * 2, max(1, n_steps - 2)}
        distractors.discard(n_steps)
        distractor_list = list(distractors)[:3]
        candidates = distractor_list + [n_steps]
        rng.shuffle(candidates)
        letters = ["A", "B", "C", "D"]
        opts = {}
        correct_letter = None
        for L, c in zip(letters, candidates):
            opts[L] = f"{c} times"
            if c == n_steps:
                correct_letter = L

        q = (
            f"ACTION1=UP, ACTION2=DOWN, ACTION3=LEFT, ACTION4=RIGHT (each moves 1 cell). "
            f"Active object at (row={r1}, col={c1}). "
            f"Target at (row={r2}, col={c2}). "
            f"How many times must you press {correct_action} to reach the target?"
        )
        probes.append({
            "id": f"T-NAV-2-{i:04d}",
            "cat": "T-NAV-2",
            "q": q,
            "options": opts,
            "correct": correct_letter,
            "_meta": {"n_steps": n_steps, "correct_action": correct_action},
        })
    return probes


def gen_T_NAV_3(n: int = 100, seed: int = 42) -> list[dict]:
    """T-NAV-3 L-shape: requires BOTH row AND col movement.

    Q: object at A, target at B with Δrow ≠ 0 AND Δcol ≠ 0.
    Options: A) only vertical; B) only horizontal; C) correct (vert + horiz);
             D) wrong direction count.
    """
    rng = random.Random(seed + 2)
    probes: list[dict] = []
    for i in range(n):
        r1 = rng.randint(10, 53)
        c1 = rng.randint(10, 53)
        dr = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5])
        dc = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5])
        r2, c2 = r1 + dr, c1 + dc
        n_vert = abs(dr)
        n_horiz = abs(dc)
        vert_action = "ACTION1" if dr < 0 else "ACTION2"
        horiz_action = "ACTION3" if dc < 0 else "ACTION4"

        options_pool = [
            f"{n_vert} {vert_action} only (wrong: misses col change)",
            f"{n_horiz} {horiz_action} only (wrong: misses row change)",
            f"{n_vert} {vert_action} + {n_horiz} {horiz_action} ({n_vert+n_horiz} total)",
            f"{n_vert+1} {vert_action} + {n_horiz} {horiz_action} ({n_vert+n_horiz+1} total, off-by-one)",
        ]
        correct_idx = 2
        # Shuffle while tracking correct
        indices = [0, 1, 2, 3]
        rng.shuffle(indices)
        letters = ["A", "B", "C", "D"]
        opts = {}
        correct_letter = None
        for L, idx in zip(letters, indices):
            opts[L] = options_pool[idx]
            if idx == correct_idx:
                correct_letter = L

        q = (
            f"ACTION1=UP, ACTION2=DOWN, ACTION3=LEFT, ACTION4=RIGHT (each 1 cell). "
            f"Active object at (row={r1}, col={c1}). "
            f"Target at (row={r2}, col={c2}). "
            f"Which is the minimum action plan?"
        )
        probes.append({
            "id": f"T-NAV-3-{i:04d}",
            "cat": "T-NAV-3",
            "q": q,
            "options": opts,
            "correct": correct_letter,
            "_meta": {"dr": dr, "dc": dc, "n_total": n_vert + n_horiz},
        })
    return probes


def gen_T_SEL_1(n: int = 100, seed: int = 42) -> list[dict]:
    """T-SEL-1 ACTION6 click hit-test.

    Q: 4 objects with bboxes. Given a target bbox, which (x, y) selects it?
    Options: 4 (x, y) pairs, only one inside the target bbox.
    """
    rng = random.Random(seed + 3)
    probes: list[dict] = []
    for i in range(n):
        # Make 4 non-overlapping bboxes
        regions = []
        for _ in range(4):
            tries = 0
            while tries < 20:
                r1 = rng.randint(0, 55)
                c1 = rng.randint(0, 55)
                h = rng.randint(2, 8)
                w = rng.randint(2, 8)
                r2 = min(63, r1 + h)
                c2 = min(63, c1 + w)
                # Check no overlap
                overlaps = any(
                    not (c2 < rg[1] or rg[3] < c1 or r2 < rg[0] or rg[2] < r1)
                    for rg in regions
                )
                if not overlaps:
                    regions.append((r1, c1, r2, c2))
                    break
                tries += 1
            if tries >= 20:
                break
        if len(regions) < 4:
            continue  # skip rare bad seeds

        # Pick target = first region
        target_idx = rng.randint(0, 3)
        target = regions[target_idx]
        # Generate one coord inside target, three inside others (or outside all)
        def _pick_inside(rg):
            (r1, c1, r2, c2) = rg
            return (rng.randint(r1, r2), rng.randint(c1, c2))
        correct_pt = _pick_inside(target)
        wrong_pts = [_pick_inside(regions[j]) for j in range(4) if j != target_idx][:3]

        all_pts = [correct_pt] + wrong_pts
        rng.shuffle(all_pts)
        letters = ["A", "B", "C", "D"]
        opts = {}
        correct_letter = None
        for L, pt in zip(letters, all_pts):
            opts[L] = f"ACTION6 x={pt[1]} y={pt[0]}"
            if pt == correct_pt:
                correct_letter = L

        # Describe objects
        def _desc(rg, idx):
            return f"Object {chr(65+idx)}: rows {rg[0]}-{rg[2]}, cols {rg[1]}-{rg[3]}"
        obj_descs = "\n  ".join(_desc(rg, j) for j, rg in enumerate(regions))
        target_name = chr(65 + target_idx)

        q = (
            f"ACTION6 takes coords (x=col, y=row) and selects whichever object "
            f"covers that cell.\n  {obj_descs}\n"
            f"Which ACTION6 selects Object {target_name}?"
        )
        probes.append({
            "id": f"T-SEL-1-{i:04d}",
            "cat": "T-SEL-1",
            "q": q,
            "options": opts,
            "correct": correct_letter,
            "_meta": {"target_bbox": target, "target_name": target_name},
        })
    return probes


def gen_T_GOAL(n: int = 100, seed: int = 42) -> list[dict]:
    """T-GOAL goal-state recognition.

    Q: Goal = "align two yellow squares vertically in left column".
    Given current positions of obj_A (yellow) and obj_B (yellow),
    answer YES/NO whether goal is achieved.
    Multi-choice with 4 options: YES same-col / YES same-row /
    NO different col / NO different row.
    """
    rng = random.Random(seed + 4)
    probes: list[dict] = []
    target_col = 0  # "left column"
    for i in range(n):
        success = rng.choice([True, False, False, False])  # 25% success
        if success:
            r1 = rng.randint(0, 30)
            r2 = rng.randint(31, 60)
            c1 = target_col
            c2 = target_col
        else:
            r1 = rng.randint(0, 30)
            r2 = rng.randint(31, 60)
            mode = rng.choice(["wrong_col_one", "wrong_col_both",
                               "wrong_row_overlap", "off_by_one"])
            if mode == "wrong_col_one":
                c1 = target_col
                c2 = rng.randint(2, 60)
            elif mode == "wrong_col_both":
                c1 = rng.randint(2, 60)
                c2 = rng.randint(2, 60)
            elif mode == "wrong_row_overlap":
                c1 = target_col
                c2 = target_col
                r2 = r1  # both on same row, not vertical alignment
            else:  # off_by_one
                c1 = target_col
                c2 = 1  # adjacent column, not target

        # 4-option answer
        if success:
            correct_idx = 0  # YES, goal achieved
        else:
            correct_idx = rng.randint(1, 3)
        options_pool = [
            "YES — both yellow objects are in the left column and vertically arranged",
            "NO — at least one object is not in the left column",
            "NO — objects are on the same row, not vertically arranged",
            "NO — objects are in different columns",
        ]
        # Shuffle
        indices = [0, 1, 2, 3]
        rng.shuffle(indices)
        letters = ["A", "B", "C", "D"]
        opts = {}
        correct_letter = None
        for L, idx in zip(letters, indices):
            opts[L] = options_pool[idx]
            if idx == correct_idx:
                correct_letter = L

        q = (
            f"Goal: 'align the two yellow squares vertically in the left column (col=0)'.\n"
            f"Current state:\n"
            f"  obj_A (yellow): row={r1}, col={c1}\n"
            f"  obj_B (yellow): row={r2}, col={c2}\n"
            f"Is the goal achieved?"
        )
        probes.append({
            "id": f"T-GOAL-{i:04d}",
            "cat": "T-GOAL",
            "q": q,
            "options": opts,
            "correct": correct_letter,
            "_meta": {"success": success, "pos_A": [r1, c1], "pos_B": [r2, c2]},
        })
    return probes


GENERATORS: dict[str, callable] = {
    "T-NAV-1": gen_T_NAV_1,
    "T-NAV-2": gen_T_NAV_2,
    "T-NAV-3": gen_T_NAV_3,
    "T-SEL-1": gen_T_SEL_1,
    "T-GOAL": gen_T_GOAL,
    # T-SEL-2 (ACTION5 switch) and T-RETRY (meta-recovery) TBD;
    # they need richer context that's harder to synthesize cleanly.
}


__all__ = ["GENERATORS", "gen_T_NAV_1", "gen_T_NAV_2",
           "gen_T_NAV_3", "gen_T_SEL_1", "gen_T_GOAL"]
