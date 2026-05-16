"""Build (state, action, label) tuples from existing trace.jsonl files.

Self-supervised: label = `frame_changed` (or `bool(real_diff)` for older
schemas). No human annotation.

Two functions:

  scan_traces(root, schema='auto') -> list[Sample]
      Walk every trace.jsonl under `root` and return one Sample per step.
      Drops rows without a parseable action.

  build_dataset(samples, action_filter=None, balance='none')
      Optionally drop degenerate (game, action) pairs (every-changed or
      never-changed) and balance pos/neg per action. Returns the same list
      filtered.

Sample is intentionally minimal - we keep `step_image_path` so the feature
extractor can reach the rendered grid if it needs to (most features come
from non-grid fields).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional


@dataclass(frozen=True)
class Sample:
    """One step worth of (state, action, label).

    `game_id` is the full id ("ar25-0c556536"); shortened later for one-hot.
    `step_image_path` may be None when the trace didn't dump per-step PNGs.
    """
    game_id: str
    step: int
    action: str
    action_coords: Optional[tuple[int, int]]    # (x, y) when action=ACTION6
    label: int                                  # 1 if frame_changed else 0
    step_image_path: Optional[Path]
    no_op_streak: int = 0
    state_revisit_count: int = 1
    primary_direction: Optional[str] = None
    primary_distance: int = 0


def _short_game(gid: str) -> str:
    """Drop the per-run hash suffix: ar25-0c556536 -> ar25."""
    return gid.split("-", 1)[0] if "-" in gid else gid


def _parse_row(row: dict, trace_dir: Path) -> Optional[Sample]:
    """Map one trace.jsonl row to a Sample, or None if unparseable."""
    game_id = str(row.get("game_id", "")) or "unknown"
    step = int(row.get("step", -1))
    if step < 0:
        return None

    # Schema dispatch: new v3.2 uses `action` + `frame_changed`; older
    # v3 / ablation use `chosen_action` + `real_diff` (list of cells).
    if "frame_changed" in row:
        action = str(row.get("action") or "?")
        label = 1 if row["frame_changed"] else 0
        coords = row.get("action_coords")
        if coords is not None and isinstance(coords, (list, tuple)) and len(coords) == 2:
            coords = (int(coords[0]), int(coords[1]))
        else:
            coords = None
    else:
        action = str(row.get("chosen_action") or "?")
        diff = row.get("real_diff") or []
        label = 1 if diff else 0
        coords = None

    if action == "?" or action == "RESET":
        return None

    img_rel = row.get("image_path")
    img_path: Optional[Path] = None
    if isinstance(img_rel, str) and img_rel:
        candidate = trace_dir / img_rel
        if candidate.exists():
            img_path = candidate
    # v3.2 schema: per-step PNGs are step_NNNN.png next to trace.jsonl
    if img_path is None:
        cand = trace_dir / f"step_{step:04d}.png"
        if cand.exists():
            img_path = cand

    return Sample(
        game_id=game_id,
        step=step,
        action=action,
        action_coords=coords,
        label=label,
        step_image_path=img_path,
        no_op_streak=int(row.get("no_op_streak", 0)),
        state_revisit_count=int(row.get("state_revisit_count", 1)),
        primary_direction=row.get("primary_direction"),
        primary_distance=int(row.get("primary_distance", 0) or 0),
    )


def scan_traces(root: Path) -> list[Sample]:
    """Walk every trace.jsonl under `root` and return per-step Samples."""
    samples: list[Sample] = []
    for tp in sorted(root.rglob("trace.jsonl")):
        try:
            for line in tp.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s = _parse_row(row, tp.parent)
                if s is not None:
                    samples.append(s)
        except OSError:
            continue
    return samples


def _per_action_balance(
    samples: list[Sample],
    seed: int = 42,
    drop_degenerate_threshold: float = 0.02,
) -> list[Sample]:
    """For each action, downsample the majority class to match the minority.

    Drops actions whose minority class is < `drop_degenerate_threshold` of
    total (e.g. ACTION2 in our data: 203 changed / 2 no-op -> drop).
    """
    import random as _rnd
    rng = _rnd.Random(seed)

    by_action: dict[str, dict[int, list[Sample]]] = {}
    for s in samples:
        by_action.setdefault(s.action, {0: [], 1: []})[s.label].append(s)

    out: list[Sample] = []
    for action, classes in by_action.items():
        pos, neg = len(classes[1]), len(classes[0])
        total = pos + neg
        if total == 0:
            continue
        minority = min(pos, neg)
        if minority / total < drop_degenerate_threshold:
            continue  # degenerate: classifier would be trivial
        keep = minority
        rng.shuffle(classes[1])
        rng.shuffle(classes[0])
        out.extend(classes[1][:keep])
        out.extend(classes[0][:keep])
    rng.shuffle(out)
    return out


def build_dataset(
    samples: list[Sample],
    balance: Literal["none", "per_action"] = "per_action",
    action_filter: Optional[Iterable[str]] = None,
    seed: int = 42,
) -> list[Sample]:
    """Apply optional filtering + class-balancing."""
    if action_filter is not None:
        allowed = set(action_filter)
        samples = [s for s in samples if s.action in allowed]
    if balance == "per_action":
        return _per_action_balance(samples, seed=seed)
    return list(samples)
