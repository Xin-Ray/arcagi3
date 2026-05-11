"""Game-id splits and run-summary writer (the data layer for Stage 0).

Two responsibilities, both pure / no I/O surprises:

1. `demo_555_split(game_ids)` — deterministic alphabetical 5-5-5 split of the
   demo set into G_base / G_train / G_val + holdout. Same input always
   produces the same output; the *result* is then frozen into
   `data/splits/demo_555.json` by `scripts/freeze_splits.py` and committed.
   Downstream scripts read the JSON, never re-derive.

2. `write_summary(out_path, **fields)` — write one `summary.json` per run
   (baseline / validation / grpo) with a fixed schema, so multiple runs are
   directly comparable across `summary["mean_f1"]` etc.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def demo_555_split(game_ids: list[str]) -> dict[str, Any]:
    """Split a list of game IDs into G_base / G_train / G_val / holdout.

    Method: sort the IDs alphabetically, then slice
        [0:5]  -> g_base
        [5:10] -> g_train
        [10:15]-> g_val
        [15:]  -> holdout

    Reproducible (same input list, same output split).

    Raises ValueError if fewer than 15 unique IDs are provided (the three
    5-game groups must each be filled). The 10 holdout games may be empty
    when the input has exactly 15 items; with the standard demo-25 set
    they are the trailing 10.
    """
    if not isinstance(game_ids, list):
        raise TypeError(f"game_ids must be a list, got {type(game_ids).__name__}")
    unique = sorted(set(game_ids))
    if len(unique) < 15:
        raise ValueError(
            f"need >= 15 unique game_ids for 5-5-5 split, got {len(unique)}"
        )
    return {
        "method": "alphabetical sort; slices [0:5]/[5:10]/[10:15]/[15:]",
        "n_input": len(unique),
        "g_base":  unique[0:5],
        "g_train": unique[5:10],
        "g_val":   unique[10:15],
        "holdout": unique[15:],
    }


_SUMMARY_REQUIRED: frozenset[str] = frozenset({
    "run_kind",            # "baseline" | "validation" | "grpo"
    "games",               # list[str] of game_ids actually run
    "n_episodes_per_game", # int
    "wall_clock_seconds",  # float
    "mean_f1",             # float in [0, 1]
    "parse_rate",          # float in [0, 1]
    "mean_rhae",           # float in [0, 1]
    "per_game",            # dict[str, dict] keyed by game_id
})


def write_summary(out_path: Path | str, **fields: Any) -> dict[str, Any]:
    """Write a run-summary JSON with the fixed schema and return the payload.

    Required fields (caller MUST pass all of these as kwargs):
      run_kind, games, n_episodes_per_game, wall_clock_seconds,
      mean_f1, parse_rate, mean_rhae, per_game

    Auto-filled by this function:
      schema_version: "1"
      run_ts: ISO-8601 UTC timestamp at write time

    Optional (caller may pass; persisted as-is if present):
      split, git_commit, notes, hypothesis, branch_decision

    The resulting JSON is the only contract for cross-run comparison —
    don't add fields ad-hoc in calling code; extend the schema here first.
    """
    missing = _SUMMARY_REQUIRED - fields.keys()
    if missing:
        raise ValueError(
            f"write_summary missing required fields: {sorted(missing)}"
        )
    payload: dict[str, Any] = {
        "schema_version": "1",
        "run_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **fields,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
