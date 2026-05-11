"""Freeze the demo-25 5-5-5 game ID split into data/splits/demo_555.json.

Run ONCE per repo (or whenever the SDK exposes a different demo set). The
output JSON is committed; downstream scripts (run_baseline.py / run_grpo.py
/ run_validation.py) read it instead of re-deriving the split, so the
G_base / G_train / G_val groups stay identical across all runs and dates.

Usage:
    .venv/Scripts/python.exe scripts/freeze_splits.py

Requires a real ARC_API_KEY in .env (https://arcprize.org/api-keys).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from arc_agi import Arcade

from arc_agent.eval_split import demo_555_split

SPLIT_PATH = REPO_ROOT / "data" / "splits" / "demo_555.json"


def main() -> None:
    key = os.getenv("ARC_API_KEY", "")
    if not key or key.startswith("your_"):
        raise RuntimeError(
            "ARC_API_KEY missing or still the .env.example placeholder. "
            "Put a real key in .env (see README.md §3)."
        )

    arc = Arcade()
    envs = arc.get_environments() or []
    game_ids = sorted({e.game_id for e in envs})
    print(f"Available demo games from API: {len(game_ids)}")

    split = demo_555_split(game_ids)
    split["frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    split["source"] = "arc.get_environments() filtered to demo set"

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(
        json.dumps(split, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Frozen -> {SPLIT_PATH}")
    for k in ("g_base", "g_train", "g_val", "holdout"):
        print(f"  {k} ({len(split[k])}): {split[k]}")
    print("\nNext: git add data/splits/demo_555.json && commit it.")


if __name__ == "__main__":
    main()
