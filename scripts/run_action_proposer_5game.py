"""Run action_proposer v0 on G_base 5 games × 2 round × 300 step each.

Wraps `scripts/run_v3_multi_round.py` to loop across games. Each game
gets its own outputs/ap5game_<game>_<ts>/ dir.

Per docs/project/2026-05-16-v0-action_proposer/architecture.md §6
(extended evaluation: G_base 5 game cross-validation).

Usage:
    .venv/Scripts/python.exe scripts/run_action_proposer_5game.py
        [--backbone microsoft/Phi-4-mini-reasoning]
        [--tag-prefix ap5game]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

GAMES = ["ar25", "bp35", "cd82", "cn04", "dc22"]

ROUNDS = 2
MAX_STEPS = 300


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="",
                        help="HF model id; empty = Qwen2.5-VL-3B default")
    parser.add_argument("--tag-prefix", default="ap5game")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_root = REPO / f"outputs/{args.tag_prefix}_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)
    log_root = run_root / "logs"
    log_root.mkdir(exist_ok=True)

    summary = []
    started = time.time()

    for i, game in enumerate(GAMES):
        game_ts = time.strftime("%Y%m%d-%H%M%S")
        tag = f"{args.tag_prefix}_{game}"
        log_file = log_root / f"{game}.log"
        print(f"\n=== [{i+1}/5] {game} (rounds={ROUNDS}, max_steps={MAX_STEPS}) ===",
              flush=True)
        if args.backbone:
            print(f"    backbone: {args.backbone}", flush=True)
        print(f"    log: {log_file}", flush=True)
        start = time.time()
        cmd = [
            str(REPO / ".venv/Scripts/python.exe"),
            str(REPO / "scripts/run_v3_multi_round.py"),
            "--game", game,
            "--rounds", str(ROUNDS),
            "--max-actions", str(MAX_STEPS),
            "--seed", "42",
            "--mask", "strict",
            "--propose", "on",
            "--tag", tag,
        ]
        if args.backbone:
            cmd.extend(["--backbone", args.backbone])
        try:
            with log_file.open("w", encoding="utf-8") as f:
                result = subprocess.run(
                    cmd, stdout=f, stderr=subprocess.STDOUT,
                    cwd=REPO, timeout=7200,  # 2h max per game
                )
            elapsed = time.time() - start
            print(f"    exit={result.returncode}  elapsed={elapsed/60:.1f}min", flush=True)
            summary.append({
                "game": game, "exit": result.returncode,
                "elapsed_min": round(elapsed / 60, 1),
            })
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT after {(time.time()-start)/60:.1f}min", flush=True)
            summary.append({"game": game, "exit": -1, "elapsed_min": 120,
                             "timeout": True})
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            summary.append({"game": game, "exit": -2, "elapsed_min": 0,
                             "error": str(e)})

    total = time.time() - started
    print(f"\n=== ALL DONE ===  total wall clock: {total/3600:.2f}h", flush=True)
    print(f"summary: {summary}", flush=True)
    import json
    (run_root / "summary.json").write_text(
        json.dumps({"games": summary, "total_min": round(total / 60, 1)},
                   indent=2), encoding="utf-8"
    )
    print(f"[done] {run_root}", flush=True)


if __name__ == "__main__":
    main()
