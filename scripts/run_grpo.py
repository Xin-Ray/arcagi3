"""GRPO training entry point — Stage D Step 7.

Reads the frozen demo_555 split, picks G_train (default) and G_val, builds
the GRPO trainer via `arc_agent.train_grpo.build_trainer`, and runs.

This script is the orchestration layer — all reusable logic lives in
`arc_agent/train_grpo.py`. Per the project rule, scripts only do I/O and
argparse wiring.

Pre-registered hypothesis (per docs/ARCHITECTURE_RL.md §5.3):
    G_val  mean RHAE post-train  ≥  pre-train + 0.05    (real generalization)
    G_train mean RHAE post-train  ≥  pre-train + 0.10   (sanity check)

CLI examples:
    # Dry run — no torch / trl imports
    .venv/Scripts/python.exe scripts/run_grpo.py --dry-run

    # Real training (needs GPU + training stack)
    .venv/Scripts/python.exe scripts/run_grpo.py \
        --steps 500 --val-every 50 --output outputs/grpo_<ts>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

SPLIT_PATH = REPO_ROOT / "data" / "splits" / "demo_555.json"
OUTPUTS_ROOT = REPO_ROOT / "outputs"

TRAIN_HYPOTHESIS = {
    "g_val_rhae_gain_min":   0.05,
    "g_train_rhae_gain_min": 0.10,
}


def _check_key() -> None:
    key = os.getenv("ARC_API_KEY", "")
    if not key or key.startswith("your_"):
        raise RuntimeError(
            "ARC_API_KEY missing or still the .env.example placeholder. "
            "Put a real key in .env (see https://arcprize.org/api-keys)."
        )


def _read_split() -> dict[str, Any]:
    if not SPLIT_PATH.exists():
        raise RuntimeError(
            f"Frozen split missing: {SPLIT_PATH}. "
            "Run `scripts/freeze_splits.py` first."
        )
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))


def _maybe(value: str, default: list[str]) -> list[str]:
    return [g.strip() for g in value.split(",") if g.strip()] if value else default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default="",
                        help="Comma list of training game ids (default: G_train from split)")
    parser.add_argument("--val-games", default="",
                        help="Comma list of validation game ids (default: G_val from split)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Total GRPO steps (matches §9 Step 7 acceptance)")
    parser.add_argument("--val-every", type=int, default=50,
                        help="Run G_val eval every N steps; early-stop on flat for 3")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=4,
                        help="K rollouts per GRPO step (the 'G' in GRPO)")
    parser.add_argument("--output", default="",
                        help="Run dir (default: outputs/grpo_<ts>)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan without importing torch/trl or loading the model.",
    )
    args = parser.parse_args()

    split = _read_split()
    train_games = _maybe(args.games,     split["g_train"])
    val_games   = _maybe(args.val_games, split["g_val"])

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output) if args.output else OUTPUTS_ROOT / f"grpo_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "schema_version": "1",
        "run_kind": "grpo_plan",
        "run_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_games": train_games,
        "val_games": val_games,
        "steps": args.steps,
        "val_every": args.val_every,
        "learning_rate": args.learning_rate,
        "num_generations": args.num_generations,
        "seed": args.seed,
        "split": str(SPLIT_PATH.relative_to(REPO_ROOT)),
        "hypothesis": TRAIN_HYPOTHESIS,
        "output": str(run_dir),
    }
    (run_dir / "plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print("=== GRPO plan ===")
    for k, v in plan.items():
        print(f"  {k}: {v}")
    print()

    if args.dry_run:
        print("--dry-run: stopping before torch/trl imports.")
        print(f"plan.json written -> {run_dir / 'plan.json'}")
        return

    _check_key()

    # Heavy imports only past --dry-run gate.
    from arc_agent.train_grpo import build_trainer, reward_fn
    from arc_agent.vlm_backbone import HFBackbone

    print("Loading Qwen2.5-VL-3B (4-bit) — this may take a minute...")
    bb = HFBackbone.load()

    trainer = build_trainer(
        model=bb.model,
        processor=bb.processor,
        reward_fn=reward_fn,
        train_games=train_games,
        val_games=val_games,
        output_dir=str(run_dir),
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_steps=args.steps,
        val_every=args.val_every,
    )
    print(f"Trainer built ({type(trainer).__name__}). Rollout loop wiring -> TODO Step 7.b")
    print("This script currently sets up the trainer; the rollout adapter and training")
    print("loop need to be glued in once the Step 6 baseline gates pass. See")
    print("docs/ARCHITECTURE_RL.md §9 Step 7.")


if __name__ == "__main__":
    main()
