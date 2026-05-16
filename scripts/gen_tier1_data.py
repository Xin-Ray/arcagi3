"""Generate Tier 1 SFT datasets: train (~190k), holdout (~10k), OOD (~5k).

See docs/arch_sft_tier1_zh.md §3.7. Seed=42 by default — re-run produces
identical files. Output paths are under outputs/finetune/ which is
gitignored; the generator code in arc_agent.finetune is the reproducible
artifact.

Usage:
    .venv/Scripts/python.exe scripts/gen_tier1_data.py
    .venv/Scripts/python.exe scripts/gen_tier1_data.py --seed 7 --out outputs/finetune_alt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arc_agent.finetune.synth_tier1 import Sample, build_ood, build_tier1


def _write_jsonl(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs/finetune", type=Path)
    p.add_argument(
        "--holdout-frac", type=float, default=0.05,
        help="fraction of the main mix held out for in-distribution eval",
    )
    p.add_argument(
        "--ood-per-task", type=int, default=1000,
        help="OOD samples per task (5 tasks -> 5x this many total)",
    )
    args = p.parse_args()

    out_dir: Path = args.out
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    print(f"[gen_tier1] building main mix with seed={args.seed} ...")
    samples = build_tier1(seed=args.seed)
    n = len(samples)
    n_holdout = int(n * args.holdout_frac)
    holdout = samples[:n_holdout]
    train = samples[n_holdout:]
    print(f"[gen_tier1] total={n}  train={len(train)}  holdout={len(holdout)}")

    print(f"[gen_tier1] building OOD set ({args.ood_per_task}/task) ...")
    ood = build_ood(seed=args.seed + 1295, n_per_task=args.ood_per_task)
    print(f"[gen_tier1] ood={len(ood)}")

    _write_jsonl(train,   out_dir / "tier1_train.jsonl")
    _write_jsonl(holdout, out_dir / "tier1_holdout.jsonl")
    _write_jsonl(ood,     out_dir / "tier1_ood.jsonl")

    print(f"[gen_tier1] wrote:")
    print(f"  {out_dir / 'tier1_train.jsonl'}")
    print(f"  {out_dir / 'tier1_holdout.jsonl'}")
    print(f"  {out_dir / 'tier1_ood.jsonl'}")


if __name__ == "__main__":
    main()
