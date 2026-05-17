"""Sequentially run multiple subtasks with one model load.

After T-NAV-1 finishes (which uses scripts/bench_subtask.py), this
batch script runs T-NAV-2 / T-NAV-3 / T-SEL-1 / T-GOAL in one Python
process so the model is loaded once.

Usage:
    .venv/Scripts/python.exe scripts/bench_subtask_batch.py
        --subtasks T-NAV-2,T-NAV-3,T-SEL-1,T-GOAL
        --model smollm3-cot
        --n-probes 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Import shared loaders from bench_subtask
from scripts.bench_subtask import (  # type: ignore[import-not-found]
    MODEL_REGISTRY, SYSTEM_PROMPT,
    parse_answer, format_probe, load_model, free_model, generate,
)
from arc_agent.subtask_probes import GENERATORS


def run_subtask(model, tokenizer, entry: dict, subtask: str,
                n_probes: int, seed: int, max_new_tokens: int,
                out_root: Path, ts: str) -> dict:
    """Run probes for one subtask using already-loaded model."""
    probes = GENERATORS[subtask](n=n_probes, seed=seed)
    out_dir = out_root / f"subtask_{subtask}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {subtask} ({len(probes)} probes) ===", flush=True)
    t0 = time.time()
    results = []
    for i, p in enumerate(probes):
        t = time.time()
        try:
            raw = generate(model, tokenizer, SYSTEM_PROMPT, format_probe(p),
                           entry, max_new_tokens=max_new_tokens)
            elapsed = time.time() - t
        except Exception as e:
            raw = ""
            elapsed = time.time() - t
            print(f"  GEN ERROR on {p['id']}: {e}", flush=True)
        guessed = parse_answer(raw)
        ok = guessed == p["correct"]
        results.append({
            "id": p["id"], "correct": p["correct"], "guessed": guessed,
            "ok": ok, "elapsed_s": round(elapsed, 2),
            "raw_tail": raw[-200:] if raw else "",
        })
        if (i + 1) % 10 == 0 or i + 1 == len(probes):
            n_ok = sum(1 for r in results if r["ok"])
            print(f"  [{i+1}/{len(probes)}] {subtask} "
                  f"running_acc={100*n_ok/(i+1):.1f}% "
                  f"last_elapsed={elapsed:.1f}s", flush=True)

    n_correct = sum(1 for r in results if r["ok"])
    accuracy = n_correct / len(probes) if probes else 0
    total_s = time.time() - t0

    # Dump per-probe + summary
    model_key = entry.get("_key", "model")
    with (out_dir / f"per_probe_{model_key}.jsonl").open("w", encoding="utf-8") as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    summary = {
        "subtask": subtask, "n_probes": len(probes), "model": entry["label"],
        "reasoning_mode": entry["reasoning_mode"],
        "n_correct": n_correct, "accuracy": accuracy,
        "total_s": round(total_s, 1),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [f"# Subtask bench — {subtask}", "",
             f"Model: {entry['label']} (reasoning={entry['reasoning_mode']})",
             f"n_probes: {len(probes)}",
             f"accuracy: **{100*accuracy:.1f}%** ({n_correct}/{len(probes)})",
             f"total: {total_s:.1f}s"]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"  [{subtask}] done: {100*accuracy:.1f}% ({n_correct}/{len(probes)}) "
          f"in {total_s:.1f}s", flush=True)
    return {
        "subtask": subtask, "out_dir": str(out_dir),
        "n_probes": len(probes), "n_correct": n_correct,
        "accuracy": accuracy, "total_s": round(total_s, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtasks", default="T-NAV-2,T-NAV-3,T-SEL-1,T-GOAL",
                        help="Comma-separated subtask names. Default = 4 not yet run.")
    parser.add_argument("--model", default="smollm3-cot",
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--n-probes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    subtasks = [s.strip() for s in args.subtasks.split(",") if s.strip()]
    for st in subtasks:
        if st not in GENERATORS:
            print(f"unknown subtask: {st}; valid: {list(GENERATORS.keys())}",
                  file=sys.stderr)
            sys.exit(1)

    entry = dict(MODEL_REGISTRY[args.model])  # copy
    entry["_key"] = args.model
    print(f"[batch] {len(subtasks)} subtasks × {args.n_probes} probes on "
          f"{entry['label']} (reasoning={entry['reasoning_mode']})", flush=True)

    print(f"\n=== loading model once ===", flush=True)
    t0 = time.time()
    model, tokenizer = load_model(entry)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_root = REPO / "outputs"

    all_results = []
    for st in subtasks:
        r = run_subtask(model, tokenizer, entry, st,
                        args.n_probes, args.seed, args.max_new_tokens,
                        out_root, ts)
        all_results.append(r)

    free_model(model)

    # Cross-subtask summary
    cross = {"model": entry["label"], "reasoning": entry["reasoning_mode"],
             "subtasks": all_results}
    cross_dir = out_root / f"subtask_batch_{ts}"
    cross_dir.mkdir(parents=True, exist_ok=True)
    (cross_dir / "metrics.json").write_text(
        json.dumps(cross, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [f"# Subtask batch — {entry['label']} ({entry['reasoning_mode']})",
             "", "| Subtask | accuracy | n_correct/n | total(s) | dir |",
             "|---|---:|---:|---:|---|"]
    for r in all_results:
        d_name = Path(r["out_dir"]).name
        lines.append(f"| {r['subtask']} | **{100*r['accuracy']:.1f}%** | "
                     f"{r['n_correct']}/{r['n_probes']} | "
                     f"{r['total_s']} | `{d_name}` |")
    (cross_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[batch done] cross-summary: {cross_dir}", flush=True)
    for r in all_results:
        print(f"  {r['subtask']}: {100*r['accuracy']:.1f}%", flush=True)


if __name__ == "__main__":
    main()
