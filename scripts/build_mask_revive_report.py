"""Generate the mask re-enable comparison report.

Compares the new `outputs/mask_revive_3x200_*` run against the historical
v3_2_ar25_3x30_v2 (mask-on canary) and preload_v3_budget_smoke_*
(mask-off recent baseline). Produces:

  outputs/reports/mask_revive_3x200.md     (markdown summary)
  outputs/reports/mask_revive_3x200/       (4 PNG plots)

Usage:
    .venv/Scripts/python.exe scripts/build_mask_revive_report.py
        --new outputs/mask_revive_3x200_20260516-014356
        --v2  outputs/v3_2_ar25_3x30_v2
        --off outputs/preload_v3_budget_smoke_20260515-004233
        --output outputs/reports/mask_revive_3x200.md

If --new isn't specified, picks the latest `outputs/mask_revive_*` dir.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load_round(run: Path, r: int) -> list[dict]:
    """Read one round's trace.jsonl as list of dicts. Empty list if missing."""
    p = run / f"round_{r:02d}" / "trace.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _round_summary(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "change_rate": 0.0, "overrides": 0,
                "actions": {}, "max_no_op_streak": 0}
    changed = sum(1 for r in rows if r.get("frame_changed"))
    overrides = sum(1 for r in rows if r.get("orch_override"))
    actions: dict[str, int] = {}
    streak = 0
    max_streak = 0
    for r in rows:
        a = str(r.get("action") or "?")
        actions[a] = actions.get(a, 0) + 1
        if not r.get("frame_changed"):
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return {
        "n": n, "change_rate": 100 * changed / n,
        "n_changed": changed, "n_no_op": n - changed,
        "overrides": overrides,
        "actions": actions,
        "max_no_op_streak": max_streak,
    }


def _run_summary(run: Path, max_rounds: int = 5) -> list[dict]:
    out = []
    for r in range(max_rounds):
        rows = _load_round(run, r)
        if not rows and r > 0:
            break
        out.append(_round_summary(rows))
    return out


def _plot_change_rate(
    runs: dict[str, list[dict]], out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, rounds in runs.items():
        xs = [f"r{i}" for i in range(len(rounds))]
        ys = [r["change_rate"] for r in rounds]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_ylabel("Round change_rate (%)")
    ax.set_xlabel("Round")
    ax.set_ylim(0, 100)
    ax.set_title("Round-level change_rate by run")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_action_dist(
    runs: dict[str, list[dict]], out_path: Path,
) -> None:
    """Stacked bar: action distribution per run (round-0)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4",
               "ACTION5", "ACTION6", "ACTION7"]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / max(len(runs), 1)
    for i, (name, rounds) in enumerate(runs.items()):
        if not rounds:
            continue
        counts = [rounds[0]["actions"].get(a, 0) for a in actions]
        n = sum(counts) or 1
        pct = [100 * c / n for c in counts]
        x = np.arange(len(actions)) + (i - len(runs) / 2) * width + width / 2
        ax.bar(x, pct, width=width, label=name)
    ax.set_xticks(np.arange(len(actions)))
    ax.set_xticklabels(actions, rotation=20)
    ax.set_ylabel("Action distribution (%, round 0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_change_rate_timeseries(
    runs: dict[str, list[dict]], out_path: Path, window: int = 20,
) -> None:
    """Rolling change_rate within round 0 over steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, rounds in runs.items():
        if not rounds:
            continue
        run_dir = rounds[0].get("_run_dir")
        # Re-read round 0 to get step-by-step
        rows = rounds[0].get("_rows", [])
        if not rows:
            continue
        ys = []
        for i in range(len(rows)):
            start = max(0, i - window + 1)
            chunk = rows[start: i + 1]
            ch = sum(1 for r in chunk if r.get("frame_changed"))
            ys.append(100 * ch / len(chunk))
        ax.plot(range(len(rows)), ys, label=name)
    ax.set_xlabel("Step (round 0)")
    ax.set_ylabel(f"Rolling change_rate (window={window}, %)")
    ax.set_title("Step-level change_rate, round 0")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", default="")
    parser.add_argument("--v2", default="outputs/v3_2_ar25_3x30_v2")
    parser.add_argument("--off", default="outputs/preload_v3_budget_smoke_20260515-004233")
    parser.add_argument("--output", default="outputs/reports/mask_revive_3x200.md")
    args = parser.parse_args()

    if args.new:
        new_run = REPO / args.new
    else:
        candidates = sorted((REPO / "outputs").glob("mask_revive_*"))
        if not candidates:
            print("No mask_revive_* directories found", file=sys.stderr)
            sys.exit(1)
        new_run = candidates[-1]
    v2_run = REPO / args.v2
    off_run = REPO / args.off

    # Pull each run's per-round summary + cache rows for the timeseries plot
    new_rounds = _run_summary(new_run)
    v2_rounds = _run_summary(v2_run)
    off_rounds = _run_summary(off_run)

    # Attach raw rows for the step-level plot (round 0 only).
    for run_dir, summary in [(new_run, new_rounds),
                             (v2_run, v2_rounds), (off_run, off_rounds)]:
        if summary:
            r0_rows = _load_round(run_dir, 0)
            summary[0]["_rows"] = r0_rows
            summary[0]["_run_dir"] = str(run_dir)

    runs = {
        f"new (3x200, mask strict)": new_rounds,
        f"v2 canary (3x30, mask strict)": v2_rounds,
        f"off baseline (preload, mask off)": off_rounds,
    }

    out_md = REPO / args.output
    img_dir = out_md.parent / out_md.stem
    img_dir.mkdir(parents=True, exist_ok=True)

    _plot_change_rate(runs, img_dir / "change_rate_per_round.png")
    _plot_action_dist(runs, img_dir / "action_dist_round0.png")
    _plot_change_rate_timeseries(runs, img_dir / "change_rate_timeseries.png")

    # Markdown
    lines: list[str] = []
    lines.append(f"# Mask re-enable — ar25 1 game × 3 round × 200 step")
    lines.append("")
    lines.append(f"Generated 2026-05-16. Source: `{new_run.name}`.")
    lines.append("")
    lines.append("## Headline comparison")
    lines.append("")
    lines.append(
        "| run | rounds | r0 change_rate | r1 change_rate | r2 change_rate | "
        "r0 overrides | r0 max_no_op_streak |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, rounds in runs.items():
        n = len(rounds)
        def cr(i: int) -> str:
            return f"{rounds[i]['change_rate']:.1f}%" if i < n else "-"
        ov = rounds[0]["overrides"] if rounds else "-"
        mns = rounds[0]["max_no_op_streak"] if rounds else "-"
        lines.append(
            f"| {name} | {n} | {cr(0)} | {cr(1)} | {cr(2)} | {ov} | {mns} |"
        )
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append(f"![per-round](./{out_md.stem}/change_rate_per_round.png)")
    lines.append("")
    lines.append(f"![action-dist](./{out_md.stem}/action_dist_round0.png)")
    lines.append("")
    lines.append(f"![timeseries](./{out_md.stem}/change_rate_timeseries.png)")
    lines.append("")
    lines.append("## Per-run action distribution (round 0)")
    lines.append("")
    for name, rounds in runs.items():
        lines.append(f"### {name}")
        if not rounds:
            lines.append("(no data)")
            continue
        ad = rounds[0]["actions"]
        n = sum(ad.values()) or 1
        for a, c in sorted(ad.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{a}`: {c} ({100*c/n:.1f}%)")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] {out_md}")


if __name__ == "__main__":
    main()
