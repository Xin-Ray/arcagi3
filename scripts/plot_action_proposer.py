"""Generate comparison chart for action_proposer v0 vs baselines.

Reads:
  - outputs/ap_v0_ar25_3x30_v2_* (this experiment)
  - outputs/v3_2_ar25_3x30_v2 (v2 canary historical)
  - outputs/preload_v3_budget_smoke_* OR mask_revive_* (main baseline)

Outputs:
  docs/project/2026-05-16-v0-action_proposer/figures/
    change_rate_comparison.png  - per-round bar chart 3 runs
    action_distribution.png     - per-round action distribution 3 runs
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _read_round(run_dir: Path, r: int) -> list[dict]:
    p = run_dir / f"round_{r:02d}" / "trace.jsonl"
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


def _summarize(run_dir: Path, max_rounds: int = 5) -> list[dict]:
    out = []
    for r in range(max_rounds):
        rows = _read_round(run_dir, r)
        if not rows and r > 0:
            break
        if not rows:
            continue
        total = len(rows)
        changed = sum(1 for x in rows if x.get("frame_changed"))
        overrides = sum(1 for x in rows if x.get("orch_override"))
        action_dist = Counter(str(x.get("action") or "?") for x in rows)
        out.append({
            "round": r, "n": total,
            "change_rate": 100 * changed / total,
            "overrides": overrides,
            "action_dist": dict(action_dist),
        })
    return out


def plot_change_rate(runs: dict[str, list[dict]], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    max_rounds = max(len(r) for r in runs.values())
    x = np.arange(max_rounds)
    width = 0.8 / len(runs)
    colors = {"propose ON": "#4caf50", "v2 canary (historical)": "#2196f3",
              "mask strict (main HEAD)": "#f44336"}
    for i, (name, rounds) in enumerate(runs.items()):
        ys = [r["change_rate"] for r in rounds]
        # Pad
        while len(ys) < max_rounds:
            ys.append(0)
        pos = x + (i - len(runs) / 2 + 0.5) * width
        ax.bar(pos, ys, width=width, label=name,
               color=colors.get(name, "#888"))
    ax.set_xticks(x); ax.set_xticklabels([f"round {i}" for i in range(max_rounds)])
    ax.set_ylim(0, 100); ax.set_ylabel("change_rate (%)")
    ax.set_title("Action Proposer v0 vs baselines — ar25 per-round change_rate")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_action_dist(runs: dict[str, list[dict]], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    actions = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]
    n_runs = len(runs)
    fig, axes = plt.subplots(1, n_runs, figsize=(4.5 * n_runs, 4.5), sharey=True)
    if n_runs == 1:
        axes = [axes]
    for ax, (name, rounds) in zip(axes, runs.items()):
        if not rounds:
            ax.set_title(f"{name} (no data)")
            continue
        r0 = rounds[0]
        total = r0["n"]
        if total == 0:
            ax.set_title(f"{name} (round 0 empty)")
            continue
        pct = [100 * r0["action_dist"].get(a, 0) / total for a in actions]
        ax.bar(range(len(actions)), pct, color="#5c6bc0")
        ax.set_xticks(range(len(actions)))
        ax.set_xticklabels(actions, rotation=20, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(f"{name}\n(round 0, n={total})")
        ax.axhline(50, color="r", ls="--", lw=0.5, alpha=0.5)
        # Annotate top action
        max_idx = pct.index(max(pct))
        ax.text(max_idx, pct[max_idx] + 2, f"{pct[max_idx]:.0f}%",
                ha="center", fontsize=9, fontweight="bold")
    axes[0].set_ylabel("action distribution (%)")
    fig.suptitle("Action diversity — round 0 of each run", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--propose", default="outputs/ap_v0_ar25_3x30_v2_*")
    parser.add_argument("--v2-canary", default="outputs/v3_2_ar25_3x30_v2")
    parser.add_argument("--main", default="outputs/mask_revive_3x200_*")
    parser.add_argument("--output-dir",
                        default="docs/project/2026-05-16-v0-action_proposer/figures")
    args = parser.parse_args()

    def _resolve(p: str) -> Path:
        if "*" in p:
            matches = list((REPO).glob(p))
            return sorted(matches)[-1] if matches else REPO / p
        return REPO / p

    propose_dir = _resolve(args.propose)
    v2_dir = _resolve(args.v2_canary)
    main_dir = _resolve(args.main)

    runs = {
        "propose ON": _summarize(propose_dir),
        "v2 canary (historical)": _summarize(v2_dir),
        "mask strict (main HEAD)": _summarize(main_dir),
    }
    out_dir = REPO / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_change_rate(runs, out_dir / "change_rate_comparison.png")
    plot_action_dist(runs, out_dir / "action_distribution.png")
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
