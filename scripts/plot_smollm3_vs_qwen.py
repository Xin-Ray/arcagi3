"""Comparison plot: SmolLM3-3B 5x2x300 vs Qwen baselines.

After the SmolLM3 5-game run completes, generate a 2x2 figure:
  TL: per-game change_rate (SmolLM3 vs Qwen+propose 3x30 baseline)
  TR: action distribution heatmap (SmolLM3 5-game round 0)
  BL: levels_won bar chart
  BR: |action_semantics| growth across rounds

Output:
  docs/project/2026-05-17-v0-model_bench/figures/smollm3_5game_results.png
  docs/project/2026-05-17-v0-model_bench/figures/qwen_vs_smollm3_change_rate.png
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

GAMES = ["ar25", "bp35", "cd82", "cn04", "dc22"]
ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4",
           "ACTION5", "ACTION6", "ACTION7"]

# Qwen+propose baselines from earlier 3x30 smoke (round 0/1 only)
# Source: outputs/ap_v0_ar25_3x30_v2_20260516-184956
QWEN_PROPOSE_AR25 = {"round_0": 60.0, "round_1": 70.0}  # ar25 only


def _read_round(round_dir: Path) -> list[dict]:
    p = round_dir / "trace.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _find_smollm3_game_run(game: str) -> Path | None:
    candidates = sorted(REPO.glob(f"outputs/smollm3_5game_{game}_*"))
    return candidates[-1] if candidates else None


def summarize(game_run: Path) -> dict:
    if game_run is None:
        return {"missing": True}
    rounds = []
    for r in range(2):
        rows = _read_round(game_run / f"round_{r:02d}")
        if not rows:
            rounds.append({"missing": True, "n": 0})
            continue
        n = len(rows)
        changed = sum(1 for x in rows if x.get("frame_changed"))
        overrides = sum(1 for x in rows if x.get("orch_override"))
        action_dist = Counter(str(x.get("action") or "?") for x in rows)
        # Knowledge size per step
        ks_path = game_run / f"round_{r:02d}" / "knowledge_per_step.jsonl"
        ks_growth = []
        if ks_path.exists():
            for line in ks_path.read_text(encoding="utf-8").splitlines():
                try:
                    j = json.loads(line)
                    k = j.get("knowledge_after") or j.get("knowledge") or {}
                    sem = k.get("action_semantics") or {}
                    ks_growth.append(len(sem))
                except json.JSONDecodeError:
                    continue
        rounds.append({
            "missing": False, "n": n,
            "change_rate": 100 * changed / n,
            "overrides": overrides,
            "action_dist": dict(action_dist),
            "ks_growth": ks_growth,
        })
    # Final Knowledge
    final_kn = {}
    kh = game_run / "knowledge_history.jsonl"
    if kh.exists():
        lines = [l for l in kh.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            try:
                final_kn = json.loads(lines[-1]).get("knowledge_at_round_end", {})
            except json.JSONDecodeError:
                pass
    return {"missing": False, "rounds": rounds, "final_knowledge": final_kn,
             "run_dir": str(game_run)}


def make_combined_figure(per_game: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)

    # TL: change_rate per game x round (SmolLM3)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(GAMES))
    width = 0.35
    r0 = []; r1 = []
    for g in GAMES:
        d = per_game.get(g, {})
        rounds = d.get("rounds", []) if not d.get("missing") else []
        r0.append(rounds[0]["change_rate"] if len(rounds) > 0 and not rounds[0].get("missing") else 0)
        r1.append(rounds[1]["change_rate"] if len(rounds) > 1 and not rounds[1].get("missing") else 0)
    ax.bar(x - width/2, r0, width, label="round 0", color="#4caf50", alpha=0.9)
    ax.bar(x + width/2, r1, width, label="round 1", color="#9c27b0", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(GAMES)
    ax.set_ylim(0, 105); ax.set_ylabel("change_rate (%)")
    ax.set_title("SmolLM3-3B 5×2×300 change_rate per game × round")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for i, (a, b) in enumerate(zip(r0, r1)):
        if a: ax.text(i - width/2, a + 1, f"{a:.0f}", ha="center", fontsize=8)
        if b: ax.text(i + width/2, b + 1, f"{b:.0f}", ha="center", fontsize=8)

    # TR: action distribution heatmap (round 0)
    ax = fig.add_subplot(gs[0, 1])
    mat = np.zeros((len(GAMES), len(ACTIONS)))
    for i, g in enumerate(GAMES):
        d = per_game.get(g, {})
        rs = d.get("rounds", []) if not d.get("missing") else []
        if not rs or rs[0].get("missing"):
            continue
        n = rs[0]["n"]
        if n == 0:
            continue
        for j, a in enumerate(ACTIONS):
            mat[i, j] = 100 * rs[0]["action_dist"].get(a, 0) / n
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(ACTIONS))); ax.set_xticklabels(ACTIONS, rotation=15, fontsize=8)
    ax.set_yticks(range(len(GAMES))); ax.set_yticklabels(GAMES)
    for i in range(len(GAMES)):
        for j in range(len(ACTIONS)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v > 50 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label="% (round 0)")
    ax.set_title("Action distribution per game (round 0)")

    # BL: rounds_won bar chart
    ax = fig.add_subplot(gs[1, 0])
    rw = []
    for g in GAMES:
        d = per_game.get(g, {})
        fk = d.get("final_knowledge", {}) if not d.get("missing") else {}
        rw.append(int(fk.get("rounds_won", 0) or 0))
    bars = ax.bar(range(len(GAMES)), rw, color="#2196f3", alpha=0.9)
    ax.set_xticks(range(len(GAMES))); ax.set_xticklabels(GAMES)
    ax.set_ylim(0, max(max(rw, default=0), 2) + 0.5)
    ax.set_ylabel("rounds_won")
    ax.set_title("Levels completed per game (rounds_won)")
    for bar, v in zip(bars, rw):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, str(v),
                ha="center", fontsize=10, fontweight="bold")

    # BR: knowledge growth lines (all 5 games)
    ax = fig.add_subplot(gs[1, 1])
    colors = plt.cm.tab10(np.arange(len(GAMES)))
    for i, g in enumerate(GAMES):
        d = per_game.get(g, {})
        if d.get("missing"):
            continue
        ks_all = []
        for r in d.get("rounds", []):
            if not r.get("missing"):
                ks_all.extend(r.get("ks_growth", []))
        if ks_all:
            ax.plot(range(len(ks_all)), ks_all, label=g, color=colors[i], lw=1.5)
    ax.set_xlabel("step (concatenated round 0 + 1)")
    ax.set_ylabel("|action_semantics|")
    ax.set_title("Knowledge growth across games")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("SmolLM3-3B × action_proposer v0 — G_base 5 game × 2 round × 300 step",
                 fontsize=13, fontweight="bold", y=0.99)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_qwen_comparison_figure(per_game: dict, out_path: Path) -> None:
    """SmolLM3 (5 games × 2 rounds) vs Qwen baselines we have data for."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))

    # SmolLM3 mean change_rate across rounds per game
    smollm_means = []
    for g in GAMES:
        d = per_game.get(g, {})
        rs = d.get("rounds", []) if not d.get("missing") else []
        valid = [r["change_rate"] for r in rs if not r.get("missing")]
        smollm_means.append(np.mean(valid) if valid else 0)

    # Qwen baselines (we have data only for ar25 from earlier experiments)
    qwen_baselines = {
        "ar25": 5.5,    # mask_revive_3x200 round 0
        "bp35": None,    # no data
        "cd82": None,
        "cn04": None,
        "dc22": None,
    }
    qwen_propose_baselines = {
        "ar25": 65.0,   # ap_v0_ar25_3x30_v2 mean of round 0 + 1
        "bp35": None,
        "cd82": None,
        "cn04": None,
        "dc22": None,
    }
    # v2 canary historical for ar25
    v2_canary_ar25 = 82.2

    x = np.arange(len(GAMES))
    width = 0.22
    ax.bar(x - width*1.5,
           [v if v is not None else 0 for v in qwen_baselines.values()],
           width, label="Qwen+mask off propose (main HEAD)",
           color="#f44336", alpha=0.8)
    ax.bar(x - width*0.5,
           [v if v is not None else 0 for v in qwen_propose_baselines.values()],
           width, label="Qwen+propose (3x30 smoke)",
           color="#ff9800", alpha=0.85)
    ax.bar(x + width*0.5,
           [v2_canary_ar25] + [0]*4,
           width, label="v2 canary (ar25 only, 3x30 history)",
           color="#9c27b0", alpha=0.85)
    ax.bar(x + width*1.5, smollm_means, width,
           label="SmolLM3+propose (5×2×300, this run)",
           color="#4caf50", alpha=0.9)

    ax.set_xticks(x); ax.set_xticklabels(GAMES, fontsize=10)
    ax.set_ylim(0, 105); ax.set_ylabel("Mean change_rate (%)")
    ax.set_title("SmolLM3 vs Qwen baselines — change_rate per game (across rounds)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(smollm_means):
        if v:
            ax.text(i + width*1.5, v + 1.5, f"{v:.0f}",
                    ha="center", fontsize=8, fontweight="bold", color="#4caf50")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",
                        default="docs/project/2026-05-17-v0-model_bench/figures")
    args = parser.parse_args()

    per_game = {}
    for g in GAMES:
        run = _find_smollm3_game_run(g)
        if run is None:
            print(f"[scan] {g}: NO run dir found", flush=True)
            per_game[g] = {"missing": True}
            continue
        print(f"[scan] {g}: {run.name}", flush=True)
        per_game[g] = summarize(run)

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    make_combined_figure(per_game, out_dir / "smollm3_5game_results.png")
    make_qwen_comparison_figure(per_game, out_dir / "qwen_vs_smollm3_change_rate.png")
    print(f"\n[done] figures: {out_dir}")
    print("\nPer-game summary:")
    for g, d in per_game.items():
        if d.get("missing"):
            print(f"  {g}: MISSING")
            continue
        rs = d.get("rounds", [])
        crs = [f"{r['change_rate']:.0f}%" if not r.get('missing') else 'missing' for r in rs]
        fk = d.get("final_knowledge", {})
        rw = fk.get("rounds_won", 0)
        sem = len(fk.get("action_semantics", {}) or {})
        print(f"  {g}: change_rate={crs} rounds_won={rw} |sem|={sem}")


if __name__ == "__main__":
    main()
