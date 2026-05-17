"""Generate 5-game × 2-round visualization report for action_proposer v0.

Reads outputs/ap5game_<ts>/ tree (created by run_action_proposer_5game.py),
generates:
  - per_game_change_rate.png      bar chart: 5 games × 2 rounds
  - cross_game_action_dist.png    heatmap: games × ACTIONs (occupancy %)
  - knowledge_growth.png          line chart: |action_semantics| over steps
  - levels_completed_summary.png  did any game pass any level?
  - 5game_summary_report.md       markdown narrative report

Usage:
    .venv/Scripts/python.exe scripts/plot_action_proposer_5game.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]

GAMES = ["ar25", "bp35", "cd82", "cn04", "dc22"]
ACTIONS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4",
           "ACTION5", "ACTION6", "ACTION7"]


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


def _find_game_run_dir(game: str) -> Path | None:
    """Find the latest outputs/ap5game_<game>_* dir."""
    candidates = sorted(REPO.glob(f"outputs/ap5game_{game}_*"))
    return candidates[-1] if candidates else None


def _summarize_game(game_run: Path) -> dict:
    """Per-round stats + final Knowledge for one game."""
    if game_run is None:
        return {"missing": True}
    rounds: list[dict] = []
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
        # WIN check
        levels = max((x.get("level", 0) or 0) for x in rows) - min(
            (x.get("level", 0) or 0) for x in rows
        )
        rounds.append({
            "missing": False,
            "n": n, "change_rate": 100 * changed / n,
            "n_changed": changed, "overrides": overrides,
            "action_dist": dict(action_dist),
            "ks_growth": ks_growth,
            "levels_delta": levels,
        })

    # Read final Knowledge from auto-report or knowledge_history
    final_kn: dict[str, Any] = {}
    kh_path = game_run / "knowledge_history.jsonl"
    if kh_path.exists():
        lines = [l for l in kh_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            try:
                last = json.loads(lines[-1])
                final_kn = last.get("knowledge_at_round_end", {})
            except json.JSONDecodeError:
                pass

    return {"missing": False, "rounds": rounds, "final_knowledge": final_kn,
             "run_dir": str(game_run)}


def _plot_per_game_change_rate(per_game: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(GAMES))
    width = 0.35
    r0 = []
    r1 = []
    for g in GAMES:
        d = per_game.get(g, {})
        if d.get("missing"):
            r0.append(0); r1.append(0)
            continue
        rounds = d.get("rounds", [])
        r0.append(rounds[0]["change_rate"] if len(rounds) > 0 and not rounds[0].get("missing") else 0)
        r1.append(rounds[1]["change_rate"] if len(rounds) > 1 and not rounds[1].get("missing") else 0)
    ax.bar(x - width/2, r0, width, label="round 0", color="#4caf50", alpha=0.9)
    ax.bar(x + width/2, r1, width, label="round 1", color="#9c27b0", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(GAMES)
    ax.set_ylim(0, 100); ax.set_ylabel("change_rate (%)")
    ax.set_title("Action Proposer v0 — change_rate per game × round\n"
                 "(G_base 5 game × 2 round × 300 step max)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for i, (a, b) in enumerate(zip(r0, r1)):
        ax.text(i - width/2, a + 1, f"{a:.0f}", ha="center", fontsize=8)
        ax.text(i + width/2, b + 1, f"{b:.0f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _plot_cross_game_action_dist(per_game: dict, out_path: Path) -> None:
    """Heatmap: rows = games, cols = actions, cell = occupancy% (round 0)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mat = np.zeros((len(GAMES), len(ACTIONS)))
    for i, g in enumerate(GAMES):
        d = per_game.get(g, {})
        if d.get("missing"):
            continue
        rounds = d.get("rounds", [])
        if not rounds or rounds[0].get("missing"):
            continue
        r0 = rounds[0]
        n = r0["n"]
        if n == 0:
            continue
        for j, a in enumerate(ACTIONS):
            mat[i, j] = 100 * r0["action_dist"].get(a, 0) / n
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(ACTIONS))); ax.set_xticklabels(ACTIONS, rotation=15)
    ax.set_yticks(np.arange(len(GAMES))); ax.set_yticklabels(GAMES)
    for i in range(len(GAMES)):
        for j in range(len(ACTIONS)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v > 50 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label="Action occupancy (%, round 0)")
    ax.set_title("Action distribution by game (round 0)\n"
                 "Healthy: red on a single column is BAD (action lock-in)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_knowledge_growth(per_game: dict, out_path: Path) -> None:
    """Line chart: |action_semantics| over steps, one line per game."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.arange(len(GAMES)))
    for i, g in enumerate(GAMES):
        d = per_game.get(g, {})
        if d.get("missing"):
            continue
        # Concatenate round 0 and round 1 ks_growth
        all_growth = []
        for r in d.get("rounds", []):
            if not r.get("missing"):
                all_growth.extend(r.get("ks_growth", []))
        if all_growth:
            ax.plot(range(len(all_growth)), all_growth, label=g,
                    color=colors[i], lw=1.5)
    ax.set_xlabel("step (across round 0 + round 1)")
    ax.set_ylabel("|action_semantics|")
    ax.set_title("Knowledge growth — propose ON keeps Knowledge populating across games")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_levels_completed(per_game: dict, out_path: Path) -> None:
    """Bar chart: did any game pass levels?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    levels = []
    for g in GAMES:
        d = per_game.get(g, {})
        if d.get("missing"):
            levels.append(0)
            continue
        fk = d.get("final_knowledge", {})
        rw = fk.get("rounds_won", 0) or 0
        levels.append(int(rw))
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(GAMES)), levels, color="#2196f3", alpha=0.9)
    ax.set_xticks(range(len(GAMES))); ax.set_xticklabels(GAMES)
    ax.set_ylim(0, max(max(levels), 2) + 0.5)
    ax.set_ylabel("rounds_won")
    ax.set_title("Levels completed per game (rounds_won field of final Knowledge)")
    for bar, v in zip(bars, levels):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, str(v),
                ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _build_report_md(per_game: dict, out_path: Path,
                     figures_subdir: str = "figures") -> None:
    lines = []
    lines.append("# Action Proposer v0 — G_base 5 game × 2 round × 300 step report")
    lines.append("")
    lines.append(f"生成: 2026-05-17")
    lines.append(f"对应架构: [`architecture.md`](./architecture.md)")
    lines.append("")
    lines.append("## TL;DR (per-game table)")
    lines.append("")
    lines.append("| Game | r0 change_rate | r1 change_rate | r0 max_action | r1 max_action | "
                 "rounds_won | final |action_semantics| |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for g in GAMES:
        d = per_game.get(g, {})
        if d.get("missing"):
            lines.append(f"| {g} | ❌ missing | | | | | |")
            continue
        rounds = d.get("rounds", [])
        def _cr(i: int) -> str:
            if i < len(rounds) and not rounds[i].get("missing"):
                return f"{rounds[i]['change_rate']:.1f}%"
            return "-"
        def _max_a(i: int) -> str:
            if i < len(rounds) and not rounds[i].get("missing"):
                ad = rounds[i]["action_dist"]
                n = rounds[i]["n"]
                if n == 0:
                    return "-"
                top = max(ad.values()) if ad else 0
                return f"{100*top/n:.0f}%"
            return "-"
        fk = d.get("final_knowledge", {})
        rw = fk.get("rounds_won", 0)
        sem = len(fk.get("action_semantics", {}) or {})
        lines.append(
            f"| {g} | {_cr(0)} | {_cr(1)} | {_max_a(0)} | {_max_a(1)} | "
            f"{rw} | {sem} |"
        )
    lines.append("")
    lines.append(f"## Plots")
    lines.append("")
    lines.append(f"![per-game change_rate](./{figures_subdir}/per_game_change_rate.png)")
    lines.append("")
    lines.append(f"![cross-game action dist](./{figures_subdir}/cross_game_action_dist.png)")
    lines.append("")
    lines.append(f"![knowledge growth](./{figures_subdir}/knowledge_growth.png)")
    lines.append("")
    lines.append(f"![levels completed](./{figures_subdir}/levels_completed.png)")
    lines.append("")

    # Per-game final Knowledge dumps
    lines.append("## Per-game final Knowledge")
    lines.append("")
    for g in GAMES:
        d = per_game.get(g, {})
        if d.get("missing"):
            continue
        fk = d.get("final_knowledge", {})
        lines.append(f"### {g}")
        lines.append("```json")
        sliced = {k: fk[k] for k in ("rounds_played", "rounds_won",
                                     "action_semantics", "goal_hypothesis",
                                     "goal_confidence", "rules",
                                     "failed_strategies", "round_history")
                  if k in fk}
        lines.append(json.dumps(sliced, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="docs/project/2026-05-16-v0-action_proposer/figures_5game",
    )
    parser.add_argument(
        "--report",
        default="docs/project/2026-05-16-v0-action_proposer/report_5game.md",
    )
    args = parser.parse_args()

    per_game: dict[str, dict] = {}
    for g in GAMES:
        run = _find_game_run_dir(g)
        if run is None:
            print(f"[scan] {g}: NO run dir found", flush=True)
            per_game[g] = {"missing": True}
            continue
        print(f"[scan] {g}: {run.name}", flush=True)
        per_game[g] = _summarize_game(run)

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_per_game_change_rate(per_game, out_dir / "per_game_change_rate.png")
    _plot_cross_game_action_dist(per_game, out_dir / "cross_game_action_dist.png")
    _plot_knowledge_growth(per_game, out_dir / "knowledge_growth.png")
    _plot_levels_completed(per_game, out_dir / "levels_completed.png")

    report = REPO / args.report
    _build_report_md(per_game, report, figures_subdir=out_dir.name)
    print(f"\n[done] report: {report}\n        figures: {out_dir}")


if __name__ == "__main__":
    main()
