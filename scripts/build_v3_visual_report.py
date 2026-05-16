"""Build the v3 visual report:

For each game in `outputs/v3_eval_visual/`:
  - Embed play.gif (composed by the baseline runner)
  - Show key metrics (entropy / uniq actions / no-op / RHAE)
  - Generate a compact step-by-step trace_view.md listing
      step | action | changed | direction | response (truncated)
    so the user can scrub the GIF AND read what the model said.

Top-level report at `outputs/v3_eval_visual/report.md` includes:
  - Comparison table (v3 vs v1 LLM baselines)
  - Per-game section with GIF + link to trace_view.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V3_DIR = REPO_ROOT / "outputs" / "v3_eval_visual"

# Compare to v1 baselines too (we computed these in report_v3_vs_v1.py)
V1_DIRS = {
    "R0 random":    REPO_ROOT / "outputs" / "ablation_overnight_" / "random",
    "A1 lite":      REPO_ROOT / "outputs" / "ablation_overnight_" / "lite",
    "A2 full":      REPO_ROOT / "outputs" / "ablation_overnight_" / "full",
    "A3 reflect":   REPO_ROOT / "outputs" / "ablation_overnight_" / "reflect",
    "A4 reflect+m": REPO_ROOT / "outputs" / "ablation_overnight_" / "reflect_mistakes",
}


def _shannon(actions: list[str]) -> float:
    if not actions:
        return 0.0
    counts = list(Counter(actions).values())
    total = sum(counts)
    return -sum((c / total) * math.log(c / total) for c in counts if c > 0)


def _direction_from_diff(diff: list | None) -> str:
    """Crude direction guess from real_diff cells.

    For v3 we don't have predicted_diff; we just describe the centroid
    of the changed cells. Returns "no-op" / "changed (small)" / "moved
    in <direction>"."""
    if not diff:
        return "no-op"
    n = len(diff)
    if n < 3:
        return f"{n} cell(s) changed"
    return f"{n} cells changed"


def _build_trace_view(game_dir: Path) -> Path:
    """Per-game compact step-by-step Markdown."""
    trace = game_dir / "trace.jsonl"
    out = game_dir / "trace_view.md"
    if not trace.exists():
        out.write_text(f"# {game_dir.name}\n\n(no trace.jsonl)\n", encoding="utf-8")
        return out

    rows: list[dict] = []
    with trace.open(encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line.strip()))
            except Exception:
                continue

    lines = [
        f"# {game_dir.name} — step-by-step trace",
        "",
        f"Total steps: **{len(rows)}**. "
        f"![play.gif](play.gif)",
        "",
        "| Step | Action | Changed | n cells | Response (raw) |",
        "|---:|---|:---:|---:|---|",
    ]
    for r in rows:
        step = r.get("step")
        action = r.get("chosen_action") or "(fallback)"
        diff = r.get("real_diff") or []
        changed = "✅" if len(diff) > 0 else "—"
        resp = (r.get("response_raw") or "").strip().replace("\n", " ").replace("|", "/")
        # Truncate response to fit table
        resp_short = (resp[:60] + "...") if len(resp) > 63 else resp
        lines.append(
            f"| {step} | `{action}` | {changed} | {len(diff)} | `{resp_short}` |"
        )

    lines += [
        "",
        "## Action distribution",
        "",
    ]
    actions = [r.get("chosen_action") for r in rows if r.get("chosen_action")]
    counts = Counter(actions)
    for action_name, n in counts.most_common():
        bar = "#" * min(40, n)
        lines.append(f"  {action_name:8s}  {n:3d}  {bar}")
    lines += [
        "",
        f"entropy = **{_shannon(actions):.3f}** (uniform-7 ceiling = 1.946)",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _summarize(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {"missing": True}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    per_game = payload.get("per_game", {})
    entropies, uniqs, no_ops, levels, scores = [], [], [], [], []
    n_steps_total = 0
    for game_id, g in per_game.items():
        if "error" in g:
            continue
        trace = run_dir / game_id / "trace.jsonl"
        actions: list[str] = []
        nop = 0
        if trace.exists():
            with trace.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line.strip())
                    except Exception:
                        continue
                    if r.get("chosen_action"):
                        actions.append(r["chosen_action"])
                    rd = r.get("real_diff")
                    if rd is not None and len(rd) == 0:
                        nop += 1
        entropies.append(_shannon(actions))
        uniqs.append(len(set(actions)))
        no_ops.append(nop / len(actions) if actions else 0)
        levels.append(int(g.get("levels_completed", 0)))
        s = g.get("score")
        if isinstance(s, (int, float)):
            scores.append(float(s))
        n_steps_total += int(g.get("actions", 0))
    wall_s = float(payload.get("wall_clock_seconds", 0))
    n = len(entropies) or 1
    return {
        "n_games":           len(entropies),
        "mean_entropy":      sum(entropies) / n,
        "mean_uniq_actions": sum(uniqs) / n,
        "mean_no_op_rate":   sum(no_ops) / n,
        "max_levels":        max(levels) if levels else 0,
        "mean_rhae":         sum(scores) / len(scores) if scores else 0.0,
        "wall_per_step":     wall_s / n_steps_total if n_steps_total else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(DEFAULT_V3_DIR),
                        help="path to a v3 run directory with images")
    args = parser.parse_args()
    V3_DIR = Path(args.run).resolve()

    if not V3_DIR.exists():
        raise SystemExit(f"v3 visual run not found at {V3_DIR}")

    payload = json.loads((V3_DIR / "summary.json").read_text(encoding="utf-8"))
    per_game = payload["per_game"]

    # Build per-game trace_view.md and collect existence of play.gif
    trace_views: dict[str, Path] = {}
    gif_paths: dict[str, Path] = {}
    for game_id in per_game:
        gdir = V3_DIR / game_id
        if not gdir.exists():
            continue
        trace_views[game_id] = _build_trace_view(gdir)
        gif = gdir / "play.gif"
        if gif.exists():
            gif_paths[game_id] = gif

    # Top-level summary table
    v3 = _summarize(V3_DIR)
    v1_rows = [(name, _summarize(d)) for name, d in V1_DIRS.items()]
    lines = [
        "# v3 TextAgent — visual report",
        "",
        f"Run: `{V3_DIR.relative_to(REPO_ROOT)}`. "
        f"5 games × 80 steps with **play.gif + per-step trace** for each game.",
        "",
        "## Aggregate (mean over 5 games)",
        "",
        "| Agent | entropy | uniq actions | no-op | levels | RHAE | s/step |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in v1_rows:
        if s.get("missing"):
            continue
        lines.append(
            f"| {name} | {s['mean_entropy']:.3f} | {s['mean_uniq_actions']:.1f} | "
            f"{s['mean_no_op_rate']:.1%} | {s['max_levels']} | "
            f"{s['mean_rhae']:.3f} | {s['wall_per_step']:.2f}s |"
        )
    lines.append(
        f"| **v3 TextAgent** | **{v3['mean_entropy']:.3f}** | "
        f"**{v3['mean_uniq_actions']:.1f}** | "
        f"**{v3['mean_no_op_rate']:.1%}** | "
        f"{v3['max_levels']} | "
        f"{v3['mean_rhae']:.3f} | "
        f"**{v3['wall_per_step']:.2f}s** |"
    )

    lines += ["", "## Per-game visual + trace", ""]
    for game_id, g in per_game.items():
        short = game_id.split("-")[0]
        lines += [
            f"### {short} ({game_id})",
            "",
            f"- entropy: **{g.get('action_entropy', 0):.3f}**, "
            f"unique actions: **{g.get('n_unique_actions', 0)}/7**, "
            f"no-op rate: **{g.get('no_op_rate', 0):.1%}**, "
            f"levels: **{g.get('levels_completed', 0)}/{g.get('total_levels', 0)}**, "
            f"RHAE: **{g.get('score', 0):.3f}**",
            "",
        ]
        if game_id in gif_paths:
            lines.append(f"![{short} play.gif]({game_id}/play.gif)")
        else:
            lines.append("_(play.gif not generated — re-run with --with-images)_")
        lines += [
            "",
            f"Full step-by-step trace: [{game_id}/trace_view.md]({game_id}/trace_view.md)",
            "",
            "---",
            "",
        ]

    out = V3_DIR / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Wrote {len(trace_views)} per-game trace_view.md files")


if __name__ == "__main__":
    main()
