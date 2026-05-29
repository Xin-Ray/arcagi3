"""Parse the user-filled annotation_form.md from bench_goal_gen and
produce stats + a visualization.

Usage:
  .venv/Scripts/python.exe scripts/bench_goal_gen_score.py [--in DIR]

If --in is omitted, the most recent outputs/bench_goal_gen_* dir is used.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


_LABEL_PAT = re.compile(
    r"^\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*\*\*?([A-Z]+)\*?\*?\s*\|",
    re.IGNORECASE | re.MULTILINE)


def parse_form(form_path: Path) -> list[dict]:
    """Extract (game, frame_type, step, k, hypothesis, label) tuples."""
    text = form_path.read_text(encoding="utf-8", errors="replace")
    # Split on the per-frame header "## <game> — frame <type> (step <n>)"
    section_re = re.compile(
        r"## (\S+) — frame (\S+) \(step (\d+)\)", re.MULTILINE)
    sections = list(section_re.finditer(text))
    rows = []
    for i, m in enumerate(sections):
        game = m.group(1); ftype = m.group(2); step = int(m.group(3))
        start = m.end()
        end = sections[i+1].start() if i+1 < len(sections) else len(text)
        body = text[start:end]
        for table_m in _LABEL_PAT.finditer(body):
            k = int(table_m.group(1))
            hyp = table_m.group(2).strip()
            label = table_m.group(3).strip().upper()
            rows.append({
                "game": game, "frame_type": ftype, "step": step,
                "k_index": k, "hypothesis": hyp, "label": label,
            })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", default="")
    args = p.parse_args()

    if args.in_dir:
        d = Path(args.in_dir)
    else:
        candidates = sorted([p for p in (REPO / "outputs").glob("bench_goal_gen_*")
                              if p.is_dir()])
        if not candidates:
            print("no bench_goal_gen dir found", file=sys.stderr); sys.exit(1)
        d = candidates[-1]
    print(f"[score] reading {d}")

    form = d / "annotation_form.md"
    if not form.exists():
        print(f"  {form} missing", file=sys.stderr); sys.exit(1)
    rows = parse_form(form)
    valid_labels = {"YES", "PARTIAL", "NO", "UNSURE"}
    filled = [r for r in rows if r["label"] in valid_labels]
    blank = [r for r in rows if r["label"] not in valid_labels]
    print(f"  rows total:        {len(rows)}")
    print(f"  labels filled:     {len(filled)}")
    print(f"  labels still blank: {len(blank)}")

    if not filled:
        print("  no labels filled yet; aborting", file=sys.stderr); sys.exit(1)

    # Aggregate stats
    per_frame: dict[tuple, Counter] = {}
    overall = Counter()
    for r in filled:
        key = (r["game"], r["frame_type"])
        per_frame.setdefault(key, Counter())[r["label"]] += 1
        overall[r["label"]] += 1

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    games = sorted({k[0] for k in per_frame.keys()})
    frame_types = ["A", "B"]
    labels = ["YES", "PARTIAL", "NO", "UNSURE"]
    colors = {"YES": "#2ca02c", "PARTIAL": "#ffae42",
              "NO": "#d62728", "UNSURE": "#888"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=130)
    for ax, ft in zip(axes, frame_types):
        # frame_type stored as e.g. "A_zero_prior" -> matches if starts with ft
        bottoms = np.zeros(len(games))
        for lbl in labels:
            heights = []
            for g in games:
                ct = Counter()
                for (gg, fff), c in per_frame.items():
                    if gg == g and fff.startswith(ft):
                        ct.update(c)
                heights.append(ct.get(lbl, 0))
            heights = np.array(heights)
            ax.bar(games, heights, bottom=bottoms, color=colors[lbl],
                    edgecolor="black", linewidth=0.5, label=lbl)
            bottoms += heights
        ax.set_title(f"Frame {ft} — {'zero prior' if ft=='A' else 'with prior'}")
        ax.set_ylabel("count")
        ax.set_ylim(0, max(5, int(bottoms.max())+1))
        ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    chart_path = d / "annotation_stats.png"
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()

    # Aggregate single chart
    fig2, ax = plt.subplots(figsize=(6, 4), dpi=130)
    keys = list(overall.keys())
    vals = [overall[k] for k in keys]
    cs = [colors.get(k, "#777") for k in keys]
    ax.bar(keys, vals, color=cs, edgecolor="black", linewidth=0.5)
    for k, v in zip(keys, vals):
        ax.text(k, v + 0.2, str(v), ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Overall label distribution (all games, both frames)")
    ax.set_ylabel("count")
    chart_overall = d / "annotation_overall.png"
    plt.savefig(chart_overall, bbox_inches="tight")
    plt.close()

    # Compute pass-rate metrics
    yes_pct = 100 * overall.get("YES", 0) / len(filled)
    yes_or_partial = 100 * (overall.get("YES", 0) + overall.get("PARTIAL", 0)) / len(filled)

    # Write summary md
    lines = [
        "# T-DISCOVER bench — Annotation Results", "",
        f"Total labels: {len(filled)} / {len(rows)} (blank: {len(blank)})", "",
        "## Overall label distribution",
        "",
        "| Label | Count | % |",
        "|---|---:|---:|",
    ]
    for lbl in labels:
        c = overall.get(lbl, 0)
        lines.append(f"| {lbl} | {c} | {100*c/len(filled):.0f}% |")
    lines += [
        "", f"**Strict pass rate (YES only): {yes_pct:.0f}%**",
        f"**Lenient pass rate (YES + PARTIAL): {yes_or_partial:.0f}%**", "",
        "## Per (game, frame_type) breakdown", "",
        "| Game | Frame | YES | PARTIAL | NO | UNSURE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (g, ft), ct in sorted(per_frame.items()):
        ft_short = ft.split("_")[0]
        lines.append(f"| {g} | {ft_short} | {ct.get('YES',0)} | "
                      f"{ct.get('PARTIAL',0)} | {ct.get('NO',0)} | "
                      f"{ct.get('UNSURE',0)} |")
    lines += [
        "", "## Charts", "",
        f"![per-frame breakdown](./{chart_path.name})",
        f"![overall distribution](./{chart_overall.name})",
    ]
    (d / "annotation_summary.md").write_text("\n".join(lines),
                                              encoding="utf-8")
    print(f"\n[done] wrote {d/'annotation_summary.md'}")
    print(f"       charts: {chart_path.name}, {chart_overall.name}")
    print(f"       YES rate: {yes_pct:.0f}%")
    print(f"       YES+PARTIAL rate: {yes_or_partial:.0f}%")


if __name__ == "__main__":
    main()
