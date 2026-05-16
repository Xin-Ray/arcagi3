"""Analyze frame-change balance across all existing trace.jsonl files.

Counts (game, action, frame_changed) tuples across every trace in
outputs/. Useful for deciding whether the dataset is balanced enough to
train a frame-change predictor on, and for picking a sampling strategy
(per-game stratified, etc).

Usage:
    .venv/Scripts/python.exe scripts/analyze_trace_balance.py
        --output outputs/reports/trace_balance.md
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _game_id_from_path(p: Path) -> str:
    """Heuristic: walk parents until we hit a dir whose name looks like a
    game id (contains a hyphen or starts with a known 4-char prefix)."""
    for part in p.parts[::-1]:
        if "-" in part and len(part) < 30:
            return part
        if part.startswith(("ar25", "bp35", "cd82", "cn04", "dc22", "ls20",
                            "tr87", "wa30", "ft09")):
            return part
    return p.parent.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/reports/trace_balance.md")
    parser.add_argument("--root", default="outputs")
    args = parser.parse_args()

    root = REPO / args.root
    traces = list(root.rglob("trace.jsonl"))

    per_game: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    per_run: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    overall = [0, 0]
    n_files = 0
    n_rows = 0

    for tp in traces:
        n_files += 1
        # Use the experiment dir name (first level under outputs/)
        rel = tp.relative_to(root)
        run = rel.parts[0]
        # Game id is recorded inside each row (`game_id` field) in most
        # schemas; fall back to path heuristic if missing.
        game = _game_id_from_path(tp)
        try:
            for line in tp.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_rows += 1
                # New v3.2 schema uses `action` + `frame_changed`; older v3
                # and ablation schemas use `chosen_action` + `real_diff`
                # (a list of [r,c,col] tuples; non-empty = changed).
                if "frame_changed" in row:
                    changed = bool(row["frame_changed"])
                    action = str(row.get("action", "?"))
                else:
                    diff = row.get("real_diff") or []
                    changed = bool(diff)
                    action = str(row.get("chosen_action", "?"))
                game_row = row.get("game_id") or game
                per_game[game_row][action][1 if changed else 0] += 1
                per_run[run][1 if changed else 0] += 1
                overall[1 if changed else 0] += 1
        except OSError:
            continue

    lines: list[str] = []
    lines.append("# Trace balance analysis")
    lines.append("")
    lines.append(f"- trace files scanned: **{n_files}**")
    lines.append(f"- total step rows: **{n_rows}**")
    if n_rows:
        cr = 100 * overall[1] / n_rows
        lines.append(f"- overall change_rate: **{cr:.1f}%** "
                     f"({overall[1]} changed / {overall[0]} no-op)")
    lines.append("")

    lines.append("## Per-run change_rate")
    lines.append("")
    lines.append("| run | rows | change_rate | balance |")
    lines.append("|---|---:|---:|---|")
    for run, (noop, changed) in sorted(
        per_run.items(), key=lambda kv: -(kv[1][0] + kv[1][1])
    ):
        n = noop + changed
        cr = 100 * changed / n if n else 0
        bal = "balanced" if 30 <= cr <= 70 else (
            "no-op heavy" if cr < 30 else "change heavy"
        )
        lines.append(f"| {run} | {n} | {cr:.1f}% | {bal} |")
    lines.append("")

    lines.append("## Per-(game, action) breakdown")
    lines.append("")
    lines.append("| game | action | n_tried | n_changed | change_rate |")
    lines.append("|---|---|---:|---:|---:|")
    for game in sorted(per_game.keys()):
        for action in sorted(per_game[game].keys()):
            noop, changed = per_game[game][action]
            n = noop + changed
            cr = 100 * changed / n if n else 0
            lines.append(f"| {game} | {action} | {n} | {changed} | {cr:.1f}% |")
    lines.append("")

    # Summary: how many balanced examples per action class do we have?
    lines.append("## Balanced sample availability per action")
    lines.append("")
    lines.append("Per action: min(n_changed, n_no_op) is the max class-balanced sample count.")
    lines.append("")
    lines.append("| action | total n | n_changed | n_no_op | max balanced |")
    lines.append("|---|---:|---:|---:|---:|")
    per_action_changed: Counter[str] = Counter()
    per_action_noop: Counter[str] = Counter()
    for game, am in per_game.items():
        for action, (noop, changed) in am.items():
            per_action_changed[action] += changed
            per_action_noop[action] += noop
    for action in sorted(set(per_action_changed) | set(per_action_noop)):
        c = per_action_changed[action]
        n = per_action_noop[action]
        lines.append(f"| {action} | {c + n} | {c} | {n} | {min(c, n)} |")
    lines.append("")

    out = REPO / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
