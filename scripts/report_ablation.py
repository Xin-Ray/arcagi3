"""Aggregate an ablation directory into the §2 comparison tables.

Walks `<run>/*/summary.json`, hands them to `arc_agent.report.aggregate`,
prints the RHAE table + per-game breakdown + decision-gate verdict, and
saves the same as `report.md` + `report.json` in the run dir.

Usage:
    .venv/Scripts/python.exe scripts/report_ablation.py --run outputs/ablation_<ts>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arc_agent.report import (  # noqa: E402
    aggregate,
    decision_gate,
    load_summary,
    render_per_game,
    render_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Path to outputs/ablation_<ts>")
    parser.add_argument(
        "--threshold", type=float, default=0.02,
        help="RHAE delta above A1 required to call a scaffold a winner (§5).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run)
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    summaries = []
    for sub in sorted(run_dir.iterdir()):
        s = sub / "summary.json"
        if s.exists():
            summaries.append(load_summary(s))
    if not summaries:
        raise SystemExit(f"no summary.json files under {run_dir}")

    rows = aggregate(summaries)

    print("\n=== RHAE table ===")
    table = render_table(rows)
    print(table)

    print("\n=== Per-game RHAE ===")
    per_game = render_per_game(rows)
    print(per_game)

    print("\n=== Decision gate (§5) ===")
    gate = decision_gate(rows, threshold=args.threshold)
    print(gate)

    md_path = run_dir / "report.md"
    md_path.write_text(
        f"# Ablation report — {run_dir.name}\n\n"
        f"## RHAE table\n\n```\n{table}\n```\n\n"
        f"## Per-game RHAE\n\n```\n{per_game}\n```\n\n"
        f"## Decision gate\n\n{gate}\n",
        encoding="utf-8",
    )
    json_path = run_dir / "report.json"
    json_path.write_text(
        json.dumps(
            {"rows": [r.as_dict() for r in rows], "decision_gate": gate},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote: {md_path}\nWrote: {json_path}")


if __name__ == "__main__":
    main()
