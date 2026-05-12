"""Classify parse failures across a baseline / validation run.

Reads every `trace.jsonl` under a run directory and buckets the
`parse_ok=false` rows by failure mode:

- empty_action        : `chosen_action` field present but ""
- missing_action      : `chosen_action` field absent from JSON
- illegal_action      : `chosen_action` had a valid ACTION name but it
                         wasn't in `available_actions` for that step
- truncated           : response doesn't end with `}` or `` ``` `` —
                         likely cut off at `max_new_tokens`
- other               : parsed but rejected for another reason
                         (malformed coords for ACTION6, etc.)

CLI:
    .venv/Scripts/python.exe scripts/analyze_failures.py \
        --run outputs/baseline_<ts>
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

_ACTION_RE = re.compile(r'"chosen_action"\s*:\s*"([^"]*)"')
_ACTION_NAME_RE = re.compile(r"\s*(ACTION[1-7])\b")


def _classify(row: dict[str, Any]) -> str:
    raw = row.get("response_raw") or ""
    if not raw:
        return "no_response"

    # Truncation: should end with } or ``` after stripping whitespace
    stripped = raw.rstrip()
    if not (stripped.endswith("}") or stripped.endswith("```")):
        return "truncated"

    m = _ACTION_RE.search(raw)
    if not m:
        return "missing_action"
    val = m.group(1)
    if not val.strip():
        return "empty_action"
    if not _ACTION_NAME_RE.match(val.upper()):
        return "garbled_action"

    # The action name parses cleanly — it must have been illegal
    return "illegal_action"


def analyze_run(run_dir: Path) -> dict[str, Any]:
    traces = sorted(run_dir.glob("*/trace.jsonl"))
    if not traces:
        raise FileNotFoundError(f"no trace.jsonl files under {run_dir}")

    per_game: dict[str, dict[str, Any]] = {}
    global_counts: Counter[str] = Counter()
    total_rows = 0
    total_fail = 0

    for tp in traces:
        game_id = tp.parent.name
        rows = [json.loads(ln) for ln in tp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        fails = [r for r in rows if not r.get("parse_ok")]
        cls = Counter(_classify(r) for r in fails)
        per_game[game_id] = {
            "n_rows": len(rows),
            "n_fail": len(fails),
            "fail_rate": round(len(fails) / len(rows), 4) if rows else 0.0,
            "fail_modes": dict(cls),
        }
        global_counts.update(cls)
        total_rows += len(rows)
        total_fail += len(fails)

    return {
        "run_dir": str(run_dir),
        "n_games": len(traces),
        "n_rows_total": total_rows,
        "n_fail_total": total_fail,
        "fail_rate_overall": round(total_fail / total_rows, 4) if total_rows else 0.0,
        "fail_modes_global": dict(global_counts),
        "per_game": per_game,
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"Run: {report['run_dir']}")
    print(f"Games: {report['n_games']}   "
          f"steps: {report['n_rows_total']}   "
          f"parse_fail: {report['n_fail_total']} "
          f"({report['fail_rate_overall']:.1%})")
    print()
    print(f"{'mode':<18} {'count':>7}")
    print("-" * 27)
    for mode, n in sorted(report["fail_modes_global"].items(),
                          key=lambda kv: -kv[1]):
        print(f"{mode:<18} {n:>7}")
    print()
    print(f"{'game':<22} {'rows':>5} {'fail':>5} {'rate':>6}  modes")
    print("-" * 70)
    for g, pg in report["per_game"].items():
        modes = ", ".join(f"{k}={v}" for k, v in pg["fail_modes"].items())
        print(f"{g:<22} {pg['n_rows']:>5} {pg['n_fail']:>5} "
              f"{pg['fail_rate']:>6.1%}  {modes}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run directory containing per-game trace.jsonl files")
    parser.add_argument("--json", action="store_true", help="Also print the report as JSON")
    args = parser.parse_args()

    report = analyze_run(Path(args.run))
    _print_report(report)
    if args.json:
        print()
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
