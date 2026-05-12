"""Side-by-side comparison of two summary.json files.

Use case: after running a baseline + a validation (or two GRPO checkpoints
back-to-back), print a compact diff so the iteration trigger in
`docs/ARCHITECTURE_RL.md` §5.2 / §5.4 can be evaluated at a glance.

CLI:
    .venv/Scripts/python.exe scripts/compare_summaries.py \
        --before outputs/baseline_<ts>/summary.json \
        --after  outputs/validation_<ts>/summary.json

Outputs a printed table + writes `<after>.compare_to_<before-stem>.json`
into the after-run's directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _delta(b: float, a: float) -> str:
    d = a - b
    sign = "+" if d > 0 else ("-" if d < 0 else " ")
    return f"{sign}{abs(d):.4f}"


def _top_table(before: dict[str, Any], after: dict[str, Any]) -> str:
    rows = [
        ("mean_f1",    ">=0.30"),
        ("parse_rate", ">=0.70"),
        ("mean_rhae",  "<=0.05"),
    ]
    out = [
        f"{'metric':<14} {'before':>10} {'after':>10} {'delta':>10}   hyp",
        "-" * 64,
    ]
    for key, hyp in rows:
        b = float(before.get(key, 0.0))
        a = float(after.get(key, 0.0))
        out.append(f"{key:<14} {b:>10.4f} {a:>10.4f} {_delta(b, a):>10}   {hyp}")
    return "\n".join(out)


def _per_game_table(before: dict[str, Any], after: dict[str, Any]) -> str:
    b_pg = before.get("per_game", {}) or {}
    a_pg = after.get("per_game", {}) or {}
    keys = sorted(set(b_pg.keys()) | set(a_pg.keys()))
    if not keys:
        return "(no per_game data)"

    out = [
        f"{'game':<22} {'f1_before':>10} {'f1_after':>10} {'d_f1':>9}   "
        f"{'parse_b':>8} {'parse_a':>8} {'d_p':>8}",
        "-" * 84,
    ]
    for k in keys:
        b = b_pg.get(k, {}) or {}
        a = a_pg.get(k, {}) or {}
        f1_b = float(b.get("mean_f1", 0.0))
        f1_a = float(a.get("mean_f1", 0.0))
        p_b  = float(b.get("parse_rate", 0.0))
        p_a  = float(a.get("parse_rate", 0.0))
        out.append(
            f"{k:<22} {f1_b:>10.4f} {f1_a:>10.4f} {_delta(f1_b, f1_a):>9}   "
            f"{p_b:>8.3f} {p_a:>8.3f} {_delta(p_b, p_a):>8}"
        )
    return "\n".join(out)


def _build_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Structured diff blob suitable for committing alongside the after run."""
    pg_diff: dict[str, dict[str, Any]] = {}
    b_pg = before.get("per_game", {}) or {}
    a_pg = after.get("per_game", {}) or {}
    for k in sorted(set(b_pg.keys()) | set(a_pg.keys())):
        b = b_pg.get(k, {}) or {}
        a = a_pg.get(k, {}) or {}
        pg_diff[k] = {
            "f1_before":    float(b.get("mean_f1", 0.0)),
            "f1_after":     float(a.get("mean_f1", 0.0)),
            "delta_f1":     round(float(a.get("mean_f1", 0.0))
                                  - float(b.get("mean_f1", 0.0)), 4),
            "parse_before": float(b.get("parse_rate", 0.0)),
            "parse_after":  float(a.get("parse_rate", 0.0)),
            "delta_parse":  round(float(a.get("parse_rate", 0.0))
                                  - float(b.get("parse_rate", 0.0)), 4),
        }
    return {
        "before_path":    before.get("__path__"),
        "after_path":     after.get("__path__"),
        "before_run_kind": before.get("run_kind"),
        "after_run_kind":  after.get("run_kind"),
        "before_run_ts":   before.get("run_ts"),
        "after_run_ts":    after.get("run_ts"),
        "top":  {
            "delta_mean_f1":    round(float(after.get("mean_f1", 0.0))
                                      - float(before.get("mean_f1", 0.0)), 4),
            "delta_parse_rate": round(float(after.get("parse_rate", 0.0))
                                      - float(before.get("parse_rate", 0.0)), 4),
            "delta_mean_rhae":  round(float(after.get("mean_rhae", 0.0))
                                      - float(before.get("mean_rhae", 0.0)), 4),
        },
        "per_game": pg_diff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, help="Earlier summary.json")
    parser.add_argument("--after",  required=True, help="Later summary.json")
    parser.add_argument("--no-write", action="store_true",
                        help="Print table but don't drop a JSON diff next to <after>.")
    args = parser.parse_args()

    b_path = Path(args.before)
    a_path = Path(args.after)
    before = json.loads(b_path.read_text(encoding="utf-8"))
    after  = json.loads(a_path.read_text(encoding="utf-8"))
    before["__path__"] = str(b_path)
    after["__path__"]  = str(a_path)

    print(f"before: {b_path}  ({before.get('run_kind')} @ {before.get('run_ts')})")
    print(f"after:  {a_path}  ({after.get('run_kind')} @ {after.get('run_ts')})")
    print()
    print(_top_table(before, after))
    print()
    print(_per_game_table(before, after))

    if not args.no_write:
        diff = _build_diff(before, after)
        out_path = a_path.with_name(
            f"{a_path.stem}.compare_to_{b_path.stem}.json"
        )
        out_path.write_text(
            json.dumps(diff, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print()
        print(f"diff written -> {out_path}")


if __name__ == "__main__":
    main()
