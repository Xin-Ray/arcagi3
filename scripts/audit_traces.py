"""Comprehensive audit across all 5 v3 traces.

Reports per-game and aggregate:
  - response_raw shape categories (already covered, but full picture)
  - frame_hash repetition (how often does agent revisit same state)
  - no-op streaks (max consecutive no-op steps)
  - action distribution + bias (which action picked most)
  - 'UI-like' elements in scipy extracts (long-thin bars touching edges)
  - state transitions: did the agent ever reach a state seen only once?
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN = REPO_ROOT / "outputs" / "v3_visual_full"
ACTION_RE = re.compile(r"\bACTION([1-7])\b", re.IGNORECASE)


def _trace_rows(game_dir: Path) -> list[dict]:
    fp = game_dir / "trace.jsonl"
    if not fp.exists():
        return []
    rows: list[dict] = []
    with fp.open(encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line.strip()))
            except Exception:
                continue
    return rows


def _frame_hash(row: dict) -> int:
    """Hash the real_diff list. Same state -> same hash (approx)."""
    rd = row.get("real_diff") or []
    return hash(tuple(tuple(x) for x in sorted(rd)))


def _audit_one(game_dir: Path) -> dict:
    rows = _trace_rows(game_dir)
    if not rows:
        return {}

    # Action distribution
    actions = [r.get("chosen_action") for r in rows if r.get("chosen_action")]
    action_counts = Counter(actions)
    action_share = {a: c / len(actions) for a, c in action_counts.items()}
    most_picked = action_counts.most_common(1)[0]   # (action, n)

    # Response shape
    response_shapes = Counter()
    for r in rows:
        raw = (r.get("response_raw") or "").strip()
        m = ACTION_RE.search(raw)
        if not m:
            response_shapes["(no ACTION token)"] += 1
            continue
        a = f"ACTION{m.group(1)}"
        has_coords = bool(re.search(rf"\bACTION{m.group(1)}\b\s+\d+\s+\d+",
                                    raw, re.IGNORECASE))
        if a == "ACTION6":
            response_shapes[f"{a} + coords" if has_coords else f"{a} no coords"] += 1
        else:
            response_shapes[f"{a} + coords" if has_coords else f"{a} only"] += 1

    # No-op streaks
    streak = 0
    max_streak = 0
    n_noop = 0
    for r in rows:
        rd = r.get("real_diff") or []
        if len(rd) == 0:
            streak += 1
            n_noop += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # Frame_hash repetition (proxy: how many unique real_diff hashes)
    hashes = [_frame_hash(r) for r in rows]
    hash_counts = Counter(hashes)
    n_repeated_hashes = sum(1 for h, c in hash_counts.items() if c > 1)
    n_unique = len(hash_counts)
    most_revisited = max(hash_counts.values())

    return {
        "n_steps":             len(rows),
        "n_unique_actions":    len(action_counts),
        "most_picked_action":  most_picked[0],
        "most_picked_share":   round(most_picked[1] / len(actions), 3) if actions else 0,
        "action_counts":       dict(action_counts),
        "response_shapes":     dict(response_shapes),
        "max_noop_streak":     max_streak,
        "n_noop":              n_noop,
        "n_unique_hashes":     n_unique,
        "n_repeated_hashes":   n_repeated_hashes,
        "max_revisits_of_one_state": most_revisited,
    }


def main() -> None:
    if not RUN.exists():
        raise SystemExit(f"missing {RUN}")
    per_game: dict[str, dict] = {}
    for gdir in sorted(d for d in RUN.iterdir() if d.is_dir()):
        per_game[gdir.name] = _audit_one(gdir)

    print(f"{'='*80}")
    print("v3 visual full — comprehensive trace audit")
    print(f"{'='*80}\n")

    print(f"{'Game':<22} {'most_action':<10} {'share':>6}  "
          f"{'noop_max':>8} {'noop_total':>10}  {'uniq_hash':>9} {'max_visits':>10}")
    for game, a in per_game.items():
        print(f"{game:<22} "
              f"{a['most_picked_action']:<10} "
              f"{a['most_picked_share']:>6.1%}  "
              f"{a['max_noop_streak']:>8} {a['n_noop']:>10}  "
              f"{a['n_unique_hashes']:>9} {a['max_revisits_of_one_state']:>10}")
    print()

    # Action share table
    print("=== Per-game action distribution ===")
    print(f"{'Game':<22} ACTION1 ACTION2 ACTION3 ACTION4 ACTION5 ACTION6 ACTION7")
    for game, a in per_game.items():
        c = a["action_counts"]
        line = f"{game:<22}"
        for i in range(1, 8):
            line += f"  {c.get(f'ACTION{i}', 0):5d} "
        print(line)
    print()

    # Response shapes
    print("=== Per-game response shapes ===")
    for game, a in per_game.items():
        print(f"\n--- {game} ---")
        for shape, n in sorted(a["response_shapes"].items(), key=lambda x: -x[1]):
            pct = 100 * n / a["n_steps"]
            print(f"  {shape:<22} {n:3d}  {pct:5.1f}%")

    out = RUN / "trace_audit.json"
    out.write_text(json.dumps(per_game, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetailed audit saved to {out}")


if __name__ == "__main__":
    main()
