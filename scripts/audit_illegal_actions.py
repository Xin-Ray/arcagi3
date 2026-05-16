"""Scan v3_visual_full traces for illegal-action evidence.

Counts three categories per game:
  1. response_action != executed_action  -> model said X, runner ran Y (fallback)
  2. response had no ACTION token        -> totally unparseable
  3. response had ACTION token but mismatched legal set at the time
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN = REPO_ROOT / "outputs" / "v3_visual_full"
ACTION_RE = re.compile(r"\bACTION([1-7])\b", re.IGNORECASE)


def _extract_legal_actions_from_prompt(prompt: str) -> set[str]:
    """Pull the [STATUS]:legal_actions line."""
    m = re.search(r"legal actions: ([^\n]+)", prompt)
    if not m:
        return set()
    return {a.strip() for a in m.group(1).split(",")}


def main() -> None:
    if not RUN.exists():
        raise SystemExit(f"missing {RUN}")
    game_dirs = sorted(d for d in RUN.iterdir() if d.is_dir())

    total = {
        "n_steps": 0,
        "mismatch_count": 0,
        "no_action_token": 0,
        "model_picked_illegal": 0,
        "model_picked_action6_no_coords": 0,
    }
    per_game: dict[str, dict] = {}

    for gdir in game_dirs:
        trace = gdir / "trace.jsonl"
        if not trace.exists():
            continue
        g_stats = {
            "n_steps": 0,
            "mismatch_count": 0,
            "no_action_token": 0,
            "model_picked_illegal": 0,
            "model_picked_action6_no_coords": 0,
            "samples_illegal": [],
            "samples_mismatch": [],
        }
        with trace.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line.strip())
                except Exception:
                    continue
                g_stats["n_steps"] += 1
                resp = (row.get("response_raw") or "").strip()
                executed = row.get("chosen_action") or ""

                m = ACTION_RE.search(resp)
                if not m:
                    g_stats["no_action_token"] += 1
                    continue

                response_action = f"ACTION{m.group(1)}"
                legal = _extract_legal_actions_from_prompt(row.get("prompt", ""))

                if response_action not in legal:
                    g_stats["model_picked_illegal"] += 1
                    if len(g_stats["samples_illegal"]) < 5:
                        g_stats["samples_illegal"].append({
                            "step": row.get("step"),
                            "response_action": response_action,
                            "legal": sorted(legal),
                            "executed": executed,
                            "raw": resp[:80],
                        })

                if response_action != executed:
                    g_stats["mismatch_count"] += 1
                    if len(g_stats["samples_mismatch"]) < 5:
                        g_stats["samples_mismatch"].append({
                            "step": row.get("step"),
                            "response_action": response_action,
                            "executed": executed,
                            "raw": resp[:80],
                        })

                # ACTION6 needs coords
                if response_action == "ACTION6":
                    coord_match = re.search(r"\b6\b[^\d-]+(\d+)\D+(\d+)", resp)
                    if not coord_match:
                        g_stats["model_picked_action6_no_coords"] += 1

        per_game[gdir.name] = g_stats
        for k in ("n_steps", "mismatch_count", "no_action_token",
                  "model_picked_illegal", "model_picked_action6_no_coords"):
            total[k] += g_stats[k]

    # Report
    print("=" * 72)
    print("ILLEGAL ACTION AUDIT — v3_visual_full")
    print("=" * 72)
    print()
    print(f"{'Game':<22} {'steps':>6} {'illegal':>8} {'mismatch':>9} "
          f"{'no_token':>9} {'ACT6_no_coords':>15}")
    for game_id, g in per_game.items():
        print(f"{game_id:<22} {g['n_steps']:>6} "
              f"{g['model_picked_illegal']:>8} {g['mismatch_count']:>9} "
              f"{g['no_action_token']:>9} {g['model_picked_action6_no_coords']:>15}")
    print(f"{'TOTAL':<22} {total['n_steps']:>6} "
          f"{total['model_picked_illegal']:>8} {total['mismatch_count']:>9} "
          f"{total['no_action_token']:>9} {total['model_picked_action6_no_coords']:>15}")
    print()

    # Dump JSON for the doc
    out = RUN / "illegal_audit.json"
    out.write_text(json.dumps(per_game, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Detailed samples saved to {out}")
    print()
    print("=== Sample illegal-action picks ===")
    for game_id, g in per_game.items():
        if not g["samples_illegal"]:
            continue
        print(f"\n--- {game_id} ---")
        for s in g["samples_illegal"]:
            print(f"  step {s['step']:>2}: model said {s['response_action']} "
                  f"(legal: {s['legal']}) -> runner ran {s['executed']}")
            print(f"           raw: {s['raw']!r}")


if __name__ == "__main__":
    main()
