"""Run v3 TextAgent on G_base games, capture all v3 metrics into a
summary.json suitable for the v3 report builder.

Adds two NEW fields beyond the baseline schema:
  action_entropy:           per-game Shannon entropy of chosen_action
  unique_frame_hashes:      coverage proxy

Usage:
  .venv/Scripts/python.exe scripts/run_v3_eval.py [--games ...] [--max-actions N]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from arc_agi import Arcade   # noqa: E402

from arc_agent.baseline import play_one_with_trace   # noqa: E402
from arc_agent.eval_split import write_summary   # noqa: E402

SPLIT_PATH = REPO_ROOT / "data" / "splits" / "demo_555.json"
OUT_ROOT = REPO_ROOT / "outputs"


def _shannon_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    out = 0.0
    for c in counts:
        if c == 0:
            continue
        p = c / total
        out -= p * math.log(p)
    return out


def _compute_per_game_extra(game_dir: Path) -> dict[str, float]:
    """Read trace.jsonl and compute action_entropy + state_coverage."""
    trace_path = game_dir / "trace.jsonl"
    actions: list[str] = []
    frame_hashes: set[int] = set()
    no_op = 0
    if not trace_path.exists():
        return {}
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            ca = row.get("chosen_action")
            if ca:
                actions.append(ca)
            rd = row.get("real_diff")
            if rd is not None and len(rd) == 0:
                no_op += 1
            # Frame hash approximation: hash sorted real_diff tuples
            # (not perfect — replays through the SDK would give exact grids
            # but this is a cheap proxy for "did the agent reach novel states")
            if rd is not None:
                frame_hashes.add(hash(tuple(tuple(x) for x in sorted(rd))))
    counts = list(Counter(actions).values())
    entropy = _shannon_entropy(counts)
    coverage = len(frame_hashes)
    no_op_rate = no_op / len(actions) if actions else 0.0
    return {
        "action_entropy": round(entropy, 4),
        "n_unique_actions": len(set(actions)),
        "unique_frame_hashes": coverage,
        "no_op_rate": round(no_op_rate, 4),
    }


def _check_key() -> None:
    key = os.getenv("ARC_API_KEY", "")
    if not key or key.startswith("your_"):
        raise RuntimeError("ARC_API_KEY missing/placeholder. See .env")


def _resolve_full_ids(arc, requested):
    env_infos = arc.get_environments() or []
    avail = [e.game_id for e in env_infos]
    out = []
    for r in requested:
        m = next((a for a in avail if a.startswith(r)), None)
        if m is None:
            raise RuntimeError(f"game id '{r}' not found")
        out.append(m)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default="", help="comma list; default = G_base")
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="", help="output dir; default = outputs/v3_eval_<ts>")
    parser.add_argument("--no-images", action="store_true", default=True)
    parser.add_argument("--with-images", dest="no_images", action="store_false")
    args = parser.parse_args()

    _check_key()

    if args.games:
        requested = [g.strip() for g in args.games.split(",") if g.strip()]
    else:
        if not SPLIT_PATH.exists():
            raise SystemExit(f"split file missing: {SPLIT_PATH}")
        requested = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["g_base"]

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else OUT_ROOT / f"v3_eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {out_dir}")

    arc = Arcade()
    games = _resolve_full_ids(arc, requested)
    print(f"Games: {games}")

    print("Loading Qwen2.5-VL-3B (4-bit) for text-only TextAgent...")
    from arc_agent.agents.text_agent import TextAgent
    from arc_agent.vlm_backbone import HFBackbone
    t0 = time.time()
    backbone = HFBackbone.load()
    print(f"Loaded in {round(time.time()-t0,1)}s")

    agent = TextAgent(backbone=backbone, seed=args.seed,
                      max_actions=args.max_actions)

    card_id = arc.open_scorecard(tags=["v3_eval", "g_base"])
    print(f"Scorecard {card_id} opened")

    per_game: dict[str, Any] = {}
    f1_pool: list[float] = []
    parse_pool: list[float] = []
    t_run = time.time()
    for gi, game_id in enumerate(games, 1):
        agent.reset()
        env = arc.make(game_id, scorecard_id=card_id)
        if env is None:
            print(f"[{gi}/{len(games)}] {game_id}: arc.make returned None — skip")
            per_game[game_id] = {"error": "arc.make returned None"}
            continue
        game_dir = out_dir / game_id
        m = play_one_with_trace(
            env, agent,
            run_dir=game_dir,
            game_id=game_id,
            max_actions=args.max_actions,
            write_images=not args.no_images,
        )
        per_game[game_id] = m.as_dict()
        per_game[game_id].update(_compute_per_game_extra(game_dir))
        f1_pool.append(m.mean_f1)
        parse_pool.append(m.parse_rate)
        e = per_game[game_id]
        print(f"[{gi}/{len(games)}] {game_id}: "
              f"actions={m.actions} final={m.final_state} "
              f"entropy={e.get('action_entropy', 0):.3f} "
              f"unique_actions={e.get('n_unique_actions', 0)} "
              f"no_op={e.get('no_op_rate', 0):.2%} "
              f"levels={m.levels_completed}/{m.total_levels}")

    scorecard = arc.close_scorecard(card_id)
    sc_dump = scorecard.model_dump() if scorecard is not None else {}
    env_cards = {e["id"]: e for e in (sc_dump.get("environments") or [])}
    rhae_pool: list[float] = []
    for game_id, m in per_game.items():
        if "error" in m:
            continue
        card = env_cards.get(game_id, {})
        m["score"] = card.get("score")
        m["level_count"] = card.get("level_count")
        if isinstance(m.get("score"), (int, float)):
            rhae_pool.append(float(m["score"]))

    wall = round(time.time() - t_run, 2)
    mean_entropy = sum(m.get("action_entropy", 0)
                       for m in per_game.values()) / len(per_game) if per_game else 0
    mean_rhae = sum(rhae_pool) / len(rhae_pool) if rhae_pool else 0.0

    write_summary(
        out_dir / "summary.json",
        run_kind="v3_eval",
        games=list(per_game.keys()),
        n_episodes_per_game=1,
        wall_clock_seconds=wall,
        mean_f1=round(sum(f1_pool) / len(f1_pool), 4) if f1_pool else 0.0,
        parse_rate=round(sum(parse_pool) / len(parse_pool), 4) if parse_pool else 0.0,
        mean_rhae=round(mean_rhae, 4),
        per_game=per_game,
        split=str(SPLIT_PATH.relative_to(REPO_ROOT)),
        agent="v3:text_agent",
        scorecard_id=card_id,
        max_actions=args.max_actions,
        mean_action_entropy=round(mean_entropy, 4),
        notes="v3 TextAgent (scipy + temporal classifier + text-only Qwen)",
    )
    print(f"\n=== v3 eval summary ({wall}s) ===")
    print(f"  mean action entropy: {mean_entropy:.4f}  (target >= 1.5)")
    print(f"  mean RHAE:           {mean_rhae:.4f}  (target > 0)")
    print(f"  summary:             {out_dir}/summary.json")


if __name__ == "__main__":
    main()
