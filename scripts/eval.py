"""
Batch evaluator for ARC-AGI-3 agents.

Runs an agent over a list of games (default = all 25 demo games visible to the
current ARC_API_KEY), N episodes per game, under a single shared scorecard.
Each (game, episode) appends one jsonl row to runs/<timestamp>_<tag>.jsonl,
and a final summary is printed and stored as the last row with `__summary__: true`.

Examples:
    .venv/Scripts/python.exe scripts/eval.py                       # RandomAgent, all games, 1 ep each
    .venv/Scripts/python.exe scripts/eval.py --episodes 3
    .venv/Scripts/python.exe scripts/eval.py --games ls20,vc33
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

from arc_agi import Arcade

from arc_agent.agents.llm import LLMAgent
from arc_agent.agents.random import RandomAgent
from arc_agent.llm import LLMClient
from arc_agent.runner import play_one as _play_one

load_dotenv(REPO_ROOT / ".env")

RUNS_DIR = REPO_ROOT / "outputs" / "runs"


def _check_key() -> None:
    key = os.getenv("ARC_API_KEY", "")
    if not key or key.startswith("your_"):
        raise RuntimeError(
            "ARC_API_KEY missing or still the .env.example placeholder. "
            "Put a real key in .env (see https://arcprize.org/api-keys)."
        )


_AGENT_MODELS = {
    "llm": "claude-opus-4-7",
    "llm-haiku": "claude-haiku-4-5",
}

_AGENTS = {
    "random": lambda _client: RandomAgent(),
    "llm": lambda client: LLMAgent(client),
    "llm-haiku": lambda client: LLMAgent(client),
}


def play_one(
    arc: Arcade,
    agent_name: str,
    game_id: str,
    card_id: str,
    max_actions: int,
    shared_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Play one episode of one game and return per-episode metrics."""
    if agent_name not in _AGENTS:
        return {"game_id": game_id, "error": f"unknown agent: {agent_name}", "actions": 0}
    agent = _AGENTS[agent_name](shared_client)
    result = _play_one(arc, agent, game_id, card_id, max_actions=max_actions)
    return {"game_id": game_id, **result}


def _extract_env_summary(env_card: dict[str, Any]) -> dict[str, Any]:
    """Pull the meaningful fields out of one scorecard.environments[i] entry."""
    runs = env_card.get("runs") or []
    last_run = runs[-1] if runs else {}
    return {
        "id": env_card.get("id"),
        "score": env_card.get("score"),
        "level_scores": last_run.get("level_scores"),
        "level_actions": last_run.get("level_actions"),
        "level_baseline_actions": last_run.get("level_baseline_actions"),
        "levels_completed": env_card.get("levels_completed"),
        "level_count": env_card.get("level_count"),
        "completed": env_card.get("completed"),
        "actions": env_card.get("actions"),
    }


def main() -> None:
    _check_key()

    parser = argparse.ArgumentParser(description="ARC-AGI-3 batch evaluator")
    parser.add_argument("--agent", default="random", choices=list(_AGENTS.keys()))
    parser.add_argument(
        "--games",
        default="",
        help="Comma-separated game IDs. Empty = all available demo games.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--max-actions",
        type=int,
        default=80,
        help="Per-episode safety cap (matches official scaffold default).",
    )
    parser.add_argument("--tag", default="random_baseline")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--budget",
        type=float,
        default=15.0,
        help="Hard stop if estimated LLM API cost (USD) exceeds this. Default $15.",
    )
    args = parser.parse_args()

    # Shared LLM client so cost accumulates across games; None for non-LLM agents.
    model = _AGENT_MODELS.get(args.agent)
    shared_client: LLMClient | None = LLMClient(model=model) if model else None

    arc = Arcade()
    # arc.get_environments() returns list[EnvironmentInfo] (pydantic), each has .game_id
    # like "ls20-9607627b" plus baseline_actions / tags / class_name. We keep a parallel
    # dict of metadata keyed by full game_id for later writes.
    env_infos = arc.get_environments() or []
    available_ids = [e.game_id for e in env_infos]
    meta_by_id = {e.game_id: e for e in env_infos}
    print(f"Available games: {len(available_ids)} -> {available_ids}")

    if args.games:
        wanted = [g.strip() for g in args.games.split(",")]
        # User can pass short id ("ls20") or full ("ls20-9607627b"); prefix-match.
        games = [gid for gid in available_ids if any(gid.startswith(w) for w in wanted)]
    else:
        games = list(available_ids)
    if not games:
        raise RuntimeError("No games to play. Check --games or your API key access.")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else RUNS_DIR / f"{ts}_{args.tag}.jsonl"

    card_id = arc.open_scorecard(tags=[args.tag, args.agent])
    print(f"Scorecard {card_id} opened. Writing jsonl to {out_path}")

    rows: list[dict[str, Any]] = []
    t_start = time.time()
    budget_exceeded = False
    with out_path.open("w", encoding="utf-8") as f:
        for gi, game_id in enumerate(games, 1):
            if budget_exceeded:
                break
            for ep in range(args.episodes):
                t0 = time.time()
                try:
                    result = play_one(arc, args.agent, game_id, card_id, args.max_actions, shared_client)
                except Exception as e:
                    result = {"game_id": game_id, "error": f"{type(e).__name__}: {e}"}
                meta = meta_by_id.get(game_id)
                cost_so_far = shared_client.estimated_cost_usd() if shared_client else 0.0
                row = {
                    "ts": time.time(),
                    "game_index": gi,
                    "game_id": game_id,
                    "episode": ep,
                    "wall_seconds": round(time.time() - t0, 2),
                    "baseline_actions": meta.baseline_actions if meta else None,
                    "tags": meta.tags if meta else None,
                    "class_name": meta.class_name if meta else None,
                    "estimated_cost_usd_cumulative": round(cost_so_far, 4),
                    **result,
                }
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                tag = result.get("error") or f"steps={result.get('actions')} state={result.get('final_state')} levels={result.get('levels_completed')}/{result.get('total_levels')}"
                cost_str = f" | ~${cost_so_far:.3f} spent" if shared_client else ""
                print(f"[{gi}/{len(games)}] {game_id} ep{ep}: {tag} ({row['wall_seconds']}s){cost_str}")
                if shared_client and cost_so_far >= args.budget:
                    print(f"\n*** Budget cap ${args.budget:.2f} reached (actual: ${cost_so_far:.3f}). Stopping. ***")
                    budget_exceeded = True
                    break

        scorecard = arc.close_scorecard(card_id)
        scorecard_dump = scorecard.model_dump() if scorecard is not None else {}

        # Per-environment scoreboard, keyed by env id
        env_summaries = [
            _extract_env_summary(e) for e in (scorecard_dump.get("environments") or [])
        ]
        score_by_envid = {s["id"]: s for s in env_summaries}

        # Aggregate stats
        scores = [s["score"] for s in env_summaries if isinstance(s.get("score"), (int, float))]
        passed = sum(1 for s in env_summaries if (s.get("levels_completed") or 0) > 0)
        mean_score = sum(scores) / len(scores) if scores else 0.0

        final_cost = shared_client.estimated_cost_usd() if shared_client else 0.0
        summary = {
            "__summary__": True,
            "ts": time.time(),
            "agent": args.agent,
            "tag": args.tag,
            "scorecard_id": card_id,
            "n_games": len(env_summaries),
            "n_episodes_per_game": args.episodes,
            "max_actions": args.max_actions,
            "wall_seconds_total": round(time.time() - t_start, 2),
            "mean_score": round(mean_score, 4),
            "pass_rate_level1": round(passed / len(env_summaries), 4) if env_summaries else 0.0,
            "estimated_cost_usd": round(final_cost, 4),
            "llm_tokens": {
                "input": shared_client.total_input_tokens if shared_client else 0,
                "output": shared_client.total_output_tokens if shared_client else 0,
                "cache_read": shared_client.total_cache_read_tokens if shared_client else 0,
                "cache_creation": shared_client.total_cache_creation_tokens if shared_client else 0,
            },
            "totals": {
                "total_actions": scorecard_dump.get("total_actions"),
                "total_levels_completed": scorecard_dump.get("total_levels_completed"),
                "total_levels": scorecard_dump.get("total_levels"),
            },
            "per_game": score_by_envid,
        }
        f.write(json.dumps(summary) + "\n")

    print()
    print(f"=== {args.tag} on {len(env_summaries)} games, {args.episodes} ep each ===")
    print(f"  mean score (overall RHAE):      {summary['mean_score']}")
    print(f"  pass rate (>=1 level cleared):  {summary['pass_rate_level1']}")
    print(f"  total actions:                  {summary['totals']['total_actions']}")
    print(f"  wall:                           {summary['wall_seconds_total']}s")
    if shared_client:
        tk = summary["llm_tokens"]
        print(f"  LLM tokens: in={tk['input']} out={tk['output']} cache_read={tk['cache_read']} cache_creation={tk['cache_creation']}")
        print(f"  estimated cost:                 ~${summary['estimated_cost_usd']:.4f} USD")
    print(f"  scorecard:                      {card_id}")
    print(f"  jsonl:                          {out_path}")


if __name__ == "__main__":
    main()
