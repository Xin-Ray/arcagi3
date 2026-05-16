"""Unified ablation runner — `docs/ARCHITECTURE_AGENTS.md` §2 + §3 Step 8.

Runs six agents on the same G_base subset under one scorecard each:

    R0   RandomAgent
    A1   VLMAgentLite (history=0)
    A1+h VLMAgentLite (history=5)
    A2   VLMAgent     (refined baseline)
    A3   PlayReflectAgent
    A4   PlayReflectMistakesAgent

Single Qwen backbone is reused across the five VLM rows so we pay the model
load + 4-bit quantization cost once. R0 needs no backbone.

Per-(agent, game) trace.jsonl + summary.json land under
`<output>/<agent_tag>/<game_id>/`. The flat summary.json files are what
`scripts/report_ablation.py` aggregates.

This script does I/O orchestration only — all reusable logic is in
`arc_agent.baseline.play_one_with_trace`, `arc_agent.eval_split.write_summary`,
and the agent classes in `arc_agent/agents/`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from arc_agi import Arcade  # noqa: E402

from arc_agent.baseline import play_one_with_trace  # noqa: E402
from arc_agent.eval_split import write_summary  # noqa: E402

SPLIT_PATH = REPO_ROOT / "data" / "splits" / "demo_555.json"
OUTPUTS_ROOT = REPO_ROOT / "outputs"


# Default ablation roster. (tag, factory) — factory is called with shared
# kwargs and must return an Agent.
ABLATION_AGENTS = (
    "random",
    "lite",
    "lite_h5",
    "full",
    "reflect",
    "reflect_mistakes",
)


def _check_key() -> None:
    key = os.getenv("ARC_API_KEY", "")
    if not key or key.startswith("your_"):
        raise RuntimeError(
            "ARC_API_KEY missing or .env.example placeholder — put a real "
            "key in .env (see https://arcprize.org/api-keys)."
        )


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _load_g_base(override: str) -> list[str]:
    if override:
        return [g.strip() for g in override.split(",") if g.strip()]
    if not SPLIT_PATH.exists():
        raise RuntimeError(
            f"Frozen split missing: {SPLIT_PATH}. "
            "Run `scripts/freeze_splits.py` first."
        )
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["g_base"]


def _resolve_full_game_ids(arc: Arcade, requested: list[str]):
    env_infos = arc.get_environments() or []
    available = [e.game_id for e in env_infos]
    resolved: list[str] = []
    for r in requested:
        match = next((a for a in available if a.startswith(r)), None)
        if match is None:
            raise RuntimeError(
                f"Game id '{r}' not found among SDK games."
            )
        resolved.append(match)
    return resolved


def _build_agent(tag: str, *, backbone: Any, seed: int):
    if tag == "random":
        from arc_agent.agents.random import RandomAgent
        return RandomAgent(seed=seed)
    if tag == "lite":
        from arc_agent.agents.vlm_lite import VLMAgentLite
        return VLMAgentLite(backbone=backbone, seed=seed, history=0)
    if tag == "lite_h5":
        from arc_agent.agents.vlm_lite import VLMAgentLite
        return VLMAgentLite(backbone=backbone, seed=seed, history=5)
    if tag == "full":
        from arc_agent.agents.vlm import VLMAgent
        return VLMAgent(backbone=backbone, seed=seed,
                        max_new_tokens=768, temperature=0.0)
    if tag == "reflect":
        from arc_agent.agents.reflect import PlayReflectAgent
        return PlayReflectAgent(backbone=backbone, seed=seed)
    if tag == "reflect_mistakes":
        from arc_agent.agents.reflect import PlayReflectMistakesAgent
        return PlayReflectMistakesAgent(backbone=backbone, seed=seed)
    raise ValueError(f"unknown agent tag {tag!r}")


def _make_backbone(needed: list[str], dry_run: bool):
    """Load the Qwen backbone once, only if any agent in the roster needs it."""
    if dry_run:
        return None
    if all(t == "random" for t in needed):
        return None
    from arc_agent.vlm_backbone import HFBackbone
    return HFBackbone.load()


def _run_one_agent(
    *,
    tag: str,
    arc: Arcade,
    games: list[str],
    backbone: Any,
    seed: int,
    out_root: Path,
    max_actions: int,
    write_images: bool,
    episodes: int,
) -> dict[str, Any]:
    """Run one agent across `games` under its own scorecard. Returns summary."""
    agent_dir = out_root / tag
    agent_dir.mkdir(parents=True, exist_ok=True)

    agent = _build_agent(tag, backbone=backbone, seed=seed)
    card_id = arc.open_scorecard(tags=[f"ablation_{tag}", "g_base"])

    f1_pool: list[float] = []
    parse_pool: list[float] = []
    per_game_metrics: dict[str, dict[str, Any]] = {}
    t0 = time.time()

    for gi, game_id in enumerate(games, 1):
        if hasattr(agent, "reset"):
            agent.reset()
        env = arc.make(game_id, scorecard_id=card_id)
        if env is None:
            per_game_metrics[game_id] = {"error": "arc.make returned None"}
            continue
        for ep in range(episodes):
            ep_dir = agent_dir / game_id / f"ep{ep:02d}" if episodes > 1 else agent_dir / game_id
            m = play_one_with_trace(
                env, agent,
                run_dir=ep_dir,
                game_id=game_id,
                max_actions=max_actions,
                write_images=write_images,
            )
            per_game_metrics.setdefault(game_id, {}).update(m.as_dict())
            f1_pool.append(m.mean_f1)
            parse_pool.append(m.parse_rate)
            print(
                f"  [{tag}] [{gi}/{len(games)}] {game_id} ep{ep}: "
                f"actions={m.actions} final={m.final_state} f1={m.mean_f1:.3f} "
                f"parse={m.parse_rate:.3f} levels={m.levels_completed}/{m.total_levels}"
            )

    scorecard = arc.close_scorecard(card_id)
    sc_dump = scorecard.model_dump() if scorecard is not None else {}
    env_cards = {e["id"]: e for e in (sc_dump.get("environments") or [])}
    rhae_pool: list[float] = []
    for game_id, m in per_game_metrics.items():
        card = env_cards.get(game_id, {})
        if "error" in m:
            continue
        m["score"] = card.get("score")
        m["level_count"] = card.get("level_count")
        if isinstance(m.get("score"), (int, float)):
            rhae_pool.append(float(m["score"]))

    wall = round(time.time() - t0, 2)
    summary_payload = write_summary(
        agent_dir / "summary.json",
        run_kind="ablation",
        games=list(per_game_metrics.keys()),
        n_episodes_per_game=episodes,
        wall_clock_seconds=wall,
        mean_f1=round(sum(f1_pool) / len(f1_pool), 4) if f1_pool else 0.0,
        parse_rate=round(sum(parse_pool) / len(parse_pool), 4) if parse_pool else 0.0,
        mean_rhae=round(sum(rhae_pool) / len(rhae_pool), 4) if rhae_pool else 0.0,
        per_game=per_game_metrics,
        split=str(SPLIT_PATH.relative_to(REPO_ROOT)),
        git_commit=_git_commit(),
        agent=tag if tag == "random" else f"vlm_qwen25vl3b:{tag}",
        scorecard_id=card_id,
        max_actions=max_actions,
        notes=f"ablation row {tag} (ARCHITECTURE_AGENTS.md §2)",
    )
    return summary_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agents", default=",".join(ABLATION_AGENTS),
        help="Comma list — subset of " + ",".join(ABLATION_AGENTS),
    )
    parser.add_argument("--games", default="", help="Comma list of game id prefixes")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="", help="Run dir (default: outputs/ablation_<ts>)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Force RandomAgent for every row (no Qwen load). Plumbing only.",
    )
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    _check_key()

    requested_agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    for tag in requested_agents:
        if tag not in ABLATION_AGENTS:
            raise SystemExit(f"unknown agent tag: {tag} (choose from {ABLATION_AGENTS})")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output) if args.output else OUTPUTS_ROOT / f"ablation_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_root}")

    arc = Arcade()
    requested_games = _load_g_base(args.games)
    games = _resolve_full_game_ids(arc, requested_games)
    print(f"Playing {len(games)} game(s): {games}")
    print(f"Agents this run: {requested_agents}")

    backbone = _make_backbone(requested_agents, args.dry_run)
    if backbone is not None:
        print("Qwen backbone loaded once and shared across rows.")
    else:
        print("No backbone needed (random-only or dry-run).")

    overall_t0 = time.time()
    for tag in requested_agents:
        print(f"=== {tag} ===")
        effective_tag = "random" if args.dry_run else tag
        _run_one_agent(
            tag=effective_tag,
            arc=arc,
            games=games,
            backbone=backbone,
            seed=args.seed,
            out_root=out_root,
            max_actions=args.max_actions,
            write_images=not args.no_images,
            episodes=args.episodes,
        )

    wall = round(time.time() - overall_t0, 2)
    print(f"\nAblation complete in {wall}s.")
    print(f"Next: `.venv/Scripts/python.exe scripts/report_ablation.py --run {out_root}`")


if __name__ == "__main__":
    main()
