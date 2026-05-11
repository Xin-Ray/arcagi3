"""Run the zero-shot Qwen2.5-VL baseline on G_base — ★ Go/no-go gate.

Reads the frozen split at `data/splits/demo_555.json`, opens one scorecard,
plays one episode per G_base game with `VLMAgent`, saves per-step PNG +
trace.jsonl + play.gif, and writes a final `summary.json` with the
pre-registered hypothesis embedded in the header.

Hypothesis (per docs/ARCHITECTURE_RL.md §5.2):
    mean F1   ≥ 0.30   (Qwen zero-shot has visual understanding)
    parse_rate ≥ 0.70  (Instruct-tuned Qwen follows JSON format)
    mean_rhae ≤ 0.05   (zero-shot unlikely to clear levels)

CLI examples:
    # Real run (needs ARC_API_KEY + GPU + transformers stack)
    .venv/Scripts/python.exe scripts/run_baseline.py

    # Dry run for plumbing test (RandomAgent, no model load)
    .venv/Scripts/python.exe scripts/run_baseline.py --dry-run \
        --max-actions 3 --output outputs/baseline_dry

    # Override games (e.g. test on a single game)
    .venv/Scripts/python.exe scripts/run_baseline.py --games ar25
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

HYPOTHESIS = {
    "mean_f1_min":   0.30,
    "parse_rate_min": 0.70,
    "mean_rhae_max": 0.05,
    "iteration_trigger": {
        "F1>=0.3 & parse>=0.7": "advance to Step 7 (GRPO training)",
        "F1<0.1":  "upgrade to two-image input; if still bad, fall back to BC",
        "parse<0.5": "add 1-2 in-context examples, rerun baseline",
        "0.1<=F1<0.3": "ablation: single-image vs two-image; with/without entity section",
    },
}


def _check_key() -> None:
    key = os.getenv("ARC_API_KEY", "")
    if not key or key.startswith("your_"):
        raise RuntimeError(
            "ARC_API_KEY missing or still the .env.example placeholder. "
            "Put a real key in .env (see https://arcprize.org/api-keys)."
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


def _make_agent(dry_run: bool, seed: int):
    """Choose the agent. --dry-run swaps in RandomAgent so we don't load Qwen."""
    if dry_run:
        from arc_agent.agents.random import RandomAgent
        return RandomAgent(seed=seed)

    from arc_agent.agents.vlm import VLMAgent
    from arc_agent.vlm_backbone import HFBackbone

    backbone = HFBackbone.load()  # Qwen2.5-VL-3B-Instruct, 4-bit
    return VLMAgent(backbone=backbone, seed=seed)


def _resolve_full_game_ids(
    arc: Arcade, requested: list[str]
) -> tuple[list[str], dict[str, Any]]:
    """Prefix-match user-supplied ids ("ar25") to full ids ("ar25-0c556536").

    Returns (resolved_ids, meta_by_id) where meta_by_id holds the SDK
    EnvironmentInfo for downstream RHAE lookup.
    """
    env_infos = arc.get_environments() or []
    available = [e.game_id for e in env_infos]
    meta_by_id = {e.game_id: e for e in env_infos}
    resolved: list[str] = []
    for r in requested:
        match = next((a for a in available if a.startswith(r)), None)
        if match is None:
            raise RuntimeError(
                f"Game id '{r}' not found among SDK games. Got {len(available)} games."
            )
        resolved.append(match)
    return resolved, meta_by_id


def _per_env_score(env_card: dict[str, Any]) -> dict[str, Any]:
    runs = env_card.get("runs") or []
    last_run = runs[-1] if runs else {}
    return {
        "score": env_card.get("score"),
        "levels_completed": env_card.get("levels_completed"),
        "level_count": env_card.get("level_count"),
        "actions": env_card.get("actions"),
        "level_baseline_actions": last_run.get("level_baseline_actions"),
        "level_actions": last_run.get("level_actions"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default="", help="Comma list of game ids/prefixes (default: G_base from frozen split)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="", help="Run dir (default: outputs/baseline_<ts>)")
    parser.add_argument("--tag", default="baseline_vlm")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use RandomAgent instead of Qwen — fast plumbing/regression test.",
    )
    parser.add_argument(
        "--no-images", action="store_true",
        help="Skip per-step PNG and play.gif (trace.jsonl still written).",
    )
    args = parser.parse_args()

    _check_key()

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output) if args.output else OUTPUTS_ROOT / f"baseline_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    arc = Arcade()
    requested = _load_g_base(args.games)
    games, meta_by_id = _resolve_full_game_ids(arc, requested)
    print(f"Playing on {len(games)} game(s): {games}")

    agent = _make_agent(args.dry_run, args.seed)

    card_id = arc.open_scorecard(tags=[args.tag, "g_base"])
    print(f"Scorecard {card_id} opened.")

    t0 = time.time()
    per_game_metrics: dict[str, dict[str, Any]] = {}
    f1_pool: list[float] = []
    parse_pool: list[int] = []  # 1/0 per step
    for gi, game_id in enumerate(games, 1):
        # Reset per-episode state of agents that keep memory across calls.
        if hasattr(agent, "reset"):
            agent.reset()
        env = arc.make(game_id, scorecard_id=card_id)
        if env is None:
            print(f"[{gi}/{len(games)}] {game_id}: arc.make returned None — skipping")
            per_game_metrics[game_id] = {"error": "arc.make returned None"}
            continue

        game_dir = run_dir / game_id
        m = play_one_with_trace(
            env, agent,
            run_dir=game_dir,
            game_id=game_id,
            max_actions=args.max_actions,
            fps=args.fps,
            write_images=not args.no_images,
        )
        per_game_metrics[game_id] = m.as_dict()
        # Pool for overall mean
        f1_pool.append(m.mean_f1)
        parse_pool.append(m.parse_rate)
        meta = meta_by_id.get(game_id)
        baseline_actions = getattr(meta, "baseline_actions", None) if meta else None
        per_game_metrics[game_id]["baseline_actions"] = baseline_actions
        print(
            f"[{gi}/{len(games)}] {game_id}: actions={m.actions} "
            f"final={m.final_state} f1={m.mean_f1:.3f} "
            f"parse={m.parse_rate:.3f} levels={m.levels_completed}/{m.total_levels}"
        )

    # Close scorecard → RHAE per env
    scorecard = arc.close_scorecard(card_id)
    sc_dump = scorecard.model_dump() if scorecard is not None else {}
    env_cards = {e["id"]: e for e in (sc_dump.get("environments") or [])}
    rhae_pool: list[float] = []
    for game_id, m in per_game_metrics.items():
        card = env_cards.get(game_id, {})
        if "error" in m:
            continue
        m.update(_per_env_score(card))
        if isinstance(m.get("score"), (int, float)):
            rhae_pool.append(float(m["score"]))

    wall = round(time.time() - t0, 2)
    mean_f1 = sum(f1_pool) / len(f1_pool) if f1_pool else 0.0
    parse_rate = sum(parse_pool) / len(parse_pool) if parse_pool else 0.0
    mean_rhae = sum(rhae_pool) / len(rhae_pool) if rhae_pool else 0.0

    branch = "no rhae" if not rhae_pool else "see hypothesis"
    if mean_f1 >= 0.3 and parse_rate >= 0.7:
        branch = "F1>=0.3 & parse>=0.7 -> advance to Step 7 (GRPO)"
    elif mean_f1 < 0.1:
        branch = "F1<0.1 -> upgrade to two-image input (or fall back to BC)"
    elif parse_rate < 0.5:
        branch = "parse<0.5 -> add in-context examples and rerun"
    elif mean_f1 < 0.3:
        branch = "0.1<=F1<0.3 -> ablation studies"

    summary_path = run_dir / "summary.json"
    write_summary(
        summary_path,
        run_kind="baseline",
        games=list(per_game_metrics.keys()),
        n_episodes_per_game=args.episodes,
        wall_clock_seconds=wall,
        mean_f1=round(mean_f1, 4),
        parse_rate=round(parse_rate, 4),
        mean_rhae=round(mean_rhae, 4),
        per_game=per_game_metrics,
        split=str(SPLIT_PATH.relative_to(REPO_ROOT)),
        git_commit=_git_commit(),
        hypothesis=HYPOTHESIS,
        branch_decision=branch,
        scorecard_id=card_id,
        agent="random" if args.dry_run else "vlm_qwen25vl3b",
        max_actions=args.max_actions,
        notes="zero-shot baseline; LoRA untrained" if not args.dry_run else "dry-run plumbing test",
    )

    print()
    print(f"=== baseline summary  ({wall}s) ===")
    print(f"  mean F1:     {mean_f1:.4f}  (hyp >= 0.30)")
    print(f"  parse rate:  {parse_rate:.4f}  (hyp >= 0.70)")
    print(f"  mean RHAE:   {mean_rhae:.4f}  (hyp <= 0.05)")
    print(f"  branch:      {branch}")
    print(f"  summary:     {summary_path}")
    print(f"  scorecard:   {card_id}")


if __name__ == "__main__":
    main()
