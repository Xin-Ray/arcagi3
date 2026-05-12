"""Post-training validation — Stage D / Step 8.

Loads the base Qwen2.5-VL + a LoRA checkpoint (from `scripts/run_grpo.py`),
runs one episode per G_val game, writes summary.json. Pair the output with
the pre-training baseline summary to compute the F1_v_pre / F1_v_post
comparison documented in docs/ARCHITECTURE_RL.md §5.4.

CLI examples:
    # Validate a checkpoint, default G_val
    .venv/Scripts/python.exe scripts/run_validation.py \
        --checkpoint outputs/grpo_<ts>/checkpoint-500

    # Side-by-side: pass both summaries through `--compare-to`
    .venv/Scripts/python.exe scripts/run_validation.py \
        --checkpoint outputs/grpo_<ts>/checkpoint-500 \
        --compare-to outputs/baseline_<ts>/summary.json

    # Dry-run plumbing (RandomAgent, no model load, no checkpoint)
    .venv/Scripts/python.exe scripts/run_validation.py --dry-run --max-actions 3
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


def _load_g_val(override: str) -> list[str]:
    if override:
        return [g.strip() for g in override.split(",") if g.strip()]
    if not SPLIT_PATH.exists():
        raise RuntimeError(
            f"Frozen split missing: {SPLIT_PATH}. "
            "Run `scripts/freeze_splits.py` first."
        )
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["g_val"]


def _make_agent(
    dry_run: bool, checkpoint: str | None, seed: int,
    max_new_tokens: int, temperature: float,
):
    """Choose the agent. --dry-run -> RandomAgent (no model load)."""
    if dry_run:
        from arc_agent.agents.random import RandomAgent
        return RandomAgent(seed=seed)

    from arc_agent.agents.vlm import VLMAgent
    from arc_agent.vlm_backbone import HFBackbone

    backbone = HFBackbone.load(lora_path=checkpoint)
    return VLMAgent(
        backbone=backbone,
        seed=seed,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def _resolve_full_ids(arc: Arcade, requested: list[str]) -> list[str]:
    env_infos = arc.get_environments() or []
    available = [e.game_id for e in env_infos]
    resolved: list[str] = []
    for r in requested:
        match = next((a for a in available if a.startswith(r)), None)
        if match is None:
            raise RuntimeError(f"Game id '{r}' not found among SDK games.")
        resolved.append(match)
    return resolved


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


def _compare(curr: dict[str, Any], prev_summary_path: Path) -> dict[str, Any]:
    """Compute pre→post deltas for the three top-level metrics."""
    if not prev_summary_path.exists():
        return {"error": f"compare-to file missing: {prev_summary_path}"}
    prev = json.loads(prev_summary_path.read_text(encoding="utf-8"))
    return {
        "prev_run_kind": prev.get("run_kind"),
        "prev_run_ts": prev.get("run_ts"),
        "delta_mean_f1":     round(curr["mean_f1"] - float(prev.get("mean_f1", 0.0)), 4),
        "delta_parse_rate":  round(curr["parse_rate"] - float(prev.get("parse_rate", 0.0)), 4),
        "delta_mean_rhae":   round(curr["mean_rhae"] - float(prev.get("mean_rhae", 0.0)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="",
                        help="Path to LoRA adapter dir (peft `save_pretrained` output)")
    parser.add_argument("--games", default="",
                        help="Comma list of validation game ids (default: G_val)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="",
                        help="Run dir (default: outputs/validation_<ts>)")
    parser.add_argument("--tag", default="validation_vlm")
    parser.add_argument(
        "--compare-to", default="",
        help="Path to an earlier summary.json (e.g. the baseline) to diff against.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use RandomAgent instead of loading the checkpoint.",
    )
    parser.add_argument(
        "--no-images", action="store_true",
        help="Skip per-step PNG and play.gif.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=768,
        help="VLM generate budget per step.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="VLM sampling temperature (0.0 = greedy).",
    )
    args = parser.parse_args()

    _check_key()

    if not args.dry_run and not args.checkpoint:
        raise SystemExit(
            "--checkpoint is required for a real validation run. "
            "Use --dry-run for a plumbing check without a checkpoint."
        )

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output) if args.output else OUTPUTS_ROOT / f"validation_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    arc = Arcade()
    games = _resolve_full_ids(arc, _load_g_val(args.games))
    print(f"Validating on {len(games)} game(s): {games}")

    agent = _make_agent(
        args.dry_run, args.checkpoint or None, args.seed,
        args.max_new_tokens, args.temperature,
    )

    card_id = arc.open_scorecard(tags=[args.tag, "g_val"])
    print(f"Scorecard {card_id} opened.")

    t0 = time.time()
    per_game_metrics: dict[str, dict[str, Any]] = {}
    f1_pool: list[float] = []
    parse_pool: list[float] = []
    for gi, game_id in enumerate(games, 1):
        if hasattr(agent, "reset"):
            agent.reset()
        env = arc.make(game_id, scorecard_id=card_id)
        if env is None:
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
        f1_pool.append(m.mean_f1)
        parse_pool.append(m.parse_rate)
        print(
            f"[{gi}/{len(games)}] {game_id}: actions={m.actions} "
            f"final={m.final_state} f1={m.mean_f1:.3f} "
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
        m.update(_per_env_score(card))
        if isinstance(m.get("score"), (int, float)):
            rhae_pool.append(float(m["score"]))

    wall = round(time.time() - t0, 2)
    mean_f1 = sum(f1_pool) / len(f1_pool) if f1_pool else 0.0
    parse_rate = sum(parse_pool) / len(parse_pool) if parse_pool else 0.0
    mean_rhae = sum(rhae_pool) / len(rhae_pool) if rhae_pool else 0.0

    summary_path = run_dir / "summary.json"
    summary_kwargs: dict[str, Any] = dict(
        run_kind="validation",
        games=list(per_game_metrics.keys()),
        n_episodes_per_game=args.episodes,
        wall_clock_seconds=wall,
        mean_f1=round(mean_f1, 4),
        parse_rate=round(parse_rate, 4),
        mean_rhae=round(mean_rhae, 4),
        per_game=per_game_metrics,
        split=str(SPLIT_PATH.relative_to(REPO_ROOT)),
        git_commit=_git_commit(),
        scorecard_id=card_id,
        agent="random" if args.dry_run else "vlm_qwen25vl3b_lora",
        checkpoint=args.checkpoint or None,
        max_actions=args.max_actions,
    )
    if args.compare_to:
        summary_kwargs["compared_to"] = str(args.compare_to)
        summary_kwargs["delta_vs_compare_to"] = _compare(
            summary_kwargs, Path(args.compare_to),
        )

    write_summary(summary_path, **summary_kwargs)

    print()
    print(f"=== validation summary  ({wall}s) ===")
    print(f"  mean F1:     {mean_f1:.4f}")
    print(f"  parse rate:  {parse_rate:.4f}")
    print(f"  mean RHAE:   {mean_rhae:.4f}")
    if args.compare_to:
        d = summary_kwargs["delta_vs_compare_to"]
        print(f"  delta vs {args.compare_to}: f1={d.get('delta_mean_f1')}, "
              f"parse={d.get('delta_parse_rate')}, rhae={d.get('delta_mean_rhae')}")
    print(f"  summary:     {summary_path}")
    print(f"  scorecard:   {card_id}")


if __name__ == "__main__":
    main()
