"""
Step 1 — collect training data from the ARC SDK (with animation path info).

Each step record now includes:
  - PNG of the current stable grid state
  - animation_text: direction/path/distance of every object that moved
  - diff_cells: exact cells that changed (old_color -> new_color)
  - n_animation_frames: how many intermediate frames the action produced

Prompt structure stored in each record:
  [current grid image]
  State / Level / Available actions
  Last action: ACTION4
  Result of last action:
    animation: 5 frames
      color 1 (blue): (row~32, col~10) -> (row~32, col~14) [RIGHT 4 cells]
    Changed cells: 4 cells  [(32,10,1,0), (32,14,0,1), ...]
  What is your next action?

Usage:
    .venv\\Scripts\\python.exe vlm_test/scripts/collect_data.py
    .venv\\Scripts\\python.exe vlm_test/scripts/collect_data.py --games ls20 --episodes 3 --max-steps 40
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

OUT_DIR   = ROOT / "vlm_test" / "data" / "train"
IMG_DIR   = OUT_DIR / "images"
JSONL_OUT = OUT_DIR / "dataset.jsonl"


def build_prompt(
    state: str,
    level: int,
    total_levels: int,
    available: list[str],
    last_action: str | None,
    animation_text: str | None,
    diff_cells: list | None,
) -> str:
    lines = [
        f"State: {state} | Level: {level}/{total_levels}",
        f"Available: {' '.join(available)}",
    ]
    if last_action and animation_text:
        lines.append(f"Last action: {last_action}")
        lines.append("Result of last action:")
        for anim_line in animation_text.splitlines():
            lines.append(f"  {anim_line}")
        n_diff = len(diff_cells) if diff_cells else 0
        if diff_cells and n_diff <= 20:
            lines.append(f"  Changed cells: {n_diff}  {diff_cells}")
        else:
            lines.append(f"  Changed cells: {n_diff}")
    lines.append("What is your next action? Reply with: ACTION: <name>")
    return "\n".join(lines)


def collect(game_id: str, episodes: int, max_steps: int, seed: int) -> int:
    from arc_agi import Arcade
    from arcengine import GameAction, GameState
    from arc_agent.observation import (
        latest_grid, available_action_names,
        grid_to_image, grid_diff, animation_to_text,
    )

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    arc  = Arcade()
    card = arc.open_scorecard(tags=["vlm_test_data"])
    rng  = random.Random(seed)
    rows: list[dict] = []

    print(f"Collecting  game={game_id}  episodes={episodes}  max_steps={max_steps}")

    for ep in range(episodes):
        env   = arc.make(game_id, scorecard_id=card)
        frame = env.reset()

        prev_grid: "np.ndarray | None" = None   # type: ignore[name-defined]
        prev_action: str | None = None
        step = 0

        while frame.state not in (GameState.WIN, GameState.GAME_OVER) and step < max_steps:
            curr_grid = latest_grid(frame)       # final stable state
            avail     = available_action_names(frame)

            # ── animation + diff from the action that produced this frame ──
            anim_text: str | None = None
            diff_cells: list | None = None

            if prev_grid is not None and len(frame.frame) >= 1:
                # animation path: only meaningful if > 1 frame
                if len(frame.frame) >= 2:
                    anim_text  = animation_to_text(frame.frame)
                else:
                    anim_text  = "animation: 1 frame (instant)"
                diff_cells = grid_diff(prev_grid, curr_grid)

            # ── save PNG of current stable state ──
            image    = grid_to_image(curr_grid, scale=8)
            img_name = f"{game_id}_ep{ep:02d}_step{step:03d}.png"
            image.save(IMG_DIR / img_name)

            # ── choose next action (random silver-label) ──
            action_name = rng.choice(avail) if avail else "ACTION1"
            action      = GameAction[action_name]
            action_data = None
            if action.is_complex():
                action_data = {"x": rng.randint(0, 63), "y": rng.randint(0, 63)}

            # ── build prompt ──
            prompt = build_prompt(
                state        = frame.state.name,
                level        = frame.levels_completed + 1,
                total_levels = frame.win_levels,
                available    = avail,
                last_action  = prev_action,
                animation_text = anim_text,
                diff_cells   = diff_cells,
            )

            rows.append({
                "step":               step,
                "game_id":            game_id,
                "episode":            ep,
                "level":              frame.levels_completed + 1,
                "state":              frame.state.name,
                "image_path":         f"images/{img_name}",
                "prompt":             prompt,
                "action":             action_name,
                "action_data":        action_data,
                "animation_text":     anim_text,
                "n_animation_frames": len(frame.frame),
                "n_diff_cells":       len(diff_cells) if diff_cells else 0,
            })

            prev_grid   = curr_grid.copy()
            prev_action = action_name

            # ── execute action ──
            step_kwargs: dict = {}
            if action_data:
                step_kwargs["data"] = action_data
            frame = env.step(action, **step_kwargs)
            step += 1

        print(f"  ep {ep}: {step} steps  final={frame.state.name}")

    arc.close_scorecard(card)

    with open(JSONL_OUT, "w", encoding="utf-8") as f:   # overwrite each run
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved {len(rows)} steps -> {JSONL_OUT}")
    print(f"Images -> {IMG_DIR}")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games",     default="ls20")
    ap.add_argument("--episodes",  type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    total = 0
    for gid in args.games.split(","):
        total += collect(gid.strip(), args.episodes, args.max_steps, args.seed)
    print(f"\nTotal: {total} training steps")


if __name__ == "__main__":
    main()
