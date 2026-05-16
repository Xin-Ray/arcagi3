"""Replay games through the SDK and save the first N clean grid PNGs.

For each game in BASELINE_RUN, replay the first N actions from its
trace.jsonl through the live SDK, render each frame's grid into a clean
512x512 RGB PNG with `arc_agent.observation.grid_to_image(scale=8)`, and
save to `outputs/qwen_object_diag/<game>/frame_<NN>.png`.

Used by `docs/OBJECT_PIPELINE_DESIGN_zh.md` §3.1 (option B) as the
input source for `scripts/qwen_object_diag.py`. Producing CLEAN frames
(no 4-quadrant overlay) so Qwen sees exactly the input the ablation
agents saw.

ACTION6 in the trace doesn't carry its (x, y) coords (the trace only
records `chosen_action.name`), so when replay sees an ACTION6 we
substitute (0, 0). The resulting frame may differ from the original
agent's run -- that's fine for object-extraction testing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from arc_agi import Arcade   # noqa: E402
from arcengine import GameAction, GameState   # noqa: E402

from arc_agent.observation import grid_to_image, latest_grid   # noqa: E402


BASELINE_RUN = REPO_ROOT / "outputs" / "baseline_20260511_200835"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "qwen_object_diag"
DEFAULT_N_FRAMES = 5


def _action_from_name(name: str) -> GameAction:
    """Map "ACTION3" / "RESET" string back to a GameAction enum value."""
    if name == "RESET":
        return GameAction.RESET
    return GameAction[name]


def _replay_one_game(arc: Arcade, game_id: str, n_frames: int,
                     trace_path: Path, out_dir: Path, card_id: str) -> int:
    """Replay one game, save n_frames PNGs. Returns number saved."""
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_rows = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trace_rows.append(json.loads(line))
    if not trace_rows:
        print(f"  [{game_id}] trace.jsonl empty -- skipping")
        return 0

    env = arc.make(game_id, scorecard_id=card_id)
    if env is None:
        print(f"  [{game_id}] arc.make returned None -- skipping")
        return 0

    saved = 0
    latest = env.reset()
    if latest.frame:
        grid = latest_grid(latest)
        img = grid_to_image(grid, scale=8)
        png_path = out_dir / f"frame_{saved:02d}.png"
        img.save(png_path)
        print(f"  [{game_id}] saved {png_path.name}  state={latest.state.name}")
        saved += 1

    for row in trace_rows:
        if saved >= n_frames:
            break
        action_name = row.get("chosen_action") or "RESET"
        try:
            action = _action_from_name(action_name)
        except KeyError:
            print(f"  [{game_id}] unknown action {action_name!r} -- skipping step")
            continue
        if action.is_complex():
            action.set_data({"x": 0, "y": 0})
        latest = env.step(
            action,
            data=action.action_data.model_dump(),
            reasoning="render_clean_frames replay",
        )
        if not latest.frame:
            print(f"  [{game_id}] no frame after step -- stopping")
            break
        grid = latest_grid(latest)
        img = grid_to_image(grid, scale=8)
        png_path = out_dir / f"frame_{saved:02d}.png"
        img.save(png_path)
        print(f"  [{game_id}] saved {png_path.name}  action={action_name}  "
              f"state={latest.state.name}")
        saved += 1
        if latest.state is GameState.WIN:
            print(f"  [{game_id}] reached WIN early -- stopping replay")
            break
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-frames", type=int, default=DEFAULT_N_FRAMES,
                        help="frames per game to save (default 5)")
    parser.add_argument("--baseline", default=str(BASELINE_RUN),
                        help="baseline run dir containing per-game trace.jsonl")
    parser.add_argument("--output", default=str(OUTPUT_ROOT),
                        help="root output dir for clean PNGs")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    game_dirs = sorted(
        d for d in baseline_dir.iterdir()
        if d.is_dir() and (d / "trace.jsonl").exists()
    )
    if not game_dirs:
        raise SystemExit(f"no per-game trace.jsonl found under {baseline_dir}")

    print(f"Found {len(game_dirs)} game(s) with trace.jsonl in {baseline_dir}")
    for d in game_dirs:
        print(f"  - {d.name}")

    arc = Arcade()
    card_id = arc.open_scorecard(tags=["render_clean_frames", "diag"])
    print(f"Scorecard {card_id} opened.")
    try:
        total = 0
        for gdir in game_dirs:
            game_id = gdir.name
            print(f"=== {game_id} ===")
            saved = _replay_one_game(
                arc, game_id, args.n_frames,
                trace_path=gdir / "trace.jsonl",
                out_dir=out_root / game_id,
                card_id=card_id,
            )
            total += saved
        print(f"\nTotal saved: {total} PNGs across {len(game_dirs)} games.")
    finally:
        arc.close_scorecard(card_id)
        print(f"Scorecard {card_id} closed.")


if __name__ == "__main__":
    main()
