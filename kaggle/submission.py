"""ARC Prize 2026 — Kaggle submission notebook (script form).

Copy this to a Kaggle notebook (or convert to .ipynb) and run inside a
Kaggle Code Competition kernel attached to:
  - competition: arc-prize-2026-arc-agi-3
  - dataset:     xinxiang000/arcagi3-code      (our arc_agent library)
  - dataset:     xinxiang000/smollm3-3b-4bit   (pre-downloaded model)

The notebook produces /kaggle/working/submission.parquet, then you run:

  kaggle competitions submit -c arc-prize-2026-arc-agi-3 \
      -f submission.parquet -k xinxiang000/<notebook-slug> \
      -v <version> -m "v4+propose attempt N"

UNKNOWNS (marked TODO below) -- fill in from Kaggle starter notebook
since the competition's expected submission schema and env wrapper are
not publicly fixed yet.
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

# ─── Kaggle paths ─────────────────────────────────────────────────────
INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
COMP = INPUT_ROOT / "arc-prize-2026-arc-agi-3"      # competition data
CODE = INPUT_ROOT / "arcagi3-code"                  # our repo as a dataset
MODEL = INPUT_ROOT / "smollm3-3b-4bit"              # pre-downloaded weights

# Add our code dir to sys.path so we can import arc_agent.*
sys.path.insert(0, str(CODE))

# Offline mode -- no HuggingFace downloads at runtime
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# ─── Load agent stack ─────────────────────────────────────────────────
from arc_agent.agents.action_agent import ActionAgent  # noqa: E402
from arc_agent.agents.reflection_agent import ReflectionAgent  # noqa: E402
from arc_agent.knowledge import Knowledge  # noqa: E402
from arc_agent.goal_evaluator import evaluate_goal  # noqa: E402
from arc_agent.action_mask import apply_action_mask, compute_action_mask  # noqa: E402
from arc_agent.observation import available_action_names, latest_grid  # noqa: E402
from arc_agent.vlm_backbone import make_backbone  # noqa: E402

# ─── Load backbone (4-bit nf4 from local Kaggle dataset) ──────────────
print("[init] loading SmolLM3-3B from", MODEL, flush=True)
t0 = time.time()
backbone = make_backbone(str(MODEL), reasoning_mode="no_think")
print(f"[init]   loaded in {time.time()-t0:.1f}s", flush=True)

action_agent = ActionAgent(
    backbone=backbone, seed=42,
    max_new_tokens=256,
)
reflection_agent = ReflectionAgent(
    backbone=backbone, max_new_tokens=1024, temperature=0.0,
)
# V4 minimal + propose ON (the validated config from 2026-05-19)
action_agent.use_proposer = True

# ─── Game iteration ───────────────────────────────────────────────────
# TODO: the actual Kaggle env API for ARC-AGI-3 is unknown until we
# see the starter notebook. Two likely shapes:
#   (a) Local copy of `arc_agi.Arcade` reading frames from
#       /kaggle/input/arc-prize-2026-arc-agi-3/, exposing the same
#       env.reset() / env.step(action) interface.
#   (b) A Kaggle harness that calls a `predict(observation)` fn from
#       this notebook each turn.
#
# Below we sketch the (a) variant. Adjust `make_env(game_id)` based on
# the actual API once you see the starter notebook.

try:
    from arc_agi import Arcade  # may be the official path on Kaggle
    arc = Arcade(offline=True)  # if Arcade supports offline mode
    def make_env(game_id):
        return arc.make(game_id)
except Exception:
    # Fallback: assume environment_files/ layout, like our local
    # vendor/ARC-AGI-3-Agents/environment_files/
    print("[init] arc_agi.Arcade not available offline, using local stub")
    def make_env(game_id):
        raise NotImplementedError(
            "TODO: paste the starter notebook's env factory here")


# Get game ids from competition data
def list_game_ids() -> list[str]:
    # TODO: replace with whatever the starter notebook does.
    # Common patterns:
    #   - sample_submission.csv has game_id column
    #   - test_games.json lists ids
    #   - directory listing of /kaggle/input/<comp>/games/
    sample = COMP / "sample_submission.parquet"
    if sample.exists():
        import pandas as pd
        df = pd.read_parquet(sample)
        if "game_id" in df.columns:
            return df["game_id"].unique().tolist()
    # Fallback: dir scan
    games_dir = COMP / "games"
    if games_dir.exists():
        return [p.name for p in games_dir.iterdir() if p.is_dir()]
    raise RuntimeError(
        "Could not find game IDs. Inspect /kaggle/input/ layout and "
        "update list_game_ids().")


# ─── Per-game agent run (V4+propose, no_think SmolLM3) ────────────────
def play_one_game(game_id: str, max_actions_total: int = 200,
                  max_actions_per_round: int = 100, max_rounds: int = 5,
                  ) -> list[dict]:
    """Run our V4+propose agent on one game. Returns a list of records.
    Each record is one step. Adjust to match Kaggle's expected schema.
    """
    knowledge = Knowledge.empty(game_id=game_id)
    records: list[dict] = []
    steps_used = 0

    for round_idx in range(max_rounds):
        if steps_used >= max_actions_total:
            break
        env = make_env(game_id)
        action_agent.reset_episode_state(knowledge=knowledge)
        if hasattr(reflection_agent, "reset"):
            reflection_agent.reset()
        latest = env.reset()
        round_cap = min(max_actions_per_round,
                        max_actions_total - steps_used)

        for step in range(round_cap):
            from arcengine import GameState
            if latest.state in (GameState.WIN, GameState.GAME_OVER):
                break
            action, _ = action_agent.choose(latest)
            data = (action.action_data.model_dump()
                    if action.is_complex() else None)
            try:
                latest = env.step(action, data=data)
            except TypeError:
                latest = env.step(action)
            records.append({
                "game_id": game_id,
                "round": round_idx,
                "step": step,
                "action": action.name,
                "action_x": (data or {}).get("x"),
                "action_y": (data or {}).get("y"),
                "state": latest.state.name,
                "levels_completed": getattr(latest, "levels_completed", 0),
            })
            steps_used += 1

        if latest.state.name == "WIN":
            break

    return records


# ─── Main loop ────────────────────────────────────────────────────────
print("[main] starting game iteration", flush=True)
t_main = time.time()
all_records: list[dict] = []
game_ids = list_game_ids()
print(f"[main]   {len(game_ids)} games to play", flush=True)
for i, gid in enumerate(game_ids):
    t_g = time.time()
    try:
        recs = play_one_game(gid)
    except Exception as e:
        print(f"[main]   {gid}: EXCEPTION {e}", flush=True)
        recs = []
    all_records.extend(recs)
    print(f"[main]   [{i+1}/{len(game_ids)}] {gid} -> "
          f"{len(recs)} steps, {time.time()-t_g:.1f}s", flush=True)

# ─── Write submission.parquet ─────────────────────────────────────────
# TODO: the actual schema (column names, types) MUST match
# /kaggle/input/arc-prize-2026-arc-agi-3/sample_submission.parquet.
# Inspect with pd.read_parquet(<sample>).columns + .dtypes and adjust.
import pandas as pd  # noqa: E402
df = pd.DataFrame(all_records)
out = WORKING / "submission.parquet"
df.to_parquet(out, index=False)
print(f"[done] {len(df)} rows -> {out} "
      f"(total {time.time()-t_main:.1f}s)", flush=True)
