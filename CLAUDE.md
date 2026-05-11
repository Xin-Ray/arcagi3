# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project

ARC-AGI-3 competition agent research (ARC Prize 2026). The goal is to build agents that autonomously explore turn-based game environments with no instructions, infer rules, discover win conditions, and complete levels efficiently. See `TASK_OVERVIEW.md` for full competition details.

## Single source of truth

The active design doc is **`docs/ARCHITECTURE_RL.md`** — read it before doing any substantive work. Everything else previously in `docs/` (ARCHITECTURE / LIBRARY / PAPER / ROADMAP / RESEARCH / EXPERIMENTS / CODE_MAP / README) was archived to `archive/docs_2026-05-11/` on 2026-05-11. Treat the archive as read-only history. New design notes, status, and Run Log entries go into `docs/ARCHITECTURE_RL.md` directly (see its §9 Run Log).

`vlm_test/README.md` is the implementation-side companion (folder layout, commands, status table).

## Library-first coding (mandatory)

All reusable logic lives in the `arc_agent/` package. Scripts (`vlm_test/scripts/`, root `.py`) do I/O orchestration only — **no reusable logic in scripts**.

Before writing any function-shaped piece of code:

1. **Search `arc_agent/` first** with Grep for the symbol or behavior. If it exists → `from arc_agent.<mod> import <fn>`.
2. **Decide its module.** Anything that will plausibly be called twice goes into `arc_agent/<mod>.py`, not into a script.
3. **Implement** with type annotations + a one-line docstring. Pure functions where possible; library code does no env / file I/O — those belong in entry scripts.
4. **Test** in `tests/test_<mod>.py`: at least normal input + one edge case. Run `.venv/Scripts/python.exe -m pytest tests/test_<mod>.py -q`.
5. **No LIBRARY.md** — discoverability is by Grep on the source. If a module is non-obvious, a comment at the top of the file is enough.

Deprecation: don't delete; mark `Status: deprecated → <replacement>` in a docstring and keep the symbol for at least two weeks.

## Repository State

Pivoted from BC training (Phase 2) to **RL with intrinsic F1 reward** on 2026-05-11 (see `docs/ARCHITECTURE_RL.md` §0 战略决策). Current step: `docs/ARCHITECTURE_RL.md` §9 Step 3 (`VLMAgent` 推理 loop) — Steps 1–2 (`grid_to_image`, `rewards.py` primitives) are merged.

- `agent_starter.py` — single-game demo, imports `RandomAgent` + `play_one` from the library
- `eval.py` — batch evaluator across N game × M episode, writes `runs/<ts>_<tag>.jsonl`; `--agent {random,llm,llm-haiku}`, `--budget` USD hard stop
- `arc_agent/` — library: `runner.py` (Agent Protocol + `play_one`), `observation.py` (grid → text/diff/image), `rewards.py` (verifier F1 primitives), `llm.py` (Anthropic wrapper), `agents/{random,llm}.py`
- `tests/` — pytest suite (passing as of 2026-05-11; verifier + grid_to_image tests added on RL branch)
- `vendor/ARC-AGI-3-Agents/` — read-only clone of the official scaffold; do **not** edit

**Backbone**: Qwen2.5-VL-3B-Instruct (chosen 2026-04-28). 64×64 grid → 512×512 PNG via `arc_agent.observation.grid_to_image`, fed to the ViT vision encoder. The text-only Qwen3-0.6B route was dropped because hex-encoded grids miss 2D spatial structure (vertically adjacent cells are ~64 tokens apart in a 1D sequence).

## Gotchas

- **`arc_agi/base.py` auto-loads `.env.example` at import time**, so `os.getenv("ARC_API_KEY")` returns the placeholder `"your_arc_api_key_here"` if the user hasn't created a real `.env`. All `if not key:` checks are bypassed and the SDK then 401s. Always check `key.startswith("your_")` too.
- **Anonymous mode is rejected** by `https://three.arcprize.org` (401 on `/api/games`). A real `ARC_API_KEY` is required from https://arcprize.org/api-keys, placed in `.env` (NOT `.env.example`).
- **`GameAction` has 8 enum members** (`RESET + ACTION1..ACTION7`); only ACTION1–ACTION7 are user-callable. `RESET` is the state-transition action when `state ∈ {NOT_PLAYED, GAME_OVER}`. Only `ACTION6` is `is_complex()` (needs x, y ∈ [0,63]). The runner passes coordinates via the action's pydantic data model: `env.step(action, data=action.action_data.model_dump())`.
- **`FrameDataRaw.win_levels` is the total level count, not the wins.** Use `levels_completed` for progress.
- **Windows console (cp1252) cannot encode Unicode arrows** like `→`. Use ASCII (`->`) in any print that may run on Windows.
- **Default `GAME_ID = "ls20"`**, version 9607627b at time of writing (SDK auto-downloads to `environment_files/ls20/9607627b/ls20.py` — game logic is local Python, not a remote service).

## Environment

`.venv/` is Python 3.12 (the prior 3.11 venv could not install `arcengine`, which requires ≥3.12). Base interpreter: `C:\Users\sshuser\AppData\Local\Programs\Python\Python312\python.exe`.

A real `ARC_API_KEY` from https://arcprize.org/api-keys must be in `.env` at the repo root (not `.env.example` — see Gotchas). `requirements.txt` pins `arc-agi>=0.9.8` and `arcengine>=0.9.3`. RL training deps (`transformers`, `peft`, `bitsandbytes`, `trl`, `accelerate`, `qwen-vl-utils`) are intentionally **not** in `requirements.txt` to avoid Kaggle conflicts — install separately when running training.

## Commands

```bash
# Single-game demo against live SDK (requires real ARC_API_KEY in .env)
.venv/Scripts/python.exe agent_starter.py

# Batch eval — RandomAgent on all available demo games, 1 episode each
.venv/Scripts/python.exe eval.py

# Batch eval — LLMAgent on three keyboard games, 3 episodes each
.venv/Scripts/python.exe eval.py --agent llm --games ls20,tr87,wa30 --episodes 3 --tag llm_keyboard

# Tests (full suite, then a single test by node id)
.venv/Scripts/python.exe -m pytest tests/ -q
.venv/Scripts/python.exe -m pytest tests/test_rewards.py -q
```

`eval.py` flags: `--agent {random,llm,llm-haiku}`, `--games <comma-list-or-prefix>` (empty = all available demo games), `--episodes N`, `--max-actions N` (default 80), `--tag <label>`, `--output <path>`. It writes one jsonl row per (game, episode) plus a final `__summary__` row to `runs/<ts>_<tag>.jsonl`.

RL-line entry points (planned per `docs/ARCHITECTURE_RL.md` §9):

```bash
# Step 6 — Baseline (Go/no-go gate)
.venv/Scripts/python.exe vlm_test/scripts/run_baseline.py --output vlm_test/outputs/baseline_<ts>

# Step 7 — GRPO training
.venv/Scripts/python.exe vlm_test/scripts/run_grpo.py --output vlm_test/outputs/grpo_<ts>

# Step 8 — Validation (post-training)
.venv/Scripts/python.exe vlm_test/scripts/run_validation.py --checkpoint <path> --output vlm_test/outputs/validation_<ts>
```

## Core Concepts

**Scoring (RHAE)**: `S = min(1.0, h/a)²` where `h` = second-best human action count, `a` = agent actions. Quadratic penalty — being 2× slower yields 0.25, not 0.5. Later levels are weighted more heavily (`w_l = l`). Hard cutoff at 5× human action budget.

**Observation**: 64×64 grid, 16 possible colors per cell. May be a sequence of frames (animations).

**Action space**: 7 actions in `GameAction`, imported `from arcengine import GameAction`:
- `ACTION1`=Up, `ACTION2`=Down, `ACTION3`=Left, `ACTION4`=Right (4 directional)
- `ACTION5`=Primary (interact / select / rotate / etc — game-specific)
- `ACTION6`=Coordinate (takes x, y ∈ [0, 63] via `env.step(action, data={"x":..., "y":...})`; check with `action.is_complex()`)
- `ACTION7`=Undo

**Datasets**: 25 public demo environments (easier), 55 semi-private (API testing), 55 fully private (competition eval). Access via `arc.make(game_id)`.

**Intrinsic F1 reward** (RL-line core idea, `docs/ARCHITECTURE_RL.md` §3): each step the agent predicts which cells will change. We compare the predicted set to the real change set extracted from `(s_t, s_{t+1})` and score with F1. That F1 becomes a dense reward signal — every step gives feedback, no waiting for WIN. Implemented in `arc_agent/rewards.py`.

## Agent Interface

The canonical loop (matches `arc_agent/runner.py:play_one`):

```python
from arc_agi import Arcade
from arcengine import GameAction, GameState

arc = Arcade()                                  # reads ARC_API_KEY from env
card_id = arc.open_scorecard(tags=["demo"])
env = arc.make(game_id, scorecard_id=card_id)   # game_id from arcprize.org/tasks

latest = env.reset()                            # returns FrameDataRaw
while latest.state is not GameState.WIN:
    action = agent.choose(latest, history)      # any object satisfying arc_agent.runner.Agent
    latest = env.step(
        action,
        data=action.action_data.model_dump(),   # action carries its own pydantic data model
        reasoning=getattr(action, "reasoning", None),
    )

scorecard = arc.close_scorecard(card_id)        # .score is overall RHAE
```

Notes:
- Action enum lives in `arcengine`, **not** `arc_agi`. Both are pinned in `requirements.txt`.
- `env.step` returns a single `FrameDataRaw` object (not a 5-tuple). Use `latest.state`, `latest.levels_completed`, `latest.guid`, `latest.frame` (and remember `latest.win_levels` is the total level count, not wins).
- A single scorecard can span many `arc.make(...)` calls — `play_one` does **not** close the scorecard so the caller can run a whole batch under one card. `eval.py` opens one card per run and closes it at the end.
- `render_mode="terminal"` is available for visual debugging; omit for speed.

## Key Links

- API docs: https://docs.arcprize.org
- Browse game IDs: https://arcprize.org/tasks
- Kaggle competition: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
- Leaderboard: https://arcprize.org/leaderboard
