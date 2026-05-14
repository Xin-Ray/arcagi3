# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ARC-AGI-3 competition agent research (ARC Prize 2026). The goal is to build agents that autonomously explore turn-based grid-puzzle environments with no instructions, infer rules, discover win conditions, and complete levels efficiently. See `TASK_OVERVIEW.md` for the full competition spec.

## Where to start reading

The repo has three layers of authoritative docs — read in this order on takeover:

1. **`README.md`** — 3-minute project overview (goal, data flow, what to run, current state, TODO board with stage gates).
2. **`docs/INDEX_zh.md`** — doc index + naming rules (`{arch|ref}_{name}_{version}_{lang}.md`). Tells you which doc is 🟢 active, 🟡 reference, ⚫ archived.
3. **`docs/arch_v3_2_zh.md`** (current main design) → **`docs/arch_v3_zh.md`** (the v3 baseline it builds on) → **`docs/ref_v3_prompt_zh.md`** (prompt block reference, read before touching prompts) → **`docs/ref_object_pipeline_zh.md`** (scipy vs Qwen-VL perception evaluation).

`docs/arch_rl_v0_zh.md` is the **older RL line** (intrinsic F1 reward + GRPO). It is **parked, not deleted** — v3 deviated from it after empirical results made the text-only-Qwen + scipy-perception route win. Read it only if you're touching `rewards.py` / `train_grpo.py` / `run_grpo.py` or you need to understand the historical decision.

## Active line: v3 / v3.2 (as of 2026-05-14)

The current architecture is **NOT** "VLM with image input + intrinsic F1 reward". It is:

```
scipy.ndimage.label  ──►  object_extractor  ──►  temporal_classifier  ──►  object_aligner (Hungarian)
        │                                                                              │
        ▼                                                                              ▼
    (perception is 100% deterministic, no LLM in the loop)              ObjectMemory + OutcomeLog
                                                                                       │
                                                                                       ▼
                                              prompts_v3.build_play_prompt  ◄──── enriched prompt
                                                                                       │
                                                                                       ▼
                                                Qwen2.5-VL-3B  in TEXT-ONLY mode  (no image content block)
                                                                                       │
                                                                                       ▼
                                            anti-collapse postprocess  →  ACTION
```

Key principle (`docs/arch_v3_zh.md` §0): **vision = deterministic algorithm; reasoning = text LLM; the two are connected by structured data, and the LLM never sees pixels.** This was decided because `ref_object_pipeline_zh.md` showed Qwen-VL at ~0% on per-frame object extraction while scipy was at 100% on the same ar25 set.

**v3.2 layer on top** (`docs/arch_v3_2_zh.md`): split into **Action Agent + Reflection Agent** with a shared `Knowledge` object that persists across rounds within the same `game_id`. Reflection runs **per step** (not per round) so the Action Agent can use within-episode discoveries. Currently being implemented; the v3 single-agent code is what's actually running.

## Library-first coding (mandatory)

All reusable logic lives in the `arc_agent/` package. Scripts in `scripts/` do I/O orchestration only — **no reusable logic in scripts**.

Before writing any function-shaped piece of code:

1. **Search `arc_agent/` first** with Grep for the symbol or behavior. If it exists → `from arc_agent.<mod> import <fn>`.
2. **Decide its module.** Anything that will plausibly be called twice goes into `arc_agent/<mod>.py`, not into a script.
3. **Implement** with type annotations + a one-line docstring. Pure functions where possible; library code does no env / file I/O — those belong in entry scripts.
4. **Test** in `tests/test_<mod>.py`: at least normal input + one edge case. Run `.venv/Scripts/python.exe -m pytest tests/test_<mod>.py -q`.
5. **No LIBRARY.md** — discoverability is by Grep on the source. If a module is non-obvious, a comment at the top of the file is enough.

Deprecation: don't delete; mark `Status: deprecated → <replacement>` in a docstring and keep the symbol for at least two weeks.

## Key modules (what to grep when you need something)

These are the load-bearing pieces of the v3 pipeline. Names match exactly; module-level docstrings have the full contract.

- **Perception** — `object_extractor.py` (scipy connected components → `ObjectRecord`), `temporal_classifier.py` (STATIC / ACTIVE / TEXTURE / CANDIDATE per object across frames), `object_aligner.py` (Hungarian cross-frame match), `object_tracker.py` (UID-keyed `ObjectMemory` for an episode).
- **Memory** — `action_inference.py` (`OutcomeLog`, `StepOutcome`, `detect_stuck`, `detect_collapse`, `summarize_action` → `LearnedActionMap`), `world_model.py` (persistent A3 state across steps), `click_candidates.py` (ACTION6 coordinate proposals).
- **Reasoner** — `prompts_v3.py` (8-block prompt builder), `agents/text_agent.py` (current main agent — text-only Qwen + anti-collapse), `vlm_backbone.py` (Qwen2.5-VL loader with lazy `torch/transformers` import).
- **Reflection / mistakes** — `agents/reflect.py` (PlayReflectAgent A3/A4), `mistakes.py` (deterministic mistake detectors).
- **Run plumbing** — `runner.py` (`Agent` Protocol + `play_one`), `baseline.py` (`play_one_with_trace` — trace.jsonl + step PNGs + play.gif), `viz.py` (4-quadrant `compose_step_image` + `write_gif`), `observation.py` (`grid_to_image`, `serialize_step`), `eval_split.py` (`demo_555_split`, `write_summary`).
- **Other agents (baselines / ablation)** — `agents/random.py`, `agents/llm.py` (Claude API), `agents/vlm.py` (image-input VLM, used in v1 ablations), `agents/vlm_lite.py` (A1).
- **RL line (parked)** — `rewards.py` (F1 verifier primitives), `train_grpo.py` (`reward_fn` + lazy `trl` trainer factory). Keep tests passing but no active iteration.
- **Report / audit** — `report.py` (per-(agent, game) summary aggregator). Companion scripts: `scripts/audit_traces.py`, `scripts/audit_illegal_actions.py`, `scripts/audit_action6_misuse.py`, `scripts/analyze_failures.py`, `scripts/compare_summaries.py`, `scripts/build_v3_visual_report.py`, `scripts/report_v3_vs_v1.py`, `scripts/report_ablation.py`.

## Repository State (2026-05-14)

- Tests: **353 collected** in `tests/`. Always run the relevant slice after editing a library module.
- Active eval entrypoint: `scripts/run_v3_eval.py` (v3 TextAgent on G_base, captures `action_entropy` and `unique_frame_hashes` beyond the baseline schema). Latest result: `outputs/v3_p0b_p1_full/report.md`.
- Frozen split: `data/splits/demo_555.json` (5 base / 5 train / 5 val game IDs, committed 2026-05-11; **do not re-randomize** — train/val comparability breaks otherwise).
- Backbone: **Qwen2.5-VL-3B-Instruct in text-only mode** (no image content block). 4-bit, `max_new_tokens=24`, greedy. The vision encoder is loaded but unused on the hot path.
- `outputs/` (gitignored) layout: `runs/<ts>_<tag>.jsonl` (eval.py), `baseline_<ts>/<game>/{step_*.png,trace.jsonl,play.gif}`, `v3_p0b_p1_full/`, `ablation_overnight_/`, `scipy_object_diag/`, `qwen_object_diag/`, `goal_inference/`.
- **`outputs/reports/` is the single browse-everything entry for all experiment reports** (see `outputs/reports/INDEX_zh.md`). Rule:
  - **Active experiments**: report stays in `outputs/<exp>/report.md`; INDEX **links** to it (relative GIFs / PNGs keep working).
  - **Archived experiments**: report is **copied** into `outputs/reports/<YYYY-MM-DD>_<name>.md` with image / sub-link paths rewritten to point back to `archive/outputs_<date>/<exp>/`. Use `scripts/copy_archived_reports.py` — add the entry to `COPIES` list and run.
  - When archiving an experiment, do all four: (1) add to `copy_archived_reports.py:COPIES`, (2) run the script, (3) move the source dir into `archive/outputs_<date>/`, (4) update `outputs/reports/INDEX_zh.md` ⚫ section.
- `archive/` is a **one-way door**: `archive/docs_2026-05-11/` (BC-era 7-doc set), `archive/docs_2026-05-14/` (v1/v2 agent designs superseded by v3), `archive/outputs_2026-05-14/` (27 stale experiment dirs + ~23 logs from 05-11 through 05-14), `archive/bc_scripts/` (paused BC pipeline), `archive/old_runs/`. The contents are indexed in `archive/INDEX_zh.md` (one-line summary per file). To revive anything, explicitly promote it back to the README §4 whitelist first.
- `vendor/ARC-AGI-3-Agents/` — read-only clone of the official scaffold; do **not** edit.

## Gotchas

- **`arc_agi/base.py` auto-loads `.env.example` at import time**, so `os.getenv("ARC_API_KEY")` returns the placeholder `"your_arc_api_key_here"` if the user hasn't created a real `.env`. All `if not key:` checks are bypassed and the SDK then 401s. Always check `key.startswith("your_")` too.
- **Anonymous mode is rejected** by `https://three.arcprize.org` (401 on `/api/games`). A real `ARC_API_KEY` is required from https://arcprize.org/api-keys, placed in `.env` (NOT `.env.example`).
- **`GameAction` has 8 enum members** (`RESET + ACTION1..ACTION7`); only ACTION1–ACTION7 are user-callable. `RESET` is the state-transition action when `state ∈ {NOT_PLAYED, GAME_OVER}`. Only `ACTION6` is `is_complex()` (needs x, y ∈ [0,63]). The runner passes coordinates via the action's pydantic data model: `env.step(action, data=action.action_data.model_dump())`.
- **`FrameDataRaw.win_levels` is the total level count, not the wins.** Use `levels_completed` for progress.
- **Windows console (cp1252) cannot encode Unicode arrows** like `→`. Use ASCII (`->`) in any print that may run on Windows.
- **Default `GAME_ID = "ls20"`**, version 9607627b at time of writing (SDK auto-downloads to `environment_files/ls20/9607627b/ls20.py` — game logic is local Python, not a remote service).
- **The text-only Qwen path bypasses the vision encoder**, but `vlm_backbone.py` still loads the full multimodal model. If you write a new agent and add an image content block back, you change the inference characteristics — keep `agents/text_agent.py` text-only unless you mean to.

## Environment

`.venv/` is Python 3.12 (the prior 3.11 venv could not install `arcengine`, which requires ≥3.12). Base interpreter: `C:\Users\sshuser\AppData\Local\Programs\Python\Python312\python.exe`.

A real `ARC_API_KEY` from https://arcprize.org/api-keys must be in `.env` at the repo root (not `.env.example` — see Gotchas). `ANTHROPIC_API_KEY` in `.env` enables `LLMAgent`. `requirements.txt` pins `arc-agi>=0.9.8` and `arcengine>=0.9.3`. Training deps (`transformers`, `peft`, `bitsandbytes`, `trl`, `accelerate`, `qwen-vl-utils`) are intentionally **not** in `requirements.txt` to avoid Kaggle conflicts — install separately when running v3 eval or RL training.

## Commands

```bash
# Tests (full suite, then a single test file)
.venv/Scripts/python.exe -m pytest tests/ -q
.venv/Scripts/python.exe -m pytest tests/test_text_agent.py -q

# Single-game demo against live SDK (RandomAgent — sanity check that the SDK + key work)
.venv/Scripts/python.exe scripts/agent_starter.py

# Batch eval — RandomAgent on all demo games, 1 episode each
.venv/Scripts/python.exe scripts/eval.py

# Batch eval — LLMAgent on three keyboard games, 3 episodes each
.venv/Scripts/python.exe scripts/eval.py --agent llm --games ls20,tr87,wa30 --episodes 3 --tag llm_keyboard

# v3 TextAgent eval on G_base (current main path; needs torch + transformers + Qwen weights)
.venv/Scripts/python.exe scripts/run_v3_eval.py
```

`scripts/eval.py` flags: `--agent {random,llm,llm-haiku}`, `--games <comma-list-or-prefix>` (empty = all demo), `--episodes N`, `--max-actions N` (default 80), `--tag <label>`, `--output <path>`. Writes one jsonl row per (game, episode) plus a final `__summary__` row to `outputs/runs/<ts>_<tag>.jsonl`.

RL-line entry points (parked, dry-run only):

```bash
.venv/Scripts/python.exe scripts/run_baseline.py    # zero-shot Qwen, was the Go/no-go gate
.venv/Scripts/python.exe scripts/run_grpo.py        # --dry-run skeleton
.venv/Scripts/python.exe scripts/run_validation.py  # --dry-run skeleton
```

**Long-running jobs over SSH must go through `scripts/run_scheduled.ps1`** — see the `scheduler-run` skill at `.claude/skills/scheduler-run/SKILL.md`. Bare `Start-Process -WindowStyle Hidden` does **not** survive SSH disconnect on Windows (the SSH session's job object kills all descendants). Pattern:

```powershell
# Wraps the python call in a Task Scheduler task (-LogonType S4U).
# First run needs an ELEVATED PowerShell; the python job itself runs as the user.
.\scripts\run_scheduled.ps1 v3_eval scripts\run_v3_eval.py `
    --output outputs\v3_eval_run --max-actions 80
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

**v3 reasoning loop** (per `docs/arch_v3_zh.md` §2): every `env.step` triggers deterministic perception → memory update → 8-block enriched prompt → text-only Qwen → anti-collapse postprocess (if last 3 actions are the same, force a different `untried` action). The prompt blocks are `[STATUS] [ACTIVE] [TEXTURE] [ACTION] [UNTRIED] [HISTORY] [GOAL] [ASK]` — `ref_v3_prompt_zh.md` is the per-block reference.

**Intrinsic F1 reward** (parked, RL-line): each step the agent predicts which cells will change; predicted set vs real change set scored with F1, used as dense reward signal. Implemented in `arc_agent/rewards.py`. **Not on the current path** — v3 doesn't predict diffs.

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
- A single scorecard can span many `arc.make(...)` calls — `play_one` does **not** close the scorecard so the caller can run a whole batch under one card. `scripts/eval.py` opens one card per run and closes it at the end.
- v3 agents (`text_agent.py`) expose `_state.last_prompt` / `last_response_raw` / `last_parse_ok` so `baseline.play_one_with_trace` can capture them into `trace.jsonl`. New agents should follow the same convention if they want to appear in trace audits.
- `render_mode="terminal"` is available for visual debugging; omit for speed.

## Key Links

- API docs: https://docs.arcprize.org
- Browse game IDs: https://arcprize.org/tasks
- Kaggle competition: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
- Leaderboard: https://arcprize.org/leaderboard
