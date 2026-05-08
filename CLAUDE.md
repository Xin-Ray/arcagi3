# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ARC-AGI-3 competition agent research (ARC Prize 2026). The goal is to build agents that autonomously explore turn-based game environments with no instructions, infer rules, discover win conditions, and complete levels efficiently. See `TASK_OVERVIEW.md` for full competition details.

## Working Workflow — `docs/` is mandatory

This project uses seven tracking docs in `docs/` (six internal + one outward-facing living paper) as durable working memory. **Every session must follow this loop:**

1. **Before doing anything substantive**, read the relevant doc(s):
   - `docs/ROADMAP.md` — what phase we're in, what's the next unchecked task
   - `docs/CODE_MAP.md` — which file does what (before modifying any module)
   - `docs/LIBRARY.md` — what functions already exist (before writing any code — see "Library-first" below)
   - `docs/ARCHITECTURE.md` — module responsibilities and data flow (before designing/changing interfaces)
   - `docs/RESEARCH.md` — past attempts, known defects, open questions (before proposing a new approach)
   - `docs/EXPERIMENTS.md` — measured numbers (before claiming improvement)
   - `docs/PAPER.md` — outward-facing narrative (before any external write-up or stakeholder report)
   - `docs/README.md` — index of the above
2. **After finishing work**, write changes back to the right doc(s):
   - New / removed / refactored module → `CODE_MAP.md`
   - New / changed / deprecated **function** → `LIBRARY.md`
   - Interface or responsibility change → `ARCHITECTURE.md`
   - Tried an approach (success **or** failure) / read a paper / found a defect → `RESEARCH.md`
   - Ran an experiment with a conclusion → `EXPERIMENTS.md` (append-only, never overwrite)
   - Completed a roadmap item / changed phase → `ROADMAP.md`
   - **Anything that crosses into outward narrative** → `PAPER.md` (see "Living paper" below)

Rules: absolute dates only (YYYY-MM-DD), append-only history, failures must be recorded, one fact lives in one doc.

### Living paper — `PAPER.md`

`PAPER.md` is the project's **outward-facing artifact**: a paper-format draft that grows with the project, ready for Milestone #1 / #2 open-source submissions and any external write-up. It **does not introduce new facts** — it re-narrates facts that already exist in the other six docs into Abstract / Intro / Related Work / Method / Experiments / Discussion / Conclusion / References.

Update triggers (mandatory — same loop, just one extra check at the end of substantive work):

- **Experiment with a conclusion lands in `EXPERIMENTS.md`** → update PAPER §5 (numbers in the table) and §6 (at least one Discussion sentence). Numbers in PAPER must be traceable back to an EXPERIMENTS entry.
- **New paper / blog read, recorded in `RESEARCH.md` 读后感** → add a Related Work paragraph in PAPER §2, **and** add a numbered URL to PAPER References. Citations in the body without a References URL are not allowed.
- **Architectural decision changes in `ARCHITECTURE.md`** → update PAPER §4 (Method).
- **Phase transition in `ROADMAP.md`** → bump the version at the top of PAPER (Phase 0 end = v0.1, Phase 1 end = v0.2, …) and append a Changelog row.
- **Failure / falsified hypothesis** in RESEARCH → goes into PAPER §6.2 "What didn't" — never silently dropped.

Writing style for PAPER: academic voice ("we", not "Claude" / "the user"), no bare claims without a numbered citation, `[TBD]` / `[pending Phase N]` markers for sections not yet ready (preferred over fluff). When in doubt, ship a stub with TBD over polished prose with hand-waved numbers.

### Library-first coding

Reusable logic lives in the `arc_agent/` package and is indexed in `docs/LIBRARY.md`. **Before writing any function-shaped piece of code, follow this loop:**

1. Search `docs/LIBRARY.md` for an existing function. If found → `from arc_agent.<mod> import <fn>` and use it. Done.
2. If not found, decide which submodule it belongs to (see the "模块划分" table in `LIBRARY.md`). Anything that will plausibly be called twice goes in the library — not in a script.
3. Implement it in `arc_agent/<mod>.py` with type annotations and a one-line docstring. Pure functions where possible; library code does no env/file I/O (those belong in entry scripts and `eval.py`).
4. Add a test in `tests/test_<mod>.py` (normal input + at least one edge case) and verify it passes.
5. Register it in `docs/LIBRARY.md` using the entry template at the top of that file.
6. If the submodule didn't exist, also add a row to `docs/CODE_MAP.md`.

Deprecation: don't delete; mark `Status: deprecated → <replacement>` in `LIBRARY.md` and keep the symbol for at least two weeks before removing.

### Scientific iteration — Hypothesize, Execute, Iterate (HEI)

The whole project — both the agent's runtime decision loop and our development process — runs on the same scientific-method loop. **There is no "let me try X and see what happens".**

**Development-process HEI** (applies to every coding task and experiment):

- **Hypothesize**: before running anything, commit to a *predicted* number or outcome. Examples: "LLMAgent will raise mean RHAE by ≥0.05 over Random on demo set"; "adding prompt cache will cut per-step token cost by ≥40%". Write the hypothesis into the task description, the EXPERIMENTS.md entry, or the RESEARCH.md attempt — whichever applies.
- **Execute**: keep the run *cheap and bounded* so iteration is fast. Cap episodes, cap tokens, cap wall-clock. Smaller experiments that resolve a hypothesis beat one giant run that doesn't.
- **Iterate**: every result, **success or failure**, must produce a documented next action. Branch the next step on the outcome:
  - *Hypothesis confirmed* → what's the next, sharper hypothesis?
  - *Hypothesis falsified* → what does that rule out, what's the new candidate?
  - *Inconclusive* → what tighter experiment disambiguates?
  Shelving a result without a next step is the failure mode to avoid. Every EXPERIMENTS.md entry has an explicit `Iteration trigger` field for this.

**Agent-runtime HEI** (the inside-the-game decision loop): the agent itself is structured as Hypothesize → Execute → Iterate at every step — it maintains falsifiable rule hypotheses, picks actions for both goal progress and information gain, and updates beliefs after each observation. Details in `docs/ARCHITECTURE.md` "核心智能体循环".

When the two loops are aligned, our experiments validate agent improvements that the agent itself uses the same logic for at runtime.

## Repository State

Phase 1 in progress (per `docs/ROADMAP.md`). Phase 0 closed 2026-04-27 with end-to-end SDK working on `ls20` and a RandomAgent baseline on demo 25 (mean RHAE 0.000, pass rate 0%; see `docs/EXPERIMENTS.md`).

- `agent_starter.py` — single-game demo, imports `RandomAgent` + `play_one` from the library
- `eval.py` — batch evaluator across N games × M episodes, writes `runs/<ts>_<tag>.jsonl`; `--agent {random,llm,llm-haiku}`, `--budget` USD hard stop
- `arc_agent/` — library: `runner.py` (Agent Protocol + `play_one`), `observation.py` (grid → text/diff; `grid_to_image` planned Phase 2), `llm.py` (Anthropic wrapper with cumulative cost tracking), `agents/{random,llm}.py`
- `tests/` — pytest suite (5 files, 36 passing tests as of 2026-04-28)
- `vendor/ARC-AGI-3-Agents/` — read-only clone of the official scaffold for reference; do **not** edit
- Not yet a git repo (no `.git/`), no linter, no CI

**Phase 2 backbone decision (2026-04-28)**: local model changed from Qwen3-0.6B (text-only) to **Qwen2.5-VL-3B-Instruct** (multimodal). Reason: the 64×64 grid encoded as hex text is a 1D sequence where vertically adjacent cells are ~64 tokens apart — a 0.6B text model cannot reliably infer 2D spatial correlations. Qwen2.5-VL-3B's ViT-style vision encoder processes the grid as a rendered 512×512 PNG with 2D patch attention, directly capturing spatial structure. The CNN encoder + slot encoder + fusion layer plan is dropped in favour of this single end-to-end model. See `docs/ARCHITECTURE.md` for full details.

For the full file-by-file map see `docs/CODE_MAP.md`; for the seven-doc index see `docs/README.md`.

**Gotchas** (full list in `docs/RESEARCH.md` §4 "SDK 已知坑"):

- **`arc_agi/base.py` auto-loads `.env.example` at import time**, so `os.getenv("ARC_API_KEY")` returns the placeholder `"your_arc_api_key_here"` if the user hasn't created a real `.env`. All `if not key:` checks are bypassed and the SDK then 401s. Always check `key.startswith("your_")` too.
- **Anonymous mode is rejected** by `https://three.arcprize.org` (401 on `/api/games`). A real `ARC_API_KEY` is required from https://arcprize.org/api-keys, placed in `.env` (NOT `.env.example`).
- **`GameAction` has 8 enum members** (`RESET + ACTION1..ACTION7`), but only ACTION1–ACTION7 are user-callable game actions. `RESET` is the state-transition action used when `state ∈ {NOT_PLAYED, GAME_OVER}`. Only `ACTION6` is `is_complex()` (needs x, y ∈ [0,63]). In practice the runner passes coordinates via the action's pydantic `action_data` model — `env.step(action, data=action.action_data.model_dump())` — rather than constructing a raw `{"x":…, "y":…}` dict.
- **`FrameDataRaw.win_levels` is the total level count for the game, not the wins.** Use `levels_completed` for progress.
- **Windows console (cp1252) cannot encode Unicode arrows** like `→`. Use ASCII (`->`) in any print statement that may run on Windows.
- **Default `GAME_ID = "ls20"`**, version 9607627b at time of writing (SDK auto-downloads to `environment_files/ls20/9607627b/ls20.py` — game logic is local Python, not a remote service).

## Environment

`.venv/` was rebuilt with **Python 3.12** on 2026-04-27 (the prior 3.11 venv could not install `arcengine`, which requires ≥3.12). The interpreter at `C:\Users\sshuser\AppData\Local\Programs\Python\Python312\python.exe` was used as the base.

A real `ARC_API_KEY` from https://arcprize.org/api-keys must be in `.env` at the repo root (not `.env.example` — see Gotchas above). `requirements.txt` pins both `arc-agi>=0.9.8` and `arcengine>=0.9.3`.

No linter/CI yet. `pytest` is wired up — run the suite directly with the venv interpreter (see Commands below).

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
.venv/Scripts/python.exe -m pytest tests/test_runner.py::test_play_one_stops_on_win -q

# Recreate the venv from scratch (rare)
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

`eval.py` flags: `--agent {random,llm,llm-haiku}`, `--games <comma-list-or-prefix>` (empty = all available demo games), `--episodes N`, `--max-actions N` (default 80), `--tag <label>`, `--output <path>`. It writes one jsonl row per (game, episode) plus a final `__summary__` row to `runs/<ts>_<tag>.jsonl`.

## Core Concepts

**Scoring (RHAE)**: `S = min(1.0, h/a)²` where `h` = second-best human action count, `a` = agent actions. Quadratic penalty — being 2× slower yields 0.25, not 0.5. Later levels are weighted more heavily (`w_l = l`). Hard cutoff at 5× human action budget.

**Observation**: 64×64 grid, 16 possible colors per cell. May be a sequence of frames (for animations).

**Action space** (corrected 2026-04-27 — see `docs/RESEARCH.md` attempt #2): 7 actions in `GameAction`, imported `from arcengine import GameAction`:
- `ACTION1`=Up, `ACTION2`=Down, `ACTION3`=Left, `ACTION4`=Right (4 directional)
- `ACTION5`=Primary (interact / select / rotate / etc — game-specific)
- `ACTION6`=Coordinate (takes x, y ∈ [0, 63] via `env.step(action, data={"x":..., "y":...})`; check with `action.is_complex()`)
- `ACTION7`=Undo

**Datasets**: 25 public demo environments (easier), 55 semi-private (API testing), 55 fully private (competition eval). Access via `arc.make(game_id)`.

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
