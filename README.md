# ARC-AGI-3 Agent — ARC Prize 2026

> **3-minute project takeover.** Read this top-to-bottom; everything else is a deeper reference.

Competition: <https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3>
Milestones: **2026-06-30** open-source · **2026-09-30** final.

---

## 1. Project Goal

Build an agent that plays ARC-AGI-3 turn-based puzzle games autonomously, with **no instructions** — it must infer object behavior and win conditions on its own.

**Current approach (since 2026-05-11):** zero-shot `Qwen2.5-VL-3B-Instruct` baseline → GRPO fine-tuning with **intrinsic F1 reward** (each step the agent predicts which cells will change; F1 between predicted and real change set becomes a dense reward, no need to wait for WIN).

---

## 2. Data Flow

```
ARC-AGI-3 SDK env  ── env.step ──▶  FrameDataRaw (grid frames, state, available actions)
        │
        ▼
arc_agent.observation.latest_grid()      → 64×64 numpy
arc_agent.observation.grid_to_image()    → 512×512 PNG
        │
        ▼
┌──────────────────────────────────────────────┐
│  arc_agent.agents.vlm.VLMAgent.choose()      │
│    • build prompt (image + state + history)  │
│    • Qwen2.5-VL-3B-Instruct (LoRA fine-tuned)│
│    • parse JSON: action + predicted_diff     │
└──────────────────────────────────────────────┘
        │
        ▼  GameAction
   env.step()  ───▶  s_{t+1}
        │
        ▼
arc_agent.rewards.real_changes(s_t, s_{t+1})    → ground-truth change set
arc_agent.rewards.verify_prediction_f1(...)     → F1 ∈ [0, 1]
        │
        ▼
intrinsic reward (training) │ rule_table update (inference)
```

The agent never sees instructions. The F1 of its prediction is the only signal it gets between sparse WIN events.

---

## 3. Which File To Run

```bash
# 0. one-time setup
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
# put a real ARC_API_KEY (https://arcprize.org/api-keys) into .env

# 1. tests must pass before anything else
.venv\Scripts\python.exe -m pytest tests/ -q

# 2. sanity check: one game with RandomAgent
.venv\Scripts\python.exe scripts/agent_starter.py

# 3. batch eval (RandomAgent on all demo games, 1 episode each)
.venv\Scripts\python.exe scripts/eval.py
#    → outputs/runs/<ts>_random_baseline.jsonl
```

**RL pipeline** — these scripts are planned, not yet built. See `docs/ARCHITECTURE_RL.md` §9 for what each one needs to do:

```bash
.venv\Scripts\python.exe scripts/run_baseline.py     # ★ Go/no-go gate
.venv\Scripts\python.exe scripts/run_grpo.py
.venv\Scripts\python.exe scripts/run_validation.py
```

---

## 4. Correct Version (allow-list)

Only these files are active. Everything not listed here is in `archive/` and **must not** be touched without first promoting it back here.

| Active | Path | Purpose |
|---|---|---|
| ✅ Library | `arc_agent/runner.py` | Agent Protocol + `play_one()` single-game loop |
| ✅ Library | `arc_agent/observation.py` | `latest_grid` / `grid_to_image` / `grid_diff` / `animation_to_text` / `summarize_frame` |
| ✅ Library | `arc_agent/rewards.py` | `changes_to_set` / `real_changes` / `verify_prediction_f1` |
| ✅ Library | `arc_agent/llm.py` | Anthropic SDK wrapper (Claude API for dev/baseline) |
| ✅ Library | `arc_agent/agents/random.py` | `RandomAgent` baseline |
| ✅ Library | `arc_agent/agents/llm.py` | `LLMAgent` (Claude API, dev / silver-label use) |
| ⬜ Planned | `arc_agent/agents/vlm.py` | `VLMAgent` — see ARCHITECTURE_RL.md §9 Step 3 |
| ⬜ Planned | `arc_agent/viz.py` | GIF / per-step composite — Step 4 |
| ⬜ Planned | `arc_agent/train_grpo.py` | GRPO trainer setup — Step 7 |
| ✅ Script | `scripts/agent_starter.py` | Single-game RandomAgent demo |
| ✅ Script | `scripts/eval.py` | Batch evaluator (`--agent {random,llm,llm-haiku}`) |
| ⬜ Planned | `scripts/run_baseline.py` | RL Step 5–6: zero-shot Qwen on G_base, GIF + summary |
| ⬜ Planned | `scripts/run_grpo.py` | RL Step 7: GRPO training on G_train |
| ⬜ Planned | `scripts/run_validation.py` | RL Step 8: post-training eval on G_val |
| ✅ Doc | `README.md` | this file |
| ✅ Doc | `docs/ARCHITECTURE_RL.md` | deep design + §9 step-by-step impl plan |
| ✅ Doc | `CLAUDE.md` | Claude Code working rules + gotchas |
| ✅ Doc | `TASK_OVERVIEW.md` | full competition rules |

**Folder shape**

```
.
├── arc_agent/        library (no I/O outside model loading)
├── scripts/          entry points (orchestration only, no reusable logic)
├── tests/            pytest (72 passing)
├── data/             inputs/ (test PNGs), train/ (silver-label JSONL)
├── outputs/          runs/, checkpoints/, baseline_<ts>/, etc. (gitignored)
├── docs/             ARCHITECTURE_RL.md only
├── archive/          everything obsolete — do not touch
└── vendor/           read-only third-party (ARC-AGI-3-Agents)
```

---

## 5. Result Summary

| Date | Step | Result |
|---|---|---|
| 2026-04-27 | RandomAgent baseline on demo 25 | mean RHAE **0.000**, pass rate **0%** |
| 2026-05-08 | Qwen2.5-VL-3B capability test (5 cases) | **4/5 pass** — T3 fail because zero-shot model outputs `"Right"` not `"ACTION4"` |
| 2026-05-08 | QLoRA smoke training | 1 epoch on 60 silver-label steps, pipeline OK, adapter saved |
| **TBD** | **Step 6 baseline** (Go/no-go gate) | not yet run — see TODO |

Pre-registered hypotheses for Step 6: F1 ≥ 0.30, parse rate ≥ 0.70, RHAE ≤ 0.05. See `docs/ARCHITECTURE_RL.md` §5.2 for the iteration triggers.

---

## 6. TODO

**Active:** `docs/ARCHITECTURE_RL.md` §9 — file-level steps with acceptance criteria.

| # | What | Status |
|---|---|---|
| 1 | `grid_to_image()` | ✅ done |
| 2 | `rewards.py` (changes_to_set / real_changes / verify_prediction_f1) | ✅ done |
| 3 | `arc_agent/agents/vlm.py` + backbone abstraction | ⬜ next |
| 4 | `arc_agent/viz.py` + `scripts/make_gif.py` | ⬜ |
| 5 | `scripts/run_baseline.py` | ⬜ |
| 6 | **Run baseline + check Hypothesis** ★ Go/no-go | ⬜ |
| 7 | `scripts/run_grpo.py` + `arc_agent/train_grpo.py` | ⬜ |
| 8 | GRPO training + validation | ⬜ |

**Known risks**

- **Step 6 may fail.** If F1 < 0.1, fallback is two-image input (prev + curr); if that also fails, fall back to BC training (paused branch in `archive/bc_scripts/`).
- **Kaggle no-network constraint.** Final submission must run offline. Qwen2.5-VL-3B 4-bit fits in T4 16GB, but full pipeline timing ≤ 10h on 110 games is not yet measured.

---

## Conventions (read once, then assumed)

- `arc_agent/` is a Python package. Anything called twice goes in here, not in `scripts/`.
- Filenames are functional (`agent_starter.py`, `run_baseline.py`), never temporal (`final.py`, `_v2.py`, `_new.py`).
- Outputs are always under `outputs/<run_kind>_<timestamp>/` — never overwritten.
- `archive/` is one-way: things go in, nothing comes out without explicit promotion to the allow-list above.
- See `CLAUDE.md` for the full library-first rule and known SDK gotchas.
