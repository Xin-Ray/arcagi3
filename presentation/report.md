# ARC-AGI-3 Agent — Project Report

> Current state of the agent + rationale + what's validated + what's next.
> Companion to `presentation/arc.md` (architecture history).
> Source: synthesis of `docs/README.md` §2 + `docs/verify.md` + Phase 4 results.

---

## 1. Problem statement

ARC-AGI-3 (ARC Prize 2026) gives the agent a **64×64 grid game with no instructions, no documentation, no examples**. The agent must:

1. Observe the grid evolve as it issues actions (ACTION1..ACTION7).
2. Infer what the game wants (the hidden win condition).
3. Reach the win state within a step budget.
4. Generalize across 110 distinct games (25 public demo + 55 semi-private + 55 fully-private).

Constraints from the competition rules:
- **No task-specific optimization**: cannot fine-tune on a specific game, cannot use demo/private game labels.
- **Kaggle final eval**: T4 16 GB, offline (no network), ≤ 10 hours total wall time.
- **Scoring (RHAE)**: `S = min(1.0, h/a)²`. `h` = second-best human step count; `a` = agent step count. Quadratic penalty — twice as slow ⇒ 0.25 score. Hard cutoff at 5× human budget.

State of the art (community):
- Symbolica Agentica: 36.08% (7 / 25 demo wins).
- Frontier LLMs (GPT / Gemini / Claude / Grok) on the fully-private set: **<1%**.

---

## 2. Module decomposition (7 modules, by data flow)

```
frame (raw 64×64 grid)
  ↓
[0] Perception      scipy.ndimage.label → ObjectRecord list
  ↓ + history
[1] Goal Generation  Reflection Agent writes goal_hypothesis
  ↓
[2] Goal Recognition parser/judge: is the hypothesis achieved?
  ↓  (if achieved=True but env != WIN)
[3] Reflection Loop  reject hypothesis → push rejected_goals → force rewrite
  ↓
[4] Action Selection Action Agent picks an ACTION aligned with hypothesis
  ↓
[5] force_cot Prompt instructs model to reason step-by-step toward goal
  ↓
[6] K=3 Candidates   action_proposer presents 3 candidates for diversity
  ↓
env.step(action) → next frame → loop back to [0]
```

**Core design principle**: vision is a deterministic algorithm; reasoning is the LLM; the bridge is `ObjectRecord` structured data. **The LLM never sees pixels** (proven by scipy 100% vs Qwen-VL 0% on ar25 object extraction).

---

## 3. Why **not** BFS / DFS / A\* as the primary action policy

A natural-looking alternative is: encode the grid as a graph, run BFS or A\* to a goal state, emit the action sequence. We do **not** use this as the primary policy. Reasons:

### 3.1 The goal is hidden

BFS / A\* needs a **goal state** to search toward. ARC-AGI-3 hides the win condition. Without a goal, BFS becomes exhaustive expansion over the state space — exponential blowup. We could "search until env.state == WIN" but with 7 actions per step and 200 steps budget, the search tree has 7²⁰⁰ ≈ 10¹⁶⁹ leaves.

### 3.2 The transition function is unknown

BFS / A\* needs `next_state = T(state, action)`. We do not know what ACTION_X does until we try it. Some games make ACTION1 move the active object up; some make ACTION1 a no-op or a cycle. The transition function must be **learned** before any planner can use it.

### 3.3 The win condition is opaque even after reaching a "goal"

Even if we infer a target state and reach it via planning, `env.WIN` may not trigger because our inferred goal was wrong. We must keep an **inference layer** (Reflection) that updates the hypothesis when the env does not WIN despite the agent thinking it should.

### 3.4 What IS valuable from BFS / A\*

Once Modules 0 (perception) + 1 (goal generation) + 2 (parser) produce a **directional, parseable hypothesis** with a concrete target (e.g., `move_to_row(target=63)`) and an **observed `action_effects` table** (which action moves which object which way, learned from `OutcomeLog`), a tiny A\* is trivially correct:

- Input: `(current_obj_pos, target_pos, action_effects)`.
- Output: shortest sequence of ACTIONs to move object to target.
- Cost: ~ 50 lines.
- Speed: 100% reach in `D = |Δrow| + |Δcol|` steps, vs current LLM 0.5 ^ D probability.

**A\* becomes Module 4** (Action Selection) **complement** — only after Modules 0/1/2 are verified. It does not replace the inference loop; it speeds up the execution loop.

We will integrate A\* **once Module 1 (Goal Generation) is verified to produce directional hypotheses** (currently blocked by BUG-1, see §5).

---

## 4. How RL fits in (later)

We have a parked RL line: GRPO + intrinsic F1 reward (`arc_agent/rewards.py`, `scripts/train_grpo.py`). 10 unit tests pass; real training has not been run. Plan:

1. **First**: verify each module passes both bench and production check (see `docs/verify.md` §3 each module's per-module check).
2. **Then**: train each module's LLM-dependent piece via RL, using the intrinsic F1 reward as the dense signal:
   - **Module 1 (Goal Generation)**: reward = match between hypothesis content and what got WIN-triggered later (sparse, needs labelled trajectories).
   - **Module 4 / 5 (Action Selection / force_cot)**: reward = (object delta dotted with hypothesis direction). Encourages goal-aligned moves.
   - **Module 6 (K=3 Candidates)**: reward = whether the chosen letter mapped to a goal-aligned action.

**Why RL after verification, not before**: training a module that does not even pass an analytical PASS definition burns compute on noise. The current 5-phase ablation gave us a deterministic baseline (V4 + propose, 82% change_rate). RL fine-tuning is the next step **once we know the deterministic bottleneck**.

---

## 5. Current validated results

### 5.1 Per-module status (2026-05-19, `docs/verify.md` §5)

| # | Module | Bench | Production | Conclusion |
|---:|---|:-:|:-:|---|
| 0 | Perception (scipy) | 100% vs VLM 0% | stable across 800+ steps | **PASS** |
| 1 | Goal Generation (Reflection) | none (game GT hidden) | waiting human label | **UNKNOWN** |
| 2 | Goal Recognition (parser) | T-GOAL 83% | 0/5 wins → can't measure precision/recall | bench PASS, prod untested |
| 3 | Reflection Loop (force-reject) | none | waiting human label | **UNKNOWN** |
| 4 | Action Selection | T-NAV-1 71%, T-NAV-3 97% | **0/794 directional steps** | **prod FAIL** (blocked by Module 1) |
| 5 | force_cot Prompt | T-NAV-3 +30pp | 51% match (49% LLM letter confusion) | **prod FAIL** |
| 6 | K=3 Candidates (proposer) | ablation +75pp | K=3 present 78% (22% < 3) | **partial PASS** |

### 5.2 V4 ablation table (Phase 3 — single-module add-back)

| Configuration | change_rate | wins | ACTION1 share | Verdict |
|---|---:|---:|---:|---|
| V4 baseline (all legacy off) | 11% | 0 | 94% | reference |
| V4 + click_targets | 11% | 0 | 92% | NEUTRAL |
| **V4 + action_proposer** | **86%** | **0** | **20%** | **CRITICAL +75pp** |
| V4 + action_semantics from LLM | 11% | 0 | 94% | NEUTRAL |
| V4 + hard_rules R1/R4/R5/R6/R7 | 11% | 0 | 94% | NEUTRAL |

Sources: `outputs/v4_ablate_{baseline,click,propose,semantics,hardrules}_s42_*/summary.json`.

### 5.3 V4 + propose × 5 G_base games (Phase 4)

| Game | rounds | total steps | change_rate | wins |
|---|---:|---:|---:|---:|
| ar25 | 2 | 172 / 200 | **87%** | 0 |
| bp35 | 2 | 72 / 200 | **85%** | 0 |
| cd82 | 2 | 200 / 200 | **79%** | 0 |
| cn04 | 2 | 150 / 200 | **74%** | 0 |
| dc22 | 2 | 200 / 200 | **84%** | 0 |
| **mean** | 2 | 159 | **82%** | **0 / 5** |

Sources: `outputs/v4_phase4_g{1..5}_*_s42_*/summary.json`.

### 5.4 Cross-baseline comparison (5 G_base games, mean change_rate)

```
main e07e7d1 (v3.2 full, /no_think, Qwen)  ──────────  5-8%   0/5 wins
SmolLM3 5×2×300 (full mods, no goal_eval) ──────────  64%    0/5 wins
det_goal v3 ar25 1×100 (/think truncated) ──────────  75%    0/1 wins
V4 + propose (this report)                 ──────────  82%    0/5 wins  +18pp
```

The +18pp delta over the SmolLM3 baseline comes from `goal_evaluator` + parser extensions + force-reject mechanism + Reflection schema validation. These are **net additive improvements**, validated by the ablation.

---

## 6. Open bugs blocking 0 → 1+ wins

(Detailed in `docs/verify.md` §2.)

### BUG-1 (fatal): Hypothesis is non-directional

- **Symptom**: 0 / 794 Phase 4 steps had a directional `goal_hypothesis` (`move_to_row / col / center / stack`). All Reflection output was `align_any` or `match X with Y` (no axis).
- **Cause**: SmolLM3 `/no_think` mode prefers "match X with Y" phrasing. `/think` mode wrote directional hypotheses but its chain failed to close in production prompts.
- **Action**: try `--validate-hypothesis-schema strict` (reject `align_any`, force the model to write a `move_to_X` kind).

### BUG-2 (severe): reasoning ↔ action 49% pure LLM disconnect

- **Symptom**: in 49% of steps with a parseable reasoning text, the ACTION mentioned in reasoning is **not** the ACTION recorded in trace.
- **Cause**: action_proposer K=3 letter mapping is shuffled per step ("A: ACTION3, B: ACTION1, C: ACTION6"). The model writes reasoning naming "ACTION1" but outputs `choice: A` (which actually maps to ACTION3). `orch_override` contributed **0 of 318 mismatches** — this is the LLM tracking the letter map incorrectly, not the orchestrator overriding.
- **Action**: fix letter mapping to a deterministic order (A=ACTION1 always) so the model does not have to track shuffled letters.

### BUG-3 (methodology root): bench-vs-production distribution shift

- bench used: sanitized prompts (fixed letter mapping, given hypothesis).
- production uses: dynamic shuffle + free Reflection output.
- → bench PASS does not imply production PASS.
- **Action**: build production-equivalent benches (T-DIALECT for Reflection dialect; T-MAPPING for dynamic letter tracking).

---

## 7. Charts (referenced)

| Chart description | Source path |
|---|---|
| 5-game V4+propose change_rate (bar) | `outputs/v4_phase4_g{1..5}_*/summary.json` (data only, no chart yet) |
| Ablation +click / +propose / +semantics / +hardrules | `docs/project/2026-05-19-v0-v4_clean_baseline/phase3_ablation.md` §1 |
| Reasoning ↔ action mismatch breakdown | `docs/project/2026-05-19-v0-v4_clean_baseline/module_validation.md` §3 |
| T-GOAL parser vs LLM judge confusion | `docs/project/2026-05-18-v0-goal_judge_ab/report.md` §3.2 |
| Subtask probe results per task | `docs/project/2026-05-17-v0-subtask_decomp/report.md` §2 |
| Step PNG visualisations + play GIFs | `outputs/v4_phase4_g{1..5}_*/round_00/step_*.png`, `play.gif` |
| Knowledge per step (hypothesis evolution) | `outputs/v4_phase4_g{1..5}_*/round_00/knowledge_per_step.jsonl` |
| Reflection raw output | `outputs/v4_phase4_g{1..5}_*/round_00/reflection_raw.txt` |
| Action raw output | `outputs/v4_phase4_g{1..5}_*/round_00/action_raw.txt` |

(Chart PNG files have not been generated yet; we have the raw JSON / JSONL data. Generating the figures is a P2 todo.)

---

## 8. Next steps (priority ordered)

| P | Action | What it fixes |
|---|---|---|
| P0 | Human annotate `docs/project/2026-05-19-v0-v4_clean_baseline/annotation_request.md` (~15 min, 25-35 labels) | Module 1 + 3 UNKNOWN → KNOWN |
| P0 | Rerun ar25 with `--validate-hypothesis-schema strict` | BUG-1: force directional hypotheses |
| P0 | Fix action_proposer letter mapping to deterministic order | BUG-2: remove letter confusion |
| P0 | Add 3 fields to `trace.jsonl`: `n_objects`, `force_reject_event`, `candidates_dump` | basis for Module 0/3/6 integration checks |
| P1 | Implement A\* pathfinder for Module 4 (~200 lines) | once BUG-1 is fixed, navigation becomes deterministic 100% reach in `D` steps |
| P1 | Build T-DIALECT bench (Reflection free-write dialect distribution) | Module 1 bench |
| P1 | Build T-REJECT bench (rejected hypothesis correctness) | Module 3 bench |
| P1 | Build T-MAPPING bench (dynamic letter shuffle tracking) | Module 5 bench |
| P2 | RL fine-tune Modules 1, 4, 5 with intrinsic F1 reward | once modules pass deterministic checks |
| P2 | Upgrade backbone to SmolLM3-7B or Qwen3-4B | model-capacity ceiling test |

---

## 9. Project goal status

- **Current**: V4 + propose, **82% mean change_rate on 5 G_base games, 0 / 5 wins**.
- **Gap to community SOTA (Symbolica Agentica 36% wins on demo)**: still 36 percentage points of wins.
- **Gap to "minimum viable submission" (≥ 1 demo win in 10h)**: blocked by BUG-1 + BUG-2. Both have well-defined P0 fixes above; both fixes are 1-2 day implementation each.

---

## 10. Source layout (for someone new to the repo)

```
docs/
  README.md              top-level overview (incl. §2 7-module map)
  verify.md              per-module verification spec (this report's source)
  GLOSSARY_zh.md         all named concepts
  project/
    2026-05-19-v0-v4_clean_baseline/   most recent ablation + 5-game eval
    2026-05-18-v0-goal_judge_ab/       Python parser vs LLM judge
    2026-05-18-v0-force_cot/           force_cot prompt A/B
    2026-05-17-v0-subtask_decomp/      5 subtask validation
    ...                                 9 more project folders, chronological

presentation/
  arc.md                 architecture history (English)
  report.md              this file

arc_agent/               library (importable)
  object_extractor.py    Module 0
  goal_evaluator.py      Module 2 (parser)
  action_proposer.py     Module 6 (K=3 candidates)
  agents/
    action_agent.py      Module 4
    reflection_agent.py  Module 1
  ...

scripts/                 entrypoints
  run_v3_multi_round.py  main orchestrator (also implements Module 3 force-reject + step-budget pooling)
  bench_subtask*.py      subtask probe benches
  bench_goal_judges.py   parser vs LLM judge A/B
  ...

outputs/                 experiment outputs (gitignored)
```
