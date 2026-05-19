# ARC-AGI-3 Agent — Architecture Evolution

> Short history of each architecture iteration we tried for the ARC-AGI-3 competition agent. Each block: **(a) what changed**, **(b) why we changed**, **(c) what problem it exposed**.

---

## v0 — RL with intrinsic F1 reward (2026-04-27, parked)

**Setup**: Qwen2.5-VL receives raw 64×64 grid as image. GRPO fine-tunes the agent. Reward signal = intrinsic F1 between the cells the agent **predicts will change** and the cells that **actually changed** after `env.step`.

**Why this design**: ARC-AGI-3 gives no instructions, so we have no win-trigger labels to supervise on. Dense, frame-by-frame intrinsic reward seemed like the only training signal available.

**Problem exposed**:
1. Qwen2.5-VL extracted objects from the 64×64 image at **0% accuracy** on ar25 (`ref_object_pipeline_zh.md`). Without correct perception, the predicted-change set was noise.
2. RL training requires a working baseline policy first; we did not yet have one.

**Parked** because the next iteration showed a deterministic perception path that obviated the need for VLM image input. Kept the RL code (`arc_agent/rewards.py`, `scripts/train_grpo.py`) for future re-integration after modules are verified.

---

## v1 — VLM with image input + 1-step ablations (2026-05-11, parked)

**Setup**: Same Qwen2.5-VL but used in a simpler "one ablation per script" setup (random / Claude API / VLM single-action). Image was still the model input.

**Why this design**: Establish a baseline before adding any structure.

**Problem exposed**:
- Qwen-VL image extraction continued at ~0% per-frame.
- The model could not name the cell positions correctly, so any instruction like "click on x,y" was garbage.

**Parked** in favor of v3.

---

## v3 — scipy perception + text-only Qwen (2026-05-11)

**Setup (core principle)**:
- **Vision = deterministic algorithm** (`scipy.ndimage.label`). Per non-background color, run 4-connected component labeling. Each component → one `ObjectRecord` with `(color, bbox, center, cells, shape_signature)`.
- **Reasoning = text LLM**. Qwen sees structured object descriptions (in text), never sees pixels.
- The bridge is structured data (`ObjectRecord`), not pixels.

8-block prompt: `[STATUS] [ACTIVE] [TEXTURE] [ACTION] [UNTRIED] [HISTORY] [GOAL] [ASK]`.

**Why this change**: `ref_object_pipeline_zh.md` benchmark showed **scipy 100% vs Qwen-VL 0%** on the same per-frame object extraction. Once perception is reliable, it can stop being the bottleneck.

**Problem exposed**:
- Single-agent loop wrote no persistent learning. Across rounds the agent forgot every action it had tested.
- High no-op rate on multi-step games (ACTION1 spam).

---

## v3.2 — Action Agent + Reflection Agent + persistent Knowledge (2026-05-14)

**Setup**: Two agents share the same backbone:
- **Action Agent** chooses the action each step.
- **Reflection Agent** runs after every step (not after every round), writes a JSON delta updating a `Knowledge` object that **persists across rounds** within one game_id.
- `Knowledge` fields: `action_semantics`, `goal_hypothesis`, `goal_confidence`, `rules`, `failed_strategies`, `rejected_goals`, `round_history`, `current_alert`, `click_targets`.

**Why this change**:
- Persistent learning across rounds. Reflection writes what it observed; next round Action reads it.
- Two agents let each one specialize: Action picks; Reflection learns.

**Problem exposed**:
- LLM **ignored advisory prompt language**. We told Reflection "do not write hypothesis = 'unknown'" and it still did. We told Action "do not pick ACTION1 if it was no-op last 3 times" and it still did.
- This produced the next architectural layer (hard rules).

---

## v3.2 + R1-R7 hard rules (2026-05-14 .. 05-16)

**Setup**: Orchestrator-level rules that **rewrite or filter** the agents' outputs, not just advise:
- **R1** sentinel filter (reject `"unknown" / "none"` as goal_hypothesis update)
- **R2** Knowledge-driven action mask (replace LLM-picked actions that are flagged ineffective)
- **R3** ActionAgent forced explore on no-op streak ≥ 5 or state-revisit ≥ 5
- **R4** contradiction filter (drop `rules_append` that contradicts `action_semantics`)
- **R5** failed_strategies cross-contamination filter
- **R6** action-described-goal filter (reject hypothesis starting with "ACTION_X")
- **R7** LOW-PRIORITY ACTIONS prompt block

**Why this change**:
- "Prompt only advises; orchestrator enforces" — direct consequence of v3.2 finding that LLM ignored prompts.

**Problem exposed**:
- These rules helped on individual issues but did not fix overall win rate. Production change_rate was 5-8% on main (v3.2 full).
- The rules were not individually ablated; we did not know which one was load-bearing.

---

## v3.2 + action_proposer K=3 (2026-05-16)

**Setup**:
- For each step, code generates **K=3 candidate actions**: one untried action (from `OutcomeLog`), one known-good (from `action_semantics`), one click-target candidate (from `Knowledge.click_targets`).
- Action prompt is restyled into multi-choice: "Pick A / B / C".

**Why this change**:
- Force the LLM to explore. Without proposer, the model anchored on ACTION1 (first in the legal-actions list) and stayed there.

**Problem exposed**:
- ar25 3×30 change_rate jumped to 60/70/75% (up from 23/17/17% on bare v3.2).
- But still 0 wins. The change_rate metric measures "did the frame change", not "did we get closer to win".

---

## det_goal v1 / v2 / v3 (2026-05-18, iterative)

**Setup**: Added two new modules on top of v3.2:
1. **`arc_agent/goal_evaluator.py`** — deterministic parser. Reads `goal_hypothesis` text, parses it into a `GoalPredicate` (e.g., `align_col target=0`), evaluates against current `ObjectRecord` list, returns True / False / None.
2. **Force-reject mechanism** — if parser says `achieved=True` but `env.state != WIN`, orchestrator clears the hypothesis and appends it to `rejected_goals`, forcing Reflection to rewrite.

**Iterations**:
- **v1**: Reflection token budget was 250 (default for `/no_think`). With `/think` enabled, the chain consumed all 250 tokens before producing JSON → empty deltas → no hypothesis stored. **Bug**: reflection truncation.
- **v2**: Reflection budget raised to 2048. Hypothesis now stored, but parser couldn't parse "to top edge" / "to center" — vocab too narrow.
- **v3**: Parser extended to 7 kinds (`align_col, align_row, align_any, move_to_row, move_to_col, move_to_center, stack, adjacent`) + verbs (`reach the X edge`, `match X with Y`).

**A/B bench**: Python parser **83% T-GOAL accuracy** vs LLM-as-judge 68% (recall on TRUE: parser 100% vs LLM 22%, 1.25M× faster).

**Problem exposed**:
- Reflection writes hypotheses in dialects the parser was not built for. Even after vocab extension, model preferred "match X with Y" (no axis) → `align_any` kind without a direction target.
- Action received hypotheses with no extractable direction.

---

## v4 clean_baseline (2026-05-19) — 5-phase ablation

**Setup**: Strip every legacy module that had never been individually validated; add them back one at a time. CLI flags:
```
--click-targets {on, off}
--action-semantics-from-llm {on, off}
--hard-rules {on, off}
--validate-hypothesis-schema {off, wide, strict}
--max-actions-total N  --max-rounds N    (step-budget pooling)
```

V4 minimal baseline: everything off except scipy perception, goal_evaluator, force_cot Action ASK block, Reflection schema validation. SmolLM3-3B `/no_think`.

**5-phase pipeline**:
1. **Phase 1**: `/think` vs `/no_think`. `/think` chain failed to close (0/10 even on short V4 prompt); `/no_think` reasoning visible 10/10. **Pick `/no_think`**.
2. **Phase 2**: V4 minimal ar25 1×200 budget-pooled. **Result: 11% change_rate, 0 wins**. Regression of 53pp vs SmolLM3 5×2×300 baseline (64%).
3. **Phase 3**: Add back one legacy module at a time. **`action_proposer` is the single critical module (+75pp)**. Other three (click_targets / action_semantics from LLM / hard_rules) each contribute +0pp alone.
4. **Phase 4**: V4 + propose on 5 G_base games × 1×200 budget pooled. **Mean 82% change_rate (+18pp vs baseline), 0/5 wins**.

**Why this change**:
- Methodology reset: previous "validated" modules were measured by proxy metrics (`change_rate / count / diversity`), not their PASS definitions.

**Problem exposed (per-module re-validation, 2026-05-19)**:
1. **BUG-1**: Reflection wrote 0 / 794 steps a directional hypothesis. Action had no direction signal.
2. **BUG-2**: 49% of steps' reasoning ≠ action choice (LLM letter-mapping confusion with the K=3 shuffled candidates), `orch_override` contributed 0/318 mismatches.
3. **BUG-3** (methodology root): bench used sanitized prompts (fixed letter mapping, given hypothesis); production used dynamic shuffle + free Reflection output → distribution shift.

**Status (open)**:
- 0/5 wins on G_base, despite +18pp change_rate.
- Module 1 (Goal Generation) and Module 3 (Reflection Loop) cannot be validated automatically. Human annotation requested in `docs/project/2026-05-19-v0-v4_clean_baseline/annotation_request.md`.
- Pathfinding (A* / BFS) is a candidate to complement Module 4 once Module 1 yields directional hypotheses (see `presentation/report.md`).

---

## What we kept across all iterations

- `scipy.ndimage.label` perception (v3 → today). Validated 100% vs Qwen-VL 0%.
- `Knowledge` persistence across rounds within one `game_id` (v3.2 → today).
- Text-only LLM (no pixel input to model) (v3 → today).
- The "vision is algorithm, reasoning is LLM" principle.

## What we tried and dropped

- Qwen-VL image input (v0, v1) — replaced by scipy perception.
- `/think` mode in production (det_goal v1) — chain does not close in long production prompts.
- `click_targets` bandit standalone (v3.2 → v4 Phase 3) — 0 / 5 production hit rate in cross-validation.
- `action_semantics` from LLM as standalone (v3.2 → v4 Phase 3) — 0pp on its own.
- Hard rules R1 / R4 / R5 / R6 / R7 in V4 baseline — 0pp on their own when proposer is the carrier.

## Reading order for new collaborators

1. `docs/README.md` — top-level 10-minute overview.
2. `docs/verify.md` — 7-module verification spec + current bugs.
3. `presentation/report.md` — achievements + future plan (this folder, English).
4. This file (`presentation/arc.md`) — architecture history.
5. `docs/project/2026-05-19-v0-v4_clean_baseline/report.md` — most recent ablation study.

## Cross-references

| Project doc | What it covers |
|---|---|
| `docs/project/2026-04-27-v0-rl/architecture.md` | v0 RL design |
| `docs/project/2026-05-11-v3-baseline/architecture.md` | scipy perception + text-only Qwen |
| `docs/project/2026-05-14-v3_2-double_agent/architecture.md` | Action + Reflection split |
| `docs/project/2026-05-16-v0-action_proposer/architecture.md` | K=3 candidate generator |
| `docs/project/2026-05-18-v0-force_cot/report.md` | force_cot Action prompt A/B |
| `docs/project/2026-05-18-v0-goal_judge_ab/report.md` | Python parser vs LLM judge |
| `docs/project/2026-05-19-v0-v4_clean_baseline/report.md` | Final v4 ablation + 5-game eval |
