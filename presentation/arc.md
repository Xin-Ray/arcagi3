# ARC-AGI-3 Agent — Architecture Evolution

> Each iteration: a diagram, the <span style="color:#1f77b4">**reason we changed**</span> (blue), and the <span style="color:#d62728">**problem it exposed**</span> (red).

---

## v0 — RL with intrinsic F1 reward (parked)

```mermaid
flowchart LR
    G[64x64 grid] --> VLM[Qwen2.5-VL<br/>image input]
    VLM --> P[Predicted change cells]
    VLM --> A[Action]
    A --> ENV[env.step]
    ENV --> O[Actual change cells]
    P --> F1{Intrinsic F1<br/>P vs O}
    O --> F1
    F1 --> RL[GRPO update<br/>policy]
    RL --> VLM
```

<span style="color:#1f77b4">**Reason for this design**</span>: ARC-AGI-3 gives no labels, so no supervised signal exists. Intrinsic F1 between predicted and actual change set is the only dense reward.

<span style="color:#d62728">**Problem exposed**</span>:
1. <span style="color:#d62728">Qwen-VL extracted objects at 0% per-frame accuracy</span> on a public game. Predicted change set was noise.
2. <span style="color:#d62728">No working baseline policy</span> to RL-train on top of.

---

## v1 — VLM with image input + 1-step ablations (parked)

```mermaid
flowchart LR
    G[64x64 grid] --> VLM[Qwen2.5-VL<br/>image input]
    VLM --> A[Action]
    A --> ENV[env.step]
    ENV --> G
```

<span style="color:#1f77b4">**Reason for this design**</span>: Establish a single-step baseline (random / Claude API / VLM) before any structure.

<span style="color:#d62728">**Problem exposed**</span>: <span style="color:#d62728">VLM cannot name cell positions reliably</span> from the 64x64 image. Instructions like "click on (x, y)" produced garbage. Same root as v0.

---

## v3 — scipy perception + text-only Qwen

```mermaid
flowchart LR
    G[64x64 grid] --> SCIPY[scipy.ndimage.label<br/>per non-bg color]
    SCIPY --> OBJ[ObjectRecord list<br/>color, bbox, center, cells]
    OBJ --> P8[8-block text prompt<br/>STATUS, ACTIVE, TEXTURE,<br/>ACTION, UNTRIED, HISTORY,<br/>GOAL, ASK]
    P8 --> QWEN[Qwen2.5-VL<br/>text-only mode<br/>NO pixel input]
    QWEN --> A[Action]
    A --> ENV[env.step]
    ENV --> G
```

<span style="color:#1f77b4">**Reason for this design**</span>:
- <span style="color:#1f77b4">Benchmark showed scipy 100% vs Qwen-VL 0%</span> on the same object-extraction task.
- <span style="color:#1f77b4">Make vision deterministic; let LLM only do reasoning</span>. Connect them via structured `ObjectRecord` data. The LLM never sees pixels.

<span style="color:#d62728">**Problem exposed**</span>:
1. <span style="color:#d62728">Single-agent loop wrote no persistent learning</span>; every new round forgot what was tested.
2. <span style="color:#d62728">High no-op rate, ACTION1 spam</span> across multi-step games.

---

## v3.2 — Action Agent + Reflection Agent + persistent Knowledge

```mermaid
flowchart LR
    G[grid_t] --> SCIPY[scipy perception]
    SCIPY --> OBJ[ObjectRecord]
    OBJ --> AA[Action Agent<br/>picks ACTION]
    AA --> ENV[env.step]
    ENV --> G2[grid_t+1]
    G2 --> RA[Reflection Agent<br/>writes delta JSON]
    RA --> K[Knowledge<br/>action_semantics<br/>goal_hypothesis<br/>rules, rejected_goals<br/>PERSISTS across rounds]
    K --> AA
    K --> RA
```

<span style="color:#1f77b4">**Reason for this design**</span>:
- <span style="color:#1f77b4">Persistent learning across rounds</span> — Reflection writes; next round Action reads.
- <span style="color:#1f77b4">Specialise the two agents</span> — Action picks, Reflection learns.

<span style="color:#d62728">**Problem exposed**</span>: <span style="color:#d62728">LLMs ignored advisory prompt language</span>. "Do not write `unknown` as goal" → it still did. "Do not pick ACTION1 if it was no-op last 3 times" → it still did. This forced the next layer — orchestrator-level hard rules.

---

## v3.2 + R1-R7 hard rules

```mermaid
flowchart TB
    subgraph LLM_LAYER [LLM layer]
        AA[Action Agent picks ACTION_X]
        RA[Reflection Agent writes delta]
    end
    subgraph ORCH [Orchestrator hard rules]
        R1[R1 sentinel filter<br/>reject unknown/none/tbd]
        R2[R2 action mask<br/>swap ineffective action]
        R3[R3 forced explore<br/>no-op streak >= 5]
        R4[R4 contradiction filter]
        R5[R5 failed cross-contam]
        R6[R6 action-described goal filter]
        R7[R7 LOW-PRIORITY prompt block]
    end
    AA --> R2 --> R3 --> A_OUT[Final action]
    RA --> R1 --> R4 --> R5 --> R6 --> KU[Knowledge update]
    R7 --> AA
```

<span style="color:#1f77b4">**Reason for this design**</span>: <span style="color:#1f77b4">"Prompt only advises; orchestrator enforces"</span>. Hard rules rewrite or filter LLM output, not just request behaviour.

<span style="color:#d62728">**Problem exposed**</span>:
1. <span style="color:#d62728">Production change_rate stayed 5-8%</span> on main with all 7 rules on.
2. <span style="color:#d62728">No individual ablation</span>: we did not know which rule was load-bearing.

---

## v3.2 + action_proposer K=3 candidates

```mermaid
flowchart LR
    OBJ[ObjectRecord] --> PROP[action_proposer<br/>generates K=3:<br/>1 untried<br/>1 known-good<br/>1 click-target]
    PROP --> P_MC[multi-choice prompt<br/>A: ACTION3<br/>B: ACTION1<br/>C: ACTION6]
    P_MC --> LLM[LLM picks letter]
    LLM --> R[Resolver:<br/>letter -> action]
    R --> A[Action]
```

<span style="color:#1f77b4">**Reason for this design**</span>: <span style="color:#1f77b4">force the LLM to explore</span> by hiding ACTION1 from the default pick. The model can no longer anchor on "first in legal list".

<span style="color:#d62728">**Problem exposed**</span>:
1. <span style="color:#1f77b4">ar25 change_rate jumped 23-17-17% to 60-70-75%</span> (good).
2. <span style="color:#d62728">But still 0 wins</span>. change_rate measures "frame moved", not "moved toward win".

---

## det_goal v1 / v2 / v3 (parser + force-reject mechanism)

```mermaid
flowchart TB
    RA[Reflection Agent<br/>writes goal_hypothesis] --> H[hypothesis text]
    H --> PARSE[goal_evaluator.parse_goal_hypothesis<br/>regex + structural parser]
    PARSE --> PRED{GoalPredicate?<br/>kind, colors, target}
    PRED -- yes --> EVAL[evaluate_predicate<br/>vs ObjectRecord]
    PRED -- no --> DROP[drop / fallback]
    EVAL --> V{verdict: True / False / None}
    V -- True + env != WIN --> REJ[FORCE-REJECT<br/>clear hypothesis<br/>push to rejected_goals]
    V -- False --> CONT[continue toward target]
    V -- None + frame_objs --> HAL[HALLUCINATED alert<br/>colors not in frame]
    REJ --> RA
    HAL --> RA
```

<span style="color:#1f77b4">**Reason for this design**</span>:
- <span style="color:#1f77b4">Goal recognition needs to be deterministic</span> — synthetic bench showed LLM scored only 30-35% on "is the hypothesis achieved?" while a Python parser scored 83% with 1.25M× speed and 100% recall on TRUE cases (vs LLM 22%).
- <span style="color:#1f77b4">When the agent thinks the goal is met but env disagrees, reject the hypothesis</span>. This is the only loop that can self-correct when the LLM hallucinates.

<span style="color:#d62728">**Problems exposed across iterations**</span>:
- <span style="color:#d62728">v1: Reflection token budget (250) truncated the JSON</span> output before it could emit a hypothesis. Empty deltas, nothing stored.
- <span style="color:#d62728">v2: parser vocab too narrow</span> — Reflection wrote "to top edge" and "to the center", but parser only recognised "in column 0" / "vertically aligned".
- <span style="color:#d62728">v3: Reflection now favours "match X with Y" dialect</span> (under `/no_think` mode) — parser sees this as `align_any` with no direction target.

---

## v4 clean_baseline (current)

```mermaid
flowchart LR
    subgraph VALIDATED [Kept: validated modules]
        P0[Module 0: scipy perception]
        P2[Module 2: parser]
        P3[Module 3: force-reject]
        P5[Module 5: force_cot prompt]
        P6[Module 6: action_proposer]
    end
    subgraph REMOVED [Stripped: legacy unvalidated]
        X1[click_targets bandit]
        X2[action_semantics from LLM]
        X3[hard_rules R1/R4/R5/R6/R7]
        X4[R2 action mask]
        X5[8-block prompt blocks]
    end
    subgraph CLI_FLAGS [V4 CLI ablation flags]
        F1[--click-targets on/off]
        F2[--action-semantics-from-llm on/off]
        F3[--hard-rules on/off]
        F4[--validate-hypothesis-schema wide/strict]
        F5[--max-actions-total N<br/>step-budget pooling]
    end
```

<span style="color:#1f77b4">**Reason for this design**</span>:
- <span style="color:#1f77b4">Methodology reset</span> — every legacy module was kept just by inertia; we never knew which was carrying load and which was dead weight.
- <span style="color:#1f77b4">CLI flags allow single-module ablation</span> so we can re-validate.

**5-phase ablation result**:

| Configuration | change_rate (ar25 100 steps) | ACTION1 share | Verdict |
|---|---:|---:|---|
| V4 baseline (all off) | 11% | 94% | reference |
| V4 + click_targets | 11% | 92% | NEUTRAL |
| <span style="color:#1f77b4">**V4 + action_proposer**</span> | <span style="color:#1f77b4">**86%**</span> | <span style="color:#1f77b4">**20%**</span> | <span style="color:#1f77b4">**CRITICAL (+75pp)**</span> |
| V4 + action_semantics | 11% | 94% | NEUTRAL |
| V4 + hard_rules | 11% | 94% | NEUTRAL |

5-game V4 + propose: **mean 82% change_rate, 0 / 5 wins**.

<span style="color:#d62728">**Problems exposed by per-module re-validation**</span>:
- <span style="color:#d62728">BUG-1 (fatal): 0 / 794 Phase 4 steps had a directional hypothesis</span> — Reflection wrote `match X with Y` instead of `move_to_row N`, so Action had no direction signal.
- <span style="color:#d62728">BUG-2 (severe): 49% of steps reasoning ≠ action</span> — model wrote "I pick ACTION1" but output `choice: A` which mapped to ACTION3. Pure LLM letter-mapping confusion; orchestrator override contributed 0 of 318 mismatches.
- <span style="color:#d62728">BUG-3 (methodology root): bench-vs-production distribution shift</span> — bench used sanitized inputs (fixed letter mapping, pre-written hypotheses); production used dynamic shuffle + free output. The 6 bench-PASS modules were not actually validated for production.

---

## What we kept across all iterations

- `scipy.ndimage.label` perception. **<span style="color:#1f77b4">Reason kept: 100% vs VLM 0%</span>.**
- `Knowledge` persistence across rounds. **<span style="color:#1f77b4">Reason kept: single-round agent forgot every observation</span>.**
- Text-only LLM (no pixel input to model). **<span style="color:#1f77b4">Reason kept: structured data is what the model can actually parse</span>.**

## What we tried and dropped

- VLM image input (v0, v1) — <span style="color:#d62728">replaced after 0% extraction benchmark</span>.
- `/think` mode in production (det_goal v1) — <span style="color:#d62728">chain never closed in long prompts</span>.
- click_targets bandit standalone — <span style="color:#d62728">0 / 5 production hit rate in cross-validation</span>.
- action_semantics from LLM standalone — <span style="color:#d62728">+0pp on its own in the ablation</span>.
- Hard rules R1/R4/R5/R6/R7 in V4 baseline — <span style="color:#d62728">+0pp on their own when proposer is the carrier</span>.

## Reading order for new collaborators

1. `presentation/report.md` — current state, why no BFS/DFS, RL plan, results.
2. `presentation/arc.md` — this file.
3. The repository README at the top level.
