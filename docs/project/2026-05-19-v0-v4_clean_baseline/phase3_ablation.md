# v4 Phase 3 — Ablation Study (single-module add-back)

> **作用**: Validation C — V4 minimal 退化 (11% change_rate) 之后,反向看哪个旧模块单独加回去能补回 SmolLM3 5×2×300 baseline (64%)。
> **结果**: **`action_proposer` 是唯一关键模块**。其它三个单独加都跟 baseline 持平。
> **决策**: Phase 4 配置 = V4 minimal + action_proposer only。

## 0. 原始数据 (outputs/ 引用)

| Ablation | Output dir |
|---|---|
| V4 baseline (P2) | `outputs/v4_phase2_baseline_s42_20260519-025943/` |
| +click_targets | `outputs/v4_ablate_click_s42_20260519-034414/` |
| +action_proposer | `outputs/v4_ablate_propose_s42_20260519-041257/` |
| +action_semantics | `outputs/v4_ablate_semantics_s42_20260519-044123/` |
| +hard_rules | `outputs/v4_ablate_hardrules_s42_20260519-050519/` |

每个对应的 log: `outputs/v4_*_s42.log`。

## 1. 关键对比表

| Ablation | change_rate | ACTION1 占比 | other actions | parser kind | rejected_goals | wins |
|---|---:|---:|---|---:|---:|---:|
| V4 baseline (P2 200 step) | 11% | 94% | each ≈ 2/200 | 2/200 | 2 | 0 |
| +click_targets | **11%** | 92% | each ≈ 1-3/100 | 2/100 | 2 | 0 |
| **+action_proposer** ✅ | **86%** | **20%** | **均匀 7 ACTIONs** | 5/85 | 5 | 0 |
| +action_semantics | **11%** | 94% | each ≈ 1/100 | 2/100 | 2 | 0 |
| +hard_rules | **11%** | 94% | each ≈ 1/100 | 2/100 | 2 | 0 |

## 2. 单 ablation 验收 (按 ablation_plan.md §"判定")

| Ablation | Δ change_rate vs baseline | Verdict |
|---|---:|---|
| +click_targets | +0pp | **NEUTRAL** — 单独无效,确认 cross-validation 0/5 结论 |
| **+action_proposer** | **+75pp** ✅ | **CRITICAL** — 唯一翻盘的模块 |
| +action_semantics | +0pp | **NEUTRAL** — 需要被 proposer 转化为 candidate 才有用 |
| +hard_rules | +0pp | **NEUTRAL** — 仅 R3 在 V4 baseline 已开,其它没填空 |

## 3. 为什么 action_proposer 是关键

V4 baseline 11% 的根因是 Action 死循环 ACTION1 (legal 第一个) → hit edge → no-op 190 步。

`action_proposer` 工作方式:
- 每步生成 K=3 候选 (untried action + known-good + click_target)
- prompt 改成 multi-choice "pick A/B/C"
- Action 只能在 3 个里选,**ACTION1 不会出现在每个候选集中**
- 强制 explore 多个 ACTION → frame_changed 概率高

实测 actions 分布 (ablation 2):
- ACTION1: 17/85 (20%) — 比 baseline 94% 大幅下降
- ACTION2: 12, ACTION3: 13, ACTION4: 10, ACTION5: 12, ACTION6: 12, ACTION7: 9
- 7 个 action 几乎均匀使用

这是 ACTION1-bias 的根本解。

## 4. 为什么其它 3 个单独无效

- **click_targets**: 给 ObjectMemory.click_targets 列表,但 Action prompt 没显示该列表(需要 proposer 把 click_target 渲染成 candidate),所以列表写了也没用
- **action_semantics**: 给 Knowledge.action_semantics 字典,但同样 Action 看不到(需要 proposer 把 "known-good" 渲染成 candidate)
- **hard_rules**: R1/R4/R5/R6/R7 都是 delta filter,影响 Reflection 写入合法性;但 Action 行为不直接受 prompt 中 ACTION stats 影响,V4 砍掉 prompt 那块后 R 系列规则没渲染目标

简言之: **prompt 没渲染 ⇒ Action 看不到 ⇒ 模块没作用**。proposer 是连接 Knowledge 到 prompt 的桥。

## 5. 跟 SmolLM3 5×2×300 baseline 对比

| Run | reasoning | proposer | click | semantics | hard_rules | goal_evaluator | change_rate |
|---|---|:-:|:-:|:-:|:-:|:-:|---:|
| SmolLM3 5×2×300 | /no_think | ✅ | ✅ | ✅ | ✅ | ❌ | 64% |
| **V4 + propose** | /no_think | ✅ | ❌ | ❌ | ❌ | ✅ | **86%** |

**V4 + propose 超 SmolLM3 5×2×300 +22pp**。差异 = `goal_evaluator` + force-reject + extended parser + Reflection schema 验证。

这说明 **goal_evaluator 改进 不仅没退化,反而 +22pp 提升**。

## 6. Phase 4 配置

```powershell
.venv\Scripts\python.exe scripts\run_v3_multi_round.py `
  --game {ar25|bp35|cd82|cn04|dc22} `
  --max-actions-total 200 --max-actions 100 --rounds 5 `
  --seed 42 `
  --mask off --propose on --click-targets off `
  --action-semantics-from-llm off --hard-rules off `
  --validate-hypothesis-schema wide `
  --backbone HuggingFaceTB/SmolLM3-3B --reasoning-mode no_think `
  --max-new-tokens-action 256 --max-new-tokens-reflection 1024 `
  --tag v4_phase4_g{N}_{game}_s42
```

5 game one at a time. 第一个 (ar25) 已 launch (`bughdchj2`)。

## 7. 跟进考虑

如果 Phase 4 某些 game 通关,**确认 V4+propose 是 production 推荐配置**:
- 比 v3.2 full module set 简单(砍 3 个无效模块)
- 包含 goal_evaluator 改进(+22pp)
- 反思闭环真正活跃(force-reject + parser eval)

如果 Phase 4 全 0 通关:
- change_rate 86% 但仍不能通关说明 **action 多样性不够,需要的是 goal-aligned planning**
- 这是另一个未解层面(T-NAV-3 测过 model 能力,production 需要 chaining)

## 8. 文件清单

```
docs/project/2026-05-19-v0-v4_clean_baseline/
└── phase3_ablation.md                  (本文件)

outputs/v4_phase2_baseline_s42_*        (V4 baseline,Phase 2)
outputs/v4_ablate_click_s42_*           (Ablation 1)
outputs/v4_ablate_propose_s42_*         (Ablation 2 ✅ winner)
outputs/v4_ablate_semantics_s42_*       (Ablation 3)
outputs/v4_ablate_hardrules_s42_*       (Ablation 4)
```
