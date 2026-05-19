# 2026-05-19 v4 clean_baseline — Final Report

> **5 phase overnight pipeline**: Phase 0 infra → Phase 1 reasoning_mode → Phase 2 baseline → Phase 3 ablation → Phase 4 5-game eval。
> **结果**: **+18pp change_rate** over SmolLM3 5×2×300 baseline (82% vs 64%),5 game G_base **0 wins**。
> **状态**: 完成全 5 phase。pipeline 没遇到 stop 条件。
> **建议**: 保守。**不立即训练**(0 通关是 hard gate);先 trace 分析 + prompt 调整看能不能拿到 ≥1 win。

## 1. 一句话总结

把 v3.2 主线的 4 个旧模块(click_targets / action_proposer / action_semantics / hard_rules)逐个 ablate,**唯一关键的是 `action_proposer`**;V4 + `action_proposer` + 新增 goal_evaluator 反思闭环 = **change_rate 82%(超 SmolLM3 5×2×300 +18pp)**,但仍 0/5 通关。

## 2. 5-phase 全表

| Phase | 内容 | 结果 |
|---|---|---|
| **0** | CLI flags + step pooling + docs | ✅ commit `57d5d5a` `eaeb89b` |
| **1** | /think vs /no_think on V4 minimal | ✅ `/no_think` wins (`/think` chain 0/10 闭合) (commit `823e7cc`, report `017e2f5`)|
| **2** | V4 minimal ar25 1×200 budget-pooled | ❌ change_rate 11% 大退化 → 反向 Phase 3 (report `f4758f0`) |
| **3** | Ablation 4 个旧模块单独加 | ✅ `action_proposer` +75pp,其它 +0pp (report `d114b1f`) |
| **4** | V4+propose × 5 game one-at-a-time × 1×200 | ✅ mean change_rate 82%,**0 wins** |

## 3. 跨实验对比 (核心数字)

| Run | reasoning | proposer | click | semantics | hard_r | goal_eval | mean change_rate | wins |
|---|---|:-:|:-:|:-:|:-:|:-:|---:|---:|
| `main` e07e7d1 (v3.2 full) | /no_think | ✅ | ✅ | ✅ | ✅ | ❌ | 5-8% | 0/5 |
| SmolLM3 5×2×300 | /no_think | ✅ | ✅ | ✅ | ✅ | ❌ | **64%** | 0/5 |
| det_goal v3 ar25 1×100 | /think (chain trunc) | ✅ | ✅ | ✅ | ✅ | ✅ | 75% | 0/1 |
| **V4 + propose (本)** | /no_think | ✅ | ❌ | ❌ | ❌ | ✅ | **82%** | **0/5** |

## 4. 5 game (Phase 4) 单 game 表

| Game | change_rate | rounds | total steps | actions used |
|---|---:|---:|---:|---:|
| ar25 | 87% | 2 | 172 | 7 |
| bp35 | 85% | 2 | 72 | 4 |
| cd82 | 79% | 2 | 200 | 6 |
| cn04 | 74% | 2 | 150 | 6 |
| dc22 | 84% | 2 | 200 | 5 |
| **mean** | **82%** | 2.0 | 159 | 5.6 |

## 5. 已 validated 模块清单 (V4+propose 配置 = 最佳已知)

| 模块 | 来源 | Phase X 证据 |
|---|---|---|
| `scipy` perception | v3 baseline | Pre-v4 |
| `goal_evaluator` (parser) | det_goal v3 + v4 Phase 1 | T-GOAL 83%, +18pp over SmolLM3 |
| Force-reject (achieved/hallucinated) | det_goal smoke 3 | 5 game 测 hypothesis 主动 reject |
| Hallucination detect | det_goal smoke 2-3 | 同上 |
| Parser v3 vocab: edge/center/align_any/reach/match | smoke 3 + Phase 1B2 | Phase 1B 修复后 hypothesis 100% parseable |
| force_cot Action ASK 措辞 | force_cot A/B | (注: V4 用 /no_think 后此措辞效果中性,留着不退化) |
| Reflection 截断 fix (token budget 配置) | det_goal v2 → v3 | Phase 1B2 1024 token 足以让 Reflection 完整 JSON |
| `action_proposer` K=3 candidates | (v3.2 旧) | **Phase 3 ablation 唯一 +75pp** |
| Reflection schema 验证 (wide) | v4 Phase 0 NEW | Phase 1B 验证 dropped invalid hypothesis |
| Step-budget pooling | v4 Phase 0 NEW | Phase 4 ar25/cn04 show pooling 跨 round 工作 |

## 6. 已证伪 / 弃用模块

| 模块 | 证据 |
|---|---|
| `click_targets` bandit (standalone) | Phase 3 ablation +0pp;cross-validation 0/5 命中 |
| `action_semantics` LLM-written (standalone) | Phase 3 ablation +0pp;需 proposer 转候选才生效 |
| `hard_rules` (R1/R4/R5/R6/R7 batch) | Phase 3 ablation +0pp;R3 在 action_agent 仍 on(safety) |
| `mask` (R2) | 没在 v4 评估,保持 off |

## 7. 0/5 通关诊断

change_rate 82% 但 0/5 wins。change_rate 衡量"frame 是否变化",不衡量"朝目标方向变化"。诊断:

1. **goal_hypothesis 不可 chain**: hypothesis 是 "yellow to bottom edge" 类型,到了 bottom edge 也没 win → 说明 hypothesis ≠ 真实 win condition
2. **win condition 在 hypothesis space 外**: 某些 game 的 win 条件是"按 ACTION5 三次"或"stack 3 obj"或更复杂,parser 模式不覆盖
3. **K=3 candidates 没 goal-directed plan**: proposer 把 untried + known-good + click_target 推给 Action,**没显式根据 hypothesis 选方向 action**

## 8. 跟进 ROI 排序

| 优先级 | 建议 | 预期成本 |
|---|---|---|
| **P0** | Trace 分析 — 找 levels_gained > 0 的 step(如果有),分析临通关时 hypothesis + action | 1-2 小时 |
| **P0** | `--validate-hypothesis-schema strict` 重跑 — 强制 hypothesis 必须有明确 target,看 ACTION 是否更对齐 | 1.5 小时 (5 game) |
| **P1** | action_proposer 加 goal-directed candidate — 当 hypothesis = `move_to_row 63`,把对应 action (ACTION2 DOWN) 显式加入 K=3 | 半天 (代码 + 测) |
| **P1** | 升级 model 试 SmolLM3-7B / Qwen3-4B | 大半天 |
| **P2** | 加 `T-CHAIN` subtask probe — 测 model 在多步 plan 上能力 | 1 天 |

## 9. 是否训练?

按 decision_tree.md §"Phase 4":

| condition | 决策 |
|---|---|
| ≥ 2 game 通关 ≥ 1 level | 训练候选 |
| 1 game 通关 + 其余 change_rate ≥ 60% | 等用户 |
| 0 通关 + change_rate ≥ 64% baseline | **报告 + 等用户** |
| change_rate < 64% | 标"V4 minimal 不够",等用户 |

我们在 row 3 — **0 通关 + 82% change_rate**。**不自动训练**,等用户 morning 看完报告决定。

## 10. 文件清单

```
docs/project/2026-05-19-v0-v4_clean_baseline/
├── architecture.md                     (设计)
├── decision_tree.md                    (Phase 决策图)
├── ablation_plan.md                    (Phase 3 详细 spec)
├── phase1_thinkvsnothink.md            (/no_think 选定)
├── phase2_v4_baseline.md               (V4 minimal 退化)
├── phase3_ablation.md                  (action_proposer winner)
├── phase4_5game_eval.md                (5 game 82% mean)
└── report.md                           (本文件)

scripts/run_v3_multi_round.py           (Phase 0 加 6 个 CLI flag + step pooling)
arc_agent/goal_evaluator.py             (Phase 1B 加 "match" 模式)
tests/test_goal_evaluator.py            (33 tests pass)

outputs/v4_phase{1,2}_*                  (Phase 1/2 数据)
outputs/v4_ablate_*                      (Phase 3 ablation)
outputs/v4_phase4_g{1..5}_*              (Phase 4 5 game)
```

## 11. 引用

- 上游 (用户提议清洗 + ablation): 2026-05-18 chat
- 上游 (5 game one-at-a-time + step pooling + wide schema): 2026-05-19 chat
- det_goal 实验链: `docs/project/2026-05-18-v0-det_goal_plus_force_cot/` + 后续
- goal_judge A/B: `docs/project/2026-05-18-v0-goal_judge_ab/`
- 全局词汇: `docs/GLOSSARY_zh.md`
- 仓库总入口: `docs/README.md`
