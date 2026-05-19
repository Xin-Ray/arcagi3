# v4 Phase 2 — V4 baseline ar25 1×200 budget-pooled

> **作用**: Validation B — V4 minimal (砍光所有旧模块) 在 ar25 full run 上 work 吗?
> **结果**: **大退化** — change_rate 11% << SmolLM3 5×2×300 baseline 64%。**砍掉的某些旧模块 IS necessary**。
> **决策**: 按 decision_tree.md §"Phase 2" 进 Phase 3 ablation 反向看哪个模块必要。

## 0. 原始数据 (outputs/ 引用)

| 文件 | 内容 |
|---|---|
| `outputs/v4_phase2_baseline_s42.log` | 运行日志 |
| `outputs/v4_phase2_baseline_s42_20260519-025943/summary.json` | per-round metrics |
| `outputs/v4_phase2_baseline_s42_20260519-025943/round_{00,01}/` | trace + action_raw + reflection_raw + step PNG |

## 1. 数字

| 指标 | round 0 | round 1 | 总 |
|---|---:|---:|---:|
| steps | 100 | 100 | 200 |
| change_rate | **11%** | **11%** | 11% |
| levels gained | 0 | 0 | 0 |
| won | False | False | 0 |

| Action 分布 | 次数 |
|---|---:|
| ACTION1 | **188** (94%) |
| ACTION2 | 2 |
| ACTION3 | 2 |
| ACTION4 | 2 |
| ACTION5 | 2 |
| ACTION6 | 2 |
| ACTION7 | 2 |

| 闭环指标 | 总 (200 step) |
|---|---:|
| Parser kind 解析成功 | 2/200 (1%) |
| Parser verdict 非 None | 0/200 |
| GOAL CHECK alert | 2/200 |
| `rejected_goals` 最终 | 2 |

## 2. 跟其他 baseline 对比

| Run | reasoning_mode | 模块 | change_rate | wins |
|---|---|---|---:|---:|
| main e07e7d1 | /no_think | full v3.2 | 5-8% | 0 |
| SmolLM3 5×2×300 | /no_think | full v3.2 (含 action_proposer + click_targets + action_semantics) | **64%** | 0 |
| det_goal v1 | /think (truncated) | full v3.2 + goal_evaluator | 75% | 0 |
| det_goal v2 round 0 | /think (refl 2048) | full v3.2 + goal_evaluator | 47% | 0 |
| **v4 Phase 2** | **/no_think** | **V4 minimal (all old off)** | **11%** | **0** |
| Phase 1B2 (10 step) | /no_think | V4 minimal | 80% | 0 |

**关键反差**:
- v4 Phase 2 比 SmolLM3 5×2×300 退 53pp
- Phase 1B2 10 step 80% vs Phase 2 200 step 11% — 长程退化

## 3. 退化机制 (确诊)

ACTION1 占 94% 是真相。机制:
1. 初始 step: ar25 yellow + gray 在 frame 中,Reflection 写 hypothesis (但 parser parse 不通过的占 98%)
2. Action 看 `[STATE]` legal_actions 显示 ACTION1 在第 1 位 + 没有 `[ACTION stats]` (砍 hard_rules 后)+ 没有 `[CLICK TARGETS]` (砍 click_targets 后) + 没有 `[CANDIDATES]` (砍 propose 后)
3. Action default-bias 选 ACTION1
4. 前几步:object 真的 UP 1-3 cell,frame_changed → 看起来 work
5. 第 5-10 步:object 撞到顶 edge → ACTION1 no-op
6. 后续 190 步:Action 继续选 ACTION1(没 history 提示它 no-op),撞同一个状态死循环
7. R3 forced explore 偶尔抓到 ≥5 streak → 强换一个 untried action (ACTION2-7 各 2 次)
8. 但 R3 一旦换完 untried,余下时间都回 ACTION1

**问题**: 砍掉 `[ACTION stats]` (来自 outcome_log,被 `--hard-rules off` 间接禁) + 砍掉 `[CLICK TARGETS]` + 砍掉 `[CANDIDATES]` 等于 **拿走了 Action 唯一的 "我刚才做过什么" 信号**。

## 4. 决策

按 decision_tree.md `change_rate < 60% (退化)` → Phase 3 ablation **反向**看哪个旧模块必要。

Phase 3 顺序 (per ablation_plan.md):
1. V4 + click_targets — 验证它真的 0/5 无效
2. V4 + action_proposer — 验证它带 ACTION1 bias (但可能也带 ACTION 多样性)
3. V4 + action_semantics from LLM — 验证 known-good 是否帮 Action 多样化
4. V4 + hard_rules — 验证 R1/R4-R7 batch 的效果

每个 ar25 1×100 step,看 delta vs V4 baseline (11%)。

启动 Ablation 1 (`v4_ablate_click_s42`),任务 `b0nj5s0p0`,monitor `bdslkxdzc`。

## 5. 跟进考虑 (early findings)

如果 Phase 3 4 个 ablation 加起来仍达不到 SmolLM3 5×2×300 的 64%,可能需要:
- 同时加 2+ 模块的组合 ablation(超出 Phase 3 范围)
- 重新检查 V4 砍模块的选择(也许我误判了哪个有用)
- 升级 model(SmolLM3 7B?)

但先把 4 个单 ablation 跑完拿到清晰证据,再决定。

## 6. 文件清单

```
docs/project/2026-05-19-v0-v4_clean_baseline/
└── phase2_v4_baseline.md               (本文件)

outputs/v4_phase2_baseline_s42_20260519-025943/
├── summary.json
├── round_00/ (100 steps)
└── round_01/ (100 steps)

outputs/v4_phase2_baseline_s42.log
```
