# 2026-05-19 v4 clean_baseline — Architecture

> **作用**: 把所有 **已验证有效** 的模块拼成最小 baseline,把所有 **未验证 / 已知有问题** 的模块默认 off,然后 ablation 单独验证每个旧模块的实际贡献。
> **触发**: 2026-05-18 chat,用户指出 "你这样混着架构，那不出问题才怪",要求清洗 + ablation。
> **设计原则**: 每个保留的模块都必须有 publish 过的 PASS evidence;每个砍掉的模块要有可复现的 FAIL/regression evidence。

## 1. 保留模块 (validated → keep)

| 模块 | 验证证据 | 来源 |
|---|---|---|
| **scipy perception** | scipy 100% vs Qwen-VL 0% per-frame extract | `docs/project/2026-05-11-v3-baseline/`、`ref_object_pipeline_zh.md` |
| **`arc_agent/goal_evaluator.py`** (parser) | T-GOAL acc 83% vs LLM 68%,recall on TRUE 100% vs 22%,1.25M× 更快 | `docs/project/2026-05-18-v0-goal_judge_ab/report.md` |
| **Parser v1 vocab 扩展** (edge / center / align_any / reach) | 100% 覆盖 v2 round 0 + smoke 3 的所有 production hypothesis | 同上 + smoke 3 commit `12fd46c` |
| **Force-reject 机制** (orchestrator 清 hypothesis + push rejected_goals) | smoke 3 验证: Reflection 真的被迫改写 hypothesis | `docs/project/2026-05-18-v0-det_goal_v3_extended_parser/report.md` §2.4 |
| **Hallucination 检测** (parser parse OK 但 verdict=None → reject) | smoke 2-3 验证: 18/20 trigger,清掉 "yellow+red" 这种 frame 不存在的 hypothesis | 同上 |
| **force_cot Action ASK 措辞** | T-NAV-3 bench 67%→97% (+30pp PASS) | `docs/project/2026-05-18-v0-force_cot/report.md` |
| **Reflection 截断 fix** (`--max-new-tokens-reflection` 配置) | v1 (250 token) → v2 (2048 token) 把 Reflection JSON 输出从 0/100 → 100/100 | `docs/project/2026-05-18-v0-det_goal_plus_force_cot/report.md` §3 |
| **R3 forced-explore safety net** (orchestrator-level) | 在多个 smoke 防 ACTION1 spam 死锁,默认 ON | (CLAUDE.md gotchas) |
| **NEW: Reflection schema 验证 (wide)** | hypothesis_update 必须能被 `parse_goal_hypothesis` 解析成 GoalPredicate(任何 kind 都行,包括 align_any);parse 失败 → 立即拒收 | 本 project 引入 |

## 2. 砍掉模块 (unvalidated / known-bad → off,留 CLI flag 可 ablate)

| 模块 | 问题证据 | CLI 关 |
|---|---|---|
| **`action_proposer` K=3 candidates** | smoke 3-4 raw 显示 Action 把 "known-good ACTION1" 标签当 prior,无视 goal direction → 65% ACTION1 bias | `--propose off` |
| **`click_targets` bandit** | 交叉验证 0/5 命中(production ACTION6 全 no-op) | `--click-targets off` (NEW CLI) |
| **`action_semantics` propagation** | Reflection 写正向 "ACTION1 moves UP" → Action 当 known-good → 死循环 ACTION1 | `--action-semantics-from-llm off` (NEW CLI) |
| **R2 mask** | smoke 3 没用上;之前 86% change_rate 但 0 通关 | `--mask off` |
| **R1/R4/R5/R6/R7 hard rules** | 没单独 ablate 过,默认 off 进 baseline | `--hard-rules off` (NEW CLI,batch) |
| **8-block prompt 冗余 (LOW-PRIORITY / TEXTURE / etc)** | 让 prompt 过长 → /think 死循环 | 自动跟 `--propose off` + 其他关 |

## 3. CLI flag 一览 (V4 baseline 命令)

```powershell
.venv\Scripts\python.exe scripts\run_v3_multi_round.py `
  --game ar25 `
  --max-actions-total 200 `              # NEW: total step budget across rounds
  --max-actions-per-round 100 `          # cap per single round
  --max-rounds 5 `                       # safety cap
  --seed 42 `
  --mask off `
  --propose off `
  --click-targets off `                  # NEW
  --action-semantics-from-llm off `      # NEW
  --hard-rules off `                     # NEW (umbrella for R1/R4/R5/R6/R7)
  --validate-hypothesis-schema wide `    # NEW
  --backbone HuggingFaceTB/SmolLM3-3B `
  --reasoning-mode {tbd by Phase 1} `
  --max-new-tokens-action {tbd} `
  --max-new-tokens-reflection 2048 `
  --tag v4_baseline_ar25
```

## 4. Step-budget pooling 机制 (NEW)

旧逻辑: `--rounds N --max-actions M` 跑 N 轮独立,每轮上限 M 步,**早终止浪费预算**(round 1 在 step 70 GAME_OVER → round 2 才开始,但 step budget 不复用)。

新逻辑: `--max-actions-total T` 是总预算。while `total_used < T` and `rounds < max_rounds`:
- 起一轮,本轮 step 上限 = `min(--max-actions-per-round, T - total_used)`
- 轮结束(WIN / GAME_OVER / 到 per-round cap) → 累计 `total_used += round_steps_actually_used`
- 继续下一轮直到 `total_used == T` 或 `rounds == max_rounds`

跟 user 2026-05-19 提的 "70 步停了但 200 还没用完,继续 round 2 直到 200" 一致。

## 5. Phase 时间表 (overnight)

| Phase | 内容 | wall (估) | 关键产出 |
|---|---|---|---|
| **0** | 文档 + CLI 实现 + 单测 | 30 min | 本文档 + decision_tree.md + ablation_plan.md + code |
| **1** | Validation A: /think vs /no_think 各 1×10 step on V4 minimal | 20-40 min | `phase1_thinkvsnothink.md` |
| **2** | Validation B: V4 minimal full ar25 1 round × 200 step (budget pooling) | 30-90 min | `phase2_v4_baseline.md` |
| **3** | Validation C: Ablation 4 个模块单加,各 ar25 1 round × 100 step | 2-4 小时 | `phase3_ablation.md` |
| **4** | (条件) 5-game G_base × 1 round × 200 step,one at a time | 2-3 小时 | `phase4_5game_eval.md` |
| **5** | 最终报告 + 决策 | 10 min | `report.md` (project root) |

## 6. 决策树 (autonomously executed)

| Phase 1 结果 | 下一步 |
|---|---|
| /think chain 在 V4 短 prompt 下 ≤ 1024 token 闭合 + reasoning 非空 | Phase 2 用 /think + 1024 tokens |
| /think 仍 ramble > 1024 tokens | 试 2048 一次;仍不行 → 用 /no_think |
| /no_think reasoning 可见 + action 多样化 (非全 ACTION1) | Phase 2 用 /no_think |
| 两种都坏 → 文档报告,**停止**,等用户 |

| Phase 2 结果 | 下一步 |
|---|---|
| ≥ 1 round 通关 ≥ 1 level | **大胜**,直接 Phase 4(5 game),跳过部分 Phase 3 |
| change_rate ≥ 60% + Force-reject 触发 + hypothesis 改写 ≥ 5 次 + 0 wins | Phase 3 ablation 看哪个旧模块能补 |
| change_rate < 60% (退化) | 报告,停止 |

| Phase 3 单 ablation 结果 | 下一步 |
|---|---|
| 加这个模块 win 率 > V4 baseline | 标"有用",留下来 |
| 加这个模块 change_rate / hypothesis 质量等价或更差 | 标"无用或有害",**继续禁用** |

| Phase 4 5-game 总结 | 下一步 |
|---|---|
| ≥ 1 game 通关 ≥ 1 level | **训练候选**,等用户 morning 决定要不要 fine-tune |
| 全 0 通关但 change_rate ≥ SmolLM3 5×2×300 (64%) | 报告 + 等用户 |
| change_rate 退化 | 报告 + 停止 + 标"validated subset 不够" |

## 7. 安全约束 (user 睡时)

- **不 push 任何分支**(用户没明确授权 v4 分支 push)
- **不动 main**
- **不删任何文件**
- 任何 Python 异常 → 保 partial + 报告 + 停止
- GPU OOM → 减小 token budget,标"未跑完"
- 并发: 每个 phase **串行**,不抢 GPU

## 8. 文件清单 (本 project 预期产出)

```
docs/project/2026-05-19-v0-v4_clean_baseline/
├── architecture.md                     (本文件)
├── decision_tree.md                    (Phase 1/2/3/4 决策图)
├── ablation_plan.md                    (Phase 3 详细 spec)
├── phase1_thinkvsnothink.md            (Phase 1 报告)
├── phase2_v4_baseline.md               (Phase 2 报告)
├── phase3_ablation.md                  (Phase 3 报告)
├── phase4_5game_eval.md                (Phase 4 报告,可选)
└── report.md                           (最终汇总)

outputs/v4_*                            (per-phase outputs)
```

## 9. 引用

- 上游决定 (用户提的清洗 + ablation): 2026-05-18 chat
- 5 subtask 拆分: `docs/project/2026-05-17-v0-subtask_decomp/`
- force_cot A/B: `docs/project/2026-05-18-v0-force_cot/`
- goal_judge A/B: `docs/project/2026-05-18-v0-goal_judge_ab/`
- det_goal v1/v2/v3 实验链: `docs/project/2026-05-18-v0-det_goal_plus_force_cot/`、`docs/project/2026-05-18-v0-det_goal_v3_extended_parser/`
- 全局词汇: `docs/GLOSSARY_zh.md`
- 仓库总入口: `docs/README.md`
