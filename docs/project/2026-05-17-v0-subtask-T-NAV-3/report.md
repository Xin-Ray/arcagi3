# 2026-05-17 v0 subtask T-NAV-3 — Verification Report

> **作用**: 验证 SmolLM3-3B CoT 在 "L 型双段路径规划" 子任务上的准确率。
> **状态**: 完成,**FAIL** (acc = 67.0%,目标 ≥ 90%)。
> **决策**: T-NAV-3 不通过 → 多段路径规划在不激活 CoT 时不可靠;需 prompt 强制激活。

## 0. 原始数据 (outputs/ 引用)

| 文件 | 说明 |
|---|---|
| `outputs/subtask_T-NAV-3_20260517-163018/metrics.json` | 汇总: 67/100, total 163.3s |
| `outputs/subtask_T-NAV-3_20260517-163018/per_probe_smollm3-cot.jsonl` | 100 条 per-probe |
| `outputs/subtask_T-NAV-3_20260517-163018/summary.md` | 表格 |
| `outputs/subtask_batch_20260517-163018/{metrics.json,summary.md}` | 跨 subtask 汇总 |
| `outputs/bench_subtask_batch_v1_think.log` | batch 日志 |

## 1. 任务定义

来源: `arc_agent/subtask_probes/__init__.py:gen_T_NAV_3`。

**Probe 形态**:
- Object at `(r1, c1)`,target at `(r2, c2)`,**`Δrow ≠ 0` 且 `Δcol ≠ 0`** (must move both axes)。
- 4 选 1:
  - "n_vert ACTION_X only" (错: 漏 col)
  - "n_horiz ACTION_Y only" (错: 漏 row)
  - "n_vert ACTION_X + n_horiz ACTION_Y (n_total)" ✅
  - "(n_vert+1) ACTION_X + n_horiz ACTION_Y (off-by-one)" (错)

**目标**: ≥ 90%。

## 2. 实验配置

| 项 | 值 |
|---|---|
| 模型 | SmolLM3-3B |
| reasoning_mode | `cot` (with `/think`) |
| max_new_tokens | 1024 |
| n_probes | 100, seed=42 |

## 3. 结果

**Accuracy: 67.0% (67/100)** — **FAIL** (-23pp)。

| 子集 | n | accuracy |
|---|---:|---:|
| CoT activated (elapsed > 10s) | **1** | 0% (单样本不可靠) |
| No CoT (elapsed ≤ 10s) | 99 | **67.7%** |

**CoT 几乎完全没激活**(1/100,total 才 163s ≈ 1.6s/probe)。模型一看 prompt "Which is the minimum action plan?" 加 4 个文本选项,直接给字母答案,没去算两轴 delta。

## 4. 错误分析

短路径 99 个里答对 67 个:
- 比随机 25% 高得多 → 模型有 **"两轴都要动" 这个常识** 的先验
- 但分不清正确的(n_vert, n_horiz)组合 vs off-by-one distractor

T-NAV-1 vs T-NAV-3 短路径 acc 对照:
- T-NAV-1 短路径 47.2%(4 选 1 单轴方向,几乎随机)
- T-NAV-3 短路径 67.7%(4 选 1 双轴 plan,因为干扰选项里有 2 个明显错的 "只走一轴",模型能排除)

→ T-NAV-3 的 67% 不是真的"会推理",是**靠排除明显错选项拿分**;触及真正难点 (n_vert vs n_vert+1 off-by-one) 就翻车。

## 5. 通过/失败判定

**FAIL**。原因:
1. acc = 67% < 90% target (-23pp)
2. CoT 激活率 1/100,无推理证据
3. 即使排除明显错项后,off-by-one 干扰仍能骗到 33%

## 6. 跟进项

| 优先级 | 行动 | 验证目标 |
|---|---|---|
| P0 | user prompt 前缀 "Let's solve step by step. Compute Δrow and Δcol explicitly." | 看 CoT 激活率能否 ≥ 80%,目标 acc ≥ 85% |
| P0 | 同 fix 应用到 T-NAV-1, T-GOAL,看 CoT 激活率响应 | 跨 subtask 一致性 |
| P1 | 若仍 < 90%,升级模型测 base rate | - |

## 7. 实践含义

T-NAV-3 在 ar25 通关链 "把物体推到对角位置" 时有用。**当前 SmolLM3 短路径下 67% 准确率,落实到 N 步 plan 容易在中途出错累积成 0%**。

实际游戏中:
- propose 提多个候选 → reflection 看 outcome 决定
- 即使 LLM 选错 plan,outcome_log 上 "no_op 5 步" 会触发 R3 → forced explore
- 所以 67% 不像 model_bench 100% T1 那样直接致命,但**多段策略很难持续做对**。

## 文件清单

```
docs/project/2026-05-17-v0-subtask-T-NAV-3/
└── report.md                                          (本文件)

outputs/subtask_T-NAV-3_20260517-163018/
├── metrics.json
├── per_probe_smollm3-cot.jsonl
└── summary.md

outputs/subtask_batch_20260517-163018/                 (跨 subtask 汇总)
└── metrics.json, summary.md

outputs/bench_subtask_batch_v1_think.log               (batch 日志)

arc_agent/subtask_probes/__init__.py                   (gen_T_NAV_3)
```

## 引用

- 主架构: `docs/project/2026-05-17-v0-subtask_decomp/architecture.md`
- 同批: T-NAV-1 (52% FAIL), T-NAV-2 (95% PASS), T-SEL-1 (78% FAIL), T-GOAL (30% FAIL)
- 全局词汇: `docs/GLOSSARY_zh.md`
- 仓库总入口: `docs/README.md`
