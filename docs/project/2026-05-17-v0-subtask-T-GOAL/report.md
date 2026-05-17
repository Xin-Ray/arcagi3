# 2026-05-17 v0 subtask T-GOAL — Verification Report

> **作用**: 验证 SmolLM3-3B 在 "判断目标是否达成 (YES/NO)" 子任务上的准确率。
> **状态**: 完成,**严重 FAIL** (acc = 30.0%,**低于 25% 随机 baseline 上一档**)。
> **决策**: T-GOAL 完全不可用 → 这是最关键的瓶颈,goal recognition 是通关闭环的核心。

## 0. 原始数据 (outputs/ 引用)

| 文件 | 说明 |
|---|---|
| `outputs/subtask_T-GOAL_20260517-163018/metrics.json` | 汇总: 30/100, total 168.6s |
| `outputs/subtask_T-GOAL_20260517-163018/per_probe_smollm3-cot.jsonl` | 100 条 per-probe |
| `outputs/subtask_T-GOAL_20260517-163018/summary.md` | 表格 |
| `outputs/subtask_batch_20260517-163018/{metrics.json,summary.md}` | 跨 subtask 汇总 |
| `outputs/bench_subtask_batch_v1_think.log` | batch 日志 |

## 1. 任务定义

来源: `arc_agent/subtask_probes/__init__.py:gen_T_GOAL`。

**Probe 形态**:
- Goal 文本: `"align the two yellow squares vertically in the left column (col=0)"`
- 给 `obj_A (yellow): row=R1, col=C1` + `obj_B (yellow): row=R2, col=C2`
- 25% probe 是 success state (双方 col=0 且 row 不等),75% 是 failure
- 4 选 1:
  - "YES — both yellow objects are in the left column and vertically arranged"
  - "NO — at least one object is not in the left column"
  - "NO — objects are on the same row, not vertically arranged"
  - "NO — objects are in different columns"

**目标**: ≥ 90% (这是 reflection agent 的输入,精度必须高才能 trigger win)。

## 2. 实验配置

| 项 | 值 |
|---|---|
| 模型 | SmolLM3-3B |
| reasoning_mode | `cot` (with `/think`) |
| max_new_tokens | 1024 |
| n_probes | 100, seed=42 (25 success / 75 failure expected from `[True, False, False, False]` 概率) |

## 3. 结果

**Accuracy: 30.0% (30/100)** — **CATASTROPHIC FAIL** (-60pp)。

| 子集 | n | accuracy |
|---|---:|---:|
| CoT activated (elapsed > 10s) | **0** | n/a |
| No CoT (elapsed ≤ 10s) | 100 | **30.0%** |

**0/100 激活 CoT**(total 168.6s ≈ 1.7s/probe)。模型完全没思考,直接按 letter pattern 猜。

实际上 30% 比 25% 随机 baseline 略高,但**类间不平衡**:25 success / 75 failure 的话,**全部猜 NO** 就能拿 75% 的 failure 子集——模型的 30% 说明它甚至猜不对类别分布。

## 4. 错误分析

| 模式 | 描述 |
|---|---|
| 字母偏好 | 模型可能集中投到某个字母 (历史 raw_tail 显示 D 偏多) |
| 不读 col=0 含义 | "vertically" 这个词触发模型选 "vertically arranged" 而忽略 col 是否真的 = 0 |
| 4 选 1 干扰文本歧义 | 4 个选项都很长,模型短路径下挑视觉相似度最高的 |

## 5. 通过/失败判定

**FAIL — 最严重的一个 subtask**。

理由:
1. acc = 30% < 90% target,**差 60pp**
2. CoT 激活率 = 0,模型完全没尝试推理
3. **这是 goal recognition 任务,直接影响 reflection agent 判断 "是否已完成关卡"**
4. 在 production 上,这个子任务的失败可能就是 **0/5 game 通关** 的核心原因

## 6. 跟进项

| 优先级 | 行动 | 验证目标 |
|---|---|---|
| **P0(最高)** | user prompt 加 "First, write down obj_A position and obj_B position. Then check each condition." | 强制 CoT,target acc ≥ 70% |
| **P0** | 简化 4 个选项,只留 YES/NO 二选一(单独的 "原因" 用另一个 probe 测试)| 测试模型能不能区分 success vs failure |
| P0 | Few-shot 2 examples (1 success + 1 failure) | target acc ≥ 85% |
| P1 | 如果都不行,在 production 把 goal-check 改成 deterministic Python(`obj_A.col == 0 and obj_B.col == 0 and obj_A.row != obj_B.row`),LLM 不参与 | 100% precise |

## 7. 实践含义 (关键)

**T-GOAL 失败是 5 game × 2 round × 300 step 0 通关的强候选根因。**

- production 中 reflection agent 每步看 outcome,**它也在做 T-GOAL** —— 没 "goal achieved" 的信号,reflection 没法触发 win
- 即使 action_proposer 选对了 ACTION → object 真的对齐到 col=0,reflection agent 看不出来,继续推下一个 action,可能把对齐状态搞砸了

**Production fix 思路**: 把 goal-check 从 LLM 移到 deterministic Python 代码(读 `ObjectMemory` + Knowledge.goal_hypothesis,做 string match + 坐标计算)。这样不依赖模型推理。

## 文件清单

```
docs/project/2026-05-17-v0-subtask-T-GOAL/
└── report.md                                          (本文件)

outputs/subtask_T-GOAL_20260517-163018/
├── metrics.json
├── per_probe_smollm3-cot.jsonl
└── summary.md

outputs/subtask_batch_20260517-163018/
└── metrics.json, summary.md

arc_agent/subtask_probes/__init__.py                   (gen_T_GOAL)
```

## 引用

- 主架构: `docs/project/2026-05-17-v0-subtask_decomp/architecture.md`
- 同批: T-NAV-1 52%, T-NAV-2 **95% PASS**, T-NAV-3 67%, T-SEL-1 78%
- model_bench 5×2×300 报告: `docs/project/2026-05-17-v0-model_bench/report_5game.md` (0/5 通关现象)
- 全局词汇: `docs/GLOSSARY_zh.md`
