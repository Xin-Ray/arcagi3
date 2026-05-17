# 2026-05-17 v0 subtask T-SEL-1 — Verification Report

> **作用**: 验证 SmolLM3-3B CoT 在 "ACTION6 click 击中目标 bbox" 子任务上的准确率。
> **状态**: 完成,**FAIL** (acc = 78.0%,目标 ≥ 90%,差 12pp)。
> **决策**: 接近通过,**最有希望的 borderline FAIL** —— prompt 工程或 1 次 few-shot 可能足以推过。

## 0. 原始数据 (outputs/ 引用)

| 文件 | 说明 |
|---|---|
| `outputs/subtask_T-SEL-1_20260517-163018/metrics.json` | 汇总: 78/100, total 3235.5s (~54 min) |
| `outputs/subtask_T-SEL-1_20260517-163018/per_probe_smollm3-cot.jsonl` | 100 条 per-probe |
| `outputs/subtask_T-SEL-1_20260517-163018/summary.md` | 表格 |
| `outputs/subtask_batch_20260517-163018/{metrics.json,summary.md}` | 跨 subtask 汇总 |
| `outputs/bench_subtask_batch_v1_think.log` | batch 日志 |

## 1. 任务定义

来源: `arc_agent/subtask_probes/__init__.py:gen_T_SEL_1`。

**Probe 形态**:
- 4 个不重叠 bbox (Object A/B/C/D),坐标 `rows r1-r2, cols c1-c2`。
- 题目: "Which ACTION6 selects Object X?"
- 4 选 1: 每个选项形如 `ACTION6 x=N y=M`,只有 1 个在 target bbox 内。

**目标**: ≥ 90%。

## 2. 实验配置

| 项 | 值 |
|---|---|
| 模型 | SmolLM3-3B |
| reasoning_mode | `cot` (with `/think`) |
| max_new_tokens | 1024 |
| n_probes | 100, seed=42 |

## 3. 结果

**Accuracy: 78.0% (78/100)** — **FAIL** (-12pp)。

| 子集 | n | accuracy |
|---|---:|---:|
| CoT activated (elapsed > 10s) | **86** | 79.1% |
| No CoT (elapsed ≤ 10s) | 14 | 71.4% |

**86% probe 激活 CoT**(total 54min,平均 ~32s/probe)。即便如此,只有 79% 准确。

## 4. 错误分析

22 个错误的成因(从 per_probe.jsonl raw_tail 抽样):
- **bbox 边界混淆**: target 是 "rows 12-18, cols 5-9",选项 x=10 y=15 (col=10 越界一格,但模型可能误以为 "rows" 含义)。
- **(x, y) vs (row, col) 顺序混淆**: prompt 明说 "ACTION6 takes coords (x=col, y=row)",但模型有时把 x 当 row。
- **off-by-one**: target 是 cols 5-9,选项 x=4 (差 1),模型误判。

## 5. 通过/失败判定

**FAIL**(-12pp)。但是:
1. CoT 激活率高 (86/100) → prompt 工程不是主因
2. 错误集中在坐标约定 (x/y vs row/col) 混淆,**修 prompt 措辞可解决**
3. 是 5 个 subtask 中 **唯一靠近 90% 的 FAIL** —— investment 性价比最高

## 6. 跟进项

| 优先级 | 行动 | 验证目标 |
|---|---|---|
| P0 | 把题面的 `(x, y)` 表达统一成 `(col, row)` 或直接 `(row, col)`,在选项里也用 `row=N col=M` | 消除约定混淆,target ≥ 88% |
| P0 | 选项格式从 `x=N y=M` 改为 `row=R col=C`,跟 prompt 一致 | 同上 |
| P1 | Few-shot 1 example | bbox 含义示范,target ≥ 90% |
| P2 | 检查 ar25 实际 game 用什么坐标系 (`env.step(ACTION6, x=, y=)` 的 x 是 col 还是 row?) | 跟实际 API 对齐,避免离线高分但上线翻车 |

## 7. 实践含义

T-SEL-1 是 ACTION6 click 的核心子任务。**78% 在多次重试场景下 still usable**(R7 + click_targets bandit 会过滤反复失败的坐标),但**单次 select 准确率不够**。

实际游戏:
- click_targets bandit 用 frame change 反馈 → 错 click 也能挽回
- 但 78% 单 step 等于每 4 次 click 1 次浪费,**在 80-step budget 下显著拖累**

## 文件清单

```
docs/project/2026-05-17-v0-subtask-T-SEL-1/
└── report.md                                          (本文件)

outputs/subtask_T-SEL-1_20260517-163018/
├── metrics.json
├── per_probe_smollm3-cot.jsonl
└── summary.md

outputs/subtask_batch_20260517-163018/
└── metrics.json, summary.md

arc_agent/subtask_probes/__init__.py                   (gen_T_SEL_1)
```

## 引用

- 主架构: `docs/project/2026-05-17-v0-subtask_decomp/architecture.md`
- 同批: T-NAV-1 (52% FAIL), T-NAV-2 (95% PASS), T-NAV-3 (67% FAIL), T-GOAL (30% FAIL)
- 全局词汇: `docs/GLOSSARY_zh.md`
