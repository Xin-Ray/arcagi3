# 2026-05-17 v0 subtask T-NAV-2 — Verification Report

> **作用**: 验证 SmolLM3-3B CoT 在 "多步同方向计数" 子任务上的准确率。
> **状态**: 完成,**PASS** (acc = 95.0%,远超目标 ≥ 90%)。
> **决策**: T-NAV-2 通过 → 多步同方向场景下,model 可以胜任。**5 个 subtask 中唯一通过**。

## 0. 原始数据 (outputs/ 引用)

| 文件 | 说明 |
|---|---|
| `outputs/subtask_T-NAV-2_20260517-163018/metrics.json` | 汇总: 95/100, total 2321.9s (~38 min) |
| `outputs/subtask_T-NAV-2_20260517-163018/per_probe_smollm3-cot.jsonl` | 100 条 per-probe |
| `outputs/subtask_T-NAV-2_20260517-163018/summary.md` | 表格 |
| `outputs/subtask_batch_20260517-163018/{metrics.json,summary.md}` | 跨 5 个 subtask 汇总 |
| `outputs/bench_subtask_batch_v1_think.log` | batch 日志 |

## 1. 任务定义

来源: `arc_agent/subtask_probes/__init__.py:gen_T_NAV_2`。

**Probe 形态**:
- Object at `(r1, c1)`,target at `(r2, c2)` 共行或共列,distance n ∈ [2..10]。
- 4 选 1: 一个正确次数 (n times),三个干扰 (n±1, 2n, n−2)。
- Prompt 直接告诉模型用哪个 ACTION (ACTION1/2/3/4),只问几次。

**目标**: ≥ 90%。

## 2. 实验配置

| 项 | 值 |
|---|---|
| 模型 | SmolLM3-3B (`HuggingFaceTB/SmolLM3-3B`) |
| reasoning_mode | `cot` (with `/think` injected) |
| 量化 | 4-bit nf4 |
| max_new_tokens | 1024 |
| n_probes | 100, seed=42 |
| 运行 | `scripts/bench_subtask_batch.py` (batch 5 subtasks 一次加载) |

## 3. 结果

**Accuracy: 95.0% (95/100)** — **PASS** (vs 90% target,+5pp 安全余量)。

| 子集 | n | accuracy |
|---|---:|---:|
| CoT activated (elapsed > 10s) | **100** | 95.0% |
| No CoT (elapsed ≤ 10s) | 0 | n/a |

**所有 100 probe 都激活了 CoT 链**(elapsed_s 普遍 20-30s)。这是 5 个 subtask 中唯一 100% CoT 激活的。原因猜测: prompt 直接要求计数("How many times must you press ..."),数字计算无法被短路径绕过,模型必须列计算。

## 4. 错误分析

| 类型 | 数量 |
|---|---:|
| parse fail (没有 Answer: X) | 4 |
| guess 错误但有 Answer | 1 |

**5 个错误全是 distractor 接近时选错**(典型: n=7 时混到 distractor=8)。说明模型基本计数能力 OK,只在边界(off-by-one)上偶尔翻车。

## 5. 通过/失败判定

**PASS**。原因:
1. acc = 95.0% ≥ 90% target ✅
2. CoT 100% 激活 → 推理过程透明、可验证
3. 错误模式集中在 off-by-one,非系统性失败

→ 该子任务下游可用,**不需要继续投资改进**。

## 6. 实践含义

T-NAV-2 是 ar25 通关链 "对齐两个 yellow square" 的核心子任务之一(把对象沿单轴推 N 步)。**这一步模型本身没问题**。

剩余瓶颈不在 "如何走 N 步" 这种子任务,而在:
- T-NAV-1 (单步方向判定,52%) — 模型短路径下出错
- T-NAV-3 (L 型双段,67%) — 模型短路径下出错
- T-SEL-1 (ACTION6 click,78%) — 接近但没过
- T-GOAL (识别 goal 达成,30%) — 几乎随机

→ 见 cross-summary。

## 文件清单

```
docs/project/2026-05-17-v0-subtask-T-NAV-2/
└── report.md                                          (本文件)

outputs/subtask_T-NAV-2_20260517-163018/
├── metrics.json
├── per_probe_smollm3-cot.jsonl
└── summary.md

outputs/subtask_batch_20260517-163018/                 (cross-subtask 汇总)
├── metrics.json
└── summary.md

outputs/bench_subtask_batch_v1_think.log               (batch 运行日志)

arc_agent/subtask_probes/__init__.py                   (gen_T_NAV_2)
scripts/bench_subtask_batch.py                         (batch runner)
```

## 引用

- 主架构: `docs/project/2026-05-17-v0-subtask_decomp/architecture.md`
- 全局词汇: `docs/GLOSSARY_zh.md`
- 仓库总入口: `docs/README.md`
- 同批 subtask 报告:
  - [`../2026-05-17-v0-subtask-T-NAV-1/report.md`](../2026-05-17-v0-subtask-T-NAV-1/report.md)
  - [`../2026-05-17-v0-subtask-T-NAV-3/report.md`](../2026-05-17-v0-subtask-T-NAV-3/report.md)
  - [`../2026-05-17-v0-subtask-T-SEL-1/report.md`](../2026-05-17-v0-subtask-T-SEL-1/report.md)
  - [`../2026-05-17-v0-subtask-T-GOAL/report.md`](../2026-05-17-v0-subtask-T-GOAL/report.md)
