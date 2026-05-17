# 2026-05-17 v0 subtask_decomp — Cross-Subtask Verification Report

> **本文档为 5 subtask 验证汇总**(T-NAV-1 / T-NAV-2 / T-NAV-3 / T-SEL-1 / T-GOAL)。
> 每个 subtask 单独 report 在 `docs/project/2026-05-17-v0-subtask-<NAME>/report.md`。
>
> **本日核心发现**: SmolLM3-3B 在 5 个原子子任务里 **只过 1 个**;其中 T-GOAL (goal 识别) 30%,几乎随机,**强烈怀疑这是 0/5 通关的根因**。

## 0. 原始数据 (outputs/ 引用)

| 文件 | 说明 |
|---|---|
| `outputs/subtask_batch_20260517-163018/metrics.json` | 跨 5 个 subtask 汇总,主入口 |
| `outputs/subtask_batch_20260517-163018/summary.md` | markdown 表格 |
| `outputs/subtask_T-NAV-1_20260517-163018/` | T-NAV-1 (52%) |
| `outputs/subtask_T-NAV-2_20260517-163018/` | T-NAV-2 (95% **PASS**) |
| `outputs/subtask_T-NAV-3_20260517-163018/` | T-NAV-3 (67%) |
| `outputs/subtask_T-SEL-1_20260517-163018/` | T-SEL-1 (78%) |
| `outputs/subtask_T-GOAL_20260517-163018/` | T-GOAL (30%) |
| `outputs/bench_subtask_batch_v1_think.log` | batch 运行日志 |
| `outputs/subtask_T-NAV-1_20260517-161834/` | T-NAV-1 v0 baseline (无 /think,48%) |
| `outputs/bench_subtask_T-NAV-1_v2.log` | v0 运行日志 |

## 1. 配置

| 项 | 值 |
|---|---|
| 模型 | SmolLM3-3B (`HuggingFaceTB/SmolLM3-3B`) |
| 量化 | 4-bit nf4 + bf16 compute |
| reasoning_mode | `cot` with `/think` system flag |
| max_new_tokens | 1024 |
| n_probes | 100 / subtask, seed=42 |
| 总 wall time | ~6,445s ≈ 1h47m,单次模型加载 + 5 subtask sequential |
| 运行 | `scripts/bench_subtask_batch.py` |

## 2. 结果一览

| Subtask | accuracy | n_correct/n | CoT activation | long_acc | short_acc | total(s) | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| **T-NAV-1** 单步方向 | 52.0% | 52/100 | 11/100 | 90.9% | 47.2% | 555 | ❌ FAIL |
| **T-NAV-2** 多步同方向计数 | **95.0%** | 95/100 | **100/100** | 95% | n/a | 2322 | ✅ **PASS** |
| **T-NAV-3** L 型双段 | 67.0% | 67/100 | 1/100 | (n=1) | 67.7% | 163 | ❌ FAIL |
| **T-SEL-1** ACTION6 click | 78.0% | 78/100 | 86/100 | 79.1% | 71.4% | 3235 | ❌ FAIL (borderline) |
| **T-GOAL** YES/NO 目标 | **30.0%** | 30/100 | **0/100** | n/a | 30.0% | 169 | ❌ SEVERE FAIL |

PASS 阈值: ≥ 90%。

**5 个 subtask 中 1 通 4 不通**。

## 3. 一句话总结 (放最上面)

**SmolLM3-3B-CoT 在 100% 激活 CoT 时表现极好 (95%);但是否激活 CoT 是模型自己决定,问题任务 (T-GOAL / T-NAV-1 / T-NAV-3) 激活率 0-11%,导致整体不可靠。**

## 4. 根因诊断

### 4.1 CoT activation 是 PASS/FAIL 的决定因素

把 5 个 subtask 按 CoT activation 率排序:

| Subtask | CoT 激活率 | 整体 acc |
|---|---:|---:|
| T-GOAL | **0%** | 30% |
| T-NAV-3 | 1% | 67% |
| T-NAV-1 | 11% | 52% |
| T-SEL-1 | 86% | 78% |
| T-NAV-2 | **100%** | 95% |

→ **Pearson r ≈ 0.9**(目测;activation 高 = acc 高,几乎单调)。

### 4.2 为什么 T-NAV-2 100% 激活?

prompt 末尾 `How many times must you press ACTION2 to reach the target?` —— 数字答案 forces 模型用 token 表达"算 8-3=5"这种过程。短路径下没法直接输出数字(模型没学会 zero-shot 立刻输出 "5 times" 而不是先列出推理)。

### 4.3 为什么 T-GOAL 0% 激活?

prompt 是 YES/NO + 4 个长 distractor 选项,**模型把它当 multiple choice 模式识别题**而非推理题,直接挑相似度高的选项 (letter pattern matching)。

### 4.4 `/think` system flag 没有解决 activation 问题

T-NAV-1 v0 (无 /think) vs v1 (有 /think):
- activation: 11/100 → 11/100 (没变)
- long-CoT acc: 54.5% → 90.9% (✅ 大涨,因 1024 tokens 不再截断 chain)
- 整体 acc: 48% → 52% (+4pp,因为大头没改)

`/think` 仅改善了 **已激活的** CoT 链;**没办法强制激活**。

## 5. 跟进项 (按 ROI 排)

### P0 — Force CoT activation across all subtasks

| 行动 | 命中的 subtask | 预期收益 |
|---|---|---|
| user prompt 前缀加 `"Let's solve this step by step. Show your reasoning."` | T-NAV-1, T-NAV-3, T-GOAL | 把 CoT 激活率拉到 ≥ 80%,根据 T-NAV-2 数据可期 acc ≥ 85% |
| 题目结尾改 "Show your work, then end with Answer: X" | 同上 | 同上 |
| 试 `tokenizer.apply_chat_template(messages, enable_thinking=True)` 如果存在 | 全 | 比 prompt hack 干净 |

### P1 — Production fix for T-GOAL (即使 CoT 起作用也可能不够)

把 goal-check 移到 deterministic Python:
```python
def goal_achieved(knowledge: Knowledge, obj_memory: ObjectMemory) -> bool:
    # 解析 knowledge.goal_hypothesis ("align two yellow squares vertically in left column")
    # 用 ObjectMemory 拿 obj_A.col, obj_B.col, obj_A.row, obj_B.row
    # 显式断言: col_A == 0 and col_B == 0 and row_A != row_B
    ...
```
LLM 写 goal 描述,但**判断**用 deterministic code。

### P2 — Refine T-SEL-1 prompt (78% → ≥ 90%)

把题面 `(x, y)` 和选项 `x=N y=M` 统一改成 `(row, col)` / `row=R col=C`,跟 prompt 一致。

### P3 — Stronger model 测 base rate

升级 SmolLM3-7B 或 Qwen2.5-7B,先跑 T-GOAL 100 probes 看 base activation 率是否更高(7B 通常 think 更稳定)。注意会拉慢 inference,要权衡。

## 6. 5 game × 2 round × 300 step (0 通关) 推论

[`../2026-05-17-v0-model_bench/report_5game.md`](../2026-05-17-v0-model_bench/report_5game.md) 显示 SmolLM3 mean change_rate 64% 但 0 levels won。**结合 subtask 数据**:

- change_rate 高: T-NAV-2 100% PASS + T-SEL-1 78%(够频繁 hit 有变化的 click)→ frame 经常变化
- 0 levels won: T-GOAL 30% → 即使 frame 变到 win state,reflection agent 也认不出来

**强假设**: T-GOAL 失败 = production 0 通关的核心瓶颈。

→ 验证方法: 跑一次 ar25,把 LLM 的 reflection 替换成 deterministic goal-check(读 hypothesis + obj 位置),看通关率是否飙升。

## 7. 分支状态

| 分支 | HEAD | 状态 |
|---|---|---|
| `feat-2026-05-17-v0-subtask-T-NAV-1` | `8f1d314` (`/think` fix + v0 report) | 验证完,等合 docs 或 deprecate |
| 其他 subtask 报告分支 | (本文件就在 T-NAV-1 分支上)  | 5 个 report 文件可一次性合到 main |
| `feat-2026-05-16-v0-action_proposer` | `b9e2e87` | 提供 batch runner 基础 |
| `main` | `e07e7d1` | 没动 |

→ **建议**: 全部 5 个 report + cross-summary 一次 commit 后,**最终用户决定是否合 main**(我不主动合)。

## 8. 文件清单

```
docs/project/
├── 2026-05-17-v0-subtask_decomp/
│   ├── architecture.md                                (7 subtask DAG 设计)
│   └── report.md                                      (本文件,cross-summary)
├── 2026-05-17-v0-subtask-T-NAV-1/report.md            (v0 48% / v1 52%)
├── 2026-05-17-v0-subtask-T-NAV-2/report.md            (95% PASS)
├── 2026-05-17-v0-subtask-T-NAV-3/report.md            (67%)
├── 2026-05-17-v0-subtask-T-SEL-1/report.md            (78%)
└── 2026-05-17-v0-subtask-T-GOAL/report.md             (30%)

outputs/
├── subtask_T-NAV-1_20260517-161834/                   (v0,无 /think)
├── subtask_T-NAV-1_20260517-163018/                   (v1,/think)
├── subtask_T-NAV-2_20260517-163018/
├── subtask_T-NAV-3_20260517-163018/
├── subtask_T-SEL-1_20260517-163018/
├── subtask_T-GOAL_20260517-163018/
├── subtask_batch_20260517-163018/                     (跨 subtask 汇总)
├── bench_subtask_T-NAV-1_v2.log
└── bench_subtask_batch_v1_think.log

arc_agent/subtask_probes/__init__.py                   (5 个 generator,~370 行)
scripts/bench_subtask.py                               (single-subtask runner,/think 注入)
scripts/bench_subtask_batch.py                         (batch runner)
tests/test_subtask_probes.py                           (8 sanity tests)
```

## 9. 引用

- 上游设计: `docs/project/2026-05-17-v0-subtask_decomp/architecture.md`
- 模型来源: [`../2026-05-17-v0-model_bench/report.md`](../2026-05-17-v0-model_bench/report.md) + [`../2026-05-17-v0-model_bench/report_5game.md`](../2026-05-17-v0-model_bench/report_5game.md)
- 全局词汇: `docs/GLOSSARY_zh.md` ([[T-NAV-1]]、[[subtask probe]]、[[/think]] 等)
- 仓库总入口: `docs/README.md`
- `CLAUDE.md` Key modules
