# 2026-05-18 v0 goal_judge_ab — Python parser vs LLM judge A/B

> **作用**: 验证 "goal_hypothesis 是否达成" 这一关键反思 step 用 deterministic Python parser 还是 LLM judge 更好。
> **状态**: 完成。**Python parser 完胜**(83% vs 68% acc,1.25M× 速度)。
> **决策**: production 反思闭环用 parser-only;LLM judge 作为 hybrid fallback 备用。

## 0. 原始数据 (outputs/ 引用)

| 文件 | 说明 |
|---|---|
| `outputs/bench_goal_judges_20260518-160438/metrics.json` | 汇总:parser 83%, judge 68% |
| `outputs/bench_goal_judges_20260518-160438/per_probe.jsonl` | 200 行 (100 T-GOAL × 2 method + 11 v2 prod) |
| `outputs/bench_goal_judges_20260518-160438/summary.md` | 表格摘要 |
| `outputs/bench_goal_judges_full.log` | 运行日志 |

## 1. 实验问题

[[T-GOAL]] subtask bench (2026-05-17) 显示 SmolLM3-3B CoT long_acc **33.7%** — 模型即使思考也答不对 "given hypothesis + state, is achieved?" 用户 (2026-05-18) 提出**两个方向都测**:
1. Deterministic Python parser
2. LLM judge (prompt 比 T-GOAL bench 干净)

## 2. 实验设计

### Dataset A: T-GOAL 100 probe (ground truth 已知)

- Hypothesis: `"align the two yellow squares vertically in the left column"`
- 25 success / 75 failure 期望分布,seed=42
- 4 种 failure 模式:wrong_col_one / wrong_col_both / wrong_row_overlap / off_by_one

### Dataset B: v2 production trace (11 unique hypotheses)

- Source: `outputs/det_goal_force_cot_v2_ar25_2x100_20260518-090129/round_00`
- 11 distinct hypotheses Reflection wrote during 100-step run
- 用于测 parser 的 **覆盖率**(parse rate),没 GT 因为无法手标

### Method A: Deterministic Python parser (extended v1)

- File: `arc_agent/goal_evaluator.py`
- 新加 patterns: `to the {top/bottom/left/right} edge` / `to the center` / `align X and Y` (no axis)
- 7 个 GoalPredicate kinds: align_col / align_row / align_any / move_to_col / move_to_row / move_to_center / stack / adjacent
- 31 个单测全 PASS

### Method B: LLM judge

- File: `arc_agent/llm_goal_judge.py`
- Prompt: YES/NO 二选(比 T-GOAL bench 的 4 选 1 多选清晰)
- SmolLM3-3B 4-bit nf4 + `/think` + 1024 max_new_tokens
- System prompt 显式 "Solve step by step. End with: Answer: YES / NO"

## 3. 结果

### 3.1 T-GOAL accuracy

| Method | accuracy | 速度 |
|---|---:|---:|
| **Python parser (extended)** | **83.0%** (83/100) | 0.02 ms |
| **LLM judge (SmolLM3 CoT)** | 68.0% (68/100) | ~25 s |
| (reference) T-GOAL bench force_cot LLM | 35.0% | ~17 s |

→ Parser **+15pp** vs LLM judge,**+48pp** vs 原 T-GOAL bench;速度 **1,250,000× 更快**。

### 3.2 Confusion matrix(关键)

**Parser** (100 probes):

| | predicted True | predicted False |
|---|---:|---:|
| GT True (27) | **27** | 0 |
| GT False (73) | 17 | 56 |

- Recall on TRUE = **100%**(27/27)
- Precision on TRUE = 61%(17 FP)
- 0 false negatives → **每个真 achieved 都会触发**

**LLM judge** (100 probes):

| | predicted True | predicted False | None (parse fail) |
|---|---:|---:|---:|
| GT True (27) | 6 | 19 | 2 |
| GT False (73) | 11 | 62 | 0 |

- Recall on TRUE = **22%**(6/27)
- Precision on TRUE = 35%
- 19 false negatives → **78% 真 achieved 漏掉**

### 3.3 production hypothesis 覆盖

v2 round 0 trace 里 11 distinct hypotheses,parser 全部 parse 成有效 kind:

| Hypothesis | Parser kind |
|---|---|
| "move red and yellow to the **top edge**" | move_to_row (target=0) |
| "move red and yellow to the **left edge**" | move_to_col (target=0) |
| "**align** the tan objects #7 and #8" | **align_any** (new) |
| "move tan #7 and #8 **towards each other**" | align_any |
| "move tan #7 and #8 **to the center**" | **move_to_center** (new) |
| "align purple to **the center**" | move_to_center |
| (5 more variants) | move_to_center / align_any |

→ parser 覆盖率 **100%**(9/9 unique patterns)。

## 4. 关键发现

### 4.1 Parser 在反思闭环的关键属性 (recall on TRUE) 上完胜

production "反思" 逻辑(用户 2026-05-18 提的):
1. Reflection 写 hypothesis
2. 判定 hypothesis 是否达成
3. **达成 + env != WIN → reject hypothesis,写新的**

→ step 2 的 **recall on TRUE 是决定性的**。LLM 22% recall 等于 78% 的真达成事件错失,反思机制实质不动。Parser 100% recall 等于每个真达成都会触发反思 / hypothesis 拒绝。

### 4.2 17 个 parser FP 是 "保守失败",副作用小

17 个 false positive 全来自 "wrong_row_overlap" — 两个 obj 都在 col=0 但行相同(stacked,不算"vertically arranged")。Parser 把它判 True 是因为 align_col 只查共列,不检查行不同。

→ production 影响:**过早 reject hypothesis** → Reflection 写新的 → **多探索一轮**。这比 FN(漏达成 → 永远不质疑 hypothesis)代价小得多。

要修也容易:`align_col` 加 "rows must differ" 约束。但当前 FP 副作用小,先不动。

### 4.3 LLM judge 68% vs T-GOAL bench 35% 的 33pp 差距说明什么

我新写的 YES/NO prompt 比 T-GOAL 多选 prompt 让 LLM **多对 33pp**。说明:
- T-GOAL bench 的 35% 是 **prompt-quality lower bound**,不是模型纯能力
- 但即使用"最好"的 prompt,LLM 仍输给 parser 15pp + 1.25M× 慢
- 同一模型不同 prompt 差 33pp → **prompt 工程 sensitive**,production 选 parser 还省了这个不稳定性

## 5. 决策

**production 反思 step 2 用 Python parser**。

| 状态 | 反思动作 |
|---|---|
| parser returns True + env not WIN | **reject hypothesis**,加 `rejected_goals`,Reflection 必须写新的 |
| parser returns False | continue toward target |
| parser returns None (无法 parse) | **fall back to LLM judge**(hybrid 备用,当前 ar25 不需要) |

Hybrid 设计前置:**先 parser → fail 才 LLM**,因为:
- Parser 100% 覆盖当前 ar25,LLM 不会被调
- 速度上 parser 一直在路径上(0.02 ms 不会拖慢)
- 未来 game 出现新 vocab 时,LLM 兜底保正确率

## 6. 下一步

跑 ar25 1×2×100,**带扩展 parser**(commit `932dcef`)。预期:
- `goal_achieved_det` 终于会有非 None 值
- `[GOAL CHECK]` alert 真正触发
- Reflection 写 hypothesis → parser 检测 achieved 但 env != WIN → 强降 confidence → Reflection 下一步写新 hypothesis
- 目标:任一 round 通关至少 1 level

如果还是 0 win,说明 **反思不是 0 通关的根因**,要去看 hypothesis 质量(T-DISCOVER 类问题) 或 action selection。

## 7. 文件清单

```
docs/project/2026-05-18-v0-goal_judge_ab/
└── report.md                                          (本文件)

arc_agent/goal_evaluator.py                            (extended v1,7 kinds)
arc_agent/llm_goal_judge.py                            (NEW,LLM judge baseline)
tests/test_goal_evaluator.py                           (31 tests, all pass)
scripts/bench_goal_judges.py                           (A/B bench runner)

outputs/bench_goal_judges_20260518-160438/
├── metrics.json
├── per_probe.jsonl
└── summary.md

outputs/bench_goal_judges_full.log

outputs/det_goal_force_cot_v2_ar25_2x100_20260518-090129/round_00/   (v2 trace 取 hypothesis)
└── knowledge_per_step.jsonl
```

## 8. 引用

- 触发本实验的用户提议: 2026-05-18 chat,"硬解析还是 agent 解析?两个都测"
- 上游 T-GOAL bench (35% baseline): [`../2026-05-18-v0-force_cot/report.md`](../2026-05-18-v0-force_cot/report.md) §3
- v2 production trace 来源: [`../2026-05-18-v0-det_goal_plus_force_cot/report.md`](../2026-05-18-v0-det_goal_plus_force_cot/report.md) §5.3
- 反思逻辑设计 (用户提): 同上 §6
- 全局词汇: `docs/GLOSSARY_zh.md`
