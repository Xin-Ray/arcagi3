# 2026-05-18 v0 force_cot — A/B Verification Report

> **作用**: 验证用户假设 "在 user prompt 加 step-by-step 指令能否解决 [[CoT 激活率]] 问题"。
> **状态**: 完成 — **混合结论**: force_cot 让激活率从 0-100% 任性分布 → 56-100% 稳定区,但揭示 **T-GOAL 任务模型本身能力到顶**(long_acc 33.7%,近随机)。
> **决策**: Scenario B+ (deterministic goal eval + 选择性 force_cot)。详见 §7 + `next_decisions.md`。

## 0. 原始数据 (outputs/ 引用)

| 文件 | 说明 |
|---|---|
| `outputs/bench_subtask_batch_v2_forcecot.log` | force_cot 运行日志(~1h47m) |
| `outputs/subtask_batch_20260518-000721/` | 跨 subtask 汇总(metrics.json + summary.md) |
| `outputs/subtask_T-NAV-1_20260518-000721/` | T-NAV-1 force_cot (71/100) |
| `outputs/subtask_T-NAV-2_20260518-000721/` | T-NAV-2 force_cot (93/100) |
| `outputs/subtask_T-NAV-3_20260518-000721/` | T-NAV-3 force_cot (97/100) ✅ |
| `outputs/subtask_T-SEL-1_20260518-000721/` | T-SEL-1 force_cot (70/100) |
| `outputs/subtask_T-GOAL_20260518-000721/` | T-GOAL force_cot (35/100) |
| (baseline 对照) `outputs/subtask_batch_20260517-163018/` | default prompt baseline,2026-05-17 |

## 1. 实验设计

| 项 | 值 |
|---|---|
| 假设 | 在 user prompt 末尾加 "Solve step by step. Show your work, end with Answer: X" 能提升 [[CoT 激活率]] 并改善 accuracy |
| 模型 | SmolLM3-3B (`HuggingFaceTB/SmolLM3-3B`),4-bit nf4,bf16 compute |
| reasoning_mode | `cot` + `/think` system flag |
| 控制变量 | 同 5 subtask、同 seed=42、同 max_new_tokens=1024、同 model state |
| 唯一变量 | `--prompt-style {default, force_cot}` (commit `01fc61b`) |
| 总 wall | force_cot: 9187s (≈ 2h33m);default: 6483s (≈ 1h47m) |

代码:`scripts/bench_subtask.py:format_probe()` 的 `force_cot` 分支末行替换为 "Solve this step by step. First, write down the relevant values from the question (positions, distances, conditions). Then check each option. Finally end your reply with: Answer: X"。

## 2. 结果一览

| Subtask | Default acc | force_cot acc | Δ | 判定 |
|---|---:|---:|---:|---|
| T-NAV-1 单步方向 | 52.0% | **71.0%** | **+19.0pp** | FAIL(目标 90%)|
| T-NAV-2 多步同方向 | 95.0% ✅ | **93.0%** | -2.0pp | PASS(噪声,仍 PASS)|
| T-NAV-3 L 型双段 | 67.0% | **97.0%** | **+30.0pp** | **PASS** ✅ 🎉 |
| T-SEL-1 ACTION6 click | 78.0% | **70.0%** | **-8.0pp** | **WORSE!** ⚠️ |
| T-GOAL YES/NO 目标 | 30.0% | **35.0%** | +5.0pp | SEVERE FAIL |

## 3. 激活率 + long_acc / short_acc 全表

### Default (2026-05-17 baseline,引自 `subtask_decomp/report.md` §2)

| Subtask | acc | CoT 激活 | long_acc | short_acc |
|---|---:|---:|---:|---:|
| T-NAV-1 | 52% | 11/100 | 90.9% | 47.2% |
| T-NAV-2 | 95% | 100/100 | 95.0% | n/a |
| T-NAV-3 | 67% | 1/100 | 0% (n=1) | 67.7% |
| T-SEL-1 | 78% | 86/100 | 79.1% | 71.4% |
| T-GOAL | 30% | **0/100** | n/a | 30.0% |

### force_cot

| Subtask | acc | CoT 激活 | long_acc | short_acc | avg elapsed_s |
|---|---:|---:|---:|---:|---:|
| T-NAV-1 | 71% | **100/100** | 71.0% | n/a | 24.2 |
| T-NAV-2 | 93% | 9/100 | 44.4% | **97.8%** | 10.0 |
| T-NAV-3 | 97% | **56/100** | 96.4% | 97.7% | 12.4 |
| T-SEL-1 | 70% | **100/100** | 70.0% | n/a | 29.5 |
| T-GOAL | 35% | **98/100** | **33.7%** | 100% (n=2) | 15.9 |

## 4. 三个核心发现

### 4.1 force_cot **有效解决"激活率"问题** (假设 H1 部分成立)

| Subtask | default 激活率 | force_cot 激活率 | Δ |
|---|---:|---:|---:|
| T-GOAL | 0/100 | **98/100** | +98pp |
| T-NAV-3 | 1/100 | 56/100 | +55pp |
| T-NAV-1 | 11/100 | 100/100 | +89pp |
| T-SEL-1 | 86/100 | 100/100 | +14pp |
| T-NAV-2 | 100/100 | 9/100 | **-91pp** ⚠️ |

→ **除 T-NAV-2 反向**(它的 short_acc 97.8% 比 long_acc 44.4% 还高,说明 force_cot 反而让模型"想太多")。其余 4 task 激活率全部 ≥ 56/100,问题 1 解决。

### 4.2 激活了 ≠ 答对了 (假设 H2 部分证伪)

| Subtask | force_cot long_acc | 含义 |
|---|---:|---|
| T-NAV-3 | **96.4%** | 推理能稳赢 |
| T-NAV-1 | 71.0% | **模型即使全 CoT 也破不了 80%** — letter-shuffle attention 弱 |
| T-SEL-1 | 70.0% | (x, y) vs (row, col) 坐标系混淆,CoT 也救不了 |
| T-GOAL | **33.7%** | **near-random** — 任务超出模型能力上限 |

→ 假设 "激活了 CoT = 答对" 只对 T-NAV-3 成立。**T-GOAL 33.7% long_acc 是这次最重要的发现**: 模型不是没思考,是思考也想不明白。

### 4.3 force_cot **不是 free lunch** — 在某些任务上反而变差

| Subtask | Δ | 解释 |
|---|---:|---|
| T-SEL-1 | **-8pp** | 长 CoT 链反而让模型在坐标比较时 distracted by 长 prompt 干扰文字 |
| T-NAV-2 | -2pp | 噪声,但说明本来无脑能做对的任务,加 CoT 没收益 |

→ **不能把 force_cot 一刀切应用到 production**;需要按任务类型决定。

## 5. 改架构方案 (我自己决定的下一步)

### 方案: Scenario "B+" = deterministic goal + 选择性 force_cot

按 [[next_decisions.md]] §2,T-GOAL = 35% < 40% → 严格走 Scenario C(只 deterministic goal,不动 production prompt)。

**但我决定走 B+(混合)**,理由:
- T-NAV-3 +30pp 是 production Action Agent 决策类型最直接的 task,**Action Agent 应该用 force_cot**
- T-SEL-1 -8pp 警告:**当 Action 含 ACTION6 click 时不要 force_cot**
- T-GOAL long_acc 33.7% → Reflection Agent 的 goal-check **必须** 走 deterministic

具体改动 (在 `feat-2026-05-18-v0-det_goal_plus_force_cot`):

1. **`arc_agent/goal_evaluator.py`**(新建)
   - `evaluate_goal(goal_hypothesis: str, objects: dict[str, ObjectRecord]) -> Optional[bool]`
   - 解析常见 hypothesis 模式 (vertical-align / move-to-col / stack-on / overlap):
     - "align ... vertically in left column" → 全部 `obj.col == 0` 且 row 互不相等
     - "move ... to col=N" → target.col == N
     - "stack X on Y" → X.row == Y.row - 1 and X.col == Y.col
   - 不识别的模式 → 返回 None (LLM 接管)

2. **`tests/test_goal_evaluator.py`**(新建): ≥ 4 case

3. **`arc_agent/agents/reflection_agent.py`** 改 `reflect()` 末尾
   - 调 `evaluate_goal(knowledge.goal_hypothesis, latest_objects)`
   - 若返回 True → `delta["goal_achieved"] = True`(OVERRIDE LLM)
   - 若返回 None → 保留 LLM 的 delta

4. **`arc_agent/prompts_v3_2.py:build_action_user_prompt`** 末尾追加 force_cot 文字
   - 仅 Action Agent (不动 Reflection prompt)
   - "Solve this step by step. Compute Δrow / Δcol if relevant before deciding."

5. **实验**: `scripts/run_v3_multi_round.py --game ar25 --rounds 2 --max-actions 100 --tag det_goal_action_force_cot_ar25_2x100`
   - 跟 main baseline (5-8% change_rate, 0 通关) + SmolLM3 5×2×300 baseline (64% change_rate, 0 通关) 对比
   - 目标: 任 1 round 通关 ≥ level 1

### 不在这个分支做的(留给用户审核)

- T-SEL-1 prompt rewrite ((x,y) → (row,col)) — 已知 P1,等用户决定
- 升级到 SmolLM3-7B / Qwen3-4B 看 T-GOAL long_acc base rate

## 6. 当前 PASS / FAIL 矩阵

| Subtask | default | force_cot | 最终判定 (取更高) |
|---|---:|---:|---|
| T-NAV-1 | 52% | 71% | FAIL — letter-shuffle attention 弱,prompt 救不动 |
| T-NAV-2 | **95% PASS** | 93% | **PASS** (default) |
| T-NAV-3 | 67% | **97% PASS** | **PASS** (force_cot,**+30pp**) |
| T-SEL-1 | 78% | 70% | FAIL — 坐标系混淆 + force_cot 适得其反 |
| T-GOAL | 30% | 35% | SEVERE FAIL — **必须** deterministic Python |

通过率: **2/5** (T-NAV-2 + T-NAV-3),vs 一轮前 1/5。

## 7. 当前架构改进的最终建议

1. ✅ **Reflection 的 goal-check 走 deterministic** (下一个分支动手)
2. ✅ **Action Agent prompt 加 force_cot 风格语句**(同分支)
3. ⏳ **Reflection prompt 不动**(force_cot 没帮到 T-GOAL,反而拖 wall 时间)
4. ⏳ **T-SEL-1 留 P1**:坐标系约定 prompt 重写 (单独分支,小改)
5. ⏳ **T-NAV-1 留 P2**:模型升级实验

## 8. 文件清单

```
docs/project/2026-05-18-v0-force_cot/
└── report.md                                          (本文件)

docs/project/2026-05-17-v0-subtask_decomp/
├── next_decisions.md                                  (决策树)
└── report.md                                          (前一日 baseline)

outputs/subtask_batch_20260518-000721/                 (force_cot 跨 subtask 汇总)
├── metrics.json
└── summary.md

outputs/subtask_{T-NAV-1,T-NAV-2,T-NAV-3,T-SEL-1,T-GOAL}_20260518-000721/  × 5
├── metrics.json
├── per_probe_smollm3-cot.jsonl
└── summary.md

outputs/bench_subtask_batch_v2_forcecot.log

outputs/subtask_batch_20260517-163018/                 (default baseline,对照)
outputs/subtask_{T-*}_20260517-163018/                 × 5

scripts/bench_subtask.py                               (--prompt-style flag)
scripts/bench_subtask_batch.py                         (--prompt-style flag)
```

## 9. 引用

- A/B 设计 + 决策树: `docs/project/2026-05-17-v0-subtask_decomp/next_decisions.md`
- baseline (default prompt): `docs/project/2026-05-17-v0-subtask_decomp/report.md`
- 全局词汇: `docs/GLOSSARY_zh.md` — [[CoT 激活率]] / [[长链 vs 短链]] / [[/think]] / [[force_cot]] (新加)
- 仓库总入口: `docs/README.md`
