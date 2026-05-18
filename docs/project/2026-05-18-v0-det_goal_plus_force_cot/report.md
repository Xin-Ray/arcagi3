# 2026-05-18 v0 det_goal_plus_force_cot — Architecture Improvement Experiment

> **状态**: v1 跑完 (Reflection 截断 bug 暴露,goal_evaluator 没拿到输入);v2 跑中 (修了 token 预算)。
> **分支**: `feat-2026-05-18-v0-det_goal_plus_force_cot` (本地,**不 push**)。
> **决定依据**: `docs/project/2026-05-18-v0-force_cot/report.md` §5+§7

## 0. 原始数据 (outputs/ 引用)

### v1 (max-new-tokens-reflection=250,默认值,**Reflection 输出被 CoT 链截断**)

| 文件 | 说明 |
|---|---|
| `outputs/det_goal_force_cot_ar25_2x100_20260518-025313/report.md` | 跑结果汇总 |
| `outputs/det_goal_force_cot_ar25_2x100_20260518-025313/summary.json` | 结构化 metrics |
| `outputs/det_goal_force_cot_ar25_2x100_20260518-025313/knowledge_history.jsonl` | Knowledge per round |
| `outputs/det_goal_force_cot_ar25_2x100_20260518-025313/round_00/trace.jsonl` | step-by-step (新字段 `goal_achieved_det`, `goal_pred_kind`) |
| `outputs/det_goal_force_cot_ar25_2x100_20260518-025313/round_00/reflection_raw.txt` | Reflection raw outputs(诊断 truncation 的关键)|
| `outputs/det_goal_force_cot_ar25_2x100_20260518-025313/round_{00,01}/play.gif` | 可视化 |
| `outputs/det_goal_force_cot_ar25_2x100.log` | 运行日志 |

### v2 (max-new-tokens-reflection=2048,跑中)

| 文件 | 说明 |
|---|---|
| `outputs/det_goal_force_cot_v2_ar25_2x100.log` | v2 运行日志 |
| `outputs/det_goal_force_cot_v2_ar25_2x100_<ts>/` | 预计完成后写入 |

## 1. 实验设计

### 假设链
1. T-GOAL bench (`2026-05-18-v0-force_cot`) 证明 SmolLM3-3B long_acc 33.7% → **LLM 做不了 goal recognition,需要 deterministic Python**
2. T-NAV-3 force_cot bench 证明 **+30pp PASS** → **Action Agent prompt 应该加 step-by-step 措辞**
3. 因此 production 应:
   - Action Agent prompt: 加 force_cot 措辞 ✓
   - Reflection 之后: 调 `evaluate_goal(hypothesis, frame_objects)`,把 `[GOAL CHECK]` alert 注入 next-step Action prompt ✓

### 实现 (commit `456f317`)

- **`arc_agent/goal_evaluator.py`** (NEW,156 行)
  - 解析 5 种 hypothesis 模式 → `GoalPredicate`
  - 不识别 → 返回 None,fall back 给 LLM
  - **17 个 unit tests 全 PASS** (`tests/test_goal_evaluator.py`)
- **`arc_agent/prompts_v3_2.py`**: `_ACTION_ASK_BLOCK` + `_ACTION_ASK_BLOCK_MC` 加 force_cot 措辞
- **`scripts/run_v3_multi_round.py`**: import `evaluate_goal`,Reflection merge 后调一次,根据结果注入 `[GOAL CHECK]` alert,**当 achieved=True 但 env != WIN 时强降 goal_confidence 到 low**

## 2. v1 结果 (有效配置,但 Reflection 被截)

### 2.1 数字

| 指标 | round 0 | round 1 |
|---|---:|---:|
| steps | 100 | 100 |
| changed | 75 | 73 |
| **change_rate** | **75%** | **73%** |
| levels gained | 0 | 0 |
| won | False | False |
| orch_override count | 51 | 46 |

**最佳 change_rate 纪录**: 75% / 73% — vs `main` (5-8%) 和 SmolLM3 5×2×300 (`/no_think`, 64%) **首次破 70%**。

### 2.2 关键观察

- **Reflection 全程 think 但产不出 JSON**: `reflection_raw.txt` 每步 800-1200 chars 都在 `<think>` 块里,被 max_new_tokens=250 截断,**JSON 输出从未触发**。结果:
  - `goal_hypothesis = ""` (空)
  - `action_semantics = {}`
  - 所有 `reflection_delta = {}`
- **`goal_pred_kind` 100/100 都是空字符串**: 因为 hypothesis 是空,evaluator 没东西可解析
- **`goal_achieved_det` 100/100 都是 None**: 同上

→ **deterministic goal evaluator 在 v1 没起作用** —— 我的 force_cot Action prompt 改动单独贡献了 75% change_rate。

### 2.3 75% change_rate 是 force_cot 单一贡献的证据

唯一变量(对比 main 5-8%):
- force_cot Action ASK block ✓
- goal_evaluator integration → 没数据(hypothesis 空)
- /think + max-new-tokens-action=1024 → 但 Reflection 还是 250

所以 +67-70pp change_rate 主要来自:
1. `/think` mode 让 Action Agent 真的推理
2. force_cot ASK block 让 Action 推理结构化("First note position. Then pick action.")
3. 1024 token budget 让 chain 不被截

orch_override 也起作用:51/100 + 46/100 步被 R3 forced-explore 等 mask 路径接管 → 不让 LLM 收敛到无效动作。

## 3. v1 暴露的架构 bug (在 v2 修复)

**bug**: `--max-new-tokens-reflection` 默认 250。这个值是为 `/no_think` mode 设的(短 JSON 输出);切到 `/think` mode 后,SmolLM3 想了 800-1200 chars (~300-400 tokens) 还没说完,250 token 早就截掉了,JSON 永远不出现。

**症状**:
- Reflection 输出全是开头的 `<think>` 段
- 没有 `}` 结尾,parse_reflection_output 返回 `({}, False)`
- 所有 delta 都是 `{}` → Knowledge 永远不更新

**fix (v2 已 launch)**: `--max-new-tokens-reflection 2048`

## 4. v2 预期 (跑中)

如果 fix 起作用:
- Reflection 输出 JSON,Knowledge 字段填充
- `goal_hypothesis` 应该出现真的英文描述
- `goal_pred_kind` 出现 align_col / align_row / 等
- `[GOAL CHECK]` alert 在某些步出现
- 如果 hypothesis 正确,achieved=True but env not WIN → 强降 confidence,逼 Reflection 改 hypothesis
- 目标: change_rate ≥ 70% (维持 v1) + 至少 1 round 通关 ≥ level 1

## 5. (v2 结果占位 — Monitor 完成后填)

_(待 `outputs/det_goal_force_cot_v2_ar25_2x100_<ts>/` 完成)_

## 6. 不管 v2 结果如何,已确认的发现

1. **change_rate 5-8% → 75% 不是模型问题,是 prompt+budget 问题** —— 同一 SmolLM3-3B,改 prompt 就拿到首个 70%+
2. **`--max-new-tokens-reflection 250` 是 Reflection-in-CoT-mode 的隐藏 blocker** —— 应该跟 reasoning_mode 联动调整
3. **goal_evaluator code 已落,单测 17/17 PASS** —— 即使本次实验没数据,代码可重用
4. **(待 v2 验证)** deterministic goal-check 在 production 是否能转化为 win

## 7. 跟进项

| 优先级 | 行动 |
|---|---|
| P0 (v2 跑完后) | 写 §5 真实结果 + 决定 next branch |
| P0 | 把 `--max-new-tokens-reflection` 默认值绑到 reasoning_mode: cot → 2048,no_think → 250 |
| P1 | Reflection 用 `/no_think` 让 JSON 快出,Action 用 `/think` 给推理预算 —— 但当前 backbone 是共享的,要做 per-call mode override |
| P2 | 如果 v2 仍 0 win:试 max-action 200/300 看 budget 是否限制 |

## 8. 文件清单

```
docs/project/2026-05-18-v0-det_goal_plus_force_cot/
└── report.md                                          (本文件)

arc_agent/goal_evaluator.py                            (NEW, 156 lines)
tests/test_goal_evaluator.py                           (17 tests, all PASS)
arc_agent/prompts_v3_2.py                              (force_cot ASK block)
scripts/run_v3_multi_round.py                          (evaluate_goal 集成)

outputs/det_goal_force_cot_ar25_2x100_20260518-025313/  (v1)
├── report.md
├── summary.json
├── knowledge_history.jsonl
├── round_00/{trace.jsonl, reflection_raw.txt, play.gif, step_*.png}
└── round_01/{...}
outputs/det_goal_force_cot_ar25_2x100.log

outputs/det_goal_force_cot_v2_ar25_2x100_*/             (v2, pending)
outputs/det_goal_force_cot_v2_ar25_2x100.log

docs/project/2026-05-18-v0-force_cot/                  (上一日 A/B,触发本实验)
docs/project/2026-05-17-v0-subtask_decomp/             (5 subtask 验证,触发 A/B)
```

## 9. 引用

- 触发本实验的 A/B 报告: `docs/project/2026-05-18-v0-force_cot/report.md`
- 决策树: `docs/project/2026-05-17-v0-subtask_decomp/next_decisions.md`
- 5 subtask 验证: `docs/project/2026-05-17-v0-subtask_decomp/report.md`
- 全局词汇: `docs/GLOSSARY_zh.md` ([[force_cot]] / [[CoT 截断]] / [[T-GOAL]])
- 仓库总入口: `docs/README.md`
