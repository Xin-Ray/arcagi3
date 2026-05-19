# v4 Phase 1 — /think vs /no_think on V4 minimal

> **作用**: Validation A — 选 Phase 2+ 用哪种 reasoning_mode。
> **结果**: `/no_think` 胜。decision_tree.md 表 row 3:`</think>_closed_rate < 80%` + `reasoning ≥ 8/10`。

## 0. 数据 (outputs/ 引用)

| 文件 | 内容 |
|---|---|
| `outputs/v4_phase1_think_s42.log` | /think 跑日志 |
| `outputs/v4_phase1_think_s42_20260519-021402/round_00/` | trace + action_raw + reflection_raw + step PNG |
| `outputs/v4_phase1_nothink_s42.log` | /no_think 初版日志 (match pattern 未支持) |
| `outputs/v4_phase1_nothink_s42_20260519-025100/round_00/` | 同上 |
| `outputs/v4_phase1b2_nothink_s42.log` | /no_think + match-pattern fix |
| `outputs/v4_phase1b2_nothink_s42_20260519-025631/round_00/` | 闭环工作的数据 |

## 1. 数字对比

| 指标 | /think (P1A) | /no_think v0 (P1B) | /no_think + match (P1B2) | 说明 |
|---|---:|---:|---:|---|
| steps | 10 | 10 | 10 | 全 ar25 1 round |
| change_rate | 80% | 80% | 80% | 持平 |
| ACTION1 占比 | 8/10 | 8/10 | 8/10 | bias 不变(下游问题)|
| **`</think>` 闭合** | **0/10** ❌ | n/a | n/a | /think 链全截 |
| **reasoning empty** | **10/10** ❌ | **0/10** ✅ | **0/10** ✅ | /no_think 解析完全 |
| Action raw chars/step | ~4100 | ~250 | ~250 | /no_think 短 16× |
| Parser kind 解析成功 | 9/10 (align_col) | 0/10 (match 不识别) | **3/10 (align_any)** | parser fix 后能识别 |
| Parser verdict 非 None | 9/10 (False) | 0/10 | 0/10 (但 alert 触发) | (备注 §2.2) |
| `rejected_goals` 累积 | 0 | 0 | **3** ✅ | hallucination 拦截工作 |
| GOAL CHECK alert | 0 | 0 | **2/10** ✅ | 反思闭环活了 |

## 2. 关键诊断

### 2.1 /think chain 在 V4 短 prompt 下仍不闭合

V4 minimal 砍掉 action_proposer / click_targets / 部分 KNOWLEDGE 后,prompt 比 v3.2 短了 30-40%。但 SmolLM3 /think 还是写 4000 chars (~1024 tokens) 不收尾。token 预算从 1024 → 2048(diag 阶段验证)仍 ramble。

**结论**: 在 production prompt (即使短化) 下,SmolLM3 `/think` 死循环不可控。**production 不能用 /think**。

### 2.2 /no_think 的 Reflection 字典

/no_think 模式下 Reflection JSON 输出完整:
```json
{
  "goal_hypothesis_update": "match every yellow 1x1 with a yellow target square",
  "action_semantics_update": null,
  "current_alert": "..."
}
```

但 hypothesis 都是 "match X with Y" 风格 — 跟之前 v2/v3 用的 "to the X edge" / "to the center" 不一样。**Reflection 在 /no_think 下选了新方言**。

Parser v1 之前没覆盖 "match"。`schema-validator wide` 把所有 hypothesis_update 当作 invalid → 丢弃 → knowledge.goal_hypothesis 一直空。

**fix**: 加 "match(es|ed|ing)?" → `align_any` 分支(commit `823e7cc`)。

### 2.3 修完后 P1B2 闭环验证

3 个 hypothesis 被 force-reject 入 rejected_goals:
1. "match every **red** dot with a red target square" (frame 没 red → hallucinated)
2. "match every **yellow object** with a yellow target square" (frame 有 yellow 但只 1 个 → align_any min_count=2 没通过 → 触发 hallucination detect)
3. "match the **yellow 1x1 (obj_002)** with a yellow target square" (obj_002 不存在? + 仍然 align_any 没法 evaluate)

→ **每次 Reflection 写新 hypothesis,parser 评估,verdict=None,orchestrator force-reject 它,Reflection 下一步改写**。闭环工作正常。

### 2.4 ACTION1 bias 仍 8/10 — 跟 reasoning_mode 无关

这层问题不在 Phase 1 解决范围。可能源头:
- model 自身偏好 ACTION1 (它在 prompt 的 "legal actions" 第一个)
- R3 forced explore 只在 stuck 时干预,大部分 step 不触发
- Knowledge 全空 (V4 砍了 action_semantics) → model 没"known-good"标签 → 但仍然偏 ACTION1

留 Phase 2 + 3 看是否 200 步后或加某个 ablation 模块能纠正。

## 3. 决策

按 decision_tree.md §"Phase 1":
- `/think` `</think>_closed_rate` 0/10 < 80%
- `/no_think` reasoning_empty 0/10 → reasoning_nonempty = 10/10 ≥ 8/10

**Decision: `/no_think` for Phase 2 onwards**。

## 4. 跟进 (Phase 2 launch)

```powershell
.venv\Scripts\python.exe scripts\run_v3_multi_round.py `
  --game ar25 --max-actions-total 200 --max-actions 100 --rounds 5 `
  --seed 42 --mask off --propose off `
  --click-targets off --action-semantics-from-llm off `
  --hard-rules off --validate-hypothesis-schema wide `
  --backbone HuggingFaceTB/SmolLM3-3B --reasoning-mode no_think `
  --max-new-tokens-action 256 --max-new-tokens-reflection 1024 `
  --tag v4_phase2_baseline_s42
```

bash task `b36q917dd`,monitor `b2mnc2jan`。

## 5. 文件清单

```
docs/project/2026-05-19-v0-v4_clean_baseline/
└── phase1_thinkvsnothink.md            (本文件)

outputs/v4_phase1_think_s42_*           /think 数据
outputs/v4_phase1_nothink_s42_*         /no_think v0 数据
outputs/v4_phase1b2_nothink_s42_*       /no_think + match-pattern fix

arc_agent/goal_evaluator.py             (parser "match" 分支)
tests/test_goal_evaluator.py            (4 个新 case,33 tests pass)
```
