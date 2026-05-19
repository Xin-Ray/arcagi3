# 2026-05-18 v0 det_goal_v3_extended_parser — production 反思闭环验证

> **作用**: 用扩展 parser (commit `932dcef`)在 production 跑,验证反思闭环 (parser→reject→新 hypothesis) 是否真触发,是否能转化为通关。
> **状态**: Phase 1 smoke 跑中。
> **方法**: 小样本 → 多 seed 并行 → 大样本,渐进式扩展(用户 2026-05-18 提)。

## 0. 原始数据 (outputs/ 引用)

### Phase 1 — smoke (1 round × 20 step, seed=42)

| 文件 | 说明 |
|---|---|
| `outputs/det_goal_v3_smoke_s42.log` | 运行日志 |
| `outputs/det_goal_v3_smoke_s42_<ts>/round_00/` | step PNG + trace + reflection_raw + GIF |
| `outputs/det_goal_v3_smoke_s42_<ts>/summary.json` | metrics |

### Phase 2 (多 seed,~50 step each)

_(待填)_

### Phase 3 (大样本,2 round × 100 step)

_(待填)_

## 1. 实验目标

跟 v1 / v2 同样配置,**唯一变量**:goal_evaluator parser 现在覆盖 100% production hypothesis vocab(commit `932dcef`)。

预期 v1 / v2 没出现的现象:
- `goal_achieved_det != None` 应该出现(parser 解析得到 verdict)
- `[GOAL CHECK]` alert 应该被注入到 next-step Action prompt
- 当 `achieved=True 且 env != WIN` → orchestrator 自动 reject hypothesis,强降 confidence,Reflection 下一步写新的

## 2. Phase 1 smoke 1 结果 (1 round × 20 step, seed=42)

`outputs/det_goal_v3_smoke_s42_20260518-172043/round_00/`

### 2.1 数字

| 指标 | 值 |
|---|---:|
| change_rate | **75%** (15/20) |
| Reflection 写 hypothesis | 19/20 |
| Parser parse 成功 (`goal_pred_kind != ''`) | **19/20** ✅ 大幅好于 v2 (0/100) |
| Parser verdict 非 None | **0/20** ⚠️ 仍 0 |
| `[GOAL CHECK]` alert 出现 | 0/20 |
| 不同 hypothesis 数 | 3 |
| `rejected_goals` 累积 | 2 (Reflection 自己写入) |
| levels won | 0 |

### 2.2 sanity check

- [x] parser **parse kind** 至少 1 次:19/20 ✅
- [ ] parser **verdict 非 None** 至少 1 次:**0/20** ❌
- [ ] `[GOAL CHECK]` alert 至少 1 次出现:**0/20** ❌
- [ ] `goal_achieved_det = True 且 env != WIN` 触发:**0** ❌

### 2.3 Bug 找到 — Reflection 在幻觉

Reflection 写的 hypothesis 跟 frame 不符:

| 项 | 实际 ar25 frame (click_targets snapshot) | Reflection 的 hypothesis |
|---|---|---|
| 对象数 | 2 (1 yellow_9x9 + 1 gray_9x9) | "yellow 1x1 AND red 1x1" |
| 颜色 | yellow + **gray** | yellow + **red** ❌ |
| 大小 | **9x9** | **1x1** ❌ |

Parser 找 (yellow, red) → frame 没 red → selected={1 yellow} → `len(selected) < min_count(2)` → 返回 None。

**所以 parser 沉默是正确的** — 它在 silent-reject 幻觉 hypothesis,但**没把信号反馈给 Reflection**,所以反思机制不动。

### 2.4 Fix (commit `<pending>`)

新增 orchestrator-level **"hallucination detection"**: parser parse 成 kind 但 verdict=None,且 frame 非空 → hypothesis 命名了 frame 不含的实体 → inject `[GOAL CHECK] Hypothesis names {colors} but frame only contains {actual_colors}. Hypothesis is HALLUCINATED.` + 强降 goal_confidence。

scripts/run_v3_multi_round.py:730-770 新加 elif 分支处理这种情况。

## 2'. Phase 1 smoke 2 结果 (advisory alert fix, 1×20 seed=42)

`outputs/det_goal_v3_smoke2_s42_20260518-182233/round_00/`

| 指标 | smoke 1 | smoke 2 |
|---|---:|---:|
| change_rate | 75% | 80% |
| Parser kind parse | 19/20 | 19/20 |
| Parser verdict 非 None | 0/20 | 0/20 |
| **HALLUCINATED alert 出现** | 0 | **18** ✅ |
| confidence 强降 low | n/a | **20/20** ✅ |
| hypothesis 改写次数 | 3 | 2 ❌ |
| Reflection 仍写 "yellow+red" 次数 | n/a | **18/20** ❌ |

### 2'.2 关键观察

✅ Alert 机制工作: 18/20 步注入 `[GOAL CHECK] Hypothesis names ['yellow', 'red'] but frame only contains ['gray', 'maroon', 'purple', 'tan', 'yellow']. Hypothesis is HALLUCINATED.`

❌ **Reflection 完全无视 alert**: 即使 17 步连续告诉它"frame 没 red,只有 [gray, maroon, purple, tan, yellow]",Reflection 仍写 "yellow+red"。这印证 CLAUDE.md 的 "prompt 只能劝,orchestrator 才能管"。

### 2'.3 Fix v2 — orchestrator 强 reject

- **不再仅"劝"**,改成: 检测到 hallucinated hypothesis 时,**直接清空** `knowledge.goal_hypothesis = ""` + **push to `rejected_goals`**
- Reflection 下一步看到 hypothesis 是空 + 它的旧提议在 rejected_goals 里 → 必须写新的(且不能写同一个,sentinel R1/R5 拦截)
- Commit: `<pending>` in `scripts/run_v3_multi_round.py` line ~730

## 2''. Phase 1 smoke 3 结果 (aggressive reject fix, 1×20 seed=42)

`outputs/det_goal_v3_smoke3_s42_<ts>/round_00/`

_(等 monitor `bsox592lf` 完成)_

### 2''.1 预期

| 指标 | smoke 2 | smoke 3 预期 |
|---|---:|---:|
| HALLUCINATED alert 出现 | 18/20 | 可能少于 18(被 reject 后 Reflection 改写,后续不再幻觉)|
| hypothesis 改写次数 | 2 | **应 > 5** (每次被 reject 必写新的)|
| `rejected_goals` 累积 | 1 | **应 > 5** |
| change_rate | 80% | 看是否能维持 |

## 3. Phase 2 多 seed (variance check)

待 Phase 1 sanity 通过后跑 3-4 个不同 seed,~50 step each。

预期目标:
- 每个 seed 都至少 1 次触发 `[GOAL CHECK]`
- change_rate 跨 seed 标准差 < 15pp(说明不是偶然)
- 至少 1 个 seed 通关 ≥ 1 level (低预期,如果有就大新闻)

## 4. Phase 3 大样本 (2 round × 100 step)

待 Phase 2 通过后跑。跟 v1 / v2 directly comparable。

## 5. 跨实验对比表

| 指标 | v1 (refl truncated) | v2 (refl works, parser 0% cov) | v3 Phase 3 (refl + 100% parser cov) |
|---|---:|---:|---:|
| change_rate | 75% | 47% | _待填_ |
| Reflection 写 hypothesis | 0/100 | 100/100 | _待填_ |
| Parser 触发 | 0/100 | 0/100 | _待填_ |
| `[GOAL CHECK]` alert 出现 | 0 | 0 | _待填_ |
| `rejected_goals` | 0 | 0 | _待填_ |
| levels won | 0 | 0 | _待填_ |

## 6. 决策树(给跑完后用)

| Phase 1 smoke 结果 | 下一步 |
|---|---|
| parser 没触发 (verdict 全 None) | **bug**: parser 集成在 orchestrator 错位。debug fix 后重 smoke |
| parser 触发但 alert 没注入到 Action | bug: alert 路径错。看 reflection_raw + Action prompt diff |
| parser 触发 + alert 注入 + hypothesis 改变,但 change_rate 跌 | **没 bug,但行为退化** — 反思机制 inconvenient,需评估改进 vs 回退 |
| parser 触发 + change_rate 维持 + 任 1 level won | **大胜利**,直接 Phase 3 验证再说 |

## 7. 文件清单

```
docs/project/2026-05-18-v0-det_goal_v3_extended_parser/
└── report.md                                          (本文件)

arc_agent/goal_evaluator.py                            (v1 extended parser, commit 932dcef)
arc_agent/prompts_v3_2.py                              (force_cot ASK block)
scripts/run_v3_multi_round.py                          (evaluate_goal 集成)
tests/test_goal_evaluator.py                           (31 tests pass)

outputs/det_goal_v3_smoke_s42.log
outputs/det_goal_v3_smoke_s42_<ts>/                   (待生成)
```

## 8. 引用

- 上游 A/B (parser 决策): [`../2026-05-18-v0-goal_judge_ab/report.md`](../2026-05-18-v0-goal_judge_ab/report.md)
- v1 / v2 实验: [`../2026-05-18-v0-det_goal_plus_force_cot/report.md`](../2026-05-18-v0-det_goal_plus_force_cot/report.md)
- 反思逻辑设计 (用户提): det_goal §6
- 全局词汇: `docs/GLOSSARY_zh.md`
- 仓库总入口: `docs/README.md`
