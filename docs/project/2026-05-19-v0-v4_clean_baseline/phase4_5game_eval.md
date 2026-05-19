# v4 Phase 4 — 5-game G_base eval

> **配置**: V4 minimal + `--propose on` (Phase 3 winner). 5 game one-at-a-time, 1×200 step budget-pooled, /no_think SmolLM3-3B.
> **结果**: change_rate **mean 82%** (vs SmolLM3 5×2×300 baseline 64%,+18pp);**0/5 wins** (跟 baseline 一样)。

## 0. 原始数据 (outputs/ 引用)

| Game | Output dir |
|---|---|
| ar25 | `outputs/v4_phase4_g1_ar25_s42_20260519-053015/` |
| bp35 | `outputs/v4_phase4_g2_bp35_s42_20260519-062440/` |
| cd82 | `outputs/v4_phase4_g3_cd82_s42_20260519-065756/` |
| cn04 | `outputs/v4_phase4_g4_cn04_s42_20260519-080429/` |
| dc22 | `outputs/v4_phase4_g5_dc22_s42_20260519-084820/` |

logs: `outputs/v4_phase4_g{1..5}_*.log`。

## 1. 5-game 数字总表

| Game | rounds | total steps | change_rate | wins | actions used | GC alerts | rejected_goals |
|---|---:|---:|---:|---:|---:|---:|---:|
| ar25 | 2 | 172/200 | **87%** | 0 | 7 | 13 | 10 |
| bp35 | 2 | 72/200 | **85%** | 0 | 4 | 62 | 5 |
| cd82 | 2 | 200/200 | **79%** | 0 | 6 | 198 | 2 |
| cn04 | 2 | 150/200 | **74%** | 0 | 6 | 4 | 3 |
| dc22 | 2 | 200/200 | **84%** | 0 | 5 | 199 | 0 |
| **mean** | 2 | 159 | **82%** | **0** | 5.6 | 95 | 4 |

## 2. 对照表 (跨实验)

| Run | reasoning | proposer | goal_evaluator | mean change_rate (G_base) | wins (G_base) |
|---|---|:-:|:-:|---:|---:|
| `main` e07e7d1 | /no_think | ✅ | ❌ | 5-8% | 0/5 |
| SmolLM3 5×2×300 (det_goal v0) | /no_think | ✅ (+ click_targets + semantics + hard_rules) | ❌ | **64%** | 0/5 |
| det_goal v1 (Reflection 截断) | /think | ✅ + all | ✅ | 75% on ar25 | 0/1 |
| det_goal v2 round 0 (Refl 2048) | /think | ✅ + all | ✅ | 47% on ar25 | 0/1 |
| **v4+propose (本 Phase 4)** | /no_think | ✅ only | ✅ | **82%** | **0/5** |

**V4+propose 是所有非通关 baseline 里最高**(+18pp vs SmolLM3 5×2×300)。但仍 **0/5 wins**。

## 3. 为什么 change_rate 大幅提升但仍 0 通关

change_rate 衡量 "frame 有没有变化",不衡量"朝目标方向变化"。从 V3 → V4 提升的是 **action 多样性**(action_proposer 强 explore + ACTION1 spam 解除),不是 **目标对齐**。

要通关,核心问题不在"是否动",而在:
1. **选哪个 object 操作** (ar25 多 obj 时 model 不知道 click 哪个)
2. **多步规划朝目标** (model 单步看似随机选,但不能跨步骤 chain)
3. **识别 win-state precondition** (env-WIN trigger 是 hidden,model 不知道何时该停)

我加的反思机制 (goal_evaluator + force-reject + hallucination detect) 解决了"hypothesis 写得对吗",但**没解决"action 序列怎么 chain"**。

## 4. 各 game 特征

### ar25 (87%, 2 rounds 172 steps)
- 含 yellow / gray / maroon / purple / tan 多色 obj
- 7 actions 都用上
- rejected_goals=10 → Reflection 写了 10+ 个 hypothesis,反思活跃
- 但通关需要 yellow + gray 各自到目标位置,model 不能链式定位

### bp35 (85%, 2 rounds 72 steps,短游戏)
- 只 4 actions 合法(ACTION1/5 可能 illegal)
- GC alert 62 (高密度) → parser 在大部分 step evaluable 但 hypothesis 仍不能驱动 win

### cd82 (79%, 2 rounds 200 steps)
- 6 actions 均衡
- GC alert 198 → 几乎每步都触发反思评估
- 但 hypothesis 始终不对(rejected_goals 只 2,表示 Reflection 写的没被 reject,但也没 trigger win)

### cn04 (74%, 2 rounds 150 steps)
- 较低的 change_rate(74% vs mean 82%)
- 可能游戏机制不同 — 需要看实际 trace

### dc22 (84%, 2 rounds 200 steps)
- 5 actions used,均衡
- GC alert 199(密集)
- rejected_goals = 0 — Reflection hypothesis 从没被 reject (要么没生成,要么生成的 parser 都认可)

## 5. 5 game wins 都是 0 的诊断

可能根因(按可能性):

1. **goal_hypothesis 没法 chain 到 win**: hypothesis 是 "yellow to bottom edge" 类型;到了 bottom edge 没 win 也没 force-reject(因为 parser verdict=True + env != WIN 路径只在 hypothesis 真的 evaluable 时触发)
2. **win condition 不在 hypothesis space**: 某 game 的 win 条件是 "press ACTION5 三次" 或 "stack 三个 obj" — 我的 parser 模式集不覆盖
3. **action_proposer K=3 没覆盖通关 path**: K=3 候选有 untried + known-good + click_target,**没有 goal-directed plan**。需要先 chain 5-10 步才看到效果,proposer 只看 1 步

## 6. 决策建议 (给用户 morning)

按 decision_tree.md §"Phase 4":
- 0 通关 + change_rate 平均 82% > 64% baseline → **平 baseline +18pp,但没胜**
- 决策: "**报告 + 等用户决定**" — 不能称为 "结果非常好",但是有意义的 +18pp 改进

**用户提的 "如果结果非常好可以开始训练 5 个游戏"** 的判定:
- 严格解读 "好" = 通关 → ❌ 不好,不训练
- 宽解读 "好" = 显著超 baseline → ✅ +18pp,可以训练但风险有
- **保守建议: 暂不训练,先 debug 通关问题**

## 7. 跟进选项 (按 ROI 排)

1. **P0**: Trace 分析 — 看 V4+propose 在每个 game 哪些步骤接近 win,什么阻止了它。具体 grep `levels_gained > 0` 的 step,看 hypothesis 和 reasoning。
2. **P0**: 加 `--validate-hypothesis-schema strict` — 只接受 `move_to_row/col/center/stack` 这种有明确 target 的 hypothesis,禁掉 `align_any` 模糊的。可能 force Reflection 写出更可 chain 的 hypothesis。
3. **P1**: action_proposer 加 "goal-directed candidate" — 当 hypothesis 是 `move_to_row 63`,把 ACTION2 (DOWN) 显式塞进 K=3,确保 Action 看到对的方向。
4. **P2**: 升级模型试 SmolLM3-7B 或 Qwen3-4B,看是否 base 通关率提升。

## 8. 文件清单

```
docs/project/2026-05-19-v0-v4_clean_baseline/
└── phase4_5game_eval.md                (本文件)

outputs/v4_phase4_g{1..5}_*_s42_*/
├── summary.json
├── round_00/{trace.jsonl, knowledge_per_step, action_raw, refl_raw, GIF, step PNG}
└── round_01/...
outputs/v4_phase4_g{1..5}_*_s42.log
```
