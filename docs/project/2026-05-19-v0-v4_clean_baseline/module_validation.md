# v4 Phase 4 — Per-module re-validation (用户 2026-05-19 提的方法学修正)

> **背景**: 用户指出之前的"6 模块 validated"用的是 proxy metric (change_rate, count, diversity),不是每个模块自己的 PASS 定义。本文档重审 Phase 4 5 game trace,**用 module 真正的 PASS 定义**验证。
> **结论**: 之前的 8 modules 实际只 2 个能从 trace 客观验证(Module 3 / 5);Module 6 只有 proxy。Module 1 + 4 需要 human labels(见 `annotation_request.md`)。

## 0. 数据来源

- `outputs/v4_phase4_g{1..5}_*` × 5 game trace.jsonl + knowledge_per_step.jsonl + action_raw.txt
- 总 794 步across 5 game
- 分析脚本: `scripts/v4_phase4_module_validation.py`
- 原始结果: `docs/project/2026-05-19-v0-v4_clean_baseline/module_validation.json`

## 1. 模块 PASS 定义 vs 之前测的 proxy

| 模块 | 真 PASS 定义 | 之前测的 | Gap |
|---|---|---|---|
| 1. Goal Generation | hypothesis ≈ game 真实 win condition | "写了几个 + 不写 sentinel" | 没核对 hypothesis 是否对 |
| 2. Goal Recognition | parser=True ↔ env-WIN | T-GOAL 83% on synthetic | 没对照真实 WIN |
| **3. Action Selection** | action 方向 ≈ hypothesis 方向 | action diversity | **本 doc 重测** |
| 4. Force-reject Loop | reject 的都是真错的 hypothesis | reject 计数 | 没核对 reject 是否对 |
| **5. force_cot 一致性** | reasoning 提的 ACTION ≈ 实际 action | reasoning 长度 | **本 doc 重测** |
| **6. action_proposer 质量** | K=3 包含 goal-aligned 选项 | change_rate 提升 | **本 doc proxy 重测** |

## 2. Module 3: Action 方向对齐

### 验证方法

对每步:
1. 取 knowledge.goal_hypothesis → `parse_goal_hypothesis(text)` 解析
2. 如果 kind ∈ {move_to_row, move_to_col, move_to_center},算 hypothesis 目标方向 vs 当前 active 对象重心位置 → 应该选 up/down/left/right
3. 实际选的 action 是 ACTION1..4 → 实际方向
4. **对齐** = hypothesis 方向 == action 方向

### 结果

```
ar25  0/172 步有 directional hypothesis
bp35  0/72
cd82  0/200
cn04  0/150
dc22  0/200
total 0/794 = 0%
```

**❌ 致命发现**: **Reflection 在 Phase 4 5 game 全 794 步中,从未写过任何 directional hypothesis**(move_to_row / move_to_col / move_to_center)。

实际写的全是:
- "align the X objects" (kind=align_any,无方向)
- "match every X with X target" (kind=align_any,无方向)
- "stack X on Y" (偶尔出现,有方向但不在我的检测)
- 各种 vague 描述,parser 返回 None

→ **Action 拿到的 hypothesis 没有方向信号**,只有"哪个色应该对齐"。模型选 ACTION 时**根本没有"朝目标方向走"的依据**。

### 含义

之前我以为反思闭环工作(GC alert 触发、rejected_goals 累积),其实**hypothesis 本身缺一个关键属性: 可执行的方向**。这解释了为什么 change_rate 高(action 多样) 但 0/5 wins(没朝正确方向行动)。

## 3. Module 5: force_cot reasoning ↔ action 一致

### 验证方法

对每步:
1. 取 trace.reasoning 字段
2. 正则 `ACTION([1-7])` 找 **最后一次** ACTION 提及(通常是模型说"I'll pick ACTION X")
3. 与实际 trace.action 比对
4. **一致** = reasoning 末尾提的 ACTION_N == 实际 action

### 结果

| Game | total | has_reasoning | mentions_action | match | rate |
|---|---:|---:|---:|---:|---:|
| ar25 | 172 | 97 | 97 | 45 | **46%** |
| bp35 | 72 | 50 | 50 | 34 | **68%** |
| cd82 | 200 | 191 | 191 | 95 | **50%** |
| cn04 | 150 | 130 | 130 | 76 | **58%** |
| dc22 | 200 | 186 | 186 | 86 | **46%** |
| **mean** | **794** | **654** | **654** | **336** | **51%** |

**❌ 关键发现**: **51% 的 step 模型 reasoning 说选 ACTION X,但实际选了别的**。

### 含义

模型的"declared intent"跟"final action"严重脱节。可能源头:

1. **action_proposer letter-to-action mapping 错位**: prompt 说 "A=ACTION3, B=ACTION1, C=ACTION6",模型说 "I pick B because ACTION1...",但 model 其实没解析 letter,output "choice: A" → orchestrator resolve A → ACTION3。reasoning 提 ACTION1 但 action 是 ACTION3
2. **R3 forced explore 干预**: model 选 ACTION1,但 stuck detection 强制改 untried action
3. **anti-collapse 干预**: 最近 3 次都是 ACTION1 → 强制不同

总之: **trace 里的 action 不一定是 model "想"的 action**。这意味着我们的 "force_cot" prompt 改进**在 production 半数时间无效**(被 orchestrator 覆盖)。

## 4. Module 6 (proxy): K=3 candidate 出现率

### 验证方法 (proxy)

K=3 候选不在 trace 里直接保存。Proxy: 数 action_raw.txt 里 `choice: A/B/C` 出现频率(只有 `--propose on` 时模型才会输出这种格式)。

### 结果

| Game | total | letter_choice 出现 | rate |
|---|---:|---:|---:|
| ar25 | 172 | 77 | **45%** |
| bp35 | 72 | 56 | **78%** |
| cd82 | 200 | 185 | **92%** |
| cn04 | 150 | 118 | **79%** |
| dc22 | 200 | 182 | **91%** |
| total | 794 | 618 | **78%** |

**⚠️ 发现**: action_proposer 在 22% 步(平均)**没产生 K=3 候选**,prompt 走 non-proposer 格式。

可能原因:
- 当 outcome_log 没足够"untried" candidate (e.g., 早期 step)
- 当 ar25 只有少数 legal action,K=3 不够 distinct
- 当 click_targets 空(我们 --click-targets off 后,bandit 永远空)

→ **可能的修复**: 即使 K<3,也强制走 multi-choice 格式。

## 5. 综合诊断 (0/5 wins 的真因)

| 层 | 我以为它 work | 它实际 work 吗 | 证据 |
|---|---|---|---|
| Reflection 写 hypothesis | ✅ JSON output 完整 | **❌ 但没方向信号** | M3 0/794 directional |
| Parser 评 hypothesis | ✅ T-GOAL 83% | (未测 vs 真 win) | Module 2 待 human label |
| Action 选 action | ✅ diversity 高 | **❌ 51% 跟 reasoning 不一致** | M5 |
| K=3 候选 | ✅ +75pp change_rate | **⚠️ 22% 步没 K=3** | M6 |
| Force-reject | ✅ rejected_goals 累积 | (是否误伤) | Module 4 待 human label |

→ **真因 (新认知)**:
1. **Reflection 写的 hypothesis "align/match" 类没方向** → Action 不知道朝哪走
2. **半数 step Action 的"reasoning intent" 跟"实际 action"脱节** → 我们的 prompt 改进半数 invalid

修这两个之前,加再多模块都不会通关。

## 6. 跟进 (P0 排序)

| 优先级 | 行动 | 预期 |
|---|---|---|
| **P0** | `--validate-hypothesis-schema strict` 重跑 ar25 — 只接受 directional kind(move_to_row/col/center/stack),拒 align_any/match。 force Reflection 写方向 | 看 Action 方向对齐率是否大幅提升 |
| **P0** | 在 Action 集成层把 model "intent action" 跟 "final action" 都 log。如果 letter-mapping 是错位源,标准化映射 | M5 一致率应该 → 100% |
| **P1** | action_proposer 当 K < 3 时填 random untried 或显式 wait,确保始终 K=3 | 确认 |
| **P1** | annotation_request.md 准备好 module 1 + 4 的人工标注 | 用户标完决定下一步 |

## 7. 文件清单

```
docs/project/2026-05-19-v0-v4_clean_baseline/
├── module_validation.md                (本文件 — Module 3/5/6 重审)
├── module_validation.json              (machine-readable detailed metrics)
└── annotation_request.md               (待写 — Module 1 + 4 标注 spec)

scripts/v4_phase4_module_validation.py  (分析脚本)
```

## 8. 引用

- 用户提的方法学修正: 2026-05-19 chat
- Phase 4 数据: [`phase4_5game_eval.md`](./phase4_5game_eval.md)
- v4 final report: [`report.md`](./report.md)
