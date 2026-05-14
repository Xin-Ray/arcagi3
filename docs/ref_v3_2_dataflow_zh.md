# v3.2 数据流参考 —— 三个板块的真实 I/O

日期: 2026-05-14
状态: 🟢 reference
前置阅读: [`arch_v3_2_zh.md`](./arch_v3_2_zh.md)（设计），本文是**实测数据**展开

> 这份文档拿 2026-05-14 真跑的 `outputs/v3_2_ar25_3x30/round_00/step=6` 做参照（ar25 第 0 轮第 7 步，`matches_reasoning=YES` 的一步——Knowledge 真实生效那一刻），把三个板块的**输入来源、SYSTEM/USER prompt、实际输出**逐字展开。改 prompt / 加新检测器 / 排查模型行为前先读这个。

---

## 目录

1. [全局架构与一步内的执行顺序](#1-全局架构与一步内的执行顺序)
2. [板块 1 —— Perception（确定性感知 → 文字块）](#板块-1--perception-确定性感知--文字块)
3. [板块 2 —— Reflection Agent](#板块-2--reflection-agent)
4. [板块 3 —— Action / Play Agent](#板块-3--action--play-agent)
5. [关键文件索引](#关键文件索引)
6. [step 6 这一步暴露的问题](#step-6-这一步暴露的问题)

---

## 1. 全局架构与一步内的执行顺序

每一步 `env.step` 之后，orchestrator 按这个顺序执行：

```
┌──── 板块 1: PERCEPTION (确定性, 无 LLM) ──────────────────────────────┐
│   FrameDataRaw.frame[-1]  → 64x64 numpy.ndarray                       │
│       ↓ scipy.ndimage.label                                            │
│   ObjectRecord[]  (颜色/bbox/size/center)                              │
│       ↓ temporal_classifier                                            │
│   Layer per object  (STATIC / ACTIVE / TEXTURE / CANDIDATE)            │
│       ↓ Hungarian align_objects(prev_active, current_active)           │
│   Match[]  (moved / appeared / disappeared / recolored + delta)        │
│       ↓ ObjectMemory.update + OutcomeLog.record                        │
│   per-uid 历史 + per-action 统计                                       │
│       ↓ prompts_v3.build_play_user_prompt(...)                         │
│   8 个文字块  [STATUS] [ACTIVE] [TEXTURE] [ACTION] [UNTRIED]           │
│              [HISTORY] [GOAL] [ASK]                                    │
└────────────────────────────────────────────────────────────────────────┘
                        ↓
┌──── 板块 3: ACTION AGENT (1 次 LLM 调用) ─────────────────────────────┐
│   USER prompt = [REFLECTION ALERT]? + [KNOWLEDGE] + 板块1输出 + [ASK] │
│       ↓ Qwen2.5-VL-3B (text-only, 4-bit, max_new_tokens=96)            │
│   "reasoning: ...\naction: ACTIONx"                                    │
│       ↓ parse_reasoning_and_action + _coerce_action                    │
│   (GameAction, reasoning_text)                                         │
└────────────────────────────────────────────────────────────────────────┘
                        ↓
                  env.step(action)
                        ↓
┌──── 板块 1 续: outcome 计算 ──────────────────────────────────────────┐
│   compute_primary_change(prev_grid, curr_grid)                         │
│       → frame_changed, primary_direction, primary_distance, deltas     │
│   compute_matches_reasoning(reasoning, ...)                            │
│       → YES / PARTIAL / NO / N/A  (orchestrator 端规则判定)            │
│   ActionAgent.no_op_streak() / state_revisit_count() / recent_steps()  │
└────────────────────────────────────────────────────────────────────────┘
                        ↓
┌──── 板块 2: REFLECTION AGENT (1 次 LLM 调用) ─────────────────────────┐
│   USER prompt = [CURRENT KNOWLEDGE] + StepSummary.render() + [ASK]    │
│       ↓ Qwen2.5-VL-3B (text-only, max_new_tokens=250)                  │
│   raw JSON 字符串(可能带 ```json fence)                                │
│       ↓ parse_reflection_output (容忍 fence + 前缀散文)                │
│   delta dict {action_semantics_update, rules_append, current_alert,...}│
│       ↓ Knowledge.merged_with_delta(delta)                             │
│   新 Knowledge(下一步 Action Agent 看到)                               │
└────────────────────────────────────────────────────────────────────────┘
```

每步 **2 次 LLM 调用**（Action + Reflection），实测在 4-bit Qwen2.5-VL-3B 上约 **8-9 秒**。

---

## 板块 1 —— Perception (确定性感知 → 文字块)

**核心原则**：视觉理解全部由 scipy 完成，**LLM 永不看像素**。LLM 只看预先算好的文字块。

### 1.1 数据流 + 函数链

| # | 函数 | 文件 | 输入 | 输出 |
|---|---|---|---|---|
| 1 | `extract_objects(grid)` | `arc_agent/object_extractor.py` | 64×64 int 网格 | `list[ObjectRecord]`（每个对象 color/bbox/size/center） |
| 2 | `update_history` + `classify_frame` | `arc_agent/temporal_classifier.py` | 多帧观察 | `dict[obj_id → Layer]`（STATIC / ACTIVE / TEXTURE / CANDIDATE） |
| 3 | `align_objects(prev_active, curr_active)` | `arc_agent/object_aligner.py` | 上一帧 + 当前帧 ACTIVE 对象 | `list[Match]`（每个匹配带 type + delta） |
| 4 | `ObjectMemory.update` | `arc_agent/object_tracker.py` | matches | UID 跨帧持久化轨迹 |
| 5 | `OutcomeLog.record(StepOutcome)` | `arc_agent/action_inference.py` | (action, changed, direction, distance) | per-action 累积统计 |
| 6 | `build_play_user_prompt` | `arc_agent/prompts_v3.py` | 以上全部 + Knowledge | 8 个 markdown 风格文字块 |

### 1.2 实际产出的文字（step 6 真实重建）

> 完整文件：`outputs/v3_2_ar25_3x30/_reconstructed_step6_action_user_prompt.txt`

```text
[STATUS]                              # _format_status(), prompts_v3.py:95
  step: 6 / 30
  level: 1 / 4
  game state: NOT_FINISHED
  legal actions: ACTION1 (no params), ACTION6 (x, y in 0..63), ACTION7 (no params)

[ACTIVE]                              # _format_active_block(), 读 ObjectMemory.alive_tracked()
  (no active objects yet — every cell looks static)
                                      # ↑ ar25 step 6 没移动对象，纯文本占位

[TEXTURE]                             # _format_texture_block(), 读 layer_by_id 中 TEXTURE 的
  (none)

[ACTION effects observed]             # render_action_block(), 读 OutcomeLog (action_inference.py)
  ACTION1: tried 2x: 2x moved UP 3 cell(s)        ← 板块 1 真正的成果:
  ACTION6: total tried 4x: all no-op;             ← 把 "ACTION1 之前两次都把对象上移 3 格"
           recent 4x: 4 no-op (100%)                这种动作语义,通过 perception 算出来送给 LLM
  ACTION7: UNTRIED

[UNTRIED legal actions]               # OutcomeLog.untried(legal_actions)
  ACTION7

[HISTORY last 5 steps]                # render_history_tail(log, n=5)
  step 1: ACTION6 -> no-op
  step 2: ACTION6 -> no-op
  step 3: ACTION1 -> CHANGED (UP 3)              ← 方向/距离都是 align_objects + StepOutcome 算的
  step 4: ACTION6 -> no-op
  step 5: ACTION1 -> CHANGED (UP 3)
```

### 1.3 设计意图

板块 1 把"模型本来要从原始 64×64 像素里推理出来的所有 spatial/temporal 信号"压缩成 LLM 直接能消费的人话。两个最重要的输出：

- **`[ACTION effects observed]`** —— per-action 历史聚合。LLM 不用扫 history 自己数 ACTION1 出现多少次往哪边走，板块 1 直接给 "tried 2x: 2x moved UP 3 cell(s)"。
- **`[UNTRIED]`** —— 显式列出还没试过的合法 action。LLM 不用自己做集合差。

📌 **设计权衡**：板块 1 的预聚合让小模型（3B）可用，但也让 LLM 失去对原始信号的访问权——`[ACTIVE]` 里说 "no active objects yet"，LLM 就不知道画面其实有 100 个静态像素。这就是 ar25 上 reasoning 偶尔幻觉的根源。

---

## 板块 2 —— Reflection Agent

**职责**：每步 `env.step` 之后调用一次，比较 "Action 说什么" vs "实际发生什么"，输出 incremental delta 更新 Knowledge。

### 2.1 SYSTEM prompt

文件：`arc_agent/prompts_v3_2.py:76-142`（常量，每步一样）

完整 1700+ 字符，关键段：

```text
You are the Reflection Agent for an in-episode learning loop.

After EACH step you see:
  - The current KNOWLEDGE
  - The Action Agent's REASONING (what it expected)
  - This step's actual OUTCOME (action, frame_changed, primary_direction,
    no_op_streak, state_revisit_count, matches_reasoning verdict)
  - The last 3 steps

Your two main jobs:

  (A) UPDATE KNOWLEDGE -- be EAGER, not cautious.
      - action_semantics[ACTION_X]: as soon as you see ONE specific
        effect, WRITE the entry. Empty action_semantics after 5+ steps
        is a FAILURE.
        Concrete rule: if primary_direction is not null, the entry MUST
        name the direction + distance + what kind of object moved.
      - failed_strategies: append when a strategy or coord region has
        clearly failed 3+ times. Keep these high-level.
      - goal_hypothesis_update: ONLY when you have a real guess. If you
        don't, set this to null (NOT the literal string "unknown").

  (B) WRITE current_alert when the Action Agent's mental model is wrong.
      Trigger an alert when ANY of these holds:
        - matches_reasoning == "NO"
        - no_op_streak >= 3
        - state_revisit_count >= 3
        - same action chosen 5+ times in a row with no progress

Output STRICT JSON only -- no prose, no markdown fences:
{
  "action_semantics_update": {"ACTION3": "..."},
  "goal_hypothesis_update": "..." or null,
  "goal_confidence_update": "low|medium|high" or null,
  "rules_append": [...],
  "failed_strategies_append": [...],
  "current_alert": ""
}
```

### 2.2 USER prompt at step 6（真实重建）

来源：`build_reflection_user_prompt(knowledge, step_summary)`（`prompts_v3_2.py:188`）

每个块的数据来源标在右侧：

```text
[CURRENT KNOWLEDGE before this step]                  ← knowledge_per_step.jsonl[step=5]
  rounds: 0 played, 0 won                                (step 5 reflection 后的快照)
  action_semantics:
    ACTION1: moves an active object UP by 3 cells
  goal_hypothesis (low): unknown
  rules:
    - ACTION6 has no effect on any tested coord.
  failed_strategies (do NOT repeat):
    - ACTION6 anywhere in the right half.

[ACTION AGENT'S REASONING]                            ← trace.jsonl[step=6].reasoning
  "Knowledge suggests ACTION1 moves the player up; goal is the top"

[ACTION AGENT'S CHOICE]                               ← trace.jsonl[step=6].action
  ACTION1

[ACTUAL OUTCOME]                                      ← orchestrator: compute_primary_change(
  frame_changed: True                                       prev_grid, curr_grid)
  primary_direction: UP (distance=3)                  ← align_objects 算出的最强 delta
  object delta: (frame changed but no tracked-object movement)
  no_op_streak: 0                                     ← ActionAgent.no_op_streak()
  state_revisit_count: 0                              ← ActionAgent.state_revisit_count()
  matches_reasoning: YES                              ← compute_matches_reasoning(
                                                           reasoning="...moves the player up",
                                                           frame_changed=True,
                                                           primary_direction="UP")
[LAST STEPS — short context]                          ← ActionAgent.recent_step_records(n=3)
  ACTION1 -> CHANGED (UP)
  ACTION6 -> no-op
  ACTION1 -> CHANGED (UP)

[ASK]
Output STRICT JSON with all six fields. Use {}/[]/""/null for "no update".
Write current_alert ONLY if Action's mental model is wrong, the agent
is stuck, or revisits keep increasing.
```

完整文件：`outputs/v3_2_ar25_3x30/_reconstructed_step6_reflection_user_prompt.txt`

### 2.3 真实 OUTPUT at step 6

来源：`outputs/v3_2_ar25_3x30/round_00/reflection_raw.txt`（带 ```json fence，被 `parse_reflection_output` 自动剥）：

```json
{
  "action_semantics_update": {"ACTION1": "moves an active object UP by 3 cells"},
  "goal_hypothesis_update": "unknown",
  "goal_confidence_update": "low",
  "rules_append": ["ACTION6 has no effect on any tested coord."],
  "failed_strategies_append": ["ACTION6 anywhere in the right half."],
  "current_alert": ""
}
```

### 2.4 输出处理

`Knowledge.merged_with_delta(delta)`（`arc_agent/knowledge.py:147`）做合并：

- `action_semantics_update` —— per-key 覆盖
- `goal_hypothesis_update` —— 非 null 替换（**但这次模型违规写了字符串 "unknown"**，被照样替换）
- `rules_append` / `failed_strategies_append` —— append + dedup + cap (10 / 5)
- `current_alert` —— 覆盖（下一步 Action 在 prompt 顶部看到）

合并后写入 `knowledge_per_step.jsonl[step=6]`，下一步 Action prompt 直接用。

---

## 板块 3 —— Action / Play Agent

**职责**：每步 `env.step` 之前调用一次，读 Knowledge + 板块 1 输出，输出 `reasoning + action` 两行。

### 3.1 SYSTEM prompt

文件：`arc_agent/prompts_v3_2.py:33-73`（常量）

```text
You are the Action Agent for a turn-based 64x64 grid game.

The Reflection Agent has provided KNOWLEDGE accumulated across previous
rounds (and the previous steps of this round):
  - action_semantics: what each ACTION does in this game (when known)
  - goal_hypothesis: the most likely goal so far
  - rules: patterns observed across previous rounds
  - failed_strategies: strategies that were tried and did NOT work

Trust the KNOWLEDGE block. Do NOT re-explore things already documented
as failed_strategies. If a [REFLECTION ALERT] block is present at the
top of the prompt, the Reflection Agent has flagged that your previous
mental model was wrong -- read it FIRST and change behavior accordingly.

The user prompt also gives you v3 enriched context:
  [STATUS]   step, level, legal actions (with parameter signatures)
  [ACTIVE]   tracked objects with movement history
  [TEXTURE]  static cells filtered out
  [ACTION]   observed effects of each action THIS round
  [UNTRIED]  legal actions you have not tried yet
  [HISTORY]  the last 5 (action, frame_changed) tuples
  [GOAL]     current hypothesis from v3 (the [KNOWLEDGE] goal is preferred)

OUTPUT FORMAT (strict, two lines, no JSON, no markdown):
  reasoning: <one sentence explaining your choice; mention the expected
             effect so the Reflection Agent can judge it>
  action: ACTION1..ACTION5 / ACTION7  (no params)
          ACTION6 <x> <y>             (x, y in 0..63 -- ACTION6 ONLY)

Valid:
  reasoning: knowledge says ACTION1 moves the player up; goal is the top
  action: ACTION1

INVALID:
  Do NOT add coordinates to ACTION1..5 or ACTION7.
  Do NOT output JSON. Just two plain lines: reasoning and action.
```

### 3.2 USER prompt at step 6（真实重建）

来源：`build_action_user_prompt(knowledge, +v3 kwargs)`（`prompts_v3_2.py:131`）

```text
[KNOWLEDGE - accumulated across rounds]               ← knowledge.render(),
  rounds: 0 played, 0 won                                knowledge = step 5 reflection 后状态
  action_semantics:
    ACTION1: moves an active object UP by 3 cells
  goal_hypothesis (low): unknown
  rules:
    - ACTION6 has no effect on any tested coord.
  failed_strategies (do NOT repeat):
    - ACTION6 anywhere in the right half.

# (这里没 [REFLECTION ALERT] 块,因为 current_alert == "" )

[STATUS]                                              ← 板块 1 输出(见 §1.2)
  step: 6 / 30
  level: 1 / 4
  game state: NOT_FINISHED
  legal actions: ACTION1 (no params), ACTION6 (x, y in 0..63), ACTION7 (no params)

[ACTIVE]                                              ← 板块 1
  (no active objects yet — every cell looks static)

[TEXTURE]                                             ← 板块 1
  (none)

[ACTION effects observed]                             ← 板块 1 (OutcomeLog)
  ACTION1: tried 2x: 2x moved UP 3 cell(s)
  ACTION6: total tried 4x: all no-op; recent 4x: 4 no-op (100%)
  ACTION7: UNTRIED

[UNTRIED legal actions]                               ← 板块 1
  ACTION7

[HISTORY last 5 steps]                                ← 板块 1
  step 1: ACTION6 -> no-op
  step 2: ACTION6 -> no-op
  step 3: ACTION1 -> CHANGED (UP 3)
  step 4: ACTION6 -> no-op
  step 5: ACTION1 -> CHANGED (UP 3)

[GOAL hypothesis]                                     ← knowledge.goal_hypothesis,
  unknown                                                这里跟 [KNOWLEDGE] 块重复显示
  confidence: low

[ASK]
  Output TWO lines (no JSON, no markdown):
    reasoning: <one sentence, mention the expected effect>
    action: <ACTION1..ACTION7 -- only ACTION6 takes x y>
```

完整文件：`outputs/v3_2_ar25_3x30/_reconstructed_step6_action_user_prompt.txt`（1115 字符）

### 3.3 真实 OUTPUT at step 6

来源：Qwen 真实返回，记录在 `trace.jsonl[step=6]` 的 `reasoning` + `action`：

```text
reasoning: Knowledge suggests ACTION1 moves the player up; goal is the top
action: ACTION1
```

### 3.4 输出处理

`parse_reasoning_and_action(text)`（`agents/action_agent.py:55`）—— 正则提取两行：

```python
reasoning = "Knowledge suggests ACTION1 moves the player up; goal is the top"
action_text = "ACTION1"
```

`_coerce_action(action_text, latest, ...)`（`agents/action_agent.py:284`）—— 把字符串映射成 `GameAction`：

1. `_ACTION_RE` 匹配 `ACTION([1-7])` → 拿到 `1`
2. `GameAction[f"ACTION1"]` → `GameAction.ACTION1`
3. 验证 `action.value in latest.available_actions` → 合法
4. `action.is_complex()` → False（ACTION1 不需要坐标），直接返回
5. orchestrator 调 `env.step(action, data=action.action_data.model_dump())`

---

## 关键文件索引

按板块归档，所有路径相对于 repo root：

### 板块 1 (Perception)
| 文件 | 角色 |
|---|---|
| `arc_agent/object_extractor.py` | `extract_objects` —— scipy.ndimage.label 找连通分量 |
| `arc_agent/temporal_classifier.py` | `classify_frame` —— STATIC/ACTIVE/TEXTURE 分层 |
| `arc_agent/object_aligner.py` | `align_objects` —— Hungarian 跨帧匹配 |
| `arc_agent/object_tracker.py` | `ObjectMemory` —— UID 持久化 |
| `arc_agent/action_inference.py` | `OutcomeLog`, `StepOutcome`, `render_action_block` |
| `arc_agent/prompts_v3.py` | 8 个 `_format_*_block` + `build_play_user_prompt` |

### 板块 2 (Reflection)
| 文件 | 角色 |
|---|---|
| `arc_agent/prompts_v3_2.py:76` | `REFLECTION_SYSTEM` 常量 |
| `arc_agent/prompts_v3_2.py:188` | `build_reflection_user_prompt` |
| `arc_agent/step_summary.py` | `StepSummary.render()` + `compute_matches_reasoning` |
| `arc_agent/agents/reflection_agent.py` | `ReflectionAgent.reflect_after_step` + `parse_reflection_output`（剥 ```json fence） |
| `arc_agent/knowledge.py` | `Knowledge.merged_with_delta` —— delta 合并规则 |

### 板块 3 (Action)
| 文件 | 角色 |
|---|---|
| `arc_agent/prompts_v3_2.py:33` | `ACTION_SYSTEM` 常量 |
| `arc_agent/prompts_v3_2.py:131` | `build_action_user_prompt`（包 v3 prompts + alert + knowledge） |
| `arc_agent/agents/action_agent.py:55` | `parse_reasoning_and_action` |
| `arc_agent/agents/action_agent.py:284` | `_coerce_action` —— 字符串 → GameAction |
| `arc_agent/agents/action_agent.py:113` | `ActionAgent.choose` —— 串联 perception + prompt + generate |

### Orchestrator
| 文件 | 角色 |
|---|---|
| `scripts/run_v3_multi_round.py:_compute_primary_change` | (s_t, s_{t+1}) → primary_direction/distance |
| `scripts/run_v3_multi_round.py:run_one_game` | N round × M step 主循环；每步串板块 3 → env → 板块 1 outcome → 板块 2 |

---

## step 6 这一步暴露的问题

这一步 `matches_reasoning=YES`，是 Knowledge 系统**正确生效**的样本。但通看整个 3 round × 30 step run 后，能从这一步的输出看出**两个潜在风险**：

### 问题 1：`goal_hypothesis_update: "unknown"` 字符串

SYSTEM prompt 明确写：

> goal_hypothesis_update: ONLY when you have a real guess. If you don't, set this to null (NOT the literal string "unknown"). Never write "unknown" / "none" / "" as a string -- use null.

但 step 6 输出依然写了字符串 `"unknown"`。这条规则被 Qwen-3B 忽略。**`Knowledge.merged_with_delta` 默认相信非 null 输入会替换原值**，所以 `goal_hypothesis` 整个 run 被永远卡在 `"unknown"` 字符串上。

**修复方向**：
- 在 `merged_with_delta` 加白名单（拒绝 `"unknown" / "none" / ""` 这种 sentinel 字符串当作 hypothesis）
- 或在 prompt 加 few-shot 反例

### 问题 2：板块 1 给的 `[UNTRIED]: ACTION7` 没被采纳

板块 1 明确告诉 Action Agent `ACTION7` 还没试过（step 6 时只试过 1 次，在 step 0 那次也是 no-op）。SYSTEM prompt 的优先级是：

> Your priorities, in order:
>   1. If [UNTRIED] is non-empty, try one of those actions

但 step 6 选 ACTION1 而不是 ACTION7。**Knowledge 的 `action_semantics[ACTION1]: "moves up by 3"` 把 `[UNTRIED]` 的优先级盖住了**。在 step 6 这是合理的（明确有效的 action 比未知 action 强），但在 round 1 末尾卡 16 步那种情况下，agent 应该回退到 `[UNTRIED]`，结果它一直 spam ACTION6。

**修复方向**：
- ActionAgent 加 orchestrator-level 强制：连续 N 步 no-op + state_revisit ≥ K → 强制 untried action（不依赖 Qwen 推理）
- 或加 `last_resort: try untried` 块，alert 触发时显示

---

## 想看其他步骤怎么办

把上面的重建 Python snippet 改一下索引就行（`outputs/v3_2_ar25_3x30/round_<R>/`）。或者直接读这些文件：

- `outputs/v3_2_ar25_3x30/round_<R>/trace.jsonl` —— 每步 reasoning / action / outcome / matches_reasoning / reflection_delta
- `outputs/v3_2_ar25_3x30/round_<R>/knowledge_per_step.jsonl` —— 每步 reflection 合并后的 Knowledge 快照
- `outputs/v3_2_ar25_3x30/round_<R>/reflection_raw.txt` —— 每步 Reflection 的完整原始返回（带 ```json fence）
- `outputs/v3_2_ar25_3x30/round_<R>/step_*.png` —— 每步合成图（grid + reasoning + reflection delta + alert，旧 512×286；下次 run 会用 896×552）
- `outputs/v3_2_ar25_3x30/knowledge_history.jsonl` —— 每 round 结束时的 Knowledge 快照（看跨 round 学习曲线）

---

*这份文档是 2026-05-14 实测后写的。当 prompts_v3_2.py 或 build_action_user_prompt 改动时，重新跑一次并把 §2.2 / §3.2 的样例更新一下，免得它跟代码漂移。*
