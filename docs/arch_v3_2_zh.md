# ARCHITECTURE v3.2 —— 双 Agent 分离 + 多轮 Knowledge 累积

日期: 2026-05-14
状态: 设计 → 实施(本文取代 v3.1 的 P2/P3 计划;P0-A、P0-B、P1 的改动**保留**)
前置阅读: [`arch_v3_zh.md`](./arch_v3_zh.md)、[`ref_v3_prompt_zh.md`](./ref_v3_prompt_zh.md)

---

## 0. 为什么要拆成两个 agent(2026-05-14 决定)

v3 单 agent 的 L6 实测发现:**一个 Qwen 同时担**

1. **看图** —— 已经由 scipy 担任 ✅
2. **总结历史**(我哪些 action 试过,效果如何)—— 仍由模型隐含完成 ❌
3. **推断目标 / 规则 / 动作语义** —— 仍隐含 ❌
4. **决策下一步 action + 写 reasoning chain** ❌

后果(ar25 L6 实测):
- 总结历史 = 隐含 → 模型在 80 步里**还是把 same state 访问 66 次**(没建立"这个状态我去过"的概念)
- 推断规则 = 隐含 → 模型每步重新决策,不积累
- **跨 episode 无任何持续学习**(reset 全清空)

v3.2 的核心:**用两个独立 agent + 一个共享 `Knowledge` 对象,让"学到的东西"跨 round 持久化**。

---

## 1. 概念表(快速查阅)

| 概念 | 含义 | 实现 |
|---|---|---|
| **Round / Episode** | 一次完整的 game play(reset → ... → WIN / 超 max_actions) | 调度 by `run_v3_multi_round.py` |
| **Knowledge** | 跨 round 持久的"学到的东西":action_semantics、goal_hypothesis、rules、failed_strategies、rounds_played/won | `arc_agent/knowledge.py:Knowledge` |
| **Reflection Agent** | **每步 env.step 后召唤一次**(2026-05-14 修订:per-step,不是 per-round),读本步 outcome + 当前 Knowledge,输出 incremental delta | `arc_agent/agents/reflection_agent.py` |
| **Action Agent** | 每步召唤,读 frame 状态 + Knowledge,输出 action + reasoning chain | `arc_agent/agents/action_agent.py`(由 v3 TextAgent 改造)|
| **Message passing** | 两 agent 之间的唯一通讯通道:Knowledge dict | 数据结构,不是网络 |
| **Reasoning chain** | Action agent 在 action token 前给的一句话推理(`reasoning: ... action: ACTIONx`) | 进 trace.jsonl 的 response_raw,人审用 |
| **action_semantics** | dict[str, str],例如 `{"ACTION1": "moves red object up 1 cell"}` | Knowledge 的核心字段 |
| **failed_strategies** | 已经试过且失败的 high-level 策略,**不再重复** | Knowledge 的负面记忆 |
| **rounds_won** | 该游戏过关次数,作 stop 条件 | Knowledge 字段 |
| **multi-round budget** | 默认 3-5 round,每 round 最多 80 step | run_v3_multi_round 配置 |

---

## 2. 系统架构图(2026-05-14 修订:**Reflection 每步执行**)

```
┌──────────────────────────────────────────────────────────────────────────┐
│  MULTI-ROUND ORCHESTRATOR (scripts/run_v3_multi_round.py)                  │
│                                                                            │
│  knowledge = Knowledge.empty(game_id)                                      │
│                                                                            │
│  for round_idx in range(n_rounds):                                         │
│    env = arc.make(game_id, scorecard_id)                                   │
│    action_agent.reset_episode_state()    # 每 round 清 ObjectMemory      │
│    # 但 knowledge 不 reset !! 跨 round 持续累积                            │
│                                                                            │
│    ┌─ episode loop (≤80 steps) ──────────────────────────────────────┐   │
│    │                                                                  │   │
│    │   frame_t = env.reset() (round 第 1 步) or env.step(prev_action)│   │
│    │            ↓                                                     │   │
│    │   PERCEPTION (scipy + Hungarian + temporal_classifier)           │   │
│    │            ↓                                                     │   │
│    │   ┌─ Action Agent ────────────────────────────────────────┐    │   │
│    │   │   input:  perception(frame_t) + knowledge + log       │    │   │
│    │   │   output: reasoning + ACTION token                     │    │   │
│    │   │   1 LLM call                                            │    │   │
│    │   └─────────────────────────────────────────────────────────┘    │   │
│    │            ↓ action                                              │   │
│    │   frame_{t+1} = env.step(action)                                 │   │
│    │   record StepOutcome → outcome_log                               │   │
│    │            ↓                                                     │   │
│    │   ┌─ Reflection Agent (PER STEP, after env.step) ────────┐     │   │
│    │   │   input:                                                │     │   │
│    │   │     - knowledge (current)                                │     │   │
│    │   │     - this step's tuple: (action, reasoning,            │     │   │
│    │   │       frame_t/frame_{t+1} diff, ObjectMemory delta,     │     │   │
│    │   │       changed/no-op)                                    │     │   │
│    │   │     - last 3 step records (short-term context)          │     │   │
│    │   │   output: incremental knowledge delta (JSON, small):    │     │   │
│    │   │     - update action_semantics[ACTION_x] if confident    │     │   │
│    │   │     - refine goal_hypothesis only if strong evidence    │     │   │
│    │   │     - append failed_strategies if applicable            │     │   │
│    │   │   1 LLM call (text-only, prompt < 1k tokens, ~0.5-1s)   │     │   │
│    │   └─────────────────────────────────────────────────────────┘     │   │
│    │            ↓                                                     │   │
│    │   knowledge ← merge(knowledge, delta)                            │   │
│    │   knowledge snapshot appended to round_<k>/knowledge_per_step.jsonl │   │
│    │                                                                  │   │
│    └──────────────────────────────────────────────────────────────────┘   │
│                                                                            │
│    play.gif + trace.jsonl saved for round_idx                              │
│    end-of-round: knowledge.rounds_played++; if WIN: rounds_won++           │
│    knowledge.round_history.append(<one-line summary>)                     │
│                                                                            │
│  multi-round report.md + per-round GIFs                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

**关键变化**(vs 早先 per-round 设计):

| 维度 | 早先 per-round | 现在 **per-step** |
|---|---|---|
| Reflection 调用频率 | 每 round 1 次 | 每步 1 次 |
| Knowledge 更新粒度 | 整个 episode 后才更新 | **每步累积** |
| Knowledge 在 episode 内可变 | ❌ 永不变 | ✅ 实时刷新 |
| Action Agent 第 t 步看到的 knowledge | 上一 round 的快照 | **本 round 前 t-1 步累积** |
| 每步 LLM 调用数 | 1(只 Action) | **2**(Action + Reflection) |
| 每步成本 | ~0.7s | ~1.5-2s |
| 5 round × 80 step 总耗时 | ~5 min | **~13-17 min** |
| Episode 内学习 | ❌ 不可能 | ✅ 可能 |

**为什么改成 per-step**(2026-05-14 用户决定):

跨 round 反思**信号丢失**问题 —— 一个 episode 80 步可能有 5-10 个 frame-changing 事件,等到 episode 结束再总结时:
- 模型记不清"step 12 的 ACTION3 让 yellow L 上移 3 格"的具体事件
- Knowledge 更新只能基于 episode-level 聚合(平均、分布),信号被稀释
- Action Agent 在 episode 内**完全不能利用本 episode 的发现**

per-step reflection 解决:
- Reflection 看到的是**最新一步的精确变化**(scipy diff、Hungarian delta、ObjectMemory snapshot),信号最强
- 每次只更新 knowledge 的一小部分(可能只是改 `action_semantics[ACTION3]`),不会爆炸
- 第 t 步的 Action Agent 看到的是**前 t-1 步累积的 knowledge**,边玩边学

---

## 3. `Knowledge` 数据结构

```python
@dataclass
class Knowledge:
    """The shared message between Action Agent and Reflection Agent.
    Persisted across rounds within one game; reset only when a new game starts."""

    # 玩了几轮 / 赢了几轮
    rounds_played: int = 0
    rounds_won: int = 0

    # 每个 ACTION 在这个游戏中的效果(reflection 持续修正)
    # 例:{"ACTION1": "moves the red 1x1 object up 1 cell", "ACTION6": "click; nothing observable yet"}
    action_semantics: dict[str, str] = field(default_factory=dict)

    # 关于游戏的一句话目标假设(从经验里推)
    goal_hypothesis: str = ""
    goal_confidence: str = "low"   # low | medium | high

    # 已观察到的规则(≤10 条短句)
    rules: list[str] = field(default_factory=list)

    # 已经试过但失败的策略(避免重复)
    failed_strategies: list[str] = field(default_factory=list)

    # 历史 round 的关键结果摘要(每 round 一行)
    round_history: list[str] = field(default_factory=list)
    # 例:["round 0: action distrib {A1:60, A6:20}; no progress; max_revisit=66",
    #     "round 1: tried different click coords; 1 level completed at step 47", ...]

    # 2026-05-14 新增:Reflection 写给 Action 的实时警报
    # 例:"step 12 你说 ACTION1 让 yellow 上移,但 yellow 没动 — action_semantics 可能错"
    # 例:"连续 5 步 no-op,reasoning 没改变 — 换一个完全不同 action category"
    # 每步 Reflection 决定要不要写;为空则下一步 Action prompt 不显示
    current_alert: str = ""

    def render(self) -> str:
        """Turn the dict into a text block for the Action Agent's prompt."""
        ...
```

### 3.1 持久化 / 序列化

- `Knowledge` 可以 `to_dict()` / `from_dict()` 用 JSON 存档
- 每 round 结束写一份到 `outputs/<run>/knowledge_history.jsonl`,append-only
- Round k 启动时,先尝试 load 最后一行;load 不到则用 `Knowledge.empty()`

---

## 4. Reflection Agent

### 4.1 何时召唤(2026-05-14 修订)

- **每步 env.step 之后,正好一次**(per-step,**不是 per-round**)
- 输入是**当前这一步**的具体观察(action / diff / delta),**不是** episode 摘要
- 输出是 **delta**(只更新 knowledge 的一小部分),不是完整 Knowledge
- Knowledge 跟 OutcomeLog 一样**在 episode 内被持续 mutate**;每步追加 Action Agent 看到的内容

设计权衡:成本翻倍(每步 2 LLM 调用 vs 1),但换来 **episode 内可学习**。

### 4.2 SYSTEM prompt(常量)

```
You are the Reflection Agent for an in-episode learning loop.

After EACH step, you see:
  - The current KNOWLEDGE (what you've learned across all prior steps
    of this round + all prior rounds of this game)
  - The Action Agent's REASONING from this step ("what it expected")
  - This step's actual OUTCOME: action taken, what changed in the grid,
    what moved in ObjectMemory, and whether it was a no-op
  - The last 3 steps for short context

Your two main jobs:

  (A) UPDATE KNOWLEDGE based on observation.
      - action_semantics[ACTION_X]: confirm or correct based on outcome
      - rules: append new patterns
      - failed_strategies: append patterns that failed 3+ times
      - goal_hypothesis: refine if new evidence

  (B) COMPARE REASONING vs OUTCOME — flag mismatches.
      - If reasoning predicted X but outcome was Y, the Action Agent's
        mental model is wrong. Write a SHORT, SPECIFIC current_alert
        that the next step's Action Agent will see at the TOP of its
        prompt.
      - Also write a current_alert when:
          * Stuck (no-op streak >= 3 OR same-state revisit >= 3)
          * Action Agent has been doing the same thing for 5+ steps
            with no progress
          * Reasoning text is generic / non-committal
      - Otherwise leave current_alert empty.

Output strict JSON, no prose, no markdown fences:

{
  "action_semantics_update": {"ACTION3": "..."},   // {} if no update
  "goal_hypothesis_update": "..." | null,
  "goal_confidence_update": "low|medium|high" | null,
  "rules_append": ["..."],                          // [] if none
  "failed_strategies_append": ["..."],
  "current_alert": ""                               // "" if no alert
}
```

### 4.3 USER prompt(每步动态)

```
[CURRENT KNOWLEDGE before this step]
<knowledge.render()>

[LAST 3 STEPS — short context]
  step t-3: ACTION1 -> CHANGED (yellow obj_002 moved UP 3 cells)
  step t-2: ACTION1 -> no-op
  step t-1: ACTION3 -> CHANGED (yellow obj_002 moved RIGHT 1 cell)

[ACTION AGENT'S REASONING at step t]
  "I'll try ACTION6 at (12, 30) — it should place a red marker
   based on action_semantics."

[ACTION AGENT'S CHOICE]
  ACTION6 (12, 30)

[ACTUAL OUTCOME at step t]
  frame_changed: True
  object delta:
    obj_005 (red 1x1, size=1) APPEARED at (12, 30)
    no other active object moved
  matches reasoning? PARTIAL  (auto-computed by orchestrator)
  no_op_streak: 0
  state_revisit_count (this frame_hash): 1

[ASK]
Output the incremental DELTA as strict JSON, AND write current_alert
if reasoning conflicted with outcome OR the agent is stuck.
```

### 4.3b reasoning vs outcome 自动预判(给 reflection 减负)

orchestrator 在调 Reflection 之前先对比 reasoning 跟 outcome,把判定结果一并塞进 prompt(让 reflection 不用自己推):

| Field 由 orchestrator 计算 | 来源 |
|---|---|
| `matches_reasoning` ∈ {YES / PARTIAL / NO / N/A} | 简单匹配:reasoning 提"obj moves up" + outcome 真上移 → YES;reasoning 提了具体方向但 outcome 反方向 → NO;reasoning 没提具体效果 → N/A |
| `no_op_streak` | OutcomeLog 末尾连续 no-op 计数 |
| `state_revisit_count` | 当前 frame_hash 在 episode 内出现次数 |

这三项让 reflection 不用从 OutcomeLog 重推。

### 4.4 输出后处理(每步 merge)

- 解析 JSON,验证 schema(用 jsonschema)
- 解析失败 → **保留旧 Knowledge,记录到 trace,继续 episode**(不阻塞)
- 解析成功 → merge:
  - `action_semantics_update`:逐个键覆盖到 `knowledge.action_semantics`
  - `goal_hypothesis_update`:非 null 则替换
  - `goal_confidence_update`:非 null 则替换
  - `rules_append`:append + dedup + cap 10
  - `failed_strategies_append`:append + dedup + cap 5
  - `current_alert`:**覆盖**`knowledge.current_alert`(下一步开头消费,见 §5.3)
- 每步 merge 后把 `knowledge.to_dict()` append 进 `round_<k>/knowledge_per_step.jsonl`(可观察学习过程)

### 4.5 性能预算

| 字段 | 大约预算 |
|---|---:|
| Reflection input(prompt 长度) | ~600-1000 token |
| Reflection output(delta JSON) | ~50-150 token |
| 每步 reflection 调用耗时(Qwen2.5-VL-3B 4-bit text-only) | **~0.5-1.0s** |
| 每步总耗时(Action + Reflection) | ~1.5-2.0s |
| 5 round × 80 step 总 wall-clock(含 SDK + load) | **~15 min** |

确认在 Kaggle 10h 上限内:5 round × 5 game × 80 step × 2s = 67 min(单 game 多 round 测试用,Kaggle 提交是 110 game × 80 step ≈ 5h)。

---

## 5. Action Agent

### 5.1 跟 v3 TextAgent 的差异

| | v3 TextAgent | v3.2 ActionAgent |
|---|---|---|
| 每步 LLM 调用 | 1 | 1(同) |
| Prompt 含 [GOAL] | 永远 "unknown" | **包含 Knowledge.goal_hypothesis(每步可能被刷新)** |
| Prompt 含 action 语义 | 模型从 OutcomeLog 推 | **直接给 Knowledge.action_semantics(每步可能扩张)** |
| 跨 round 状态 | 完全清空 | 清 ObjectMemory / OutcomeLog,**Knowledge 保留(跨 round + 跨 step)** |
| Knowledge 在 episode 内 | (不适用) | **每步被 Reflection Agent mutate**(per-step learning) |
| 输出格式 | 仅 action token | **reasoning + action**(两行)|

### 5.2 SYSTEM prompt(常量,继承 v3 的硬约束)

```
You are the Action Agent. Pick the next action in a turn-based grid game.

The Reflection Agent has provided KNOWLEDGE from previous rounds:
  - action_semantics: what each ACTION does in this game (when known)
  - goal_hypothesis: the most likely goal so far
  - rules: patterns observed across previous rounds
  - failed_strategies: strategies that were tried and did NOT work

Trust the KNOWLEDGE block. Do NOT re-explore things already documented
as failed_strategies.

OUTPUT FORMAT (strict, two lines):
  reasoning: <one sentence explaining your choice>
  action: ACTION_  (or "ACTION6 x y" — ACTION6 is the ONLY one with coords)

Valid examples:
  reasoning: knowledge says ACTION1 moves the player up; goal is to reach top
  action: ACTION1

  reasoning: failed_strategies says clicking near (32,32) doesn't help; try edge
  action: ACTION6 5 60

Do NOT add coordinates to ACTION1..5 or ACTION7. Do NOT output JSON or
markdown. Just two plain lines: reasoning and action.
```

### 5.3 USER prompt 结构(v3 8 块 + 2 块新增)

```
[REFLECTION ALERT]        ← NEW: knowledge.current_alert,只在 alert 非空时出现
                            放在最显眼位置(prompt 第一块),让模型先看到
[KNOWLEDGE]              ← NEW: knowledge.render() 的余下内容(action_semantics、
                            goal、rules、failed_strategies)
[STATUS]                  ← unchanged from v3
[ACTIVE]                  ← unchanged
[TEXTURE]                 ← unchanged
[ACTION effects observed] ← unchanged (this-round only stats)
[UNTRIED]                 ← unchanged
[HISTORY]                 ← unchanged
[CLICK CANDIDATES]        ← unchanged (only when ACTION6 legal)
[STUCK SIGNALS]           ← unchanged (conditional)
[ASK]                     ← changed to require reasoning + action format
```

**渲染规则**:
- `current_alert` 在 prompt 渲染时如果非空 → 输出 `[REFLECTION ALERT]` 块(在 [KNOWLEDGE] 之前)+ 单独高亮一行
- 渲染完后 **不立即清空** alert(让本步 Action 看到),下一步 Reflection 决定是否覆盖
- 如果连续 3 步 alert 内容相同,Action Agent prompt 加一句 "⚠️ The Reflection Agent has repeated this alert N times — change behavior."

### 5.4 解析 reasoning + action

新增 `_parse_reasoning_and_action(text)`:

```python
def _parse_reasoning_and_action(text: str) -> tuple[str, str]:
    """Pulls 'reasoning:' line and 'action:' line. Tolerant of order."""
    reasoning_match = re.search(r"reasoning\s*:\s*(.+)", text, re.IGNORECASE)
    action_match    = re.search(r"action\s*:\s*(.+)", text, re.IGNORECASE)
    return (
        reasoning_match.group(1).strip() if reasoning_match else "",
        action_match.group(1).strip()    if action_match else text.strip(),
    )
```

`reasoning` 进 trace.jsonl 的 `chain_of_thought` 字段(新增),`action` 同 v3 进 coerce 流程。

---

## 5.5 GIF 可视化设计(2026-05-14 新增,替换 v3 的 4 象限版)

v3 旧的 `compose_step_image` 是 4 象限(`grid_now / predicted_diff overlay / grid_next / JSON text`)。**v3.2 没有 predicted_diff,且 reasoning + reflection 才是关键信息**,所以重新设计。

### 5.5.1 单步 PNG 布局

```
┌─────────────────────────────────────────────────────────┐
│  HEADER:  game_id  step=12/80  action=ACTION6 12 30      │ <- 30 px header
├──────────────────────┬──────────────────────────────────┤
│                      │  [ACTION] ACTION6 (12, 30)        │
│                      │                                    │
│                      │  [REASONING]                       │
│       GRID           │    "click on red target slot,      │
│   (256x256 RGB,      │     knowledge says it advances     │
│    64x64 grid at     │     the level"                     │
│    4x scale)         │                                    │
│                      │  [REFLECTION DELTA]                │
│                      │    action_semantics_update:        │
│                      │      ACTION6: "click; reveals red" │
│                      │    current_alert: ""               │
│                      │    matches_reasoning: PARTIAL      │
│                      │                                    │
│  256x256             │  256x256 (text panel,wrap 32 char) │
└──────────────────────┴──────────────────────────────────┘
total: 512 wide × 286 tall per step PNG
```

- **图**只占左半;比 v3 的 4 象限 PNG 大,看得清楚
- **action + reasoning + reflection** 在右半,分 3 块,每块顶端有 label
- 如果 `current_alert` 非空,右半最顶上用红色背景框出来,**视觉上一眼看到**
- 不显示 predicted_diff(v3 / v3.2 都没有)

### 5.5.2 GIF 合成

- 每步存一张这种 286-高 PNG,episode 结束拼成 GIF(fps=2,跟 v3 一致)
- PNG 命名:`step_NNNN.png`(跟 v3 同惯例)
- GIF 路径:`round_<k>/play.gif`

### 5.5.3 实现位置

新增 `arc_agent/viz_v3_2.py:compose_step_image_v32(grid, action, reasoning, reflection_delta, alert, header)`。旧的 `viz.compose_step_image` **保留**,baseline runner 通过 `agent_kind` flag 选用哪个。

### 5.5.4 文字 panel 内容来源

| 字段 | 来源 |
|---|---|
| header.action | `chosen_action.name`(及 coords) |
| panel.ACTION | 同上 |
| panel.REASONING | Action Agent 输出的 `reasoning:` 行(可能空) |
| panel.REFLECTION DELTA | Reflection Agent 输出的 JSON(精简成 2-3 行,过长 truncate) |
| panel.current_alert | `delta.current_alert`,若非空则**红框高亮** |
| panel.matches_reasoning | orchestrator 算的 (YES / PARTIAL / NO / N/A) |

trace.jsonl 同步新增字段:
- `reasoning`(action agent 的推理)
- `reflection_raw`(reflection 的完整 raw 输出)
- `reflection_delta`(reflection 解析后的 dict)
- `current_alert_active`(本步 Action 看到的 alert,可能跟 reflection_delta.current_alert 不同 — 因为 alert 是上一步留下的)
- `matches_reasoning`(orchestrator 自动判定)

---

## 6. 文件 / 模块清单

| 文件 | 状态 | 责任 |
|---|---|---|
| `arc_agent/knowledge.py` | 🆕 NEW | `Knowledge` dataclass(含 `current_alert`)+ render/to_dict/from_dict/merged_with_delta |
| `arc_agent/agents/reflection_agent.py` | 🆕 NEW | ReflectionAgent class:**reflect_after_step**(knowledge, step_outcome, reasoning, recent_steps) → delta dict |
| `arc_agent/agents/action_agent.py` | 🆕 NEW | 由 v3 `TextAgent` 改造:接 Knowledge,输出 `reasoning + action` 两行 |
| `arc_agent/agents/text_agent.py` | 🟡 保留 deprecated | 旧 v3 agent,继续可用 |
| `arc_agent/prompts_v3_2.py` | 🆕 NEW | SYSTEM + user 模板(2 套:Action / Reflection) |
| `arc_agent/step_summary.py` | 🆕 NEW | 把单步状态压成 reflection 可读(action、outcome、delta、reasoning、matches_reasoning 判定) |
| `arc_agent/viz_v3_2.py` | 🆕 NEW | `compose_step_image_v32(grid, action, reasoning, reflection_delta, alert, header)` — §5.5 布局 |
| `scripts/run_v3_multi_round.py` | 🆕 NEW | 调度 N round,持久化 Knowledge,生成 N GIFs + multi-round report |
| `tests/test_knowledge.py` | 🆕 NEW | render / merge_with_delta / current_alert 行为 |
| `tests/test_reflection_agent.py` | 🆕 NEW | mock backbone,验证 JSON parse + delta merge 逻辑 + alert 产出 |
| `tests/test_action_agent.py` | 🆕 NEW | 跟 test_text_agent 类似但加 reasoning 解析 + alert 渲染 |
| `tests/test_viz_v3_2.py` | 🆕 NEW | 合成 PNG 的关键文本块是否包含 reasoning / reflection / alert |

代码估计:~1700 行(含测试 ~700 行)。预估 1-1.5 工作日。

---

## 7. 多轮调度逻辑(伪代码,per-step reflection)

```python
def run_multi_round(game_id, n_rounds=5, max_actions=80, out_dir):
    arc = Arcade()
    card_id = arc.open_scorecard(tags=["v3_2_multi"])
    backbone = HFBackbone.load()   # 共享一个 Qwen 实例

    action_agent = ActionAgent(backbone=backbone)
    reflection_agent = ReflectionAgent(backbone=backbone)

    knowledge = Knowledge.empty(game_id)
    knowledge_history_path = out_dir / "knowledge_history.jsonl"

    for r in range(n_rounds):
        env = arc.make(game_id, scorecard_id=card_id)
        action_agent.reset_episode_state(knowledge=knowledge)
        round_dir = out_dir / f"round_{r:02d}"
        knowledge_step_path = round_dir / "knowledge_per_step.jsonl"

        # ── episode loop ────────────────────────────────────────────────
        latest = env.reset()
        for step in range(max_actions):
            if latest.state == GameState.WIN:
                break

            # 1) Action Agent reads CURRENT knowledge (incl current_alert)
            alert_active = knowledge.current_alert    # 本步看到的 alert(上一步留下的)
            action_agent.attach_knowledge(knowledge)
            action, reasoning = action_agent.choose(latest)

            # 2) env.step
            frame_t = latest
            latest = env.step(action, data=action.action_data.model_dump())

            # 3) orchestrator 预判 reasoning vs outcome,塞进 reflection prompt
            matches = compute_matches_reasoning(reasoning, frame_t, latest, action)
            # matches ∈ {YES, PARTIAL, NO, N/A}

            outcome = build_step_outcome(
                action, frame_t, latest,
                reasoning=reasoning,
                matches_reasoning=matches,
                no_op_streak=action_agent.no_op_streak(),
                state_revisit=action_agent.state_revisit_count(latest),
            )

            # 4) Reflection Agent updates knowledge — PER STEP, with reasoning
            delta, refl_raw = reflection_agent.reflect_after_step(
                knowledge=knowledge,
                step_outcome=outcome,
                reasoning=reasoning,           # 显式传入,reflection 据此纠错
                recent_steps=action_agent.recent_step_records(n=3),
            )
            knowledge = knowledge.merged_with_delta(delta)

            # 5) 写 per-step PNG(新 5.5 布局)
            png = compose_step_image_v32(
                grid=latest_grid(frame_t),
                action=action.name,
                reasoning=reasoning,
                reflection_delta=delta,
                alert=alert_active,           # 本步 Action 实际看到的 alert
                header=f"{game_id} step={step} action={action.name}",
            )
            png.save(round_dir / f"step_{step:04d}.png")

            # 6) Snapshot 进 trace
            append_jsonl(knowledge_step_path, {
                "step": step,
                "action": action.name,
                "reasoning": reasoning,
                "reflection_raw": refl_raw,
                "reflection_delta": delta,
                "current_alert_active": alert_active,
                "matches_reasoning": matches,
                "knowledge_after": knowledge.to_dict(),
            })

        # ── end of round bookkeeping ───────────────────────────────────
        knowledge.rounds_played += 1
        if latest.state == GameState.WIN:
            knowledge.rounds_won += 1
        knowledge.round_history.append(
            build_round_summary_one_line(round_dir, r)
        )
        write_gif_for_round(round_dir)
        append_jsonl(knowledge_history_path, {
            "round": r,
            "knowledge_at_round_end": knowledge.to_dict(),
        })

    arc.close_scorecard(card_id)
    build_multi_round_report(out_dir, n_rounds, knowledge)
```

---

## 8. 评估方法

### 8.1 单次 run 的指标(每个 round)

| 指标 | 含义 | 期望趋势 |
|---|---|---|
| `entropy` | round 内 action 分布的香农熵 | round0 较高,后面随 Knowledge 收敛**逐渐降低** |
| `no_op_rate` | round 内 no-op 比例 | round 增加应**下降**(更精准的 action) |
| `unique_frame_hashes` | round 内 unique state | round 增加应**上升**(更多探索) |
| `max_state_revisit` | round 内同一 state 最多访问次数 | round 增加应**下降**(stuck 模式被 Knowledge 知道) |
| `levels_completed` | SDK 返回 | 任一 round > 0 = 重大胜利 |
| `parse_rate_reasoning` | reasoning 行能解析出来的比例 | ≥ 0.9 |
| `reflection_parse_rate` | 每步 reflection JSON 解析成功率 | ≥ 0.9(< 0.9 要重写 prompt)|
| `knowledge_grow_in_episode` | 一个 episode 内 `len(action_semantics)` 净增长 | ≥ 2 表示在 episode 内学到东西 |
| `alert_trigger_rate` | reflection 写 `current_alert` 的步数比例 | 10%-30% 健康;> 60% 说明 reflection 在乱报警 |
| `reasoning_outcome_match_rate` | `matches_reasoning ∈ {YES, PARTIAL}` 的步数比例 | 应**随 round 增长**(reasoning 越来越准)|

### 8.2 跨 round + 跨 step 的趋势指标

| 指标 | 期望 |
|---|---|
| `len(knowledge.action_semantics)` 随 **step** 增加 | step 0 = 0;step 80 ≥ 3 |
| `len(knowledge.action_semantics)` 随 **round** 增加 | round 0 末尾 ≥ 3;round 4 末尾 ≥ 5 |
| `len(knowledge.rules)` 随 round 增加 | 0 → 3-5 |
| `goal_confidence` 随 round 提升 | low → medium → (理想)high |
| 平均 wall_clock_per_step | **~1.5-2 s/step**(per-step reflection 翻倍开销;符合预算)|

### 8.3 决策门(canary 通过后才推进)

| 门 | 条件 | 失败应对 |
|---|---|---|
| **G1 (Knowledge 在 episode 内累积可见)** | round 0 跑完 80 step 后,`len(action_semantics) ≥ 3` | reflection prompt 太弱 / parse 失败率过高 |
| **G2 (行为受 Knowledge 影响)** | round 末后 30 步 vs 前 30 步:`no_op_rate` 下降 ≥ 20% | Action Agent 没在读 knowledge;查 prompt 渲染 |
| **G3 (Kaggle 友好)** | 5 round × 80 step + per-step reflection ≤ **20 min** | reflection prompt 太大,裁 |
| **G4 (任意 round 通关)** | levels_completed > 0 至少一次 | scope 升级:更多 round 或换游戏 |

### 8.4 失败模式 list(预先想清楚怎么诊断)

| 症状 | 可能原因 | 验证方法 |
|---|---|---|
| Knowledge 不积累(每 round 还是空) | Reflection JSON parse 总失败 | 看 `outputs/<run>/round_*/reflection_raw.txt` |
| Knowledge 一直变化但没用 | Action Agent 没读 Knowledge | grep prompt 看 `[KNOWLEDGE]` 是否非空 |
| 模型一直输出空 reasoning | format 不严格;reasoning 解析正则太严 | 看 trace.jsonl 的 `chain_of_thought` 字段 |
| ar25 多轮还是不通关 | Knowledge 累积了但游戏机制更难 | 看 `goal_hypothesis` 随 round 是否改善;改进 reflection 提示 |

---

## 9. 实施顺序(canary 工作流照 ref_v3_prompt §10.2 走)

| # | 步骤 | 文件 | 验证 |
|---|---|---|---|
| 1 | `Knowledge` dataclass + 单测 | `arc_agent/knowledge.py`, `tests/test_knowledge.py` | L0 pytest |
| 2 | `step_summary.py`(把单步状态压成 reflection 可读,**per-step**) | new | unit test |
| 3 | `prompts_v3_2.py` 两套 SYSTEM/USER prompt(action / reflection) | new | unit test |
| 4 | `ReflectionAgent.reflect_after_step(...)` + 单测(fake backbone) | new | L0 pytest |
| 5 | `ActionAgent` = TextAgent + reasoning 解析 + `attach_knowledge` 接口 | new | L0 pytest |
| 6 | `run_v3_multi_round.py` 调度(注意:loop 内 2 个 LLM 调用)| new | ar25 1 round × 10 step canary,先确认 reflection 真的每步被调用 |
| 7 | 跑 **ar25 × 3 round × 20 step** smoke + GIF | — | 看 Knowledge 是否**在 episode 内**累积(`knowledge_per_step.jsonl` 应该 80 行) |
| 8 | 跑 **ar25 × 5 round × 80 step** + 完整 multi-round report + 5 GIFs | — | G1/G2/G3 决策门 |
| 9 | 通过 G1+G2 → 扩到 G_base 5 game | — | G3/G4 检查 |

**关键时间点**(per-step reflection 后 wall-clock 翻倍):
- 步骤 1-6 完成 ≈ 实施一天
- 步骤 7 跑一次 ≈ **5-8 分钟**(3 round × 20 step × ~2s + GIF)
- 步骤 8 跑一次 ≈ **15-20 分钟**(5 round × 80 step × ~2s)
- 步骤 9 跑一次 ≈ **1-2 小时**(5 game × 上面 8 的时间)

---

## 10. 与 v3 / v3.1 的关系

| | v3 baseline | v3.1 (现在的 P0-A/B + P1) | **v3.2** |
|---|---|---|---|
| 视觉感知 | scipy + Hungarian | 同 | 同 |
| 每步 LLM 调用数 | 1 (Action) | 1 (Action) | **2 (Action + Reflection per-step)** |
| Action 决策 | 1 个 Qwen 调用 | 1 个 Qwen 调用 | **1 个 Action Agent** |
| 历史总结 | 隐含在 prompt 里 | 同 + click candidates + stuck signal | **显式由 Reflection Agent 每步做** |
| 跨 round 学习 | ❌ 每 episode 全清 | ❌ 同 | ✅ **Knowledge 持久化** |
| 输出格式 | bare action token | 同 | **reasoning + action(两行)** |
| GIF | 可选(--with-images) | 同 | **强制每 round 都生成** |

v3.1 的 P0-A、P0-B、P1 改动**全部继承**。v3.2 在它们的基础上加 Knowledge / Reflection / 多轮调度。

---

## 11. 本设计**不**承诺的事

- **跨 game 知识迁移**(只在同一 game 内多 round 累积,新 game 仍从空开始)
- **Reflection 用 VL 模式**(继续 text-only;Reflection 看 ObjectMemory snapshot 文本就够)
- **多 agent 并行**(顺序调用,Action 每步,Reflection 每 round 末尾)
- **RL 训练**(纯 in-context learning,不调 model weights)
- **改 P2 / P3**(模式检测、环境动态)— 留到 v3.3 / v3.4

---

## 12. 读完这个文档你应该能回答的问题

1. v3 → v3.2 的根本变化是什么?(单 agent → 双 agent + Knowledge 持久化)
2. Reflection Agent 何时被调用?(每个 episode 结束一次)
3. Knowledge 里有哪些字段?(action_semantics, goal_hypothesis, rules, failed_strategies, rounds_played/won, round_history)
4. 多轮调度的关键不变量是什么?(episode 状态每 round 清,Knowledge 不清)
5. 怎么验证 v3.2 真的"在学"?(8.2 跨 round 趋势 + 8.3 决策门)

---

## 13. 🟡 实施进展 + 已知 Bug (2026-05-14 update)

> 以下都是 v3.2 已**实施**并跑过真模型后发现的事。黄色高亮部分是**还没修**或**待澄清**的 bug,蓝色是已落地的硬规则。

### 13.1 已落地的硬规则 (orchestrator-level, 实测有效)

<span style="color: #0066cc">✅ **R1** sentinel filter</span> —— `Knowledge.merged_with_delta` 拒绝 `"unknown"/"none"/"n/a"/"tbd"/"exploring"` 等字面量作为 `goal_hypothesis_update`。`arc_agent/knowledge.py`。

<span style="color: #0066cc">✅ **R4** 自相矛盾过滤</span> —— `rules_append` / `failed_strategies_append` 含 "ACTION_X has no effect / anywhere in ..." 时,若 `action_semantics[X]` 已有正面语义,DROP 该规则。配合 `compute_action_mask` 的 `_has_positive_semantic` 交叉检查 —— OutcomeLog 真实成功的 action 永远不会被 Knowledge 文字反向标记。

<span style="color: #0066cc">✅ **R5** failed_strategies 交叉污染过滤</span> —— `goal_hypothesis_update` 若与 (existing 或 about-to-be-appended) failed_strategies 字符串匹配则拒绝。`arc_agent/knowledge.py`。

<span style="color: #0066cc">✅ **R6** action-described goal 过滤</span> —— `goal_hypothesis_update` 若以 "ACTION1-7" 开头或含 "should move/advance/click/push" 则拒绝。

<span style="color: #0066cc">✅ **R7** [LOW-PRIORITY ACTIONS] 块</span> —— Action prompt 显示 mask 结果给 LLM (advisory,不替换)。`arc_agent/prompts_v3_2.py`。

<span style="color: #0066cc">✅ **A** Reflection 看完整 state</span> —— `build_reflection_user_prompt` 现在含 [STATUS] / [ACTIVE] / [OBJECT RELATIONS] / [ACTION effects observed] / [HISTORY]。Reflection 第一次能写出真 goal (1×30 smoke: "reshape objects")。

<span style="color: #0066cc">✅ **B** mask 转 advisory</span> —— orchestrator **不再静默替换** action。`apply_action_mask` 不再被 orchestrator 调用。决策权回 LLM,适合 ar25 这种 state-dependent 游戏 (ACTION1 在天花板时 no-op,离开后恢复)。

<span style="color: #0066cc">✅ **C** orchestrator 自动 stuck alert</span> —— `state_revisit >= 5` 或 `no_op_streak >= 5` 时 deterministic 写 alert。`scripts/run_v3_multi_round.py:_build_stuck_alert`。

<span style="color: #0066cc">✅ **D** 自然终止</span> —— round 在 `latest.state ∈ {WIN, GAME_OVER}` 时结束,`max_actions` 默认 80 → 500 仅作安全上限。

测试套件 570+ 通过,零回归。

### 13.2 🟡 实测发现的 Bug (2026-05-14 ar25 2×80 ABCD run)

> 来源: `outputs/v3_2_ar25_2x80_ABCD/round_00/trace.jsonl` 实测 80 步数据。每条 bug 标了 严重度 / 复现数据 / 候选修法。

#### <span style="color: #b58900">🟡 BUG-1 — Action reasoning 91% N/A</span>

**严重度**: 高 (反馈信号大量丢失)
**复现**: round 0 共 80 步,`matches_reasoning` 分布: **N/A=73 / YES=6 / NO=1 / PARTIAL=0**
**原因**: Action Agent reasoning 大多是泛泛 ("I'll try ACTION1 to make progress"),没具体说预期 direction,所以 `compute_matches_reasoning` 标 N/A。Reflection 拿不到"预期 vs 实际"的强信号。
**候选修法**: Action SYSTEM prompt 加硬约束: reasoning **必须**含 direction 词 (UP/DOWN/LEFT/RIGHT) 或 "no change expected"。`compute_matches_reasoning` 也可放宽匹配 (色彩/对象 id 关键词)。

#### <span style="color: #b58900">🟡 BUG-2 — action_semantics 多面效果被覆写</span>

**严重度**: 高 (用户提出)
**复现**:
- step 5 写: `ACTION7: "reshapes objects"` (1×30 smoke 也观察到)
- step 8 覆写: `ACTION7: "moved an active object DOWN by 3 cells"`
- step 26/28 重复写 "DOWN" —— "reshapes" 的洞察永久丢失
- ACTION7 真实 stats: 18/19 changed,但效果不止 DOWN

**原因**: `action_semantics` 是单字符串 `dict[str, str]`,Reflection 用最新观察覆写。多面行为 (例如 ACTION7 在不同位置做不同事) 记不下。
**候选修法**:
1. 改 schema → `action_semantics: dict[str, list[ObservedEffect]]`,每条带条件 / 出现次数 / 置信度
2. 简单版: 改 prompt 让 Reflection 写 conditional 形式: `"ACTION7: sometimes reshapes (when objects same color), sometimes moves DOWN (default)"`
3. 引入 **orchestrator 自动 summarizer**: 基于 OutcomeLog 确定性生成 semantic,跳过 LLM 主观判断

#### <span style="color: #b58900">🟡 BUG-3 — 相同 semantic 被反复重写</span>

**严重度**: 中 (浪费 LLM)
**复现**: ACTION1 的 semantic `"moves an active object UP by 3 cells"` 被写了 **29 次** (内容完全相同)。ACTION7 写 27 次。
**原因**: Reflection 没"我已写过相同内容,跳过"的去重逻辑。
**候选修法**: `Knowledge.merged_with_delta` 在合并 `action_semantics_update[X]` 时,若新值 == 现值,no-op。已经事实上是 no-op 但每次 Reflection LLM 调用产生这些 token 是浪费 —— 改 Reflection prompt 加 "已有相同 entry 时设 `action_semantics_update: {}`"。

#### <span style="color: #b58900">🟡 BUG-4 — ACTION6 坐标无 memory</span>

**严重度**: 中 (探索效率)
**复现**: ACTION6 在 round 0 试了 (5,60) 4 次、(31,63) 4 次,LLM 看不到自己之前试过哪些 coord 都失败了。
**原因**: 当前 `[CLICK CANDIDATES]` 块只列候选,不显示已 tried 坐标。`tried_action6_coords` 在 ActionAgent 内部存了但没渲染到 prompt。
**候选修法**: `build_play_user_prompt` 加 `[ACTION6 TRIED COORDS]` 块,列出已 tried 的 (x,y) + 结果,避免重试。

#### <span style="color: #ff4444">🔴 BUG-5 — Round 撞 max_actions 就停,游戏未结束</span> **用户标为非常严重**

**严重度**: 关键 (用户原话: "相当于你中途放弃")
**复现**: 2×80 round 0 跑满 80 步,最后一步 `latest.state == NOT_FINISHED`,游戏没分胜负就停了。每张 PNG 显示游戏进度 ~10/N,远没完。
**原因**: D 改动已支持自然终止,但 CLI 默认还在用 `--max-actions 80` (从旧的 human-step budget 继承)。80 步对 ar25 远远不够 —— 60+ 步在等 player 撞墙后重新探索方向。
**候选修法**:
1. ✅ 已改: `max_actions` 默认 80 → 500
2. 还需改: 老的 launch 命令里不再传 `--max-actions 80`,用默认
3. **或**: 完全去掉 max_actions 强制上限,只信任游戏的 `GAME_OVER` 信号 (但有死循环风险,需要 C alert 兜底)

#### <span style="color: #b58900">🟡 BUG-6 — Reflection 输出无 per-claim 置信度</span> (用户提出)

**严重度**: 中 (Knowledge 质量)
**复现**: 当前 Reflection JSON 只有一个 `goal_confidence_update: low/medium/high`。action_semantics / rules / failed_strategies 都无置信度。
**原因**: schema 设计简单。Reflection 写 `"ACTION1: moves UP"` 时,基于 1 次观察还是 25 次,Action Agent 无法分辨。
**候选修法**:
1. action_semantics schema → `dict[str, {"value": str, "confidence": str, "n_obs": int}]`
2. 简单版: 让 Reflection prompt 在 semantic 字符串内加置信度词 (e.g. "ACTION1: moves UP 3 (high-conf, 25 obs)")

#### <span style="color: #b58900">🟡 BUG-7 — C stuck alert 阈值偏高 / 未触发</span>

**严重度**: 低 (但暴露设计 gap)
**复现**: 2×80 round 0 max `state_revisit=4`,刚好低于 C 阈值 5。C 全程没触发。
**原因**: ar25 这局 LLM 探索还算多样,没真正卡死。但阈值 5 可能略宽。
**候选修法**: 阈值调到 3-4,**或**改为"在最近 K 步内出现 >= 50% no-op 即触发"。

### 13.3 🟡 下一步优先级 (建议)

按用户反馈强度排序:

1. **BUG-5 (round 提早结束)** —— 关键,立即修。CLI 不再传 --max-actions 80,游戏自己决定何时停。
2. **BUG-2 (action_semantics 覆写)** —— 用户明确提到 + 影响 Knowledge 质量。先做简单版 (Reflection prompt conditional 形式),后考虑 schema 升级。
3. **BUG-6 (per-claim confidence)** —— 用户明确提到。配合 BUG-2 一起改 Reflection schema。
4. **BUG-1 (reasoning N/A 91%)** —— Action SYSTEM prompt 加 direction 强制。
5. **BUG-4 (ACTION6 coord memory)** —— [ACTION6 TRIED COORDS] 块。
6. **BUG-3 (重复 semantic)** —— Reflection prompt 微调。
7. **BUG-7 (C 阈值)** —— 调 3-4。

---

*文档历史: 2026-05-14 初稿; 2026-05-14 加 §13 实测 Bug 清单 (黄色高亮的是待修).*
