# v3 Prompt 参考(逐块来源 + 实战例子 + 卡死检测缺陷)

日期: 2026-05-14
配套: `arch_v3_zh.md` §2 数据流,本文是它的 prompt 层快照 + 改进建议

---

## 0. 顶层结构

每步喂给 Qwen 的实际消息长这样:

```
[SYSTEM]  ← 常量,静态,英文,中性(不告诉 action 语义)
[USER]    ← 每步动态构造,8 个块
```

发送时用 chat-template 拼成 messages:
```python
[{"role": "system", "content": SYSTEM},
 {"role": "user",   "content": USER}]
```

---

## 1. SYSTEM prompt(常量,跨步跨游戏不变)

**来源**: `arc_agent/prompts_v3.py:PLAY_SYSTEM`(模块级常量,~30 行英文,纯 ASCII)

**完整内容**:

```
You play an unfamiliar turn-based grid game on a 64x64 grid.
You do not know in advance what each action does. Different games map
ACTION1..ACTION7 to different effects. Discover by trying actions and
observing how the grid changes.

The user prompt summarizes what has already been computed for you:
  [STATUS]   current step, level, and pre-computed aggregates
  [ACTIVE]   moving / interactive objects with movement history
  [TEXTURE]  static background pattern (filtered for you)
  [ACTION]   what each action has done so far this episode
  [UNTRIED]  legal actions you have NOT tried yet
  [HISTORY]  the last few action outcomes
  [GOAL]     a hypothesis you may refine

Your priorities, in order:
  1. If [UNTRIED] is non-empty, try one of those actions
  2. Otherwise, pick an action whose history is most consistent with
     making progress (frame changes, level advances)
  3. Never repeat the same action 3+ times in a row unless evidence
     proves it advances toward the goal

Output ONE action token only (e.g. "ACTION3" or "ACTION6 12 30"). No JSON.
No prose. No explanation.
```

**遵守的三条硬约束**(传承 v2 红字):
1. ✅ 不预告 action 语义(无 "ACTION1=up" 这种)
2. ✅ 全英文 ASCII,跨 agent 统一
3. ✅ 全静态,任何动态内容都在 user prompt

---

## 2. USER prompt(每步动态,8 块)

下面用 **ar25 step 16** 的真实 prompt 作为 case。每块标:**来源代码 / 数据源 / 例子**。

### 2.1 `[STATUS]` 块

**来源代码**: `prompts_v3.py:_format_status()`

| 字段 | 来源 |
|---|---|
| `step: 16 / 80` | runner 计数器 + CLI `--max-actions` |
| `level: 1 / 8` | `FrameDataRaw.levels_completed + 1` / `win_levels` |
| `game state: NOT_FINISHED` | `FrameDataRaw.state.name` |
| `legal actions: ...` | `observation.available_action_names(latest)` |
| `topmost active: obj_002 (yellow)` | `min(active_objects, key=...bbox[0])` —— **pre-computed** |
| `largest active: obj_002 (yellow, size=45)` | `max(active_objects, key=...size)` —— **pre-computed** |

**例子**:
```
[STATUS]
  step: 16 / 80
  level: 1 / 8
  game state: NOT_FINISHED
  legal actions: ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION6, ACTION7
  topmost active: obj_002 (yellow)
  largest active: obj_002 (yellow, size=45)
```

### 2.2 `[ACTIVE]` 块

**来源代码**: `prompts_v3.py:_format_active_block()`

| 字段 | 来源 |
|---|---|
| `obj_002: yellow (size=45, bbox=...)` | `ObjectMemory.alive_tracked()` 的每个 UID 的最新 snapshot |
| `last step: moved 3 cell(s) RIGHT` | 用 `history[-1].center - history[-2].center` 算出 dy/dx + 翻成 UP/DOWN/LEFT/RIGHT —— **pre-computed** |

**数据流**:
1. scipy `extract_objects(grid)` 出 candidate
2. `temporal_classifier.classify_frame()` 标 STATIC/ACTIVE/TEXTURE
3. 只 ACTIVE 进 [ACTIVE] 块
4. `object_tracker.ObjectMemory.update()` 跨帧合并 UID

**例子**:
```
[ACTIVE]
  obj_002: yellow (size=45, bbox=[15,54,23,62])
      last step: moved 3 cell(s) RIGHT (dy=+0, dx=+3)
  obj_003: gray (size=40, bbox=[15,0,23,8])
      last step: moved 3 cell(s) LEFT (dy=+0, dx=-3)
```

### 2.3 `[TEXTURE]` 块

**来源代码**: `prompts_v3.py:_format_texture_block()` + `temporal_classifier.texture_summary()`

| 字段 | 来源 |
|---|---|
| `total: N cells (C green, ...)` | `temporal_classifier.is_likely_texture()`:同色同形小对象 ≥ 20 个 |

ar25 上是 `(none)`,bp35 上会有大量 texture 被聚合。

**例子**(bp35 风格):
```
[TEXTURE] (treated as background, filtered out)
  total: 187 cells (187 lime cells)
```

### 2.4 `[ACTION effects observed]` 块

**来源代码**: `prompts_v3.py:_format_action_block()` + `action_inference.render_action_block()`

| 字段 | 来源 |
|---|---|
| 每个 action 一行汇总 | `OutcomeLog.by_action[ACTION_X]` 列表 + `summarize_action()` 聚合 |

**例子**(step 16 真实):
```
[ACTION effects observed]
  ACTION1: tried 1x: 1x changed (no clear direction)
  ACTION2: tried 1x: 1x changed (no clear direction)
  ACTION3: tried 10x: 8x moved RIGHT 3 cell(s), 2x no-op
  ACTION4: tried 1x: 1x moved LEFT 3 cell(s)
  ACTION5: tried 1x: 1x changed (no clear direction)
  ACTION6: tried 1x: all no-op
  ACTION7: tried 1x: 1x changed (no clear direction)
```

### 2.5 `[UNTRIED legal actions]` 块

**来源代码**: `prompts_v3.py:_format_untried_block()` + `OutcomeLog.untried()`

**例子**(step 16 — 已经全试过):
```
[UNTRIED legal actions]
  (none — every legal action has been tried)
```

### 2.6 `[HISTORY last 5 steps]` 块

**来源代码**: `prompts_v3.py:_format_history_block()` + `action_inference.render_history_tail()`

| 字段 | 来源 |
|---|---|
| `step N: ACTION_X -> CHANGED/no-op` | `OutcomeLog.all_steps[-5:]` 每行 |

**例子**(step 16 真实):
```
[HISTORY last 5 steps]
  step 11: ACTION3 -> no-op
  step 12: ACTION5 -> CHANGED
  step 13: ACTION3 -> no-op
  step 14: ACTION7 -> CHANGED
  step 15: ACTION3 -> CHANGED (RIGHT 3)
```

### 2.7 `[GOAL hypothesis]` 块

**来源代码**: `prompts_v3.py:_format_goal_block()`

**当前状态**:`_TextAgentState.goal_hypothesis` 在整个 episode 内**保持空字符串**,因为我们还没接 Reflection。所以这块永远是:

```
[GOAL hypothesis]
  (unknown — still exploring)
```

未来要做的:每 K 步调一次 Qwen reflection,更新这块。

### 2.8 `[ALERT]` 块(条件性,只在卡死时出现)

**来源代码**: `agents/text_agent.py:choose()` 的 `diversification` 变量 + `detect_collapse()`

触发条件:**最近 `COLLAPSE_WINDOW=3` 步都是同一个 action**。

**例子**:
```
[ALERT] You have picked ACTION3 3 times in a row without progress.
        STOP repeating it. Pick a different action.
```

⚠️ **这就是问题所在 —— 见 §4**。

### 2.9 `[ASK]` 块(常量结尾)

总是:
```
[ASK]
  Output ONE action token now.
```

---

## 3. ar25 step 16 完整 prompt(实战例子)

把所有块拼起来,以下就是 Qwen 在 step 16 收到的**完整 user 消息**(我从 trace.jsonl 抽的):

```
[STATUS]
  step: 16 / 80
  level: 1 / 8
  game state: NOT_FINISHED
  legal actions: ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION6, ACTION7
  topmost active: obj_002 (yellow)
  largest active: obj_002 (yellow, size=45)

[ACTIVE]
  obj_002: yellow (size=45, bbox=[15,54,23,62])
      last step: moved 3 cell(s) RIGHT (dy=+0, dx=+3)
  obj_003: gray (size=40, bbox=[15,0,23,8])
      last step: moved 3 cell(s) LEFT (dy=+0, dx=-3)

[TEXTURE]
  (none)

[ACTION effects observed]
  ACTION1: tried 1x: 1x changed (no clear direction)
  ACTION2: tried 1x: 1x changed (no clear direction)
  ACTION3: tried 10x: 8x moved RIGHT 3 cell(s), 2x no-op
  ACTION4: tried 1x: 1x moved LEFT 3 cell(s)
  ACTION5: tried 1x: 1x changed (no clear direction)
  ACTION6: tried 1x: all no-op
  ACTION7: tried 1x: 1x changed (no clear direction)

[UNTRIED legal actions]
  (none — every legal action has been tried)

[HISTORY last 5 steps]
  step 11: ACTION3 -> no-op
  step 12: ACTION5 -> CHANGED
  step 13: ACTION3 -> no-op
  step 14: ACTION7 -> CHANGED
  step 15: ACTION3 -> CHANGED (RIGHT 3)

[GOAL hypothesis]
  (unknown — still exploring)

[ASK]
  Output ONE action token now.
```

**模型这步的输出**: `ACTION3`
**实际发生**:**no-op**(我从下一行 trace 看到 step 16 的 real_diff=[],frame 没变)
**正确答案应该是**:试 ACTION1 / ACTION2 / ACTION4 / ACTION5(都改变过 frame 但还没充分探索),或重新评估 yellow 的位置 —— yellow 已经移到 col 62(几乎贴右壁),ACTION3 = RIGHT 已经撞墙了。

---

## 4. ❗模型为啥发现不了自己卡了

### 4.1 现有的卡死检测只有"3 步连续相同 action"

代码:`text_agent.py:detect_collapse(log, window=3)` → 触发 `[ALERT]`。

但 ar25 step 11 / 13 / 16 / 17 这些卡死步**都没触发** —— 因为最近 3 步是 `ACTION3, ACTION7, ACTION3`(不连续相同),或者 `ACTION3, ACTION3` 但前面夹了个 ACTION7。

### 4.2 真正的卡死信号被埋没

看 step 16 的 prompt,真相全在,但**信号太散**:

| 信号 | 在哪 | 是否突出 |
|---|---|---|
| ACTION3 最近 10 步里 2 次 no-op | `[ACTION effects observed]` 行末尾的 `2x no-op` | ❌ 跟"8x success"混着说,模型 over-weight 成功 |
| ACTION3 最近 5 步里 2 次 no-op | `[HISTORY]` 块 | ❌ 5 步里只有 2 个 no-op,看着不严重 |
| yellow 当前在 col 62(贴墙) | `[ACTIVE]` 的 bbox | ❌ 模型不会主动算"还能不能继续 RIGHT" |
| frame 哈希在最近重复过 | **没有此字段** | ❌ |
| no-op streak(连续多少步 no-op) | **没有此字段** | ❌ |

### 4.3 模型行为解读

模型看到 `ACTION3: tried 10x: 8x moved RIGHT 3 cell(s), 2x no-op`,会"概率上"觉得 ACTION3 是最有用的(80% 成功率),所以继续选它。**它没有"上一次成功是几步前"的概念**。

---

## 5. 建议的卡死检测增强(具体改动)

### 5.1 在 prompt 里加 `[STUCK SIGNALS]` 块(优先)

新增一块,把所有卡死信号集中放在最显眼的地方,**ALERT 升级版**:

```
[STUCK SIGNALS]  ← 只在以下任一条件下出现
  no-op streak: 0/1/2/...     (累计连续 no-op 步数)
  frame_hash repeated: N times in last 20 steps
  last successful action: <name> at step <K>(已经 X 步没新变化)
  same-state revisits: <count>
  diversification advice: <strong text>
```

**触发条件**(任一即出现):
- 最近 3 步至少 2 次 no-op
- 最近 5 步至少 3 次 no-op
- 当前 frame_hash 在最近 20 步内出现过 ≥ 3 次
- 自上次 level_completed 变化以来已 ≥ 20 步无进展

### 5.2 在 `[HISTORY]` 加可视化连续 no-op

把 5 步 history 改成更长(10 步)+ 突出 no-op streak:

```
[HISTORY last 10 steps]
  step 7:  ACTION3 -> CHANGED (RIGHT 3)
  step 8:  ACTION3 -> CHANGED (RIGHT 3)
  step 9:  ACTION6 -> no-op
  step 10: ACTION3 -> CHANGED (RIGHT 3)
  step 11: ACTION3 -> no-op       ← stuck started
  step 12: ACTION5 -> CHANGED (1 cell)
  step 13: ACTION3 -> no-op       ← still stuck on ACTION3
  step 14: ACTION7 -> CHANGED
  step 15: ACTION3 -> CHANGED (RIGHT 3)
  step 16: ACTION3 -> no-op       ← stuck this step
  no-op rate in window: 4/10 = 40% (RISING)
```

### 5.3 在 `[ACTION effects observed]` 里加"近期窗口"

现在是 episode 累计统计。改成 "总体 + 最近 10 步":

```
[ACTION effects observed]
  ACTION3: total 10x (8 RIGHT/2 no-op);
           recent 5x: 2 RIGHT, 3 no-op   ← 反映"现在不工作了"
```

### 5.4 在 `[ACTIVE]` 加"碰壁"信号

如果 active object 的 bbox 已经贴 grid 边界 + 上一步是同方向移动,加一行警告:

```
obj_002: yellow (size=45, bbox=[15,54,23,62])
    last step: moved 3 cell(s) RIGHT (dy=+0, dx=+3)
    ⚠️ bbox col_max=62 -- AT RIGHT WALL, further RIGHT will likely fail
```

### 5.5 强化 `[ALERT]` 触发条件(代码改 `detect_collapse`)

把"连续 3 次相同"改成"3 个并集条件之一":

```python
def detect_stuck(log: OutcomeLog,
                 frame_hashes: list[int],   # 新加 — agent 维护
                 ) -> Optional[str]:
    """Return a human-readable stuck reason, or None."""
    # Condition A: 3 consecutive same action
    if last_3_same_action(): return "3 same actions in a row"
    # Condition B: no-op streak >= 3
    if last_n_no_ops(3): return "no-op streak of 3"
    # Condition C: frame_hash repeated >= 3 times in last 20
    if hash_repeats(frame_hashes, 20, 3): return "you have visited this exact state 3+ times recently"
    # Condition D: action with recent-window no-op rate > 60%
    for a in tried_actions:
        if recent_no_op_rate(a, window=5) > 0.6:
            return f"{a} has been mostly no-op recently — try something else"
    return None
```

---

## 6. 实施清单(短)

| # | 修改 | 文件 | 估时 |
|---|---|---|---|
| 1 | `text_agent` 增加 `frame_hash_history` per-episode 列表 | `text_agent.py` | 10 min |
| 2 | 重命名/扩展 `detect_collapse` -> `detect_stuck`(4 条触发) | `action_inference.py` | 30 min |
| 3 | `[HISTORY]` 块 5 步 -> 10 步,带 no-op rate 行 | `prompts_v3.py` | 15 min |
| 4 | `[ACTION effects observed]` 块加 recent-window 子项 | `prompts_v3.py` + `action_inference.summarize_action` | 30 min |
| 5 | `[ACTIVE]` 加 bbox 贴墙警告 | `prompts_v3.py:_format_active_block` | 20 min |
| 6 | 新增 `[STUCK SIGNALS]` 块(可选,如果 5 个改动还不够) | `prompts_v3.py` | 30 min |
| 7 | 单元测试更新 + 重跑 ar25 smoke | tests + run | 20 min |

总:**~2.5 小时代码 + 5 min 重跑 ar25**

---

## 7. 怎么验证有效

把当前 v3 跑 ar25 的指标记下来(baseline):

| 指标 | v3 当前(无 stuck 检测) | v3+stuck 预期 |
|---|---:|---:|
| no-op rate | 43.8% | < 30% |
| ACTION3 连用次数(总) | 14 | ≤ 10 |
| ACTION3 在 step 11+ 后的占比 | 占主导 | 应明显下降 |
| ACTION1/2 试过次数 | 各 1 | 应该多试几次因为 ACTION3 卡 |
| levels_completed | 0 | 目标 ≥ 1(但不保证) |
| action_entropy | 0.836 | ≥ 1.0 |

---

## 8. ❗发现的额外问题:Action 格式过度泛化(2026-05-14 audit)

跑完 5 G_base × 80 步(`outputs/v3_visual_full/`)用 `scripts/audit_action6_misuse.py` 审计 400 步,**严格违反 legal_actions 集合的"非法 action" 数 = 0**,但发现另一类问题:

### 8.1 数据

| 严格违法(action_value ∉ legal_actions) | **0 / 400 (0%)** |
| ACTION6 调用时缺 coords | **0 / 9** |
| **非 ACTION6 actions 带尾部坐标(格式滥用)** | **272 / 400 (68%)** |

Qwen 输出形状分布(实际从 400 步 trace 抽出):

| Response 形状 | 次数 | 占比 |
|---|---:|---:|
| `ACTION1 <coords>` | 97 | 24.2% |
| `ACTION3 only` | 79 | 19.8% |
| `ACTION3 <coords>` | 60 | 15.0% |
| `ACTION4 <coords>` | 56 | 14.0% |
| `ACTION2 <coords>` | 54 | 13.5% |
| `ACTION2 only` | 19 | 4.8% |
| `ACTION4 only` | 10 | 2.5% |
| `ACTION6 <coords>` | 9 | 2.2% |
| `ACTION1 only` | 8 | 2.0% |
| `ACTION5 <coords>` | 3 | 0.8% |
| `ACTION7 only` | 3 | 0.8% |
| `ACTION7 <coords>` | 2 | 0.5% |

**典型例子**(ar25 step 0):
- 模型输出: `ACTION1 10 20`
- 正确格式: `ACTION1`(ACTION1 不需要坐标)
- ACTION6 才接 `<x y>`,但模型把它泛化成了 "ACTIONx [coords]" 的通用模板

### 8.2 为什么这样输出

看 SYSTEM prompt 最后一行:

```
Output ONE action token only (e.g. "ACTION3" or "ACTION6 12 30"). No JSON.
```

模型读到两个例子(`ACTION3`、`ACTION6 12 30`)后**没有学到"只有 ACTION6 接坐标"的区分**,而是把"action token + 数字"当作了通用规范。再看 [STATUS] 块:

```
legal actions: ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION6, ACTION7
```

**只列了名字,没有标谁需要 coords**。所以模型把所有 action 当成"格式上对称"。

### 8.3 后果

**直接后果**:`_coerce_action` 对非 ACTION6 静默忽略多余坐标,**实际执行没问题**(action 还是合法的),所以审计里 illegal=0。

**间接后果**(更严重):
1. **每个响应多 2 个 token**(`12 30` 等),输出从 4 token 变 6+ token,大致 1.5× 慢
2. **模型对 action 语义的内部表征是错的** —— 它"以为"ACTION1 需要 (10, 20),那它选 ACTION1 时其实在想"点 (10,20)";真正的 game action 1 跟坐标无关,模型脑子里却带着假信息
3. **可能影响 reflection 推理**:未来加 reflection 时,模型可能会写出"ACTION1 at (10,20) makes the player go up"这种错误归因

### 8.4 修复方案(具体改动)

#### 方案 A:SYSTEM prompt 明确区分(最小改动)

把现在的最后一行:

```
Output ONE action token only (e.g. "ACTION3" or "ACTION6 12 30"). No JSON.
```

改成:

```
Output ONE action token. Only ACTION6 takes coordinates.

Valid formats:
  ACTION1            (no params)
  ACTION2            (no params)
  ACTION3            (no params)
  ACTION4            (no params)
  ACTION5            (no params)
  ACTION6 <x> <y>    (REQUIRED: x and y in 0..63)
  ACTION7            (no params)

Do NOT append numbers to ACTION1..5 or ACTION7 — they take no parameters.
```

预期效果:`<coords>` 出现率从 67.5% → < 10%。

#### 方案 B:在 USER prompt 的 [STATUS] 加上参数标注

```
[STATUS]
  legal actions: ACTION1 (no params), ACTION2 (no params), ACTION3 (no params),
                 ACTION4 (no params), ACTION5 (no params),
                 ACTION6 (x, y in 0..63), ACTION7 (no params)
```

#### 方案 C:不接受非 ACTION6 的尾部坐标,强制 retry

更严:`_coerce_action` 检测到 `ACTION1 10 20` 直接 reject → 进 fallback random。这会让 OutcomeLog 记录失败,模型下次有 history 提示"上次格式错了"。**风险**:fallback random 会污染 action_entropy 指标,容易把训练带偏。**建议先 A + B,如果效果还不够再加 C**。

### 8.5 实施清单

| # | 改动 | 文件 | 估时 |
|---|---|---|---|
| 1 | 改写 PLAY_SYSTEM 末尾 5 行(方案 A) | `prompts_v3.py:PLAY_SYSTEM` | 10 min |
| 2 | 改写 `_format_status` legal_actions 行(方案 B) | `prompts_v3.py:_format_status` | 15 min |
| 3 | 更新 `tests/test_prompts_v3.py` 的硬约束 grep 测试 | `tests/test_prompts_v3.py` | 10 min |
| 4 | 重跑 ar25 smoke,跑 audit_action6_misuse,看 `<coords>` 比例 | — | 10 min |

总:**~45 分钟**。

### 8.6 验证标准

跑完 ar25 后再跑 audit:

| 指标 | 当前 | 目标 |
|---|---:|---:|
| 非 ACTION6 带 coords 比例 | 67.5% | **< 10%** |
| ACTION6 缺 coords 比例 | 0% | 保持 0% |
| 严格 illegal action 比例 | 0% | 保持 0% |
| 平均响应 token 数 | ~6 | ~4 |

---

## 9. §5 + §8 两类问题的合并优先级

| 类别 | 严重性 | 修复成本 | 优先级 |
|---|---|---|---|
| §5 卡死检测缺陷(no-op streak 不识别) | 高(直接影响 RHAE) | 1.5-2.5h | **P0** |
| §8 Action 格式滥用(coords 乱加) | 中(token 浪费 + 内部表征错) | 0.75h | **P1** |

建议:**先做 §8(45 min,清掉 prompt 噪音),再做 §5**。理由:§8 修完后 prompt 表达更准确,§5 的 stuck signal 也会更有效。

---

*这份文档可以直接拿去 review。要不要按 §5 + §8 实施清单顺序动手?共 ~3 小时,然后重跑 ar25 看两个 audit 指标都是否改善。*

---

## 10. v3.1 升级清单(2026-05-14 决定的优先级)

经过对 5 个游戏 trace 的深度审计(`outputs/v3_visual_full/trace_audit.json`),除了 §5 / §8,还发现 3 个用户视角的关键缺陷:

| 缺陷 | 例子 | 当前 v3 怎么处理 |
|---|---|---|
| **C. 点击按钮(ACTION6 智能坐标)** | dc22 的交互按钮没人点对地方 | ACTION6 无坐标时填 random (x,y),命中 1×1 按钮概率 1/4096 |
| **D. 模式识别(形状/颜色相似 → 移动)** | 多数 ARC 游戏要求 "把红 L 推到绿 L 旁" | 完全没做,靠 LLM 自己看 bbox 推理(它做不到)|
| **E. 环境动态识别(bp35 海浪等)** | 海浪每步上涨,跟 action 无关 | 当作 ACTIVE 对象,模型会误以为是它的 action 触发的 |

### 10.1 优先级表(用户 2026-05-14 决定)

| 序号 | 任务 | 估时 | ROI | 备注 |
|---|---|---:|---|---|
| **P0-A** | §8 ACTION 格式滥用修复 | 45 min | **极高**(45 min 清掉 68% 噪音)| 已完整诊断 |
| **P0-B** | §5 卡死检测增强(state revisit + recent window)| 2-2.5 h | 高 | ar25 上 35 次 state 重访,完全没检测 |
| **P1** | **C. 点击候选**(`click_candidates.py`,ACTION6 智能默认 + `[CLICK CANDIDATES]` prompt 块) | ~3 h | 高 | 解 dc22 类游戏 |
| **P2** | **D. 模式检测**(`pattern_detector.py`,shape/color/overlap/mirror 检测,加 `[PATTERN HINTS]` 块) | ~6 h | 中(通用)| 给 LLM 具体 goal 假设可选 |
| **P3** | **E. 环境动态检测**(`environment_detector.py` + `[ENVIRONMENT]` 块) | ~4 h | 中 | bp35 海浪。**用户决定排最后**,因为前 4 个先做完更容易看到收益 |

### 10.2 ⭐ 小数据 canary 工作流(从今往后强制执行)

**用户明确要求**:
- **每改一次代码,先在小数据上验证**
- **canary 顺序:先测别的游戏,最后测 bp35**(bp35 最特殊,留到最后)
- **每次都生成 GIF 给用户验证**

具体 canary 阶梯:

| 关 | 跑什么 | 期望耗时 | 验证什么 |
|---|---|---:|---|
| **L0 单元测试** | `pytest tests/ -q` | 3-5s | 任何代码改动不破坏现有测试(336 个 baseline) |
| **L1 ar25 20 步** | `run_v3_eval --games ar25 --max-actions 20 --with-images` | ~20s | 基础探索 + GIF 验证 |
| **L2 cd82 20 步** | `--games cd82 --max-actions 20 --with-images` | ~20s | 简单对象游戏 |
| **L3 cn04 20 步** | `--games cn04 --max-actions 20 --with-images` | ~20s | 真有运动的游戏 |
| **L4 dc22 20 步** | `--games dc22 --max-actions 20 --with-images` | ~20s | 点击按钮类游戏(C 改完后会显著好转) |
| **L5 bp35 20 步** | `--games bp35 --max-actions 20 --with-images` | ~20s | **最后跑** —— 191 candidates 压力测试 + 环境动态 |
| **L6 全 5 game × 80 步 + GIF** | `--max-actions 80 --with-images` | ~6 min | 最终验收,生成 5 个 play.gif |

**每改一处,L0 + L1-L4 必须全部通过,才允许跑 L5。L5 通过才允许跑 L6**。

### 10.3 ⭐ GIF 生成是强制环节

从今往后每次跑 canary **都加 `--with-images`**,生成 `play.gif`,**让用户对照 GIF 验证 prompt 改动是否有视觉效果**。

GIF 在 markdown 报告里嵌入(用现有 `scripts/build_v3_visual_report.py --run <dir>`)。

### 10.4 当前进度

- [x] §10 优先级讨论 + 数据审计完成 (`outputs/v3_visual_full/trace_audit.json`)
- [ ] **P0-A**: §8 ACTION 格式 → 下一步开始
- [ ] **P0-B**: §5 卡死检测
- [ ] **P1**: 点击候选
- [ ] **P2**: 模式检测
- [ ] **P3**: 环境动态检测

### 10.5 提交 commit 前 checklist

- [ ] L0(pytest)绿
- [ ] L1-L4(4 个游戏 20 步)无 crash
- [ ] L1-L4 的 GIF 文件存在且大小合理(0.3-1 MB)
- [ ] 报告里 entropy / no-op rate / format-error rate 三个指标方向正确
- [ ] L5 bp35 通过
- [ ] L6 全 5 game × 80 步:对比 `outputs/v3_visual_full/` 关键指标至少持平或更好

---

*v3.1 工作就按 §10 顺序展开。先做 P0-A(45 min)+ 完整 L0-L5 canary(~5 min)+ L6 验收(~6 min)= 约 1 小时内拿出第一个改进 commit。*
