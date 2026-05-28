# 2026-05-28 v0 — Goal Generation (Module 1) — Spec for Validation

> **作用**: 把 Module 1 (Reflection Agent 写 goal_hypothesis) 的输入 / 输出 / 代码位置 / prompt / 之前测试不靠谱的地方 全部摊开,让你判断**该如何重新验证**。
> **作量**: 这个文件不是验证,只是 spec。验证方案等你看完决定。
> **触发**: 用户 2026-05-28 chat:"之前的几个分支的测试，我觉得有些不靠谱的地方"。

## 1. Module 1 (Goal Generation) 是什么

在 v3.2 / v4 架构里,Module 1 = **Reflection Agent 写 goal_hypothesis 这件事**。每个 env.step 之后跑一次 Reflection,它要回答:**这个游戏的赢条件可能是什么?**

数据流位置:

```
frame (64x64 grid)
  → Module 0 scipy perception → ObjectRecord list
  → 【Module 1: Reflection Agent reads + writes goal_hypothesis】  ← 本 doc 测这个
  → Module 2 parser evaluates hypothesis
  → Module 3 force-reject loop
  → Module 4 Action Agent picks ACTION using hypothesis
  → env.step → next frame → loop
```

## 2. Input — Reflection Agent 每步看到什么

来源代码: `arc_agent/agents/reflection_agent.py:ReflectionAgent.reflect_after_step` 调用 `arc_agent/prompts_v3_2.py:build_reflection_user_prompt`。

### 2.1 函数签名

```python
def reflect_after_step(
    self,
    *,
    knowledge: Knowledge,                          # 跨 round persistent
    step_summary: StepSummary,                      # 这一步的 action+outcome+reasoning
    step: Optional[int] = None,                     # 当前 step index
    max_steps: Optional[int] = None,                # 本 round 步数上限
    level: Optional[int] = None,                    # 当前 level
    total_levels: Optional[int] = None,             # 总 level 数
    state_name: Optional[str] = None,               # IN_PROGRESS / WIN / GAME_OVER 等
    legal_actions: Optional[list[str]] = None,      # 当前帧的 legal actions
    frame_objects: Optional[list[ObjectRecord]] = None,   # scipy 抽出的所有 obj
    layer_by_id: Optional[dict[int, Any]] = None,   # ACTIVE/STATIC/TEXTURE 分类
    object_memory: Optional[ObjectMemory] = None,    # 跨步 obj tracking
    outcome_log: Optional[OutcomeLog] = None,        # 跨步 action outcomes 统计
    object_relations: Optional[ObjectRelations] = None,   # same-color / same-shape / 距离对
    exploration_hint: Optional[str] = None,         # untried actions + uninteracted objs
) -> tuple[dict, str]:  # (delta, raw_response_text)
```

### 2.2 实际 prompt 长啥样 (传给 LLM 的 user message)

由 `build_reflection_user_prompt` 拼出来,8 个 block,按这个顺序:

```
[CURRENT KNOWLEDGE before this step]
  rounds_played: 1
  goal_hypothesis (low): match the moving yellow with the static yellow target
  rejected_goals: ["align two red squares vertically", ...]
  action_semantics:
    ACTION1: moves the yellow 1x1 (obj_002) UP by 3 cells
  rules: ["ACTION6: no-op on every tested coord"]
  failed_strategies: [...]
  ...

[EXPLORATION HINT]
  STUCK: same state seen 3x in last 6 steps
  Actions NOT tried this round: ACTION5, ACTION7
  ...

[STATUS]
  step: 47 / 100
  level: 1 / 8
  game state: IN_PROGRESS
  legal actions: ACTION1, ACTION2, ..., ACTION7

[ACTIVE]                                         ← 当前活动对象 (from object_memory)
  obj_032 (yellow 9x9) at (15, 39), bbox=[12,36,20,44]
  obj_033 (gray 9x9) at (15, 23), bbox=[12,18,20,26]
  ...

[OBJECT RELATIONS]                                ← from object_relations
  same-color groups:
    yellow: obj_032, obj_005
    gray:   obj_033, obj_006
  same-shape groups:
    9x9_size45: obj_032, obj_033, obj_005, obj_006
  closest pairs: obj_032-obj_005 dist=12, ...

[ACTION effects observed]                         ← from outcome_log
  ACTION1: 12 calls, 75% changed, primary_direction=UP majority
  ACTION6: 5 calls, 0% changed (no-op)
  ACTION2: 0 calls (untried)
  ...

[STEP SUMMARY]
  step 46 action=ACTION1 reasoning="moves yellow 1x1 UP, goal is top edge"
  outcome: frame_changed=True, primary_direction=UP, distance=3
  matches_reasoning: YES
  recent 3 steps: ACTION1, ACTION1, ACTION1

[ASK]
  Output STRICT JSON with three fields...
  goal_hypothesis_update should describe the WIN STATE in plain language...
  Patterns to use:
    * Same-color groups → goal often bringing them together
    * Static obj near edge → may be targets
    ...
```

(完整 ASK block 在 `arc_agent/prompts_v3_2.py:470-490`)

### 2.3 System prompt (Reflection Agent)

由 `arc_agent/prompts_v3_2.py:REFLECTION_SYSTEM`(约 100 行)。要点摘要:

- 你是 Reflection Agent,更新 Knowledge
- 看到 `KNOWLEDGE`、state context、本步 action+outcome+reasoning、最近几步、`[EXPLORATION HINT]`
- 输出 **3 字段** JSON: `goal_hypothesis_update`, `action_semantics_update`, `current_alert`
- goal_hypothesis_update 规则:
  - **描述 TARGET STATE 不是 action**(Good: "match every red dot with red target"; Bad: "ACTION1 should move up")
  - 不能重复 `rejected_goals` 里的(orchestrator 会 drop)
  - 跟当前 KNOWLEDGE.goal_hypothesis 一样 → 写 `null`(省 token)
  - 没真的猜想 → `null`(不能写 "unknown" / "none" / "")

## 3. Output — Reflection 每步产 (delta, raw_response_text)

### 3.1 期望 schema

```json
{
  "goal_hypothesis_update": "match the moving blue square to the static blue target",
  "action_semantics_update": {"ACTION1": "moves the yellow 1x1 (obj_002) UP by 3 cells"},
  "current_alert": ""
}
```

任何字段可以是 `null` / `{}` / `""` 表示"无更新"。

### 3.2 解析路径 (orchestrator 处理)

1. `parse_reflection_output(raw)` → `(delta_dict, parse_ok)`
   - 容忍 ```json fence
   - 容忍 leading prose
   - parse 失败 → `({}, False)` 当作 no-op
2. v4 引入的 schema validation: `--validate-hypothesis-schema {off, wide, strict}`
   - off: 接收任何 hypothesis text
   - wide: 必须能 `parse_goal_hypothesis` 成 GoalPredicate(任何 kind)
   - strict: 必须 kind ∈ {move_to_row, move_to_col, move_to_center, stack}
3. 通过 → `knowledge = knowledge.merged_with_delta(delta)`
4. R1 sentinel filter 拦截 `"unknown"/"none"/"tbd"/""/...`
5. R6 拦截以 "ACTION_X" 开头的 action-described goal

### 3.3 真实样本 (Phase 4 ar25 round 0 step 5)

```json
{
  "goal_hypothesis_update": "match the moving blue square to the static blue target",
  "action_semantics_update": null,
  "current_alert": ""
}
```

**问题:** ar25 frame 里**没有 blue 对象**,只有 yellow / gray / maroon / purple / tan。这是幻觉 hypothesis(BUG-2 之前发现的)。

## 4. 之前的"分支测试"为什么不靠谱(用户 2026-05-28 提的)

### 4.1 T-GOAL bench(2026-05-17 做的)

**bench 给的输入:**
```
"Goal: 'align the two yellow squares vertically in the left column'
 Current state:
   obj_A (yellow): row=12, col=0
   obj_B (yellow): row=32, col=0
 Is the goal achieved? A) YES ... B) NO ... C) NO ... D) NO"
```

**但 Module 1 的真实任务是:**
- **WRITE** goal_hypothesis from observation
- 不是 "given hypothesis, judge YES/NO"

→ T-GOAL 测的是 **Module 2 (Goal Recognition,parser/judge)**,**不是** Module 1。
→ Module 1 从来没有过"专门为它设计的 bench"。

### 4.2 T-DISCOVER 提过但没做

`docs/GLOSSARY_zh.md` 里 [[T-DISCOVER]]:"测 Reflection Agent 的首步推理能力:从单帧 0 prior 生成第一个 hypothesis"。**Status 是 "待建"** — 用户 2026-05-18 决定放弃 (`docs/project/2026-05-19-v0-v4_clean_baseline/architecture.md`)。

### 4.3 v4 Phase 4 cross-validation

`docs/project/2026-05-19-v0-v4_clean_baseline/module_validation.md` §2:
- 我数了 794 步,**全部 Reflection 写 align_any / match X with Y 类型,0 步写 directional kind**
- 这说明 **dialect distribution**,**没说 hypothesis 对不对**

### 4.4 annotation_request.md 没填

`docs/project/2026-05-19-v0-v4_clean_baseline/annotation_request.md` 列了 25 条 Phase 4 实际产生的 hypothesis,等用户标 YES/NO/PARTIAL。**到现在 (2026-05-28) 还没填**。

→ 我们目前对"Module 1 写出来的 hypothesis 对不对"**没有任何客观数据**。所有结论都基于 proxy(parseable rate、kind 分布、change_rate)。

### 4.5 距离 production 远的 4 个差异

| Bench 给的输入 | Production 实际输入 |
|---|---|
| 单帧 + 0 prior | 累积 N 步 KNOWLEDGE(action_semantics, rejected_goals, rules, ...) |
| 1 个写好的 hypothesis | Reflection 自由写,可能多步同一个、可能幻觉 |
| 固定 frame schema | scipy 抽的可能漏 obj、可能 unstable |
| 单步 single query | 跨步累积 outcome_log + relations |

→ 即便有 bench,bench-vs-production 分布偏移让 bench 数字不可信(2026-05-19 已经发现这是 6 模块共同问题)。

## 5. 一个**靠谱的** Module 1 验证需要什么

按 `docs/verify.md` §2.2 Module 1 真 PASS 定义: **hypothesis ≈ game 真实 win 条件**。

要测这个,需要:

### 5.1 Ground truth = game 真实 win 条件

- 来源:**人工玩通关后总结**(或者参考已有 walkthrough,但 ar25 / bp35 等私有 game 没有公开通关攻略)
- 用户(你)需要给出 e.g.: "ar25 level 1 win 条件 = 把两个 yellow obj 移到 (5, 5) 和 (5, 20),把两个 gray obj 移到对应位置"
- 每个 game 每个 level 都要标(工作量 = N levels × N games)

### 5.2 比对方法

| 方法 | 优 | 劣 |
|---|---|---|
| **逐字精确匹配** | 客观 | 太严:"yellow to col 0" vs "left edge for yellow" 都对但字面不同 |
| **语义匹配 by LLM judge** | 容忍措辞 | 引入新 model 误差(我们之前 LLM-judge 68% < parser 83%)|
| **结构匹配 by parser** | 跟我们 production 解析对齐 | parser 模式不全 → 漏判 |
| **人工 YES/NO/PARTIAL/UNSURE** | 最准 | 慢,主观 |

我们之前 [annotation_request.md] 提的是**人工 YES/NO/PARTIAL/UNSURE**。

### 5.3 测什么 input distribution

| Distribution | 怎么取样 |
|---|---|
| (a) 单帧 0 prior(round 开始) | 从 Phase 4 trace 取每 round 第 0 步的 frame + Reflection 写的 hypothesis |
| (b) 累积 N 步 KNOWLEDGE(中后期)| 取每 round 第 50 步 |
| (c) 多次 reject 之后 | 取 rejected_goals 累积 ≥ 3 之后的步 |

→ **覆盖三种 distribution 才能定位 bug**:首步弱(b1=DISCOVER 能力差)/ 累积后弱(BUG-1 dialect lock-in)/ reject 后弱(BUG-3 force-reject 没真改 mental model)。

## 6. 数据已经准备好,等你决定方案

`outputs/v4_phase4_g{1..5}_*/round_00/` 各有:
- 100 步 reflection_raw.txt(每步 Reflection 真实 JSON 输出)
- 100 步 knowledge_per_step.jsonl(每步 Knowledge 快照,含 goal_hypothesis 演化)
- 100 张 step_NNNN.png(每步 4-quadrant 可视化:before-frame、after-frame、Knowledge 摘要、metrics)
- play.gif(整 round 动画)

要做 validation,我可以从这些数据采样 + 准备 annotation 模板。**先等你说**:

1. **你愿意人工标 ground truth 吗?** 5 game,每 game 至少 3 个 level 的 win 条件
2. **你要测哪个 distribution**(单帧 0 prior / 累积 KNOWLEDGE / reject 后)?或者三个都覆盖
3. **比对方法用哪个**(LLM judge / parser / 人工)?
4. **每 game 取多少样本**(5? 10? 全 100)?
5. **跑新的 trace 还是用现有 Phase 4 trace**?(Phase 4 是 V4+propose `/no_think`;如果换 prompt 或换 mode,得重新跑)

回了我才动手。

## 7. 文件清单

```
docs/project/2026-05-28-v0-goal_gen_validation/
└── architecture.md             (本文件 - Module 1 spec)

(后续等你拍板 valid 方案后,会补:)
├── validation_protocol.md     (你的方案细化)
├── samples.json               (Phase 4 trace 采样数据)
├── annotation_form.md         (你填的地方)
└── report.md                  (跑完后的发现)
```

## 8. 引用

- `docs/verify.md` §3.2 Module 1 PASS 定义
- `docs/project/2026-05-17-v0-subtask-T-GOAL/report.md` (T-GOAL bench 测的不是 Module 1)
- `docs/project/2026-05-19-v0-v4_clean_baseline/module_validation.md` §2 (V4 Phase 4 重审)
- `docs/project/2026-05-19-v0-v4_clean_baseline/annotation_request.md` (之前提的 25 条人工标,没填)
- 代码: `arc_agent/agents/reflection_agent.py` + `arc_agent/prompts_v3_2.py`
- 系统 prompt: `prompts_v3_2.py:REFLECTION_SYSTEM` (line 105-203)
- 用户 prompt builder: `prompts_v3_2.py:build_reflection_user_prompt` (line 379-467)
- 全局词汇: `docs/GLOSSARY_zh.md`
