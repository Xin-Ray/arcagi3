# 项目术语表 (GLOSSARY_zh.md)

> 项目里出现过的所有术语 / 缩写 / 概念都收录在这里。**每个术语只在这里有定义,其他文档用 `[[term]]` 链接过来**。

最近更新: 2026-05-16

按字母排序。状态码:🟢 当前活的;🟡 参考 / 历史但被引用;⚫ 已弃用。

---

## A

### `Action Agent`

🟢 v3.2 双 agent 架构里负责「每步选 action」的 agent。读 perception + Knowledge + Reflection alert,输出 `reasoning: ... action: ACTIONx`(两行)。实现在 `arc_agent/agents/action_agent.py`。

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §5。
**相关**: [[Reflection Agent]]、[[Knowledge]]、[[ActionAgent.choose]]。

### `action_semantics`

🟢 `Knowledge` 字段。`dict[str, str]`,记录每个 ACTION 在当前游戏里的效果。例:`{"ACTION1": "moves the yellow 1x1 (obj_002) UP by 3 cells"}`。Reflection 写,Action 读。

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §3。
**相关**: [[Reflection Agent]]、[[BUG-2]]、[[BUG-9]]。

### `ACTION1..ACTION7`

🟢 ARC-AGI-3 的游戏动作枚举(`arcengine.GameAction`)。
- `ACTION1-4`:Up/Down/Left/Right(具体方向因游戏而异)
- `ACTION5`:Primary(interact / select / 等)
- `ACTION6`:Coordinate,需要 `(x, y) ∈ [0, 63]`,唯一 `is_complex()` 的 action
- `ACTION7`:Undo

**出处**: `TASK_OVERVIEW.md` + `CLAUDE.md`。

### `ar25` / `bp35` / `cd82` / `cn04` / `dc22`

🟢 我们目前实测用的 5 个公开 demo 游戏 ID 的短前缀。完整 ID 形如 `ar25-0c556536`。**ar25 是默认基准**(最多代码引用)。

**出处**: `data/splits/demo_555.json`。

---

## B

### `BUG-1..BUG-13`

🟡 v3.2 实施过程中的 13 个已知 bug,在 [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §13 集中列出。
- BUG-1: Action reasoning 91% 输出 N/A(没具体说预期)
- BUG-2: `action_semantics` 多面效果被覆写
- BUG-3: 相同 semantic 反复重写
- BUG-4: ACTION6 坐标没 memory(被 BUG-10 完整覆盖)
- BUG-5: round 撞 max_actions 提早结束(已修)
- BUG-6: Reflection 输出无 per-claim 置信度
- BUG-7: C stuck alert 阈值偏高
- BUG-8: `goal_hypothesis` 直接覆盖(已修,加 `rejected_goals`)
- BUG-9: `action_semantics` 写"an active object"无主语(已修)
- BUG-10: ACTION6 无 per-object 置信度(已修,引入 `click_targets`)
- BUG-11/12/13: click_targets 跨 round 持久化 + canonical rules + eager Reflection

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §13.

---

## C

### `change_rate`

🟢 一个 round / run 内 `frame_changed=True` 步数占总步数的百分比。**核心健康指标**。

- v3.2 baseline (mask off): 3-4%
- v2 canary (3×30 mask strict): 60-100%
- 当前 mask strict 3×200: 5-8%

**出处**: 几乎所有实验报告。

### `click_targets`

🟢 v3.2 BUG-10 引入的 ACTION6 命名 target bandit。把 ACTION6 从「4096 像素盲选」压缩为「~10 个 named target 的 confidence × tries 优先级」。实现 `arc_agent/click_targets.py`,prompt 块 `[CLICK TARGETS]`。

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §13.2 BUG-10。
**相关**: [[BUG-10]]、[[ACTION6]]。

### `CNN-small`

🟢 [[Predictor v0]] 的第四个架构。17 channel(16 色 one-hot + 1 action) × 64 × 64 → 32 → 64 → 64 → 1。~80K 参数。val AUC 0.894,比 hand-feature MLP +6 pp。

**出处**: [`reports/predictor_v0.md`](../outputs/reports/predictor_v0.md)。

---

## D

### `demo_555`

🟢 5+5+5 = 15 个游戏的固定划分(5 baseline + 5 train + 5 val),冻结在 `data/splits/demo_555.json`,**不允许重新随机**。

**出处**: `arc_agent/eval_split.py`、commit `2026-05-11`。

---

## F

### `F1 (intrinsic reward)`

🟡 [[RL v0]] 的内在奖励:agent 预测 `predicted_changes`,跟真实 `real_changes` 算 F1 ∈ [0, 1]。实现 `arc_agent/rewards.py:verify_prediction_f1`。**当前 v3.2 不用 F1**,被 parked 路线。

**出处**: [`architecture/rl_v0_zh.md`](./architecture/rl_v0_zh.md) §3。
**相关**: [[GRPO]]、[[parked]]。

### `frame_changed`

🟢 trace.jsonl 每行的核心 boolean:`grid_t != grid_{t+1}`?决定一步是 no-op 还是真正改变了状态。

**出处**: `arc_agent/observation.py`、所有 trace 写入处。

---

## G

### `Gate (G1, G2, ...)`

🟢 决策门。每份架构设计在 §6.2 列出实验需通过的 G1/G2/... 条件。失败 → 走对应 fallback 而不是继续推进。

**出处**: [`CONVENTIONS_zh.md`](./CONVENTIONS_zh.md) §3。

### `goal_hypothesis` / `goal_confidence`

🟢 `Knowledge` 字段。Reflection 推断的「这个游戏的目标」一句话假设 + 置信度 `low|medium|high`。

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §3。
**相关**: [[BUG-8]]、[[rejected_goals]]。

### `GRPO`

🟡 Group Relative Policy Optimization。RL 训练算法。[`architecture/grpo_v0_zh.md`](./architecture/grpo_v0_zh.md) 定位为**单游戏诊断实验**(只 ar25),不用于 Kaggle 提交。

**出处**: TRL 库,[`architecture/rl_v0_zh.md`](./architecture/rl_v0_zh.md) §6。

---

## H

### `Hungarian (object aligner)`

🟢 跨帧 object 匹配。`arc_agent/object_aligner.py` 用匈牙利算法匹配 frame_t / frame_{t+1} 的 [[ObjectRecord]],保持 obj_id 稳定。

**出处**: [`reference/object_pipeline_zh.md`](./reference/object_pipeline_zh.md)。

---

## K

### `Knowledge`

🟢 v3.2 核心 dataclass。**跨 round 持久化**的"学到的东西":`action_semantics` / `goal_hypothesis` / `rules` / `failed_strategies` / `rejected_goals` / `click_targets` / `current_alert` / `rounds_played` / `rounds_won` / `round_history`。 实现 `arc_agent/knowledge.py`。

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §3。
**相关**: [[Action Agent]]、[[Reflection Agent]]、所有 BUG-X。

---

## M

### `mask` / `R2 mask`

🟢 [[R2]] orchestrator-级硬规则。在 `env.step` 前拦截 LLM 选的 action,如果该 action 已经 `n_tried ≥ 5 AND n_changed == 0` → 替换。实现 `arc_agent/action_mask.py`。

- `compute_action_mask`:决定哪些 action 该 block
- `apply_action_mask`:决定替换成什么(优先级:untried > known-good > random)

**注意**:2026-05-16 发现「替换后 OutcomeLog 归因 bug」并已修(commit `01a7227`)。

**出处**: [`reference/v3_2_hardrules_results_zh.md`](./reference/v3_2_hardrules_results_zh.md)、commit `1bac4be`。

### `MLP-S / MLP-L`

🟢 [[Predictor v0]] 的两个 MLP 架构。S = 128 hidden,L = 512 hidden。val AUC 0.834 / 0.830,均饱和在 hand-feature 信号。

**出处**: [`reports/predictor_v0.md`](../outputs/reports/predictor_v0.md)。

---

## N

### `no_op_streak`

🟢 连续多少步 `frame_changed=False`。健康值 < 5,异常 > 20。`ActionAgent.no_op_streak()` 返回当前值。

**出处**: `arc_agent/agents/action_agent.py`。

---

## O

### `ObjectMemory`

🟢 v3 引入。一个 episode 内 UID-keyed 的 object 追踪。实现 `arc_agent/object_tracker.py:ObjectMemory`。每个 round reset。

**出处**: [`architecture/v3_zh.md`](./architecture/v3_zh.md) §5。

### `ObjectRecord`

🟢 一个 object 的描述:`bbox` / `mass` / `color` / `state ∈ {STATIC, ACTIVE, TEXTURE, CANDIDATE}` 等。`arc_agent/object_extractor.py` 用 `scipy.ndimage.label` 抽出来。

**出处**: [`architecture/v3_zh.md`](./architecture/v3_zh.md) §5、[`reference/object_pipeline_zh.md`](./reference/object_pipeline_zh.md)。

### `orch_override`

🟢 trace.jsonl 字段。当 [[R2 mask]] 替换了 LLM 的选择时,记录替换原因(如 `"untried ACTION5 over masked ACTION6"`)。空字符串表示没替换。

**出处**: `scripts/run_v3_multi_round.py`。

### `OutcomeLog`

🟢 一个 round 内所有 step 的 `(action, frame_changed, primary_direction, ...)` 记录。Mask 用它判断 `n_tried` / `n_changed`。Round 之间 reset。

**出处**: `arc_agent/action_inference.py`。

---

## P

### `parked`

🟡 状态标记:本路线代码留着 + 测试通过,但**当前不主动迭代**。例如 RL/GRPO/SFT 都被 park 过。要复活先 promote。

**出处**: [`CONVENTIONS_zh.md`](./CONVENTIONS_zh.md) §6。

### `perception (deterministic)`

🟢 v3 决策:**视觉用算法,不用 LLM**。scipy.ndimage.label + temporal_classifier + Hungarian aligner。源于 [`reference/object_pipeline_zh.md`](./reference/object_pipeline_zh.md) 实测 Qwen-VL 在 ar25 上 ~0% 而 scipy 100%。

**出处**: [`architecture/v3_zh.md`](./architecture/v3_zh.md) §0。

### `Predictor v0`

🟢 frame-change 预测器。给 `(state, action)` 输出 `P(frame_change)`。四个架构对比:LogReg / MLP-S / MLP-L / CNN-small。val AUC 0.80-0.89。

**出处**: [`architecture/predictor_v0_zh.md`](./architecture/predictor_v0_zh.md)、[`reports/predictor_v0.md`](../outputs/reports/predictor_v0.md)。

### `primary_direction` / `primary_distance`

🟢 一步 env.step 的主方向(UP/DOWN/LEFT/RIGHT)+ 距离。由 orchestrator 通过 `object_aligner` 计算。trace.jsonl 字段。

**出处**: `scripts/run_v3_multi_round.py:_compute_primary_change`。

---

## R

### `R1..R7` (hard rules)

🟢 v3.2 orchestrator-级硬规则。**Prompt 只能劝,orchestrator 才能管**的设计原则下产生。
- **R1** Knowledge sentinel filter:拒绝 `"unknown"/"none"/"tbd"` 作为 goal_hypothesis
- **R2** Knowledge-driven action mask:见 [[mask]]
- **R3** ActionAgent 卡死强制 explore:`no_op_streak ≥ 5` 或 `state_revisit ≥ 5` → 选 untried
- **R4** 自相矛盾过滤:`rules_append` "ACTION_X no effect" 跟 action_semantics 矛盾 → drop
- **R5** failed_strategies 交叉污染过滤:goal_hypothesis 不能等于 failed_strategies 字面量
- **R6** action-described goal 过滤:goal_hypothesis 不能以 "ACTION_X" 开头
- **R7** [LOW-PRIORITY ACTIONS] prompt 块:把 mask 结果给 LLM 看

**出处**: commits `1bac4be` (R1+R2+R3)、`7a315c7` (R5)、`d66ce37` (R4+R6+R7)。
**详细对比**: [`reference/v3_2_hardrules_results_zh.md`](./reference/v3_2_hardrules_results_zh.md)。

### `Reflection Agent`

🟢 v3.2 双 agent 架构里负责「每步反思 + 更新 Knowledge」的 agent。读上一步 outcome + 当前 Knowledge,输出 incremental delta JSON。每步调一次(per-step,不是 per-round)。

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §4。

### `rejected_goals`

🟢 `Knowledge` 字段(BUG-8 修复时加的)。被 Reflection 改口替换掉的旧 `goal_hypothesis` 存这里(cap 10),Reflection 下次不会再提同样的目标。

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §13 BUG-8。

### `RHAE` (scoring)

🟢 ARC-AGI-3 评分:`S = min(1.0, h/a)²`。`h` = 第二好人类的 action 数,`a` = agent 的 action 数。Quadratic penalty。

**出处**: `TASK_OVERVIEW.md`。

### `RL v0`

🟡 早期 RL 设计(intrinsic F1 reward + GRPO)。**parked**;现在不跑训练。

**出处**: [`architecture/rl_v0_zh.md`](./architecture/rl_v0_zh.md)。

### `round` / `episode`

🟢 一次完整的 game play (reset → ... → WIN / GAME_OVER / max_actions)。v3.2 multi-round 调度:跨 round Knowledge 保留,ObjectMemory + OutcomeLog 每 round 重置。

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §1。

---

## S

### `scipy.ndimage.label`

🟢 v3 perception 的根:64×64 grid → 连通分量。比 Qwen-VL 在 ar25 上 100% vs ~0% 的对比由 [`reference/object_pipeline_zh.md`](./reference/object_pipeline_zh.md) 实测。

**出处**: `arc_agent/object_extractor.py`。

### `state_revisit_count`

🟢 当前 frame_hash 在本 episode 已经出现过几次。`> 5` 触发 R3 / orchestrator stuck alert。

**出处**: `arc_agent/agents/action_agent.py:state_revisit_count`。

### `step PNG` / `viz_v3_2`

🟢 每 step 一张 512×286 的 composite PNG:左半 256×256 grid (4× 放大),右半 256×286 文字 panel (action / reasoning / reflection delta / alert / matches_reasoning)。`arc_agent/viz_v3_2.py:compose_step_image_v32`。

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §5.5。
**相关**: [[CNN-small]] 用 PNG 反解码出原 grid。

### `STATIC / ACTIVE / TEXTURE / CANDIDATE`

🟢 [[ObjectRecord]] 的 state 枚举。
- STATIC:静止 + 单帧
- ACTIVE:跨帧动过
- TEXTURE:大面积均匀背景(已被 prompt 过滤)
- CANDIDATE:还在判定

**出处**: `arc_agent/temporal_classifier.py`。

### `StochasticGoose`

⚫ ARC-AGI-3 preview 第一名(Tufa Labs)。CNN 预测 P(frame_change) + RL。Preview 12.58%,全集 0.25%(过拟合)。我们 [[Predictor v0]] 借鉴它的核心信号。

**出处**: [`reports/predictor_v0.md`](../outputs/reports/predictor_v0.md)。

### `Symbolica Agentica`

⚫ ARC-AGI-3 public demo 25 game 36.08%(7/25 通关)。Orchestrator + subagent 架构,基于 Agentica SDK + 大模型。我们的 v3.2 Knowledge / Reflection 思路接近。

**出处**: web search 2026-05-15 + [`reports/grpo_v0_ar25_plan.md`](../outputs/reports/grpo_v0_ar25_plan.md)。

---

## T

### `Tier 1 SFT`

🟡 [`architecture/sft_tier1_zh.md`](./architecture/sft_tier1_zh.md)。在合成数据上 LoRA fine-tune Qwen2.5-VL-3B 的空间推理基础能力(T1-方向 / T2-算术 / T3-相对位置 / T4-格式 / T8-一致性)。

**状态**:第一轮失败(planning probe 完全没修 + gsm8k 退化 -14.7pp,根因 T4 模板锁死)。F3 修法待跑。

**出处**: [`architecture/sft_tier1_zh.md`](./architecture/sft_tier1_zh.md) §12 复盘。

### `TextAgent` (v3)

🟡 v3 单 agent 实现。`arc_agent/agents/text_agent.py`。v3.2 拆出 [[Action Agent]] + [[Reflection Agent]] 后被 deprecate,代码留着可用。

**出处**: [`architecture/v3_zh.md`](./architecture/v3_zh.md) §5。

### `trace.jsonl`

🟢 每个 round 一份的逐步记录。每行一个 JSON,字段含 `step / action / frame_changed / orch_override / reasoning / reflection_delta / ...`。

**出处**: `arc_agent/baseline.py:play_one_with_trace`、`scripts/run_v3_multi_round.py`。

---

## V

### `v3`

🟡 单 agent 架构。scipy perception + Qwen text-only。被 v3.2 取代,代码留着。

**出处**: [`architecture/v3_zh.md`](./architecture/v3_zh.md)。

### `v3.2`

🟢 当前活的架构。Action Agent + Reflection Agent + 跨 round Knowledge + click_targets bandit + R1-R7 硬规则。

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md)。

### `v3.2 canary` (v2 canary)

🟡 commit `1bac4be` 时的实测状态:R1+R2+R3,无 click_targets,无 canonical rules,prompt 17 块。在 ar25 3×30 step 上 change_rate 60/100/97%。**目前已知的最佳状态**(虽然没通关 levels)。

**出处**: [`reference/v3_2_hardrules_results_zh.md`](./reference/v3_2_hardrules_results_zh.md)。
**相关**: 当前 v3.2 跟它的 diff = 12 commits / 6300 行。

---

## 文档历史

- 2026-05-16 初稿。涵盖到 commit `01a7227`。每次架构 / 报告新增一个概念,在这里加一条。
