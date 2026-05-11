# ARCHITECTURE_RL — RL with Intrinsic Reward 路线设计

最近一次更新:2026-05-11
状态:**方案设计** — 替代 ROADMAP.md Phase 2 BC 子计划

> **本文与 `ARCHITECTURE.md` 的分工**:`ARCHITECTURE.md` 写"模块职责"(稳定不变);本文写"训练 + 推理的具体步骤"(可演进)。本文是 RL 路线的**核心实现文档**,后续代码以此为蓝本。

---

## 战略决策(2026-05-11)

| 决策点 | 原 ROADMAP | 新方案 | 理由 |
|---|---|---|---|
| 监督学习预热(BC) | Phase 2:人类 trace BC | **暂停** | trace 收集脚本未实现,前置成本高;先看 zero-shot Qwen 表现 |
| LLM silver-label | 备选数据源 | **暂停** | 准确率低 + API 贵 |
| 主线 | BC → RL(PPO+KL)→ 提交 | **zero-shot Qwen baseline → RL with intrinsic reward → 提交** | 跳过 BC,把 verifier-F1 当密集 reward,直接训 |
| 训练 RL 算法 | PPO | **GRPO** | A4500 20GB 装不下 PPO critic |
| 评估切分 | demo 25 整体训完一次性评 | **5-5-5 切分** | 小步快跑;避免一次性投入失败 |

**如果 baseline F1 < 0.1 → 回退**:这时再考虑做 BC 或换 backbone。BC 留作 fallback,不死。

---

## 第 0 节:概念词典

| 术语 | 一句话定义 | 项目里具体是什么 |
|---|---|---|
| **Agent** | 在游戏里做决策的程序 | `arc_agent/agents/vlm.py` |
| **Policy** | "状态 → 动作"的函数 | Qwen2.5-VL-3B 模型 |
| **State / Frame / Grid** | 游戏当前的样子 | 64×64 numpy 数组,值 0–15(颜色) |
| **FrameDataRaw** | env 返回的 pydantic 对象 | 见 §0.1 |
| **Action** | agent 能做的操作 | `ACTION1..ACTION7` |
| **Episode / Rollout** | 一局游戏 | reset 到 WIN/GAME_OVER/超步数 |
| **Step** | 一步 | `(s_t, a_t, s_{t+1})` |
| **GRPO** | RL 算法 | DeepSeek-R1 用的,比 PPO 省 50% 显存 |
| **LoRA / QLoRA** | 训"小补丁"参数,主模型冻结 | Qwen 3B 主体不动,只训 ~1M 参数 |
| **Rule Table** | agent 对游戏规则的"理解清单" | list of dict,最多 20 条 |
| **Verifier** | "比较预测和真实" 函数 | `verify_prediction_f1()` |
| **Predicted Diff** | Qwen 猜的"哪些 cell 会变" | `set[(row, col, new_color)]` |
| **Real Diff** | 真实变化集合 | 从 `s_t`, `s_{t+1}` 算出 |
| **F1** | 预测集 vs 真实集 的吻合度 ∈ [0,1] | precision/recall 调和平均 |
| **Reward** | 给 RL 训练的打分 | 标量 |
| **Intrinsic Reward** | agent 自己产生的 reward | 密集,每步都有(主要靠 F1) |
| **Extrinsic Reward** | env 给的 reward | 稀疏,通关 +1 |
| **G_base / G_train / G_val** | 5-5-5 三组游戏 | 见 §5 |

### 0.1 FrameDataRaw 详解

`arcengine` SDK 定义的 pydantic 模型,`env.reset()` / `env.step()` 都返回这个对象:

```python
FrameDataRaw(
    game_id            = "ls20",
    state              = GameState.NOT_FINISHED,  # 或 WIN / GAME_OVER
    levels_completed   = 0,                       # 已通过的关卡数
    win_levels         = 7,                       # 该游戏总关卡数(易混淆!)
    available_actions  = [1, 2, 3, 4],            # 这一步合法的 ACTION id
    frame              = [np.ndarray (64, 64)],   # 帧序列,动画可能多帧
    guid               = "abc-...",               # 本局唯一 id
    full_reset         = False,
    action_input       = ActionInput(...)
)
```

我们只用 5 个字段:
- `state` — 是否结束
- `levels_completed` / `win_levels` — 关卡进度(注意 win_levels 是**总关卡数不是胜利数**)
- `available_actions` — 本步合法动作集
- `frame[-1]` — 最新一帧 `(64, 64)`,每 cell 是 0–15 颜色编号

---

## 第 1 节:Prompt 设计(6 段)

每步给 Qwen 的 prompt:

```
[system]
你是 ARC-AGI-3 游戏 agent。每步需要:
(1) 识别画面中的实体
(2) 反思上一步预测的对错
(3) 预测这一步选某动作后会发生什么
(4) 选一个动作
(5) 如果发现新规则,输出新规则

已知规则:
{rule_table 序列化为 JSON,可能为空}

[user]
<image: 当前 grid 512×512 PNG>

【段 1: 场景元信息】
游戏: ls20
关卡: 1 / 3
状态: NOT_FINISHED
可用动作: ACTION1, ACTION2, ACTION3, ACTION4

【段 2: 历史动作】
最近 3 步: ACTION3, ACTION1, (start)

【段 3: 上一轮反思】   ← 新增
上次 predicted_diff: [(10, 3, 2)]
上次 real_diff:      []
F1 = 0.0   (预测完全错,P 没动)
推断: ACTION3 不总是向左移,可能被障碍物挡

【段 4: 实体识别请求】   ← 新增
请识别画面中所有实体:
- shape: 占据哪几个 cell
- color: 颜色编号
- count: 该色块出现几次
- type: 推测属于哪类 (player / wall / goal / enemy / movable_obj / ...)
- function: 你认为它的作用
- position: 中心 cell 坐标

【段 5: 输出格式】
请输出 JSON:
{
  "entities":       [{shape, color, count, type, function, position}, ...],
  "reflection":     "上轮推断(自然语言一句)",
  "predicted_diff": [{row, col, to_color}, ...],
  "chosen_action":  "ACTIONx",
  "new_rule":       null | {trigger_action, subject_color, ..., confidence}
}
```

### 1.1 新增 3 段的作用

| 段 | 作用 | 是否影响 reward |
|---|---|---|
| 反思(段 3) | 让 Qwen 显式利用上轮 F1 信号,避免重复犯错 | 间接(它会让下一步 F1 更高) |
| 实体识别(段 4) | 把"看图" 拆成结构化感知,比 free-form 描述可控 | 可选辅助:entity 自洽度 +0.05 |
| 输出格式(段 5) | 强制结构化,verifier 才能 parse | **JSON parse 失败 = -0.5 reward**(见 §3) |

### 1.2 一次输出多个动作?

| 风格 | 描述 | 何时用 |
|---|---|---|
| A. 开环多动作 | 一次输出 `[ACTION3, ACTION1, ACTION3]` 全部执行 | ❌ 误差累积,verifier 放大问题 |
| B. 闭环 imagined trajectory | 输出 `[(action, predicted_after), ...]`,只执行第一个 | baseline 跑通后试 |
| **C. 单步**(默认) | 每次只输出 1 动作 | ✅ **第一版** |

### 1.3 一次输入多张图?

Qwen2.5-VL 支持 interleaved 多图。每张 512×512 ≈ 334 visual tokens,32K context 可塞 5–10 张图。

| 姿势 | 输入 | 用途 |
|---|---|---|
| 单图 | 当前帧 | **第一版** |
| 双图 | 上一帧 + 当前帧 | 让模型"看到"变化(若单图 F1 太低,首选升级) |
| 多图 | 当前帧 + 多种假想后帧 | 内部 lookahead(可选未来) |

---

## 第 2 节:训练流程(RL only)

### 2.1 训练总览

```
原始 Qwen2.5-VL-3B(HuggingFace,加 LoRA 未训)
       │
       │ GRPO with Intrinsic Reward
       │  - 训练游戏: G_train(5 个)
       │  - 每 step 跑 K=4 局 rollout
       │  - 奖励组成: F1(密集) + 通关(稀疏) + JSON 合规(罚分)
       │  - 训练: ~3-4 GPU-days
       ▼
checkpoints/grpo_final/
```

### 2.2 GRPO 一步详解

```
Step 1. 当前 Qwen 在 G_train 上跑 K=4 局 rollout

Step 2. 每一步算 reward(见 §3)

Step 3. 4 局之间相对比较(这是 GRPO 的"G"=Group):
        - 哪几局奖励高 → 鼓励对应轨迹的动作分布
        - 哪几局奖励低 → 抑制

Step 4. 反向传播 → 更新 Qwen 的 LoRA 参数

Step 5. 每 N 步在 G_val 上跑评估(early-stop 用)
```

**为什么 GRPO 不是 PPO**:PPO 需要单独的 critic 网络估"每步值多少分",A4500 20GB 装不下 critic + actor + 4-bit Qwen + LoRA。GRPO 用一组 rollout 的相对排名替代 critic,显存省 ~50%。

---

## 第 3 节:Reward 设计

```python
def compute_reward(step) -> float:
    r = 0.0

    # 主项: 通关(稀疏)
    if step.state == WIN:
        r += 1.0

    # 主项: F1(密集,每步都有)
    if step.parsed_json_ok:
        r += 0.2 * step.f1                  # 0 .. 0.2

    # 惩罚: JSON 不合法 → 模型学不会输出格式
    if not step.parsed_json_ok:
        r -= 0.5

    # 惩罚: 输出非法动作(不在 available_actions)
    if step.action not in step.available_actions:
        r -= 0.3

    # (可选) 辅助: entity 识别自洽
    if step.entity_recognition_consistent:
        r += 0.05

    return r
```

### 3.1 为什么 JSON 罚分这么重要

- 训练初期 Qwen 会乱输出,若只 fallback 到随机不罚,**它永远学不会输 JSON**
- 给 −0.5 罚分后,几轮内模型稳定输出合法 JSON
- 推理阶段仍保留 fallback(rollout 不能崩),**仅训练时罚**

### 3.2 推理时 F1 干什么

- 推理时 F1 **不是 reward**(那时不在训练)
- F1 仅用于决定 `rule_table` 怎么更新:
  - F1 ≥ 0.8 → 增加现有规则 evidence_count
  - F1 < 0.5 → 若 Qwen 提了 new_rule,加进去;confidence 跌破 0.3 的旧规则 evict
- rule_table 上限 20 条,超过 evict 最低 confidence 的

---

## 第 4 节:推理流程(9 步)

> 加载训后 checkpoint,在新游戏上玩。每步执行下面 9 件事。

```
Step 1.  s_t = latest_grid(frame)
Step 2.  image = grid_to_image(s_t, scale=8)
Step 3.  prompt = build_prompt(image, rule_table, history, last_round_state)
            ↑ 含上轮反思和 entity 请求(§1)
Step 4.  response = qwen.generate(image, prompt)
         parse JSON → entities, reflection, predicted_diff, chosen_action, new_rule
            ↑ parse 失败 → fallback 到随机动作 + 空 predicted_diff
Step 5.  frame_next = env.step(chosen_action, data=...)
         s_tp1 = latest_grid(frame_next)
Step 6.  real_diff = diff_grid(s_t, s_tp1)
Step 7.  f1 = verify_prediction_f1(predicted_diff, real_diff)
Step 8.  update_rule_table(f1, new_rule, rule_table)
Step 9.  last_round_state = (predicted_diff, real_diff, f1)
         s_t = s_tp1
         → 回到 Step 1
```

**终止条件**(3 选 1):
- `state == WIN` → 通关
- `state == GAME_OVER` → 失败
- `step_count ≥ 5 × level_baseline_actions` → 主动 give-up(评测端也会截断)

### 4.1 数据流图

```
        s_t (64×64 numpy)
          │
          ▼
   grid_to_image() → image (512×512 PNG)
          │
   build_prompt(image, rule_table, history, last_round_state)
          │
          ▼
   ┌──────────────────┐
   │   Qwen2.5-VL-3B  │  ← 训后 LoRA
   └────────┬─────────┘
            │ JSON
   ┌────────┼────────┬───────────┬─────────────┐
   ▼        ▼        ▼           ▼             ▼
entities  reflection predicted   chosen      new_rule (可 null)
                    _diff       _action
                                  │
                                  ▼
                          env.step() → s_{t+1}
                                  │
                                  ▼
                          real_diff (从 s_t, s_{t+1})
                                  │
            ┌─────────────────────┘
            ▼
      F1 score
            │
      update rule_table
            │
            ▼
    last_round_state ────────► 下一步 prompt 用
            │
       s_t = s_{t+1}
            │
            ▼
       下一步循环
```

---

## 第 5 节:5-5-5 实验计划

### 5.1 三组游戏切分

```
demo 25 (公开)
  │
  ├── 5 → G_base   : baseline,zero-shot Qwen + GIF 可视化
  ├── 5 → G_train  : GRPO 训练         (与 G_base 不重叠)
  └── 5 → G_val    : 训前训后都跑,看进步  (与 G_base, G_train 都不重叠)
剩 10 → 留给最终扩展评估
```

### 5.2 阶段 1:Baseline(不训练,先看 Qwen 在干什么)

```
scripts/run_baseline.py
  - 对 G_base 中每个游戏跑 1 episode
  - 用当前 prompt + LoRA 未训的 Qwen
  - 每步保存:
      * 当前 grid 渲染图(PNG)
      * Qwen 看到的完整 prompt
      * Qwen 输出的完整 JSON
      * F1 + parse_ok 标志
      * 真实 next_grid 渲染图
  - 输出:
      * runs/baseline_<ts>/<game_id>/step_<n>.png   合成图
      * runs/baseline_<ts>/<game_id>/play.gif       整局 GIF
      * runs/baseline_<ts>/summary.json             各游戏 mean F1/RHAE
```

**合成图布局**(每步一张 PNG):
```
┌──────────────────────────────────────────────────┐
│  Game: ls20  |  Step 5  |  F1: 0.62              │
├─────────────────────┬────────────────────────────┤
│   当前 grid          │   Qwen 预测的 diff(叠层)    │
├─────────────────────┼────────────────────────────┤
│   真实下一帧         │   Qwen 输出 JSON(文字)      │
└─────────────────────┴────────────────────────────┘
```

#### Baseline 的 Hypothesis(预注册,HEI 规矩)

| 指标 | 预测 | 含义 |
|---|---|---|
| Mean F1 在 G_base | **≥ 0.30** | Qwen zero-shot 有视觉理解 |
| Mean RHAE 在 G_base | **≤ 0.05** | zero-shot 不大可能通关 |
| JSON parse 成功率 | **≥ 0.70** | Instruct-tuned Qwen 指令遵循能力 |

#### Iteration trigger

- **F1 ≥ 0.3 且 parse ≥ 0.7** → 进入阶段 2 GRPO 训练
- **F1 < 0.1** → Qwen zero-shot 太弱:试双图输入 / few-shot prompt;仍不行 → 回退 BC
- **parse < 0.5** → 改 prompt 模板,加 1-2 个 in-context 示例
- **0.1 ≤ F1 < 0.3** → 中间态:消融实验(单图 vs 双图,有 entity 请求 vs 无)

### 5.3 阶段 2:训练(GRPO on G_train)

```
scripts/run_grpo.py
  - 在 G_train 5 个游戏上轮训
  - 每 GRPO step: 从 G_train 随机选 1 个游戏跑 K=4 局
  - 每 N=50 step 在 G_val 上跑评估(不是 G_base,避免数据污染)
  - Early stop: G_val mean RHAE 连续 3 次没涨
  - 训练总时长 ≤ 4 GPU-days
```

#### 训练 Hypothesis

- G_val mean RHAE 从 baseline 提升 **≥ 0.05** → 算成功
- G_train mean RHAE 提升 **≥ 0.10** → sanity check(应该比 val 涨得多)

### 5.4 阶段 3:Validation(训后对比)

```
scripts/run_validation.py
  - 加载训后 checkpoint
  - 在 G_val 5 个游戏上跑 1 episode
  - 同样生成 GIF + summary
  - 与训前 G_val 数据横向比较
```

**评估矩阵**:

|         | G_base       | G_train             | G_val               |
|---------|--------------|---------------------|---------------------|
| 训前    | F1_b (阶段1) | F1_t_pre (临时跑)   | F1_v_pre (临时跑)   |
| 训后    | —            | F1_t_post (阶段3)   | F1_v_post (阶段3)   |

**关键判断**:
- `F1_v_post > F1_v_pre` → 训出了真本事(在没见过的游戏上有进步)
- `F1_t_post > F1_t_pre` but `F1_v_post ≈ F1_v_pre` → **过拟合**(StochasticGoose 失败模式重现)
- 都没涨 → 路线有问题,诊断

---

## 第 6 节:实施顺序

按依赖关系:

| # | 任务 | 文件 | 阻塞 | 预计 |
|---|---|---|---|---|
| 1 | `grid_to_image()` 实现+测试 | `arc_agent/observation.py` | 2-6 | 0.5 day |
| 2 | `verify_prediction_f1()` 实现+测试 | `arc_agent/rewards.py` | 3-6 | 0.5 day |
| 3 | VLMAgent 推理 loop(含 entity + reflection) | `arc_agent/agents/vlm.py` | 4-6 | 1 day |
| 4 | GIF 合成工具(per-step 合成图 + ffmpeg/PIL → gif) | `arc_agent/viz.py` | 5 | 0.5 day |
| 5 | **跑 baseline + 验证 Hypothesis** | `scripts/run_baseline.py` | 6 | 0.5 day |
| 6 | GRPO 训练脚本(用 `trl.GRPOTrainer`) | `scripts/run_grpo.py` | 7 | 2 days |
| 7 | 跑 GRPO + Validation | — | — | 4 days |

**总预算**:~10 天(基础设施 3 days + baseline 验证 0.5 day + 训练 4 days + 评估 0.5 day + buffer 2 days)。

---

## 第 7 节:FAQ

**Q1: 为什么暂停 BC?**
A:(1) trace 录制脚本未实现,开始 BC 要先建数据管线,前置成本高;(2) LLM silver-label 准确率低 + API 贵;(3) zero-shot Qwen + RL 路线如果 baseline F1 ≥ 0.3 就证明 Qwen 看得懂,**不需要 BC**。如果 baseline F1 < 0.1 再回头考虑 BC。BC 留作 fallback,不死。

**Q2: Qwen zero-shot 真能跑通吗?**
A: 风险点。Hypothesis 预测 F1 ≥ 0.3,实际可能更低。第一阶段 baseline 就是要测这个数字。所以**先做 baseline,再决定要不要 BC**。

**Q3: 一次输出 3 个动作能省时间吗?**
A: 能,但**开环模式不推荐**(误差累积)。建议第一版单步,baseline 跑通后试闭环 imagined trajectory(§1.2 风格 B)。

**Q4: F1 推理时是 reward 吗?**
A: **不是**。F1 训练时是 reward,推理时只用来决定 rule_table 是否更新。同一个数字,两种用途。

**Q5: rule_table 训练时用吗?**
A: 第一版**不用**(只在推理时用)。先证明"纯 GRPO + F1 reward"涨分,再考虑把 rule_table 也放进训练 prompt。复杂化要分步加。

**Q6: JSON 罚分会让模型一开始就崩吗?**
A: 不会。Qwen2.5-VL-Instruct 经过指令微调,初始 JSON 输出能力较强,parse 成功率应 ≥ 0.7。罚分是"防止退步",不是"从 0 教会"。

**Q7: 5-5-5 用哪些游戏?**
A: 实施 §6 第 5 步前先 `arc.list_games()` 拉取 demo 25 完整列表,然后按 game_id 字母序固定切前 5 / 中 5 / 后 5(可复现,不随机)。

---

## 第 8 节:文档现状(2026-05-11 重组)

为了保持简单,旧的七件套(ARCHITECTURE / LIBRARY / PAPER / ROADMAP / RESEARCH / EXPERIMENTS / CODE_MAP / README)已**整体归档**到 `archive/docs_2026-05-11/`,**只读保留作历史**。

当前 `docs/` 目录**只有本文件**。新代码、新决策、新实验的所有正式文档需求,统一写在本文件内或对应的代码注释 / 测试断言里。项目根目录的 `README.md` 是 3 分钟接管入口,本文件是深度参考;两者职责不重叠。

CLAUDE.md 的"库优先"规则仍然强制:任何会被复用的函数必须进 `arc_agent/`,在源码注释里写清签名和用途;不再要求维护单独的 LIBRARY.md 索引。

---

## 第 9 节:实施步骤(可执行清单)

每步给出:**文件路径 / 关键签名 / 测试 / Acceptance criteria / 当前状态**。完成一步 → 把状态从 ⬜ 改成 ✅ 即可,无需另外更新文档。

库优先约束(每一步都适用):任何要复用的函数必须放进 `arc_agent/` 包,**不允许**写在 script 里;脚本只负责 I/O 编排。

### Step 1 — `grid_to_image()`  ✅(已合入 main)

- **文件**:`arc_agent/observation.py`
- **签名**:`def grid_to_image(grid: np.ndarray, scale: int = 8) -> PIL.Image.Image`
- **行为**:64×64 int 数组(0..15)→ 512×512 RGB,使用 ARC 标准 16 色调色板
- **测试**:`tests/test_observation.py::test_grid_to_image_*`(形状、调色板、scale 参数)
- **Acceptance**:测试全过 + 在 baseline 脚本中调用不报错

### Step 2 — `rewards.py` 三件套  ✅(已合入 main)

- **文件**:`arc_agent/rewards.py`
- **签名**:
  - `changes_to_set(changes: Any) -> ChangeSet`(容错解析 Qwen JSON)
  - `real_changes(s_t, s_tp1) -> ChangeSet`(从两帧计算真实变化)
  - `verify_prediction_f1(predicted, real) -> float`(F1 ∈ [0,1],含空集边界)
- **测试**:`tests/test_rewards.py`(全等、全错、部分重叠、空集、shape mismatch)
- **Acceptance**:`pytest tests/test_rewards.py -q` 全过

### Step 3 — `VLMAgent` 推理 loop  ✅(2026-05-11 合入 main)

- **文件**:`arc_agent/agents/vlm.py`(新建);依赖已存在的 `arc_agent/vlm_backbone.py` 抽象 — 若该文件不存在则同步建立
- **关键签名**:
  - `class VLMAgent: def __init__(self, *, model_path: str | None = None, max_rules: int = 20); def choose(self, latest, history) -> GameAction; def reset(self) -> None`
  - 内部:`_build_prompt(...)`(§1 6 段)、`_parse_response(text)→dict`(JSON 容错)、`_update_rule_table(f1, new_rule)`(§3.2)
  - backbone:`arc_agent/vlm_backbone.py::load_model(quantize: str | None = "4bit") -> (model, processor)`、`generate(model, processor, image, prompt, *, system) -> str`
- **测试**:`tests/test_agents_vlm.py`
  - 必测(无 GPU):`_build_prompt` 6 段顺序齐全、`_parse_response` 在合法/非法/部分缺失 JSON 上不抛异常、`_update_rule_table` 上限 20 条 + evict 逻辑
  - 可选(GPU,`@pytest.mark.gpu`):一次 `choose` 端到端
- **Acceptance**:无 GPU 测试全过 + 在 baseline 脚本里跑 1 局不崩

### Step 4 — GIF 合成工具  ✅(已合入 main)

- **文件**:`arc_agent/viz.py`(库)+ `scripts/make_gif.py`(脚本入口)
- **签名**:
  - `compose_step_image(grid_now: np.ndarray, predicted_diff: ChangeSet, grid_next: np.ndarray, json_text: str, *, header: str) -> PIL.Image.Image`(§5.2 四宫格布局)
  - `write_gif(frames: list[PIL.Image.Image], out_path: str | Path, *, fps: int = 2) -> None`
- **测试**:`tests/test_viz.py`
  - `compose_step_image` 输出非空 + 尺寸正确(宽 ≥ 2× 单图,高同理)
  - `write_gif` 在 tmp_path 写出文件,大小 > 0
- **Acceptance**:测试过 + baseline 跑完后 `outputs/baseline_<ts>/<game_id>/play.gif` 能在浏览器播放

### Step 5 — `scripts/run_baseline.py`  ✅(2026-05-11 合入,--dry-run 烟雾验证过)

- **文件**:`scripts/run_baseline.py`
- **职责**(脚本只编排,逻辑全在库里):
  1. `arc.list_games()` → 按字母序固定切 G_base / G_train / G_val(可复现);切分逻辑放进 `arc_agent/eval_split.py::demo_555_split() -> dict[str, list[str]]`
  2. 对 G_base 5 个游戏各跑 1 episode,用 `arc_agent.runner.play_one` + `VLMAgent`
  3. 每步保存 `step_<n>.png`(`viz.compose_step_image`)+ JSONL 一行(prompt / response / f1 / parse_ok)
  4. 整局结束 `viz.write_gif(...)`,生成 `play.gif`
  5. 全部完成写 `summary.json`:每个游戏的 mean F1 / parse_rate / RHAE
- **CLI**:`--games <ids> --episodes 1 --output outputs/baseline_<ts>`(--games 默认从 5-5-5 切分取 G_base)
- **Acceptance**:run 一次后产物齐全(每游戏一个文件夹,含 PNG、GIF、trace.jsonl、summary.json)

### Step 6 — 跑 Baseline + 验证 Hypothesis  ⬜  ★ Go/no-go gate

- **预注册 Hypothesis**(运行前**必须**在 commit message 或 summary.json 头部写下):
  - Mean F1 在 G_base ≥ 0.30
  - Mean RHAE 在 G_base ≤ 0.05
  - JSON parse 成功率 ≥ 0.70
- **执行**:`python scripts/run_baseline.py`
- **Iteration trigger**(根据 summary.json 里的实测数字分支):
  - F1 ≥ 0.3 且 parse ≥ 0.7 → 进入 Step 7(GRPO 训练)
  - F1 < 0.1 → 升级双图输入(上帧 + 当前帧);仍不行 → 回退 BC 路线
  - parse < 0.5 → 加 1–2 个 in-context 示例,重跑 baseline
  - 0.1 ≤ F1 < 0.3 → 消融实验(单图 vs 双图;有 entity 段 vs 无)
- **Acceptance**:summary.json + 触发的分支结论写在本文件 §9 末尾的"Run Log"小节(append-only)

### Step 7 — `scripts/run_grpo.py`  ⬜

- **文件**:`scripts/run_grpo.py`(脚本)+ `arc_agent/train_grpo.py`(库,封装 trl.GRPOTrainer 的 setup)
- **签名**:
  - `arc_agent/train_grpo.py::build_trainer(model, processor, reward_fn: Callable, train_games: list[str], **grpo_kwargs) -> GRPOTrainer`
  - `arc_agent/train_grpo.py::reward_fn(rollout_step) -> float`(§3 公式:1.0 win + 0.2*F1 − 0.5 parse_fail − 0.3 illegal_action)
- **CLI**:`--games <5 train ids> --val-games <5 val ids> --steps N --val-every 50 --output outputs/grpo_<ts>`
- **测试**:`tests/test_train_grpo.py::test_reward_fn_*`(每条惩罚/奖励路径单独覆盖)
- **Acceptance**:reward_fn 单测过 + 在 G_train 上能跑出至少 50 个 GRPO step 不 OOM(显存日志记 peak ≤ 18GB)

### Step 8 — GRPO 训练 + Validation  ⬜

- **预注册 Hypothesis**:G_val mean RHAE 训后比训前提升 ≥ 0.05;G_train 提升 ≥ 0.10(sanity check)
- **训练**:`python scripts/run_grpo.py ...`,wall-clock ≤ 4 GPU-days,early stop 看 G_val mean RHAE 连续 3 次 val 不涨
- **Validation**:`scripts/run_validation.py --checkpoint <path> --games <G_val>`(复用 Step 5 的脚本骨架,换 checkpoint)
- **关键判断**:
  - `F1_v_post > F1_v_pre` → 训出真本事
  - `F1_t_post 涨 / F1_v_post ≈ pre` → 过拟合,记录失败模式
  - 都没涨 → 路线问题,回头诊断 prompt / reward / 数据
- **Acceptance**:训前训后两份 summary.json + 一句结论 append 到本文件 §9 末尾 Run Log

---

### Run Log(append-only)

每次跑完 Step 6 / Step 8 在这里追加一行:`<日期> | <step#> | <核心数字> | <触发分支> | <下一步>`。

| 日期 | Step | 关键数字 | 触发分支 | 下一步 |
|---|---|---|---|---|
| _(待填)_ | | | | |
