# vlm_test — RL with Intrinsic Reward 实施工作区(核心方案文档)

最近一次更新:2026-05-11
状态:**方案设计 + Day 1 已实施**(`arc_agent/rewards.py` 已建,18 个测试全过)

> **本文档是 RL 路线的核心实现蓝本**,后续代码以此为准。设计原理、Hypothesis、5-5-5 切分、实施顺序等全部在本文。

> **与 `docs/ARCHITECTURE.md` 的分工**:`docs/ARCHITECTURE.md` 写"模块职责"(稳定不变);本文写"训练 + 推理的具体步骤"(可演进)。

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

## 总览图

```
zero-shot Qwen2.5-VL-3B
  │
  │  Step A — Baseline(不训练,先看 Qwen 在干什么)
  │           输出: GIF + F1/parse 数字
  │           Go/no-go gate
  ▼
有进展 → 走 GRPO 训练; 没进展 → 改 prompt 或回退 BC
  │
  │  Step B — GRPO + Intrinsic Reward(F1 当密集 reward)
  │           训练: ~3-4 GPU-days on G_train (5 个游戏)
  ▼
checkpoints/grpo_final/
  │
  │  Step C — Validation
  │           在 G_val 上对比训前训后
  ▼
最终评估
```

5-5-5 切分:
- **G_base**: 5 个游戏 — 跑 baseline,生成 GIF
- **G_train**: 5 个游戏 — GRPO 训练
- **G_val**: 5 个游戏 — 训前+训后都跑,看进步
- 三组互不重叠;剩 10 个 demo 留作最终评估

---

## 文件夹结构(实施期)

```
vlm_test/
├── README.md                      本文件(核心方案)
├── data/
│   ├── inputs/                    旧 smoke-test 输入(保留作参考)
│   └── train/                     旧 BC 训练数据(暂停后不再增加)
│
├── outputs/
│   ├── baseline_<ts>/             阶段 A 输出
│   │   └── <game_id>/
│   │       ├── step_<n>.png       每步合成图(grid + 预测 + 真实 + JSON)
│   │       ├── play.gif           整局 GIF
│   │       └── trace.jsonl        每步原始数据
│   ├── grpo_<ts>/                 阶段 B 训练产物
│   │   └── checkpoint/            LoRA 权重
│   ├── validation_<ts>/           阶段 C 输出
│   └── test_results.json          旧 smoke-test
│
└── scripts/
    ├── collect_data.py            旧 BC 数据收集(暂停)
    ├── test_vlm.py                旧 capability 测试(2026-05-08 通过)
    ├── tiny_train.py              旧 QLoRA smoke test(暂停)
    │
    ├── run_baseline.py            ★ 新:阶段 A 入口
    ├── make_gif.py                ★ 新:把 step_<n>.png 合成 play.gif
    ├── run_grpo.py                ★ 新:阶段 B 入口
    └── run_validation.py          ★ 新:阶段 C 入口
```

旧 BC 脚本**保留但暂停**,不删除。如果 baseline F1 < 0.1 触发回退,这些可能复用。

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
Step 6.  real_diff = real_changes(s_t, s_tp1)
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

## 第 6 节:实施顺序与状态

| # | 任务 | 文件 | 状态 |
|---|---|---|---|
| 1 | `grid_to_image()` 实现+测试 | `arc_agent/observation.py` | ✅ **Day 1 完成**(9 个测试) |
| 2 | `verify_prediction_f1()` + `real_changes()` + `changes_to_set()` | `arc_agent/rewards.py` | ✅ **Day 1 完成**(18 个测试全过) |
| 3 | VLMAgent 推理 loop(含 entity + reflection) | `arc_agent/agents/vlm.py` | 🟡 **Day 2 下一步** |
| 4 | GIF 合成工具 | `arc_agent/viz.py` + `scripts/make_gif.py` | ⬜ Day 3 |
| 5 | `scripts/run_baseline.py` | `vlm_test/scripts/run_baseline.py` | ⬜ Day 3 |
| 6 | **跑 baseline + 验证 Hypothesis** | — | ⬜ **Day 4 ★ Go/no-go gate** |
| 7 | `scripts/run_grpo.py` | `vlm_test/scripts/run_grpo.py` | ⬜ Day 5-6 |
| 8 | GRPO 训练 + Validation | — | ⬜ Day 7-10 |

**总预算**:~10 天(基础设施 3 days + baseline 验证 0.5 day + 训练 4 days + 评估 0.5 day + buffer 2 days)。

---

## 第 7 节:怎么跑(命令清单)

⚠️ Day 3-4 后才能跑通,目前是计划态。

```bash
# 阶段 A — Baseline(每个游戏 1 episode,生成 GIF)
.venv/Scripts/python.exe vlm_test/scripts/run_baseline.py \
    --games <5 game ids> --episodes 1 --output vlm_test/outputs/baseline_<ts>

# 阶段 B — GRPO 训练
.venv/Scripts/python.exe vlm_test/scripts/run_grpo.py \
    --games <5 train game ids> --val-games <5 val game ids> \
    --output vlm_test/outputs/grpo_<ts>

# 阶段 C — Validation(训后)
.venv/Scripts/python.exe vlm_test/scripts/run_validation.py \
    --checkpoint vlm_test/outputs/grpo_<ts>/checkpoint \
    --games <5 val game ids> --output vlm_test/outputs/validation_<ts>
```

---

## 第 8 节:FAQ

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

## 第 9 节:与其它文档的关系

| 文档 | 关系 |
|---|---|
| `docs/ARCHITECTURE.md` | 父文档(模块职责)。本文是"训练+推理具体流程"补充 |
| `docs/ROADMAP.md` | 本文使 Phase 2 BC 子计划失效;Phase 4 RL 提前到当前 |
| `docs/RESEARCH.md` | 本路线立项条目:方案 #5("Qwen 自验证 + intrinsic F1 reward") |
| `docs/EXPERIMENTS.md` | Baseline / 训练 / Validation 三个阶段各有预注册条目(待加) |
| `docs/PAPER.md` | §4 Method 大改;§5 Experiments 表新增 baseline / trained 两行(待加) |
| `docs/LIBRARY.md` | 已注册 `changes_to_set` / `real_changes` / `verify_prediction_f1` / `grid_to_image` |

实施时按 `CLAUDE.md` 工作流,每完成一步同步更新对应 doc。
