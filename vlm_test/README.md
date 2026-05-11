# vlm_test — RL with Intrinsic Reward 实施工作区

最近一次更新:2026-05-11(战略转向:暂停 BC,改走 zero-shot Qwen → GRPO + F1 verifier)

> **本文 = `docs/ARCHITECTURE_RL.md` 的实施侧 README**。设计原理、Hypothesis、5-5-5 切分等**正式定义在** [`../docs/ARCHITECTURE_RL.md`](../docs/ARCHITECTURE_RL.md);本文聚焦"这个文件夹里有什么 + 怎么跑"。

---

## 1. 总览:我们要做什么

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

5-5-5 切分(详见 ARCHITECTURE_RL.md §5):
- **G_base**: 5 个游戏 — 跑 baseline,生成 GIF
- **G_train**: 5 个游戏 — GRPO 训练
- **G_val**: 5 个游戏 — 训前+训后都跑,看进步
- 三组互不重叠;剩 10 个 demo 留作最终评估

---

## 2. 文件夹结构(实施期)

```
vlm_test/
├── README.md                      本文件
├── data/
│   ├── inputs/                    旧 smoke-test 的输入(保留作参考)
│   └── train/                     旧 BC 训练数据(暂停后不再增加)
│
├── outputs/
│   ├── baseline_<ts>/             阶段 A 输出
│   │   └── <game_id>/
│   │       ├── step_<n>.png       每步合成图(grid + 预测 + 真实 + JSON)
│   │       ├── play.gif           整局 GIF
│   │       └── trace.jsonl        每步原始数据(prompt, response, F1)
│   ├── grpo_<ts>/                 阶段 B 训练产物
│   │   └── checkpoint/            LoRA 权重
│   ├── validation_<ts>/           阶段 C 输出(同 baseline 结构)
│   └── test_results.json          旧 smoke-test(2026-05-08 之前)
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

旧 BC 脚本(collect_data.py / tiny_train.py)**保留但暂停**,不删除。如果 baseline F1 < 0.1 触发回退,这些可能复用。

---

## 3. Prompt 模板(6 段)

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
游戏: ls20  /  关卡: 1/3  /  状态: NOT_FINISHED
可用动作: ACTION1, ACTION2, ACTION3, ACTION4

【段 2: 历史动作】
最近 3 步: ACTION3, ACTION1, (start)

【段 3: 上一轮反思】
上次 predicted_diff: [(10, 3, 2)]
上次 real_diff:      []
F1 = 0.0   (预测完全错,P 没动)
推断: ACTION3 不总是向左移,可能被障碍物挡

【段 4: 实体识别请求】
请识别画面中所有实体: shape / color / count / type / function / position

【段 5: 输出格式】
{
  "entities":       [{shape, color, count, type, function, position}, ...],
  "reflection":     "上轮推断(自然语言一句)",
  "predicted_diff": [{row, col, to_color}, ...],
  "chosen_action":  "ACTIONx",
  "new_rule":       null | {trigger_action, subject_color, ..., confidence}
}
```

完整设计原理见 ARCHITECTURE_RL.md §1。

---

## 4. Reward 公式(训练时)

```python
def compute_reward(step):
    r = 0.0
    if step.state == WIN:               r += 1.0       # 稀疏外在
    if step.parsed_json_ok:             r += 0.2 * step.f1   # 密集内在
    if not step.parsed_json_ok:         r -= 0.5       # JSON 罚分
    if step.action not in step.available_actions: r -= 0.3   # illegal action
    return r
```

推理时 F1 不是 reward,仅用于决定 rule_table 是否更新。详见 ARCHITECTURE_RL.md §3。

---

## 5. 实施顺序与状态

| # | 任务 | 文件 | 状态 |
|---|---|---|---|
| 1 | `grid_to_image()` 实现+测试 | `arc_agent/observation.py` | 🟡 进行中(Day 1) |
| 2 | `verify_prediction_f1()` + `diff_grid()` | `arc_agent/rewards.py` | 🟡 进行中(Day 1) |
| 3 | VLMAgent 推理 loop(6 段 prompt) | `arc_agent/agents/vlm.py` | ⬜ Day 2 |
| 4 | GIF 合成工具 | `arc_agent/viz.py` + `scripts/make_gif.py` | ⬜ Day 3 |
| 5 | `scripts/run_baseline.py` | `vlm_test/scripts/run_baseline.py` | ⬜ Day 3 |
| 6 | **跑 baseline + 验证 Hypothesis** | — | ⬜ Day 4 ★ Go/no-go gate |
| 7 | `scripts/run_grpo.py` | `vlm_test/scripts/run_grpo.py` | ⬜ Day 5-6 |
| 8 | GRPO 训练 + Validation | — | ⬜ Day 7-10 |

---

## 6. Baseline 阶段的 Hypothesis(预注册,HEI 规矩)

跑 baseline 前必须先 commit 数字预测:

| 指标 | 预测 | 含义 |
|---|---|---|
| Mean F1 在 G_base | **≥ 0.30** | Qwen zero-shot 视觉理解 |
| Mean RHAE 在 G_base | **≤ 0.05** | zero-shot 不大可能通关 |
| JSON parse 成功率 | **≥ 0.70** | Qwen2.5-VL-Instruct 指令遵循 |

**Iteration trigger**(三种结果分支):
- F1 ≥ 0.3 且 parse ≥ 0.7 → 进入阶段 B(GRPO 训练)
- F1 < 0.1 → 升级单图为双图(上帧 + 当前帧);再不行 → 回退 BC
- parse < 0.5 → 改 prompt 模板,加 1-2 个 in-context 示例
- 0.1 ≤ F1 < 0.3 → 消融实验(单图 vs 双图,有 entity 段 vs 无)

跑完写 `docs/EXPERIMENTS.md` 一条新记录,带最终数字和触发的分支。

---

## 7. 怎么跑(命令清单)

⚠️ 以下命令在 Day 3-4 后才能跑通,目前是计划态。

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

## 8. 重要参考

- 设计原理:[`../docs/ARCHITECTURE_RL.md`](../docs/ARCHITECTURE_RL.md)(本文档的"父文档")
- 模块职责:[`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)
- 路线变更说明:[`../docs/ROADMAP.md`](../docs/ROADMAP.md) 头部"2026-05-11 战略转向"
- 函数库索引:[`../docs/LIBRARY.md`](../docs/LIBRARY.md)
- 失败方案与已知坑:[`../docs/RESEARCH.md`](../docs/RESEARCH.md)
