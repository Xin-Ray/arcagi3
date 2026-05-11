# ARC-AGI-3 Agent — ARC Prize 2026

> **3 分钟接管整个项目**。从上读到下,其它都是更深的参考。

比赛:<https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3>
里程碑:**2026-06-30** 开源奖 · **2026-09-30** 终评

---

## 1. 项目目标

做一个能自主玩 ARC-AGI-3 回合制谜题游戏的 agent — **没有任何说明书**,必须自己看出物体行为和通关条件。

**当前路线(2026-05-11 起)**:zero-shot `Qwen2.5-VL-3B-Instruct` 跑 baseline → 用 **intrinsic F1 reward** 做 GRPO 微调(每步 agent 预测哪些格子会变,预测集与真实集的 F1 当作密集 reward — 不用等通关才有信号)。

---

## 2. 数据流

```
ARC-AGI-3 SDK env  ── env.step ──▶  FrameDataRaw (帧序列, 状态, 可用动作)
        │
        ▼
arc_agent.observation.latest_grid()      → 64×64 numpy
arc_agent.observation.grid_to_image()    → 512×512 PNG
        │
        ▼
┌──────────────────────────────────────────────┐
│  arc_agent.agents.vlm.VLMAgent.choose()      │
│    • 构造 prompt(图像 + 状态 + 历史)         │
│    • Qwen2.5-VL-3B-Instruct(LoRA 微调过)    │
│    • 解析 JSON:动作 + predicted_diff         │
└──────────────────────────────────────────────┘
        │
        ▼  GameAction
   env.step()  ───▶  s_{t+1}
        │
        ▼
arc_agent.rewards.real_changes(s_t, s_{t+1})    → 真实变化集合
arc_agent.rewards.verify_prediction_f1(...)     → F1 ∈ [0, 1]
        │
        ▼
intrinsic reward(训练时)│ rule_table 更新(推理时)
```

agent 永远看不到说明书。它在两次稀疏 WIN 之间唯一拿到的信号,就是自己预测的 F1。

---

## 3. 跑哪个文件

```bash
# 0. 一次性安装
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
# 把真实 ARC_API_KEY(https://arcprize.org/api-keys)放进 .env

# 1. 任何事之前先确认测试过
.venv\Scripts\python.exe -m pytest tests/ -q

# 2. sanity check:RandomAgent 跑一局
.venv\Scripts\python.exe scripts/agent_starter.py

# 3. 批量评估(RandomAgent 跑全部 demo,每局 1 ep)
.venv\Scripts\python.exe scripts/eval.py
#    → outputs/runs/<ts>_random_baseline.jsonl
```

**RL 流水线** — 以下脚本是计划中,尚未实现。每个脚本要做什么见 `docs/ARCHITECTURE_RL.md` §9:

```bash
.venv\Scripts\python.exe scripts/run_baseline.py     # ★ Go/no-go gate
.venv\Scripts\python.exe scripts/run_grpo.py
.venv\Scripts\python.exe scripts/run_validation.py
```

---

## 4. 当前在用的版本(白名单)

只有下面这些文件是活的。**没列出来的全部在 `archive/`,不允许直接动**;要用必须先显式 promote 回这张表。

| 状态 | 路径 | 用途 |
|---|---|---|
| ✅ 库 | `arc_agent/runner.py` | Agent Protocol + `play_one()` 单局循环 |
| ✅ 库 | `arc_agent/observation.py` | `latest_grid` / `grid_to_image` / `grid_diff` / `animation_to_text` / `summarize_frame` |
| ✅ 库 | `arc_agent/rewards.py` | `changes_to_set` / `real_changes` / `verify_prediction_f1` |
| ✅ 库 | `arc_agent/llm.py` | Anthropic SDK 封装(开发期 / silver-label 用) |
| ✅ 库 | `arc_agent/agents/random.py` | `RandomAgent` 基线 |
| ✅ 库 | `arc_agent/agents/llm.py` | `LLMAgent`(Claude API) |
| ⬜ 待建 | `arc_agent/agents/vlm.py` | `VLMAgent` — 见 ARCHITECTURE_RL.md §9 Step 3 |
| ⬜ 待建 | `arc_agent/viz.py` | GIF / 单步合成图 — Step 4 |
| ⬜ 待建 | `arc_agent/train_grpo.py` | GRPO trainer 封装 — Step 7 |
| ✅ 脚本 | `scripts/agent_starter.py` | 单局 RandomAgent demo |
| ✅ 脚本 | `scripts/eval.py` | 批量评估(`--agent {random,llm,llm-haiku}`) |
| ⬜ 待建 | `scripts/run_baseline.py` | RL Step 5–6:zero-shot Qwen 在 G_base 上跑 + GIF |
| ⬜ 待建 | `scripts/run_grpo.py` | RL Step 7:GRPO 训练在 G_train 上 |
| ⬜ 待建 | `scripts/run_validation.py` | RL Step 8:训后在 G_val 上对比 |
| ✅ 文档 | `README.md` | 本文 |
| ✅ 文档 | `docs/ARCHITECTURE_RL.md` | 深度设计 + §9 文件级实施步骤 |
| ✅ 文档 | `CLAUDE.md` | Claude Code 工作规则 + 已知坑 |
| ✅ 文档 | `TASK_OVERVIEW.md` | 比赛规则全文 |

**目录结构**

```
.
├── arc_agent/        库(除模型加载外不做 I/O)
├── scripts/          入口脚本(只编排,不放可复用逻辑)
├── tests/            pytest(72 个全过)
├── data/             inputs/(测试 PNG)、train/(silver-label JSONL)
├── outputs/          runs/、checkpoints/、baseline_<ts>/ 等(gitignore)
├── docs/             只有 ARCHITECTURE_RL.md
├── archive/          所有过时的东西 — 不要碰
└── vendor/           只读第三方(ARC-AGI-3-Agents)
```

---

## 5. 当前结果

| 日期 | Step | 结果 |
|---|---|---|
| 2026-04-27 | RandomAgent 在 demo 25 上 baseline | mean RHAE **0.000**、pass rate **0%** |
| 2026-05-08 | Qwen2.5-VL-3B 能力测试(5 项) | **4/5 通过** — T3 失败因为 zero-shot 模型回 `"Right"` 而不是 `"ACTION4"` |
| 2026-05-08 | QLoRA 烟雾训练 | 60 步 silver-label 跑 1 epoch,流水线通,adapter 已存 |
| **TBD** | **Step 6 baseline**(Go/no-go gate) | **尚未跑** — 见 TODO |

Step 6 的预注册 hypothesis:F1 ≥ 0.30、parse 成功率 ≥ 0.70、RHAE ≤ 0.05。三种迭代分支见 `docs/ARCHITECTURE_RL.md` §5.2。

---

## 6. TODO

**当前活跃工作**:`docs/ARCHITECTURE_RL.md` §9 — 文件级 acceptance criteria。

| # | 内容 | 状态 |
|---|---|---|
| 1 | `grid_to_image()` | ✅ 完成 |
| 2 | `rewards.py`(changes_to_set / real_changes / verify_prediction_f1) | ✅ 完成 |
| 3 | `arc_agent/agents/vlm.py` + backbone 抽象 | ⬜ 下一步 |
| 4 | `arc_agent/viz.py` + `scripts/make_gif.py` | ⬜ |
| 5 | `scripts/run_baseline.py` | ⬜ |
| 6 | **跑 baseline + 验证 Hypothesis** ★ Go/no-go | ⬜ |
| 7 | `scripts/run_grpo.py` + `arc_agent/train_grpo.py` | ⬜ |
| 8 | GRPO 训练 + Validation | ⬜ |

**已知风险**

- **Step 6 可能失败**。如果 F1 < 0.1,fallback 是双图输入(上帧 + 当前帧);仍不行则回退到 BC 训练(暂停的分支在 `archive/bc_scripts/`)。
- **Kaggle 终评无网**。最终提交必须离线。Qwen2.5-VL-3B 4-bit 能装进 T4 16GB,但 110 个游戏 ≤ 10 小时的整体时延还没实测。

---

## 约定(读一次,以后默认)

- `arc_agent/` 是 Python 包。任何会被调用第二次的逻辑必须放进去,**不准**放在 `scripts/`。
- 文件名用功能描述(`agent_starter.py`、`run_baseline.py`),**禁止**用情绪或时间(`final.py`、`_v2.py`、`_new.py`、`_fix.py`)。
- 所有产物落 `outputs/<kind>_<timestamp>/`,**永不覆盖**。
- `archive/` 是**单向门**:东西进得去,但出来必须显式 promote 到上面的白名单。
- 完整的库优先规则和 SDK 已知坑见 `CLAUDE.md`。
