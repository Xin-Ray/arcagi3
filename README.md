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
| ✅ 库 | `arc_agent/observation.py` | `latest_grid` / `grid_to_image` / `grid_diff` / `animation_to_text` / `summarize_frame` / `serialize_step` |
| ✅ 库 | `arc_agent/rewards.py` | `changes_to_set` / `real_changes` / `verify_prediction_f1` |
| ✅ 库 | `arc_agent/eval_split.py` | `demo_555_split` / `write_summary` |
| ✅ 库 | `arc_agent/llm.py` | Anthropic SDK 封装(开发期 / silver-label 用) |
| ✅ 库 | `arc_agent/agents/random.py` | `RandomAgent` 基线 |
| ✅ 库 | `arc_agent/agents/llm.py` | `LLMAgent`(Claude API) |
| ⬜ 待建 | `arc_agent/agents/vlm.py` | `VLMAgent` — 见 ARCHITECTURE_RL.md §9 Step 3 |
| ✅ 库 | `arc_agent/viz.py` | `compose_step_image`(4 宫格合成图)+ `write_gif` |
| ⬜ 待建 | `arc_agent/train_grpo.py` | GRPO trainer 封装 — Step 7 |
| ✅ 脚本 | `scripts/agent_starter.py` | 单局 RandomAgent demo |
| ✅ 脚本 | `scripts/eval.py` | 批量评估(`--agent {random,llm,llm-haiku}`) |
| ✅ 脚本 | `scripts/freeze_splits.py` | 一次性冻结 demo-25 5-5-5 划分到 `data/splits/demo_555.json` |
| ✅ 脚本 | `scripts/make_gif.py` | 把一个 run 文件夹的 `step_*.png` 合成 `play.gif` |
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

## 6. TODO(按阶段板块)

6 个阶段顺序推进。**每个阶段有明确出口准则**(达到才算这阶段完),阶段内的任务都对应 `docs/ARCHITECTURE_RL.md` §9 的某个 Step,文件级 acceptance criteria 在那里看。

> **关于数据(读一次)**:这是 RL 不是监督学习,**没有静态训练集**。GRPO 训练数据是 agent 在 G_train 上实时生成的 rollouts(epoch 间不复用、不持久化)。"训练/验证/测试" 在本项目里 = **游戏 ID 划分**(下面 Stage 0)。唯一持久化的是每次 run 的产物(trace.jsonl + 合成图 + GIF + summary,落 `outputs/<kind>_<ts>/`)。BC 收集 / 处理流程只在 C 阶段失败回退时才启用,代码已在 `archive/bc_scripts/` 备着。

### Stage 0. 数据基础(划分 + 存储 schema)

> **出口(人话)**:跑任何 baseline / GRPO / validation 之前,先把这三件事定下来 —
> ① 哪 5 个游戏是 baseline、哪 5 个是训练、哪 5 个是验证 → 写成一个 JSON 文件 commit 进 git,以后不许改。
> ② 每步 agent 留下的 JSONL 行长什么样(字段名 + 类型) → 写在写入函数的 docstring 里,所有读写都按这个走。
> ③ 每次 run 跑完的 summary.json 长什么样 → 同上。
>
> **进度**:2 / 3(代码全在,只差跑一次 freeze_splits 把 JSON 写下来 commit)。

| 状态 | 任务 | 路径 / 备注 | §9 Step |
|---|---|---|---|
| 🟡 | **划分冻结**:函数已实现 + 单测过;**还需跑一次** `scripts/freeze_splits.py`(需 ARC_API_KEY)产出 `data/splits/demo_555.json` 并 commit。**不允许重新随机或临时换游戏**(否则训前训后没法对比) | `arc_agent/eval_split.py::demo_555_split()` + `scripts/freeze_splits.py` → `data/splits/demo_555.json` | 5(从 5 拆出) |
| ✅ | **trace schema**:`serialize_step()` 实现 + 6 个测试覆盖(全字段、parse 失败 shape、JSON round-trip、None 透传、类型强制)。每步一行 JSONL,12 字段固定。所有写入用同一个 helper,所有分析脚本按这个字段名读。 | `arc_agent/observation.py::serialize_step()` | 5(子任务) |
| ✅ | **summary schema**:`write_summary()` 实现 + 3 个测试覆盖(必填字段全在、缺字段 raise、optional 透传)。每个 run 一份 `summary.json`,8 个必填字段固定,多次 run 直接对比 `summary['mean_f1']` 不会因字段名打架而崩。 | `arc_agent/eval_split.py::write_summary()` | 5(子任务) |

**数据落盘约定**(写一次,所有 run 必须遵守):

```
data/
├── splits/
│   └── demo_555.json           ★ 冻结划分 — 不允许重新随机
├── inputs/                     5 个能力测试 PNG(2026-05-08 留下)
└── train/
    └── dataset.jsonl           BC fallback 用的 60 行 silver-label(暂封存)

outputs/
├── runs/<ts>_<tag>.jsonl       eval.py 的批量评估输出
├── baseline_<ts>/<game_id>/
│   ├── step_<n>.png            合成图(grid + 预测 + 真实 + JSON)
│   ├── trace.jsonl             每步一行,字段见上
│   └── play.gif                整局 GIF
├── grpo_<ts>/
│   ├── checkpoint/             LoRA adapter
│   ├── train_log.jsonl         GRPO 每 step 的 reward / loss
│   └── val_<step>.json         每 N step 在 G_val 上的快照
└── validation_<ts>/<game_id>/  同 baseline 结构,加载训后 checkpoint
```

### A. 通用工具(基础积木)

> **出口**:其它阶段需要的所有库函数已实现且单测过。
> **进度**:3 / 3 完成 ✅。

| 状态 | 任务 | 路径 | §9 Step |
|---|---|---|---|
| ✅ | 渲染:`grid_to_image()` | `arc_agent/observation.py` | 1 |
| ✅ | F1 verifier 三件套(`changes_to_set` / `real_changes` / `verify_prediction_f1`) | `arc_agent/rewards.py` | 2 |
| ✅ | 可视化:`compose_step_image`(4 宫格)+ `write_gif`(13 个测试覆盖)+ standalone `make_gif.py` | `arc_agent/viz.py` + `scripts/make_gif.py` | 4 |

### B. Agent 推理引擎

> **出口**:`VLMAgent.choose()` 端到端能跑 1 局不崩,JSON 解析失败有 fallback。
> **进度**:0 / 1 完成。**当前阶段(可与 Stage 0、A 的 viz 并行做)。**

| 状态 | 任务 | 路径 | §9 Step |
|---|---|---|---|
| ⬜ | `VLMAgent` 推理 loop(6 段 prompt + JSON 解析 + rule_table) | `arc_agent/agents/vlm.py` | 3 |
| ⬜ | Qwen2.5-VL 加载 + generate 抽象 | `arc_agent/vlm_backbone.py` | 3(子任务) |

### C. Baseline 评估 ★ Go/no-go gate

> **出口**:在 G_base 5 个游戏上跑出 mean F1、parse 成功率、mean RHAE 三个数字,根据预注册 hypothesis 决定下一步走哪条分支(D 训练 / 改 prompt / 双图 / 回退 BC)。
> **进度**:0 / 2 完成。**依赖 B 通过 + Stage 0 freeze_splits 跑过。**(A.viz 已 ✅)

| 状态 | 任务 | 路径 | §9 Step |
|---|---|---|---|
| ⬜ | baseline 脚本(5-5-5 切分 + 每步合成图 + summary.json) | `scripts/run_baseline.py` | 5 |
| ⬜ | **跑 baseline + 写结论到 §9 Run Log** | — | 6 |

**预注册 Hypothesis**(运行前必须 commit):F1 ≥ 0.30、parse ≥ 0.70、RHAE ≤ 0.05。
**分支决策**:F1 ≥ 0.3 + parse ≥ 0.7 → 进 D / F1 < 0.1 → 双图 → 仍不行回退 BC / parse < 0.5 → 改 prompt 重跑 / 中间态 → 消融。

### D. RL 训练(GRPO + intrinsic F1 reward)

> **出口**:G_val mean RHAE 训后比训前涨 ≥ 0.05;否则记录失败模式回头诊断。
> **进度**:0 / 2 完成。**依赖 C 通过。**

| 状态 | 任务 | 路径 | §9 Step |
|---|---|---|---|
| ⬜ | GRPO trainer 封装(reward_fn 单测全过 + 不 OOM) | `arc_agent/train_grpo.py` + `scripts/run_grpo.py` | 7 |
| ⬜ | 跑 GRPO 训练(≤ 4 GPU-days,early stop)+ Validation 对比 | `scripts/run_validation.py` | 8 |

### E. 部署提交(Kaggle 离线)

> **出口**:能在 Kaggle T4 16GB 离线 notebook 上跑完 110 个游戏 ≤ 10 小时,提交得分。
> **进度**:0 / 3 完成。**依赖 D 通过。**

| 状态 | 任务 | 备注 |
|---|---|---|
| ⬜ | 验证 4-bit 量化模型在 T4 16GB 加载 | 显存余量 ≥ 2GB |
| ⬜ | 打包离线 notebook(模型权重 + LoRA + 全部依赖)| 不允许联网 |
| ⬜ | 实测 110 games 整体时延 ≤ 10h | 含 reset / 失败重启 buffer |

---

**已知风险**(可能改变路线的事)

- **C 阶段 Go/no-go 可能不通**。F1 < 0.1 时先升级双图输入,仍不行回退 BC(`archive/bc_scripts/`)。这会把整体路线推后 2-3 周。
- **Kaggle 终评无网**。E 阶段还没实测,Qwen 4-bit 装得下但时延有风险。

---

## 约定(读一次,以后默认)

- `arc_agent/` 是 Python 包。任何会被调用第二次的逻辑必须放进去,**不准**放在 `scripts/`。
- 文件名用功能描述(`agent_starter.py`、`run_baseline.py`),**禁止**用情绪或时间(`final.py`、`_v2.py`、`_new.py`、`_fix.py`)。
- 所有产物落 `outputs/<kind>_<timestamp>/`,**永不覆盖**。
- `archive/` 是**单向门**:东西进得去,但出来必须显式 promote 到上面的白名单。
- 完整的库优先规则和 SDK 已知坑见 `CLAUDE.md`。
