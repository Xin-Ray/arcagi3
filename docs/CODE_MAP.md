# CODE_MAP — 代码目录与简介

最近一次更新:2026-05-08(Phase 2 进度:新增 vlm_test/ 子项目；observation.py 补全 grid_to_image + animation_to_text；arc_agent/agents/vlm.py + vlm_backbone.py + train_bc.py 计划中)

每个文件/模块**一行**简介,写"做什么 / 关键依赖 / 是否稳定"。新增、删除、改职责时 **必须** 同步本表。

---

## 顶层

| 路径 | 简介 | 状态 |
|------|------|------|
| `agent_starter.py` | 单局 demo 入口,从 `arc_agent` 库 import `RandomAgent` 和 `play_one`,打印 scorecard | **working** |
| `eval.py` | 批量评估入口,跨 N game × M ep,落 `runs/<ts>_<tag>.jsonl`;`--agent {random,llm,llm-haiku}`,`--budget` 硬截止 | **working** |
| `requirements.txt` | 依赖:arc-agi≥0.9.8, arcengine≥0.9.3, anthropic, Pillow, python-dotenv 等(训练依赖 transformers/peft/bitsandbytes/trl 不在此,见 ROADMAP §2C) | **working** |
| `.env` | 真实 ARC_API_KEY + ANTHROPIC_API_KEY,git-ignore,不入版控 | 用户本地 |
| `.env.example` | 占位模板。**坑**:arc_agi/base.py import 时自动加载本文件 → 坑详见 RESEARCH §4 | 稳定 |
| `TASK_OVERVIEW.md` | 比赛规则全文(评分 RHAE、数据集、奖金) | 稳定 |
| `CLAUDE.md` | Claude Code 工作指南 + docs/ 工作流约定 | 稳定 |

---

## `arc_agent/` — 可复用库

| 路径 | 简介 | 状态 |
|------|------|------|
| `arc_agent/__init__.py` | 包入口(空) | stable |
| `arc_agent/runner.py` | Agent Protocol + `play_one()`:单局核心循环,不做 scorecard open/close | stable |
| `arc_agent/observation.py` | 观测处理:`latest_grid` / `grid_to_text` / `grid_diff` / `available_action_names` / `summarize_frame` / `analyze_animation` / `animation_to_text` / `grid_to_image` | stable |
| `arc_agent/llm.py` | Anthropic SDK 封装:LLMClient + LLMResponse,prompt cache,累计计费 | stable |
| `arc_agent/agents/__init__.py` | 包入口(空) | stable |
| `arc_agent/agents/random.py` | RandomAgent:均匀随机可用动作,RESET 状态机,可再现(seed) | stable |
| `arc_agent/agents/llm.py` | LLMAgent:Claude API + HEI prompt,解析失败 fallback random | stable |
| `arc_agent/agents/vlm.py` | **(Phase 3 计划)** VLMAgent:加载 BC checkpoint → HEI prompt → choose() → parse | 未实现 |
| `arc_agent/vlm_backbone.py` | **(Phase 2C 计划)** Qwen2.5-VL-3B-Instruct 加载 + QLoRA + `generate(image, prompt)->str` | 未实现 |
| `arc_agent/train_bc.py` | **(Phase 2C 计划)** SFTTrainer BC 训练 + RHAE validation callback | 未实现 |
| `arc_agent/data/human_traces.py` | **(Phase 2B 计划)** 读取 human trace JSONL → (image, prompt, action) 三元组 + 增广 ×8 | 未实现 |

---

## `tests/` — pytest 测试套件（36 个测试全过，2026-05-08）

| 路径 | 覆盖 |
|------|------|
| `tests/test_runner.py` | play_one 停止条件、错误路径 |
| `tests/test_agents_random.py` | RandomAgent 状态机、ACTION6 坐标合法性 |
| `tests/test_agents_llm.py` | LLMAgent parse、fallback、HEI prompt 结构 |
| `tests/test_observation.py` | latest_grid / grid_to_text / grid_diff / available_action_names / summarize_frame |
| `tests/test_llm.py` | LLMClient complete + 计费字段 |

---

## `vlm_test/` — VLM 烟雾测试子项目（Phase 2 前置验证，已完成 2026-05-08）

| 路径 | 简介 | 状态 |
|------|------|------|
| `vlm_test/scripts/collect_data.py` | SDK 录制 (image, prompt, action) 三元组到 data/train/；含 animation_to_text | done |
| `vlm_test/scripts/test_vlm.py` | 5 项能力测试，结果写 outputs/test_results.json | done |
| `vlm_test/scripts/tiny_train.py` | 1-epoch QLoRA 烟雾训练，adapter 写 outputs/checkpoint/ | done |
| `vlm_test/data/train/dataset.jsonl` | 60 条 silver-label 训练步骤（ls20，2 ep × 30 steps，旧格式无 animation） | data |
| `vlm_test/data/train/images/` | 对应 PNG 帧（60 张，512×512） | data |
| `vlm_test/data/inputs/` | test_vlm.py 保存的 5 项测试输入图像 + prompt txt | data |
| `vlm_test/outputs/test_results.json` | 5 项测试结果（4/5 passed；T3 format defect 记录） | data |
| `vlm_test/outputs/checkpoint/` | QLoRA adapter 权重（adapter_model.safetensors，PEFT 0.19.1） | data |

---

## `vendor/` — 只读参考

| 路径 | 简介 |
|------|------|
| `vendor/ARC-AGI-3-Agents/` | 官方脚手架 git clone（--depth=1，2026-04-27），只读，不编辑 |

---

## `docs/` — 工作记忆七件套

见 `docs/README.md`。

---

## `archive/` — 归档文档快照

| 路径 | 简介 |
|------|------|
| `archive/EXPERIMENTS.md` | EXPERIMENTS.md 2026-04-28 快照（已迁回 docs/EXPERIMENTS.md 并续写） |
| `archive/CODE_MAP.md` | CODE_MAP.md 2026-04-28 快照（已迁回 docs/CODE_MAP.md 并续写） |
| `archive/HUMAN_TRACE_GUIDE.md` | 人工 think-aloud 标注操作手册（Phase 2A 人工 trace 录制用） |

---

## `traces/` — 人工 trace 数据（Phase 2A 计划）

| 路径 | 说明 |
|------|------|
| `traces/human/<game_id>_NNN.jsonl` | schema_version:2 人工 trace，每步一行（计划，尚未录制） |
| `traces/human/images/<trace_id>/step_NNN.png` | 每步网格 PNG（不入版控，见 HUMAN_TRACE_GUIDE.md） |

---

## `runs/` — 评估日志

| 路径 | 说明 |
|------|------|
| `runs/<YYYYMMDD>_<tag>.jsonl` | eval.py 输出，每行一个 (game,episode) 结果 + 末行 `__summary__` |

---

## 命名约定

- 包名 `arc_agent`（下划线）；agent 子类放 `arc_agent/agents/<name>.py`，类名 `<Name>Agent`
- 实验日志统一 `runs/<YYYYMMDD>_<short_tag>.jsonl`，一行一 step 或一 episode
- 训练 checkpoint 统一 `checkpoints/<tag>_epoch_N/`（正式训练）或 `vlm_test/outputs/checkpoint/`（烟雾测试）

## 不要做的事

- 不要把 LLM key 硬编码在文件里；只走 `.env` 和 `os.getenv`
- 不要在 `arc_agent/` 内部依赖 `agent_starter.py`（它只是 demo 入口）
- 不要编辑 `vendor/ARC-AGI-3-Agents/`（只读参考）
