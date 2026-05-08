# CODE_MAP — 代码目录与简介

最近一次更新:2026-04-27(Phase 1 起步:`arc_agent/__init__.py` + `runner.py` + `agents/random.py` 实际入库;`agent_starter.py` 和 `eval.py` 都改为从库 import)

每个文件/模块**一行**简介,写"做什么 / 关键依赖 / 是否稳定"。新增、删除、改职责时 **必须** 同步本表。

---

## 顶层

| 路径 | 简介 | 状态 |
|------|------|------|
| `agent_starter.py` | 单局 demo 入口,从 `arc_agent` 库 import `RandomAgent` 和 `play_one`(2026-04-27 重构) | **working** |
| `eval.py` | 批量评估入口,跨 N game × M ep,落 `runs/<ts>_<tag>.jsonl`,从库 import 完整 agent + runner | **working** |
| `arc_agent/` | 主包(2026-04-27 起步)。子模块:`runner.py`(Agent Protocol + play_one)、`observation.py`(grid → text)、`llm.py`(Claude wrapper)、`agents/random.py`、`agents/llm.py`(LLMAgent)、`agents/__init__.py`、`__init__.py` | active |
| `tests/` | pytest 测试目录:`test_runner.py` + `test_agents_random.py` + `test_observation.py` + `test_llm.py` + `test_agents_llm.py`,**共 34 测试全过**(2026-04-27) | active |
| `requirements.txt` | 依赖:arc-agi, numpy, Pillow, anthropic, openai, gymnasium, python-dotenv;**arcengine 还未列入,需补**(Phase 0 收尾) | 待补 |
| `.env` | 真 `ARC_API_KEY`,**git-ignore 不要提交**(Phase 0 末加 .gitignore) | 用户本地,不入版本控制 |
| `.env.example` | 模板:`ARC_API_KEY` + 可选 LLM key。**坑**:`arc_agi/base.py` import 时自动加载本文件,placeholder 会污染 `os.getenv` —— 详见 RESEARCH §4 SDK 已知坑 | 稳定 |
| `TASK_OVERVIEW.md` | 比赛规则全文(评分 RHAE、数据集、奖金) | 稳定 |
| `CLAUDE.md` | Claude Code 工作指南 + docs/ 工作流约定 | 稳定 |
| `vendor/ARC-AGI-3-Agents/` | 官方脚手架 git clone(--depth=1,2026-04-27),只读参考。规范模式:Agent ABC + Swarm + Recorder + Playback。`agent_starter.py` 的协议借鉴它但简化为单局 | **read-only**,不要编辑 |
| `environment_files/<game_id>/<version>/` | SDK 自动下载的游戏 Python 类(例:`ls20/9607627b/ls20.py`) | SDK 管理,不要编辑 |
| `recordings/` | (将被 SDK 创建)单局 jsonl 录像 | SDK 管理 |

## docs/

见 [docs/README.md](README.md)。

## 计划中(尚未存在)

按 ARCHITECTURE.md 三层混合架构 + 三阶段训练范式拆分:

### Phase 0–1(脚手架 + 开发期 LLM Agent)

| 路径 | 计划用途 | 计划阶段 |
|------|----------|----------|
| `arc_agent/__init__.py` | 主包入口 | Phase 1 |
| `arc_agent/agents/random.py` | 从 `agent_starter.py` 拆出 | Phase 1 |
| `arc_agent/agents/llm.py` | API LLM(Claude)驱动的开发期 agent;**不进 Kaggle 提交** | Phase 1 |
| `arc_agent/observation.py` | 观测帧序列化(grid → 文本/ASCII/diff) | Phase 1 |
| `arc_agent/data/human_traces.py` | 加载 458 ARC-AGI-3 人类 trace + 增广(颜色 shuffle / 旋转 / 翻转) | Phase 1 末 |
| `arc_agent/data/silver_rules.py` | 用 Claude API 在 trace 上生成 silver-label 规则文本(开发期) | Phase 1 末 |
| `eval.py` | 批量跑 game_id × episode,落 `runs/*.jsonl` | Phase 0 末 |
| `runs/` | 每次实验的 jsonl 日志 | Phase 0 末 |

### Phase 2(Qwen2.5-VL-3B + BC 预训练)

**架构变更 2026-04-28**:原 CNN+Qwen3-0.6B+槽位三层方案改为 Qwen2.5-VL-3B-Instruct 单模型。`visual_encoder.py`、`llm_backbone.py`、`slot_encoder.py`、`fusion.py` 均不再需要,详见 ARCHITECTURE.md。

| 路径 | 计划用途 | 计划阶段 |
|------|----------|----------|
| `arc_agent/vlm_backbone.py` | Qwen2.5-VL-3B-Instruct 加载(BF16/4-bit)+ LoRA 配置 + `generate(image, prompt)->str` | Phase 2 |
| `arc_agent/agents/vlm.py` | VLMAgent:图像+文本双模态 HEI agent,与 LLMAgent prompt 格式兼容 | Phase 2 |
| `arc_agent/data/human_traces.py` | 加载 458 ARC-AGI-3 人类 trace + 增广;输出 (PIL.Image, prompt, action) 三元组 | Phase 2 |
| `arc_agent/data/silver_rules.py` | 用 Claude API 在 trace 上生成 silver-label 规则文本(开发期辅助) | Phase 2 |
| `arc_agent/train_bc.py` | Stage A:QLoRA BC 训练,主损失动作 CE + 辅助下一帧预测 | Phase 2 |

### Phase 3(RL 微调 + 在线 LoRA + 搜索)

| 路径 | 计划用途 | 计划阶段 |
|------|----------|----------|
| `arc_agent/train_rl.py` | Stage B:PPO + KL 正则到 BC 策略 | Phase 3 |
| `arc_agent/intrinsic_reward.py` | dense intrinsic:帧变化 + 规则 entropy + 槽位增长 | Phase 3 |
| `arc_agent/online_adapt.py` | Stage C:测试时每游戏开 LoRA,前 30% 预算微调 | Phase 3 |
| `arc_agent/search.py` | UNDO 回退 + 浅 BFS/MCTS(可选,基于规则表前瞻 1–3 步) | Phase 3 |

### Phase 4(Kaggle 提交)

| 路径 | 计划用途 | 计划阶段 |
|------|----------|----------|
| `kaggle_submit/notebook.ipynb` | 提交 notebook 入口,本地权重 + 无网 inference | Phase 4 |
| `kaggle_submit/weights/` | 打包好的 base 权重 + Stage B RL 策略权重 | Phase 4 |
| `LICENSE` | MIT-0 或 CC0(prize 必需) | Phase 4 |
| `README.md`(项目根) | 复现说明 | Phase 4 |

---

## 命名约定

- 包名 `arc_agent`(下划线);agent 子类放 `arc_agent/agents/<name>.py`,类名 `<Name>Agent`
- 实验日志统一 `runs/<YYYYMMDD>_<short_tag>.jsonl`,一行一 step 或一 episode

## 不要做的事

- 不要把 LLM key 硬编码在文件里;只走 `.env` 和 `os.getenv`
- 不要在 `arc_agent/` 内部依赖 `agent_starter.py`(它只是 demo 入口)
