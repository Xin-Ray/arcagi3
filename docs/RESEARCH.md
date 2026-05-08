# RESEARCH — 文献、已尝试方案与缺陷

最近一次更新:2026-04-27(跑通 agent_starter.py 后 §4 关闭 6 个开放问题、新增"SDK 已知坑"段;新增 4 个待验证问题)

**核心原则:失败比成功更要写下来。** 这是避免下次重蹈覆辙的唯一方式。

---

## 一、相关文献 / 资源

### 必读官方
- ARC-AGI-3 Launch Blog:https://arcprize.org/blog/arc-agi-3-launch
- 技术论文 arXiv:https://arxiv.org/abs/2603.24621(论文号待确认,正文里给出的)
- API 文档:https://docs.arcprize.org
- ARC Prize 2026 总览:https://arcprize.org/competitions/2026
- ARC Prize Verified Testing Policy:https://arcprize.org/policy
- Preview 30-Day Learnings:https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings
- ARC-AGI-3 Human Dataset(458 人 trace 已开源):https://arcprize.org/blog/arc-agi-3-human-dataset

### 已知 SOTA / 开源方案

| 来源 | Preview 期分数 | Launch 期分数(OOD) | 链接 |
|------|----------------|---------------------|------|
| StochasticGoose (Tufa Labs) | **12.58%(1st)** | **0.25%**(掉了 50×) | [code](https://github.com/DriesSmit/ARC3-solution) / [blog](https://medium.com/@dries.epos/1st-place-in-the-arc-agi-3-agent-preview-competition-49263f6287db) |
| dolphin-in-a-coma "just-explore" | 3rd | 未披露 | [code](https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore) |
| Symbolica AI Arcgentica | (community) | — | [code](https://github.com/symbolica-ai/ARC-AGI-3-Agents) |
| Frontier API 裸跑 (GPT-5/Claude/Gemini) | — | <1% | [Official LB](https://three.arcprize.org/leaderboard) |

### 相关方向(暂列,待逐篇精读后填读后感)

| 方向 | 代表工作 | 启发点 | 是否读过 |
|------|----------|--------|----------|
| 模型基 RL / 世界模型 | Dreamer 系列、MuZero | 学转移函数 + 用搜索决策 | ❌ |
| LLM agent 探索 | Voyager (Minecraft)、ReAct | 用 LLM 做 high-level 探索决策 + 自我反思 | ❌ |
| 抽象推理 | ARC-AGI-1/2 winners (Hodel, Greenblatt) | 程序合成、test-time compute | ❌ |
| Test-Time Adaptation | TTT (Test-time training) | ARC-AGI-1/2 上验证可让性能翻倍以上 | ⚠️ 见读后感 #2 |
| ARC-AGI-1/2 小 LLM 微调 | Iteration 46 (Ironbar) — Qwen3 系列 | 0.6B–4B 已是 ARC-AGI-1/2 SOTA;迁移到 3 未验证 | ⚠️ 见读后感 #1 |

> 每读一篇,在下面 **读后感** 段落新增一条:论文标题 + 一句话贡献 + 一句话能不能用上 + 日期。

### 读后感(append-only)

#### #1 — 2026-04-27 — Iteration 46. Revisit small LLMs(Ironbar / arc24)
- **链接**:https://ironbar.github.io/arc24/modeling/Iteration_46_revisit_small_llms/
- **核心贡献**:在 ARC-AGI-1/2 上系统对比小 LLM 微调效果。Qwen3-4B-Instruct-2507 当前榜首,Qwen3-0.6B / Llama-3.2-1B 微调收益最大。
- **能不能用**:**间接相关**。1/2 是单 shot 函数映射,3 是 sequential decision,迁移性未证。但**它证明了"小 LLM + 微调"是 ARC 系列任务上一条已验证可行路线**,我们的"Qwen3-0.6B + LoRA"路线借此立足。
- **行动**:作为 PAPER §2.3 的引用之一(分数证据);Phase 2 选 backbone 时直接选 Qwen3-0.6B / SmolLM2-360M。

#### #2 — 2026-04-27 — Boosting Performance on ARC is a Matter of Perspective(test-time training)
- **链接**:https://arxiv.org/html/2505.07859v1
- **核心贡献**:在 ARC-AGI 上,推理时用任务给的少量例子做 fine-tune,**性能翻倍以上**。
- **能不能用**:**强相关**。我们的"在线 LoRA fine-tune"就是 TTT 的应用,把 5h 探索预算的一部分用作模型微调成本,而不是只做 inference。
- **行动**:Phase 3 的核心机制;PAPER §2.3 + §4 引用。

#### #3 — 2026-04-27 — StochasticGoose 1st-place blog(Dries Smit / Tufa Labs)
- **链接**:https://medium.com/@dries.epos/1st-place-in-the-arc-agi-3-agent-preview-competition-49263f6287db / [code](https://github.com/DriesSmit/ARC3-solution)
- **核心贡献**:Preview 期 1st(12.58%),纯 CNN(4 层 32→64→128→256)+ 在线 RL,无离线训练、无预训练权重、每个游戏从零起、每 level 清 buffer。Hash 去重经验池,层次化采样(先选动作类型,再 conv 出 64×64 坐标 heatmap)。
- **关键发现**:Launch 期掉到 **0.25%**(50× 坍塌),证明"纯在线 + 无知识迁移"是个上限低的路径。
- **能不能用**:**两件事直接搬**——① CNN 预测帧变化的辅助头(cheap 信号强);② 层次化坐标采样(64×64 conv heatmap 而不是 4096-class softmax)。**整体范式不搬**(纯在线 → 我们走"离线 BC + RL → 在线 LoRA")。
- **行动**:Phase 0 末读源码,Phase 2 借用上述两个 trick;PAPER §2.1 引用,作为 Discussion §6.2 的对照案例。

#### #4 — 2026-04-27 — Symbolica AI Arcgentica(orchestrator + subagent)
- **链接**:https://github.com/symbolica-ai/ARC-AGI-3-Agents
- **核心贡献**:LLM orchestrator 不直接动手,把任务拆给 subagent,subagent 返回**压缩过的文字摘要** → 控住 context 增长。
- **能不能用**:**间接相关**。它走 API LLM 路线(无奖)。**洞察可借用**:不要让模型直接读 raw 帧序列,中间需要一层归纳/压缩——我们的"动态规则表"是这层的持久化版本。
- **行动**:PAPER §2.2 引用;架构上确认"规则表 + 帧"双路输入是合理的。

#### #5 — 2026-04-27 — ARC-AGI-3 Human Dataset(458 名参与者 trace 开源)
- **链接**:https://arcprize.org/blog/arc-agi-3-human-dataset
- **核心贡献**:458 人 × 2,893 attempts 的完整交互轨迹,**开源**。
- **能不能用**:**直接核心数据**。我们的 BC 预训练就吃这个数据集。具体可用样本量待真正下载后确认(估算 ~22K 轨迹)。
- **行动**:Phase 1 末或 Phase 2 头下载并 inspect;PAPER §3 / §4 引用;LIBRARY 加 `human_traces.py` 模块负责加载。

---

## 二、已尝试方案

遵循 HEI:每条记录 **方案 → Hypothesis(预测,可证伪) → 执行 → 结果 → 结论 → Iteration trigger(下一步)**。哪怕是 1 行 RandomAgent 也写。这里和 EXPERIMENTS.md 的分工:**EXPERIMENTS** 记数字和实验配置,**这里** 记方法论判断("这条路通不通、为什么、接下来去哪")。

### 1. RandomAgent(2026-04-27,**已闭合**)
- **方案**:从 `available_actions` 均匀随机(排除 `RESET`),`ACTION6` 配随机 (x,y) ∈ [0,63]²
- **Hypothesis**:作为下界基线;预测平均 RHAE ≤ 0.10、过关率 ≤ 30%
- **执行**:demo 25 games × 1 episode × MAX_ACTIONS=80,wall 17.46s,2,000 步 API 调用
- **结果**:平均 RHAE = **0.000**,过关率 = **0%**,所有 25 局 final state = NOT_FINISHED(被 80 步 cap 截断)
- **结论**:**Hypothesis 确认在下界**(实测正好是 0,而不是接近预测上限 0.10)。这意味着:① 80 步内随机决策无法在任何 demo 游戏上过 level 1;② SDK 没有给出非预期的 reward / 截断逻辑(否则会偏离);③ 任何后续 agent 与此基线相比都是"非负数 vs 0",清晰可读
- **Iteration trigger**:**已触发"假设确认 → 进入下一阶段"分支** → Phase 1 LLMAgent。新一组 Hypothesis 在 ROADMAP Phase 1 + 见 EXPERIMENTS 第一条 Iteration trigger
- **写入其它文档**:EXPERIMENTS 首条 + PAPER §5.2 表格 Random 行 + ROADMAP Phase 0 最后一项已勾

### 2. LLMAgent — Claude API + HEI prompt(2026-04-28,Phase 1,进行中)

- **方案**:`arc_agent/agents/llm.py::LLMAgent`,用 Claude API(claude-opus-4-7 / claude-haiku-4-5)+ HEI 三段式 prompt,hex 文本表示 64×64 grid,`summarize_frame()` 生成 per-step 用户消息
- **Hypothesis**:LLMAgent 在 3 个 keyboard 游戏(ls20/tr87/wa30)上平均 RHAE ≥ 0.05,至少 1/3 游戏完成 level 1
- **执行**:2026-04-28 跑(需 Python 3.12 + arcengine,正在配置环境)
- **结论**:[待填 — 实验跑完后补]
- **Iteration trigger**:[待填]
- **写入其它文档**:EXPERIMENTS.md 实验 #2/#3(llm_haiku_phase1 / llm_opus_phase1)预注册条目已建

### 3. Qwen2.5-VL-3B-Instruct + LoRA — 多模态单模型方案(2026-04-28,提案,替代原三层混合)

- **背景**:原三层方案(CNN+Qwen3-0.6B+槽位)在 2026-04-28 被放弃。根本原因:Qwen3-0.6B 是纯文本模型,hex 文本网格按 1D token 序列处理,行间的空间关联(垂直相邻 cell 在序列中相距 ~64 token)无法被 0.6B 模型可靠捕获。强行用 cross-attention 连接 CNN 和文本的融合层加大开发复杂度却不能根治根本缺陷。
- **新方案概述**(详见 `ARCHITECTURE.md` §模型架构 修订版):
  - `grid_to_image()`:64×64 grid → 512×512 PIL Image(ARC 标准 16 色,scale=8)
  - **Qwen2.5-VL-3B-Instruct** vision encoder:2D patch attention,直接感知空间关联
  - LLM backbone:3B Qwen2.5,与 Phase 1 HEI prompt 完全兼容
  - LoRA:rank 8–16 在 text q/v 上,每游戏独立,episode 结束清空
  - 训练三阶段不变(BC → RL → 在线 LoRA),输入从 (text) 改为 (image + text)
- **Hypothesis(预测,可证伪)**:
  - **demo 25 上 BC-only**:平均 RHAE ≥ 0.10(Qwen2.5-VL-3B 视觉感知 + 人类 trace BC,应优于纯文本 LLMAgent)
  - **semi-private 55 上完整系统**:平均 RHAE ≥ 0.05(20× vs StochasticGoose 0.25%)
- **执行**:Phase 2(2026-05-11 → 2026-06-08)开始实现
- **结论**:[待填]
- **Iteration trigger**:
  - 假设确认 → 进入 Phase 3 RL 微调
  - 假设驳斥(BC-only RHAE < 0.05) → 检查 vision encoder 是否在纯颜色网格上欠拟合;考虑冻结 vision encoder 只 fine-tune text backbone;或升至 7B 变体
  - 不确定 → 消融:仅 text-only Qwen2.5-3B vs image+text Qwen2.5-VL-3B,量化视觉输入贡献

### 4. 双路混合 Agent — Qwen3-0.6B + LoRA + 结构化规则槽位(2026-04-27,**已放弃,改为方案 #3**)

- **方案概述**(详细架构见 `ARCHITECTURE.md` §核心智能体循环 修订版):
  - **视觉编码**:小 CNN(借 StochasticGoose 的 4 层 32→64→128→256,~1M 参数)
  - **规则表(双路)**:
    - **Path A — LLM 文本规则**:Qwen3-0.6B 作为 backbone,规则以自然语言累积在 context 里,LLM 自己输出新增/修改;在线时用 LoRA 适配
    - **Path B — 结构化槽位**:固定模板的规则向量(`trigger_action / subject_color / subject_shape / effect_type / effect_param / confidence / evidence_count`),小 transformer (~5–20M) 编码,作为辅助通道
  - **融合**:cross-attention 融合视觉 token + LLM 文本 token + 槽位向量
  - **Heads**:动作头(7 类 + 64×64 conv heatmap,借 StochasticGoose);规则更新头(LLM 生成新规则文本 + 槽位更新);辅助头(下一帧预测、过关概率)
  - **训练**:三阶段 — (A) 离线 BC on 458 人类 trace + demo rollouts;(B) 离线 RL fine-tune (PPO + KL 正则) on demo;(C) 在线 LoRA fine-tune at test time(每游戏开始时重置 LoRA,base 不动)
  - **泛化防御**:训练时游戏增广(颜色 shuffle、旋转/翻转);规则表测试时初始空白 → 强制"用规则,而非记规则";LoRA 在线适应

- **Hypothesis(预测,可证伪)**:
  - **demo 25 上**:平均 RHAE ≥ 0.20(相对 StochasticGoose Preview 期 12.58% 的 1.6×;依靠 BC 引入人类先验 + 双路规则表的归纳偏置)
  - **semi-private 55 上**:平均 RHAE ≥ 0.05(相对 StochasticGoose 0.25% 的 20×;泛化防御机制必须至少抬这一档,否则等于没解决坍塌问题)
  - **过关率 (level 1) 在 semi-private**:≥ 30%
  - **训练成本**:单卡 A100 ~ 50–100 GPU-h(BC) + 100–200 GPU-h(RL)
  - **推理成本**:单游戏 < 5 min on T4,110 个游戏总 < 10 h(留 2h buffer 应对在线 LoRA 的额外开销)

- **Iteration trigger**(三种结果分支预先承诺):
  - **若 demo ≥0.20 且 semi ≥0.05(假设确认)** → 下一步:① ablate 双路(Path A only / Path B only / 两路)看哪一路贡献大;② push backbone 到 Qwen3-1.7B 看是否抬天花板;③ 准备 Kaggle 提交
  - **若 demo ≥0.20 但 semi <0.05(过拟合 demo)** → 这驳斥了"双路规则表 + LoRA 足以泛化"。新候选:① 加更激进的训练时游戏增广;② 加显式的 OOD 正则(如 IRM / 元学习);③ 缩小 base 模型,扩大 LoRA 容量(防记忆)
  - **若 demo <0.10(BC 都没学好)** → 这驳斥了"458 trace 数据量足够 BC"。新候选:① 增加自玩 rollouts;② 用 API LLM 在 trace 上生成 silver-label 规则增广;③ 改用更小 backbone(SmolLM2-360M)以减少过拟合
  - **若 demo 0.10–0.20(中间态,假设不确定)** → 加对照实验:换 backbone(Qwen3-0.6B vs SmolLM2-360M)、关掉辅助损失、关掉结构化槽位

- **依赖 / 阻塞**:
  - 需先完成 Phase 0(`agent_starter.py` 重写、跑通基线)
  - 需 458 人类 trace 数据集已下载并 inspect 通过
  - 需 GPU 资源到位(单卡 A100 / 4090,~250 GPU-h 量级)

- **执行**:**未开始**(2026-04-27 立项)
- **结果 / 结论**:待填

### 2. Phase 0 API 探索(2026-04-27)
- **方案**:读 `.venv` 里安装的 `arc_agi` 包源码 + WebFetch 查 docs.arcprize.org,搞清真实 API
- **Hypothesis**:`arc_agi` SDK 包含 `GameAction` 枚举和 `Arcade` 类,`env.step` 返回 gym 风格 5-tuple
- **执行**:
  - 看 `.venv/Lib/site-packages/arc_agi/__init__.py` → 仅 `from arc_agi_core import *`
  - 看 `arc_agi_core` → 只有 ARC-AGI-1/2 静态拼图(Grid/Pair/Task/Dataset),**没有任何交互式 API**
  - WebFetch `docs.arcprize.org/actions` 和 `/toolkit/minimal.md` → 拿到真实 minimal example
- **结果**:**Hypothesis 被驳斥**。真实情况:
  1. `GameAction` 和 `GameState` 来自独立的 **`arcengine`** 包(我们 venv 没装)
  2. `Arcade` 来自 `arc_agi`,但本地装的 0.0.7 版的 `arc_agi/__init__.py` 只 re-export 了 AGI-1/2,没有 Arcade。可能上游版本不同步,需要进一步排查
  3. 动作空间是 **ACTION1–ACTION7**,共 7 个:1=Up, 2=Down, 3=Left, 4=Right(**4 个方向键,不是 5 个**), 5=Primary(交互/选择), 6=Coordinate(坐标点击,需 x/y∈[0,63]), 7=Undo
  4. `env.step` 签名是 `env.step(action, data=action_data)`,其中 `action_data = {"x": ..., "y": ...}` 仅当 `action.is_complex()` 为 True 时填
  5. 返回值是单个 `obs` 对象(有 `obs.state ∈ {GameState.WIN, GameState.GAME_OVER, ...}`),**不是 gym 风格 5-tuple**
  6. `env.action_space` 给当前可用 GameAction 的列表
- **结论**:agent_starter.py 多处错误。我们文档(CLAUDE.md / TASK_OVERVIEW / ARCHITECTURE)关于"5 个方向键 + UNDO + 坐标"和"obs, reward, done, truncated, info"的描述也错了。**正确版本见 ARCHITECTURE.md 修订**
- **Iteration trigger**(已驳斥后的下一步,**优先级最高**):
  1. `pip install arcengine` 在我们 venv 里跑;如成功 → 进入 2,如失败 → 排查它是不是私有包/需要授权
  2. 重写 agent_starter.py 用真实 API
  3. 把 `arcengine` 加进 requirements.txt
  4. 跑一次,看能不能拿到 scorecard
  5. 修文档里所有"5 directional keys"和"5-tuple"的过时描述

---

## 三、当前已知缺陷(按优先级)

| # | 缺陷 | 影响 | 修复阶段 |
|---|------|------|----------|
| C7 | **agent_starter.py 的多处 API 调用是错的**(详见下条尝试方案 #2):import 路径错(`from arc_agi import GameAction` 应为 `from arcengine import GameAction`)、`UNDO` 不存在(应为 `ACTION7`)、`env.step` 签名错(应为 `env.step(action, data=action_data)`)、返回值结构错(返回 `obs` 对象有 `.state` 而非 5-tuple)、动作总数错(7 而非 6) | 整个脚手架跑不起来 | Phase 0,**优先级最高** |
| C8 | **`arcengine` 包未安装**:requirements.txt 只列了 `arc-agi`,但交互式 API 在独立的 `arcengine` 包里 | 阻塞 C7 | Phase 0,需要先 `pip install arcengine` 验证可行 |
| C1 | ~~`agent_starter.py` 的 `ALL_ACTIONS` 漏了坐标选择动作~~ → **被 C7 取代** (问题更深:整个 ALL_ACTIONS 设计就不对) | — | 已并入 C7 |
| C2 | 没有 `.env`,只有 `.env.example`(注:文档说"匿名 key 也可以",但建议有真 key) | 部分功能受限 | Phase 0 |
| C3 | 没有评估脚本,只能跑单局 | 无法批量得到 scorecard | Phase 0 |
| C4 | 没有 LLM agent,仅 Random | 必然垫底 | Phase 1 |
| C5 | 没有跨步骤记忆(observe_result 没有 hook) | 无法学转移函数 | Phase 1–2 |
| C6 | ~~不知道 `arc_agi` SDK 的 GameAction 完整枚举~~ → **已解决**(2026-04-27 通过 docs.arcprize.org 查到) | — | 闭合 |

---

## 四、未解的问题(开放)

### 已闭合(2026-04-27 一次跑通 agent_starter.py 后)

- ~~**`arcengine` 包能否直接 pip 装?**~~ **是**。`arcengine 0.9.3` 在 PyPI,要求 Python ≥3.12;重建 venv 后 `pip install arcengine` 一行装上。
- ~~**`arc_agi` 升级到带 `Arcade` 的版本?**~~ **是**。Py3.12 上 `arc-agi 0.9.8` 含 `Arcade` 类,签名 `Arcade(arc_api_key, arc_base_url='https://three.arcprize.org', operation_mode, environments_dir, recordings_dir, logger)`。
- ~~**`obs`(实际名 `FrameDataRaw`)的字段?**~~ **已 inspect**。pydantic 模型,字段:`game_id: str` / `state: GameState` / `levels_completed: int` / `win_levels: int`(注:**该游戏的总关卡数**,**不是**胜利数!) / `action_input: ActionInput` / `guid: Optional[str]` / `full_reset: bool` / `available_actions: list[int]` / `frame: list[ndarray]`,每帧 shape `(64, 64)` int 值。**注意**:可能是帧序列(动画),取 `frame[-1]` 或处理整列表。
- ~~**`env.action_space`?**~~ **动态可用集**。`available_actions` 字段是 `list[int]`,值为当前帧合法的动作 ID(1=ACTION1, …, 7=ACTION7,可能不含某些)。例:ls20 reset 后是 `[1,2,3,4]`,只有 4 个方向键合法。
- ~~**RHAE 中 `h` 怎么取到?**~~ **scorecard 里**,不是 obs 里。`scorecard.environments[i].runs[j].level_baseline_actions: list[int]`,每 level 一个 `h`。例:**ls20 七关 h = [22, 123, 73, 84, 96, 192, 186]**(总 776,5× = 3880 是预算上限)。
- ~~**5h 截断 SDK 强制吗?**~~ **不强制(2026-04-27 实测)**。我们用 MAX_ACTIONS=80 跑完没被 SDK 中断;5× 截断应在评测端按 `level_baseline_actions × 5` 判分。我们要自己实现 give-up 逻辑或信任评测端。
- ~~**比赛允许的 LLM 调用形式?**~~ Kaggle 沙盒**完全无网**,LLM 必须本地权重(已记录在 ARCHITECTURE.md 决策表)。
- ~~**人类游戏 trace 是否开源?**~~ **是**,458 参与者 × 2,893 attempts(读后感 #5)。

### SDK 已知坑(append 到这里,避免重蹈)

- **`arc_agi/base.py` 在 import 时自动 `load_dotenv(".env.example")`**(filename 强匹配)。后果:如果只有 `.env.example`(占位 `your_arc_api_key_here`)而没有真 `.env`,`os.getenv("ARC_API_KEY")` 会返回占位串而非空,所有 `if not key:` 风格的检查会被绕过,SDK 拿假 key 去请求,报 401。**修复**:`agent_starter.py` 同时检查 `not key` 和 `key.startswith("your_")`;长期建议在 `.env` 真实存在。
- **匿名 mode(`Arcade(arc_api_key="")`)在 0.9.x 上被 `https://three.arcprize.org` 拒绝**(401 Unauthorized 在 `/api/games`)。docs 老说"anonymous key works too"已过时。
- **GameAction 总数实际为 8**(`RESET + ACTION1–ACTION7`),其中 `RESET` 是 `is_simple()=True` 但语义为"开始/重启游戏"(NOT_PLAYED 或 GAME_OVER 时调用)。"用户实际可选动作"仍为 7(ACTION1–ACTION7)。我们之前文档说"7 个 GameAction"是 USER-callable 数,RESET 是状态转移操作。
- **Windows 默认控制台 cp1252** 不能 print Unicode 箭头(`→`)等;脚本用 ASCII。
- **官方 scaffold 的 `MAX_ACTIONS=80` 是脚本自定义的安全 cap**,不是 SDK 限制。每游戏跨 levels 共享。生产 agent 应按 `5 × sum(level_baseline_actions)` 设上限,或干脆不设让评测端截断。

### 仍开放

- **Kaggle GPU 真实型号?** 公开说 T4 / L4 / P100 都有可能;不同型号显存(16GB / 24GB / 16GB)和 throughput 差异显著,影响 backbone 选型上限。**何时关闭**:Phase 4 准备提交前确认。
- **模型权重打包大小限制?** 没看到明确上限,Qwen3-0.6B fp16 仅 1.2GB,常识 Kaggle 数据集 < 100GB,问题不大但需 Phase 4 前明确。
- **`OperationMode.COMPETITION` 与 `NORMAL` 行为差异?** 知道 X 推文说 Kaggle 必须用 COMPETITION 模式,但具体差异(是否屏蔽 scorecard / 是否更严格的预算)待 Phase 4 前实测。
- **每个 game 的 levels 推进规则?** 已知 ls20 = 7 levels;不知道是否有"过 level 1 才解锁 level 2"的串行约束,还是可任意挑。Phase 1 跑过几关后就清楚。
- **`OperationMode.OFFLINE` 是否能跑下载好的 game?** 已下载 `environment_files/ls20/9607627b/ls20.py` 实质是本地 Python 类。理论 OFFLINE 模式可绕过 `/api/games` 仅用本地缓存。**潜在用途**:Kaggle 沙盒 + 预下载游戏,避免比赛时网络请求。**何时验证**:Phase 4 提交前必查。

---

## 五、成本估算(LLM API)

(待 Phase 1 实现后补)
- 每步 prompt 大小估计:64×64 grid = 4096 token text + 历史 N 帧 = 几万 token / step
- 一局可能 100–500 step
- 一次完整评估 25 game × 6+ levels × 一些 step → 估算总 token / 总美元
- 对 prompt cache 命中率的依赖:**极强**(共享前缀的历史帧)
