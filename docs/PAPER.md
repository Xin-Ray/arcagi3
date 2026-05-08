# PAPER — 活论文(Living Paper)

最近一次更新:2026-04-27(v0.06:§5.2 表格 Random 行 + §6.1 first "What worked" 句子;由 EXPERIMENTS.md 第一条实验触发)
状态:**Draft v0.06**(Phase 0 末,Method 主体定稿,Experiments 表 Random 行已填,后续 Agent 行待 Phase 1+)
目标投稿地:Kaggle write-up + 可能的 arXiv preprint;Milestone #1(2026-06-30)前争取到 v0.5。

---

## 这份文档的定位

`PAPER.md` 是**对外讲清楚**项目的那份文档,与其它六份内部跟踪文档分工如下:

| 来源 | → 同步到 PAPER.md 的哪一节 |
|------|--------------------------|
| `RESEARCH.md` 文献条目 | Related Work |
| `RESEARCH.md` 已尝试方案 | Method(成功的)+ Discussion(失败的、教训) |
| `ARCHITECTURE.md` | Method |
| `EXPERIMENTS.md` 数字 | Experiments(表格)+ Discussion(结论) |
| `ROADMAP.md` 完成的里程碑 | Contributions / Conclusion |
| `TASK_OVERVIEW.md` | Problem Setting(简述,不照抄) |

**更新触发**(强制):
1. 任何 EXPERIMENTS 出新结论 → 更新 Experiments 表 + Discussion 至少一句
2. 任何 RESEARCH 新增文献条目(读后感) → 更新 Related Work 段(且 References 加链接)
3. 任何 ARCHITECTURE 重要决策变化 → 更新 Method
4. 阶段切换(进入 Phase X)→ 顶部 status 改版本号(Phase 0 末 = v0.1, Phase 1 末 = v0.2, …)

**写作规则:**
- 学术论文风格,主语用"we",**不要写"用户" / "Claude"**
- 每个数字必须能在 EXPERIMENTS.md 找到出处(不许编)
- 每个引用必须在 References 段有 URL(不许只写作者)
- 失败的方案放 Discussion,**不要藏着**;比赛规则要求开源,审稿人也会看负面结果
- 当前缺的、未完成的、待验证的,统一标 `[TBD]` 或 `[pending Phase N]`

---

# Title (working)

**Hypothesize, Execute, Iterate: A Modular Agent for ARC-AGI-3 Without Rewards**

(标题是 working title,Phase 1 末再定。)

**Authors:** [TBD] · **Date drafted:** 2026-04-27 · **Last revised:** 2026-04-27

---

## Abstract

[TBD — Phase 1 末写第一版。占位提纲:]

> ARC-AGI-3 提出一个挑战:在无指令、无规则说明、几乎无 reward 的回合制游戏环境中,自主探索、推断规则、识别胜利条件并高效通关。本文提出一个把**科学方法 (Hypothesize–Execute–Iterate, HEI) 直接作为 agent 决策骨架**的模块化架构,并在 ARC-AGI-3 公共 demo 集 (25 environments) 上评估。我们贡献:(1) 一个用 LLM 显式输出可证伪假设的 agent prompt 模板;(2) 一个把信息增益与目标推进联合优化的动作选择器;(3) 一份 baseline → LLM → world-model 三档对比的实证分析。我们的最强系统达到 [TBD] 平均 RHAE,相对随机基线提升 [TBD]×。

---

## 1. Introduction

[TBD — Phase 1 末写]

写作大纲(供未来填):
- 痛点:LLM agent 在 ARC-AGI-3 上 ~0.51%(2026-03 frontier 数据,见 References [1]),距离人类 100% 极远
- 为什么传统 RL 不行:reward 几乎只在通关时给,5h 截断后无任何信号
- 我们的洞察:把"科学方法"显式化 → 假设可证伪、动作含信息增益、迭代有触发器
- 贡献清单(三条):见 Abstract
- 论文结构

---

## 2. Related Work

[append-only:每读一篇相关论文 → 在 RESEARCH.md 写读后感 → 这里加一段 + 在 References 加 URL]

### 2.1 ARC-AGI-3 已有方案

The Preview competition winner, **StochasticGoose** by Tufa Labs [8, 9], adopted a 4-layer convolutional network (32→64→128→256 channels) trained online with reinforcement learning to predict frame-changing actions. Hierarchical sampling first selects an action type and, for the coordinate action, emits a 64×64 probability map via a final convolution rather than a 4096-way softmax. This system reached 12.58% on the Preview hidden set but collapsed to **0.25%** on the official Launch evaluation [10] — a roughly 50× degradation that we treat as the canonical evidence that purely-online learning, without any offline knowledge transfer, hits a hard generalization ceiling on this benchmark. We adopt the action-prediction auxiliary head and the 64×64 conv heatmap idea, but reject the from-scratch-per-game training paradigm.

A second open-source line of work, **Arcgentica** by Symbolica AI [11], uses an LLM orchestrator that delegates sub-tasks to specialized agents returning compressed textual summaries. While the system relies on hosted-model APIs and is therefore not Kaggle-eligible, its architectural insight — that raw frame sequences are too low-bandwidth for direct LLM consumption and require an inductive intermediate layer — directly motivates our use of an explicit, persisted **rule table** as that layer.

### 2.2 LLM-driven Agents 与探索
- [pending] Voyager (Minecraft)、ReAct —— 用 LLM 做高层探索决策 + 自我反思。我们的 Phase 1 (development-only) LLMAgent 借用 ReAct 的"思维-动作"结构,但要求 LLM **显式输出可证伪 hypothesis**,与 §4.3 的 prompt-level HEI 对齐。

### 2.3 ARC 系列上的小 LLM 微调与测试时训练

On ARC-AGI-1/2 (the static input/output-pair predecessors of ARC-AGI-3), recent work has demonstrated that small open-weight language models can be highly competitive after fine-tuning. The Iteration-46 micro-benchmark by Ironbar [12] reports Qwen3-4B-Instruct-2507 as the top fine-tuned model and shows that smaller checkpoints (Qwen3-0.6B, Llama-3.2-1B) close most of the gap to larger models when fine-tuned. Independently, **test-time training** [13] — fine-tuning on a task's few-shot examples before generating an answer — has been shown to more than double ARC-AGI accuracy. We borrow both findings: we select Qwen3-0.6B as our backbone and apply test-time LoRA adaptation per game (§4.5, Stage C). However, we caution that ARC-AGI-1/2 are single-shot input-output mappings while ARC-AGI-3 is a sequential decision problem, so the transfer is suggestive rather than proven, and forms a central hypothesis to test.

### 2.4 模型基强化学习与世界模型
- [pending] Dreamer 系列、MuZero —— 学转移函数 + 用搜索决策。我们的 WorldModel 借用其"显式预测下一状态"思路,但具象化为符号 + 神经混合(§4.4 双路规则表),而不是纯神经隐变量。

### 2.5 抽象推理与程序合成
- [pending] ARC-AGI-1/2 winners (Hodel, Greenblatt 等) —— 程序合成、test-time compute。在 ARC-AGI-3 的交互场景下程序合成是否还适用是开放问题;我们的结构化槽位(§4.4 Path B)是该方向的轻量化探索。

### 2.6 ARC-AGI-3 数据资源
- ARC-AGI-3 比赛 launch blog [1]
- 技术论文 [2]
- **ARC-AGI-3 Human Dataset(458 名参与者 trace,开源)** [14] —— 我们的 BC 预训练核心数据,详见 §4.5 Stage A

---

## 3. Problem Setting

ARC-AGI-3 提供 135 个交互式环境(25 公开 demo / 55 半私有 / 55 完全私有),每个 ≥6 levels。
- **Observation**: 64×64 grid,每 cell 一种 16 色之一,可能为帧序列(动画)
- **Action**: `ACTION1–5`、`UNDO`、`COORD(x, y)` 共 7 类
- **No instructions**:agent 完全不知道目标、规则
- **Score (RHAE)**: `S = min(1.0, h/a)²`,`h` = 第 2 名人类动作数,`a` = agent 动作数;quadratic penalty;5h 硬截断
- **Aggregation**: per-environment 是各 level 的加权平均(`w_l = l`),per-dataset 是各 environment 的均值

详细规则见 `TASK_OVERVIEW.md`,本节只取建模需要的最小集。

---

## 4. Method

[TBD — 以 ARCHITECTURE.md 为蓝本,实验进展后逐节填]

### 4.1 Overview: HEI Loop as Decision Backbone

我们用 Hypothesize → Execute → Iterate 作为 agent 每一步的决策循环,替代传统的 state-action-reward 闭环。
形式化:agent 维护一个假设集 $\mathcal{H} = \{h_i, c_i\}$,每条假设 $h_i$ 带置信度 $c_i \in [0,1]$。每步动作 $a$ 选择最大化加权目标:

$$ a^* = \arg\max_a \big[\, \alpha \cdot V(s'\mid s,a) + (1-\alpha) \cdot \mathrm{IG}(\mathcal{H}; a) \,\big] $$

其中 $V$ 是 LLM 给出的 heuristic 价值,$\mathrm{IG}$ 是动作对 $\mathcal{H}$ 的预期信息增益,$\alpha$ 在 episode 内从小到大调度(早期重探索、晚期重开发)。

### 4.2 Modules

| 模块 | 实现 | Phase |
|------|------|-------|
| ObservationParser | grid → text/diff/tensor 三视图 | 1 |
| WorldModel(假设管理) | 自然语言假设 + (可选)程序判定 | 2 |
| Planner(V 估计 + IG 估计) | LLM heuristic + counterfactual rollout | 2–3 |
| ActionSelector(加权融合 + UNDO 回退) | argmax + ε-noise | 1 |
| Memory(三元组缓存) | per-episode + per-environment | 1 |

### 4.3 Phase 1 Implementation: Prompt-level HEI

第一版 LLMAgent 不实现完整循环,而是让 LLM 在 prompt 里显式输出三段(Hypothesis / Execute / Iterate),把推理过程"打卡"出来。这样的好处:零基础设施成本就能起跑,且 LLM 输出本身就是 Phase 2 WorldModel 的训练/对齐数据。

### 4.4 Three-Tier Hybrid Architecture (Phase 2 Implementation)

For the deployed system on Kaggle (which forbids external API calls [4]), we replace the prompt-only LLMAgent with a three-tier hybrid that runs entirely on local weights.

**Tier 1 — Visual encoder.** A 4-layer convolutional network with 32→64→128→256 channels (≈1M parameters) over a 64×64×16 one-hot grid input, optionally augmented with the previous-frame difference as additional channels. This architecture is borrowed from StochasticGoose [8] for its proven ability to encode the ARC-AGI-3 grid structure, but we keep it small enough to leave compute headroom for the language tier. The CNN is trained from scratch; we found no benefit from natural-image pretraining (e.g. ImageNet, CLIP) given the discrete 16-color domain.

**Tier 2 — Dual-path rule representation.** Rules — agent beliefs about how each action affects the world — are represented through two parallel paths:

- **Path A (LLM-text):** A Qwen3-0.6B backbone [12], extended with a low-rank adapter (LoRA, rank 8–16). Rules are accumulated as natural-language strings in the model's context. Path A provides expressive flexibility — the model can describe rules in any form it can articulate.

- **Path B (structured slots):** Each rule is also encoded as a fixed-length vector with seven categorical/numerical fields: `trigger_action`, `subject_color`, `subject_shape`, `effect_type`, `effect_param`, `confidence`, `evidence_count`. A small transformer (~5–20M parameters) embeds the (N, D) rule matrix into a fixed-size context. Path B trades expressive ceiling for inference speed, interpretability, and a robust fallback when the language path fails to produce a useful rule.

We deliberately accept the redundancy: Path A's empirical failure modes (slow generation, hallucinated rules) are different from Path B's (limited template expressiveness). In ablation we predict that single-path systems will underperform the dual-path combination, particularly on out-of-distribution games where one path may fail.

**Tier 3 — Fusion and heads.** A cross-attention layer fuses visual tokens, Path-A text tokens, and Path-B slot vectors into a unified representation. From this, we predict (i) an action distribution over ACTION1–7, (ii) a 64×64 coordinate heatmap for ACTION6 (Coordinate), via convolution to preserve 2D structure (following StochasticGoose [8]), (iii) rule updates — a textual edit produced by Path A and a slot-write index produced by Path B, and (iv) two auxiliary outputs: a next-frame predictor and a level-success-probability estimator, both used as auxiliary losses during training.

### 4.5 Three-Stage Training Paradigm (Phases 2–3)

To avoid StochasticGoose's catastrophic preview-to-launch generalization gap [10], we explicitly split training into offline knowledge transfer and online adaptation:

**Stage A — Offline behavior cloning (BC) pretraining.** We use the open-sourced ARC-AGI-3 human-trace dataset [14] (458 participants, 2,893 attempts) augmented with random-agent rollouts on the 25 demo environments. Color-shuffle and grid rotation/reflection augmentations are applied since ARC-AGI colors carry no semantics. The training objective combines (i) action cross-entropy against human actions, (ii) next-frame MSE for grounding the world-model representation, (iii) a self-supervised slot loss requiring the structured rules to predict the observed (s, a, s') diff, and (iv) optional silver-label supervision for the rule-text head, where development-time API calls produce candidate rule descriptions for each trace [TBD; pending Phase 1 data preparation].

**Stage B — Offline reinforcement-learning fine-tuning.** Starting from the Stage A checkpoint, we fine-tune with Proximal Policy Optimization, regularized by a KL term against the BC policy to prevent forgetting. The reward signal combines the sparse RHAE win signal with three dense intrinsic rewards: frame-change magnitude (StochasticGoose [8]), reduction in hypothesis-set entropy, and growth in the structured rule-slot count. Rollouts are mixed across all 25 demo games to discourage per-game overfitting.

**Stage C — Online test-time LoRA adaptation.** At evaluation, each new environment receives a freshly initialized LoRA of rank 8–16 on top of the frozen Stage B weights. We allocate the first ~30% of the per-level action budget to BC-style fine-tuning on the in-game trajectory accumulated so far, and the remaining ~70% to forward-only inference. The LoRA is reset between environments to comply with the no-task-specific-optimization rule [4]. This stage operationalizes the test-time-training principle that has been shown to more than double accuracy on ARC-AGI-1/2 [13], while bounding the additional compute within the 12-hour Kaggle budget.

**Compute budget (estimated).** Stage A: 50–100 GPU-hours on a single A100; Stage B: 100–200 GPU-hours; Stage C: amortized into the Kaggle 12-hour evaluation envelope. Final inference cost is well below the official $50 / 120-task ceiling [3].

### 4.6 [TBD] Phase 3 Implementation: Search with World Model

[pending Phase 3] If Stage C alone is insufficient, we plan a shallow lookahead (BFS / MCTS, depth 1–3) over the Path-B rule table, with UNDO as the natural backtrack operator.

---

## 5. Experiments

### 5.1 Setup

- 数据集:ARC-AGI-3 public demo (25 environments)
- 预算:每 level 5h(human action budget),每 episode token 上限 [TBD]
- 重复:每 environment N=[TBD] 次,报均值 ± 标准误

### 5.2 Main Results

[完成第一组实验后填表;数字必须从 EXPERIMENTS.md 抄过来]

| Agent | 平均 RHAE | 过关率 (level 1) | 平均 step / episode | API 成本 / episode |
|-------|-----------|------------------|----------------------|--------------------|
| Random (Phase 0, 1 ep × 80-step cap) [Exp 2026-04-27] | **0.000** | **0%** | 80 (capped) | $0 (within free RPM) |
| LLMAgent-prompt-HEI (Phase 1) | [TBD] | [TBD] | [TBD] | $[TBD] |
| +Hybrid Architecture (Phase 2) | [TBD] | [TBD] | [TBD] | $[TBD] |
| +Online LoRA (Phase 3) | [TBD] | [TBD] | [TBD] | $[TBD] |

### 5.3 Ablations

[TBD — Phase 2/3 后]
计划的 ablation:
- 关掉 IG 项(只优化 V)→ 测信息增益的贡献
- 关掉 Hypothesis 显式化(让 LLM 自由输出)→ 测 prompt 结构的贡献
- 关掉 cross-level Memory → 测跨关复用规则的贡献

---

## 6. Discussion

[随每次实验追加,失败案例尤其重要]

### 6.1 What worked
- **Random baseline establishes a clean floor.** Our pre-committed Hypothesis (mean RHAE ≤ 0.10, level-1 pass rate ≤ 30% under an 80-step cap) was confirmed at the absolute floor (0.000 / 0%) on the demo-25 set. This validates that all subsequent reported gains are over a non-trivial baseline rather than over a buggy or accidentally-strong random.

### 6.2 What didn't
- [pending Phase 1] —— 把 RESEARCH.md 里 "假设驳斥" 的方案提炼到这里

### 6.3 Limitations
- 终评无网,API LLM 不可用 → 提交时必须替换为本地权重模型,代价是能力下降。我们当前没有定量评估这个 gap。
- HEI 循环依赖 LLM 给出的 V 和 IG 估计本身的准确性;一旦 LLM 在某类游戏失校准,整套循环失效。

### 6.4 Failure modes
[TBD]

---

## 7. Conclusion

[TBD — Phase 3 末写]

---

## References

引用编号在正文里用 `[N]`,这里维护 N → URL 的对照。新增文献时:**先在 RESEARCH.md "读后感" 段写一句话评价,再在这里 append 一行**。

[1] ARC Prize 2026 — ARC-AGI-3 Launch Blog. https://arcprize.org/blog/arc-agi-3-launch
[2] ARC-AGI-3 Technical Paper. https://arxiv.org/abs/2603.24621 *(论文号待二次核对)*
[3] Kaggle Competition page. https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
[4] ARC Prize 2026 Overview. https://arcprize.org/competitions/2026
[5] API Documentation. https://docs.arcprize.org
[6] Browse Tasks. https://arcprize.org/tasks
[7] Leaderboard. https://arcprize.org/leaderboard
[8] D. Smit. *1st Place in the ARC-AGI-3 Agent Preview Competition.* Medium blog, 2025. https://medium.com/@dries.epos/1st-place-in-the-arc-agi-3-agent-preview-competition-49263f6287db
[9] D. Smit. *ARC3-solution* (StochasticGoose source code, Tufa Labs). https://github.com/DriesSmit/ARC3-solution
[10] ARC Prize. *ARC-AGI-3 Preview: 30-Day Learnings.* 2025. https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings
[11] Symbolica AI. *Arcgentica — ARC-AGI-3 agent harness on the Agentica SDK.* https://github.com/symbolica-ai/ARC-AGI-3-Agents
[12] G. Barbadillo (Ironbar). *Iteration 46. Revisit small LLMs* (ARC-AGI-1/2 small-LLM micro-benchmark). https://ironbar.github.io/arc24/modeling/Iteration_46_revisit_small_llms/
[13] *Boosting Performance on ARC is a Matter of Perspective* (test-time training on ARC). arXiv:2505.07859. https://arxiv.org/html/2505.07859v1
[14] ARC Prize. *Measuring Human Performance on ARC-AGI-3* (open-source 458-participant trace dataset). https://arcprize.org/blog/arc-agi-3-human-dataset

[N+] [pending] 后续添加的文献(Dreamer, MuZero, Voyager, ReAct, ARC-AGI-1/2 winners 等)

---

## Changelog

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-04-27 | v0.0 | 骨架建立,全部章节为 [TBD] 占位 |
| 2026-04-27 | v0.05 | §2 Related Work 填入 SOTA(StochasticGoose [8–10]、Arcgentica [11])与方法学(Qwen3 [12]、TTT [13]、Human Dataset [14]);§4 Method 增 4.4 三层混合架构与 4.5 三阶段训练范式;References 加 [8]–[14]。触发于 ARCHITECTURE.md 重要决策变化(本地权重 + 双路规则 + BC→RL→在线 LoRA) |
| 2026-04-27 | v0.06 | §5.2 表格 Random 行填入首组实测(0.000 / 0% / 80 / $0,1 ep × 80-step cap on demo 25);§6.1 What worked 加 Random 基线作为干净下界的论述。触发于 EXPERIMENTS.md 第一条实验入库 |
