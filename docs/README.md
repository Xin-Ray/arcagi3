# 项目总入口 (README.md)

> 10 分钟看完知道现状。状态码 🟢 当前活、🟡 参考、⚫ 历史。

最近更新: 2026-05-16

---

## 0. 项目目标

做一个能自主玩 ARC-AGI-3 回合制谜题游戏的 agent —— **没有任何说明书**,必须自己看出物体行为和通关条件。

- 比赛: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
- 里程碑: 2026-06-30 开源奖 / 2026-09-30 终评
- 当前 SOTA(社区): Symbolica Agentica 36.08%(7/25 通关);前沿大模型(GPT/Gemini/Claude/Grok)在全私集 < 1%

---

## 1. 边界和限制

| 维度 | 限制 |
|---|---|
| 模型 | Qwen2.5-VL-3B-Instruct(4-bit + LoRA),text-only 模式 |
| 视觉 | 不让模型看图;scipy.ndimage.label 抽对象,LLM 看结构化文本 |
| 规则 | **No task-specific optimization** —— 不能为特定 game 训 LoRA;不能用 demo / private game label 监督训练 |
| Kaggle 终评 | 110 game,T4 16GB,**离线**(无网),≤ 10 小时总耗时 |
| 工程 | Windows 11 + PowerShell + .venv (Python 3.12);GPU 一台(RTX A4500/3090) |

---

## 2. 当前分支状态

| 分支 | 状态 | 在做什么 |
|---|---|---|
| `main` | 🟢 主线 | v3.2 + mask + Knowledge + click_targets + predictor v0.1 + 文档规范化 |
| `docs-reorg` | 🟡 本次(此分支) | 文档体系迁移到 `docs/project/<name>/` 结构 + README + data.md。完成后合 main |
| `v2-canary` | 🟡 历史快照 | `1bac4be` 的 read-only 引用,R1+R2+R3 引入点 |
| `v2-canary-verify` | 🟡 已完成,不合并 | 在 v2-canary 上复现历史 86% change_rate(✅),作为对照基线保留 |

---

## 3. 当前方向(2026-05-16)

整体瓶颈结论:**ar25 change_rate 7%(main)和 86%(v2 canary)都 0 通关 —— action 选择不是真正的瓶颈,目标推断 / Qwen-3B 能力天花板才是**。

按 ROI 排的下一步:

1. **action_proposer**:把 action 决策从「Qwen 自由生成」改成「代码生成 N 候选 + Qwen N 选 1」,playing to Qwen's strengths。预计 1 天,在 `docs/project/<date>-v0-action_proposer/` 起新分支
2. **换 game 验证模型上限**:Qwen-3B 在 bp35 / cd82 / cn04 上能不能通任何一关?如果别的 game 通,说明 ar25 是特例;全通不了,Qwen 路线封顶
3. **Predictor v0.1 扩量**:加 RandomAgent 跑 25 game × 200 step 收集 PNG 训练数据,重训 CNN

---

## 4. 验证结果汇总

### ✅ 已验证有用(保留在主线)

| 决策 | 实测证据 | Project |
|---|---|---|
| 视觉用 scipy 不用 LLM | scipy 100% vs Qwen-VL 0%(ar25 单帧抽取) | v3-baseline |
| Knowledge 跨 round 持久化(设计) | 双 agent 框架可跑通,plumbing 稳定 | v3_2-double_agent |
| R2 mask 拦 ACTION6 spam | ar25 3×30 change_rate 23/17/17% → 60/100/97% | v3_2-double_agent |
| BUG-10 click_targets bandit | ACTION6 在 ar25 上从「4096 像素盲选」→ 10 个 named target | v3_2-double_agent |
| R2 mask 归因 fix | 修了「OutcomeLog 错归因导致 mask 自废」的 bug | v3_2-double_agent |
| CNN > 手工特征(predictor) | val AUC 0.894 vs LogReg/MLP 0.834,+6pp | v0-predictor |
| v2 canary 状态可复现 | 86.3% 长跑均值,跟历史 85.6% 一致(差 3pp) | v2_canary_verify(分支) |

### ❌ 已验证无用 / 失败 / 不解决问题

| 尝试 | 失败证据 | Project |
|---|---|---|
| Tier 1 SFT 第一轮(T4 模板锁死) | planning probe 0/5 没修 + gsm8k -14.7pp | v0-sft_tier1 |
| Reflection 早写 action_semantics | LLM 95% commit ACTION1 → main change_rate 跌 60-100% 到 5-8% | v3_2-double_agent |
| `[ACTION FORECAST]` 接入 prompt | per-action AUC 0.46(ACTION1 比随机差),信号太弱 | v0-predictor |
| 单靠 mask 提通关率 | v2 canary 86% change_rate 也是 0 levels won | v2_canary_verify |
| Knowledge 满 ↔ 高 change_rate | trade-off,不是叠加;充满 ACTION1 anchor LLM | v3_2 vs v2_canary_verify |

### ⚠️ 部分有用 / 待补充验证

| 尝试 | 状态 | Project |
|---|---|---|
| Predictor v0.1 CNN(raw grid) | AUC 高但 val n=14 太小,需扩量数据 | v0-predictor |
| GRPO 单游戏诊断 | 仅文档,Phase 0 plumbing 未做 | v0-grpo |
| v2_canary_ablation A/B 计划 | 设计写完,3 个 toggle 未跑 | v0-v2_canary_ablation |
| ar25 上的 ceiling | 不确定是模型问题还是 game 特例,需换 game 实测 | (没专门 project) |

### 🔬 没测过(下一步可能做)

- **action_proposer**: 代码 propose N 候选 + Qwen N 选 1
- **换 game 测 Qwen 能力上限**(bp35/cd82/cn04)
- **Claude API 跑 v3.2 框架**: 看大模型在我们 harness 下的天花板
- **Predictor v0.1 扩量**: RandomAgent 跑 25 game × 200 step 收集 PNG 训练数据

---

## 5. 文档结构

```
docs/
├── README.md             ← 本文件(总入口,10 分钟读完)
├── CONVENTIONS_zh.md     ← 文档规范(写新 doc 前看)
├── GLOSSARY_zh.md        ← 唯一术语字典
├── data/
│   └── data.md           ← 数据来源 + 内容 + 用法
└── project/              ← 每个 project = 1 个分支 = 1 个版本,命名 <date>-<version>-<name>/
    ├── 2026-04-27-v0-rl/                  ← 早期 RL(parked)
    ├── 2026-05-11-v3-baseline/            ← scipy + text-only Qwen 单 agent
    ├── 2026-05-14-v3_2-double_agent/      ← Action+Reflection 双 agent + Knowledge(主线)
    ├── 2026-05-15-v0-sft_tier1/           ← 合成数据 LoRA(第一轮 FAIL)
    ├── 2026-05-16-v0-predictor/           ← frame-change CNN 预测器
    ├── 2026-05-16-v0-grpo/                ← 单游戏 GRPO 诊断(仅设计)
    └── 2026-05-16-v0-v2_canary_ablation/  ← 找 v2 canary → main 退化主因(未启动)
```

每个 project 子文件夹典型布局:
- `architecture.md` —— 设计(人工核心管理)
- `reference_*.md` —— prompt / 数据流 / 评测细节(人工核心管理)
- `report*.md` —— 实测结果(Claude auto 写,人工审)
- `figures/` —— 报告引用的 PNG / GIF

---

## 5. 版本历史(每个 project 一段)

按时间顺序,最新在最下面。

---

### v3 — `docs/project/v3/`

- **路径**: [`docs/project/v3/architecture.md`](./project/2026-05-11-v3-baseline/architecture.md) + `reference_prompt.md` + `reference_object_pipeline.md`
- **状态**: 🟡 被 v3.2 取代,代码留着
- **关键 commits**: `01a7e58` (v3 初版) → `2cbd1e1` (P0-A/B) → `0aa92c4` (P1)
- **一句话**: scipy perception + Qwen text-only 单 agent。**确立了「视觉用算法,推理用 LLM」的设计原则**

---

### rl_v0 — `docs/project/rl_v0/`

- **路径**: [`docs/project/rl_v0/architecture.md`](./project/2026-04-27-v0-rl/architecture.md)
- **状态**: ⚫ parked
- **一句话**: 最早的 RL 路线设计(intrinsic F1 reward + GRPO),实测后被 v3 取代

---

### sft_tier1 — `docs/project/sft_tier1/`

- **路径**: [`docs/project/sft_tier1/architecture.md`](./project/2026-05-15-v0-sft_tier1/architecture.md)
- **状态**: 🟡 第一轮 FAIL,F3 修法待跑
- **关键 commits**: 在 `622f4bd` 快照内
- **一句话**: 合成数据 LoRA 修 Qwen 空间推理基础能力。第一轮 holdout 100% 但 planning probe 0% + gsm8k -14.7pp

---

### v3_2 — `docs/project/v3_2/`

- **路径**: [`docs/project/v3_2/architecture.md`](./project/2026-05-14-v3_2-double_agent/architecture.md) + `reference_dataflow.md` + `report_hardrules.md` + `report_mask_revive.md`
- **状态**: 🟢 当前主线,正在迭代
- **关键 commits**: `1bac4be` (R1+R2+R3) → `21bca81` (mask advisory) → `4680508` → `ec86881` → `439ca59` → `eac5f1a` → `01a7227` (mask 归因 fix)
- **一句话**: Action Agent + Reflection Agent + 跨 round Knowledge + click_targets bandit + R1-R7 硬规则。实测 ar25 change_rate 5-8%,0 通关
- **报告**:
  - [`report_hardrules.md`](./project/2026-05-14-v3_2-double_agent/report_hardrules.md) —— R1+R2+R3 引入前后对比(60-100% change_rate)
  - [`report_mask_revive.md`](./project/2026-05-14-v3_2-double_agent/report_mask_revive.md) —— 2026-05-16 mask 重启 + 归因 bug 修

---

### predictor_v0 — `docs/project/predictor_v0/`

- **路径**: [`docs/project/predictor_v0/architecture.md`](./project/2026-05-16-v0-predictor/architecture.md) + `report.md` + `report_trace_balance.md`
- **状态**: 🟢 已完成 v0 评测,v0.1 等扩量
- **关键 commits**: `aa75ed9` (predictor v0 base) → `3def117` (CNN +6pp) → `dfc2ee1` (tests)
- **一句话**: 训小模型(LogReg / MLP / CNN)预测 `P(frame_change | state, action)`。CNN val AUC 0.894 最好,但 ablation 显示 13pp 信号来自 game_id 先验,真正状态信号弱。不接入 prompt,等扩量数据

---

### grpo_v0 — `docs/project/grpo_v0/`

- **路径**: [`docs/project/grpo_v0/architecture.md`](./project/2026-05-16-v0-grpo/architecture.md) + `report_skeleton.md`
- **状态**: 🟡 仅设计,Phase 0 plumbing 未开工
- **一句话**: 单游戏 ar25 GRPO 诊断 —— 看 LLM-RL 路线在 ar25 上有没有 ceiling。**不是 Kaggle 提交方案**

---

### v2_canary_ablation — `docs/project/v2_canary_ablation/`

- **路径**: [`docs/project/v2_canary_ablation/architecture.md`](./project/2026-05-16-v0-v2_canary_ablation/architecture.md)
- **状态**: 🟡 仅设计,A/B 未跑
- **一句话**: 在 main 上加 `--reflect-semantics / --click-targets / --prompt-format` 3 个 toggle,A/B 找 v2 canary → main 退化主因

---

### v2_canary_verify — 仅在 `v2-canary-verify` 分支

- **路径**: (主分支无;`git checkout v2-canary-verify` 后看)
- **状态**: 🟡 已完成,不合并主线
- **关键 commits**: 分支 `v2-canary-verify` `38d7a7d` (复原模块) → `cda68f9` (实验报告)
- **一句话**: 在 `1bac4be` 上跑 ar25 3×30 + 2×200,复现 86% change_rate ✅。**确认 main 退化是代码引起,不是环境**。但 86% 也 0 通关 —— **trade-off:Knowledge 空 ↔ 高探索 / Knowledge 满 ↔ ACTION1 commit**

---

### docs-reorg — (本分支,未合并)

- **路径**: 本文件 + `data/data.md` + `project/` 目录结构本身
- **状态**: 🟡 进行中
- **一句话**: 把扁平 `docs/architecture/` + `docs/reference/` + `outputs/reports/` 收编进 `docs/project/<name>/`,加 README + GLOSSARY + CONVENTIONS 三件套

---

## 6. 3 分钟接管路径

刚加入项目?按这个顺序读:

1. 本文(项目总览 + 当前进展)
2. [`CONVENTIONS_zh.md`](./CONVENTIONS_zh.md)(文档规范)
3. [`project/v3_2/architecture.md`](./project/2026-05-14-v3_2-double_agent/architecture.md)(当前主线设计)
4. [`project/v3_2/report_mask_revive.md`](./project/2026-05-14-v3_2-double_agent/report_mask_revive.md)(最新实测)
5. 不懂的术语 → [`GLOSSARY_zh.md`](./GLOSSARY_zh.md)

---

## 7. 关键代码文件(grep 入口)

- `arc_agent/knowledge.py` — Knowledge dataclass
- `arc_agent/action_mask.py` — R2 mask
- `arc_agent/agents/{action_agent,reflection_agent}.py` — v3.2 双 agent
- `arc_agent/predictor/` — frame-change predictor
- `scripts/run_v3_multi_round.py` — v3.2 主 runner
- `scripts/train_predictor.py` — predictor 训练
- `arc_agent/rewards.py` — intrinsic F1 reward primitives (parked)

完整模块清单见 `CLAUDE.md` 的「Key modules」。

---

## 8. 维护

- 新方向 → 在 `docs/project/<name>/` 起目录 + 1 个 git 分支 + 写 `architecture.md`
- 项目跑出实验 → 在该目录写 `report.md` + 把图放 `figures/`
- 概念落地 → 加进 [`GLOSSARY_zh.md`](./GLOSSARY_zh.md)
- 项目完结 → 回到本文 §5 加一段
- 失败的项目 → 用户判断,在另一个分支记录代码片段 + 经验

更多见 [`CONVENTIONS_zh.md`](./CONVENTIONS_zh.md)。

---

*历史:*
- *2026-05-16 重构:扁平 docs → `project/<name>/` 结构。从 INDEX_zh.md 改名而来,加分支状态表 + 版本历史。*
