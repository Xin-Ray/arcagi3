# 项目总入口 (README.md)

> 10 分钟看完知道现状。状态码 🟢 当前活、🟡 参考、⚫ 历史。

最近更新: 2026-05-17 14:55

> **昨晚跨分支汇报**: [`tonight_summary.md`](./tonight_summary.md) + [`figures/tonight_summary.png`](./figures/tonight_summary.png)

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
| 模型 | Qwen2.5-VL-3B-Instruct / SmolLM3-3B(候选)/ Phi-4-mini-reasoning(实测劣)。4-bit + LoRA,text-only |
| 视觉 | 不让模型看图;scipy.ndimage.label 抽对象,LLM 看结构化文本 |
| 规则 | **No task-specific optimization** —— 不能为特定 game 训 LoRA;不能用 demo / private game label 监督训练 |
| Kaggle 终评 | 110 game,T4 16GB,**离线**(无网),≤ 10 小时总耗时 |
| 工程 | Windows 11 + PowerShell + .venv (Python 3.12);GPU 一台(RTX A4500/3090) |

---

## 2. 当前分支状态(2026-05-17 14:55)

| 分支 | 状态 | 顶端 commit | 在做什么 |
|---|---|---|---|
| `main` | 🟢 主线 | `e07e7d1` (2026-05-16) | v3.2 + mask + Knowledge + click_targets。**没有任何 push/merge 发生**;所有新工作都在 feature 分支 |
| `feat-2026-05-16-v0-action_proposer` | 🟢 **活跃** | (本分支)`b29mzqh3m` 跑中 | 包含 action_proposer + model_bench + SmolLM3 5×2×300 + CoT 1×2×100 跑中 |
| `docs-reorg` | 🟡 等合 | `181d0b6` (2026-05-16) | 文档体系迁移,昨天写完 |
| `feat-2026-05-16-v01-predictor` | ❌ 不合 | `722db78` (2026-05-16) | CNN OOD AUC 0.16 失败结论存档 |
| `feat-2026-05-16-v0-grpo_train` | ⏳ Phase 0 | `11ed21f` (2026-05-16) | rollout_wrapper + 10 unit tests pass。真训练未跑 |
| `v2-canary` | 🟡 历史快照 | `1bac4be` (2026-05-14) | R1+R2+R3 引入点,read-only |
| `v2-canary-verify` | 🟡 已完成 | `cda68f9` (2026-05-16) | 86% change_rate 复现成功,不合主线 |

---

## 3. 当前方向(2026-05-17)

**今天重大发现**:SmolLM3 +17pp 优势**全部来自 reasoning chain (CoT)**。生产用 `/no_think` 模式 → SmolLM3 **55.2% = Qwen baseline**。但 SmolLM3 5×2×300 仍拿 64% mean change_rate(vs Qwen 5-8%),所以**真正起作用的不是空间推理而是 action_proposer + Knowledge + instruction following 的协同**。

按 ROI 排的下一步:

1. **CoT 1×2×100 ar25 实测中** — 验证 CoT 模式实际通关能力(慢 10×,但精度高)
2. **subtask 拆分**(用户建议,关键) — 把「通关」拆成 7 个可测的小任务(T-NAV-1..3, T-SEL-1..2, T-GOAL, T-RETRY),每个一个分支
3. **Qwen+propose 5×2×300 对照** — 确认 SmolLM3 优势是模型还是 action_proposer 结构

---

## 4. 验证结果汇总

### ✅ 已验证有用(保留)

| 决策 | 实测证据 | Project / 日期 |
|---|---|---|
| 视觉用 scipy 不用 LLM | scipy 100% vs Qwen-VL 0%(ar25 单帧抽取) | v3-baseline (2026-05-11) |
| Knowledge 跨 round 持久化 | 5/5 game `rounds_played=2`,`goal_hypothesis` 跨 round 保留 | v3_2 (2026-05-14) + 实测 2026-05-17 |
| R2 mask 拦 ACTION6 spam | ar25 3×30 change_rate 23/17/17% → 60/100/97% | v3_2 (2026-05-14) |
| BUG-10 click_targets bandit | ACTION6 命名 target,从 4096 像素盲选 → ~10 个 target | v3_2 (2026-05-14) |
| R2 mask 归因 fix | OutcomeLog 错归因 bug 修复 | v3_2 (2026-05-16) |
| v2 canary 状态可复现 | 86.3% 长跑均值,跟历史 85.6% 一致 | v2_canary_verify (2026-05-16) |
| **action_proposer N 选 1** | ar25 3×30 60/70/75%,Knowledge 4 entries + medium confidence | action_proposer (2026-05-16) |
| **SmolLM3-3B replace Qwen 跨 5 game** | **5×2×300 mean change_rate 64% (vs main 5-8%, +57pp);ar25 r0 87%** | model_bench (2026-05-17) |
| **SmolLM3 探索多样性** | ar25 r0 ACTION1 占比 21% (vs main 95%);7 action 都用到 | model_bench (2026-05-17) |
| **goal_hypothesis 质量飞跃** | "align the two yellow squares vertically in the left column" + medium confidence(Qwen 通常 low) | model_bench (2026-05-17) |

### ❌ 已验证无用 / 失败 / 不解决问题

| 尝试 | 失败证据 | Project / 日期 |
|---|---|---|
| Tier 1 SFT 第一轮(T4 模板锁死) | planning probe 0/5 没修 + gsm8k -14.7pp | sft_tier1 (2026-05-15) |
| Reflection 早写 action_semantics | LLM 95% commit ACTION1 → main change_rate 跌到 5-8% | v3_2 (2026-05-14..16) |
| 单靠 mask 提通关率 | v2 canary 86% change_rate 也是 0 levels won | v2_canary_verify (2026-05-16) |
| Predictor v0:`[ACTION FORECAST]` prompt | per-action AUC 0.46(ACTION1 比随机差) | predictor_v0 (2026-05-16) |
| **Predictor v0.1:CNN OOD 泛化** | 5-fold LOO-game CNN AUC 0.16 (worse than random) | predictor_v01 (2026-05-16) |
| **Phi-4-mini-reasoning 在 spatial probe** | bench accuracy **41.4%**(低于 Qwen 55.2%);reasoning chain 走偏 | model_bench (2026-05-17) |
| **SmolLM3 `/no_think` 模式 spatial 推理** | bench 55.2% **完全等于 Qwen baseline** —— CoT 的 17pp 优势消失 | model_bench (2026-05-17) |
| **任何 backbone 单换都解决通关** | SmolLM3 5×2×300 mean 64% change_rate 但 **0 levels won 全 5 game** | model_bench (2026-05-17) |

### ⚠️ 部分有用 / 待补充验证

| 尝试 | 状态 | Project / 日期 |
|---|---|---|
| **SmolLM3 CoT 模式生产可行性** | bench 72.4% 是好的;1×2×100 CoT 跑中(`b29mzqh3m`),~90 min | model_bench (2026-05-17) |
| GRPO Phase 0 plumbing | rollout_wrapper + 10 测试,真训练未做 | grpo_train (2026-05-16) |
| v2_canary_ablation A/B 计划 | 设计完,3 个 toggle 未跑 | v2_canary_ablation (2026-05-16) |
| Qwen+propose **5×2×300 干净对照** | **缺失** —— 只有 3×30 smoke + 我昨天 kill 的 step 151 部分数据 | (待跑) |

### 🔬 没测过(下一步可能做)

- **Subtask 拆分实验** —— ar25 通关分解为 7 个小任务,每个独立验证
- **换 game 测 Qwen 能力上限** —— bp35/cd82/cn04 任一通关吗?5×2×300 已部分回答(0/5 通关)
- **Claude API 跑 v3.2+propose** —— 大模型在我们 harness 下的天花板
- **Predictor 重做** —— 不同 target / 更多数据(被 v0.1 失败结论暂停)

---

## 5. 文档结构

```
docs/
├── README.md             ← 本文件(总入口,10 分钟读完)
├── tonight_summary.md    ← 2026-05-16 跨分支汇总
├── CONVENTIONS_zh.md     ← 文档规范(写新 doc 前看;2026-05-17 加报告必须列 outputs/ 路径)
├── GLOSSARY_zh.md        ← 唯一术语字典
├── figures/              ← 跨项目可视化图
├── data/
│   └── data.md           ← 数据来源 + 内容 + 用法
└── project/              ← 每个 project = 1 个分支 = 1 个版本,命名 <date>-<version>-<name>/
    ├── 2026-04-27-v0-rl/
    ├── 2026-05-11-v3-baseline/
    ├── 2026-05-14-v3_2-double_agent/
    ├── 2026-05-15-v0-sft_tier1/
    ├── 2026-05-16-v0-predictor/             v0(in-distribution假象)
    ├── 2026-05-16-v01-predictor/            v0.1(LOO-game CV 证伪)
    ├── 2026-05-16-v0-grpo/
    ├── 2026-05-16-v0-grpo_train/            Phase 0 plumbing
    ├── 2026-05-16-v0-v2_canary_ablation/
    ├── 2026-05-16-v0-action_proposer/        K=3 propose + N 选 1
    └── 2026-05-17-v0-model_bench/            🆕 3 model bench + SmolLM3 5×2×300 + /no_think 发现
```

---

## 6. 版本历史(最新在上)

---

### 2026-05-17 14:55 — 🟢 model_bench v0 — `docs/project/2026-05-17-v0-model_bench/`

- **分支**: `feat-2026-05-16-v0-action_proposer`(复用)
- **关键 commits**: `f037b03`(28 spatial probes) → `bb701f4`(SmolLM3 winner +17pp CoT) → `d7740e0`(5×2×300 报告) → 今天(/no_think reveals = Qwen baseline,CoT 1×2×100 跑中)
- **一句话**: 28 道 spatial probe + 3 model 对照。SmolLM3 CoT 72.4% / `/no_think` 55.2% / Qwen 55.2% / Phi-4 41.4%。**CoT 是关键**,/no_think SmolLM3 跟 Qwen 等价
- **5×2×300 实测**: SmolLM3 `/no_think` mean change_rate **64%** vs main 5-8%。**0 levels won 全 5 game**
- **关键 outputs**:
  - `outputs/model_bench_20260517-015413/` (CoT bench)
  - `outputs/model_bench_20260517-144223/` (`/no_think` bench)
  - `outputs/smollm3_5game_20260517-022946/` (5×2×300 orchestrator)
  - `outputs/smollm3_5game_<game>_*/` × 5 games (per-game traces + 10 play.gif)
  - `outputs/smollm3_cot_ar25_2x100_*` (跑中)

---

### 2026-05-16 19:30 — 🟢 action_proposer v0 — `docs/project/2026-05-16-v0-action_proposer/`

- **分支**: `feat-2026-05-16-v0-action_proposer`(未合)
- **关键 commits**: `d101e14`(主实现)→ `74481f8`(ClickTarget fix)→ `6a4bdd1`(GameAction scoping)→ `75af99d`(报告)
- **一句话**: 代码 propose K=3 候选 + Qwen N 选 1。ar25 3×30 change_rate **60/70/75%** vs main 5-8%;Knowledge 满 4 entries,confidence medium
- **关键 outputs**:
  - `outputs/ap_v0_ar25_3x30_v2_20260516-184956/`(主 smoke 数据)

---

### 2026-05-16 19:30 — 🟢 docs-reorg — `docs/project/` 目录结构

- **分支**: `docs-reorg`(未合)
- **关键 commits**: `7a13cab`(INDEX → README)→ `4f3cb36`(目录重组)→ `3178c37`(日期前缀)→ `5dc99ab`(tonight summary)
- **一句话**: 扁平 `docs/architecture/` + `docs/reference/` + `outputs/reports/` → `docs/project/<date>-<v>-<name>/`;加 README + GLOSSARY + CONVENTIONS + data.md
- **关键 outputs**: 无(纯 docs 改动)

---

### 2026-05-16 19:15 — ⏳ grpo_train v0(Phase 0) — `docs/project/2026-05-16-v0-grpo_train/`

- **分支**: `feat-2026-05-16-v0-grpo_train`(部分,未合)
- **关键 commits**: `67f8c6c`(架构)→ `11ed21f`(plumbing + 10 tests)
- **一句话**: rollout_wrapper + JSON prompt + reward 计算单测。**真训练 16-20h 未跑**
- **关键 outputs**: 无(无训练数据,Phase 1 才有)

---

### 2026-05-16 18:50 — ❌ predictor v0.1 — `docs/project/2026-05-16-v01-predictor/`

- **分支**: `feat-2026-05-16-v01-predictor`(不合)
- **关键 commits**: `3d9865d`(CV 框架 + CNN-deep)→ `722db78`(报告 + 图)
- **一句话**: 5-fold leave-one-game-out CV:CNN mean AUC **0.16** (比随机差);路线 deprecate
- **关键 outputs**:
  - `outputs/predictor_v01_cv/`(metrics + 6 plots)

---

### 2026-05-16 17:25 — 🟡 v2_canary_verify(仅在 v2-canary-verify 分支)

- **分支**: `v2-canary-verify`(不合)
- **关键 commits**: `38d7a7d`(复原模块)→ `cda68f9`(实验报告)
- **一句话**: 在 `1bac4be` 上跑 ar25 3×30 + 2×200,复现 86% change_rate ✅,**确认主线退化是代码引起**。但 86% 也 0 通关
- **关键 outputs**:
  - `outputs/v2_canary_verify_3x30_20260516-170819/`
  - `outputs/v2_canary_verify_2x200_20260516-170927/`

---

### 2026-05-16 03:09 — 🟢 v3_2(包括 mask_revive)— `docs/project/2026-05-14-v3_2-double_agent/`

- **分支**: `main`(已 commit)
- **关键 commits**: `1bac4be`(R1+R2+R3)→ ... → `eac5f1a` → `01a7227`(mask 归因 fix)
- **一句话**: Action Agent + Reflection Agent + 跨 round Knowledge + click_targets + R1-R7 硬规则。**ar25 5-8% change_rate 0 通关**
- **关键 outputs**:
  - `outputs/v3_2_ar25_3x30_v2/`(v2 canary 历史)
  - `outputs/mask_revive_3x200_20260516-014356/`(2026-05-16 重启 mask)
  - `outputs/preload_v3_budget_smoke_20260515-004233/`(mask off baseline)

---

### 2026-05-16 ≤ 02:00 — 🟢 predictor v0 — `docs/project/2026-05-16-v0-predictor/`

- **分支**: `main`(已 commit)
- **关键 commits**: `aa75ed9`(predictor base)→ `3def117`(CNN)→ `dfc2ee1`(tests)
- **一句话**: CNN val AUC 0.894 但 ablation 显示 13pp 来自 game_id 先验;in-distribution 假象,v0.1 证伪
- **关键 outputs**:
  - `outputs/predictor_v0/` + `outputs/predictor_v0_no_game/` + `outputs/predictor_v0_with_cnn/`

---

### 2026-05-15 — 🟡 sft_tier1 — `docs/project/2026-05-15-v0-sft_tier1/`

- **分支**: `main`(已 commit)
- **关键 commits**: 在 `622f4bd` 快照内
- **一句话**: 合成数据 LoRA。第一轮 holdout 100% 但 planning probe 0% + gsm8k -14.7pp。F3 修法未跑
- **关键 outputs**:
  - `outputs/finetune/qwen3b-tier1-lora/`(LoRA adapter)
  - `outputs/finetune/{tier1_train,tier1_holdout,tier1_ood}.jsonl`

---

### 2026-05-14 ≤ — 🟡 v3_baseline + v3_2 早期 — `docs/project/2026-05-11-v3-baseline/`

- **分支**: `main`
- **关键 commits**: `01a7e58` (v3 初版) → `2cbd1e1` (P0-A/B) → `0aa92c4` (P1) → `1bac4be` (R1+R2+R3)
- **一句话**: scipy + Qwen text-only 单 agent。**确立「视觉用算法,推理用 LLM」**
- **关键 outputs**: `outputs/v3_p0b_p1_full/`(5 game × 80 step + 5 GIF)

---

### 2026-04-27 — ⚫ rl_v0(parked)— `docs/project/2026-04-27-v0-rl/`

- **分支**: `main`
- **一句话**: 最早 RL 路线(intrinsic F1 reward + GRPO),被 v3 取代
- **关键 outputs**: 无活产物

---

## 7. 3 分钟接管路径

刚加入项目?按这个顺序读:

1. 本文(项目总览 + 当前进展)
2. [`tonight_summary.md`](./tonight_summary.md)(2026-05-16 跨分支汇报)
3. [`CONVENTIONS_zh.md`](./CONVENTIONS_zh.md)(文档规范)
4. [`project/2026-05-17-v0-model_bench/report_5game.md`](./project/2026-05-17-v0-model_bench/report_5game.md)(最新 5×2×300 + SmolLM3 结果)
5. [`project/2026-05-14-v3_2-double_agent/architecture.md`](./project/2026-05-14-v3_2-double_agent/architecture.md)(主线设计)
6. 不懂的术语 → [`GLOSSARY_zh.md`](./GLOSSARY_zh.md)

---

## 8. 关键代码文件(grep 入口)

- `arc_agent/knowledge.py` — Knowledge dataclass
- `arc_agent/action_mask.py` — R2 mask
- `arc_agent/agents/{action_agent,reflection_agent}.py` — v3.2 双 agent
- `arc_agent/action_proposer.py` — K=3 候选生成(action_proposer 分支)
- `arc_agent/vlm_backbone.py` — Qwen + CausalLMBackbone factory;`reasoning_mode` 参数(2026-05-17 加)
- `arc_agent/bench_probes/__init__.py` — 28 道 spatial probe(2026-05-17 加)
- `arc_agent/rollout_wrapper.py` — GRPO Phase 0 plumbing(grpo_train 分支)
- `arc_agent/predictor/` — frame-change predictor(v0 / v0.1,deprecated)
- `scripts/run_v3_multi_round.py` — v3.2 主 runner(支持 `--mask` / `--propose` / `--backbone` / `--reasoning-mode`)
- `scripts/bench_models_spatial.py` — sequential model bench(2026-05-17 加)
- `scripts/run_action_proposer_5game.py` — G_base 5 game orchestrator
- `scripts/plot_smollm3_vs_qwen.py` — SmolLM3 vs Qwen 对照图
- `scripts/train_predictor_cv.py` — predictor 跨游戏 CV
- `arc_agent/rewards.py` — intrinsic F1 reward primitives(parked)

完整模块清单见 `CLAUDE.md` 的「Key modules」。

---

## 9. 维护

- 新方向 → 在 `docs/project/<date>-<version>-<name>/` 起目录 + 1 个 git 分支 `feat-<date>-<v>-<name>` + 写 `architecture.md`
- 项目跑出实验 → 在该目录写 `report.md` + 把图放 `figures/`;**报告头部列出所有 `outputs/` 路径**(2026-05-17 新规)
- 概念落地 → 加进 [`GLOSSARY_zh.md`](./GLOSSARY_zh.md)
- 项目完结 → 回到本文 §6 加一段(**最新在最上面**)
- 失败的项目 → 用户判断,在该分支留报告 + 经验,主分支不合并代码,只合并经验文档

更多见 [`CONVENTIONS_zh.md`](./CONVENTIONS_zh.md)。

---

*历史:*
- *2026-05-17 14:55 大更新:加 model_bench v0(SmolLM3 winner / /no_think 反转 / 5×2×300 结果)。版本历史改成倒序最新在上 + 时间戳。承认昨晚没及时 sync 是我违反规则*
- *2026-05-16 重构:扁平 docs → `project/<name>/` 结构。从 INDEX_zh.md 改名,加分支状态表 + 版本历史*
