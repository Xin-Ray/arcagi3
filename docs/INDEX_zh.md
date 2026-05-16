# 文档总入口 (INDEX_zh.md)

> 进项目前先读这个。状态码 🟢 当前活、🟡 参考、⚫ 历史。

最近更新: 2026-05-16

---

## 0. 项目目标

做一个能自主玩 ARC-AGI-3 回合制谜题游戏的 agent —— **没有任何说明书**,必须自己看出物体行为和通关条件。

- 比赛: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
- 里程碑: 2026-06-30 开源奖 / 2026-09-30 终评
- 当前 SOTA (社区): Symbolica Agentica 36.08% (7/25 通关);前沿大模型(GPT/Gemini/Claude/Grok)在全私集 < 1%

---

## 1. 整体进展(2026-05-16)

| 路线 | 状态 | 最新数字 |
|---|---|---|
| **v3.2 mask + Knowledge + click_targets** | 🟢 主线在跑 | ar25 3×200 step change_rate 5-8%;0 levels won;比 mask-off baseline (2.6-3.8%) 好;远低于 v2 canary (60-100%) |
| **Predictor v0.1 (frame-change CNN)** | 🟢 数据收集中 | CNN val AUC 0.894 (+6pp vs MLP),不接入 prompt;等扩量 |
| **Tier 1 SFT (空间推理 LoRA)** | 🟡 第一轮 FAIL | planning probe 完全没修 + gsm8k -14.7pp,根因 T4 模板锁死;待 F3 重训 |
| **GRPO v0 (ar25 单游戏)** | 🟡 仅设计 | 定位为诊断,Phase 0 plumbing 未开工 |
| **v2 canary 路线复盘** | 🟡 待 A/B | 12 commit / 6300 行改动后 change_rate 退化;3 个候选回滚特征待测 |
| **RL v0 (intrinsic F1 + GRPO)** | ⚫ parked | 被 v3 取代,代码留着 |
| **BC pipeline** | ⚫ archive | 已归档,不复用前必须 promote |

---

## 2. 文档结构

```
docs/
├── INDEX_zh.md            ← 本文件
├── CONVENTIONS_zh.md      ← 文档规范(必读 1 次)
├── GLOSSARY_zh.md         ← 唯一术语字典
├── architecture/          ← 架构设计文档
└── reference/             ← prompt / 数据流 / 评测细节

outputs/reports/
├── INDEX_zh.md            ← 报告总入口
├── <experiment>.md        ← 每次实验一份
└── <experiment>/          ← 该报告的图
```

**3 分钟接管路径**:
1. 本文 §1 (现状 + 进展)
2. [`CONVENTIONS_zh.md`](./CONVENTIONS_zh.md) (文档规范)
3. [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) (当前主线)
4. [`reports/INDEX_zh.md`](../outputs/reports/INDEX_zh.md) (最近实验)

---

## 3. 架构文档 `docs/architecture/`

按时间倒序(新的在上)。

| 状态 | 文档 | 一句话定位 |
|---|---|---|
| 🟢 | [`predictor_v0_zh.md`](./architecture/predictor_v0_zh.md) | 训一个小 CNN 预测 P(frame_change \| state, action),给 LLM 当软推荐 |
| 🟢 | [`grpo_v0_zh.md`](./architecture/grpo_v0_zh.md) | 单游戏 ar25 上跑 GRPO,看 LLM-RL 路线有没有 ceiling(诊断,不参赛) |
| 🟢 | [`v3_2_zh.md`](./architecture/v3_2_zh.md) | **主线**:Action Agent + Reflection Agent + 跨 round Knowledge + click_targets bandit + R1-R7 硬规则 |
| 🟡 | [`v3_zh.md`](./architecture/v3_zh.md) | v3 单 agent:scipy perception + Qwen text-only。被 v3.2 取代,代码留着 |
| 🟡 | [`sft_tier1_zh.md`](./architecture/sft_tier1_zh.md) | 在合成数据上 LoRA SFT 修 Qwen 的空间推理基础能力。第一轮 FAIL,F3 待跑 |
| 🟡 | [`rl_v0_zh.md`](./architecture/rl_v0_zh.md) | 早期 RL 设计(intrinsic F1 reward + GRPO)。parked |

---

## 4. 参考文档 `docs/reference/`

| 文档 | 一句话定位 |
|---|---|
| [`v3_prompt_zh.md`](./reference/v3_prompt_zh.md) | v3 / v3.2 prompt 逐块拆解 + 历次改造记录。**改 prompt 前必读** |
| [`v3_2_dataflow_zh.md`](./reference/v3_2_dataflow_zh.md) | v3.2 三板块真实 I/O 走查(perception / reflection / action) |
| [`v3_2_hardrules_results_zh.md`](./reference/v3_2_hardrules_results_zh.md) | R1+R2+R3 硬规则在 ar25 3×30 上的实测对比(change_rate 23/17/17% → 60/100/97%) |
| [`object_pipeline_zh.md`](./reference/object_pipeline_zh.md) | 视觉感知层评测:scipy 100% vs Qwen-VL 0% |

---

## 5. 实验报告(总入口在 `outputs/reports/INDEX_zh.md`)

最近 5 份:

| 报告 | 对应架构 | 一句话结论 |
|---|---|---|
| [`mask_revive_3x200`](../outputs/reports/mask_revive_3x200.md) | v3_2 | R2 mask 重启,change_rate 5-8% < v2 canary;发现归因 bug 已修 |
| [`predictor_v0`](../outputs/reports/predictor_v0.md) | predictor_v0 | 4 架构对比,CNN 0.894 AUC 最好;不接入 prompt 待扩量 |
| [`trace_balance`](../outputs/reports/trace_balance.md) | predictor_v0 | 全部 6132 step 实测 change_rate 42%,数据足够训预测器 |
| [`grpo_v0_ar25_plan`](../outputs/reports/grpo_v0_ar25_plan.md) | grpo_v0 | GRPO 单游戏诊断的 6 张可视化布局 + 决策门预案 |

---

## 6. 重要历史决策(看这里就不用翻 12 个 commit)

| 决策 | 日期 | 出处 |
|---|---|---|
| 视觉用 scipy 不用 LLM | 2026-05-11 前后 | [`reference/object_pipeline_zh.md`](./reference/object_pipeline_zh.md) |
| v3 → v3.2 拆双 agent | 2026-05-14 | [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §0 |
| R1+R2+R3 orchestrator > prompt | 2026-05-14 (commit `1bac4be`) | [`reference/v3_2_hardrules_results_zh.md`](./reference/v3_2_hardrules_results_zh.md) |
| R2 mask 临时转 advisory | 2026-05-14 (commit `21bca81`) | -- |
| R2 mask 重新启用 + 归因 bug 修复 | 2026-05-16 (commit `01a7227`) | [`reports/mask_revive_3x200`](../outputs/reports/mask_revive_3x200.md) |
| Tier 1 SFT 转 F3 | 2026-05-15 (FAIL 复盘) | [`architecture/sft_tier1_zh.md`](./architecture/sft_tier1_zh.md) §12 |

---

## 7. 关键文件(写代码前 grep)

- `arc_agent/knowledge.py` — Knowledge dataclass
- `arc_agent/action_mask.py` — R2 mask
- `arc_agent/agents/{action_agent,reflection_agent}.py` — v3.2 双 agent
- `arc_agent/predictor/` — frame-change predictor
- `scripts/run_v3_multi_round.py` — 主 runner
- `scripts/train_predictor.py` — 训 predictor
- `arc_agent/rewards.py` — intrinsic F1 reward primitives (parked)

完整模块清单见 `CLAUDE.md` 的「Key modules」。

---

## 8. 怎么用本文档(reader / writer 各一段)

**Reader**(刚接手项目):
- 按上面 §2 的 3 分钟接管路径走
- 任何概念不懂 → 直接到 [`GLOSSARY_zh.md`](./GLOSSARY_zh.md) 查
- 想知道某次实验的细节 → 到 [`outputs/reports/INDEX_zh.md`](../outputs/reports/INDEX_zh.md) 找

**Writer**(要写新文档):
- 先读 [`CONVENTIONS_zh.md`](./CONVENTIONS_zh.md)
- 新架构 → `docs/architecture/<name>_zh.md`,套 §3 模板
- 新实验 → `outputs/reports/<name>.md`,套 §4 模板
- 出现新概念 → 在 [`GLOSSARY_zh.md`](./GLOSSARY_zh.md) 加一条
- 写完后在**本文 §3 / §4 / §5 表格里加一行**

---

## 9. 命名 / 状态约定

(完整规则在 [`CONVENTIONS_zh.md`](./CONVENTIONS_zh.md))

- 文件名:`{arch|ref}_{name}_{version?}_{lang}.md`,**全部小写**
- 状态:🟢 当前活 / 🟡 参考 / ⚫ 历史
- 任何 v 前缀 (v1/v2/v3/v3_2) 表示**设计版本**,新的取代旧的
- archive/ 是单向门;promote 回来要显式

---

*历史:*
- *2026-05-16 重构:加 GLOSSARY + CONVENTIONS,把 arch_*.md / ref_*.md 移到 architecture/ + reference/。这一版按新 conventions 写。*
- *2026-05-15 增加 Tier 1 SFT 条目*
- *2026-05-14 初始版本*
