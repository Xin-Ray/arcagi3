# ARCHITECTURE — GRPO Single-Game Training (v0)

日期: 2026-05-16
状态: 设计 → 试点(可在 v3.2 + mask 跑稳后启动)
前置阅读: [`arch_v3_2_zh.md`](./arch_v3_2_zh.md), [`arch_rl_v0_zh.md`](./arch_rl_v0_zh.md), [`arch_predictor_v0_zh.md`](./arch_predictor_v0_zh.md)

---

## 0. 为什么做 + 为什么是「单游戏」

ARC Prize 2026 规则原文(`TASK_OVERVIEW.md`):「No task-specific optimization」。**全 demo 25 game 联训会构成 task-specific optimization**。

但是 — 单游戏试点是合法的诊断手段:**回答一个具体问题:「v3.2 框架本身的 ceiling 在哪里? 一个用 GRPO + intrinsic reward 学过的 Qwen LoRA 能不能在 ar25 上稳定通关?」**

如果**单游戏过拟合都达不到 50% pass rate**,我们就知道:
- Qwen-3B 即使被 RL 调过,在 ar25 这种游戏上也不行 → 路线得换
- 不是 prompt 问题,不是 perception 问题,是模型能力问题

如果**单游戏过拟合到 80%+** → 知道至少决策上限存在,可以投资 Tier 2 / 通用化方法。

**这是研究价值**,不是参赛方案。本文档明确:

- 训练只在 **ar25** 上做
- LoRA adapter 不进 Kaggle 提交
- 训练数据 = ar25 rollouts(同 game 训 + eval,过拟合 OK)
- 文档命名 `arch_grpo_v0_zh.md` 提醒「v0 是诊断」

---

## 1. 范围

### 1.1 In-scope

- 在 ar25 单 game 上跑 GRPO,目标:让 LoRA 学到「在 ar25 上选 ACTION 让 frame 改变 + 推进关卡」
- intrinsic F1 reward 已实现(`arc_agent/rewards.py`),复用
- LoRA adapter 训完 dump 到 `outputs/grpo_v0_ar25_<ts>/checkpoint/`,与 base Qwen 解耦
- 可视化:训练 loss / reward / KL / advantage,episode-level pass_rate / change_rate / unique_state
- A/B 对比 baseline(同 v3.2 框架,不加 LoRA)在 ar25 上的表现

### 1.2 Out-of-scope

- 全 25 game 联训(违反规则)
- 在线 rollout 期间更新 reward model(用 ground-truth F1)
- 将 LoRA 用于 Kaggle 提交
- predictor (`arch_predictor_v0_zh.md`) 集成 — 留待 v0.1,先各自验证

### 1.3 规则合规

| 选择 | 合规理由 |
|---|---|
| 训练只在 ar25 上做 | 单游戏诊断,不是 task-specific submission strategy |
| **不**用 LoRA 提交 Kaggle | 我们清楚标记 v0 是研究、不参评。Kaggle 用 base Qwen + v3.2 framework |
| 开源 reward + LoRA + 训练代码 | 满足 open-source 条款 |
| 用 intrinsic F1(自监督) | 不靠人工 label,通用方法 |

---

## 2. 概念回顾(从 `arch_rl_v0_zh.md` §3)

每步 reward:

```
r = R_WIN          if state == WIN              # +1.0
    + R_F1_COEFF * f1   if parse_ok               # +0.2 * f1
    + R_PARSE_FAIL      if not parse_ok           # -0.5
    + R_ILLEGAL_ACTION  if action not in legal   # -0.3
    + R_ENTITY_BONUS    if entity_recog consistent # +0.05
```

实现在 `arc_agent/rewards.py:reward_fn(step: StepRecord) -> float`,**已单测,不动**。

GRPO trainer skeleton 在 `arc_agent/train_grpo.py:build_trainer()`,惰性导入 `trl`,目前是 stub 状态。

---

## 3. 数据流(rollout → reward → policy update)

```
ar25 SDK env  ──env.step──▶  FrameDataRaw
                                    │
                                    ▼
                          v3.2 ActionAgent(LoRA-wrapped Qwen)
                                    │
                                    ▼  prompt → JSON
                          {"action": "ACTION3", "predicted_changes": [{"row":...}...]}
                                    │
                                    ▼
                          env.step(action) → FrameDataRaw_next
                                    │
                                    ▼
                       rewards.real_changes(s_t, s_{t+1})
                                    │
                                    ▼
                       rewards.verify_prediction_f1(predicted, real) → f1 ∈ [0,1]
                                    │
                                    ▼
                       rewards.reward_fn(StepRecord) → r ∈ [-0.8, 1.25]
                                    │
                                    ▼
                       GRPO trainer 收集 group (8 个 sample/state) 计算 advantage
                                    │
                                    ▼
                       LoRA 反传更新(rank=8,alpha=16,target= q/k/v/o)
```

**关键点**:
- agent 此时**必须**预测 `predicted_changes`(不像 v3.2 的 bare action token);prompt 改回 v3 旧的 JSON 模式
- GRPO 的 group 同 state 跑 8 次,每次得到不同 (action, predicted_changes) 组合,reward 差异驱动 advantage
- 单 game 训练:每个 rollout 都是 ar25 一个 fresh episode,**不复用旧 episode**
- 每 100 rollout 跑一次 val(ar25 跑 10 episode,记 pass_rate + mean reward)

---

## 4. 训练配置(单 GPU,RTX A4500 / RTX 3090)

| 项 | 值 |
|---|---|
| Base model | Qwen2.5-VL-3B-Instruct |
| LoRA | rank=8, alpha=16, dropout=0.05, target=q/k/v/o_proj |
| 量化 | 4-bit (bnb-nf4),compute_dtype=bf16 |
| Optimizer | AdamW, lr=5e-5(LoRA params only),weight_decay=0.0 |
| Scheduler | cosine,warmup=10% |
| Batch | 1 episode/forward * 8 group-replicas = 8 effective |
| GRPO β | KL coef=0.04(防止 LoRA 跑偏 base 太远) |
| Max rollout steps | 80/episode |
| Rollouts/epoch | 100 |
| Epochs | 10 |
| 总 rollouts | 1000 → 1000 × ~60s = ~16h wall clock |
| Val cadence | 每 100 rollout,跑 10 fresh ar25 episode |
| Early stop | val pass_rate plateau ≥ 200 rollout |

**注**:实测 wall clock 可能因 SDK rate limit 而拉长。SDK 每 episode ~ 5s overhead(reset + scorecard 维护)。

---

## 5. 实施清单

### 5.1 新增 / 改的代码

| 文件 | 状态 | 责任 |
|---|---|---|
| `arc_agent/train_grpo.py` | 🟡 扩 | `build_trainer` 实现完整 + `make_rollout_fn` |
| `arc_agent/rollout_wrapper.py` | 🆕 NEW | 包装 v3.2 ActionAgent 暴露 generate + compute_logprobs(trl 需要的接口)|
| `arc_agent/prompts_v3_grpo.py` | 🆕 NEW | v3 旧 JSON prompt(`predicted_changes` 必须输出)|
| `scripts/run_grpo.py` | 🟡 扩 | 接 `build_trainer`,跑训练 loop;`--dry-run` 已能跑 |
| `scripts/eval_grpo.py` | 🆕 NEW | 加载 LoRA + 跑 N episode + 写 metrics.json |
| `scripts/plot_grpo.py` | 🆕 NEW | 从 train_log.jsonl + val_*.json 画 4 张图 |
| `tests/test_train_grpo_full.py` | 🆕 NEW | 真训练 5 step on mock env,验证 loss 下降 |

### 5.2 数据落盘

```
outputs/grpo_v0_ar25_<ts>/
├── checkpoint/                LoRA adapter (PEFT 格式)
├── train_log.jsonl            每 step 一行:loss, reward_mean, kl, lr
├── val_<rollout_idx>.json     每 100 rollout 一份:episode metrics
├── meta.json                  config + git commit + base model id
└── plots/
    ├── loss.png
    ├── reward.png
    ├── kl.png
    └── val_pass_rate.png
```

---

## 6. 评估方法

### 6.1 训练指标(每 100 step / 100 rollout)

| 指标 | 健康范围 | 警报 |
|---|---|---|
| `loss` | 单调下降 | 上升 ≥ 3 step → stop |
| `mean_reward` | 上升 | < 0 持续 → reward 设计错 |
| `kl_to_ref` | < 0.5 | > 1.0 → β 升 |
| `entropy` | 缓慢下降 | 跳水 → 模式坍缩 |
| `n_parse_fail` | < 5% rollout | > 20% → prompt format 不稳 |

### 6.2 Val 指标(每 100 rollout)

| 指标 | baseline (ar25 v3.2) | 目标 (GRPO 训后) |
|---|---|---|
| `pass_rate@10 episode` | 0% (5×500 step) | **≥ 30%** (5×80 step) |
| `mean_episode_reward` | ~ 0.2(只来自 f1 残值)| ≥ 0.5 |
| `mean_change_rate` | 3-4% | **≥ 50%** |
| `mean_unique_states` | ~ 20 | ≥ 60 |

### 6.3 决策门 G

| 门 | 条件 | 失败应对 |
|---|---|---|
| **G1 (training works)** | loss 在前 200 step 下降 ≥ 30%;mean_reward 上升 | 数据 / hyperparam 重调 |
| **G2 (overfit possible)** | 训完 val pass_rate ≥ 30% on ar25 | 模型容量天花板,GRPO 路线终结 |
| **G3 (transfer hint)** | LoRA 在 ar25 上 ≥ 30% 后,在 **bp35**(未训过的)上 val pass_rate ≥ 5% | 真过拟合(还需更多 trick) |

**G2 是核心问号**。如果 ar25 单游戏过拟合都失败,LLM 路线对这个 benchmark 基本无解,得换 Symbolica 风格大模型或者 StochasticGoose 风格 CNN+RL。

---

## 7. 可视化报告

最终产出 `outputs/reports/grpo_v0_ar25.md`,含:

1. **train_curves.png**:loss / reward / kl / entropy 时间序列(4 子图)
2. **val_metrics.png**:pass_rate / change_rate / unique_states 随 rollout 变化(3 子图)
3. **action_distribution_evolution.png**:训前 / 训中 / 训后,ar25 episode 的 action 分布柱状图(看模式坍缩)
4. **reward_decomposition.png**:reward 5 个组件(win / f1 / parse / illegal / entity)各自贡献占比
5. **a_b_comparison.png**:baseline (no LoRA) vs trained 的 5 个游戏(只 ar25 训过)pass_rate
6. **failure_episodes.gif** ×3:训后 3 个失败 episode 的 GIF 拼接

---

## 8. 风险

| 风险 | 缓解 |
|---|---|
| GRPO trainer 在 trl 5.x 上接口不一样 | `tests/test_train_grpo_full.py` 在 mock env 上先验证 |
| ar25 SDK rate limit 拖慢 rollout | 本地 game 文件 SDK 不走网络(已确认 ar25 在 `environment_files/ar25/`)|
| LoRA 训歪让 base 通用能力崩 | KL 监控 + 训前 / 训后跑 mmlu_mini + gsm8k_mini(已有 regression suite) |
| `predicted_changes` JSON 解析失败率高 → reward = -0.5 卡住 | 前 100 rollout 监控 parse_rate;< 50% → revisit prompt 格式 |
| 单游戏过拟合后 prompt 进了 ar25 特有词汇,失掉通用性 | LoRA 不进 Kaggle 提交,这事不是问题 |
| 16 小时训练太长 | 用 Task Scheduler 跑后台(`scripts/run_scheduled.ps1` 模式) |

---

## 9. 实施顺序

| # | 步骤 | 时间 | 出口 |
|---|---|---|---|
| 1 | 写 `rollout_wrapper.py` + `prompts_v3_grpo.py` | 1d | unit test pass |
| 2 | 扩 `train_grpo.py:build_trainer` 完整 + `make_rollout_fn` | 1d | mock env 上跑 10 step 不崩 |
| 3 | 写 `run_grpo.py` 主循环(load env / step / collect group / update LoRA) | 0.5d | dry-run pass |
| 4 | 跑 100 rollout smoke,看 loss 是否下降(G1)| 1.5h GPU | G1 |
| 5 | 跑 1000 rollout full,看 val pass_rate (G2)| 16h GPU,后台 | G2 |
| 6 | 写 `plot_grpo.py` + 出报告(`outputs/reports/grpo_v0_ar25.md`)| 0.5d | 报告含 6 张图 |
| 7 | (若 G2 过)在 bp35/cd82 上 eval LoRA(G3)| 1h GPU | G3 |

**总预算**:实施 ~3 工作日 + 训练 ~16-20h(后台)。

---

## 10. 跟其它路线的关系

```
mask (live)                  ←  必须先稳定(基线)
predictor v0 (arch_predictor) ←  独立 add-on,不影响 GRPO
GRPO v0 (本文档)              ←  ar25 单游戏诊断;决定 LLM 路线还有没有 ceiling
   ↓ G2 过
GRPO v1                       ←  bp35 / cd82 上各训一个 LoRA(per-game adapter pool?)
   ↓
Tier 1 SFT F3 (parked)       ←  正交,改 LLM spatial reasoning 能力
   ↓ 全失败
切换路线                       ←  Symbolica orchestrator(用 Claude API)或 StochasticGoose CNN+RL
```

GRPO v0 的**唯一目标是诊断**:它的成功 / 失败信号决定接下来投资哪个路线。

---

## 11. Phase 0 立刻可以做的事(不依赖任何决策)

1. ⬜ 写 `rollout_wrapper.py`(把 v3.2 ActionAgent 拆成 generate + compute_logprobs 给 trl 用)
2. ⬜ 改 `prompts_v3_grpo.py`(强制输出 JSON 含 `predicted_changes`)
3. ⬜ 在 mock env 上跑 GRPO trainer 10 step,验证 loss 下降
4. ⬜ 写 `plot_grpo.py`(从 `train_log.jsonl` + `val_*.json` 出 4 张图)

完成 1-4 才能跑训练。整套实施工程量约 3 日。

---

*文档历史:*
- *2026-05-16 初稿:确立单游戏诊断定位 + 6 张可视化 + 三档决策门*
