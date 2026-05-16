# ARCHITECTURE — Frame-Change Predictor (v0)

日期: 2026-05-16
状态: 设计 → 试点(Phase 0 / Phase 1 验证后决定是否继续)
前置阅读: [`architecture/v3_2_zh.md`](./v3_2_zh.md), [`reference/v3_2_hardrules_results_zh.md`](../reference/v3_2_hardrules_results_zh.md)

---

## 0. 为什么做这个

ARC Prize 2026 preview 阶段第一名 StochasticGoose (Tufa Labs) 用的就是「**CNN 预测哪个 action 会让 frame 改变**」+ 简单 RL,preview 拿了 12.58%。最新 ARC 官方 30 天复盘明确把「Smart Exploration / learned action prediction」列为 **#1 actionable advice for agent designers**。

我们 v3.2 当前 (`outputs/preload_v3_budget_smoke_20260515-004233`) change_rate **只有 3-4%**(2 round × 500 step,0 levels won),95%+ 的 LLM 决策结果是 no-op。**Knowledge / Reflection / click_targets 没把这个数字拉起来,因为决策模型本身(Qwen2.5-VL-3B)做不到对「某个 state 下哪个 action 会有效」的可靠判断**。

本文档定义一个**辅助 predictor**:对每个 (state, action) 打一个 P(frame_change) 概率,把高概率 action 通过 prompt 推荐给 LLM(或在 mask off 时直接做选 action 的指导)。

---

## 1. 范围

### 1.1 In-scope

- 训练一个小模型(参数 < 1 M)估算 `P(frame_change=True | state, action) ∈ [0, 1]`
- 三种架构对照:Logistic Regression (hand features) / MLP / shallow CNN
- 训练数据**自监督**,标签 = `frame_changed`(已存在于 trace.jsonl)
- 在 `prompts_v3_2.py` 加 `[ACTION FORECAST]` 块,把 P 排序后送给 Action Agent
- A/B 在 ar25 + 4 个 demo game 上对照 v3.2 mask-on baseline

### 1.2 Out-of-scope

- 训练大模型 / fine-tune Qwen — 本文档**不替换** Action Agent,只是 prompt 加块
- 多模态(图像 + text)— predictor 输入是 perception 已抽出的结构化特征 + 离散 action,**不吃 raw RGB**
- 跨游戏迁移学习 — 第一版**每个 game 独立训练**(参考 StochasticGoose 的 per-level retrain 设计),先证明可行
- 在线学习(rollout 时增量更新) — 留待 v0.1

### 1.3 规则合规

| 选择 | 合规理由 |
|---|---|
| 用**自监督** trace data 训练 | 不是 task-specific optimization;预测「frame 会不会变」是通用的环境动力学 |
| 不用 demo / private game label | 没有 ground-truth label,纯 self-supervised |
| Per-game retrain 在 Kaggle 上 | 5-10 min per game 训练时间在 10h 总预算内 |
| 不调 Qwen 权重 | 跟 Tier 1 SFT 独立,不冲突 |

---

## 2. 数据现状(2026-05-16 实测,来自 `outputs/reports/trace_balance.md`)

```
trace files scanned: 73
total step rows:    6132
overall change_rate: 42.4%   (2599 changed / 3533 no-op)
```

**Per-game change_rate 范围**(5 个公开 demo game):

| game | rows (approx) | change_rate | 备注 |
|---|---:|---:|---|
| ar25 | ~1500 | 13-86% | v3.2 实测主战场;mask on 时 60-100%,off 时 < 20% |
| bp35 | ~560 | ~70% | RandomAgent baseline |
| cd82 | ~560 | ~64% | RandomAgent baseline |
| cn04 | ~560 | ~89% | RandomAgent baseline |
| dc22 | ~560 | ~62% | RandomAgent baseline |

**Per-action balanced sample availability**(min(n_changed, n_no_op)):

| action | total | n_changed | n_no_op | max balanced | 状态 |
|---|---:|---:|---:|---:|---|
| ACTION1 | 2070 | 529 | 1541 | **529** | 充足 |
| ACTION3 | 543 | 182 | 361 | **182** | 够用 |
| ACTION4 | 742 | 587 | 155 | **155** | 够用 |
| ACTION5 | 705 | 629 | 76 | 76 | 边缘 |
| ACTION6 | 1456 | 255 | 1201 | **255** | 够用 |
| ACTION7 | 389 | 192 | 197 | **192** | 够用 |
| ACTION2 | 205 | 203 | 2 | 2 | 退化(几乎永远 changed)|
| RESET | 22 | 22 | 0 | 0 | 退化 |

**结论**:对 ACTION1/3/4/6/7 在 5 个游戏上有 **1389 个 balanced 样本对**(扔掉 ACTION2/RESET 的退化情况)。够训一个 < 1 M 参数的小模型。

### 2.1 数据偏差 / 用户提的「失败数据为主」问题

用户原话:「现在的trace基本都是失败数据,我觉得你得另外想写办法」

**响应**:已确认过, 假设错误。实测是 42% changed / 58% no-op,实际上比平衡数据(50/50)只偏 8 pp,**远好于 LLM 视角下的 3-4% change_rate**。原因:

- 6132 步中只有 1715 步是 v3.2 LLM agent 跑的(那部分确实 5-20% change_rate)
- **2400 步**来自 `ablation_overnight_` 的 RandomAgent + 其它 baseline agent,这些 random 选 action 时,bp35/cd82/cn04/dc22 几乎所有 action 都让 frame 变(那些游戏更"宽容"); ar25 random 时也能拿到 ACTION5=100%(151 次都 changed)
- 另外 v3_p0b_p1_full 也是 53% change_rate

所以**数据足够,问题不在数量**。

但**ar25 单游戏**的 ACTION6 仍然是 0% changed(0/466),这是真的「单游戏对单 action 完全没有正例」。Predictor 在 ar25 上预测 ACTION6 → 总是 P(change)=0;在 cn04/cd82 上 → 50-70%。**这正是我们想要的**:让 predictor 学会 game-conditional 行为。

### 2.2 数据补充策略(如果发现某些 (game, action) 类太少)

第一版**不补充**,直接训。若 Phase 1 时发现某个 (game, action) 类的 n < 30 导致预测 nan,再加:

1. **RandomAgent 一夜跑** on 25 demo games × 200 step = 5000 fresh 样本,无 LLM 调用,~ 30 min wall clock(纯 SDK + 决策时间)
2. **Lite agent (vlm_lite)** 跑同样配置 = 5000 + 5000 = 10000 样本

这两个 fallback 已经做过(`outputs/ablation_overnight_/`)样本就来自这里。要扩量执行 `scripts/eval.py --agent random --games '' --episodes 5 --max-actions 200` 即可。

---

## 3. 特征工程

### 3.1 State 是什么(每行 trace 都需要)

trace 当前**没有完整 grid**(只有 action / frame_changed / direction)。两个补法:

**A. 从 step PNG 反解码**(快但脏):
- `viz_v3_2` 把 64×64 grid 用 4× upscale 渲染成 256×256
- 写 `arc_agent/png_decoder.py:decode_grid(png_path) -> np.ndarray(64, 64)`
- ARC 调色板已经在 `observation.py:_ARC_PALETTE`,反查表即可
- 风险:渲染时加了 header / panel,grid 区不一定在 (0,0)-(255,255);写一个 fixture 验证

**B. 重新跑数据采集,trace 加 `grid_before_hex`(慢但干净):
- 改 `scripts/run_v3_multi_round.py` 在 trace 行加 `"grid_before": grid.tolist()`(每行 ~16 KB,4 KB 压缩后)
- 跑 5 game × 200 step = 1000 step × ~16 KB = 16 MB,可接受
- 用 `scripts/eval.py --agent random` 在 25 game × 200 step 也跑一遍

**推荐 A 做第一版**(0 编码改动,直接 reuse 现有 73 个 trace + PNG),Phase 1 之后再考虑 B。

### 3.2 候选特征集

按复杂度从低到高:

**F1 (hand features, ~32 维):**
- Action one-hot (7 维)
- Action6 coords normalized: `x/64, y/64`(2 维,非 ACTION6 时 = 0)
- Object stats(从 scipy perception 重算或 trace 已有):
  - `n_active_objects`(连通分量数)
  - `n_static_objects`
  - `most_common_color_id` (one-hot 16 维)
  - `last_3_frame_changed` (3 维)
- Optional `no_op_streak`, `state_revisit`(若 trace 有)
- Game id one-hot(5 维 for demo games,Kaggle 时 110 维)

**F2 (raw grid, 4096 维):**
- Flatten 64×64 grid → 4096 channels(各 cell 的 16 色,one-hot 是 65536 维太大;用 16-dim embedding)
- 训 CNN 时用 `(B, 16, 64, 64)` shape

### 3.3 Label

`y = 1 if frame_changed else 0`。

---

## 4. 候选架构

| 架构 | 输入 | 参数 | 训练时长 | 预期 AUC | 推理时长 |
|---|---|---:|---|---|---|
| **M1 LogReg** | F1 hand features | ~250 | < 10 s CPU | 0.65-0.80 | < 1 ms |
| **M2 MLP-128** | F1 hand features | ~5 K | 30 s CPU / 5 s GPU | 0.75-0.85 | < 1 ms |
| **M3 MLP-512** | F1 hand features | ~30 K | 60 s GPU | 0.80-0.88 | < 1 ms |
| **M4 CNN-small** | raw grid 16ch×64×64 | ~80 K | 3-5 min GPU | 0.82-0.90 | 5 ms |
| **M5 CNN-deep** | 同 M4 | ~300 K | 8-12 min GPU | 0.85-0.92 | 10 ms |

**M5 上限可能不会比 M3 高很多**,因为 hand features 已经把 perception 的核心信号(object 数量、位置)总结了。**M1/M2 先跑,看是否需要 M4/M5**。

### 4.1 LogReg / MLP 实现细节

- sklearn `LogisticRegression(class_weight='balanced', max_iter=200)`
- PyTorch MLP: `Linear(32, 128) → ReLU → Dropout(0.2) → Linear(128, 1) → Sigmoid`
- 损失: `BCEWithLogitsLoss(pos_weight=...)` 反映 class imbalance
- 优化: Adam lr=1e-3, batch=256, epochs=30, early stop on val loss

### 4.2 CNN 实现细节

- 输入 `(B, 17, 64, 64)`:16 颜色 channel + 1 个 action one-hot 广播到 64×64
- Block: `Conv3x3(17→32) → BN → ReLU → MaxPool2x2 → Conv3x3(32→64) → BN → ReLU → MaxPool2x2 → Conv3x3(64→64) → BN → ReLU → GlobalAvgPool → Linear(64→1)`
- 训练同上

---

## 5. 训练 / 验证协议

### 5.1 划分

- **Train**: 4 个 game(ar25/bp35/cd82/cn04) 全部 step + 1 game(dc22) 70% step
- **Val**: dc22 余下 30% step + 1 个 OOD game(从 demo 中随机选未训练过的 — 但当前 5 个 demo game 我们都见过,所以 OOD 是空集,留待 Kaggle 110 game 时再分)

注意:**game id 是特征**,所以 cross-game generalization 是 OOD,not in-distribution。Phase 1 只验证 in-distribution accuracy。

### 5.2 评估指标

| 指标 | 含义 | 目标 |
|---|---|---|
| **ROC-AUC** | val 集全局 AUC | M1 ≥ 0.70, M3 ≥ 0.80, M5 ≥ 0.85 |
| **Per-action AUC** | 7 个 action 各自的 AUC | 每个 ≥ 0.65(若某 action 退化 nan,记录但不强制) |
| **Per-game AUC** | 5 个 game 各自的 AUC | ar25 ≥ 0.75(我们最关心的) |
| **Top-1 action accuracy** | 在每步预测「最有可能 change 的 action」并对照真实选的(若选的是 top-1 且 changed,记 hit)| ≥ baseline + 20pp |
| **Calibration (ECE)** | 预测概率 vs 实际频率的误差 | ECE ≤ 0.10 |

### 5.3 决策门 G

| 门 | 条件 | 失败应对 |
|---|---|---|
| **G1 (信号存在)** | M1 val AUC ≥ 0.65 | < 0.65 → 数据太少 / 特征太差,补 §3.2 fallback,或 stop |
| **G2 (架构有效)** | M3 val AUC > M1 AUC + 0.05 | 否则 hand features 已饱和,不上 CNN |
| **G3 (CNN 值得)** | M5 val AUC > M3 AUC + 0.05 | 否则 CNN 不值,用 M3 |
| **G4 (端到端有效)** | A/B run 上 change_rate 比 mask-on baseline +10pp 以上 | 否则 predictor 没真正改善行为,改 prompt 集成 |

**G1 + G2 任一失败 → 停止 predictor 路线,把 GPU 时间还给 GRPO**。

---

## 6. 集成方式(predictor 怎么影响 Action Agent)

新增 prompt 块 `[ACTION FORECAST]`,在 `prompts_v3_2.py:build_action_user_prompt` 中位于 `[ACTION effects observed]` 之后:

```
[ACTION FORECAST — model-predicted P(frame_change | state, action)]
  ACTION1: 0.82  ←  most likely
  ACTION4: 0.61
  ACTION7: 0.43
  ACTION3: 0.18
  ACTION2: 0.08
  ACTION5: 0.04
  ACTION6: 0.02  ←  least likely
(higher = more confident this action will change the frame in this state.
 Predictions ignore reward; use with goal_hypothesis.)
```

**实施细节**:
- 渲染前调 `predictor.predict_proba(state, legal_actions)` → dict[action, float]
- 按 P 降序排
- 「most likely / least likely」标记加在最高/最低两条
- 不修改 mask 逻辑;mask 仍按 R2 规则走

**ACTION6 特殊处理**(coords-dependent):
- 第一版只对 ACTION6 整体打一个 score(不分坐标),即 `P(change | state, ACTION6)`
- 若 P 高,Action Agent 仍按 `[CLICK TARGETS]` 选坐标
- 真坐标级 prediction 留到 v0.1(需 ACTION6 标签按 (x, y) 分桶,数据稀疏)

---

## 7. 实施顺序

| # | 步骤 | 文件 | 验证 |
|---|---|---|---|
| 1 | 写 dataset builder | `arc_agent/predictor/dataset.py` | unit test:从 3 个示例 trace 抽出 expected sample count |
| 2 | 写特征抽取器 F1 | `arc_agent/predictor/features.py` | unit test:输入 3 种合成 state,输出固定 shape |
| 3 | 写训练 script | `scripts/train_predictor.py` | dry-run on 100 fake samples |
| 4 | 训 M1 (LogReg) → 写 metrics + ROC plot | `outputs/predictor_v0/m1/`  | val AUC ≥ 0.65 → 进 4;否则 stop |
| 5 | 训 M2 + M3 (MLP) | `outputs/predictor_v0/m2/`, `.../m3/` | val AUC vs M1 |
| 6 | 决策门 G2 → 训 M4 + M5(可选) | `.../m4/`, `.../m5/` | G3 |
| 7 | 加 `[ACTION FORECAST]` 块到 prompt + 加载 best model | `arc_agent/prompts_v3_2.py` + `arc_agent/predictor/inference.py` | 1×30 step canary,看 prompt 渲染 OK |
| 8 | A/B run: ar25 3 round × 80 step,mask on,predictor on/off | — | G4 |
| 9 | 写报告 + 4 张图(loss / ROC / per-action AUC / change_rate 对比) | `outputs/reports/predictor_v0.md` | — |

**预算**:Phase 0-4(到 G1 决策点)= ~3 小时(数据 + LogReg + 看 AUC)。
Phase 5-9(若 G1 过) = ~6-8 小时,可并行 GRPO 设计。

---

## 8. 可视化(Phase 4 起每步必做)

| 图 | 内容 | 工具 |
|---|---|---|
| `train_curves.png` | loss / accuracy / ROC-AUC vs epoch,4 个 model 叠加 | matplotlib |
| `roc_curves.png` | 4 个 model 在 val 集的 ROC | sklearn + matplotlib |
| `per_action_auc.png` | 每个 model 在每个 action 上的 AUC 柱状图 | matplotlib |
| `confusion_per_game.png` | best model 在每个 game 的混淆矩阵 | seaborn |
| `change_rate_ab.png` | A/B run 的 change_rate 时间序列(predictor on vs off) | matplotlib |
| `calibration_plot.png` | best model 的 reliability diagram | matplotlib |

所有图 dump 到 `outputs/reports/predictor_v0/`(随报告 commit)。

---

## 9. 已知风险

| 风险 | 概率 | 缓解 |
|---|---|---|
| Hand features 太弱(M1 AUC < 0.65)| 中 | 加 perception-derived 特征(object delta、active obj 数等) |
| ar25 上 ACTION6 全负样本 → predict 永远 0,误伤其它 game 的 ACTION6 | 高 | 加 game id 特征,模型 per-game 学到不同 prior |
| 训练数据集中在 5 个 game,Kaggle 110 game 完全 OOD | 高 | 接受;Kaggle 跑前用 RandomAgent 收集 200 step × 110 game ≈ 22000 sample 现训 |
| LLM 不读 `[ACTION FORECAST]` 块,A/B 看不出差异 | 中 | A/B 不仅看 change_rate,还看 LLM 选 top-1 predicted action 的频率 |
| Predictor 推理 5 ms × 200 step × 5 round = 5 s 增量,可接受 | 低 | 已纳入 Kaggle 预算 |
| 跟 mask 互动:mask 已经禁了 0% 历史 action,predictor 多此一举 | 中 | 把 mask 阈值放宽,让 predictor 接手细化决策 |

---

## 10. 跟其它路线的关系

```
mask (R2 + recovery)        ←  当前活的,负责硬剪枝(0% change_rate 的 action 完全屏蔽)
   ↓
predictor v0 (本文档)         ←  软推荐,把活下来的 action 按 P(change) 排序给 LLM
   ↓
GRPO (arch_grpo_v0_zh)       ←  下一档,用 predictor 的输出当 dense reward 信号(可选)
   ↓
Tier 1 SFT F3 (parked)       ←  跟 predictor 独立,改 LLM 本身能力
```

predictor 是**独立 add-on**,不替换任何现有组件。失败也只是 prompt 多了个无用块,可秒级回滚。

---

## 11. Phase 0 立刻可以做的事

不依赖任何决策:

1. ✅ 数据 balance 报告(`outputs/reports/trace_balance.md`,已完成)
2. ⬜ 写 `arc_agent/predictor/dataset.py` —— 从 trace 抽 (state, action, changed)
3. ⬜ 写 `arc_agent/predictor/features.py` —— F1 hand features
4. ⬜ 写 `scripts/train_predictor.py` —— 训 M1/M2/M3,dump metrics + ROC
5. ⬜ 跑 Phase 4 决策门 G1

完成 5 之后,如果 G1 pass → 继续;不 pass → stop,写报告说明,把 GPU 还给 GRPO。

---

*文档历史:*
- *2026-05-16 初稿(基于 trace_balance.md 实测数据 + 用户「数据是否平衡」质疑的澄清)*
