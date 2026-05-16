# ARCHITECTURE v3 —— 探索型 ARC-AGI-3 Agent

日期: 2026-05-14
状态: 设计 + 实施(本文是 single source of truth,旧的 v2 红字 / arch_agents_v2 都被取代)
前置阅读: `ref_object_pipeline_zh.md`(scipy 评测结果)

---

## 0. 一段话设计目标

让 agent 在面对**完全陌生**的 grid 游戏时:

1. 输出**尽可能多样**的动作(高 action_entropy,杜绝 ACTION1-spam)
2. **统计推断**每个 ACTION 在该游戏中的实际效果
3. **统计推断**哪些对象是 active(可被操作)、哪些是 static(背景/纹理)
4. **持续修正** goal hypothesis,直到通关

核心原则:**视觉感知 = 确定性算法,推理 + 决策 = 文本 LLM,二者中间用结构化数据连接,LLM 不看图**。

---

## 1. 概念表(快速查阅)

| 概念 | 含义 | 实现 |
|---|---|---|
| **STATIC** | 多帧位置 + 形状 + 颜色完全不变的对象 | `temporal_classifier` |
| **ACTIVE** | 至少 1 次发生过移动/变形/变色的对象 | 同上 |
| **CANDIDATE** | 观察帧数 < N(默认 3),还不够分类 | 同上 |
| **TEXTURE** | 大量同色同形的小对象(默认 size≤2, count≥20) | `is_likely_texture` |
| **ObjectMemory** | per-episode 的 active 对象历史(UID 持久化跨帧) | `object_tracker.ObjectMemory` |
| **ActionOutcome** | 一次 (state_hash, action) → 帧变化的统计记录 | `action_inference.OutcomeLog` |
| **LearnedActionMap** | 每个 ACTION 的人话效果描述,例如 `"ACTION1: moves objects UP by 3 cells"` | `action_inference.summarize_action` |
| **GoalHypothesis** | 一句话假设 + confidence,每 K 步 Qwen 更新 | 由 Reflection 维护(或暂时静态)|
| **enriched prompt** | 把方向/距离/topmost/largest 等聚合**预先计算**进 prompt | `prompts_v3.py:build_play_prompt` |
| **action_entropy** | 一个 episode 内 chosen_action 分布的香农熵(对 7 个 action 均匀 ≈ 1.946) | `report_v3.compute_entropy` |
| **state_coverage** | 一个 episode 内 unique frame_hash 数量 | `report_v3.unique_frames` |
| **untried_actions** | 该 episode 内还没尝试过的合法 action 集合 | `OutcomeLog.untried(legal)` |
| **diversification penalty** | Qwen 输出跟最近 3 步相同时,在 prompt 里强制要它选不同的 | `text_agent._force_diversify` |
| **R0/A1/A2/A3** | 旧 ablation 的 6 agent,只作为对比基线 | `outputs/ablation_overnight_/` |
| **TextAgent (v3)** | 本架构的主 agent —— text-only Qwen + 全套 enrich | `arc_agent/agents/text_agent.py` |

---

## 2. 数据流(每一步发生什么)

```
┌───────────────────────────────────────────────────────────────────────┐
│  env.step(prev_action) → FrameDataRaw                                  │
│                          ↓                                              │
│   ┌──────────────  PERCEPTION (deterministic, no LLM)  ──────────────┐ │
│   │                                                                  │ │
│   │  grid (64x64)                                                    │ │
│   │     ↓                                                            │ │
│   │  extract_objects(grid)   → list[ObjectRecord]  (~ms)             │ │
│   │     ↓                                                            │ │
│   │  is_likely_texture()     → tag candidates that look like texture │ │
│   │     ↓                                                            │ │
│   │  temporal_classifier()   → STATIC / ACTIVE / CANDIDATE per obj   │ │
│   │     ↓                                                            │ │
│   │  align_objects(prev_active, current_active)  Hungarian + cost    │ │
│   │     ↓                                                            │ │
│   │  matches with direction/distance/description labels (enriched)   │ │
│   └─────────────────────────────────────────────────────────────────┘ │
│                          ↓                                              │
│   ┌──────────────  MEMORY (per-episode state)  ─────────────────────┐ │
│   │   ObjectMemory.update(matches)        # UID-keyed history       │ │
│   │   OutcomeLog.record(prev_action, frame_changed, n_active, ...)  │ │
│   │   frame_hashes.add(hash(grid))                                  │ │
│   └─────────────────────────────────────────────────────────────────┘ │
│                          ↓                                              │
│   ┌──────────────  PROMPT BUILDER  ─────────────────────────────────┐ │
│   │   [STATUS]    step/level/legal/budget                            │ │
│   │   [ACTIVE]    only the ACTIVE objects + UID-keyed history        │ │
│   │   [TEXTURE]   "+ N static texture cells, treated as background"  │ │
│   │   [ACTION]    per-action observed effects (LearnedActionMap)     │ │
│   │   [UNTRIED]   explicit list of untried legal actions             │ │
│   │   [HISTORY]   last 5 (action, frame_changed) tuples              │ │
│   │   [GOAL]      current hypothesis (or "unknown — exploring")      │ │
│   │   [ASK]       "pick next action. prefer untried."                │ │
│   └─────────────────────────────────────────────────────────────────┘ │
│                          ↓                                              │
│   ┌──────────────  REASONER  ───────────────────────────────────────┐ │
│   │   Qwen2.5-VL-3B in text-only mode (no image)                    │ │
│   │   max_new_tokens=24,greedy                                       │ │
│   │   output: ACTION token + optional 1-line reasoning               │ │
│   └─────────────────────────────────────────────────────────────────┘ │
│                          ↓                                              │
│   ┌──────────────  ACTION POSTPROCESS (anti-collapse)  ──────────────┐ │
│   │   if (action == last_3_actions).all():                          │ │
│   │       force a different action from untried set                  │ │
│   │   else: pass-through                                             │ │
│   └─────────────────────────────────────────────────────────────────┘ │
│                          ↓                                              │
│   return action                                                         │
└───────────────────────────────────────────────────────────────────────┘
```

可选的**周期性 Reflection 调用**(每 K=10 步):

```
build_reflection_prompt(...) → Qwen text-only → 更新 GoalHypothesis
```

Reflection **不参与每步循环**,所以即使它失败,Play loop 也能跑下去。

---

## 3. 三条硬约束(传承 v2 红字)

| # | 约束 | 实现保障 |
|---|---|---|
| 1 | **System prompt 中性** —— 不预告 action 语义 | `prompts_v3.PLAY_SYSTEM` 不含 "ACTION1=up" |
| 2 | **跨 agent 同语种 + 同 tone** —— 全英文 | 静态常量 |
| 3 | **动态信息只入 user prompt** —— 不渗 system | system 是常量 |

加 v3 新增的**第 4 条**:

| 4 | **所有聚合/排序/方向换算 pre-compute** —— LLM 只做决策不做扫描 | 见 §4 |

---

## 4. enriched prompt 规范(LLM 不算的事都替它算好)

| 模型很弱的事 | v3 提前算好 | 字段位置 |
|---|---|---|
| dy/dx → "up/down/left/right" | `Match.direction` | `[ACTIVE]` 块 |
| 扫列表找最小 row | `top_active_id` | `[STATUS]` 块 |
| 扫列表找最大 size | `largest_active_id` | `[STATUS]` 块 |
| 统计 ACTION 历次效果 | `LearnedActionMap[ACTION1]` | `[ACTION]` 块 |
| 区分 active vs 背景 | `temporal_classifier` 已经分好 | 只 `[ACTIVE]` 列出 |
| 计算"还没试过哪些 action" | `OutcomeLog.untried()` | `[UNTRIED]` 块 |
| 检测 "上一步 frame 变了吗" | scipy diff 即得 | `[HISTORY]` 块 |

---

## 5. 模块和文件清单

| 模块 | 文件 | 行数估 | 责任 |
|---|---|---|---|
| 视觉感知层 | `arc_agent/object_extractor.py` ✅ | ~180 | scipy extract |
| 视觉感知层 | `arc_agent/object_aligner.py` ✅ | ~190 | Hungarian align |
| 时间分类层 | `arc_agent/temporal_classifier.py` 🆕 | ~150 | STATIC/ACTIVE/TEXTURE |
| UID 持久化 | `arc_agent/object_tracker.py` 🆕 | ~120 | 跨帧 UID + ObjectMemory |
| 动作归因 | `arc_agent/action_inference.py` 🆕 | ~120 | OutcomeLog + summarize_action |
| Prompt 构造 | `arc_agent/prompts_v3.py` 🆕 | ~150 | 4 段 prompt builder |
| Agent | `arc_agent/agents/text_agent.py` 🆕 | ~180 | TextAgent (text-only Qwen) |
| 单元测试 | `tests/test_temporal_classifier.py` 等 4 个 🆕 | ~400 | 覆盖每个新模块 |
| 评估脚本 | `scripts/run_v3_eval.py` 🆕 | ~150 | 跑 G_base × 80,产报告 |
| 报告生成 | 扩展 `arc_agent/report.py` | +50 | action_entropy 列 |

新增 ~1500 行代码 + 400 行测试。预估 2 工作日。

---

## 6. 评估方法(本文重点 —— 用什么数据回答"v3 是否比 v1 好")

### 6.1 跑什么

- 同 5 个 G_base 游戏(ar25/bp35/cd82/cn04/dc22)
- 每个游戏 1 episode, 80 actions max
- seed=0, greedy decoding
- max_new_tokens=24(只输出 action token + 短 reasoning)

### 6.2 必报指标(写进 `summary.json`)

| 指标 | 类别 | 定义 | v3 目标 |
|---|---|---|---|
| **action_entropy** | 必报 | 一 episode 内 chosen_action 频率的香农熵(对 7 action 均匀 = 1.946) | **≥ 1.5**(显著优于 ablation 平均的 ~0.5)|
| **state_coverage** | 必报 | unique frame_hash 数量 | **比 random 至少 +50%** |
| **n_action_pairs_explored** | 必报 | unique (frame_hash, action) 数量 | **比 random 至少 +50%** |
| **levels_completed** | 必报 | SDK 返回 | **任何 >0 都是巨大胜利**(ablation 全 0)|
| **RHAE (Kaggle score)** | 必报 | SDK scorecard | 同上 |
| **mean_F1_gated** | 诊断 | 仅当 v3 仍维护 predicted_diff 时;若不维护标 n/a | 不作硬目标 |
| **wall_clock_per_step** | 必报 | env.step → action 的总耗时均值 | **< 2 s**(满足 Kaggle 10h)|
| **no_op_rate** | 必报 | 当步 `frame_changed = False` 的比例 | **< 30%**(比 ablation 的 75-100% 大幅低)|
| **untried_at_end** | 诊断 | episode 结束时仍未尝试的 action 数 | **0**(理论上 80 步够把 7 个 action 都试过)|

### 6.3 对比 baseline

并排比下面这些:

| Agent | 来源 | 备注 |
|---|---|---|
| **R0 (random)** | 已跑(`ablation_overnight_/random/`) | 探索的"地板" |
| **A1 lite (旧 prompt)** | 已跑(`ablation_overnight_/lite/`) | 看 prompt 改进是否有效 |
| **A2 full (旧 prompt)** | 已跑 | 看 scaffold 是否反而拖累 |
| **v3 TextAgent** | 新跑 | 主角 |

### 6.4 评估流程(可重复脚本化)

```bash
.venv/Scripts/python.exe scripts/run_v3_eval.py \
    --games ar25,bp35,cd82,cn04,dc22 \
    --max-actions 80 \
    --output outputs/v3_eval_<ts>
# 产出 outputs/v3_eval_<ts>/{<game>/trace.jsonl, summary.json}

.venv/Scripts/python.exe scripts/report_v3.py \
    --new outputs/v3_eval_<ts> \
    --baselines outputs/ablation_overnight_/random,outputs/ablation_overnight_/lite,outputs/ablation_overnight_/full
# 产出 outputs/v3_eval_<ts>/comparison.md(并排表 + 关键诊断)
```

### 6.5 决策门

| 第几步后 | 门 | 通过 → 下一步 | 失败 → 应对 |
|---|---|---|---|
| smoke test(ar25 单游戏 20 步) | `action_entropy ≥ 1.3` 且无 crash | 进 G_base 全跑 | 修 prompt;若仍 collapse,prompt 加 epsilon-greedy 强制 |
| G_base 全跑完 | `mean(action_entropy) ≥ 1.5` 且 `n_action_pairs_explored ≥ R0` | 任意游戏 `levels_completed > 0` 即成功 | 检查 Reflection 是否真的产出有用的 LearnedActionMap |
| 任意游戏过关 | `levels_completed > 0` | 进 G_train 验证泛化 | 看 trace 找瓶颈 |

---

## 7. SmokeTest(最小可验证场景)

跑前先确认基础 sanity:

```bash
.venv/Scripts/python.exe scripts/run_v3_eval.py \
    --games ar25 --max-actions 20 \
    --output outputs/v3_smoke
```

期望:
- 20 步全部 parse 成功(text 输出 ACTION 字符)
- action_entropy ≥ 1.0(至少试过 3+ 种 action)
- 没有 1 个 action 占 > 50% 的步数
- wall_clock_per_step < 2s

如果 smoke 通过,再跑全 G_base。

---

## 8. 这个架构**不**承诺的事

| 不承诺 | 理由 |
|---|---|
| 跨 episode 记忆 | 故意每 episode reset,避免污染 |
| YOLO / 大模型替换 | v3 已经把视觉做完,这里没有 YOLO 的位置 |
| 多步前瞻 / MCTS | 单步 reactive 先做出非零 RHAE,再考虑 |
| 跨游戏先验 | 每个游戏都从 zero exploration 开始 |
| 自定义 reward shaping | RHAE 和 levels_completed 已有,不再发明 |

---

## 9. 实施顺序(我接下来要做的事)

| 阶段 | 任务 | 文件 | 用时 |
|---|---|---|---|
| **W1** | temporal_classifier + 测试 | `temporal_classifier.py`, `tests/test_temporal_classifier.py` | 半天 |
| **W2** | object_tracker (UID 持久化) + 测试 | `object_tracker.py`, `tests/test_object_tracker.py` | 半天 |
| **W3** | action_inference + 测试 | `action_inference.py`, `tests/test_action_inference.py` | 半天 |
| **W4** | prompts_v3 + 测试 | `prompts_v3.py`, `tests/test_prompts_v3.py` | 半天 |
| **W5** | TextAgent + smoke test on ar25 (20 步) | `text_agent.py` | 半天 |
| **W6** | run_v3_eval + report_v3 + 跑全 G_base | `scripts/...` | 半天 + ~30 min GPU |
| **W7** | 写 comparison.md + 决策门评估 | (报告)| 半天 |

总:**~4 工作日 + ~30 min GPU 验证**(纯文本推理快)。

每完成一个模块跑一次现有单元测试,确保没破坏旧代码(253 个测试 baseline 必须保持绿)。

---

*本文档(v3)是当前的设计基准。所有之前的 v2 红字、arch_agents_v2_zh.md 都被本文取代;v2 保留作历史。*
