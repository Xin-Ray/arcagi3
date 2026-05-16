# data.md — 数据资源说明

> 项目用到的所有数据源、内容、用法。每加一个新数据源就在这里加一节。

最近更新: 2026-05-16

---

## 1. ARC-AGI-3 demo games(主数据)

**来源**:`arc.make(game_id)` SDK,真实环境(本地缓存到 `environment_files/<game>/<hash>/`)。

**内容**:25 个公开 demo 游戏。划分冻结在 `data/splits/demo_555.json`(2026-05-11 commit,**不允许重新随机**):
- **G_base** (5 个):baseline / mask 实验用 → `ar25, bp35, cd82, cn04, dc22`
- **G_train** (5 个):未来 GRPO 训练用
- **G_val** (5 个):评估用

**用法**:
- `scripts/eval.py --agent random --games ar25,bp35 --episodes 5` —— 跑随机基线
- `scripts/run_v3_multi_round.py --game ar25 --rounds 3 --max-actions 200` —— v3.2 主 runner

**关键坑**:
- Anonymous 模式 401。`.env` 必须有真 `ARC_API_KEY`(从 https://arcprize.org/api-keys)
- 终评数据集(110 game)Kaggle 离线 evaluate,不可访问

---

## 2. trace 数据(实验产物)

**来源**:每次跑 `run_v3_multi_round.py` 或 `eval.py` 都生成。

**位置**:`outputs/<tag>_<YYYYMMDD-HHMMSS>/round_<k>/trace.jsonl` (v3.2) 或 `outputs/<run>/<game>/trace.jsonl` (v3)。

**schema**(v3.2,2026-05-16):

```json
{
  "step": 12,
  "action": "ACTION1",
  "action_coords": null,
  "reasoning": "(LLM 的一句话推理)",
  "orch_override": "untried ACTION5 over masked ACTION6",
  "frame_changed": true,
  "primary_direction": "UP",
  "primary_distance": 3,
  "matches_reasoning": "YES",
  "current_alert_active": "",
  "reflection_delta": {...},
  "no_op_streak": 0,
  "state_revisit_count": 1
}
```

**v3 旧 schema**(`outputs/v3_p0b_p1_full/<game>/trace.jsonl`):用 `chosen_action` + `real_diff`(cell list)替代 `action` + `frame_changed`。`arc_agent/predictor/dataset.py` 两种 schema 都吃。

**用法**:
- 训练 frame-change predictor:`arc_agent/predictor/dataset.py:scan_traces`
- 平衡分析:`scripts/analyze_trace_balance.py` → `outputs/reports/trace_balance.md`
- 报告:`scripts/build_mask_revive_report.py` 等

**当前累积**:73 个 trace.jsonl,6132 step row,5 个 game,42% overall change_rate(2026-05-16 实测)。详情见 [`project/predictor_v0/report_trace_balance.md`](../project/predictor_v0/report_trace_balance.md)。

---

## 3. Tier 1 SFT 合成数据(parked)

**来源**:`arc_agent/finetune/synth_tier1.py` 合成生成,不依赖 ARC 数据。

**位置**:`outputs/finetune/tier1_train.jsonl` (190K rows) + `tier1_holdout.jsonl` + `tier1_ood.jsonl`。

**内容**:5 个空间推理任务的合成样本:
- T1: 方向(UP/DOWN 二选一)
- T2: 整数除法 + 计数
- T3: 8 方向相对位置 → 移动方向
- T4: 格式遵守
- T8: reasoning 跟 action 一致性

**用法**:
- 生成:`scripts/gen_tier1_data.py`(种子 42 可复现)
- 训练:`scripts/train_tier1_sft.py`(LoRA + 4-bit)
- 评测:`scripts/eval_tier1.py` → `outputs/finetune/tier1_eval_report.md`

**当前状态**:第一轮训练 FAIL,F3 修法待跑。详情 [`project/sft_tier1/architecture.md`](../project/sft_tier1/architecture.md) §12 复盘。

---

## 4. Regression mini benchmark

**来源**:从公开 benchmark 抽样,用作 LoRA 训练前后的通用能力监测。

**位置**:`data/regression/{mmlu_mini, gsm8k_mini, zh_qa_mini}.jsonl`(已 commit)。

**用途**:Tier 1 SFT 前后跑 → 看 LoRA 有没有把通用能力训坏(实测 gsm8k 退化 14.7 pp 是 FAIL 信号)。

---

## 5. 视觉数据(step PNG / play GIF)

**来源**:`arc_agent/viz_v3_2.py:compose_step_image_v32` 渲染。

**位置**:实验 dir 下 `round_<k>/step_<n>.png` + `play.gif`。

**用途**:
- 人工 debug
- Predictor CNN 训练:`arc_agent/predictor/png_decoder.py:decode_grid` 反解 64×64 grid

**注意**:viz_v3_2 的 PNG 是 composite(grid 256×256 + 文字 panel)。CNN 解码只取左上 grid 区域。

---

## 6. Knowledge / OutcomeLog(运行时数据,不持久化)

跑实验时在内存里维护:

- `Knowledge`:跨 round 累积(`action_semantics / goal_hypothesis / rules / failed_strategies / click_targets / ...`)。落地到 `knowledge_history.jsonl` + 每 step `knowledge_per_step.jsonl`
- `OutcomeLog`:per-round。每次 `env.step` append 一个 `StepOutcome`,mask 用它判断 `n_tried` / `n_changed`

---

## 7. 不在本项目里但相关的数据

- ARC-AGI-3 全私评数据集:**不可访问**(Kaggle 终评才用)
- ARC-AGI-1/2 数据:跟本竞赛不直接相关
- 任何「task-specific 优化数据」:**违规**(竞赛规则禁止)

---

*历史:*
- *2026-05-16 初稿,跟 docs reorg 一起加入*
