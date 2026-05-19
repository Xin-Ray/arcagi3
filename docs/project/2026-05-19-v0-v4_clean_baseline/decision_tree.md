# 2026-05-19 v4 — Decision Tree (autonomous overnight)

> **配套**: `architecture.md` 是设计;本文件是 runtime 决策 — Phase 跑完拿到数据后,我会按这个表选下一动作。
> **用户授权**: 2026-05-19 chat "我要准备去睡了，验证要谨慎";"如果验证出了问题，你还是多验证一下"。

## Phase 1: /think vs /no_think (on V4 minimal)

**Setup**: ar25 1 round × 10 step, V4 全砍 flags,各跑一次。

**判定指标** (from action_raw.txt + trace.jsonl):

| 指标 | 算法 |
|---|---|
| `</think>_closed_rate` | % of steps where raw 含 `</think>` |
| `reasoning_nonempty_rate` | trace 里 reasoning ≠ '' 的比例 |
| `action_diversity` | unique actions / 10 |
| `change_rate` | frame_changed / total |

**Decision**:

| Phase 1A (/think) | Phase 1B (/no_think) | Decision |
|---|---|---|
| `</think>_closed_rate ≥ 80%` AND reasoning ≥ 8/10 | (any) | **win**: 用 /think for Phase 2 |
| `</think>_closed_rate ≥ 80%` 但 reasoning < 8/10 | (any) | parsing 还有 bug;debug → 重测;**不进 Phase 2** |
| `</think>_closed_rate < 80%` | reasoning ≥ 8/10 | **win**: 用 /no_think for Phase 2 |
| `</think>_closed_rate < 80%` | reasoning < 8/10 | 两种都坏;**停止 + 报告**,等用户 |

## Phase 2: V4 baseline full run (ar25 1 round × 200 step, budget pooled)

**Setup**: 用 Phase 1 选定 mode,跑 ar25 with `--max-actions-total 200`。

**Decision**:

| change_rate | levels won | parser_triggered | hypothesis_changes | 决策 |
|---:|---:|---:|---:|---|
| ≥ 60% | ≥ 1 | ≥ 1 | ≥ 3 | **大胜** → 跳过 Phase 3 部分,直接 Phase 4 (5-game) |
| ≥ 60% | 0 | ≥ 1 | ≥ 3 | **闭环 work but 不够** → Phase 3 ablation 看哪个旧模块补 |
| ≥ 60% | 0 | 0 | < 3 | 反思机制没真触发 → debug,**不进 Phase 3** |
| < 60% | (any) | (any) | (any) | **退化** → 砍掉的某个模块其实是必要的 → 回 Phase 3 反向看 |

## Phase 3: Ablation (single-module add-back)

按顺序逐个开,跟 V4 baseline 对照。每个跑 ar25 1 round × 100 step。

**顺序** (按已有 evidence 强度):
1. `+click_targets` (`--click-targets on`) — 交叉验证 0/5 命中,预期 **无效或更差**
2. `+action_proposer` (`--propose on`) — smoke 3 ACTION1 bias 根因,预期 **更差**
3. `+action_semantics` (`--action-semantics-from-llm on`) — 同上,known-good prior 制造
4. `+hard_rules` (`--hard-rules on`) — R1/R4-R7 batch,没分别测过

**单 ablation 判定**:

| acc vs V4 baseline | 决策 |
|---|---|
| change_rate +5pp + 或 wins +1 | 标"**有用**",留下用于 Phase 4 |
| 等价 (±5pp) + wins 同 | 标"**中性**",留下可选 |
| change_rate −5pp 或 wins −1 | 标"**有害**",**禁用** |

**最佳组合**: V4 baseline + 所有"有用"模块,作为 Phase 4 配置。

## Phase 4: 5-game eval (条件)

只有 Phase 2 或 Phase 3 任一展现 **≥ 1 win** 才进。Setup: 5 game (ar25 / bp35 / cd82 / cn04 / dc22) × 1 round × 200 step pooled,**串行跑** (不并发 GPU)。

**Decision**:

| 5 game results | 决策 |
|---|---|
| ≥ 2 game 任 1 round 通关 ≥ 1 level | **训练候选** → 早上交给用户决定 fine-tune |
| 1 game 通关 + 其余 change_rate ≥ 60% | **部分胜** → 等用户 |
| 0 通关 + change_rate ≥ 64% (SmolLM3 5×2×300 baseline) | **平 baseline** → 报告 + 等用户 |
| change_rate < 64% | **退化** → 标"V4 minimal 不够,需要旧模块",等用户 |

## Phase 5: 最终汇总

无论结果,生成:
- `docs/project/2026-05-19-v0-v4_clean_baseline/report.md` (总报告)
- 每 phase 子报告
- 跨 phase 对比表
- **建议下一步**: 训练 / 继续 debug / 改模型 / 改架构

## Stop 条件 (autonomous mode 下我会停而非继续)

- Python 异常 in any phase → 停 + 保存 partial + 报告
- GPU OOM → 减 token budget,重 1 次,再 OOM → 停
- Disk full → 停
- /think + /no_think 都坏 (Phase 1 fail) → 停
- Phase 2 退化 (< 50% change_rate) → 停
- 累计 wall > 12 小时 → 停

## 不做的事 (avoid scope creep)

- 不 push 任何分支(用户没明确授权 v4 分支 push)
- 不动 main
- 不删任何文件
- 不重新设计 Action 或 Reflection prompt(超出 V4 baseline 定义)
- 不并发跑(GPU 共享)
- Phase 4 之外不跑 multi-game

## 引用

- `architecture.md` (V4 设计 + flag 一览)
- `ablation_plan.md` (Phase 3 详细 spec)
- 上游 chat: 2026-05-18 (用户提清洗 + ablation 方法),2026-05-19 (用户提 wide schema + 1×200 + step pooling + 5 game one-by-one)
