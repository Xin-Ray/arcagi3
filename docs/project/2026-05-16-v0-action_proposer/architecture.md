# ARCHITECTURE — Action Proposer v0

日期: 2026-05-16
状态: 实施中
前置阅读: [v3_2 architecture](../2026-05-14-v3_2-double_agent/architecture.md), [v2 canary verify (在 v2-canary-verify 分支)](#)

---

## 0. 为什么做这个

[v3_2 mask_revive](../2026-05-14-v3_2-double_agent/report_mask_revive.md) + v2 canary verify(在 v2-canary-verify 分支)实验暴露了 v3.2 当前的核心问题:

| 状态 | change_rate | ACTION1 占比 | Knowledge |
|---|---|---|---|
| main HEAD(R2 mask on) | **5-8%** | **95%** | 满 |
| v2 canary 复现(1bac4be 代码) | 86% | ~18% | 空 |

**LLM 决策被 Knowledge.action_semantics 的 "ACTION1: moves UP" anchor 永久 commit**。这是 Qwen2.5-VL-3B 的弱项("free-form action token generation in open vocabulary")的体现。

Qwen2.5-VL-3B 已知强项(参考公开 benchmark + 我们 Tier 1 实测):
- **多选 / 分类**: 强
- **N 选 1 ranking**: 强(few-shot 例子下尤佳)
- **结构化文本理解 + 摘要**: 中-强

弱项:
- 开放词表 token 生成
- 多步空间规划
- 自由 reasoning chain

**本设计**:把 action 决策从「开放生成」改成「N 选 1」,playing to model's strengths,期望同时获得 v2 canary 的高 change_rate + main 的 Knowledge 累积能力。

---

## 1. 范围

### 1.1 In-scope

- 新模块 `arc_agent/action_proposer.py`:基于代码启发式生成 K=3-5 个候选 action(每个带 reason)
- 改 `arc_agent/prompts_v3_2.py`:Action prompt 加 `[CANDIDATES]` 块,要求 LLM 在 [A] [B] [C] 选 1
- 改 `arc_agent/agents/action_agent.py`:parse "A"/"B"/"C" 而不是 ACTION token
- 改 `scripts/run_v3_multi_round.py`:加 `--propose {on,off}` flag,默认 on
- 跑 ar25 3×30 step canary 对照:`--propose off`(baseline,=main 行为)vs `--propose on`(本设计)

### 1.2 Out-of-scope

- 不改 Knowledge schema(还是 action_semantics + goal_hypothesis 等)
- 不改 Reflection Agent(还是写 Knowledge update)
- 不接 predictor v0 的 P(change|action)信号(留待 v0.1)
- 不改 mask(R2 仍工作,但很少触发因为 propose 已经引导)

---

## 2. 概念表

| 概念 | 一句话 |
|---|---|
| **candidate action** | 代码 propose 的 1 个 (action, reason) 二元组 |
| **proposer strategy** | 生成候选的策略,例如 untried-first / known-good-mix / explore-stuck |
| **N 选 1** | Action Agent prompt 提供 K 候选 [A][B][C],输出 "A" 或 "B" 或 "C" |

---

## 3. 数据流

```
env.step(prev_action) → frame_t
        │
        ▼
perception (scipy + Hungarian) → ObjectMemory
        │
        ▼
action_proposer.propose(latest, knowledge, outcome_log) → [
    Candidate(action=ACTION3, reason="untried this round"),
    Candidate(action=ACTION1, reason="Knowledge: moves UP"),
    Candidate(action=ACTION6 (32, 12), reason="click_target: yellow_1x1 high conf"),
]
        │
        ▼
prompts_v3_2.build_action_user_prompt(
    ...,
    candidates=[A, B, C],   ← NEW
)
        │  prompt 含:
        │   [CANDIDATES]
        │     [A] ACTION3   reason: untried this round
        │     [B] ACTION1   reason: known-good (moves UP)
        │     [C] ACTION6 (32, 12)   reason: click_target yellow_1x1
        │   [ASK]
        │     Reply with a single letter A/B/C, then 1 sentence reasoning.
        │
        ▼
Qwen output: "B. reasonable because ..."
        │
        ▼
action_agent: parse "B" → resolve to candidates[1] = ACTION1
        │
        ▼
env.step(ACTION1)
```

---

## 4. 关键决策

| 决策 | 为什么 |
|---|---|
| K = 3(不是 5 或 7) | 3 张选项是 Qwen-3B 的 sweet spot;5+ 模型注意力分散;1 不算 N 选 |
| 每次 propose 必含 1 个 untried(若有) | 强制保留探索路径 |
| 每次 propose 必含 1 个 known-good(若 Knowledge 有) | 利用学到的语义 |
| 第 3 个候选填 click_target / random / no-op | 多样性占位 |
| **不**改 prompt 的 reasoning 部分 | reasoning 是 Qwen 强项,继续要 |
| 候选 letter 用 A/B/C 而不是 1/2/3 | Qwen 2.5 实测在 letter MC 上更稳 |
| `--propose off` 默认值是 `on` | A/B 时显式关 |

---

## 5. 模块清单

| 文件 | 状态 | 责任 |
|---|---|---|
| `arc_agent/action_proposer.py` | 🆕 NEW | `Candidate` dataclass + `propose(latest, knowledge, outcome_log, recent) -> list[Candidate]` |
| `arc_agent/prompts_v3_2.py` | 🟡 改 | 加 `_render_candidates_block` + 改 `[ASK]` 块文案 |
| `arc_agent/agents/action_agent.py` | 🟡 改 | `_parse_multichoice` + `attach_candidates`;原 free-text parse 留 fallback |
| `scripts/run_v3_multi_round.py` | 🟡 改 | `--propose {on,off}` flag,默认 on |
| `tests/test_action_proposer.py` | 🆕 NEW | 单测 proposer 策略覆盖 |

---

## 6. 评估方法

### 6.1 指标(对比 `--propose off` baseline + v2 canary 参考)

| 指标 | propose off (main) | propose on (本) | v2 canary (历史) | 目标 |
|---|---:|---:|---:|---|
| change_rate (3×30 mean) | 7% | **目标 ≥ 60%** | 82% | 接近 v2 canary |
| max ACTION1 占比 | 95% | **目标 ≤ 50%** | 18% | 多样性恢复 |
| len(action_semantics) round 2 | 7(满) | **目标 ≥ 3** | 0(空) | 不空,但不爆 |
| 7 个 action 都被尝试 | ❌ 没有 | **目标 ✅** | ✅ | 多样性 |
| levels_completed | 0 | 0(预期还是 0) | 0 | 不期待突破 |

### 6.2 决策门

| 门 | 条件 | 失败应对 |
|---|---|---|
| **G1** plumbing | propose on dry-run 跑通,trace 字段含 candidate letter | 实现 bug,修 |
| **G2** change_rate 改善 | propose on 3×30 mean change_rate ≥ 40% | 改善 < 30pp → proposer 没解决 anchoring;深挖 candidate 多样性 |
| **G3** ACTION 分布健康 | 最大单 action 占比 ≤ 50% | LLM 仍只挑 [B] (known-good) → 强化 [A] (untried) 优先级 |
| **G4** Knowledge 仍学 | round 2 action_semantics 有非空条目 | 写不进去 → 看 Reflection 输出 |

---

## 7. 已知风险

| 风险 | 缓解 |
|---|---|
| LLM 总是选 [B](known-good)→ 还是 commit ACTION1 | candidate ordering 随机化;加 [A] explicit "try this untried" 标签 |
| K=3 太少,真正好的 action 落选 | propose 兜底:若所有候选都 known-bad,加 random legal |
| ACTION6 coords:propose 时怎么定 (x, y)? | 用 click_target bandit 已有逻辑(BUG-10);若无 target,跳过 ACTION6 候选 |
| Multi-choice parse 失败率 | 加正则容错 + LLM 输出 "A" 单字符也算 |
| 跟 R2 mask 冲突 | propose 阶段就已过滤 known-bad,mask 几乎不会触发(预期) |

---

## 8. 跟其它路线的关系

- 跟 [v3_2](../2026-05-14-v3_2-double_agent/architecture.md):本设计**改 Action Agent 的输入/输出格式**,不动 Reflection / Knowledge / mask
- 跟 [predictor v0](../2026-05-16-v0-predictor/architecture.md):predictor 输出 P(change)可以**未来**给 candidate 排序权重。本 v0 不集成
- 跟 [v2_canary_ablation](../2026-05-16-v0-v2_canary_ablation/architecture.md):那个 ablation 是**关掉**问题特征找原因,本设计是**正面**给 LLM 更好的输入格式

---

## 9. 文档历史

- 2026-05-16 初稿。在 feat-2026-05-16-v0-action_proposer 分支上。
