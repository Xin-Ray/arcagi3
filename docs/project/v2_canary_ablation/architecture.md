# ARCHITECTURE — v2 Canary Ablation (v0)

日期: 2026-05-16
状态: 设计 → 待跑(用户决定后启动)
取代关系: 不取代任何文档 — 是 [v3_2](../v3_2/architecture.md) 的诊断分支
前置阅读: [`v3_2_zh.md`](../v3_2/architecture.md)、[`../v3_2/report_hardrules.md`](../v3_2/report_hardrules.md)

---

## 0. 为什么做这个

[v2 canary](../GLOSSARY_zh.md) (commit `1bac4be`,3×30 步) 在 ar25 上拿到 change_rate **60 / 100 / 97 %**。当前 HEAD(`01a7227`)同样开 mask strict 跑 3×200 步,change_rate 只有 **5.5 / 7.0 / 8.5 %**。

二者中间隔了 12 个 commit、6300+ 行改动。其中哪个 commit / 哪个功能导致了退化?

直接 `git checkout 1bac4be` 跑一遍**不现实**:
- 测试套件大改(后来加了 200+ 个测试),v2 commit 的测试不会通过
- 失去 BUG-8/9/10/11/12/13 修复,以及 R4/R5/R6/R7 硬规则
- 失去当前 mask 归因 bug 修复(commit `01a7227`)

但我们可以**保留当前代码**,把 v2 之后引入的几个**核心功能**做成 CLI flag,A/B 比较。这是诊断,不是回退。

---

## 1. 范围

### 1.1 In-scope

- 在当前 HEAD 上,给 3 个候选功能加可关闭 flag
- ar25 2 round × 200 step 跑 4 个组合,统一对照(同 seed、同 mask strict)
- 出表:change_rate / overrides / max_no_op_streak / action_distribution / levels_completed
- 出图:per-round bar + per-step rolling change_rate
- 决策:哪个功能是退化主因?要不要 patch?

### 1.2 Out-of-scope

- 真 rollback 代码到 `1bac4be`(测试坏 + 失去 mask 归因 fix)
- 改 prompt 文案(P0/P1 文案不动)
- 训练 LoRA / GRPO

### 1.3 规则合规

只在 ar25 上诊断 + 不修改模型权重,合规无问题。

---

## 2. 概念表

| 概念 | 一句话定义 |
|---|---|
| **A/B knob** | 一个 CLI flag,默认 ON(=当前 HEAD 行为),OFF 时回到 v2 canary 时代该功能不存在的状态 |
| **退化主因** | 通过 A/B 找到的「关掉它后 change_rate 显著回升 ≥ +20 pp」的那个 flag |

其它术语见 [`../GLOSSARY_zh.md`](../GLOSSARY_zh.md)。

---

## 3. 候选 A/B 功能

按假设强度排序,**最可能是退化主因的在最上**。

### A. `--reflect-semantics={on,off}`(假设最强 ⭐⭐⭐)

**功能**:Reflection Agent 每步是否往 `Knowledge.action_semantics` 写入 update。

**假设**:v2 时代 Reflection 不写 semantics(因为 BUG-9 还没修,写出来都是 "an active object" 没主语,被过滤丢弃)。现在 BUG-9 修了,Reflection 在前 5-10 步就把 `ACTION1: moves UP` 写进 Knowledge,Action Agent 看到这条「known-good」后**永久 commit ACTION1**(95 % 的 round 0 picks)。v2 时 Knowledge 是空的,LLM 没有 anchor,所以**探索更均匀**(50% ACTION3 / 27% ACTION2 / 13% ACTION1 / 10% ACTION6)。

**实现**:`scripts/run_v3_multi_round.py` 接 `--reflect-semantics off` 时,把 Reflection 返回的 delta 里的 `action_semantics_update` 字段统一覆盖成 `{}`,merge 时跳过更新。10 行改动。

**预测**:OFF 后 change_rate +30~50 pp(回到 30-50% 量级)。

### B. `--click-targets={on,off}`(假设强 ⭐⭐)

**功能**:[click_targets bandit](../GLOSSARY_zh.md#click_targets) (BUG-10) 是否启用。

**假设**:v2 时 LLM 反复试 ACTION6,被 R2 mask 替换,**替换出去的 ACTION 多样**(untried 优先)。现在 click_targets 直接把 ACTION6 标 "AVOID" 写到 [CLICK TARGETS] prompt 块,LLM 不再试 ACTION6,**也就不触发 mask 替换**,反而失去了多样化的机会。

**实现**:`run_v3_multi_round.py` 接 `--click-targets off` 时,跳过 `update_click_targets` 调用 + Action prompt 不渲染 `[CLICK TARGETS]` 块。15 行改动。

**预测**:OFF 后 ACTION6 spam 回来(可能 30-50% 的 picks),但 R2 mask 会替换到 untried,**间接拉升 change_rate +10~20 pp**。

### C. `--prompt-format={v3_2,v3_2_legacy}`(假设中等 ⭐)

**功能**:prompt 用当前 7-block 还是 commit `439ca59` 前的 17-block。

**假设**:commit `439ca59` 把 17 块压成 7 块,可能漏掉了 v2 时代的某个 "试 untried" 暗示。

**实现**:已经有 v3.2_legacy 的代码在 git history,把它作为 `prompts_v3_2_legacy.py` 复活。25 行改动。

**预测**:OFF 后 change_rate +5~15 pp(假设较弱)。

### 不上 A/B 的功能

- BUG-8/9/10/11/12/13 修复:这些是**对的**改动,关掉只会让 Knowledge 质量更差(回到无主语 semantics、ACTION6 无 memory)。不测。
- R2 mask 归因 fix(commit `01a7227`):刚修的 bug,关掉只是放回 bug,显然不该测。
- R4/R5/R6/R7 硬规则:这些是细节过滤器,不影响 change_rate 主轴。不测。

---

## 4. 关键决策

### 4.1 为什么先测 A 不测 B / C?

**A 的假设最具解释力**:它**直接预测**了我们观察到的 ACTION1 95% 现象。如果 A 通过,B/C 可能就不用测。

**A 实现最便宜**:10 行代码。B/C 都要 15-25 行。

**A 风险最低**:关 Reflection 写 semantics 不会让其它路线崩,只是 Knowledge 一直空。

### 4.2 为什么不全部并行测?

跑 4 个组合(2 round × 200 step × 4 = 1600 step × 6-8 s/step = ~3 hours wall clock)。3 小时一次实验合理;并行 4 组占 GPU 4 倍时间总和。**串行 A → 看是否解释了大部分退化 → 再决定是否测 B/C**。

### 4.3 为什么不直接 fix?

「不写 semantics」如果真的解决了 change_rate 但**也牺牲 Knowledge 学习能力**(到 round 2 还是空的),那是另一种坏。我们要的是**信息**(A 是不是退化主因),不是**马上 ship 一个 fix**。

---

## 5. 数据流

```
ar25 SDK env
    │ env.step
    ▼
v3.2 ActionAgent (LoRA 不变) ── 选 action ──→
                                              │
                                              ▼
v3.2 ReflectionAgent ────── 每步反思 ──→ Knowledge merge
                                              │
                                              ▼
              ┌─── 当 --reflect-semantics=off ─→ DROP action_semantics_update
              └─── 当 --click-targets=off ─→ 不调 update_click_targets,
                                             [CLICK TARGETS] 块不渲染
```

---

## 6. 评估方法

### 6.1 指标

(每个组合都记)

| 指标 | 单位 | 期望(当 flag OFF 解释了退化) |
|---|---|---|
| `change_rate` (per round) | % | OFF >> ON,且接近 v2 canary 的 60-100% |
| `max_no_op_streak` (per round) | step | OFF << ON,接近 v2 的 3-5 |
| `action_entropy` (per round) | bits | OFF >> ON(更分散) |
| `n_overrides` (R2 mask 触发次数) | int | 不直接预测;但 OFF 时可能更高(因为更多 ACTION6 spam) |
| `levels_completed` | int | 都是 0 的概率高;若 OFF 出现 > 0 → 突破 |

### 6.2 决策门

| 门 | 条件 | 失败应对 |
|---|---|---|
| **G1 信号显著** | 至少一个 A/B 的 change_rate Δ ≥ 20 pp(2 round 平均)| 没显著差异 → 说明退化主因不在这 3 个候选里,扩到 D/E/F |
| **G2 主因定位** | 单个 A/B(A 或 B 或 C)就解释 ≥ 30 pp 退化 | 未定位 → 多因素叠加,分别小幅修都得做 |
| **G3 不引入新坏处** | OFF 后 Knowledge / Reflection 输出仍合理,无 crash | 有 crash → 实现 bug,修后再跑 |

### 6.3 终止条件

- A 跑完后 change_rate 回到 ≥ 40% → **不用测 B/C**,直接 ship 「Reflection 延迟写 semantics 直到 N 步」的 fix
- A 没改善 → 测 B → 同样标准
- A/B/C 都没改善 → 写报告说明,扩到 D(prompt 详细文案) / E(masked-hash stuck) / F(orchestrator auto rules)

---

## 7. 已知风险

| 风险 | 概率 | 缓解 |
|---|---|---|
| `--reflect-semantics off` 让 Knowledge 永远为空 → 表面 change_rate 高但**学不到东西**,后续 round 1+ 也学不到 | 高 | A/B 报告里明确写「**这不是 ship fix,这是诊断**」。真 fix 要更精细(例如 N 步后才允许写) |
| 单次 ar25 run 方差大,不能下结论 | 中 | 每个组合跑 2 seed,取均值 + 标准差 |
| Reflection 不写 semantics,但 Action 还能从 OutcomeLog 推 → 实际差异比预期小 | 中 | Action prompt 的 `[ACTION effects observed]` 是从 OutcomeLog 渲染,确实有兜底。我们看的是**第一次写**的延迟效应 |
| 3 小时 × 4 组合 = 12 小时,Kaggle 跑挂 SDK | 低 | 用 `scripts/run_scheduled.ps1` 后台 + 失败重启 |

---

## 8. 实施顺序

| # | 步骤 | 文件 | 验证 |
|---|---|---|---|
| 1 | 加 `--reflect-semantics {on,off}` flag | `scripts/run_v3_multi_round.py` + Reflection merge 逻辑 | 单测:OFF 时 delta 的 action_semantics_update 不进 Knowledge |
| 2 | 跑 ar25 2×200 with `--reflect-semantics off`(对照 ON 用现有 mask_revive_3x200 数据)| 后台 ~3h | trace.jsonl 写成,无 crash |
| 3 | 写 `outputs/reports/v2_canary_ablation_A.md`(按 CONVENTIONS §4 模板)| 新报告 | 8 节齐 + 3 张图 |
| 4 | 决策门:G1 / G2 通过? | -- | 通 → 继续 5;不通 → 测 B |
| 5 | (若 A 通)写「延迟写 semantics 直到 step N」的 fix 设计 | 新 architecture doc | -- |
| 6 | (若 A 不通)加 `--click-targets off` flag,跑 B | -- | 同 §3-4 |
| 7 | (若 A/B 都不通)加 `--prompt-format v3_2_legacy` flag,跑 C | -- | 同 §3-4 |

---

## 9. 跟其它路线的关系

```
v3.2 (当前主线)
   ├── mask_revive (commit 01a7227) ←—  当前实测在用
   │
   ├── v2_canary_ablation (本文)    ←—  诊断:为什么 mask_revive 不到 v2 水平?
   │       └─ 若 A/B 通过 → ship 后续 fix(延迟写 semantics 等)
   │
   ├── predictor_v0                  ←—  独立 add-on,补 LLM 选 action 信号
   │
   └── grpo_v0                       ←—  独立诊断,看 LLM-RL ceiling
```

本路线**不**改 Knowledge / Reflection / Action Agent 的核心实现,只加 toggle flag。

---

## 10. 文档历史

- *2026-05-16 初稿。用户提出「v2 canary 路线选几个功能验证改进」后写。等用户确认 A/B/C 顺序后启动。*
