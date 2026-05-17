# ARCHITECTURE — Subtask Decomposition v0(把通关拆成小任务)

日期: 2026-05-17
状态: 设计 → 待用户 review 后立项分支
前置阅读: [`tonight_summary.md`](../../tonight_summary.md);[`model_bench/report_5game.md`](../2026-05-17-v0-model_bench/report_5game.md)

---

## 0. 为什么做这个(用户提出的核心论点)

> 「我觉得核心只有一件事情:**根据目前的空间和目标,可以输出正确的行为**。一次实现不了,就拆解小任务依次验证。」—— 用户 2026-05-17

实测证据支持「整体跑死局」:

| 跑过的全 game 实验 | 结果 | 我们从中学到什么 |
|---|---|---|
| Qwen mask off baseline | 5-8% change_rate / 0 通关 | LLM 决策固化 |
| Qwen+propose 3×30 smoke | 60-75% / 0 通关 | proposer 修了决策但不通关 |
| v2 canary 3×30 | 86% / 0 通关 | 空 Knowledge + 高探索也不通关 |
| SmolLM3+propose 5×2×300 | 64% mean / 0 通关 | 换模型也不通关 |

**结论**:任何「整体跑通」实验都得到「change_rate 改善 + 0 通关」。无法定位是哪一步出问题。

**Subtask 拆分**:把「通关 ar25」分解成可独立测的小能力,每个小能力定义清晰的 input/output + 合成数据集 + PASS/FAIL 标准。哪个 sub-task 过 哪个不过,就知道下一步精确改哪。

---

## 1. 范围

### 1.1 In-scope

- 把 ar25 (基于读源码已知机制) 通关分解成 **7 个独立子任务**(T-NAV-1..3, T-SEL-1..2, T-GOAL, T-RETRY)
- 每个子任务:
  - 独立 git 分支 `feat-2026-05-17-v0-subtask-<name>`
  - 独立 architecture.md(用 §3 的模板填)
  - 独立合成数据集(基于 ar25 机制人工 / 程序生成,100-200 题)
  - 独立 PASS/FAIL 准则(例:N 选 1 accuracy ≥ 80%)
  - 独立 report.md
- 跑顺序:**单步类先**(T-NAV-1, T-SEL-1)→ **状态类**(T-GOAL)→ **多步类**(T-NAV-2, T-NAV-3)→ **元学习类**(T-SEL-2, T-RETRY)

### 1.2 Out-of-scope

- 不在 SDK 真跑 game(子任务都用合成数据集 + 离线评估)
- 不改 v3.2 主线 framework(子任务结论会**回头**影响 framework 设计)
- 不做 RL / SFT(纯 zero-shot prompt + Knowledge 协同)
- 不跨多个 game(只针对 ar25 机制;别的 game 通过后再扩)

### 1.3 规则合规

子任务用**合成数据**(由 ar25 已知机制 + 随机坐标合成),**不**用 demo game label。符合 ARC Prize **no task-specific optimization** 条款(可争论:任务是「教会通用空间技能」,prompt 改进会泛化到别的 game)。

---

## 2. 概念表

| 概念 | 一句话 |
|---|---|
| **subtask** | 一个独立可测试的 LLM 能力,例:「给 (object, target) 返回 first ACTION」 |
| **subtask probe set** | 该 subtask 的合成数据集(input → expected output) |
| **PASS** | LLM 在 probe set 上准确率 ≥ 阈值(每子任务定) |
| **transfer** | sub-task PASS 后,该能力在真 game 里 measurable 改善 |

---

## 3. 7 个 Subtask 定义(按依赖顺序)

### T-NAV-1 单步导航(最基础)

- **能力**: 给 object 当前位置 + target 位置 + ACTION 映射,LLM 选**单个 ACTION** 缩短 distance
- **input format**:
  ```
  Object at (row=10, col=20). Target at (row=10, col=25).
  ACTION1=UP, ACTION2=DOWN, ACTION3=LEFT, ACTION4=RIGHT (each 1 cell).
  Pick the ACTION that reduces distance most.
  ```
- **output format**: `Answer: ACTION_X`
- **probe set**: 100 题(50 纯 x-方向 + 50 纯 y-方向),target 1-10 cells away
- **PASS**: ≥ 80% (Qwen no_think 现在 ~50%,SmolLM3 CoT ~100%)
- **难度**: ★ 低 — 等价于 model_bench T1+T2 合并
- **预计**: SmolLM3 CoT/auto 必过;Qwen 可能过可能不过

### T-NAV-2 多步序列(线性)

- **能力**: 给 object + target 在**单维度**上隔 N cells,输出 **N 个相同 ACTION**
- **input**:
  ```
  Object at (10, 20). Target at (10, 25). 5 cells right.
  Output the sequence of ACTIONs as a comma-separated list.
  ```
- **output**: `Answer: ACTION4, ACTION4, ACTION4, ACTION4, ACTION4`
- **PASS**: ≥ 70%(允许长度对 + ACTION 名对 但顺序错)
- **难度**: ★★ — 测 LLM 是否会计数
- **预计**: Qwen 失败(整除不会);SmolLM3 CoT 大概过

### T-NAV-3 多维计划(L 型)

- **能力**: object + target 隔 (Δx, Δy) 都非零,输出**混合序列**
- **input**: 同 T-NAV-2 但 target 在 (15, 25),object 在 (10, 20)
- **output**: 5 ACTION2 + 5 ACTION4(顺序不限,**或任意 interleave**)
- **PASS**: ≥ 60%(最难,允许顺序自由)
- **难度**: ★★★ — Tier 1 SFT 五个 planning probe 失败的就是这类
- **预计**: 大多数 3B 模型失败 → 是真正瓶颈

### T-SEL-1 ACTION6 click 命中

- **能力**: 给 object bbox (rows 5-10, cols 10-15),返回让 ACTION6 选中该 object 的 (x, y)
- **input**:
  ```
  Object A bbox: rows 5-10, cols 10-15.
  Object B bbox: rows 30-35, cols 40-45.
  Output ACTION6 coords to select Object A.
  ```
- **output**: `Answer: ACTION6 12 7` (或任何在 A 的 bbox 内的 (x, y))
- **PASS**: ≥ 90%(很简单)
- **难度**: ★ — 用 GIF 实测 SmolLM3 已经精准点击,验证
- **预计**: SmolLM3 / Qwen 都过

### T-SEL-2 ACTION5 切换决策

- **能力**: 给 ObjectMemory 当前 active object 是 A,有 3 个可选 object {A, B, C},LLM 决定**何时**用 ACTION5 切换
- **input**:
  ```
  Active: obj_001 (yellow 1x1 at (10,10)).
  Other objects: obj_002 (yellow 1x1 at (15,15)), obj_003 (red 1x1 at (30,30)).
  Goal: align both yellow objects.
  Current strategy: keep moving obj_001? OR switch to obj_002?
  ```
- **output**: `Answer: [switch=YES/NO]; if YES, target obj_id; reason`
- **PASS**: ≥ 70%(meta-reasoning,允许多种合理答案)
- **难度**: ★★★ — 元学习
- **预计**: 大多数模型失败

### T-GOAL 目标状态识别

- **能力**: 给当前 frame 的 object 状态,判断**是否达到 goal**
- **input**:
  ```
  Goal: align two yellow squares vertically in left column.
  Current: obj_001 (yellow) at (5, 0). obj_002 (yellow) at (10, 0).
  Are we done? Answer YES/NO.
  ```
- **output**: `Answer: YES / NO + 1 sentence`
- **PASS**: ≥ 80%(分类型简单)
- **难度**: ★★ — 测「读 ObjectMemory 状态对照 goal」
- **预计**: SmolLM3 大概过;Qwen 取决于 prompt 写法

### T-RETRY 失败重试(元能力)

- **能力**: 给最近 5 步全 no-op 的 trace,LLM 判断**应该换策略**还是继续
- **input**:
  ```
  Last 5 actions: ACTION1, ACTION1, ACTION1, ACTION1, ACTION1.
  All resulted in no-op (object hit ceiling).
  What should you do?
  (A) Continue with ACTION1
  (B) Switch to ACTION2 (try opposite)
  (C) Switch to ACTION5 (cycle to different object)
  (D) Random
  ```
- **output**: `Answer: B` (or another reasonable)
- **PASS**: ≥ 60%(不期待完美,但应该比随机 25% 好)
- **难度**: ★★ — 测自我反思
- **预计**: 中等难度,可能过

---

## 4. 关键决策

| 决策 | 为什么 |
|---|---|
| **离线** probe set 评估 而不是 SDK 真跑 | 一个 probe 0.1 秒 vs SDK 一步 30 秒 → 200 题 < 1 分钟 vs 90 分钟 |
| **multi-choice (A/B/C/D)** 不开放生成 | 容易 score,避免 parse 噪声;跟 model_bench probe 一致 |
| **每子任务一个分支** | 各自有 commit history,失败的留作经验,成功的可 merge |
| **顺序跑而非并行** | 一个 GPU,串行 load model 即可。每 subtask 一次跑 ≤ 5 分钟 |
| **数据合成 不抄 game source** | 合规;source 只用于「validate 我们对机制的理解对」,probe 数据用合成的 |
| **跑两个模型** Qwen + SmolLM3 CoT | 看到 model 维度的差异,验证 SmolLM3 CoT 真有差距还是 probe 太简单 |

---

## 5. 模块清单

| 文件 | 状态 | 责任 |
|---|---|---|
| `arc_agent/subtask_probes/__init__.py` | 🆕 NEW | 7 个 subtask probe 生成函数 + `get_probes(subtask)` |
| `arc_agent/subtask_probes/generate.py` | 🆕 NEW | 程序合成各 subtask 的 100-200 题(seed 固定可复现) |
| `scripts/bench_subtask.py` | 🆕 NEW | 对每个 (model, subtask) 跑 probes,出 metrics.json + plot |
| `scripts/plot_subtask_results.py` | 🆕 NEW | 跨 subtask + 跨 model 的 heatmap |
| `tests/test_subtask_probes.py` | 🆕 NEW | 每个 subtask 的 1-2 题 sanity:correct answer 真的对 |
| `docs/project/2026-05-17-v0-subtask_decomp/architecture.md` | 🆕 NEW | 本文件 |
| `docs/project/2026-05-17-v0-subtask_decomp/report_*.md` | 🆕 NEW | 每 subtask 跑完一份 |

---

## 6. 评估方法

### 6.1 单 subtask 指标

| 指标 | 含义 | 目标 |
|---|---|---|
| accuracy | n_correct / n_probes(100-200) | 见每 subtask PASS 阈值 |
| per-model | Qwen 5.5% / SmolLM3 /no_think / SmolLM3 CoT | 看模型差异 |
| confusion | 错的题集中在哪几类 | 找 LLM 的弱点 |

### 6.2 总 decision

| 通过/失败 | 含义 |
|---|---|
| **T-NAV-1 + T-SEL-1 都过** | 单步能力 OK,可以扩到 multi-step |
| **T-NAV-2 / T-NAV-3 失败** | 多步规划是真瓶颈 → 上 CoT 或工具化(planner 在 LLM 之外) |
| **T-GOAL 过 + T-NAV-* 失败** | LLM 知道何为成功但不会执行 → 需要更结构化的 action sequence template |
| **全部过但 5×2×300 仍 0 通关** | 子任务 ≠ 整体能力。需要看「在 game 里这些子任务的 input/output 真的 wire 起来了吗」 |

---

## 7. 已知风险

| 风险 | 缓解 |
|---|---|
| 合成 probe 跟真 game 不匹配 | 每 subtask 第一组 5 题手写从真 ar25 trace 抽,跟程序合成版对照 |
| 模型在 probe 上过但 game 里仍失败 | 这是**重点要发现的事**;能定位「在 game 里如何 wire 不对」 |
| 7 个 subtask 太多,GPU 太慢 | 每 subtask 一次 100 题 × 5s/probe × 2 model = 17 min;总 17×7 = 2h。可接受 |
| 子任务太「合成」失去现实意义 | 每个 subtask 完成后,**回头跑** ar25 1×1×30 看真 trace 里相应步骤是否真的改善 |

---

## 8. 跟其它路线的关系

```
v3.2 (主线)
  │
  ├── action_proposer  ←— K=3 N 选 1
  │
  ├── model_bench       ←— 选 backbone (SmolLM3 winner)
  │
  └── subtask_decomp (本文)  ←— 拆解通关能力为 7 小任务
        │
        └─ 每子任务结果 → 影响 prompt / Knowledge schema / proposer 策略
              ↓
        回喂到主线
```

不取代任何路线,**是诊断工具**。

---

## 9. 文档历史

- *2026-05-17 15:00 初稿。基于用户 14:50 提出的「拆解小任务」原则。等用户 review 后启动 T-NAV-1 实施*
