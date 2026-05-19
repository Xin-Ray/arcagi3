# 项目总入口 (README.md)

> 10 分钟看完知道现状。状态码 🟢 当前活、🟡 参考、⚫ 历史。

最近更新: 2026-05-19 morning (v4 + per-module re-validation: bench-vs-production distribution shift 暴露两个 hidden bug)

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

## 2. 目标拆解 + 解法验证地图

> **作用**: 顶层目标怎么拆,每个子目标的解法在哪验证,当前 PASS / FAIL / UNKNOWN 状态。**这一节是项目方法学心脏** — 来自 2026-05-19 用户提的修正(之前的 6 模块"已验证"用的是 proxy metric,不是模块自身 PASS 定义)。

### 2.1 顶层目标 → 6 模块拆解

```
通关 1 game (env.state == WIN)
   ↓ 必要条件
推断 win 条件 + 高效执行到该 state
   ↓ 拆 6 个模块
1. Goal Generation       Reflection 从帧观察 + history 推断 win 条件
2. Goal Recognition      parser/judge 判 "hypothesis 是否已达成"
3. Action Selection      Action 给 hypothesis,选朝 hypothesis 方向的 action
4. Reflection Loop       achieved=True 但 env!=WIN → reject hypothesis 重写
5. force_cot Prompt      Action prompt 加 step-by-step + goal-target 措辞
6. K=3 Candidates        action_proposer 提供 explore 多样性,防 ACTION1 死循环
```

### 2.2 每模块 PASS 定义 + bench 验证 + production 验证 + 当前结论

| # | 模块 | 真 PASS 定义 | bench 验证 | production 验证 | 当前结论 |
|---|---|---|---|---|---|
| 1 | **Goal Generation** | hypothesis 描述的状态 = game 真实 win 条件 | ❌ 没 bench(game GT 隐藏) | ⏳ 等用户标 `annotation_request.md` | **UNKNOWN** |
| 2 | **Goal Recognition** (parser) | parser=True ↔ env-WIN | T-GOAL 83% on synthetic ✅ | 0/5 wins → 没法测 precision/recall | bench ✅, prod 待测 |
| 3 | **Action Selection** | action 方向 = hypothesis 方向 | T-NAV-1 71%, T-NAV-3 97% (synthetic) ✅ | **0/794 步有 directional hypothesis** ❌ | **prod FAIL** (上游 bug:hypothesis 没方向) |
| 4 | **Reflection Loop** | rejected 的 hypothesis 都是真错的 | ❌ 没 bench | ⏳ 等用户标 `annotation_request.md` Module 4 | **UNKNOWN** |
| 5 | **force_cot Prompt** | reasoning 提的 ACTION = 实际选的 action | T-NAV-3 force_cot +30pp synthetic ✅ | **51% match,49% 纯 LLM 字母 confusion** ❌ | **prod FAIL** (letter mapping shuffle) |
| 6 | **K=3 Candidates** | 每步 K=3 包含 goal-aligned 选项 | Phase 3 ablation +75pp change_rate ✅ | K=3 出现率 78% (22% 步缺) | prod **partial PASS** (proxy) |

### 2.3 当前已知的 production-level bug

| bug | 模块 | 表现 | 根因 |
|---|---|---|---|
| **B1**: hypothesis 无方向 | Module 1 / 3 | Reflection 0/794 步写 `move_to_row/col/center`,全 `align_any`/`match` | `/no_think` SmolLM3 偏好 "match X with Y" dialect |
| **B2**: reasoning ↔ action 49% LLM 脱节 | Module 5 | model reasoning 说 ACTION X 但 output `choice: Y` 选了别的字母 | action_proposer K=3 letter 动态 shuffle,model 跟踪不准 |

### 2.4 bench-vs-production distribution shift(方法学根因)

bench 砍掉了 production 的几样东西,所以 bench 全 PASS 不代表 production work:

| bench 用的 | production 加了 | 后果 |
|---|---|---|
| 预写好的 hypothesis (T-GOAL) | Reflection 自由写 | hypothesis dialect 不可控 → B1 |
| 固定 letter mapping (A→ACTION1) | K=3 候选每步 shuffle | model letter confusion → B2 |
| (current, target) 直接给 | Reflection 推断 target | target 可能是幻觉 (smoke 2/3 发现) |
| 一次性 single probe | 跨步骤累积 Knowledge | 累积错误信息污染 |

→ **修补思路**: 每个 production-level bug 都应该有对应的 production-level bench (T-DIALECT 测 hypothesis dialect / T-MAPPING 测 letter tracking),不能只 bench sanitized 形态。

### 2.5 已验证有用的解法模块(可放心保留)

| 模块 | 验证证据 | 状态 |
|---|---|---|
| scipy perception | scipy 100% vs Qwen-VL 0% | ✅ PASS |
| parser (goal_evaluator) | T-GOAL 83% vs LLM 68% (recall on TRUE 100% vs 22%, 1.25M× 快) | ✅ PASS |
| Parser v1 vocab (edge/center/align_any/reach/match) | 100% v2 round 0 + smoke 3 production hypothesis | ✅ PASS |
| Force-reject 机制 | smoke 3 验证 Reflection 真改写 | ✅ PASS (但 reject correctness 待标注 Module 4) |
| Hallucination 检测 | smoke 2-3 18/20 触发清掉幻觉色 hypothesis | ✅ PASS |
| Reflection 截断 fix (token budget) | v1→v2 JSON 输出从 0/100 → 100/100 | ✅ PASS |
| Step-budget pooling | Phase 4 ar25 round 0 早终 → round 1 继续 | ✅ PASS |
| `action_proposer` K=3 | Phase 3 ablation +75pp(唯一关键 v3.2 旧模块) | ✅ PASS |
| Reflection schema validation (wide) | Phase 1B 丢掉 invalid hypothesis | ✅ PASS |

### 2.6 已验证无用 / 有害的模块(砍掉)

| 模块 | 证据 | 状态 |
|---|---|---|
| `click_targets` bandit | cross-validation 0/5 命中 + Phase 3 ablation +0pp | ❌ 砍 |
| `action_semantics` from LLM | Phase 3 ablation +0pp;只在 propose 转化后才间接有用 | ❌ 砍 |
| `R1/R4/R5/R6/R7` hard rules | Phase 3 ablation +0pp;R3 在 action_agent 仍留 | ❌ 砍 |
| `R2 mask` | 没单独 ablate,保留 off 跟 V4 一致 | ❌ 砍 |
| 8-block prompt 的 LOW-PRIORITY / TEXTURE | 让 prompt 过长 → /think 死循环 | ❌ 砍 |

### 2.7 下一步验证 (按 ROI)

| P | 行动 | 验证什么模块 |
|---|---|---|
| P0 | 用户标 `annotation_request.md` (~15 min, 25-35 条) | Module 1 + 4 PASS rate |
| P0 | `--validate-hypothesis-schema strict` 重跑 ar25 | Module 1 经 schema 强制后是否变 directional |
| P0 | 固定 action_proposer letter mapping (A永远=ACTION1) | Module 5 letter confusion 是否消除 |
| P1 | A\* 寻路接管 Module 3 navigation (200 行,半天) | Module 3 从 prod FAIL → PASS;预期 ar25/dc22 等可通关 |
| P1 | T-DIALECT bench + T-MAPPING bench | bench 跟 production 距离补足 |
| P2 | 升级模型试 SmolLM3-7B / Qwen3-4B | 模型本身的 dialect / mapping tracking |

---

## 3. 当前分支状态(2026-05-19 morning)

| 分支 | 状态 | 顶端 commit | 在做什么 |
|---|---|---|---|
| `main` | 🟢 主线 | `e07e7d1` (2026-05-16) | v3.2 + mask + Knowledge + click_targets。**没有任何 push/merge 发生**;所有新工作都在 feature 分支 |
| `feat-2026-05-16-v0-action_proposer` | 🟡 完成,本地 | `b9e2e87` | 包含 action_proposer + model_bench + SmolLM3 5×2×300 + CoT 1×2×100 |
| `feat-2026-05-17-v0-subtask-T-NAV-1` | 🟢 已 push origin | `c61f309` | 5 subtask 验证 + force_cot A/B 全跑完。终判 2 PASS / 3 FAIL |
| `feat-2026-05-18-v0-det_goal_plus_force_cot` | 🟢 完成,本地 | `12fd46c` | det_goal + force_cot + goal_judge A/B + parser 多次扩展。多版 smoke 暴露 LLM 不听 advisory prompt + action_semantics propagation bug |
| `feat-2026-05-19-v0-v4_clean_baseline` | 🟢 **当前** | `ff01d1b` | **5-phase ablation + per-module re-validation**:V4+propose mean change_rate **82%**(+18pp);action_proposer 是唯一关键;模块重审暴露 **2 hidden bugs** (hypothesis 无方向 + reasoning↔action 49% 脱节);**等用户标 Module 1+4** |
| `docs-reorg` | 🟡 等合 | `181d0b6` (2026-05-16) | 文档体系迁移,昨天写完 |
| `feat-2026-05-16-v01-predictor` | ❌ 不合 | `722db78` (2026-05-16) | CNN OOD AUC 0.16 失败结论存档 |
| `feat-2026-05-16-v0-grpo_train` | ⏳ Phase 0 | `11ed21f` (2026-05-16) | rollout_wrapper + 10 unit tests pass。真训练未跑 |
| `v2-canary` | 🟡 历史快照 | `1bac4be` (2026-05-14) | R1+R2+R3 引入点,read-only |
| `v2-canary-verify` | 🟡 已完成 | `cda68f9` (2026-05-16) | 86% change_rate 复现成功,不合主线 |

---

## 4. 当前方向(2026-05-19)

**v4 5-phase 跑完 + per-module re-validation 揭示方法学 gap**:之前 6 个 "validated" 模块用的都是 proxy metric (change_rate / count / diversity),不是模块自身 PASS 定义。用真定义重审 Phase 4 5 game trace 暴露 **2 hidden bugs**:

1. **Hypothesis 无方向** — Reflection 在 `/no_think` 模式下偏好 "match X with Y" 句式(0/794 步写了 directional kind)。Action 拿到的 hypothesis **没有方向信号** → 没法朝目标走
2. **reasoning ↔ action 49% LLM 脱节** — model reasoning 说选 ACTION X,但实际 output "choice: Y" 选了别的字母(orch_override 0 次,纯 LLM 字母映射 confusion)

**共同根因**: bench 用 sanitized prompt(固定 letter mapping + 给定 hypothesis),production 用动态 shuffle + 自由 Reflection 输出 — **bench-vs-production distribution shift**。

按 ROI 排下一步:

1. **等用户标 Module 1+4**(`docs/project/2026-05-19-v0-v4_clean_baseline/annotation_request.md`)→ 算 6 模块 PASS rate 完整表
2. **`--validate-hypothesis-schema strict`** 重跑 5 game — 强制 Reflection 写 directional kind,看 wins 是否打开
3. **action_proposer letter mapping 固定**(A→ACTION1 永远),不 shuffle — 消除 LLM 字母 confusion
4. 加 **T-DIALECT bench**(Reflection 自由写时 kind 分布)+ **T-MAPPING bench**(动态 letter shuffle 跟踪)— 补 bench 跟 production 的距离

---

## 5. 验证结果汇总

### 🔑 关键新发现 (2026-05-19,overnight v4 + 方法学修正)

#### 🚨 Hidden bugs(用户提的 per-module 重审才暴露)

| 发现 | 来源 |
|---|---|
| **Hypothesis 0/794 步写 directional kind** —— Reflection 在 /no_think 全写 align/match 无方向句式 → Action 无法朝目标走 | module_validation §2 |
| **reasoning ↔ action 49% pure LLM disconnect** —— model 说"选 ACTION X" 实际 output "choice: Y" 选了别的字母,orch_override 0 次 | module_validation §3 |
| **方法学错误**:之前 6 模块都用 proxy metric 测,不是 PASS 真定义 → bench 跟 production 严重 distribution shift | module_validation §1 |

#### 📊 v4 5-phase 成果

| 发现 | 来源 |
|---|---|
| **V4+propose 5 game mean change_rate 82% (vs SmolLM3 5×2×300 baseline 64%, +18pp)** | v4_clean_baseline Phase 4 |
| **Ablation: action_proposer 是 v3.2 旧模块里唯一关键** —— click_targets / action_semantics / hard_rules 单独加 +0pp | v4_clean_baseline Phase 3 |
| **V4 minimal (砍光) ar25 退化 11% change_rate** —— Action 死循环 ACTION1 (94%) hit edge | v4_clean_baseline Phase 2 |
| **`/think` 在 production prompt 下 chain 仍 0/10 闭合** —— 即使 V4 短化 prompt | v4_clean_baseline Phase 1 |
| **Parser "match X with Y" 模式 (Phase 1B 新发现)** —— Reflection /no_think 模式喜欢用这种方言 | Phase 1B |
| **Step-budget pooling 工作** —— round 早终止后续 round 继续耗用预算 | Phase 0/4 |
| **5 game G_base 全 0/5 wins** —— change_rate 提升不等于通关,真正 gap 在上 2 个 hidden bug | Phase 4 §7 + module_validation |

### 🔑 早前发现 (2026-05-18)

| 发现 | 来源 |
|---|---|
| **Python parser 完胜 LLM judge on goal-achievement** —— 83% vs 68% acc,**100% recall on TRUE vs 22%**,1.25M× 更快 | goal_judge_ab |
| **Parser extended v1 覆盖 100% v2 production hypothesis** —— "edge" / "center" / "align (no axis)" 等 9/9 patterns 都 parse | goal_judge_ab |
| **5 subtask PASS ≠ production wins** —— 拆分覆盖 "given target → execute" 下游,**完全没测 "from frame → hypothesize → revise" 上游** | det_goal §6 |
| **v1 (Reflection 截断) 75% change_rate > v2 (Reflection 正常) 47%** —— **真正主导的是 orchestrator R3 / mask,不是 LLM** | det_goal §5.2 |
| **T-NAV-1 production 93% > bench 71%** —— bench letter-shuffle 4-选-1 比 production "pick ACTION1..7" 还难 | det_goal §5.4 |
| **T-SEL-1 production 0% << bench 70%** —— click_targets bandit 选错 object,**bench 没测"选哪个"这一步** | det_goal §5.4 |
| **goal_evaluator 100/100 都 None** —— Reflection 实际写 "to top edge" / "towards center",我的 parser 找 "col=N" → 输入分布错 | det_goal §5.3 |
| **架构含义**: 下一个分支应该测 **T-REVISE** (Reflection 是否会 evidence-driven 修目标),而不是再补 prompt | det_goal §6 |

### 🔑 早前发现 (2026-05-18 02:00,force_cot A/B)

| 发现 | 来源 |
|---|---|
| **force_cot prompt 让激活率从 0-100% 不稳定 → 56-100% 稳定** | force_cot (2026-05-18) |
| **T-NAV-3 67% → 97% (+30pp PASS)** —— prompt 单 fix 直接转 PASS | force_cot |
| **T-GOAL long_acc = 33.7%** —— 即使 98/100 激活 CoT,模型也答不对,**任务超出 SmolLM3-3B 能力**,prompt 救不了 | force_cot |
| **T-SEL-1 force_cot **倒退** 78% → 70%** —— force_cot 不是 free lunch | force_cot |

### 🔑 早前发现 (2026-05-17 17:50)

| 发现 | 来源 |
|---|---|
| **T-NAV-2 (多步同方向) 95% PASS** —— CoT 100% 激活 | subtask-T-NAV-2 |
| **T-GOAL (识别 goal 达成) 30% 严重 FAIL** —— CoT 0% 激活,几乎随机 | subtask-T-GOAL |
| **0/5 通关的强候选根因 = T-GOAL** —— change_rate 64% 但 reflection 认不出 win state | subtask_decomp |
| **CoT 激活率 vs accuracy 高度相关** —— Pearson 接近 1,activation 才是 Pass/Fail 决定因素 | subtask_decomp |
| **`/think` system flag 只能让已激活的 CoT 不截断,不能强制激活** | subtask-T-NAV-1 |

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
| **T-NAV-1 单步方向选择 / SmolLM3 CoT(无 `/think` 注入)** | 48.0%(目标 ≥ 90%);只 11/100 真激活 CoT,其余短路径 | subtask-T-NAV-1 (2026-05-17) |
| **T-NAV-1 v1 (/think 注入)** | 52% —— 整体没改善;activation 仍 11/100,但激活的 90.9% | subtask-T-NAV-1 (2026-05-17) |
| **T-NAV-3 L 型路径** | 67% FAIL;CoT 1/100 激活 → 模型短路径不算 delta | subtask-T-NAV-3 (2026-05-17) |
| **T-SEL-1 ACTION6 click hit-test** | 78% FAIL(-12pp,接近);(x,y) vs (row,col) 约定混淆 | subtask-T-SEL-1 (2026-05-17) |
| **T-GOAL YES/NO 目标判定** | **30% SEVERE FAIL**(-60pp);CoT 0% 激活,4 选 1 → 字母模式 | subtask-T-GOAL (2026-05-17) |
| **force_cot user prompt 路线** | T-NAV-3 67→97 PASS,T-GOAL 30→35 (long_acc 33.7% near-random),T-SEL-1 78→70 倒退 | force_cot A/B (2026-05-18) |
| **LLM 单做 T-GOAL 的极限** | force_cot 让 98/100 真激活 CoT,long_acc 仍 33.7% → SmolLM3-3B 本身做不了 goal recognition | force_cot §4.2 |

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

## 6. 文档结构

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
    ├── 2026-05-17-v0-model_bench/            3 model bench + SmolLM3 5×2×300 + /no_think 发现
    ├── 2026-05-17-v0-subtask_decomp/         7 subtask DAG + cross-summary 报告
    ├── 2026-05-17-v0-subtask-T-NAV-1/        单步方向 verify(v0 48% / v1 52%,FAIL)
    ├── 2026-05-17-v0-subtask-T-NAV-2/        🆕 多步同方向(95% **PASS**)
    ├── 2026-05-17-v0-subtask-T-NAV-3/        🆕 L 型路径(67% FAIL)
    ├── 2026-05-17-v0-subtask-T-SEL-1/        ACTION6 click(78% FAIL, borderline)
    ├── 2026-05-17-v0-subtask-T-GOAL/         YES/NO 目标(30% SEVERE FAIL,疑似 0 通关根因)
    ├── 2026-05-18-v0-force_cot/              force_cot A/B(T-NAV-3 PASS,T-GOAL long_acc 证伪 LLM)
    ├── 2026-05-18-v0-det_goal_plus_force_cot/ 集成 + 交叉验证(5 subtask PASS != production wins 的证据)
    ├── 2026-05-18-v0-goal_judge_ab/           Python parser vs LLM judge A/B(parser 完胜)
    └── 2026-05-19-v0-v4_clean_baseline/       🆕 5-phase ablation + 5 game eval(V4+propose mean 82%)
```

---

## 7. 版本历史(最新在上)

---

### 2026-05-19 morning (latest) — 🟢 v4 + per-module re-validation — `docs/project/2026-05-19-v0-v4_clean_baseline/`

- **分支**: `feat-2026-05-19-v0-v4_clean_baseline`(本地)
- **关键 commits**: `57d5d5a`(CLI flags)→ `eaeb89b`(docs)→ `823e7cc`(parser match)→ `017e2f5`(Phase 1)→ `f4758f0`(Phase 2)→ `d114b1f`(Phase 3)→ `bb1c8f7`(Phase 4 + final)→ `ff01d1b`(per-module re-validation)
- **5-phase 结果**:
  - Phase 1: `/no_think` wins(`/think` chain 0/10 闭合)
  - Phase 2: V4 minimal ar25 200 step 退化到 11% → 触发反向 Phase 3
  - Phase 3: ablation 4 模块,**action_proposer 唯一关键 +75pp**
  - Phase 4: V4+propose 5 game mean change_rate **82%**(+18pp vs baseline),0/5 wins
- **方法学修正(用户 morning 提)**: 之前 "validated 6 modules" 用 proxy metric。用 PASS 真定义重审 Phase 4 trace,**发现 2 hidden bugs**:
  - **Hypothesis 无方向**: 0/794 步是 directional kind,全 align/match
  - **reasoning↔action 49% LLM 脱节**: 字母 mapping confusion,orch_override 0 次贡献
- **共同根因**: bench 用 sanitized prompt(固定 letter / 给定 hypothesis),production 用动态 shuffle / 自由 Reflection 输出 → **distribution shift**
- **下一步**: 等用户标 Module 1+4(`annotation_request.md`),然后试 `--validate-hypothesis-schema strict` + 固定 letter mapping
- **关键 outputs**:
  - `outputs/v4_phase4_g{1..5}_*/` 5 game runs
  - `outputs/v4_ablate_*/` Phase 3 ablation
  - `docs/project/2026-05-19-v0-v4_clean_baseline/module_validation.md` 重审报告
  - `docs/project/2026-05-19-v0-v4_clean_baseline/annotation_request.md` 待用户填

---

### 2026-05-18 — 🟢 goal_judge A/B — `docs/project/2026-05-18-v0-goal_judge_ab/`

- **分支**: `feat-2026-05-18-v0-det_goal_plus_force_cot`(本地,复用)
- **关键 commit**: `932dcef`(parser 扩展 + LLM judge + 全套 bench 代码)+ 本次(报告)
- **一句话**: 反思关键 step "is hypothesis achieved?" 对比 Python parser vs LLM judge,**parser 83% > LLM 68%,recall on TRUE 100% vs 22%,速度 1.25M×**。production 反思闭环用 parser-only。
- **关键 outputs**:
  - `outputs/bench_goal_judges_20260518-160438/metrics.json`
  - `outputs/bench_goal_judges_20260518-160438/per_probe.jsonl`(200 行 side-by-side)
- **下一步**: 用扩展 parser 跑 ar25 1×2×100,看 `[GOAL CHECK]` alert 是否真触发 + 任 1 round 通关

---

### 2026-05-18 — 🟢 det_goal_plus_force_cot + 交叉验证 — `docs/project/2026-05-18-v0-det_goal_plus_force_cot/`

- **分支**: `feat-2026-05-18-v0-det_goal_plus_force_cot`(本地)
- **关键 commits**: `456f317`(集成) → `f894d71`(v1 报告) → 本次(v2 + 交叉验证)
- **一句话**: ar25 2 round × 100 step,首次 change_rate 破 70%(v1 75% / v2 47%),但仍 **0 wins**。**核心发现:5 subtask PASS ≠ production wins;真正主导的是 orchestrator hard rules,不是 LLM 推理**;**T-SEL-1 production 0% (vs bench 70%) 暴露 selector 是真瓶颈**
- **关键 outputs**:
  - `outputs/det_goal_force_cot_ar25_2x100_20260518-025313/`(v1,reflection_tokens=250 被截断)
  - `outputs/det_goal_force_cot_v2_ar25_2x100_20260518-090129/round_00/`(v2 round 0,reflection_tokens=2048,带 play.gif)
- **架构含义**: 下一个分支放弃补 evaluator,改测 **T-REVISE**(Reflection 是否会 evidence-driven 修目标)

---

### 2026-05-18 02:00 — 🟢 force_cot A/B — `docs/project/2026-05-18-v0-force_cot/`

- **分支**: `feat-2026-05-17-v0-subtask-T-NAV-1`(复用)
- **关键 commits**: `01fc61b`(`--prompt-style` flag)→ `644e604`(决策树)→ 本次 commit(报告 + sync)
- **一句话**: 同 5 subtask × 100 probe,只换 user prompt 加 step-by-step → T-NAV-3 67→**97% PASS** (+30pp);T-GOAL 30→35 但 long_acc 33.7% **证伪 LLM 能做 goal recognition**;T-SEL-1 78→70 **倒退**
- **结果**:

  | Subtask | Default | force_cot | Δ | 终判 |
  |---|---:|---:|---:|---|
  | T-NAV-1 | 52% | 71% | +19pp | FAIL |
  | T-NAV-2 | **95% PASS** | 93% | -2pp | PASS |
  | T-NAV-3 | 67% | **97%** | **+30pp** | **PASS** ✅ |
  | T-SEL-1 | 78% | 70% | -8pp | WORSE |
  | T-GOAL | 30% | 35% | +5pp | SEVERE FAIL |

- **关键 outputs**:
  - `outputs/subtask_batch_20260518-000721/`(跨 subtask 汇总,主入口)
  - `outputs/subtask_T-{NAV-1,NAV-2,NAV-3,SEL-1,GOAL}_20260518-000721/` × 5
  - `outputs/bench_subtask_batch_v2_forcecot.log`
- **架构含义**: Reflection goal-check 必须迁 deterministic Python(LLM 能力到顶);Action Agent prompt 可加 force_cot;不能一刀切
- **下一步**: 在新分支 `feat-2026-05-18-v0-det_goal_plus_force_cot` 实现 `goal_evaluator.py` + 加 Action force_cot,跑 ar25 1×2×100

---

### 2026-05-17 17:50 — 🟢 5 subtask 验证完成 — `docs/project/2026-05-17-v0-subtask_decomp/`

- **分支**: `feat-2026-05-17-v0-subtask-T-NAV-1`(当前)
- **关键 commits**: `8f1d314`(T-NAV-1 + /think fix)+ 本次写 5 个 report + cross-summary
- **一句话**: 5 subtask × 100 probe SmolLM3-CoT 实跑。**T-NAV-2 95% PASS** 是唯一通过的;**T-GOAL 30% SEVERE FAIL** 是 0 通关疑似根因。CoT activation 率(0-100% 因任务而异)= Pass/Fail 决定因素。`/think` flag 不能强制激活,只能避免截断已激活的 chain
- **结果一览**:

  | Subtask | acc | CoT activation | 判定 |
  |---|---:|---:|---|
  | T-NAV-1 单步方向 | 52.0% | 11/100 | FAIL |
  | T-NAV-2 多步同方向 | **95.0%** | **100/100** | ✅ PASS |
  | T-NAV-3 L 型双段 | 67.0% | 1/100 | FAIL |
  | T-SEL-1 ACTION6 click | 78.0% | 86/100 | FAIL (borderline) |
  | T-GOAL YES/NO 目标 | **30.0%** | **0/100** | SEVERE FAIL |

- **关键 outputs**:
  - `outputs/subtask_batch_20260517-163018/`(跨 subtask 汇总,主入口)
  - `outputs/subtask_T-{NAV-1,NAV-2,NAV-3,SEL-1,GOAL}_20260517-163018/` × 5(每个 subtask metrics + per_probe + summary)
  - `outputs/subtask_T-NAV-1_20260517-161834/`(v0 baseline,无 /think)
  - `outputs/bench_subtask_batch_v1_think.log`(batch 运行日志)
  - `outputs/bench_subtask_T-NAV-1_v2.log`(v0 运行日志)
- **下一步**: P0 - 在 user prompt 加 "Let's solve step by step" 强制 CoT;P1 - 把 production T-GOAL 改成 deterministic Python(不让 LLM 做 goal recognition)

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

## 8. 3 分钟接管路径

刚加入项目?按这个顺序读:

1. 本文(项目总览 + 当前进展)
2. [`tonight_summary.md`](./tonight_summary.md)(2026-05-16 跨分支汇报)
3. [`CONVENTIONS_zh.md`](./CONVENTIONS_zh.md)(文档规范)
4. [`project/2026-05-17-v0-model_bench/report_5game.md`](./project/2026-05-17-v0-model_bench/report_5game.md)(最新 5×2×300 + SmolLM3 结果)
5. [`project/2026-05-14-v3_2-double_agent/architecture.md`](./project/2026-05-14-v3_2-double_agent/architecture.md)(主线设计)
6. 不懂的术语 → [`GLOSSARY_zh.md`](./GLOSSARY_zh.md)

---

## 9. 关键代码文件(grep 入口)

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

## 10. 维护

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
