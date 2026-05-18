# 项目术语表 (GLOSSARY_zh.md)

> 项目里出现过的所有术语 / 缩写 / 概念都收录在这里。**每个术语只在这里有定义,其他文档用 `[[term]]` 链接过来**。

最近更新: 2026-05-16

按字母排序。状态码:🟢 当前活的;🟡 参考 / 历史但被引用;⚫ 已弃用。

---

## A

### `Action Agent`

🟢 v3.2 双 agent 架构里负责「每步选 action」的 agent。读 perception + Knowledge + Reflection alert,输出 `reasoning: ... action: ACTIONx`(两行)。实现在 `arc_agent/agents/action_agent.py`。

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §5。
**相关**: [[Reflection Agent]]、[[Knowledge]]、[[ActionAgent.choose]]。

### `action_semantics`

🟢 `Knowledge` 字段。`dict[str, str]`,记录每个 ACTION 在当前游戏里的效果。例:`{"ACTION1": "moves the yellow 1x1 (obj_002) UP by 3 cells"}`。Reflection 写,Action 读。

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §3。
**相关**: [[Reflection Agent]]、[[BUG-2]]、[[BUG-9]]。

### `ACTION1..ACTION7`

🟢 ARC-AGI-3 的游戏动作枚举(`arcengine.GameAction`)。
- `ACTION1-4`:Up/Down/Left/Right(具体方向因游戏而异)
- `ACTION5`:Primary(interact / select / 等)
- `ACTION6`:Coordinate,需要 `(x, y) ∈ [0, 63]`,唯一 `is_complex()` 的 action
- `ACTION7`:Undo

**出处**: `TASK_OVERVIEW.md` + `CLAUDE.md`。

### `ar25` / `bp35` / `cd82` / `cn04` / `dc22`

🟢 我们目前实测用的 5 个公开 demo 游戏 ID 的短前缀。完整 ID 形如 `ar25-0c556536`。**ar25 是默认基准**(最多代码引用)。

**出处**: `data/splits/demo_555.json`。

---

## B

### `BUG-1..BUG-13`

🟡 v3.2 实施过程中的 13 个已知 bug,在 [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §13 集中列出。
- BUG-1: Action reasoning 91% 输出 N/A(没具体说预期)
- BUG-2: `action_semantics` 多面效果被覆写
- BUG-3: 相同 semantic 反复重写
- BUG-4: ACTION6 坐标没 memory(被 BUG-10 完整覆盖)
- BUG-5: round 撞 max_actions 提早结束(已修)
- BUG-6: Reflection 输出无 per-claim 置信度
- BUG-7: C stuck alert 阈值偏高
- BUG-8: `goal_hypothesis` 直接覆盖(已修,加 `rejected_goals`)
- BUG-9: `action_semantics` 写"an active object"无主语(已修)
- BUG-10: ACTION6 无 per-object 置信度(已修,引入 `click_targets`)
- BUG-11/12/13: click_targets 跨 round 持久化 + canonical rules + eager Reflection

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §13.

---

## C

### `change_rate`

🟢 一个 round / run 内 `frame_changed=True` 步数占总步数的百分比。**核心健康指标**。

- v3.2 baseline (mask off): 3-4%
- v2 canary (3×30 mask strict): 60-100%
- 当前 mask strict 3×200: 5-8%

**出处**: 几乎所有实验报告。

### `click_targets`

🟢 v3.2 BUG-10 引入的 ACTION6 命名 target bandit。把 ACTION6 从「4096 像素盲选」压缩为「~10 个 named target 的 confidence × tries 优先级」。实现 `arc_agent/click_targets.py`,prompt 块 `[CLICK TARGETS]`。

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §13.2 BUG-10。
**相关**: [[BUG-10]]、[[ACTION6]]。

### `CNN-small`

🟢 [[Predictor v0]] 的第四个架构。17 channel(16 色 one-hot + 1 action) × 64 × 64 → 32 → 64 → 64 → 1。~80K 参数。val AUC 0.894,比 hand-feature MLP +6 pp。

**出处**: [`reports/predictor_v0.md`](../outputs/reports/predictor_v0.md)。

### `CoT (Chain-of-Thought)`

🟢 推理模式 — 模型在 final answer 前先生成一段 reasoning chain(`<think>...</think>` for Phi-4-reasoning,默认 mode for SmolLM3)。我们用 [[bench_models_spatial]] 实测:SmolLM3 CoT **72.4%** accuracy vs `/no_think` **55.2%** —— **17pp 差距全在 CoT chain 里**。生产太慢(~26s/step),只在 bench 用。

**出处**: [`project/2026-05-17-v0-model_bench/report.md`](./project/2026-05-17-v0-model_bench/report.md)。
**相关**: [[/no_think]]、[[/think]]、[[CoT 激活率]]、[[SmolLM3-3B]]、[[reasoning_mode]]。

### `CoT 激活率` (CoT activation rate)

🟢 **单个 probe / step 上 CoT chain 是否真的被生成的比例**。具体度量(目前用):**probe 的 `elapsed_s > 10s`** —— SmolLM3 短路径直接答 Answer: X 通常 ≤ 1.5s,激活 CoT 链一般 20-40s,**阈值放 10s 即可区分**。

**为什么重要**: [[subtask probe]] 实测 5 subtask × 100 probe,**激活率几乎单调决定 accuracy**:

| Subtask | 激活率 | accuracy |
|---|---:|---:|
| T-GOAL | 0/100 | 30% (≈随机) |
| T-NAV-3 | 1/100 | 67% |
| T-NAV-1 | 11/100 | 52%(激活时 90.9%) |
| T-SEL-1 | 86/100 | 78% |
| T-NAV-2 | 100/100 | 95% ✅ |

**关键性质**:
1. **任务驱动,非系统 flag 驱动** — 同一 model + 同一 prompt 配置下,不同 subtask 激活率从 0% 到 100% 不等;模型按问题"看起来需不需要算"自行决定
2. **`/think` 不能强制激活** — T-NAV-1 加/不加 `/think` 都是 11/100;`/think` 只能避免已激活的 chain 在 1024 tokens 处被截
3. **激活了几乎都对** — 5 subtask 里激活子集 acc 普遍 79-95%,所以问题不是"推理弱",而是"该推理时没推理"

**用法**: 拿 per_probe.jsonl 后 `awk '$elapsed_s>10'` 看激活率;低 (< 50%) 就该改 prompt 而不是改模型。

**Production fix 思路**: user prompt 加 `"Let's solve step by step. Show your reasoning."` 把激活率推到 80%+;或把 "判断" 类任务(如 [[T-GOAL]])改成 deterministic Python,绕开 LLM 推理。

**出处**: [`project/2026-05-17-v0-subtask_decomp/report.md`](./project/2026-05-17-v0-subtask_decomp/report.md) §4,5 个 [[subtask probe]] report。
**相关**: [[CoT]]、[[/think]]、[[/no_think]]、[[reasoning_mode]]、[[subtask probe]]、[[短路径]]、[[elapsed_s]]、[[per_probe.jsonl]]、[[CoT 截断]]。

### `短路径` (short-path)

🟢 模型 **跳过 [[CoT]] chain 直接给 Answer: X** 的行为(SmolLM3 / Phi-4-mini-reasoning 等双模型一致)。表现是 `<think>...</think>` 区段为空或几乎为空,生成只有 1-3 个 token,耗时 ≤ 1.5s。

**对照 (5 subtask × 100 probe SmolLM3)**:

| 模式 | elapsed_s | tokens 生成数 | accuracy 范围 |
|---|---|---|---|
| 短路径 | ≤ 1.5s | < 30 tokens | 30-71%(看任务) |
| 激活 CoT | 20-40s | 300-1000 tokens | 79-95% |

**为什么模型选短路径**: 它把 prompt 模式识别成 "multiple choice letter pattern" 而非 "推理题",尤其是 4 选 1 中有显眼的不正确选项时(distractor 太明显)。

**怎么对抗**: user prompt 加 "Let's solve step by step. Show your work." 或题目结尾改 "Show your reasoning, then end with Answer: X"。

**出处**: [`project/2026-05-17-v0-subtask-T-NAV-1/report.md`](./project/2026-05-17-v0-subtask-T-NAV-1/report.md) §4、[`project/2026-05-17-v0-subtask_decomp/report.md`](./project/2026-05-17-v0-subtask_decomp/report.md) §4.1。
**相关**: [[CoT]]、[[CoT 激活率]]、[[long_acc / short_acc]]、[[elapsed_s]]。

### `long_acc / short_acc`

🟢 把 [[per_probe.jsonl]] 按 [[elapsed_s]] 是否 > 10s 二分,分别算 accuracy:
- `long_acc` = "激活了 CoT 的 probe 子集" 的正确率
- `short_acc` = "走 [[短路径]] 的 probe 子集" 的正确率

**用法 (诊断)**: 整体 accuracy 低有两种原因 ——
1. `long_acc` 低 → 模型推理能力不够,**该换模型**
2. `long_acc` 高 + [[CoT 激活率]] 低 → 模型有能力但没用,**该改 prompt**

**实测例 (T-NAV-1 v1)**: 整体 52%,但 `long_acc = 90.9%`(n=11),`short_acc = 47.2%`(n=89) → 第 2 种情况,problem 是没激活。

**怎么算 (Python)**:
```python
import json
rows = [json.loads(l) for l in open("per_probe_smollm3-cot.jsonl")]
long  = [r for r in rows if r["elapsed_s"] > 10]
short = [r for r in rows if r["elapsed_s"] <= 10]
long_acc  = sum(r["ok"] for r in long)  / max(1, len(long))
short_acc = sum(r["ok"] for r in short) / max(1, len(short))
```

**出处**: 5 个 subtask report §3、`docs/project/2026-05-17-v0-subtask_decomp/report.md` §4.1。
**相关**: [[CoT 激活率]]、[[per_probe.jsonl]]、[[elapsed_s]]、[[短路径]]。

### `elapsed_s`

🟢 [[per_probe.jsonl]] 每行的字段,**单次 model.generate() 调用的 wall time(秒)**。`scripts/bench_subtask.py` 在每条 probe 推理外面 `time.time()` 包一层量出来。

**用途**:
1. 区分 [[短路径]] vs 激活 CoT:阈值 10s(实测 SmolLM3 中间区基本空,要么 ≤ 1.5s 要么 ≥ 20s)
2. 估算 GPU 吞吐 (`total_s / n_probes`)
3. 诊断 max_new_tokens 设置 —— 若 elapsed_s 接近"1024 tokens / 模型 tps"上限,大概率发生了 [[CoT 截断]]

**阈值参数说明**: 10s 是 SmolLM3-3B 4-bit 在 RTX A4500 上的经验值;**换 GPU / 模型 / quant 都要重测**。

**出处**: `scripts/bench_subtask.py:166-186` (timing 代码)。
**相关**: [[per_probe.jsonl]]、[[CoT 激活率]]、[[短路径]]、[[CoT 截断]]。

### `per_probe.jsonl`

🟢 [[subtask probe]] 和 [[spatial probe]] bench 的**逐 probe 输出文件**。每行一个 JSON object,无序无 schema 强约束,字段:

| 字段 | 含义 |
|---|---|
| `id` | probe ID (如 "T-NAV-1-0042") |
| `correct` | ground-truth 字母 (A/B/C/D) |
| `guessed` | 模型选的字母,parse 失败为 `null` |
| `ok` | bool, 是否选对 |
| `elapsed_s` | 推理 wall time,秒;见 [[elapsed_s]] |
| `raw_tail` | 最后 200 chars,用来事后诊断 |

**文件名格式**: `per_probe_<model_key>.jsonl`(如 `per_probe_smollm3-cot.jsonl`)。**多 model 并跑时一个 model 一个文件**。

**位置**: `outputs/subtask_<TASK>_<ts>/per_probe_<model>.jsonl` 或 `outputs/bench_<ts>/per_probe_<model>.jsonl`。

**配合**: 同目录还有 `metrics.json` (汇总 accuracy / load_s / total_s) + `summary.md` (markdown 表格)。

**出处**: `scripts/bench_subtask.py:228-231`、`scripts/bench_subtask_batch.py:71-74`。
**相关**: [[subtask probe]]、[[spatial probe]]、[[long_acc / short_acc]]、[[elapsed_s]]、[[model_bench]]。

### `CoT 截断` (CoT truncation)

🟢 模型生成的 [[CoT]] chain **超过 `max_new_tokens` 上限,被 generate() 强制截断**;表现是 `<think>...</think>` 没闭合或末尾没有 `Answer: X` → parse 出 `guessed=null`。

**实测影响**: T-NAV-1 v0 用 `max_new_tokens=512`,有些 long-CoT response 卡在 chain 中间(`raw_tail` 显示推理还在继续就被切了)。提高到 1024 后:
- T-NAV-1 [[long_acc]] 54.5% → 90.9%(同一批 11 个激活样本,把"被截的"救活变正确)
- T-NAV-1 整体 acc 48% → 52%(因为大头是 [[短路径]] 没被截断的)

**诊断**: per_probe.jsonl 里 `guessed == null` 的 probe 多半就是这种情况;或者 [[elapsed_s]] 顶到"1024 tokens / 模型 tps"上限 (~38s for SmolLM3 4-bit) 就要警惕。

**设置建议**:
- bench: `max_new_tokens = 1024`(留余量给 think chain)
- production: 256-512,因为 think chain 在 production prompt 里通常被 `/no_think` 关掉

**出处**: [`project/2026-05-17-v0-subtask-T-NAV-1/report.md`](./project/2026-05-17-v0-subtask-T-NAV-1/report.md) §7 (v0 vs v1 对照表)。
**相关**: [[CoT]]、[[CoT 激活率]]、[[/think]]、[[long_acc / short_acc]]、[[elapsed_s]]。

### `CausalLMBackbone`

🟢 `arc_agent/vlm_backbone.py` 中的类,wrap 任意 `AutoModelForCausalLM`(SmolLM3 / Phi-4 / DeepSeek-R1-Distill)成 [[VLMBackbone]] 接口给 ActionAgent 用。带 `reasoning_mode` 参数(`auto` / `cot` / `no_think`)。

**出处**: `arc_agent/vlm_backbone.py`(2026-05-17 加)。
**相关**: [[HFBackbone]](Qwen-VL 用)、[[make_backbone]]、[[reasoning_mode]]。

### `Candidate`

🟢 `arc_agent/action_proposer.py` 的 dataclass:`letter` + `action_name` + `coords` + `reason`。Code propose K=3 个候选,Qwen N 选 1,letter 顺序 shuffled 避免位置偏好。

**出处**: [[action_proposer]]。
**相关**: [[propose K=3]]、[[N 选 1]]。

---

## D

### `demo_555`

🟢 5+5+5 = 15 个游戏的固定划分(5 baseline + 5 train + 5 val),冻结在 `data/splits/demo_555.json`,**不允许重新随机**。

**出处**: `arc_agent/eval_split.py`、commit `2026-05-11`。

---

## F

### `F1 (intrinsic reward)`

🟡 [[RL v0]] 的内在奖励:agent 预测 `predicted_changes`,跟真实 `real_changes` 算 F1 ∈ [0, 1]。实现 `arc_agent/rewards.py:verify_prediction_f1`。**当前 v3.2 不用 F1**,被 parked 路线。

**出处**: [`architecture/rl_v0_zh.md`](./architecture/rl_v0_zh.md) §3。
**相关**: [[GRPO]]、[[parked]]。

### `frame_changed`

🟢 trace.jsonl 每行的核心 boolean:`grid_t != grid_{t+1}`?决定一步是 no-op 还是真正改变了状态。

**出处**: `arc_agent/observation.py`、所有 trace 写入处。

### `5×2×300`

🟢 我们的「跨游戏完整评测」标准配置:G_base 5 个 game × 2 round × max 300 step。**最早**(2026-05-16 提出),**首次完整跑** 2026-05-17 SmolLM3,wall clock 2.88h。Qwen 同配置数据**仍缺失**(只有 3×30 smoke 和被 kill 的部分跑)。

**出处**: `scripts/run_action_proposer_5game.py`、[`project/2026-05-17-v0-model_bench/report_5game.md`](./project/2026-05-17-v0-model_bench/report_5game.md)。

---

## G

### `Gate (G1, G2, ...)`

🟢 决策门。每份架构设计在 §6.2 列出实验需通过的 G1/G2/... 条件。失败 → 走对应 fallback 而不是继续推进。

**出处**: [`CONVENTIONS_zh.md`](./CONVENTIONS_zh.md) §3。

### `goal_hypothesis` / `goal_confidence`

🟢 `Knowledge` 字段。Reflection 推断的「这个游戏的目标」一句话假设 + 置信度 `low|medium|high`。

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §3。
**相关**: [[BUG-8]]、[[rejected_goals]]。

### `GRPO`

🟡 Group Relative Policy Optimization。RL 训练算法。[`architecture/grpo_v0_zh.md`](./architecture/grpo_v0_zh.md) 定位为**单游戏诊断实验**(只 ar25),不用于 Kaggle 提交。

**出处**: TRL 库,[`architecture/rl_v0_zh.md`](./architecture/rl_v0_zh.md) §6。

---

## H

### `Hungarian (object aligner)`

🟢 跨帧 object 匹配。`arc_agent/object_aligner.py` 用匈牙利算法匹配 frame_t / frame_{t+1} 的 [[ObjectRecord]],保持 obj_id 稳定。

**出处**: [`reference/object_pipeline_zh.md`](./reference/object_pipeline_zh.md)。

---

## K

### `Knowledge`

🟢 v3.2 核心 dataclass。**跨 round 持久化**的"学到的东西":`action_semantics` / `goal_hypothesis` / `rules` / `failed_strategies` / `rejected_goals` / `click_targets` / `current_alert` / `rounds_played` / `rounds_won` / `round_history`。 实现 `arc_agent/knowledge.py`。

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §3。
**相关**: [[Action Agent]]、[[Reflection Agent]]、所有 BUG-X。

---

## M

### `make_backbone`

🟢 `arc_agent/vlm_backbone.py` 工厂函数。根据 model_path 自动 dispatch:Qwen2.5-VL-* → `HFBackbone`,其它 → `CausalLMBackbone`。接受 `reasoning_mode` 参数。

**出处**: `arc_agent/vlm_backbone.py`(2026-05-17 加)。

### `model_bench` (v0)

🟢 `2026-05-17-v0-model_bench` project。在 28 道 [[spatial probe]] 上 sequential bench 3 候选 model:Qwen2.5-VL-3B / Phi-4-mini-reasoning / SmolLM3-3B。winner: SmolLM3 CoT 72.4%。但 `/no_think` 模式 SmolLM3 = 55.2% = Qwen。

**出处**: [`project/2026-05-17-v0-model_bench/`](./project/2026-05-17-v0-model_bench/)、`scripts/bench_models_spatial.py`。

### `mask` / `R2 mask`

🟢 [[R2]] orchestrator-级硬规则。在 `env.step` 前拦截 LLM 选的 action,如果该 action 已经 `n_tried ≥ 5 AND n_changed == 0` → 替换。实现 `arc_agent/action_mask.py`。

- `compute_action_mask`:决定哪些 action 该 block
- `apply_action_mask`:决定替换成什么(优先级:untried > known-good > random)

**注意**:2026-05-16 发现「替换后 OutcomeLog 归因 bug」并已修(commit `01a7227`)。

**出处**: [`reference/v3_2_hardrules_results_zh.md`](./reference/v3_2_hardrules_results_zh.md)、commit `1bac4be`。

### `MLP-S / MLP-L`

🟢 [[Predictor v0]] 的两个 MLP 架构。S = 128 hidden,L = 512 hidden。val AUC 0.834 / 0.830,均饱和在 hand-feature 信号。

**出处**: [`reports/predictor_v0.md`](../outputs/reports/predictor_v0.md)。

---

## N

### `no_op_streak`

🟢 连续多少步 `frame_changed=False`。健康值 < 5,异常 > 20。`ActionAgent.no_op_streak()` 返回当前值。

**出处**: `arc_agent/agents/action_agent.py`。

---

## O

### `ObjectMemory`

🟢 v3 引入。一个 episode 内 UID-keyed 的 object 追踪。实现 `arc_agent/object_tracker.py:ObjectMemory`。每个 round reset。

**出处**: [`architecture/v3_zh.md`](./architecture/v3_zh.md) §5。

### `ObjectRecord`

🟢 一个 object 的描述:`bbox` / `mass` / `color` / `state ∈ {STATIC, ACTIVE, TEXTURE, CANDIDATE}` 等。`arc_agent/object_extractor.py` 用 `scipy.ndimage.label` 抽出来。

**出处**: [`architecture/v3_zh.md`](./architecture/v3_zh.md) §5、[`reference/object_pipeline_zh.md`](./reference/object_pipeline_zh.md)。

### `orch_override`

🟢 trace.jsonl 字段。当 [[R2 mask]] 替换了 LLM 的选择时,记录替换原因(如 `"untried ACTION5 over masked ACTION6"`)。空字符串表示没替换。

**出处**: `scripts/run_v3_multi_round.py`。

### `OutcomeLog`

🟢 一个 round 内所有 step 的 `(action, frame_changed, primary_direction, ...)` 记录。Mask 用它判断 `n_tried` / `n_changed`。Round 之间 reset。

**出处**: `arc_agent/action_inference.py`。

---

## P

### `parked`

🟡 状态标记:本路线代码留着 + 测试通过,但**当前不主动迭代**。例如 RL/GRPO/SFT 都被 park 过。要复活先 promote。

**出处**: [`CONVENTIONS_zh.md`](./CONVENTIONS_zh.md) §6。

### `perception (deterministic)`

🟢 v3 决策:**视觉用算法,不用 LLM**。scipy.ndimage.label + temporal_classifier + Hungarian aligner。源于 [`reference/object_pipeline_zh.md`](./reference/object_pipeline_zh.md) 实测 Qwen-VL 在 ar25 上 ~0% 而 scipy 100%。

**出处**: [`architecture/v3_zh.md`](./architecture/v3_zh.md) §0。

### `Phi-4-mini-reasoning`

❌ Microsoft 3.8B 模型,公开 ARC-Challenge 83.7%。我们实测 [[model_bench]] **41.4%** —— **低于 Qwen baseline 55.2%**。原因:`<think>` reasoning chain 在 multi-choice 算术题上经常走偏 / 被 512 token 截断。**不选用**。

**出处**: [`project/2026-05-17-v0-model_bench/report.md`](./project/2026-05-17-v0-model_bench/report.md) §4。

### `propose / proposer (action_proposer v0)`

🟢 `arc_agent/action_proposer.py`。代码 propose K=3 [[Candidate]](slot 1: untried;slot 2: known-good 排除 over-committed;slot 3: click_target 或 random)。Qwen N 选 1,letter shuffle 避免位置偏好。

**出处**: [`project/2026-05-16-v0-action_proposer/`](./project/2026-05-16-v0-action_proposer/)。
**相关**: [[N 选 1]]、[[Candidate]]。

### `Predictor v0`

🟢 frame-change 预测器。给 `(state, action)` 输出 `P(frame_change)`。四个架构对比:LogReg / MLP-S / MLP-L / CNN-small。val AUC 0.80-0.89。

**出处**: [`architecture/predictor_v0_zh.md`](./architecture/predictor_v0_zh.md)、[`reports/predictor_v0.md`](../outputs/reports/predictor_v0.md)。

### `primary_direction` / `primary_distance`

🟢 一步 env.step 的主方向(UP/DOWN/LEFT/RIGHT)+ 距离。由 orchestrator 通过 `object_aligner` 计算。trace.jsonl 字段。

**出处**: `scripts/run_v3_multi_round.py:_compute_primary_change`。

---

## R

### `reasoning_mode`

🟢 `arc_agent/vlm_backbone.py` 的 `CausalLMBackbone` 参数:
- `auto` — 模型默认(SmolLM3: `no_think`,生产 ~3-4s/step)
- `cot` — 强制 chain-of-thought,准但慢(~26s/step)
- `no_think` — 显式注入 `/no_think` (SmolLM3)

CLI: `scripts/run_v3_multi_round.py --reasoning-mode {auto,cot,no_think}`(2026-05-17 加)。

**出处**: 同 [[CausalLMBackbone]]。

### `/no_think`

🟢 SmolLM3 的系统 flag。`/no_think` 前置到 system prompt 让模型跳过 `<think>...</think>` chain,直接答。**生产模式必须开**(不然 22h 跑 5×2×300)。但实测 accuracy 从 72.4% 掉到 55.2% (= [[Qwen-VL-3B]] baseline) —— **空间推理优势全在 CoT chain 里**。

**出处**: [`project/2026-05-17-v0-model_bench/report.md`](./project/2026-05-17-v0-model_bench/report.md)、HF SmolLM3-3B docs。
**相关**: [[CoT]]、[[reasoning_mode]]、[[SmolLM3-3B]]。

### `/think`

🟢 SmolLM3 的另一个系统 flag,显式打开 reasoning chain。**`bench_subtask.py:generate()` 在 reasoning_mode='cot' 时必须显式注入 `/think`**(2026-05-17 16:00 修复)。

**重要修正(2026-05-17 17:50)**: 注入 `/think` **不能提升 [[CoT 激活率]]**(T-NAV-1 注入前后都是 11/100),只能让已激活的 chain 在 1024 tokens 内不被截断,从而把"激活时 accuracy"从 54.5% 提到 90.9%。整体 acc 仅 +4pp。**要强制激活,得改 user prompt(加 "Let's solve step by step"),不是改 system flag**。

**出处**: [`project/2026-05-17-v0-subtask-T-NAV-1/report.md`](./project/2026-05-17-v0-subtask-T-NAV-1/report.md) §4 + §7。
**相关**: [[CoT 激活率]]、[[/no_think]]、[[CoT]]、[[reasoning_mode]]、[[subtask probe]]、[[SmolLM3-3B]]。

### `subtask probe`

🟢 把「通关」拆成 7 个原子可测子任务的合成 probe 集。每个 subtask 一个 `gen_*()` 函数,生成 100 道 4 选 1 题。当前实现: T-NAV-1(单步方向)、T-NAV-2(多步同方向计数)、T-NAV-3(轴切换两段路径)、T-SEL-1(bbox 内坐标 ACTION6)、T-GOAL(yes/no 判定目标达成)。每子任务一个 git 分支 + report,PASS 阈值 ≥ 90%(允许 LLM 失误率高但下游纠错)。

**出处**: [`project/2026-05-17-v0-subtask_decomp/architecture.md`](./project/2026-05-17-v0-subtask_decomp/architecture.md)、`arc_agent/subtask_probes/__init__.py`、`scripts/bench_subtask.py` / `bench_subtask_batch.py`。
**相关**: [[spatial probe]]、[[/think]]、[[T-NAV-1]]、[[T-NAV-2]]、[[T-NAV-3]]、[[T-SEL-1]]、[[T-GOAL]]。

### `cross-validation` (subtask 跨验证)

🟢 用 production trace.jsonl 反向验证 [[subtask probe]] 测的能力是否真在 production work。**2026-05-18 首次做,结论是混合的**:

| Subtask | Bench | Production | Δ |
|---|---:|---:|---:|
| T-NAV-1 | 71% | **93%** | +22pp 反而更好 |
| T-NAV-2 | 95% | ~OK(streaks 合理)| - |
| **T-SEL-1** | 70% | **0%** | **-70pp 崩** |
| T-GOAL | 35% | 0(evaluator parse=None)| - |
| Hyp-Action coherence | n/a | 80% | - |

**关键启示**:
1. bench 高 ≠ production work — T-SEL-1 70% → 0% 的代价
2. bench 低 ≠ production fail — T-NAV-1 71% bench 但 93% prod
3. **拆分本身的覆盖**才是真问题:5 subtask 全在测 "given target → execute",**完全没测 "from frame → hypothesize → revise"**

**做法** (`scripts/cross_validate_subtasks.py` 待建): 读 trace.jsonl,逐步重建 "如果 subtask probe 形态,这一步会答对吗",汇总成跨验证表。

**出处**: [`project/2026-05-18-v0-det_goal_plus_force_cot/report.md`](./project/2026-05-18-v0-det_goal_plus_force_cot/report.md) §5.4。
**相关**: [[subtask probe]]、[[T-NAV-1]]、[[T-SEL-1]]、[[T-REVISE]]、[[T-DISCOVER]]。

### `T-REVISE` (proposed)

🟢 **新 [[subtask probe]]** — 测 Reflection Agent 的核心能力:**根据 outcome 修目标**。

**输入**: (prior hypothesis, last N (action, outcome) pairs)
**问**: hypothesis 是否被证伪?如果是,新 hypothesis 应该是什么方向?

设计动机: ARC-AGI-3 没有 instruction,**通关靠 trial → reflection → revise**。5 个旧 subtask **没测这一步**。v2 round 0 trace 显示 Reflection 100 步写了 11 个不同 hypothesis,但**没人验证这 11 次 revise 是 evidence-driven 还是随机抖**。

**Status**: 待建 (`feat-2026-05-18-v0-revise_bench` 分支)。

**相关**: [[cross-validation]]、[[T-DISCOVER]]、[[goal_hypothesis]]、[[Reflection Agent]]。

### `T-DISCOVER` (proposed)

🟢 **新 [[subtask probe]]** — 测 Reflection Agent 的首步推理能力:**从单帧 0 prior 生成第一个 hypothesis**。

**输入**: 一帧 + extract 出来的 objects + (legal actions)
**问**: 写一个合理的 goal_hypothesis,理由要可解释。

跟 [[T-REVISE]] 互补:一个测"猜",一个测"改"。

**Status**: 待建。

**相关**: [[T-REVISE]]、[[Reflection Agent]]、[[goal_hypothesis]]。

### `force_cot`

🟢 user-prompt 工程手段:在题目末尾换 "Solve step by step. First write down values, then check each option. End with Answer: X"。配合 [[/think]] 把 [[CoT 激活率]] 推到 56-100%(原本 0-100% 任性分布)。

**实测 A/B (2026-05-18,SmolLM3-3B 同 seed)**:

| Subtask | default | force_cot | Δ |
|---|---:|---:|---:|
| T-NAV-3 | 67% | **97%** | **+30pp ✅ PASS 转化** |
| T-NAV-1 | 52% | 71% | +19pp(仍 FAIL)|
| T-NAV-2 | 95% | 93% | -2pp(噪声)|
| T-GOAL | 30% | 35% | +5pp(long_acc 33.7%,LLM 做不了)|
| T-SEL-1 | 78% | **70%** | **-8pp ⚠️ 倒退** |

**结论**: 不是 free lunch。**对推理类 (T-NAV-3) 大赢,对模式识别类 (T-GOAL) 救不动,对坐标对比类 (T-SEL-1) 反而干扰**。production 改架构时需要按任务类型选择性应用。

**实现**: `scripts/bench_subtask.py:format_probe(prompt_style="force_cot")`,commit `01fc61b`。

**出处**: [`project/2026-05-18-v0-force_cot/report.md`](./project/2026-05-18-v0-force_cot/report.md)。
**相关**: [[CoT 激活率]]、[[/think]]、[[long_acc / short_acc]]、[[短路径]]、[[subtask probe]]。

### `T-NAV-1`

🟢 [[subtask probe]] 之一: **单步方向选择**。给定 `(r1,c1) → (r2,c2)`,target 共行或共列,delta ∈ ±[1..6]。4 选 1 from {ACTION1=UP, ACTION2=DOWN, ACTION3=LEFT, ACTION4=RIGHT}。

**实测**:
- SmolLM3 default (无 `/think`): 48%
- SmolLM3 default + `/think` + 1024 tokens: **52% FAIL**,激活率 11/100,long_acc 90.9%
- SmolLM3 [[force_cot]] + `/think`: **71% FAIL**,激活率 100/100,long_acc 71.0% → **即使全 CoT 模型也破不了 80%**,letter-shuffle attention 是真瓶颈

**出处**: `arc_agent/subtask_probes/__init__.py:gen_T_NAV_1`、[`project/2026-05-17-v0-subtask-T-NAV-1/report.md`](./project/2026-05-17-v0-subtask-T-NAV-1/report.md)、[`project/2026-05-18-v0-force_cot/report.md`](./project/2026-05-18-v0-force_cot/report.md)。

### `T-NAV-2`

🟢 [[subtask probe]] 之一: **多步同方向计数**。给 object + target 共行或共列,distance n ∈ [2..10],已知 action,问几次。4 选 1: 正确 n times + 3 个 distractor。

**实测**:
- SmolLM3 default: **95.0% PASS** ✅,CoT 激活率 100/100
- SmolLM3 [[force_cot]]: 93%(短链 91/100 反而拿到 97.8%,长链 9/100 只有 44.4%)→ **数数任务不需要 CoT,反而干扰**

**出处**: `arc_agent/subtask_probes/__init__.py:gen_T_NAV_2`、[`project/2026-05-17-v0-subtask-T-NAV-2/report.md`](./project/2026-05-17-v0-subtask-T-NAV-2/report.md)、[`project/2026-05-18-v0-force_cot/report.md`](./project/2026-05-18-v0-force_cot/report.md)。

### `T-NAV-3`

🟢 [[subtask probe]] 之一: **L 型双段路径**。给定 `Δrow ≠ 0` 且 `Δcol ≠ 0` 的两点,4 选 1: 正确组合 vs 只走单轴 vs off-by-one。

**实测**:
- SmolLM3 default: 67% FAIL,激活率 1/100
- SmolLM3 [[force_cot]]: **97.0% PASS** ✅ **+30pp**,激活率 56/100,long_acc 96.4% → **force_cot 路线最有说服力的胜利**

**出处**: `arc_agent/subtask_probes/__init__.py:gen_T_NAV_3`、[`project/2026-05-17-v0-subtask-T-NAV-3/report.md`](./project/2026-05-17-v0-subtask-T-NAV-3/report.md)、[`project/2026-05-18-v0-force_cot/report.md`](./project/2026-05-18-v0-force_cot/report.md)。

### `T-SEL-1`

🟢 [[subtask probe]] 之一: **ACTION6 click bbox 击中**。4 个 object 各有 bbox,要选哪个 `ACTION6 x=N y=M` 在 target bbox 内。

**实测**:
- SmolLM3 default: 78% FAIL,激活率 86/100
- SmolLM3 [[force_cot]]: **70% FAIL ⚠️ -8pp 倒退**,激活率 100/100,long_acc 70% → **CoT 链反而 distract 模型在坐标对比上**。force_cot 不适用此类任务

**P1 P fix**: 把题面 `(x, y)` 跟选项 `x=N y=M` 都改成 `(row, col)` / `row=R col=C`,消除约定混淆。

**出处**: `arc_agent/subtask_probes/__init__.py:gen_T_SEL_1`、[`project/2026-05-17-v0-subtask-T-SEL-1/report.md`](./project/2026-05-17-v0-subtask-T-SEL-1/report.md)、[`project/2026-05-18-v0-force_cot/report.md`](./project/2026-05-18-v0-force_cot/report.md)。

### `T-GOAL`

🟢 [[subtask probe]] 之一: **判定 goal 是否达成 (YES/NO)**。题面给 goal_hypothesis ("align two yellow squares vertically in left column") + 两个 object 当前位置,4 选 1 YES/NO + 原因。

**实测**:
- SmolLM3 default: 30% SEVERE FAIL,激活率 0/100
- SmolLM3 [[force_cot]]: **35% 仍 SEVERE FAIL**,激活率 **98/100**,**long_acc 33.7%** → **激活了 CoT 仍然 near-random**,LLM 本身做不了 goal recognition

**架构含义** (决定性证据): 不管怎么改 prompt,LLM 都答不对 T-GOAL → **必须** 把 goal-check 移到 deterministic Python (读 hypothesis + obj 坐标做断言)。

**疑似 0/5 通关的根因**: reflection agent 实际在做 T-GOAL,**它认不出 win state → 不触发 win**。

**出处**: `arc_agent/subtask_probes/__init__.py:gen_T_GOAL`、[`project/2026-05-17-v0-subtask-T-GOAL/report.md`](./project/2026-05-17-v0-subtask-T-GOAL/report.md)、[`project/2026-05-18-v0-force_cot/report.md`](./project/2026-05-18-v0-force_cot/report.md) §4.2。

### `R1..R7` (hard rules)

🟢 v3.2 orchestrator-级硬规则。**Prompt 只能劝,orchestrator 才能管**的设计原则下产生。
- **R1** Knowledge sentinel filter:拒绝 `"unknown"/"none"/"tbd"` 作为 goal_hypothesis
- **R2** Knowledge-driven action mask:见 [[mask]]
- **R3** ActionAgent 卡死强制 explore:`no_op_streak ≥ 5` 或 `state_revisit ≥ 5` → 选 untried
- **R4** 自相矛盾过滤:`rules_append` "ACTION_X no effect" 跟 action_semantics 矛盾 → drop
- **R5** failed_strategies 交叉污染过滤:goal_hypothesis 不能等于 failed_strategies 字面量
- **R6** action-described goal 过滤:goal_hypothesis 不能以 "ACTION_X" 开头
- **R7** [LOW-PRIORITY ACTIONS] prompt 块:把 mask 结果给 LLM 看

**出处**: commits `1bac4be` (R1+R2+R3)、`7a315c7` (R5)、`d66ce37` (R4+R6+R7)。
**详细对比**: [`reference/v3_2_hardrules_results_zh.md`](./reference/v3_2_hardrules_results_zh.md)。

### `Reflection Agent`

🟢 v3.2 双 agent 架构里负责「每步反思 + 更新 Knowledge」的 agent。读上一步 outcome + 当前 Knowledge,输出 incremental delta JSON。每步调一次(per-step,不是 per-round)。

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §4。

### `rejected_goals`

🟢 `Knowledge` 字段(BUG-8 修复时加的)。被 Reflection 改口替换掉的旧 `goal_hypothesis` 存这里(cap 10),Reflection 下次不会再提同样的目标。

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §13 BUG-8。

### `RHAE` (scoring)

🟢 ARC-AGI-3 评分:`S = min(1.0, h/a)²`。`h` = 第二好人类的 action 数,`a` = agent 的 action 数。Quadratic penalty。

**出处**: `TASK_OVERVIEW.md`。

### `RL v0`

🟡 早期 RL 设计(intrinsic F1 reward + GRPO)。**parked**;现在不跑训练。

**出处**: [`architecture/rl_v0_zh.md`](./architecture/rl_v0_zh.md)。

### `round` / `episode`

🟢 一次完整的 game play (reset → ... → WIN / GAME_OVER / max_actions)。v3.2 multi-round 调度:跨 round Knowledge 保留,ObjectMemory + OutcomeLog 每 round 重置。

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §1。

---

## S

### `scipy.ndimage.label`

🟢 v3 perception 的根:64×64 grid → 连通分量。比 Qwen-VL 在 ar25 上 100% vs ~0% 的对比由 [`reference/object_pipeline_zh.md`](./reference/object_pipeline_zh.md) 实测。

**出处**: `arc_agent/object_extractor.py`。

### `SmolLM3-3B`

🟢 **2026-05-17 实测候选 backbone**。HuggingFace 开源 3B,dual-mode reasoning。spatial probe **CoT 72.4%** / **`/no_think` 55.2%**。**5×2×300 在生产 `/no_think` 模式下 mean change_rate 64%**(vs Qwen baseline 5-8%),0/5 game 通关。但优势可能不在空间推理(等于 Qwen),而在 instruction following + action_proposer 协同。

**出处**: [`project/2026-05-17-v0-model_bench/`](./project/2026-05-17-v0-model_bench/)、HF `HuggingFaceTB/SmolLM3-3B`。
**相关**: [[/no_think]]、[[CoT]]、[[CausalLMBackbone]]、[[model_bench]]。

### `spatial probe` / `T1..T8`

🟢 28 道 4-multi-choice spatial reasoning 测试题,基于真实 game 机制(64×64 grid, ACTION1=UP, ACTION3=LEFT 等,1 cell per move)校准。8 类:T1 direction / T2 single-step / T3 multi-step plan / T4 boundary / T5 multi-dim / T6 selection / T7 inverse plan / T8 distance。`arc_agent/bench_probes/__init__.py`。

**出处**: 2026-05-17 加,详 [`project/2026-05-17-v0-model_bench/architecture.md`](./project/2026-05-17-v0-model_bench/architecture.md) §2。

### `state_revisit_count`

🟢 当前 frame_hash 在本 episode 已经出现过几次。`> 5` 触发 R3 / orchestrator stuck alert。

**出处**: `arc_agent/agents/action_agent.py:state_revisit_count`。

### `step PNG` / `viz_v3_2`

🟢 每 step 一张 512×286 的 composite PNG:左半 256×256 grid (4× 放大),右半 256×286 文字 panel (action / reasoning / reflection delta / alert / matches_reasoning)。`arc_agent/viz_v3_2.py:compose_step_image_v32`。

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md) §5.5。
**相关**: [[CNN-small]] 用 PNG 反解码出原 grid。

### `STATIC / ACTIVE / TEXTURE / CANDIDATE`

🟢 [[ObjectRecord]] 的 state 枚举。
- STATIC:静止 + 单帧
- ACTIVE:跨帧动过
- TEXTURE:大面积均匀背景(已被 prompt 过滤)
- CANDIDATE:还在判定

**出处**: `arc_agent/temporal_classifier.py`。

### `StochasticGoose`

⚫ ARC-AGI-3 preview 第一名(Tufa Labs)。CNN 预测 P(frame_change) + RL。Preview 12.58%,全集 0.25%(过拟合)。我们 [[Predictor v0]] 借鉴它的核心信号。

**出处**: [`reports/predictor_v0.md`](../outputs/reports/predictor_v0.md)。

### `Symbolica Agentica`

⚫ ARC-AGI-3 public demo 25 game 36.08%(7/25 通关)。Orchestrator + subagent 架构,基于 Agentica SDK + 大模型。我们的 v3.2 Knowledge / Reflection 思路接近。

**出处**: web search 2026-05-15 + [`reports/grpo_v0_ar25_plan.md`](../outputs/reports/grpo_v0_ar25_plan.md)。

---

## T

### `Tier 1 SFT`

🟡 [`architecture/sft_tier1_zh.md`](./architecture/sft_tier1_zh.md)。在合成数据上 LoRA fine-tune Qwen2.5-VL-3B 的空间推理基础能力(T1-方向 / T2-算术 / T3-相对位置 / T4-格式 / T8-一致性)。

**状态**:第一轮失败(planning probe 完全没修 + gsm8k 退化 -14.7pp,根因 T4 模板锁死)。F3 修法待跑。

**出处**: [`architecture/sft_tier1_zh.md`](./architecture/sft_tier1_zh.md) §12 复盘。

### `TextAgent` (v3)

🟡 v3 单 agent 实现。`arc_agent/agents/text_agent.py`。v3.2 拆出 [[Action Agent]] + [[Reflection Agent]] 后被 deprecate,代码留着可用。

**出处**: [`architecture/v3_zh.md`](./architecture/v3_zh.md) §5。

### `trace.jsonl`

🟢 每个 round 一份的逐步记录。每行一个 JSON,字段含 `step / action / frame_changed / orch_override / reasoning / reflection_delta / ...`。

**出处**: `arc_agent/baseline.py:play_one_with_trace`、`scripts/run_v3_multi_round.py`。

---

## V

### `v3`

🟡 单 agent 架构。scipy perception + Qwen text-only。被 v3.2 取代,代码留着。

**出处**: [`architecture/v3_zh.md`](./architecture/v3_zh.md)。

### `v3.2`

🟢 当前活的架构。Action Agent + Reflection Agent + 跨 round Knowledge + click_targets bandit + R1-R7 硬规则。

**出处**: [`architecture/v3_2_zh.md`](./architecture/v3_2_zh.md)。

### `v3.2 canary` (v2 canary)

🟡 commit `1bac4be` 时的实测状态:R1+R2+R3,无 click_targets,无 canonical rules,prompt 17 块。在 ar25 3×30 step 上 change_rate 60/100/97%。**目前已知的最佳状态**(虽然没通关 levels)。

**出处**: [`reference/v3_2_hardrules_results_zh.md`](./reference/v3_2_hardrules_results_zh.md)。
**相关**: 当前 v3.2 跟它的 diff = 12 commits / 6300 行。

---

## 文档历史

- 2026-05-16 初稿。涵盖到 commit `01a7227`。每次架构 / 报告新增一个概念,在这里加一条。
