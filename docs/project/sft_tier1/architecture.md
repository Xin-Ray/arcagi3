# ARCHITECTURE — Tier 1 SFT(LLM spatial reasoning fine-tune)

日期: 2026-05-15
状态: 设计 → 待实施
前置阅读: [`architecture/v3_2_zh.md`](../v3_2/architecture.md), [`reference/v3_prompt_zh.md`](../v3/reference_prompt.md)

---

## 0. 为什么做这个

2026-05-15 跑了 5 次 **裸模型 planning probe**(全部在 `scripts/test_clean_qwen_*.py`),把 v3.2 的 perception / Knowledge / Reflection 全部抽走,**只给 Qwen 2.5-VL-3B 一个完整 SYSTEM prompt**:坐标系、action 表、起点终点、输出格式。模型还是失败。详见 `outputs/qwen_simple_spatial_probe.txt` + `outputs/vlm_planning_test_board.png`。

| Probe | 输入 | 输出 | 失败模式 |
|---|---|---|---|
| 1 (松 prompt) | text only,完整描述 | prose 不收尾、UP/DOWN 反、出格 | 推理 + 格式都崩 |
| 2 ((40,19)→(55,49), 严格 prompt) | text only | `TOTAL_ACTIONS=4 ACTION3,ACTION3,ACTION3,ACTION5` | dx=3/5 错,**完全忽略 dy** |
| 3 ((49,19)→(55,49), 严格 prompt) | text only | `TOTAL_ACTIONS=4 ACTION1,ACTION3,ACTION3,ACTION5` | dx=2/2 ✓,**y 反向(ACTION1=UP 而非 ACTION2=DOWN)** |
| 4 (probe 3 同题 + 图)| VLM (text + 64×8 RGB) | `TOTAL_ACTIONS=3 ACTION1,ACTION3,ACTION5` | 图反而更差 |
| 5 ("黄在绿左上,往哪走?") | 一句话中文 | "向上" | **空间常识反向** |

**核心观察:Qwen 2.5-VL-3B 对这个任务的推理能力 ≈ 0**。失败模式三类:
- (A) **方向 ⇄ 坐标增减反向**(y 增 = 向下,但模型先验是数学坐标 y 增 = 向上)
- (B) **大整数除法不会**(dx=15 / step=3 = 5,模型常算成 3)
- (C) **多维组合不会**(同时算 dx 和 dy,通常只算 1 个维度)

这三类**对一个 3B 模型来说不该是天花板**,SFT 在合成数据上**很可能**能修。本文档定义如何修 + 如何**严格量化是否真的修了**。

---

## 1. 范围

### 1.1 Tier 1 in-scope(本文档覆盖)

| 任务 | 学什么 | 数据量 |
|---|---|---|
| **T1** | 给坐标系约定 + src/dst y 值,输出 UP 或 DOWN | 50k |
| **T2** | 给 Δ + step,输出整数次数 | 40k |
| **T3** | 给"A 在 B 的 {方位}",输出移动方向(8 方向) | 40k |
| **T4** | 严格遵守 prompt 给的输出模板(任意 schema) | 30k |
| **T8** | reasoning 行的方向跟 action token 的方向一致 | 40k |
| **合计** | | **200k** |

### 1.2 Tier 1 out-of-scope(后面 tier / 别的方案)

| 失败模式 | 为什么不放 Tier 1 |
|---|---|
| Self-monitor / anti-spam(连续 fail 不知道换) | 行为问题,SFT 不擅长;放 GRPO 或 orchestrator 代码 |
| 复合对象识别(L + 内嵌 marker 合并) | perception 算法问题,改 `object_extractor` |
| 跨 game meta-strategy(先 explore 4 方向再 ACTION5) | 需要 game env rollout,放 Tier 3 GRPO |
| 真实 ARC game 数据上训练 | **违规** —— 竞赛 "no task-specific optimization" |
| Goal hypothesis(从 scene 推 win condition) | 放 Tier 2,数据更复杂 |
| 写条件性 semantics("works when ... no-op at wall") | 需要 OutcomeLog 合成,放 Tier 2 |

---

## 2. 规则合规

ARC Prize 2026 规则原文(`TASK_OVERVIEW.md`):
- "No task-specific or domain-specific optimization (for leaderboard — must be general-purpose)"
- "All code must be open-sourced for prize eligibility"

| Tier 1 选择 | 合规理由 |
|---|---|
| 用**合成数据**,完全脱离 ARC | 不是 task-specific,是通用空间推理能力 |
| 不用 ARC demo / private game 数据 | 不违反 task-specific 条款 |
| LoRA 权重 + 训练代码 + 生成数据 全部 open-source | 满足 open-source 条款 |
| 不用闭源模型(本地 Qwen) | 满足 "no internet during eval" 条款 |

合成数据 **永远不**含字符串 `arc25`、`ar25-`、ARC SDK 任何 game_id;**永远不**含 ARC palette specific colors 之外的特殊语义;**永远不**含"This is for ARC-AGI-3"之类的元提示。

---

## 3. 数据生成(合成,纯 Python,无 LLM 依赖)

### 3.1 文件位置

```
arc_agent/finetune/
    __init__.py
    synth_tier1.py            # 5 个生成器 + mix
    chat_format.py            # 包成 ChatML
tests/
    test_synth_tier1.py       # 验生成器(标签正确 / 格式有效)
scripts/
    gen_tier1_data.py         # 命令行入口
outputs/finetune/
    tier1_train.jsonl         # 训练集 ~190k
    tier1_holdout.jsonl       # in-distribution holdout ~10k
    tier1_ood.jsonl           # OOD ~5k
```

每行一个 JSON: `{"system": ..., "user": ..., "assistant": ..., "task": "T1"}`。

### 3.2 T1: 方向 ⇄ y/x 增减(50k)

```python
def gen_T1():
    convention = random.choice([
        "y increases from top to bottom",   # ARC / 屏幕坐标系
        "y increases from bottom to top",   # 数学坐标系(对照)
    ])
    src_y = random.randint(0, 60)
    dst_y = random.choice([y for y in range(0, 64) if y != src_y])
    delta_y = dst_y - src_y
    answer = ("DOWN" if delta_y > 0 else "UP") if convention.endswith("bottom") \
             else ("UP" if delta_y > 0 else "DOWN")
    return {
        "system": "Answer in one word: UP or DOWN. No explanation.",
        "user": f"{convention}. Object at y={src_y}. Target at y={dst_y}. "
                f"To reach the target, the object must move which direction?",
        "assistant": answer,
        "task": "T1",
    }
```

40k 英文 + 10k 中文同 schema。**反向先验是最大 trap,所以 50% 样本 convention 翻转**,模型必须读 prompt 而非依先验。

### 3.3 T2: 小整数除法 + 计数(40k)

```python
def gen_T2():
    step = random.choice([1, 2, 3, 4, 5])
    n = random.randint(1, 20)
    delta = n * step
    return {
        "user": f"Each move shifts the object by {step} cells. "
                f"The object needs to travel {delta} cells in one direction. "
                f"How many moves?",
        "assistant": str(n),
        "task": "T2",
    }
```

变种(每个变种用**独立 task tag**,这样 eval parser 不会因输出格式不同而误判):

| Tag | 模板 | 期望输出 | 用途 |
|---|---|---|---|
| `T2`         | 主体单整数(英文 + 中文同 schema) | `"5"` | 主指标:exact_match + MAE |
| `T2_remainder` | `"delta={d} step={s}, 几次完整,余几?"` | `"完整 5 余 1"`(固定 token) | 余数任务,独立 exact_match |
| `T2_multi`     | `"dx={dx} dy={dy} step={s}, 各几次?"`  | `"x:4 y:6"`(无空格,冒号分隔) | 多维任务,独立 regex 解析 |

主体 `T2` 占 30k,两个变种各 5k。eval suite 主要按 `T2` 评估,变种独立报告但不进决策门。

### 3.4 T3: 8 方向相对位置 → 移动方向(40k)

```python
RELATIONS = [
    ("top-left",    "right-down"),    ("top",         "down"),
    ("top-right",   "left-down"),     ("left",        "right"),
    ("right",       "left"),          ("bottom-left", "right-up"),
    ("bottom",      "up"),            ("bottom-right","left-up"),
]
SHAPES = ["square", "circle", "L-shape", "triangle", "rectangle"]
COLORS = ["yellow","red","blue","green","purple","cyan","gray"]

def gen_T3():
    rel, ans = random.choice(RELATIONS)
    c1, c2 = random.sample(COLORS, 2)
    s1, s2 = random.choices(SHAPES, k=2)
    return {
        "user": f"The {c1} {s1} is at the {rel} of the {c2} {s2}. "
                f"To move the {c1} {s1} to reach the {c2} {s2}, "
                f"output the direction (one of: up / down / left / right / "
                f"left-up / right-up / left-down / right-down).",
        "assistant": ans,
        "task": "T3",
    }
```

加中文同 schema(`"{color} 的 {shape} 在 {color} 的 {shape} 的左上,要走哪边?"`)。

### 3.5 T4: 严格格式遵守(30k)

```python
TEMPLATES = [
    ("TOTAL_ACTIONS=<N>\nACTION_CHAIN=<a,b,c>",
     "TOTAL_ACTIONS=4\nACTION_CHAIN=ACTION3,ACTION2,ACTION3,ACTION5"),
    ("reasoning: <one line>\naction: <token>",
     "reasoning: move yellow object UP\naction: ACTION1"),
    ('{"direction": "X", "count": N}',
     '{"direction": "down", "count": 5}'),
    ("DIRECTION:<X> COUNT:<N>",
     "DIRECTION:down COUNT:5"),
    ("- <line1>\n- <line2>",
     "- yellow 1x1\n- moves UP 3"),
]

def gen_T4():
    template, target = random.choice(TEMPLATES)
    return {
        "system": f"Output format exactly: {template}.\n"
                  "Output ONLY the answer line(s). No markdown, no prose, "
                  "no numbered steps, no explanation, no extra lines.",
        "user": "Now produce one valid output following the format.",
        "assistant": target,
        "task": "T4",
    }
```

**故意**让 user prompt 跟 system 描述的任务无关 —— supervises 模型**单纯按 system 的格式照抄**,而非"理解内容"。这正好对症我们 probe 1 的"prose 满天飞"。

### 3.6 T8: reasoning ⇄ action 一致(40k)

**关键设计约束**:**T8 不得 hard-code 任何固定的 ACTION→方向绑定**。原因:
- v3 的整套设计(`architecture/v3_zh.md` + `knowledge.py:action_semantics` + `OutcomeLog`)建立在 "action 语义是 per-game 的、必须靠 OutcomeLog 当场学" 之上;`agents/llm.py:47` 注释 "ACTION1=Up..." 时也明确写 *"likely; verify by experiment"*。
- 任何固定绑定都会让 SFT 后的模型对那种 binding 不一致的游戏(例如某 game 里 ACTION3=Up)系统性反向。
- T8 真正要教的是 "把 user prompt 当场给的 binding 拷贝到 reasoning + action 两行,且方向词一致" —— 即 **in-context binding copying**,不是死记某个映射。

```python
DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_TOKENS = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

def gen_T8():
    # 每个样本随机一个 (token -> direction) 绑定;模型不能依权重里的先验,只能读 prompt
    direction = random.choice(DIRECTIONS)
    token = random.choice(ACTION_TOKENS)
    color = random.choice(COLORS)
    shape = random.choice(SHAPES)
    return {
        "system": "Output exactly two lines:\n"
                  "  reasoning: <one sentence with a subject and direction>\n"
                  "  action: <ACTION1..ACTION7>\n"
                  "The direction word in `reasoning` must match the action's "
                  "direction as defined by the user prompt.",
        "user": f"In this game, {token} moves objects {direction}. "
                f"The {color} {shape} needs to go {direction}.",
        "assistant": f"reasoning: move the {color} {shape} {direction.lower()} by 3 cells\n"
                     f"action: {token}",
        "task": "T8",
    }
```

**校验(在 `tests/test_synth_tier1.py` 里写死)**:从 40k T8 样本里,任意 token 关联到 4 个方向的频率分布必须每个 ≥ 20%(即任何 token 都不会被偏向某个方向)。 跑生成器后这一项是 hard-fail 测试。

### 3.7 切分 + seed

```python
SEED = 42                     # 固定 -> 任何人复现一致
random.seed(SEED)

samples = []
samples += [gen_T1() for _ in range(50_000)]
samples += [gen_T2() for _ in range(40_000)]
samples += [gen_T3() for _ in range(40_000)]
samples += [gen_T4() for _ in range(30_000)]
samples += [gen_T8() for _ in range(40_000)]
random.shuffle(samples)

# 95% train, 5% holdout
n = len(samples)
holdout = samples[: n // 20]
train   = samples[n // 20 :]

# OOD: 单独生成,确保 delta / phrasing 在训练分布之外
ood = gen_ood_set(n=5_000)    # 见 §6.2
```

OOD 单独生成器,确保数值范围 / 措辞 跟 train 不重叠 —— **测真泛化,不是记忆**。

---

## 4. 训练设置

### 4.1 Base + LoRA

```python
# scripts/train_tier1_sft.py
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from arc_agent.vlm_backbone import load_model

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    bias="none", task_type="CAUSAL_LM",
)
cfg = SFTConfig(
    output_dir="outputs/finetune/qwen3b-tier1-lora",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,     # effective batch = 32
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    save_steps=500,
    logging_steps=20,
    max_seq_length=512,
)
```

| 资源 | 估算 |
|---|---|
| 显存(4-bit base + bf16 LoRA) | ~7 GB |
| Step 数 (190k / 32 × 2 epoch) | ~12k |
| 训练时间 T4 (~1.5s/step) | ~5h |
| 训练时间 A10 (~0.8s/step) | ~3h |
| LoRA artifact 大小 | ~50 MB |

### 4.2 LoRA 集成回 vlm_backbone

```python
# arc_agent/vlm_backbone.py (现有,line 263 已支持 lora_path)
HFBackbone.load(
    model_path=BASE,
    quantize="4bit",
    lora_path="outputs/finetune/qwen3b-tier1-lora",
)
```

部署:`scripts/run_v3_multi_round.py` 加 `--lora-path` flag,默认 None。

---

## 5. 评估协议(严格 baseline-first)

### 5.1 三层 eval suite

| 套件 | 样本 | 评估指标 | 目的 |
|---|---:|---|---|
| **T1_holdout** | 2000 | exact_match("UP"/"DOWN") | T1 学会了吗 |
| **T2_holdout** | 2000 | exact_match(整数) + MAE | T2 学会了吗 |
| **T3_holdout** | 2000 | exact_match(8 方向) | T3 学会了吗 |
| **T4_holdout** | 1500 | regex 命中 + 完整字符串匹配 | T4 学会了吗 |
| **T8_holdout** | 2000 | reasoning_dir == action_dir | T8 学会了吗 |
| **T1_OOD** (delta 60-120) | 200 | exact_match | 不靠记忆 |
| **T3_OOD** (颜色/形状未见组合) | 200 | exact_match | 不靠 surface form |
| **T1+T2_combined** | 300 | 两阶段都对 | 组合泛化 |
| **planning_probes** | 5 | 见 §5.2 | 我们关心的真任务 |
| **regression_general** | 450 | 见 §5.1.1(MMLU mini + GSM8K mini + 中文 QA mini,固定 150 行 × 3 套) | 通用能力没破 |
| **v3_2_game_mock** | 100 | mock Action prompt(无 ARC 数据)| 真用得上 |

### 5.1.1 Regression suite 数据来源(固化 artifact)

为让 regression 检查可复现,以下 3 个 mini-set **必须在 base baseline 跑之前**落盘到 git:

```
data/regression/
├── mmlu_mini.jsonl     150 行,seed=42 从 cais/mmlu test split 均匀抽 5 个子科目
├── gsm8k_mini.jsonl    150 行,seed=42 从 gsm8k test 抽,只保留 final answer 是整数的
└── zh_qa_mini.jsonl    150 行,seed=42 从 CMMLU test 抽,均匀覆盖文 / 理 / 社科
```

格式统一为 `{"prompt": str, "answer": str, "source": str}`。eval 时按 prompt 单 turn 输出,正则匹配最后一行 / `\boxed{}` / 选项字母即可,**不允许联网拉数据**。生成脚本 `scripts/build_regression_mini.py` 一次性跑完 commit 这 3 个文件,后续不再改。

### 5.2 Planning probes 拆分指标(每个 probe 7 个 binary)

| 维度 | 计算 |
|---|---|
| format_valid | 输出含 `TOTAL_ACTIONS=` AND `ACTION_CHAIN=` |
| direction_x_correct | RIGHT/LEFT 方向对 |
| direction_y_correct | UP/DOWN 方向对 **← y 反向就在这扣分** |
| count_x_correct | dx/step 数量对 |
| count_y_correct | dy/step 数量对 |
| terminator_correct | 最后一个是 ACTION5 |
| final_distance | Manhattan(|预测最终位置 - 目标|) |

### 5.3 执行顺序(强制 baseline 在 train 之前)

```
Step 1: scripts/gen_tier1_data.py     # 生成 train + holdout + ood,seed=42
Step 2: scripts/eval_tier1.py --backbone base
        → outputs/finetune/base_metrics.json     [BEFORE training]
        ★ 这一步必须先跑,否则后面没基线对比
Step 3: scripts/train_tier1_sft.py    # 3-5h GPU
Step 4: scripts/eval_tier1.py --backbone lora --lora-path .../qwen3b-tier1-lora
        → outputs/finetune/lora_metrics.json
Step 5: scripts/report_tier1.py base_metrics.json lora_metrics.json
        → outputs/finetune/tier1_eval_report.md
```

### 5.4 报告格式(`tier1_eval_report.md`)

```markdown
# Tier 1 SFT Evaluation Report

## In-distribution holdout
| Suite | N | Base | LoRA | Δ | Target |
|---|---:|---:|---:|---|---|
| T1_holdout | 2000 | 28% | XX% | +YY pp | ≥ 95% |
| T2_holdout | 2000 | 51% | XX% | +YY pp | ≥ 98% |
...

## OOD generalization
| Suite | N | Base | LoRA | Δ |
| T1_OOD large numbers | 200 | XX% | XX% | YY pp |
...

## Planning probes (the real test)
| Probe | format | x_dir | y_dir | x_cnt | y_cnt | term | final_dist |
| 1 base | ❌ | ✓ | ❌ | n/a | n/a | n/a | n/a |
| 1 LoRA | XX | XX | XX | XX | XX | XX | XX |
...

## Regression (LoRA 不能伤通用能力)
| Suite | Base | LoRA | Acceptable Δ |
| MMLU mini | 47% | XX% | > 42% (-5pp) |
| GSM8K mini | 30% | XX% | > 25% (-5pp) |
...
```

---

## 6. 决策门(Tier 1 通过 / 失败标准)

### 6.1 通过

**全部满足才算通过**(此前版本写的是 "任一提升 ≥ 50pp",太宽 —— 任意一项达标都过门会掩盖 y 反向没修的根本问题。已收紧):
- ✅ **T1 holdout 提升 ≥ 50 个百分点**(y 反向是核心失败,T1 是唯一对症数据,**必须**学会)
- ✅ T2 / T3 / T4 / T8 holdout **各自**提升 ≥ 30 个百分点(辅助指标,允许较弱)
- ✅ planning probes:**`direction_y_correct` 从 0/5 → ≥ 4/5**(端到端在我们真正关心的任务上看到效果)
- ✅ planning probes:**`count_y_correct` 从 0/5 → ≥ 3/5**
- ✅ regression(MMLU mini / GSM8K mini / 中文 QA)**全部**退化 ≤ 5 个百分点(注意是 "全部" 不是 "任一")

### 6.2 失败 / 红线

- ❌ planning probes y 方向**仍全错** → SFT 数据 T1 不够 / 没学到反向先验破除 → 不要 deploy,重新设计 T1
- ❌ regression **任一退化 > 10 pp** → LoRA 破坏通用能力 → 降 LR 或减 epoch 重训
- ❌ holdout 准确率 > 90% 但 OOD 暴跌 → 模型在记忆,数据生成器需多样化

---

## 7. 跟 v3.2 怎么对接

| 阶段 | 改动 | 风险 |
|---|---|---|
| Tier 1 训完通过 § 6.1 决策门 | `vlm_backbone.HFBackbone.load(lora_path=...)` 加载 LoRA | 几乎无 |
| 跑 v3.2 smoke (`scripts/run_v3_multi_round.py --lora-path ...`) | 对比 base 模型 smoke 的 ACTION1 spam / change_rate / matches_reasoning | 中:LoRA 可能让 model 出 wired 输出,需要监控 |
| 整合进 default backbone | CLAUDE.md / README.md 加 "v3.2 推荐用 Tier 1 LoRA" | 低 |

**Tier 1 不会改 Action prompt / Reflection prompt / orchestrator 任何代码** —— 只换 model 权重。这意味着如果发现 LoRA 没生效,可以**立即换回 base**,零迁移成本。

---

## 8. 实施步骤(时间估算)

| 步 | 内容 | 时长 |
|---|---|---:|
| 1 | 写 `synth_tier1.py` 5 个生成器 + `test_synth_tier1.py` | 2h |
| 2 | 写 `gen_tier1_data.py`,产 train + holdout + ood | 0.5h |
| 3 | 写 `eval_tier1.py` 全套 suite + probe 集成 | 2h |
| 4 | **跑 base baseline** (Step 2 of §5.3) | 1-2h(纯 inference) |
| 5 | 写 `train_tier1_sft.py` LoRA 配置 | 1h |
| 6 | **跑训练**(后台,不占人) | 3-5h GPU |
| 7 | 跑 LoRA eval + 出 report.md | 1h |
| 8 | LoRA 接 vlm_backbone + smoke 对比 | 1-2h |

**人工时间:~8h(1 工作日);GPU wall:~5-8h(挂后台,人不在也跑)**

---

## 9. 已知风险 / 未解决问题

| 风险 | 缓解 |
|---|---|
| LoRA 把通用能力训坏(catastrophic forgetting on MMLU)| eval suite 强制 regression check,>10pp 退化就回滚 |
| 合成数据"太干净"(模型在 toy 上表现好,实 game prompt 还是崩) | v3_2_game_mock 套件兜底,真实 prompt pattern 上测 |
| 5 个 planning probes 是我们手挑的,可能 cherry-pick | 加 50 个 procedurally-generated probes 当扩展 eval |
| LoRA 只学到 surface pattern(看到 "y" + "top to bottom" 就答 "DOWN")| OOD 套件强制 paraphrase(中英 / 不同词序 / 隐含约定)|
| 训练时长不够(2 epoch 不收敛) | logging_steps=20 实时看 loss,不收敛就加 epoch |
| **本 Tier 不修 self-monitoring** | 这是设计 —— anti-spam 留 orchestrator code / GRPO 做,Tier 1 别试 |

---

## 10. 跟更大计划的关系

```
Tier 1 (本文档)          —— LLM 单步基础能力(空间、算术、格式、一致性)
   ↓ 通过 § 6.1 决策门后
Tier 2(规划中)          —— LLM 单步语义能力(goal 推断、conditional semantics、subject 识别)
   ↓
Tier 3(规划中)          —— LLM 行为策略(GRPO on synthetic grid envs,self-monitor / 跨 game meta)
   ↓
完整 Kaggle 提交           —— Tier 1+2+3 LoRA + v3.2 orchestrator + planner + perception 全套

正交并行项(本文档不涵盖):
- orchestrator_planner.py (确定性 BFS,代码做 planning)
- anti-spam rescue guard (代码做 self-monitor)
- object compound merger (perception 改算法)
```

---

## 11. 验收 checklist

实施完每一步后打勾(自查用):

- [ ] `arc_agent/finetune/synth_tier1.py` 存在,5 个生成器有 type hints + docstring
- [ ] `tests/test_synth_tier1.py` 全部 pass,覆盖每个 T 的 1 正例 + 1 边界
- [ ] `outputs/finetune/tier1_train.jsonl` 行数 ≈ 190k,seed=42 可复现
- [ ] `outputs/finetune/tier1_holdout.jsonl` 与 train 完全不交
- [ ] `outputs/finetune/base_metrics.json` 存在(**在 train 之前**)
- [ ] LoRA 训练 loss 单调下降,无 NaN
- [ ] `outputs/finetune/lora_metrics.json` 完整 5 套件
- [ ] `outputs/finetune/tier1_eval_report.md` 满足 § 6.1 决策门
- [ ] LoRA 接进 `vlm_backbone.py`,`scripts/run_v3_multi_round.py --lora-path` 跑通
- [ ] CLAUDE.md / README.md / `docs/INDEX_zh.md` 各加一行说明
- [ ] **T8 不含固定 ACTION→方向先验**(由 `test_synth_tier1.py::test_T8_no_fixed_binding` 保证;见 §3.6)
- [ ] `data/regression/{mmlu_mini,gsm8k_mini,zh_qa_mini}.jsonl` 三份在 base baseline 之前 commit(见 §5.1.1)

---

---

## 12. 2026-05-15 实测复盘(第一次跑完整流程)

状态: **§6.1 决策门 FAIL**,**不 deploy**,等修了 §12.4 的根因再跑第二轮。

完整产物:
- `outputs/finetune/base_metrics.json` / `lora_metrics.json`
- `outputs/finetune/tier1_eval_report.md`(自动 base vs LoRA 对比 + 门裁定)
- `outputs/finetune/qwen3b-tier1-lora/`(LoRA adapter 14 MB,保留作对照)

### 12.1 训练实测

- 时间: 7h24m (2 epochs / 11876 steps,RTX A4500 / bf16 LoRA + 4-bit base)
- Loss: step 0 = **5.47** → step 1000 = **0.13** → step 11876 = **0.12**(前 1000 步暴跌 40×,之后一直平台期)
- mean_token_accuracy: 0.29 → 0.94 → 0.94
- grad_norm: 4.56 → 0.10(稳定收敛,无 NaN)
- LoRA artifact: `adapter_model.safetensors` 14.1 MB(0.24% 可训参数)
- **训练本身完美**,问题在数据设计。

### 12.2 Eval 结果(cap=500/suite)

| Suite           | Base  | LoRA   | Δ        |
|---|---:|---:|---|
| T1 holdout      | 36.2% | 100.0% | +63.8 pp |
| T2 holdout      | 92.2% | 100.0% | +7.8 pp  |
| T2_remainder    | 64.8% | 100.0% | +35.2 pp |
| T2_multi        | 39.5% | 100.0% | +60.5 pp |
| T3 holdout      |  5.2% | 100.0% | +94.8 pp |
| T4 holdout      |  0.0% | 100.0% | +100 pp  |
| T8 holdout      |  0.0% | 100.0% | +100 pp  |
| T1_ood          | 73.2% |  39.4% | **-33.8 pp** ⚠️ |
| T2_ood          | 93.4% |  87.8% | -5.6 pp  |
| T3_ood          | 21.0% |  34.4% | +13.4 pp |
| T4_ood          |  3.2% |   2.4% | -0.8 pp  |
| T8_ood          |  0.0% | 100.0% | +100 pp ★ |
| planning y_dir  | 1/5   | 1/5    | 0 pp **★ 核心目标没动** |
| planning y_cnt  | 0/5   | 1/5    | +20 pp(远低于 ≥ 3/5 门) |
| planning final_dist | 46.2 | 68.4 | +22 cells(更远) |
| mmlu_mini       | 55.3% |  56.0% | +0.7 pp  |
| gsm8k_mini      | 30.7% |  16.0% | **-14.7 pp** ⚠️ |
| zh_qa_mini      | 69.6% |  74.5% | +4.9 pp  |

决策门 10 项:**4 项失败**(T2 holdout/饱和 + planning y_dir + planning y_cnt + gsm8k regression)。

### 12.3 哪些信号是真的有效的(下次保留)

1. **T8 in-context binding 工作了**:OOD T8 用了**训练里没见过的 token**(ACTION5/6/7)仍 100%。说明 §3.6 的 "binding 当场随机,不固定" 设计真的让模型学会了"读 prompt 的 binding 而非记权重",这是 §12.4 修法里要**保留**的核心结构。
2. T1/T3/T4 holdout 全部 100%:模型有能力把简单空间映射学到熟。
3. 中文 QA 反而升 4.9 pp:zh prompt 数据让模型在中文 MC 上小幅改进。

### 12.4 根因(为什么 planning probe 完全没修)

从 `_raw` 字段看 5 个 planning probe 的实际输出:

```
Base 输出(全部 5 个 probe 完全一样,与 src/dst 无关):
  TOTAL_ACTIONS=7
  ACTION_CHAIN=ACTION1,ACTION1,ACTION1,ACTION1,ACTION1,ACTION1,ACTION5

LoRA 输出(全部 5 个 probe 几乎一样,只 N 不同):
  TOTAL_ACTIONS=18 或 9
  ACTION_CHAIN=ACTION3,ACTION3,...,ACTION3   (全是 ACTION3,无 ACTION5)
```

**两个模型都在 spam 单一动作,完全没读 src/dst**。LoRA 只是从 spam ACTION1 切换到 spam ACTION3。

根本原因在 §3.5 的 T4 设计:T4 的 5 个 template 之一(`TOTAL_ACTIONS=<N>\nACTION_CHAIN=<a,b,c>`)的 canonical target 写死成 `ACTION_CHAIN=ACTION3,ACTION2,ACTION3,ACTION5`,这一对 (template, target) 在 30k T4 样本里反复出现约 6k 次。**这个 format 跟 planning probe 的 SYSTEM prompt 输出格式 100% 重合**,而 target 固定到死。

模型 T4 holdout 100%,代价是把 "TOTAL_ACTIONS / ACTION_CHAIN format" 这个解空间锁死到固定答案附近。当 planning probe 用同款 format 提问,greedy 解码就一路按 T4 学到的高概率序列走 —— 跟 src/dst 内容**完全无关**。

这是文档自己 §9 风险表里第二条 "**模型在记忆**" 的实例化。

### 12.5 下次修法(三档,2026-05-16+ 决策)

| 方案 | 改动 | 工作量 | 预期 |
|---|---|---|---|
| **F1**(轻)| T4 那条 `TOTAL_ACTIONS` 模板的 target 改成**每条样本独立随机的 N + random 动作序列** | 1 行 `synth_tier1.py:_T4_TEMPLATES` | 7h 重训 | 拆掉"看到 format 就背诵固定答案"的锚 |
| **F2**(中)| **完全删除** T4 那条跟 planning probe 重合的模板,只保留另外 4 个无关格式 | 5 行 | 7h 重训 | 避免任何 format 重合污染 |
| **F3**(重,推荐)| F1 + 新增 5k **planning-format 训练样本**:用合成坐标算正确 chain(`gen_T9_planning(rng)`),教模型在该格式下真的算路径 | ~50 行新生成器 + 调 mix + 加 §3.8 章节 | 7h 重训 | 不止避免污染,**让 planning prompt 真有用** |
| 别再 SFT | 转 Tier 2 / GRPO | — | — | 接受 SFT 路线天花板 |

**F3 的合规边界**:planning 样本必须用 1) 合成 src/dst 坐标 2) 任意 step 大小 3) 任意 action token 绑定(per §3.6 in-context binding)4) 不带 ARC game_id 或 ar25 调色板语义。这样**不违反 "no task-specific optimization"**。

### 12.6 §6.1 决策门要不要改

T2 holdout 这一项("≥ +30 pp")**门设得不合理** —— base 已经在 92%,LoRA 100%,只剩 8pp 空间根本不可能 +30 pp。建议改成 "T2 holdout ≥ 99% 或 Δ ≥ +5 pp(取松)"。其它 9 项门保留(planning y_dir、gsm8k regression 真的失败,不要放水)。

---

*文档历史:*
- *2026-05-15 初稿(based on 5 planning probes failure analysis)*
- *2026-05-15 review patch:T8 改成 in-context binding(原版 hard-code 的 ACTION3=RIGHT 与本仓库 `agents/llm.py:47` 的 ACTION3=Left 反向,且违反 v3 "action semantics is per-game" 原则);T2 变种独立 task tag;§6.1 决策门收紧;§5.1.1 固定 regression 数据来源。*
- *2026-05-15 实测复盘(§12):第一轮 SFT FAIL 决策门;根因 = T4 `TOTAL_ACTIONS` 模板与 planning probe 共享格式 + 固定 target,锁死了 planning 解空间。下次走 F3:加 planning-format 合成样本 + T4 target 随机化。T8 in-context binding 已验证有效(OOD T8 100%),保留。*
