# ARCHITECTURE — Model Bench v0(换 backbone 路线探针)

日期: 2026-05-17
状态: 实施中
分支: `feat-2026-05-16-v0-action_proposer`(复用现有分支,bench 跟 proposer 是同一条路线 — 都是「换什么模型」的探索)
前置阅读: [`action_proposer/architecture.md`](../2026-05-16-v0-action_proposer/architecture.md);[`sft_tier1/architecture.md`](../2026-05-15-v0-sft_tier1/architecture.md) §0(裸 Qwen 失败模式)

---

## 0. 为什么做这个

[Tier 1 SFT §0](../2026-05-15-v0-sft_tier1/architecture.md#0-为什么做这个) 实测显示 Qwen2.5-VL-3B 在 5 个 planning probe 上 **0/5 通过**:y 反向、整数除法、多维组合都不会。

Web 搜得(2026-05-17):同尺寸开源模型有比 Qwen-3B 强一档的:
- **Phi-4-mini-reasoning (3.8B)**: ARC-Challenge **83.7%**(SLM 最强),GSM8K 88.6%,Microsoft 专门带 spatial benchmark(Maze, SpatialMap)
- **SmolLM3-3B**: 击败 Qwen2.5-3B + Llama-3.2-3B,双模式 reasoning
- **DeepSeek-R1-Distill-Qwen-7B**: MATH-500 92.8%,7B 在 T4 16GB 上边缘

换模型可能比改 prompt / 调 SFT 更直接有效。

---

## 1. 范围

### 1.1 In-scope

- 写 28 个 game-content-based spatial probe(`arc_agent/bench_probes/__init__.py`)。**不靠特定 game mechanic**,测「直角坐标 + ACTION→方向映射 + 多步规划」的通用技能
- 写多模型 bench(`scripts/bench_models_spatial.py`),sequential GPU 串行 load + bench
- bench 3 候选:Qwen2.5-VL-3B(baseline)/ Phi-4-mini-reasoning / SmolLM3-3B
- 出对比图 + 选 top 1 model
- 用 top 1 跑 G_base 5 game × 2 round × 300 step(用 `action_proposer/run_action_proposer_5game.py`)

### 1.2 Out-of-scope

- 不调 prompt(用同一个 system + chat template,公平对照)
- 不 fine-tune(纯 zero-shot 对比)
- 不上 7B+(T4 16GB 4-bit 边界,Phi-4 3.8B 已经是上限)
- 不接 vision input(我们 v3 决定 text-only;benchmark 也是文本 MC)

---

## 2. Probe 设计(8 类 ≥ 28 题)

| 类 | 测什么 | 题数 |
|---|---|---:|
| T1 direction | UP/DOWN/LEFT/RIGHT 根据 coord delta 判方向 | 5 |
| T2 single-step | 1 个 ACTION 后 object 的新坐标(整数加减)| 4 |
| T3 multi-step plan | A→B 最短 action 序列(单维)| 4 |
| T4 boundary | edge 上 ACTION 是不是 no-op | 3 |
| T5 multi-dim plan | A→B 需要 x + y 两维移动 | 4 |
| T6 selection | ACTION6 click 哪个 object | 3 |
| T7 inverse plan | 给 A、B,首 action 是啥 | 3 |
| T8 distance | manhattan / euclidean 距离 | 3 |

**全部 4-选 1**(letter A/B/C/D)。random baseline = 25%。

### 2.1 题目设计原则

- **不靠 game-specific mechanic**:每题给「ACTION1=UP, ACTION3=LEFT」这种映射,然后问数学,模型只需通用推理
- **直角坐标系 y 向下**(ARC SDK 标准):y 增 = 向下;符合 game source 实际行为
- **整数 1 cell per move**(看 ar25 source `axiknfnrlv = self.yvifanjrcyu.x + qlqwojsdxcs` 确认 1 cell)
- **不需要看图**:全部 text-only,3 个候选模型都能跑

---

## 3. 数据流

```
arc_agent/bench_probes/__init__.py    ← 28 道题
        │
        ▼
scripts/bench_models_spatial.py
        │
        │  for entry in MODELS:
        │      load_model(entry)        ← 4-bit nf4 量化,T4 16GB 跑得动
        │      for probe in probes:
        │          messages = [system, user_msg]
        │          prompt = tokenizer.apply_chat_template(...)
        │          out = model.generate(...)
        │          guessed = parse_answer(out)   ← 找 "Answer: X" 或独立 A/B/C/D
        │          score += (guessed == correct)
        │      free_model(model)        ← gc + torch.cuda.empty_cache()
        ▼
outputs/model_bench_<ts>/
├── metrics.json          per-model + per-category accuracy
├── summary.md            markdown table
├── per_probe_<id>.jsonl  raw outputs per model
└── figures/
    ├── accuracy_by_model.png
    └── accuracy_by_category.png
```

---

## 4. 关键决策

| 决策 | 为什么 |
|---|---|
| 用 multi-choice (A/B/C/D) 而不是 open generation | Qwen / Phi / SmolLM 都强于「N 选 1」(参见 [action_proposer](../2026-05-16-v0-action_proposer/architecture.md));减少 parse 噪声 |
| Sequential GPU(不同时 load 多 model)| T4 16GB 装一个 4-bit 模型 + KV cache 已经 ~10 GB,装俩可能 OOM |
| 4-bit nf4 量化 | 跟我们生产配置一致(`vlm_backbone.py` 用 bnb 4-bit);bench 跟 production 同样的精度 |
| 28 题(而不是 200+) | bench 总时长 ≤ 30 min × 3 model = 1.5h,跟 5×2×300 GPU 预算比可忽略 |
| 题目用「数学计算」体感 | Qwen-VL 在 GSM8K 上 ~30%;Phi-4-mini 88.6% — 这个差距应该直接反映在 probe acc 上 |

---

## 5. 模块清单

| 文件 | 状态 | 责任 |
|---|---|---|
| `arc_agent/bench_probes/__init__.py` | 🆕 NEW | 28 道 probe + `get_probes()` 接口 |
| `scripts/bench_models_spatial.py` | 🆕 NEW | sequential bench runner + viz |
| `tests/test_bench_probes.py` | 🆕 NEW | sanity:probe count、required fields、unique ids |

---

## 6. 评估方法

### 6.1 指标

| 指标 | 单位 | 期望 |
|---|---|---|
| 总体 accuracy | % (n_correct / 28) | random=25%;Qwen 预测 30-50%;Phi-4 预测 70-90% |
| per-category accuracy | % 每 T1..T8 | 看模型在哪类弱 |
| load time | 秒 | 3-30 秒(model 大小决定)|
| generate per probe | 秒 | 1-5 秒(影响 5×2×300 wall clock 估计)|

### 6.2 决策门

| 门 | 条件 | 失败应对 |
|---|---|---|
| **G1 winner exists** | 任一模型 accuracy ≥ Qwen baseline + 15pp | < 15pp → 换模型 ROI 不够,放弃这条线 |
| **G2 winner makes sense** | winner 在 T1+T2+T3(基础空间)≥ 60% | 全是 T6+T7 类「猜对」的话不可信;重测 |
| **G3 inference 速度可接受** | winner 每 probe ≤ 5s | > 5s → 5×2×300 跑 60h+,不可行;选第二名 |

---

## 7. 已知风险

| 风险 | 缓解 |
|---|---|
| Phi-4 / SmolLM3 HF 下载失败 / repo 改名 | bench 框架优雅 fail,跳过该模型,继续下一个 |
| chat_template 在某个 model 上 raise | 已加 fallback `f"{system}\n\nUser: {user}\nAssistant:"` |
| Phi-4-reasoning 模式 token 多 → 慢 | max_new_tokens=96(够找 "Answer: X");必要时 32 |
| answer parser 失败率高 | 双层 fallback:`Answer: X` regex → 前 200 字找独立 A/B/C/D |
| GPU OOM | sequential load + `free_model()` (gc + empty_cache);4-bit 量化 |

---

## 8. 跟其它路线的关系

- 跟 [action_proposer](../2026-05-16-v0-action_proposer/architecture.md):**正交**。proposer 改 prompt 结构;model bench 选 backbone。winner 进 proposer pipeline 跑 5×2×300
- 跟 [predictor_v01](../2026-05-16-v01-predictor/architecture.md):**正交**。predictor 失败了,我们换模型不靠 predictor
- 跟 [sft_tier1](../2026-05-15-v0-sft_tier1/architecture.md):**替代方向**。SFT 改权重;model bench 直接换模型。如果换模型有效,SFT 路线降低优先级

---

## 9. 文档历史

- *2026-05-17 初稿:web 搜得 Phi-4-mini-reasoning 是 ARC-C SLM 最强后写。bench 立项。*
