# ROADMAP — 任务清单

最近一次更新:2026-05-08(Phase 2 补全 Agent 架构细节、输入输出、训练机制、参考文献)
当前阶段:**Phase 2 — 数据管线 + BC 训练**

---

## 关键日期

| 日期 | 事件 |
|------|------|
| 2026-06-30 | **Milestone #1 截止**(开源奖 $25K/$10K/$2.5K) |
| 2026-09-30 | **Milestone #2 截止** |

---

## 提交路径(主线)

```
收集人工 trace  →  数据处理  →  BC fine-tune  →  Kaggle 提交
```

VLMAgent zero-shot 分数太低不够用;RL 留有时间再做。

**评估指标 RHAE** = `min(1.0, h/a)²`
- h = 人类第二好步数,a = agent 步数
- 慢 2× → 0.25 分;超过 5× → 0 分
- 目标:BC fine-tune ≥ 0.10,提交版 ≥ 0.20

---

## Agent 架构

### 总览

```
游戏环境 (ARC-AGI-3)
        │  env.step(action) → FrameDataRaw
        ▼
┌──────────────────────────────────────────────────────────┐
│                     VLMAgent.choose()                    │
│                                                          │
│  FrameDataRaw                                            │
│    │                                                     │
│    ├─ latest_grid() ──► 64×64 numpy ──► grid_to_image() │
│    │                                        │            │
│    │                               512×512 PNG           │
│    │                                        │            │
│    └─ summarize_frame() ──────► text prompt │            │
│         state / level /                     │            │
│         available_actions /                 │            │
│         last 3 steps history                │            │
│                                             │            │
│                               ┌─────────────▼──────────┐ │
│                               │  Qwen2.5-VL-3B-Instruct│ │
│                               │  (QLoRA fine-tuned)    │ │
│                               │                        │ │
│                               │  [ViT Vision Encoder]  │ │
│                               │  14px patches → ~340   │ │
│                               │  visual tokens         │ │
│                               │         +              │ │
│                               │  [3B Qwen2.5 LLM]     │ │
│                               │  text tokens           │ │
│                               │         ↓              │ │
│                               │  "ACTION: ACTION3"     │ │
│                               └────────────────────────┘ │
│                                             │            │
│                         parse_action() ◄────┘            │
│                               │                          │
└───────────────────────────────┼──────────────────────────┘
                                ▼
                        GameAction.ACTION3
                                │
                        env.step(action)
```

### 为什么用图像而不是文本 grid

64×64 grid 编码为 hex 文本时,每行 64 个字符,上下相邻的格子在 token 序列里相距 ~64 个位置。
3B 文本模型的 attention 虽然可以跨越这个距离,但它没有 2D 归纳偏置,无法自然地"看到"行列的空间结构。

Qwen2.5-VL 的 ViT 视觉编码器将 512×512 图像切成 14×14 像素的 patch,每个 patch 对应 grid 上的一小块区域。
视觉 patch 通过 2D Rotary Position Embedding (2D-RoPE) 编码绝对坐标,attention 矩阵直接在空间相邻的 patch 之间建立联系。
这是文本 token 序列做不到的。(参考:[1] Qwen2.5-VL paper §3.1)

### 模型输入(每一步)

```
system: |im_start|system
You are an AI agent playing a puzzle game. You see the game grid as an image.
Each color represents a different element. No instructions are given — you must
infer the rules by observing how your actions change the grid.
Think step by step using Hypothesis → Execute → Iterate reasoning.
|im_end|

user:   |im_start|user
<image>                          ← 512×512 RGB PNG (来自 grid_to_image)
State: NOT_FINISHED
Level: 2 / 5
Available actions: ACTION1 ACTION2 ACTION3 ACTION4 ACTION5
Last actions: ACTION3, ACTION1, ACTION3

Hypothesis: There seems to be a movable object (color 2) that needs to reach
            the target cell (color 5).
What is your next action? Reply exactly with: ACTION: <action_name>
|im_end|
```

**图像构造**:`grid_to_image(grid, scale=8)` — 64×64 grid → 512×512 PNG
- 每个 cell 渲染为 8×8 像素正方形
- 使用 ARC 标准 16 色调色板(0=黑,1=蓝,2=红,3=绿,4=黄…)
- 输出 RGB PIL Image

**文本 prompt 构造**:
- `state`:来自 `frame.state`(NOT_FINISHED / GAME_OVER)
- `level`:来自 `frame.levels_completed` / `frame.win_levels`
- `available_actions`:来自 `available_action_names(frame)`
- `last 3 actions`:agent 内部维护的 history deque
- `Hypothesis`:agent 内部维护的当前假设(HEI loop 的 H 部分)

### 模型输出

```
assistant: |im_start|assistant
ACTION: ACTION3
|im_end|
```

模型只输出一行。`parse_action()` 用正则提取 `ACTION\s*:\s*(ACTION\d+)` → `GameAction` enum。
解析失败时 fallback 到随机可用动作(与 LLMAgent 相同策略)。

**为什么输出这么短**:只预测一个动作 token 比预测长推理链更稳定,训练样本数据也直接对应动作标签,不需要 chain-of-thought 蒸馏。推理过程体现在 prompt 的 Hypothesis 字段(下一步的 H),而不是输出。

### ViT 视觉编码器细节

- **架构**:Qwen2.5-VL 使用 "Window Attention ViT"(非标准 ViT,窗口内做 local attention 降低计算量)
- **patch 大小**:14×14 像素
- **图像大小**:512×512
- **patch 数量**:(512 ÷ 14)² = 1337 patches
- **Spatial Merge**:相邻 2×2 patch 合并 → ~334 visual tokens(节省 LLM backbone 的序列长度)
- **位置编码**:2D-RoPE,patch (row, col) 分别编码,天然理解行列位置
- **输出**:~334 个 visual token,与文本 token 拼接后送入 3B Qwen2.5 backbone
- (参考:[1] §3.1, §3.2)

### QLoRA:为什么这样训练

**问题**:Qwen2.5-VL-3B 完整 BF16 权重 ~6GB;全参数微调需要梯度+优化器状态 ≈ 18–24GB,超出 A4500 20GB 上限。

**QLoRA 方案**([2]):
1. **4-bit 量化基础权重**:用 NF4(NormalFloat4)格式把 3B 参数压到 ~1.5GB,推理时 on-the-fly 反量化
2. **LoRA 低秩适配器**([3]):在目标层(q_proj, v_proj)旁边加小矩阵 A(d×r) 和 B(r×d),r=8
   - 前向:实际权重 = W_base + α/r × B·A (α=16)
   - 参数量:2 × (3072×8 + 8×3072) × 层数 ≈ 几百万参数 vs 30 亿基础参数
3. **只有 LoRA 权重有梯度**:基础权重 frozen + 4-bit = 显存占用极小
4. **显存估算**:
   - 4-bit 模型权重:~1.5GB
   - LoRA 权重 + 梯度 + Adam 状态:~1GB
   - 激活值(gradient checkpointing 开启):~8GB
   - 总计:~10–12GB ✅ A4500 20GB 充裕

**训练工具栈**:
- `transformers` — Qwen2.5-VL 模型加载
- `peft` — LoRA 配置 `LoraConfig(r=8, target_modules=["q_proj","v_proj"])` ([4])
- `bitsandbytes` — 4-bit 量化 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` ([5])
- `trl.SFTTrainer` — 处理 loss masking(只对 assistant 部分计算 loss) ([6])

### BC 训练:逐步机制

```
for each batch of (image, prompt, action_str) from human traces:

  step 1 — tokenize
    messages = [
      {"role": "system",    "content": SYSTEM_PROMPT},
      {"role": "user",      "content": [{"type":"image","image":pil_img},
                                         {"type":"text", "text":prompt}]},
      {"role": "assistant", "content": "ACTION: " + action_str},
    ]
    input_ids, pixel_values, labels = processor(messages, return_tensors="pt")

  step 2 — loss masking (SFTTrainer 自动处理)
    labels[labels == PROMPT_TOKEN_IDS] = -100   # 只对 "ACTION: ACTION3" 计算 loss
                                                 # system + user 部分忽略

  step 3 — forward pass
    logits = model(input_ids=input_ids, pixel_values=pixel_values).logits

  step 4 — loss
    loss = cross_entropy(logits, labels)         # 只有 action token 贡献梯度

  step 5 — backward (只流向 LoRA 参数)
    loss.backward()
    optimizer.step()   # AdamW,只更新 A, B 两个小矩阵
    scheduler.step()
```

**直觉**:模型看着图像 + 游戏状态,学会"人类在这种情况下会按哪个键"。
反复看几千步人类操作后,模型获得了对游戏规律的隐式理解。

### 推理时的游戏循环

```python
# arc_agent/agents/vlm.py 的核心逻辑
def choose(self, latest: FrameDataRaw, history: list[FrameDataRaw]) -> GameAction:
    grid  = latest_grid(latest)                          # 64×64 numpy
    image = grid_to_image(grid, scale=8)                 # 512×512 PIL Image
    prompt = self._build_prompt(latest, history)         # text with state/level/history
    text_output = self.backbone.generate(image, prompt)  # Qwen2.5-VL 推理
    action = self._parse(text_output, latest)            # extract ACTION: xxx
    self._update_hypothesis(text_output)                 # 更新 HEI 状态
    return action
```

---

## Phase 0 ✅ 已完成(2026-04-27)

- [x] SDK 端到端跑通,RandomAgent baseline demo 25 mean RHAE 0.000
- [x] `arc_agent/` 包建立(runner/observation/llm/agents),36 测试全过
- [x] LLMClient 计费 + eval.py `--budget` 硬截止
- [x] 架构决策:Qwen2.5-VL-3B-Instruct,vision encoder 直接处理 512×512 PNG

---

## Phase 1 ✅ 已完成(2026-05-08)

- [x] Python 3.12.10 确认可用,重建 `.venv`
- [x] `pytest tests/ -q` — 36 个测试全过

---

## Phase 2 — 数据管线 + BC 训练(当前)

### 2A — 数据收集

**目标**:为每个训练 game_id 积累足够的 (frame, action) 对。

**来源 1 — 人工 trace**

现有文件:`trace/human/ls20_001.json` — 只有 meta 头,**没有步数据**。
需要录制脚本写入步数据,JSONL 格式如下:

```jsonl
{"__meta__": true, "schema_version": 2, "trace_id": "ls20_human_001", "game_id": "ls20", "annotator": "xin", "outcome": "WIN", "total_steps": 0}
{"step": 0, "level": 1, "state": "NOT_FINISHED", "action": "ACTION3", "action_data": null, "grid": [[0,1,2,...],[...]], "reasoning": "角色在左,向右移"}
{"step": 1, "level": 1, "state": "NOT_FINISHED", "action": "ACTION1", "action_data": null, "grid": [[...]], "reasoning": "..."}
```

字段:
- `grid` — `latest_grid(frame).tolist()`,shape (64,64),值 0–15
- `action` — 字符串 ACTION1–ACTION7
- `action_data` — 仅 ACTION6 需要 `{"x":int,"y":int}`,其余 null
- `reasoning` — 可选,不进入训练 loss

需要实现 `scripts/record_trace.py`:打印 ANSI 彩色 grid → 等待键盘输入 → 执行 → 写一行 JSONL → 循环。

**来源 2 — LLM silver-label(备选,快但质量低)**

```bash
.venv\Scripts\python.exe eval.py --agent llm --games ls20,tr87,wa30 --episodes 5 --tag silver
```
`runs/*.jsonl` 可转成训练数据(缺 reasoning 字段,不影响)。

**数据量目标**:
- 最低:每个训练 game ≥ 5 条完整/接近完整 trace
- 理想:每个训练 game ≥ 20 条,覆盖多条探索路径
- demo 25 取 20 个 game 为 train,固定 5 个 hold-out 为 val

- [ ] 实现 `scripts/record_trace.py`
- [ ] 为至少 5 个 game 录制 ≥5 条 trace(或跑 LLM silver-label)

### 2B — 数据处理(`arc_agent/data/human_traces.py`)

```
trace JSONL (step 行)
  │
  ▼ 1. 读 step 行(跳过 __meta__ 行),提取 (game_id, level, grid_list, action_str, action_data)
  │
  ▼ 2. grid = np.array(grid_list)  →  grid_to_image(grid, scale=8)  →  512×512 PIL Image
  │
  ▼ 3. 构造文本 prompt
  │    "State: NOT_FINISHED | Level 1/5 | Available: ACTION1..ACTION5
  │     Last actions: ACTION3, ACTION1
  │     Hypothesis: [前一步 reasoning 或空]
  │     What is your next action? Reply with: ACTION: <name>"
  │
  ▼ 4. target = "ACTION3"  (action_str)
  │
  ▼ 5. 数据增广 × 8(每条变 8 条):
  │    • 颜色置换:随机打乱 0–15 的颜色索引映射(ARC 颜色无语义)
  │    • 旋转 0°/90°/180°/270°:np.rot90(grid) + Image.rotate
  │         ⚠ ACTION1(Up)/ACTION2(Down)/ACTION3(Left)/ACTION4(Right) 需跟随旋转
  │         ⚠ ACTION6 坐标 (x,y) 需按旋转矩阵变换后 clip 到 [0,63]
  │    • 水平翻转:np.fliplr(grid) + ImageOps.mirror
  │         ⚠ ACTION3(Left) ↔ ACTION4(Right)
  │
  ▼ 6. 输出三元组: (PIL.Image, prompt_str, action_str)
  │
  ▼ 7. 按 game_id 固定划分:
       train_games = [前 20 个 demo game id]  (写死,不随机)
       val_games   = [后 5 个 demo game id]   (永不参与训练)
```

- [ ] 实现 `grid_to_image(grid, scale=8)` in `arc_agent/observation.py`(已有 stub)
- [ ] 实现 `arc_agent/data/human_traces.py`(7 步,纯函数)
- [ ] `tests/test_data_human_traces.py`:三元组形状 + 增广数量 + ACTION6 坐标 + train/val 无交叉

### 2C — BC 训练(`arc_agent/train_bc.py`)

**安装额外依赖**(训练专用,不加 requirements.txt 避免 Kaggle 冲突):
```bash
pip install transformers peft bitsandbytes trl accelerate qwen-vl-utils
```

**代码结构**:
```python
# arc_agent/vlm_backbone.py
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

def load_model(quantize="4bit"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        quantization_config=bnb_config, device_map="auto"
    )
    lora_cfg = LoraConfig(r=8, lora_alpha=16,
                          target_modules=["q_proj","v_proj"],
                          task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    return model, processor
```

```python
# arc_agent/train_bc.py
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="checkpoints/bc",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,   # effective batch = 16
        learning_rate=2e-4,
        num_train_epochs=3,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
    ),
    train_dataset=train_ds,              # HuggingFace Dataset with (image, prompt, action)
    eval_dataset=val_ds,
    data_collator=collator,              # pads sequences, stacks pixel_values
)
trainer.train()
```

**Validation callback**:每 epoch 用 `eval.py` 跑 val 5 games → 记录 mean RHAE → early stop if plateau

**硬件**:RTX A4500 20GB,~10–12GB 使用,~10k 条数据约 5–10 GPU-h

- [ ] 实现 `arc_agent/vlm_backbone.py`:`load_model` + `generate(image, prompt) -> str`
- [ ] 实现 `arc_agent/train_bc.py`:SFTTrainer + RHAE val callback
- [ ] 跑 BC 训练,checkpoint 存 `checkpoints/bc_epoch_N/`
- [ ] val 5 games 评估 — **Hypothesis: mean RHAE ≥ 0.10**

---

## Phase 3 — VLMAgent + 提交

- [ ] 实现 `arc_agent/agents/vlm.py`:加载 BC checkpoint → HEI prompt → `choose()` → parse action
- [ ] `eval.py` 加 `--agent vlm --checkpoint <path>`
- [ ] demo 25 全量评估,结果写 EXPERIMENTS.md
- [ ] Kaggle:验证 4-bit 模式在 T4/L4(16GB)可加载;打包离线 notebook;110 games ≤ 10h
- [ ] **Milestone #1 提交(2026-06-30)**

---

## Phase 4 — 提升(有时间再做)

- [ ] Stage B 离线 RL(PPO + KL 正则到 BC 策略)→ 目标 RHAE ≥ 0.15
- [ ] Stage C 在线 LoRA(前 30% 步收 buffer → mini SGD → 冻结后推理)
- [ ] UNDO + 浅 BFS 前瞻

---

## 参考文献

| # | 内容 | 链接 |
|---|------|------|
| [1] | Qwen2.5-VL 论文(ViT 架构、2D-RoPE、spatial merge) | https://arxiv.org/abs/2502.13923 |
| [2] | QLoRA 论文(4-bit NF4 量化 + LoRA 组合) | https://arxiv.org/abs/2305.14314 |
| [3] | LoRA 原始论文(低秩适配器原理) | https://arxiv.org/abs/2106.09685 |
| [4] | PEFT 库文档(get_peft_model, LoraConfig) | https://huggingface.co/docs/peft/conceptual_guides/lora |
| [5] | BitsAndBytes(NF4 量化实现) | https://huggingface.co/docs/bitsandbytes/main/en/index |
| [6] | TRL SFTTrainer 文档(loss masking, 多模态) | https://huggingface.co/docs/trl/sft_trainer |
| [7] | Qwen2.5-VL-3B-Instruct 模型(HuggingFace) | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct |
| [8] | ARC-AGI-3 Human Dataset 说明 | https://arcprize.org/blog/arc-agi-3-human-dataset |
| [9] | ARC Prize 2026 Kaggle 竞赛页 | https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3 |

---

## 已完成里程碑(append-only)

- 2026-04-27 — Phase 0 搭脚手架:arcengine Python ≥3.12,动作空间确认,API 跑通
- 2026-04-27 — RandomAgent baseline demo 25:mean RHAE 0.000,pass rate 0%
- 2026-04-28 — `arc_agent/` 包:runner/observation/llm/agents,36 测试全过
- 2026-04-28 — LLMClient 累计计费 + eval.py `--budget` 硬截止
- 2026-04-28 — 架构决策:Qwen2.5-VL-3B-Instruct(VLM 视觉编码器解决 2D 空间感知)
- 2026-05-08 — Phase 1 完成:Python 3.12.10 + venv 重建,36 测试全过
