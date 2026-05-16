# 对象提取 + 对象对齐 —— Qwen 视觉能力诊断(v2)

日期: 2026-05-13
状态: 设计稿(实施前请先 review)
目的: 在加任何 RL / scaffold 之前,先看 Qwen2.5-VL-3B 能不能正确"看到对象"+"跨帧追踪对象"。**因为目前没有人类标注,这一轮只让 Qwen 跑结果,把结果格式化给人审,不做自动评分**。

---

## 0. 一句话流程

1. 从 `outputs/baseline_20260511_200835/` 每个游戏取**前 5 张干净 grid 图**(共 4 游戏 × 5 帧 = 20 张)
2. 对每张图,让 Qwen 列出图上的对象 → **提取**
3. 对相邻两帧 (0→1, 1→2, 2→3, 3→4),让 Qwen 把 BEFORE 的对象对应到 AFTER → **对齐**
4. 输出一份 Markdown 报告,每张图旁边贴 Qwen 的 JSON 输出,**人去判断对错**

无 GT,无自动评分。这一轮只看"Qwen 输出长什么样、看起来对不对"。

---

## 1. 概念表

| 概念 | 含义 | 这一轮怎么处理 |
|---|---|---|
| **对象提取**(Extract) | 输入 1 帧,输出"这帧里有哪些对象" | Qwen 跑,输出 JSON |
| **对象对齐**(Align) | 输入 2 帧,输出"哪个对象对应哪个 + 怎么变了" | Qwen 跑,输出 JSON |
| **干净 grid 图** | 64×64 grid 直接用 `grid_to_image(scale=8)` 渲染成 512×512 RGB PNG | 从现有 4 象限 PNG **裁出左上区域**或从 trace 重渲染(见 §3.1) |
| **对象**(Object) | 颜色相同 + 4-连通 的格子集合 | Qwen 自己判定,不强制 ground truth 规则 |
| **bbox** | 包围框 `[row_min, col_min, row_max, col_max]` | Qwen 输出 |
| **cells** | 对象占据的所有 `[row, col]` 坐标 | Qwen 输出(可能不全,看模型能力) |
| **match type** | `moved` / `recolored` / `reshaped` / `disappeared` / `appeared` / `unchanged` | Qwen 自己分类 |
| **delta** | 量化变化:位移 `{dy, dx}`、颜色变化 `{from, to}` | Qwen 输出 |
| **人工验证** | 你拿到报告,对照 PNG 看 Qwen 说的对不对 | 这一轮的"评测"形式 |
| **下一轮才做的** | scipy GT、IoU、P/R/F1 等自动评测 | v2 之后再加 |

---

## 2. 数据范围(具体到帧)

### 2.1 输入数据

| 游戏 | 来源文件夹 | 用前几帧 |
|---|---|---|
| ar25-0c556536 | `outputs/baseline_20260511_200835/ar25-0c556536/` | `step_0000.png` ~ `step_0004.png` |
| bp35-0a0ad940 | 同上 `/bp35-0a0ad940/` | 同上 |
| cd82-fb555c5d | 同上 `/cd82-fb555c5d/` | 同上 |
| cn04-2fe56bfb | 同上 `/cn04-2fe56bfb/` | 同上 |

**注意**:`dc22-fdcac232` 不在这个 baseline 文件夹里(那一轮跑断了),所以只测 4 个游戏。

### 2.2 总调用数

- **提取**: 4 游戏 × 5 帧 = **20 次 Qwen 调用**
- **对齐**: 4 游戏 × 4 对相邻帧 = **16 次 Qwen 调用**
- **总计**: 36 次,每次 ~12-30 s → 约 10-20 分钟 GPU

---

## 3. 关键技术细节

### 3.1 干净 grid 图哪里来

⚠️ **`baseline_20260511_200835/<game>/step_*.png` 不是干净 grid**,而是 4 象限 debug 拼图(`arc_agent/viz.py:compose_step_image`):

```
┌────────── header ──────────┐
│ game_id step=N action=...   │
├─────────────┬───────────────┤
│ grid_now    │ predicted     │  ← 左上是干净的 64×64 grid (4×scale = 256×256)
│  (256×256)  │ overlay       │
├─────────────┼───────────────┤
│ grid_next   │ JSON text     │
└─────────────┴───────────────┘
```

**两种取干净图的办法,选一个**:

| 办法 | 怎么做 | 优缺点 |
|---|---|---|
| **A. 裁左上象限** | 用 PIL 把现有 PNG 裁出 `(0, 40, 256, 296)` 区域 | 立刻可用;但分辨率 256×256(原本喂 Qwen 是 512×512) |
| **B. 从 SDK 重放再渲染** | 调 SDK 重跑这 4 游戏前 5 步,用 `grid_to_image(scale=8)` 直接渲染干净 512×512 PNG | 跟实际 ablation 输入完全一致;但要 5-10 min 跑 SDK,耗 API quota |

**建议办法 B**,这样 Qwen 看到的输入跟 ablation 时一模一样,实验结论才能直接迁移。脚本会把干净 PNG 存到 `outputs/qwen_object_diag/<game>/frame_<N>.png`。

### 3.2 Qwen 提取的 prompt(完整,逐字复制)

#### System prompt

```
You are looking at a single frame from a turn-based grid game. The image
shows a 64x64 grid rendered with one of 16 colors per cell:

  0 = black (background)
  1 = blue
  2 = red
  3 = green
  4 = yellow
  5 = gray
  6 = magenta
  7 = orange
  8 = light blue
  9 = maroon
  10 = purple
  11 = tan
  12 = teal
  13 = lime
  14 = rose
  15 = navy

An "object" is a connected group of cells with the same color, where
"connected" means cells share an edge (up/down/left/right, NOT diagonal).
Black cells (color 0) are background — do NOT include them as objects.

Your job: identify every object in the image and return strict JSON.
```

#### User prompt(每帧一样)

```
List every object in this frame. For each object, give:
  - id:         integer index starting at 0
  - color:      integer 1-15 (skip 0/black/background)
  - color_name: one word from the palette above
  - bbox:       [row_min, col_min, row_max, col_max] (row=y, col=x, 0-indexed, both endpoints inclusive)
  - size:       number of cells in the object
  - description: a short human-readable phrase, e.g. "L-shape", "3x1 horizontal bar", "single cell"

Output strict JSON only — no prose, no markdown:

{
  "objects": [
    {"id": 0, "color": 2, "color_name": "red", "bbox": [10,3,12,5], "size": 5, "description": "L-shape"},
    ...
  ]
}
```

### 3.3 Qwen 对齐的 prompt

#### System prompt

```
You are looking at TWO consecutive frames (BEFORE and AFTER) from a
turn-based grid game. Both frames use the same 64x64 grid with the
16-color palette (black=background).

The first image is BEFORE, the second image is AFTER. Some objects may
have moved, changed color, changed shape, appeared, or disappeared
between the two frames.

Match types:
  - unchanged:    object exists in both frames at the same position with same color and shape
  - moved:        same shape and color, different position
  - recolored:    same shape and position, different color
  - reshaped:     overlapping position, same color, different shape (a few cells added/removed)
  - disappeared:  object in BEFORE has no plausible match in AFTER
  - appeared:     object in AFTER did not exist in BEFORE

Your job: produce the match list. Use 4-connected same-color groups as
objects (same definition as the per-frame extractor).
```

#### User prompt(每对相邻帧一样)

```
For every object in BEFORE find its match in AFTER (or mark it disappeared).
For every NEW object in AFTER that has no match in BEFORE, mark it appeared.

Output strict JSON only:

{
  "matches": [
    {
      "before_id": 0,
      "after_id":  2,
      "type": "moved",
      "color": 2,
      "delta": {"dy": -1, "dx": 0}
    },
    {
      "before_id": 1,
      "after_id":  null,
      "type": "disappeared",
      "color": 3,
      "delta": null
    },
    {
      "before_id": null,
      "after_id":  4,
      "type": "appeared",
      "color": 4,
      "delta": null
    }
  ]
}

Rules for delta:
  - moved:     {"dy": int, "dx": int}   row/col displacement of the centroid
  - recolored: {"from": int, "to": int}  color change
  - reshaped:  {"cells_added": int, "cells_removed": int}
  - unchanged / disappeared / appeared: null

`before_id` and `after_id` refer to indices in each frame's extraction
list. You may either re-extract objects implicitly OR (if I gave you the
extraction output) reuse those ids.
```

---

## 4. 输出格式(给人看的报告)

每个游戏一份 Markdown 报告,长这样:

```
# qwen_object_diag report — ar25-0c556536

## Frame 0
![](frame_00.png)

**Qwen extract output:**

```json
{
  "objects": [
    {"id": 0, "color": 2, "color_name": "red", "bbox": [10,3,12,5], ...},
    ...
  ]
}
```

Human check:
- [ ] all objects found?
- [ ] colors correct?
- [ ] bboxes plausible?
- [ ] sizes plausible?

## Frame 0 → Frame 1
![](frame_00.png) → ![](frame_01.png)

**Qwen align output:**

```json
{"matches": [...]}
```

Human check:
- [ ] all matches correct?
- [ ] match types correct?
- [ ] deltas correct?
```

文件位置: `outputs/qwen_object_diag/<game>/report.md`,图就在同目录。你打开就能边看图边看 JSON 边打勾。

**额外**:再加一份汇总 `outputs/qwen_object_diag/SUMMARY.md`,4 游戏 × 5 帧 × extract + 4 对 × align 总共 36 个结果一页放完,方便快速扫。

---

## 5. 实施清单(等你说"做"才会动)

| # | 任务 | 文件 | 估时 |
|---|---|---|---|
| 1 | 写 `scripts/render_clean_frames.py` —— 用 SDK 重放 4 游戏前 5 步,把 grid 渲染成 512×512 干净 PNG 存到 `outputs/qwen_object_diag/<game>/frame_<N>.png` | new | 0.5h GPU(SDK 调用比较快) |
| 2 | 写 `scripts/qwen_object_diag.py` —— 对每张干净 PNG 跑 extract,对每对相邻 PNG 跑 align,raw JSON 存盘 | new | 写代码 1h + 跑 GPU 15-30 min |
| 3 | 写 `scripts/build_diag_report.py` —— 把 raw JSON + PNG 拼成 Markdown 报告 | new | 0.5h,不需要 GPU |
| 4 | 在文档中写明:**所有 prompt 都从 `arc_agent/prompts.py` 的常量读取**(extract / align prompt 各一个),方便后续微调 | new file `arc_agent/prompts.py` | 0.2h |

总:写代码 ~2h + 跑 ~30 min + 你看报告。

---

## 6. 不承诺的事

- **不做自动评分**(无 GT)
- **不做 IoU / P / R / F1**(下一轮加)
- **不做形状归一化**(旋转/翻转不识别为同一形状)
- **不做合并 / 分裂对齐**(只 1-1 + 出现 + 消失)
- **不裁现有 4 象限 PNG**(用办法 B 重放,因为分辨率要跟 ablation 一致)
- **不动 ablation_overnight_ 里的数据**(那个保留作 ablation 历史)

---

## 7. 你需要 review 的 4 个点

1. **数据范围**: 4 游戏 × 5 帧够不够?要不要每游戏前 10 帧?
2. **Prompt 内容**: §3.2 / §3.3 的 prompt 措辞 OK 吗?颜色映射要列在 system 里吗(或者干脆让 Qwen 自己描述颜色)?
3. **输出格式**: §4 的 Markdown 报告够用吗?要不要 HTML 单页能交互勾选?
4. **是否走办法 B(重放 SDK)**: 同意花 5-10 min API 调用换 512×512 高分辨率干净图?还是直接裁 256×256 拼图够?

review 完告诉我哪个要改 / 哪个 OK,我再动 §5 的代码。

---

## 8. 人工验证结果 (2026-05-13,cn04 游戏 5 帧)

跑完 EXP-1/EXP-2 后人工对照 PNG 检查 Qwen 输出。下面是评测表。**"正确"按严格标准:JSON 不光能解析,还要和图片实际变化/物体一致**。

### 8.1 文件级评测

| 文件 | 类型 | JSON 格式 | 预测内容 | 实际内容 | 是否正确 | 主要错误 |
|---|---|---:|---|---|---:|---|
| `pair_00_01.json` | 帧间对齐 | ✅ | 有移动/消失/出现 | frame_00 和 frame_01 无变化 | ❌ | 把无变化误判成变化 |
| `pair_01_02.json` | 帧间对齐 | ✅ | `dy=-1`,还有消失/出现 | 灰色物体和黄色 L 整体上移约 3 格 | ❌ | 位移量错,变化类型也错 |
| `pair_02_03.json` | 帧间对齐 | ✅ | 有移动/消失/出现 | 无变化 | ❌ | 把无变化误判成变化 |
| `pair_03_04.json` | 帧间对齐 | ✅ | 有移动/消失/出现 | 无变化 | ❌ | 把无变化误判成变化 |
| `frame_00.json` | 单帧提取 | ✅ | 2 个对象:red L、rose cell | 至少应有灰色物体、黄色 L、浅黄 L、紫色墙、底部灰线等 | ❌ | 颜色、位置、数量都错 |
| `frame_01.json` | 单帧提取 | ✅ | 同上 | 同 frame_00 | ❌ | 颜色、位置、数量都错 |
| `frame_02.json` | 单帧提取 | ✅ | 同上 | 灰色物体和黄色 L 已上移 | ❌ | 没反映真实位置 |
| `frame_03.json` | 单帧提取 | ✅ | 同上 | 同 frame_02 | ❌ | 没反映真实位置 |
| `frame_04.json` | 单帧提取 | ✅ | 同上 | 同 frame_02 | ❌ | 没反映真实位置 |

### 8.2 总体正确率

| 指标 | 结果 |
|---|---:|
| JSON 可解析率 | **9 / 9 = 100%** |
| 文件内容严格正确率 | **0 / 9 = 0%** |
| 帧间对齐文件正确率 | **0 / 4 = 0%** |
| 单帧物体提取文件正确率 | **0 / 5 = 0%** |

### 8.3 帧间"是否有变化"的粗粒度准确率

只看它有没有判断"这两帧发生变化",不看 motion 细节:

| 帧对 | 实际是否变化 | JSON 是否预测变化 | 是否正确 |
|---|---:|---:|---:|
| frame_00 → frame_01 | 否 | 是 | ❌ |
| frame_01 → frame_02 | 是 | 是 | ✅ |
| frame_02 → frame_03 | 否 | 是 | ❌ |
| frame_03 → frame_04 | 否 | 是 | ❌ |

| 指标 | 数值 |
|---|---:|
| 是否变化分类准确率 | **1 / 4 = 25%** |
| 变化预测 Precision | **1 / 4 = 25%** |
| 变化预测 Recall | **1 / 1 = 100%** |
| 无变化识别准确率 | **0 / 3 = 0%** |

### 8.4 Verdict

```
格式正确,但语义标注基本不可用。
```

最致命的问题不是 JSON 坏了,而是 Qwen **总是倾向于编造变化** —— 即使两帧完全一样,它也输出 moved / disappeared / appeared。**假阳性率 75%**(3 个无变化帧对里 3 个都被误报为有变化)。这种 false-positive 的偏差对 RL agent 是有毒的:它会让 reward signal 永远非零,agent 学到的全是噪声。

**结论**:Qwen-VL-3B 的视觉解析层(extract + align)在 ARC-AGI-3 任务上**不能作为可靠的对象状态分析模块**。继续在 prompt 层调整或换更大 backbone 都只能边际改善,治标不治本。

---

## 9. 新方案 —— 把视觉解析从 Qwen 卸下来,改用确定性 CV

<font color="blue">

### 9.1 模型 / 方法选择

**对象提取**:`scipy.ndimage.label` —— 按颜色 + 4-连通做连通块分析。
- 没有训练数据需求(纯算法)
- 在"同色 4-连通块"这个定义下**输出 100% 准确**(因为这就是数学定义,不是感知判断)
- 推理延迟 < 1 ms,比 Qwen 快约 10000 倍
- 输出字段完全可控:`color`、`cells`、`bbox`、`center`、`size`

**对象对齐**:Hungarian algorithm + 自定义 cost function —— `scipy.optimize.linear_sum_assignment`。
- cost = `α·shape_mismatch + β·color_mismatch + γ·distance + δ·size_diff`
- 形状权重最高(α 大)→ 优先匹配同形状对象;距离权重小(γ 小)→ 容忍位置变化
- 输出每个 match 的 `(before_id, after_id, type, delta)`,跟现在 Qwen-align 的 schema 完全一致

**Qwen 的新职责**:**只做 action selection**。Play agent 拿到 user prompt 包含已经算好的 `[OBJECTS]` 和 `[MATCHES]` 块,基于这些事实选下一步动作。Qwen 不再做"看图猜对象"的事。

### 9.2 为什么不上 YOLO / ByteTrack

YOLO 是好东西,但在 64×64 grid 上是**杀鸡用牛刀**:

| 维度 | scipy 方案 | YOLO11n-seg + ByteTrack |
|---|---|---|
| 训练数据 | 不需要 | 要标注 ARC 颜色块(没有数据集) |
| 准确率 | 100%(数学定义) | 看训练质量,大概率 < 100% |
| 推理延迟 | < 1 ms | ~5-10 ms |
| 部署复杂度 | 一个函数 | 模型文件 + tracker 配置 |
| Kaggle 兼容 | scipy 已在 requirements | YOLO 要额外打包 |

YOLO 适用于:跨颜色组合对象、形状先验对齐、非彩色块画面。**当前 ARC-AGI-3 demo 25 个游戏没有这些复杂度**,日后升级到 ARC-AGI-2 静态题目或者出现新类型对象再考虑。

### 9.3 实施计划

| # | 任务 | 文件 |
|---|---|---|
| 1 | `arc_agent/object_extractor.py`:`extract_objects(grid: np.ndarray) -> list[ObjectRecord]`,内部用 `scipy.ndimage.label` | new |
| 2 | `arc_agent/object_aligner.py`:`align_objects(before, after) -> list[Match]`,用 Hungarian + cost function | new |
| 3 | `tests/test_object_extractor.py` / `tests/test_object_aligner.py`:在合成小 grid 上验证 100% 正确 | new |
| 4 | **回测**:用同一份 5 帧 PNG 跑新流水线,人工核对,目标 100% 正确率 | rerun |
| 5 | `build_play_prompt()` 增加 `[OBJECTS]` + `[MATCHES]` 两块,内容来自 (1)(2) 的输出文本化 | edit `prompts.py` + agent 各文件 |
| 6 | 删 `arc_agent/world_model.py` + `_run_reflection`(Reflection agent 不再需要) | cleanup |

预估:写代码 + 测试 半天;回测 30 分钟(没 GPU 调用,纯 CPU)。

### 9.4 验证标准(比上次更严)

按 §8.1 同样的人工评测格式,跑完后预期:

| 指标 | 现在(Qwen) | 新方案(scipy + Hungarian)预期 |
|---|---:|---:|
| 文件内容严格正确率 | 0/9 = 0% | **9/9 = 100%** |
| 是否变化分类准确率 | 1/4 = 25% | **4/4 = 100%** |
| 假阳性率 | 75% | **0%** |

如果回测达不到 100%,说明 grid 上有跨颜色组合对象 / 旋转对称 / 其他需要语义判断的情况 —— 那时候再考虑加 YOLO 或回 Qwen。

### 9.5 Qwen 还能做什么(角色重定位)

| 模块 | 原计划 | 新计划 |
|---|---|---|
| 视觉对象提取 | Qwen extract | **scipy(确定性)** |
| 跨帧对象对齐 | Qwen align | **Hungarian(确定性)** |
| 规则推理 | Qwen reflection 写 `rules` | Qwen 在 Play prompt 里直接基于 `[OBJECTS]+[MATCHES]` 文本推理 |
| 动作选择 | Qwen Play | **Qwen Play(保留)** |
| Goal 假设 | Qwen reflection 写 `goal` | Qwen 在 Play prompt 里基于 `[MATCHES]` 累积观察推断 |

Qwen 从"视觉 + 推理 + 决策"三合一被卸成"只做推理 + 决策",对它本来就不擅长的视觉解析任务就此放手。这跟 2025 ARC Prize 前 3 名(NVARC / ARChitects / MindsAI)的混合架构方向一致 —— 没有人是纯 VLM。

</font>

