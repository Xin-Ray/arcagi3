# HUMAN_TRACE_GUIDE — 人工 think-aloud 标注操作手册

最近一次更新:2026-04-28(schema_version 升至 2:主视觉字段改为图像路径 `frame_image_path`,适配 Qwen2.5-VL-3B-Instruct 多模态输入;`frame_text` 降为可选备用字段)
适用阶段:Phase 1.5 对照实验(与 Phase 1 LLMAgent 评估并行)
配套登记:`RESEARCH.md` 提案 #4(待补) / `EXPERIMENTS.md` 占位条目(待补)
本文件**不是**七件套之一,只是操作手册。一切方法论判断仍以 RESEARCH/ARCHITECTURE 为准。

---

## 1. 目的

本手册支撑一个**可证伪的对照实验**:

> 用 ~10 条人工 think-aloud trace 对 Qwen2.5-VL-3B-Instruct 做小规模 BC,在 hold-out demo 集上的平均 RHAE 比同等数量的 Claude 自动 silver-label trace 高 ≥0.05。

- 假设成立 → `ARCHITECTURE.md` §训练范式里的 silver-label(Claude 标注)路径降级为 fallback。
- 假设被证伪 → 保持现状,把 Claude silver-label 当主路径继续做。

**为什么 schema 升版(v1 → v2)**:原 schema 以 `frame_text`(hex 字符网格)为主视觉字段,目标模型是文本模型 Qwen3-0.6B。架构切换为 **Qwen2.5-VL-3B-Instruct** 后,模型输入是图像+文本;训练数据的主视觉字段必须是 **PNG 图像路径**(`frame_image_path`)。`frame_text` 保留为可选字段,供 ablation 研究(text-only 对照)使用。

**HEI 规则**:正式假设和 iteration trigger 必须先在 `RESEARCH.md` 提案 #4 锁住,**才**能开始批量标注。在那之前可以先标 1-2 条试格式,但不要刷量。

---

## 2. 任务概述

1. 选 5-10 个 demo 游戏(参见 §3)
2. 每关玩一遍,尽量打到 WIN 或自然 GAME_OVER(timeout 也行)
3. **每一步**写 think-aloud:动作前 `hypothesis`+`intent`,动作后 `observed`(+ 可选 `rule_update`)
4. 每步自动保存/截图当前网格为 PNG → 路径写入 `frame_image_path`

预算:50 步以内的关卡每条 ~30-40 分钟。10 条 trace ≈ 5-7 小时人时。

---

## 3. 选哪些游戏

- 必含 `ls20`(已知 LLMAgent 失败案例,baseline 在 `EXPERIMENTS.md`)
- 优先 `EXPERIMENTS.md` baseline 表里 RHAE=0 的游戏(对照实验信号最强)
- 至少跨 5 个不同 `game_id`,不要全选规则相似的
- 选定后写到 `EXPERIMENTS.md` 占位条目里(锁住游戏列表,事后不能换)

---

## 4. 标注格式

### 4.1 文件位置与命名

```
traces/
└── human/
    ├── ls20_001.jsonl
    ├── ls20_002.jsonl       # 同一游戏多次玩,序号递增
    ├── tr87_001.jsonl
    └── images/              # 每步网格 PNG(由录入工具自动生成)
        ├── ls20_001/
        │   ├── step_001.png
        │   ├── step_002.png
        │   └── ...
        └── tr87_001/
            └── ...
```

`runs/` 是 eval 输出,**不要**把人工 trace 混进去。

图像路径约定:`images/<trace_id>/step_<NNN>.png`(三位零填充),相对于 `traces/human/`。

### 4.2 文件结构(JSONL,每行一条 JSON)

**第 1 行 — metadata**(`__meta__: true`):

```json
{"__meta__": true,
 "schema_version": 2,
 "trace_id": "ls20_human_001",
 "game_id": "ls20",
 "annotator": "<your handle>",
 "started_at": "2026-04-28T10:00",
 "ended_at": "2026-04-28T10:35",
 "outcome": "WIN",
 "levels_completed": 1,
 "total_steps": 47,
 "notes": "第 1 关通,第 2 关时间不够"}
```

**第 2 行起 — 每步一条**:

```json
{"step": 7,
 "level": 0,
 "frame_image_path": "images/ls20_human_001/step_007.png",
 "frame_text": null,
 "diff_from_prev": [[5, 10, 0, 3], [5, 11, 0, 3]],
 "available_actions": ["ACTION1","ACTION3","ACTION4","ACTION5"],
 "hypothesis": "ACTION5 是拾取键,只在玩家相邻 8 格内对黄块有效",
 "intent": "先左移一格,把玩家挪到黄块相邻范围,再 ACTION5 验证 8 格半径假设",
 "action": "ACTION3",
 "action_data": null,
 "observed": "玩家左移成功,黄块未消失 — 假设暂未充分检验,需再走一步 ACTION5",
 "rule_update": null}
```

### 4.3 字段来源

| 字段 | 来源 | 说明 |
|------|-----|------|
| `step`, `level` | 工具 | 自动 |
| `frame_image_path` | 录入工具(`grid_to_image`) | **主视觉字段**。相对 `traces/human/` 的 PNG 路径;Qwen2.5-VL-3B 训练时读这个文件 |
| `frame_text` | `arc_agent.observation.grid_to_text` | **可选**。hex 文本网格;方式 B 若有余力可填,用于 text-only ablation;`null` 完全合法 |
| `diff_from_prev` | `arc_agent.observation.grid_diff` | `[[row,col,old,new],...]`;首步为 `[]`;标注时帮助你定位变化了什么 |
| `available_actions` | `arc_agent.observation.available_action_names` | 当前可用动作名列表 |
| `action`, `action_data` | 工具(按你的输入) | `"ACTION1"`–`"ACTION7"`;ACTION6 才有 `action_data: {"x":..,"y":..}` |
| `hypothesis` | **你写** | 当前对规则的假设,可证伪、具体、动词性 |
| `intent` | **你写** | 这一步**为什么**选这个动作 |
| `observed` | **你写** | 动作后**看到了什么**(尤其意外) |
| `rule_update` | **你写,可空** | 仅当真的修正了规则信念时填 |

**v1 → v2 变更摘要**:
- `frame_text`(原必填)→ 现在 `null` 合法,降为可选
- `frame_image_path`(原无)→ 现在**必填**,PNG 路径
- `diff_from_prev` 格式从字符串改为 `[[row,col,old,new],...]` 数组(机器可直接读)
- `schema_version: 1` → `2`

---

## 5. 写作要点(HEI 三段式)

`hypothesis`/`intent`/`observed` 是 **Qwen2.5-VL-3B-Instruct** 要模仿的核心文体,质量决定实验上限。模型同时看到网格图像和你的文字,所以文字中**不需要描述网格长什么样**,只需写推理过程。

### Hypothesis — 写假设,不写情绪

- ✓ "ACTION5 在玩家相邻 8 格(含对角)内对黄块有效;更远无响应"
- ✗ "我感觉应该是拾取吧"
- **可证伪**(能被某个具体动作的结果反驳)
- 多条假设并存可编号:`1) ... 2) ...`

### Intent — 写目的,不写动作字面

- ✓ "左移一格让玩家进入黄块相邻范围,下一步用 ACTION5 验证 8 格半径假设"
- ✗ "按 A"
- 必须回答:这一步**推进目标**(exploitation)还是**验证假设**(exploration)?

### Observed — 写差异,不写流水账

- ✓ "玩家左移成功,但黄块未消失 — 推翻了'移动即自动拾取';ACTION5 才是拾取动作"
- ✗ "玩家动了"
- 重点:**预期 vs 现实**,意外比正常更值钱
- **不要描述图像本身的颜色/位置**;模型自己能看到图像,你的价值在于推理链

### Rule_update — 不要滥用

- 多数步骤为 `null`
- 写时带版本号:`"R3 v2: 拾取条件从'相邻'改为'相邻 8 格含对角'"`
- 一关里 `rule_update` 出现 >5 次说明假设太松,该重写 hypothesis

---

## 6. 录入工作流

### 方式 A — 浏览器手玩 + 手动截图(MVP,试格式用)

1. 在 https://three.arcprize.org 玩(需 `ARC_API_KEY` 登录)
2. 一半屏幕浏览器,一半编辑器
3. 每动一步:
   a. 截图当前网格区域 → 保存到 `traces/human/images/<trace_id>/step_NNN.png`
   b. 手写 jsonl 一行,`frame_image_path` 填上述路径,`frame_text` 填 `null`
4. `diff_from_prev` 可视觉估算填写,或首步填 `[]`

适合**先标 1-2 条**确认你认同 schema,再上方式 B。

### 方式 B — `record_human.py` 录入工具(实验正式开跑用)

**现状:未实现**(待 LIBRARY 落位 + 用户点头后实现)。

设计概要(实现前不在此固化细节,以 LIBRARY.md 为准):

1. SDK `env.reset()` 获取首帧
2. 调用 `grid_to_image(latest_grid(frame))` → 保存为 `step_NNN.png`,路径写入记录
3. 终端用 ANSI 16 色块打印网格概览(低保真,供快速定位颜色分布)
4. 打开系统图像查看器显示实际 PNG(Windows: `os.startfile(path)`)
5. 提示输入 `hypothesis` + `intent`
6. 提示按键选动作(`W/A/S/D` = ACTION1-4,`E` = ACTION5,`C x y` = ACTION6,`U` = ACTION7)
7. SDK `env.step()` 执行动作,获取新帧
8. 调用 `grid_diff(prev_grid, latest_grid(new_frame))` → 填 `diff_from_prev`
9. 显示新帧图像 + diff 摘要
10. 提示输入 `observed`(+ 可选 `rule_update`)
11. 追加写入 jsonl;循环至 WIN / GAME_OVER / 手动退出

ANSI 16 色块打印是备用的低保真视图,**不能替代 PNG**;PNG 才是 VLM 训练的实际输入。

---

## 7. 收工 checklist

**单条 trace 完成后**:
- [ ] JSONL 文件落在 `traces/human/<game_id>_NNN.jsonl`
- [ ] metadata 行 `schema_version: 2`,所有字段填满,日期 ISO 8601
- [ ] 每步 `frame_image_path` 对应的 PNG 文件存在于 `traces/human/images/<trace_id>/`
- [ ] 每步 `hypothesis`+`intent`+`observed` 三字段非空(`rule_update` / `frame_text` 可空)
- [ ] 在 `EXPERIMENTS.md` 占位条目把进度 ++

**全部 10 条完成后**:
- [ ] 在 `RESEARCH.md` 提案 #4 标"标注完成,等 BC 训练"
- [ ] **不要**写 `PAPER.md` — 训练实验出数后才写(PAPER 不引入新事实)
- [ ] 确认 `traces/human/` 已加入 `.gitignore`(图像文件大,不入版本控制)

---

## 8. 反模式(不要做)

- ❌ 沿用 `schema_version: 1` 的 trace — v1 缺 `frame_image_path`,训练时会报错
- ❌ `frame_image_path` 填绝对路径 — 路径必须相对于 `traces/human/` 才能跨机器复现
- ❌ 标到一半改 schema — 一旦定就锁本批次,改动开 `schema_version: 3`
- ❌ 同一游戏刷 N 条凑数 — 至少跨 ≥5 个不同 `game_id`
- ❌ Hypothesis 写"看起来可能" — 不可证伪 = 没用
- ❌ Observed 里描述图像颜色/位置("左上角有个蓝块") — 模型能看图,你的价值在推理链
- ❌ 只标 WIN trace — GAME_OVER 同样有用,"我假设错在哪"反而是最值钱的训练样本
- ❌ 跳过失败步骤、事后剪辑 — 错的动作和对的动作都全程录

---

## 9. 配套登记(本文档完成后下一步)

按 docs/ 维护规则,本手册落位后还差两步,等你点头再做:

1. `docs/RESEARCH.md` 加**提案 #4**:正式 hypothesis(具体数字)+ iteration trigger(成功/失败/不显著三分支)。
2. `docs/EXPERIMENTS.md` 加**占位条目**:实验编号、选定游戏列表、标注进度计数器。

`record_human.py` 是再下一步,等 §6 方式 B 设计在 ARCHITECTURE / LIBRARY 落位后开工。同时需要先把 `grid_to_image()` 实现并入库(`arc_agent/observation.py`,见 LIBRARY.md 登记条目)。
