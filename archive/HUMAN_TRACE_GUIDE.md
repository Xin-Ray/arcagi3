# HUMAN_TRACE_GUIDE — 人工 think-aloud 标注操作手册

最近一次更新:2026-05-08(8 项改进:新增完整三步示例 §4.4、新增 VLM 训练 prompt 格式 §5.0、Way A 截图规范、每步自检清单 §5.5、修正 §9 过时信息、标注时序说明、diff 填写规范、反模式补充)
适用阶段:Phase 2A 数据收集
配套登记:`RESEARCH.md` 方案 #5 / `EXPERIMENTS.md` Exp #4/#5
本文件**不是**七件套之一,只是操作手册。一切方法论判断仍以 RESEARCH/ARCHITECTURE 为准。

---

## 1. 目的

本手册支撑一个**可证伪的对照实验**:

> 用 ~10 条人工 think-aloud trace 对 Qwen2.5-VL-3B-Instruct 做小规模 BC,在 hold-out demo 集上的平均 RHAE 比同等数量的 Claude 自动 silver-label trace 高 ≥0.05。

- 假设成立 → `ARCHITECTURE.md` §训练范式里的 silver-label(Claude 标注)路径降级为 fallback。
- 假设被证伪 → 保持现状,把 Claude silver-label 当主路径继续做。

**关于训练目标与 RHAE 权重**:训练损失是 action token 的 CE loss。但每条 trace 的权重 = `min(1, h/total_steps)²`(即该 trace 的 RHAE 分数)。因此：
- **必须尽量打赢**:WIN trace 且接近人类步数 → 权重 ≈ 1.0;打了很多冤枉步才赢 → 权重低;不赢 → 权重 0(仅保留用于未来 RL)。
- **RHAE 在标注结束时由工具自动计算**,你只需填 `total_steps` + `outcome`。

**HEI 规则**:正式假设和 iteration trigger 必须先在 `RESEARCH.md` 方案 #5 锁住,**才**能开始批量标注。在那之前可以先标 1-2 条试格式,但不要刷量。

---

## 2. 任务概述

1. 选 5-10 个 demo 游戏(参见 §3)
2. 每关玩一遍,**尽量打赢**;至少完成 level 1(否则 BC 权重为 0)
3. **每一步**写 think-aloud:动作前 `hypothesis`+`intent`,动作后 `observed`(+ 可选 `rule_update`)
4. 每步保存当前网格 PNG(Way A 手动截图，Way B 工具自动)→ 路径写入 `frame_image_path`
5. 标完后在 `EXPERIMENTS.md` 占位条目里把进度 ++

预算:50 步以内的关卡每条 ~30-40 分钟。10 条 trace ≈ 5-7 小时人时。

---

## 3. 选哪些游戏

- 必含 `ls20`(已知 LLMAgent 失败案例,baseline 在 `EXPERIMENTS.md`)
- 优先 `EXPERIMENTS.md` baseline 表里 RHAE=0 的游戏(对照实验信号最强)
- 至少跨 5 个不同 `game_id`,不要全选规则相似的
- **每个 game 至少收两条策略不同的 WIN trace**(一条只学一条路径,两条不同路径才让模型知道"规则有多种通法")
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
    └── images/              # 每步网格 PNG
        ├── ls20_001/
        │   ├── step_001.png
        │   ├── step_002.png
        │   └── ...
        └── tr87_001/
            └── ...
```

`runs/` 是 eval 输出,**不要**把人工 trace 混进去。

图像路径约定:`images/<trace_id>/step_<NNN>.png`(三位零填充),相对于 `traces/human/`。

**PNG 规格(Way A 截图必须满足)**:
- 只截游戏网格区域,不含浏览器 chrome / 按钮栏
- 分辨率不限,但推荐与 `grid_to_image(scale=8)` 的输出一致:**512×512 像素**
- 文件名严格按 `step_NNN.png`(三位,001 起)

### 4.2 文件结构(JSONL,每行一条 JSON)

**第 1 行 — metadata**(`__meta__: true`):

```json
{"__meta__": true,
 "schema_version": 2,
 "trace_id": "ls20_human_001",
 "game_id": "ls20",
 "annotator": "<your handle>",
 "started_at": "2026-05-08T10:00",
 "ended_at": "2026-05-08T10:35",
 "outcome": "WIN",
 "levels_completed": 1,
 "total_steps": 47,
 "rhae": 0.42,
 "notes": "第 1 关通,第 2 关时间不够"}
```

> `rhae` 由工具计算:`min(1, h/total_steps)²`，h 从 scorecard `level_baseline_actions` 取。手动标注时可先填 `null`，标完后补算。

**第 2 行起 — 每步一条**:

```json
{"step": 7,
 "level": 1,
 "frame_image_path": "images/ls20_human_001/step_007.png",
 "frame_text": null,
 "diff_from_prev": [[5, 10, 0, 3], [5, 11, 0, 3]],
 "available_actions": ["ACTION1","ACTION3","ACTION4","ACTION5"],
 "hypothesis": "ACTION5 是拾取键,只在玩家相邻 8 格内对黄块有效",
 "intent": "左移一格让玩家进入黄块相邻范围,下一步用 ACTION5 验证 8 格半径假设 [exploration]",
 "action": "ACTION3",
 "action_data": null,
 "observed": "玩家左移成功,黄块未消失 — 假设暂未充分检验,需再走一步 ACTION5",
 "rule_update": null,
 "quality": 1}
```

### 4.3 字段来源

| 字段 | 来源 | 说明 |
|------|-----|------|
| `step`, `level` | 工具 | 自动 |
| `frame_image_path` | 录入工具(`grid_to_image`) | **主视觉字段**。相对 `traces/human/` 的 PNG 路径;VLM 训练时读这个文件 |
| `frame_text` | `arc_agent.observation.grid_to_text` | **可选**。hex 文本网格;`null` 完全合法 |
| `diff_from_prev` | `arc_agent.observation.grid_diff` | `[[row,col,old,new],...]`;首步为 `[]`;Way A 见 §6 填写规则 |
| `available_actions` | `arc_agent.observation.available_action_names` | 当前可用动作名列表 |
| `action`, `action_data` | 工具(按你的输入) | `"ACTION1"`–`"ACTION7"`;ACTION6 才有 `action_data: {"x":..,"y":..}` |
| `hypothesis` | **你写,动作前** | 当前对规则的假设,可证伪、具体、动词性 |
| `intent` | **你写,动作前** | 为什么选这个动作;末尾标 `[exploration]` 或 `[exploitation]` |
| `observed` | **你写,动作后** | 看到了什么(尤其意外);预期 vs 现实 |
| `rule_update` | **你写,动作后,可空** | 仅当真的修正了规则信念时填 |
| `quality` | **你写,可选** | `1`=好棋 / `0`=中性 / `-1`=事后认为是错的;用于训练时 per-step 加权 |

**v1 → v2 变更摘要**:
- `frame_text`(原必填)→ 现在 `null` 合法,降为可选
- `frame_image_path`(原无)→ 现在**必填**,PNG 路径
- `diff_from_prev` 格式从字符串改为 `[[row,col,old,new],...]` 数组
- 新增 `rhae`(meta)和 `quality`(步级)字段

---

### 4.4 完整三步示例(ls20 开局,读完直接抄格式)

> 背景:ls20 第 1 关,玩家(蓝色)在左下角,疑似目标(黄色)在右上角,灰色物体散布。

**meta 行**:
```json
{"__meta__": true, "schema_version": 2, "trace_id": "ls20_human_001",
 "game_id": "ls20", "annotator": "xin", "started_at": "2026-05-08T14:00",
 "ended_at": null, "outcome": null, "levels_completed": null,
 "total_steps": null, "rhae": null, "notes": "recording in progress"}
```

**step 0** — 首步,无上一帧,diff 为空:
```json
{"step": 0, "level": 1,
 "frame_image_path": "images/ls20_human_001/step_001.png",
 "frame_text": null,
 "diff_from_prev": [],
 "available_actions": ["ACTION1","ACTION2","ACTION3","ACTION4"],
 "hypothesis": "玩家是蓝色块;黄色是目标。ACTION1-4 是方向键(Up/Down/Left/Right)。胜利条件未知,先移向黄色测试",
 "intent": "向右移动一步,观察蓝色是否跟随指令移动,验证 ACTION4=Right [exploration]",
 "action": "ACTION4",
 "action_data": null,
 "observed": "蓝块向右移动了 1 格,符合预期 — 确认 ACTION4=Right,方向映射正确",
 "rule_update": "R1 v1: ACTION4=Right(已验证)",
 "quality": 1}
```

**step 1** — 上一步移动后:
```json
{"step": 1, "level": 1,
 "frame_image_path": "images/ls20_human_001/step_002.png",
 "frame_text": null,
 "diff_from_prev": [[32, 10, 1, 0], [32, 11, 0, 1]],
 "available_actions": ["ACTION1","ACTION2","ACTION3","ACTION4"],
 "hypothesis": "ACTION4=Right 确认。现在蓝块(玩家)需要到达黄块 — 但不知道是否要走到正上/正下/相邻才触发。先走到同行再接近测试",
 "intent": "继续向右逼近黄块,路径最短;这步是 exploitation — 若触发 WIN 则假设'到达即胜'确认 [exploitation]",
 "action": "ACTION4",
 "action_data": null,
 "observed": "蓝块又右移 1 格,黄块未消失、无新事件 — '到达即胜'未验证,需继续接近",
 "rule_update": null,
 "quality": 1}
```

**step 2** — 蓝块与黄块同行相邻:
```json
{"step": 2, "level": 1,
 "frame_image_path": "images/ls20_human_001/step_003.png",
 "frame_text": null,
 "diff_from_prev": [[32, 11, 1, 0], [32, 12, 0, 1]],
 "available_actions": ["ACTION1","ACTION2","ACTION3","ACTION4"],
 "hypothesis": "蓝块现在与黄块相邻。如果'碰到即胜',下一步 ACTION4 应触发 WIN。如果没有,说明需要额外动作(ACTION5?)或需要从特定方向进入",
 "intent": "向右踏上黄块位置,直接检验'碰到黄色 = WIN'假设;若失败则转 exploration [exploitation]",
 "action": "ACTION4",
 "action_data": null,
 "observed": "蓝块移入黄色格子,level 1 通关!WIN 触发 — '到达黄块格子即胜'确认",
 "rule_update": "R2 v1: 胜利条件 = 玩家移入黄色格子",
 "quality": 1}
```

> **注意**:以上 `diff_from_prev` 是根据位移手填的近似值。Way A 允许估填或留 `[]`；Way B 工具自动精确填写。

---

## 5. 写作要点(HEI 三段式)

### 5.0 先看模型实际看到的 prompt(理解这个,后面的规则自然成立)

你的标注最终被拼成这样的训练样本送给 VLM:

```
[system]
You are an AI agent playing a turn-based puzzle game.
You see the game grid as an image. Each color represents a different game element.
No instructions are given — you must infer the rules by observing the grid.
Think step by step using Hypothesis → Execute → Iterate reasoning.
Always end your reply with exactly: ACTION: <action_name>

[user]
<image: 512×512 PNG — 这就是你保存的 step_NNN.png>

State: NOT_FINISHED | Level: 1/7
Available: ACTION1 ACTION2 ACTION3 ACTION4
Last actions: ACTION4, ACTION4
Hypothesis: 蓝块是玩家;黄块是目标;ACTION1-4 是方向键     ← 来自你的 hypothesis 字段
What is your next action? Reply with: ACTION: <name>

[assistant — 训练 target]
ACTION: ACTION4                                              ← 来自你的 action 字段
```

**结论**:
- 模型**看到完整图像**,所以你不需要在 hypothesis/intent/observed 里描述颜色或位置
- `hypothesis` 字段直接出现在 prompt 里,质量=模型输入质量
- `intent` 和 `observed` **不进入 prompt**,但你写它们是为了保证 hypothesis 是真实推理的结果,而不是事后凑字

---

### 5.1 标注时序(动作前/后各写什么)

```
看到新帧
  │
  ├─ 写 hypothesis  (现在我认为游戏规则是什么?)
  ├─ 写 intent      (这一步为什么选这个动作?)
  │
  按键/执行动作
  │
  ├─ 写 observed    (实际发生了什么?预期 vs 现实)
  └─ 写 rule_update (仅当信念真的改变时填)
```

**不能颠倒顺序**:先按键再补写 hypothesis 会变成事后合理化,失去 think-aloud 的价值。

---

### 5.2 Hypothesis — 写假设,不写情绪

- ✓ `"ACTION5 在玩家相邻 8 格(含对角)内对黄块有效;更远无响应"`
- ✗ `"我感觉应该是拾取吧"`
- **可证伪**:能被某个具体动作的结果反驳
- 多条假设并存可编号:`"1) 黄色=目标 2) ACTION5=拾取"`

---

### 5.3 Intent — 写目的,末尾标 exploration/exploitation

- ✓ `"左移一格让玩家进入黄块相邻范围,下一步用 ACTION5 验证 8 格半径假设 [exploration]"`
- ✗ `"按 A"`
- 必须回答:这一步**推进目标**(exploitation)还是**验证假设**(exploration)?
- 加 `[exploration]` 或 `[exploitation]` 后缀(训练时可用于课程学习或 reward shaping)

---

### 5.4 Observed — 写差异,不写流水账

- ✓ `"玩家左移成功,但黄块未消失 — 推翻了'移动即自动拾取';ACTION5 才是拾取动作"`
- ✗ `"玩家动了"`
- 重点:**预期 vs 现实**,意外比正常更值钱
- **不要描述图像本身的颜色/位置**:模型自己能看图,你的价值在推理链

---

### 5.5 每步完成后的自检清单(60 秒内过完)

写完一步后,快速检查这 4 个问题。全部 ✓ 再按下一步:

- [ ] **Hypothesis 可证伪?** — "能被下一步结果反驳的具体陈述" vs "感觉/情绪/模糊描述"
- [ ] **Intent 说了为什么,不只是说了什么?** — "向右移向目标验证接触胜利 [exploitation]" vs "向右走"
- [ ] **Observed 写了预期 vs 现实的差距?** — 即使结果符合预期也要写"符合 R2 v1 的预测"
- [ ] **有没有在 hypothesis/observed 里描述图像?** — 如果有"左上角蓝块"这类词,删掉

---

### 5.6 Rule_update — 不要滥用

- 多数步骤为 `null`
- 写时带版本号:`"R3 v2: 拾取条件从'相邻'改为'相邻 8 格含对角'"`
- 一关里 `rule_update` 出现 >5 次说明假设太松,该重写 hypothesis

---

## 6. 录入工作流

### 方式 A — 浏览器手玩 + 手动截图(MVP,试格式用)

**适合先标 1-2 条确认格式,再换方式 B。**

1. 在 https://three.arcprize.org 玩(需 `ARC_API_KEY` 登录,见 [arcprize.org/api-keys](https://arcprize.org/api-keys))
2. 屏幕左半浏览器,右半文本编辑器(VS Code 或记事本)

3. **动作前**:在编辑器里写好当前步的 `hypothesis` + `intent`(不能在按键后补写)

4. **执行动作**,然后立即:
   a. **截图网格区域** — 只截游戏网格,不含浏览器工具栏
      - Windows 推荐:`Win + Shift + S` → 矩形截图 → 保存为 `step_NNN.png`
      - 目标分辨率:512×512。如果浏览器网格不是正方形,截完后用画图/IrfanView 裁剪到 512×512
      - 保存路径:`traces/human/images/<trace_id>/step_NNN.png`(三位零填充,从 `001` 开始)
   b. **写 `observed`**(+ 可选 `rule_update`)
   c. **拼成 JSON 一行追加到 jsonl 文件**

5. **`diff_from_prev` 填写规则(Way A)**:
   - step 0(首步):填 `[]`
   - 后续步骤:如果你能目测出移动了哪些格子,填 `[[row,col,old,new],...]`;**如果不确定,填 `[]`** — diff 是辅助调试字段,不影响训练
   - 行/列编号从 0 起,左上角为 (0,0)

6. **meta 行在标注结束后补全**:`outcome`, `levels_completed`, `total_steps`, `rhae`

---

### 方式 B — `record_human.py` 录入工具(实验正式开跑用)

**现状:未实现,但 `grid_to_image()` 已就绪。** 实现后的工作流:

1. SDK `env.reset()` 获取首帧
2. `grid_to_image(latest_grid(frame), scale=8)` → 保存为 `step_NNN.png`(512×512),路径写入记录
   > `grid_to_image` 已在 `arc_agent/observation.py` 实现并测试,直接 import 可用
3. 终端用 ANSI 16 色块打印网格概览(低保真,供快速定位颜色分布)
4. 打开系统图像查看器显示实际 PNG(Windows: `os.startfile(path)`)
5. 提示输入 `hypothesis` + `intent`
6. 提示按键选动作(`W/A/S/D` = ACTION1-4,`E` = ACTION5,`C x y` = ACTION6,`U` = ACTION7)
7. SDK `env.step()` 执行动作,获取新帧
8. `grid_diff(prev_grid, latest_grid(new_frame))` → 精确填 `diff_from_prev`
9. 显示新帧图像 + diff 摘要
10. 提示输入 `observed`(+ 可选 `rule_update` + 可选 `quality`)
11. 追加写入 jsonl;循环至 WIN / GAME_OVER / 手动退出
12. 标注结束时自动算 RHAE 并补写 meta 行

**当前阻塞**:`record_human.py` 脚本本身尚未实现,等 §9 中的两步 RESEARCH/EXPERIMENTS 登记完成后开工。

ANSI 色块是备用的低保真视图,**不能替代 PNG**;PNG 才是 VLM 训练的实际输入。

---

## 7. 收工 checklist

**单条 trace 完成后**:
- [ ] JSONL 文件落在 `traces/human/<game_id>_NNN.jsonl`
- [ ] metadata 行 `schema_version: 2`,所有字段填满(包括 `outcome`/`total_steps`/`rhae`),日期 ISO 8601
- [ ] 每步 `frame_image_path` 对应的 PNG 文件存在于 `traces/human/images/<trace_id>/`
- [ ] PNG 是网格区域截图,不含浏览器 chrome
- [ ] 每步 `hypothesis`+`intent`+`observed` 三字段非空(`rule_update`/`frame_text`/`quality` 可空)
- [ ] intent 末尾有 `[exploration]` 或 `[exploitation]` 标记
- [ ] 在 `EXPERIMENTS.md` 占位条目把进度 ++

**全部 10 条完成后**:
- [ ] 在 `RESEARCH.md` 方案 #5 标"标注完成,等 BC 训练"
- [ ] **不要**写 `PAPER.md` — 训练实验出数后才写(PAPER 不引入新事实)
- [ ] 确认 `traces/human/` 已加入 `.gitignore`(图像文件大,不入版本控制)

---

## 8. 反模式(不要做)

- ❌ 沿用 `schema_version: 1` 的 trace — v1 缺 `frame_image_path`,训练时会报错
- ❌ `frame_image_path` 填绝对路径 — 路径必须相对于 `traces/human/` 才能跨机器复现
- ❌ 标到一半改 schema — 一旦定就锁本批次,改动开 `schema_version: 3`
- ❌ 同一游戏刷 N 条凑数 — 至少跨 ≥5 个不同 `game_id`,每个 game 至少 2 条策略不同的 WIN trace
- ❌ Hypothesis 写"看起来可能" — 不可证伪 = 没用
- ❌ Observed 里描述图像颜色/位置("左上角有个蓝块") — 模型能看图,你的价值在推理链
- ❌ 先按键再补写 hypothesis — 变成事后合理化,失去 think-aloud 价值
- ❌ 只标 WIN trace 且 RHAE 很低 — 高步数 WIN trace 的 BC 权重极低(RHAE²),不如少而精
- ❌ 跳过失败步骤、事后剪辑 — 错的动作和对的动作都全程录(用 `quality: -1` 标记,不要删)
- ❌ 每步都填 rule_update — `rule_update` 只在信念真的改变时填;高频出现说明 hypothesis 没有真正更新,该回头重写

---

## 9. 配套登记(本文档完成后下一步)

按 docs/ 维护规则,本手册落位后还差两步,等你点头再做:

1. `docs/RESEARCH.md` 加**方案 #5 正式条目**:正式 hypothesis(具体数字)+ iteration trigger(成功/失败/不显著三分支)。
2. `docs/EXPERIMENTS.md` 加**占位条目**:实验编号、选定游戏列表、标注进度计数器。

**`record_human.py` 是再下一步**。当前阻塞是脚本本身未实现,**不是** `grid_to_image()` —— 该函数已在 `arc_agent/observation.py` 实现并通过 vlm_test 验证。实现 `record_human.py` 前需要：
- 确认 §9 中的 RESEARCH/EXPERIMENTS 两步完成
- 确认 Way A 已用 1-2 条 trace 验证了 schema 格式

相关链接：
- ARC Prize 游戏浏览器：https://three.arcprize.org
- API key 申请：https://arcprize.org/api-keys
- 可用游戏列表：https://arcprize.org/tasks
- PEFT / LoRA 文档（训练用）：https://huggingface.co/docs/peft/conceptual_guides/lora
- Qwen2.5-VL 模型页：https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
