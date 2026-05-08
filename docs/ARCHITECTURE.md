# ARCHITECTURE — 架构逻辑

最近一次更新:2026-04-27(新增"模型架构:三层混合"+"训练范式:离线 BC→RL→在线 LoRA"两节,把 WorldModel/Planner 的实现路径具体化为 Qwen3-0.6B + 结构化槽位双路设计)
当前实现:**只有 RandomAgent**,以下大部分是规划。

---

## 设计目标

ARC-AGI-3 的核心难点是 4 个能力的同时具备:**探索、建模、目标自设、规划**。架构按这 4 个职责切分,而不是按"前端/后端"或"模型/控制器"切分。

## 核心智能体循环:Hypothesize → Execute → Iterate (HEI)

ARC-AGI-3 几乎没有 reward 信号(只在通关时给),传统 RL 的"状态-动作-reward"闭环用不上。我们用**科学方法循环**作 agent 决策骨架,把"探索"和"建模"提升为一等公民:

```
                       +---------------------+
                  +----| Hypothesize         |<---+
                  |    |  · 维护规则假设集    |    |
                  |    |  · 维护 win 候选     |    |
                  |    |  · 每条带置信度+证据 |    |
                  |    +----------+----------+    |
                  |               | hypotheses    |
                  |               v               |
                  |    +---------------------+    | 更新假设集
                  |    | Execute             |    | (新增/驳斥/合并)
                  |    |  · 选 action 同时最大化:
                  |    |    (1) 目标推进 V(s')   |
                  |    |    (2) 信息增益 IG(a)   |
                  |    +----------+----------+    |
                  |               | action        |
                  |               v               |
                  |    +---------------------+    |
                  |    | Iterate             |    |
                  |    |  · 观察 (s, a, s')   |----+
                  |    |  · 比对预测 vs 实际  |
                  |    |  · 用 diff 反推规则  |
                  |    +---------------------+
                  |               |
                  +---------------+
```

**三个阶段的具体职责**

1. **Hypothesize**(对应 `WorldModel`)
   维护一组**可证伪假设**:规则、物体身份、win condition 候选。每条假设带:
   - 置信度 ∈ [0, 1]
   - 支持样本数 / 反对样本数
   - 一句自然语言描述(便于 LLM 操作)+ (可选) 程序化判定函数
   形式上类似经典 belief revision:新证据来时按 Bayes 或简单计数更新。

2. **Execute**(对应 `Planner` + `ActionSelector`)
   每一步同时优化两个目标的加权和:
   - **目标推进** `V(s')`:预期能让你接近某个 win 候选(LLM 给 heuristic 评分)
   - **信息增益** `IG(a)`:这个动作的预期结果在多大程度上能区分相竞假设
   权重在 episode 内动态调整:**早期偏 IG(纯探索),发现稳定假设后偏 V(开发)**。预算意识:5h 是硬上限,留 30% buffer。

3. **Iterate**(对应 `Memory.observe_result`)
   每次 `env.step` 后:
   - 把 (s, a, s') 写入三元组缓存
   - 用每个未驳斥假设预测 s',与实际比较 → 增加/减少置信度
   - 置信度跌破阈值 → 假设"驳斥"(append 到 RESEARCH.md 的失败方案,内部 attempted 集)
   - 多个假设解释力相同 → 触发 Execute 阶段下次优先 IG

**与开发流程 HEI 的对齐**:两者用同一套语言。开发实验里我们对"换 prompt 模板"做假设,agent 在游戏里对"按 ACTION3 是否旋转"做假设。这种对齐让 EXPERIMENTS.md 的迭代经验能直接映射到 agent 内部的迭代逻辑。

**起步技巧**:Phase 1 的 LLMAgent 不需要真实现这个循环,只需让 LLM 在 prompt 里**显式输出三段**:
> Hypothesis: 我认为 ACTION1 是"上移光标"。
> Execute: 我先按 ACTION1,预期光标 y 减 1。
> (观察后)Iterate: y 真的减了 1 → 假设置信度 +1;否则记录反例。

这样我们 Phase 1 就能积累后续 Phase 2/3 需要的"假设-观察"对齐数据。

任何一个 agent 实现都对外暴露一个统一接口:

```python
class Agent(Protocol):
    def reset(self, env_meta: dict) -> None: ...
    def act(self, frame: FrameDataRaw) -> GameAction: ...
    def observe_result(self, frame: FrameDataRaw) -> None: ...
```

`frame` 是 `arcengine.FrameDataRaw`(pydantic 模型,字段 `game_id / state / levels_completed / win_levels / available_actions: list[int] / frame: list[ndarray (64,64)] / guid / full_reset / action_input`,详见 `RESEARCH.md` §4 闭合问题)。`GameAction` 共 8 个枚举值(`RESET + ACTION1..ACTION7`),其中 7 个是用户实际可选动作:`ACTION1=Up / 2=Down / 3=Left / 4=Right / 5=Primary / 6=Coordinate(需 `action.set_data({"x":...,"y":...})`)/ 7=Undo`;`RESET` 是 `state ∈ {NOT_PLAYED, GAME_OVER}` 时的状态转移动作。SDK 的 `env.step(action, data=action.action_data.model_dump(), reasoning=...)` 返回单个 `FrameDataRaw`(无 reward/done/truncated/info 元组——稀疏奖励通过 `state==WIN` 判断)。详见 `RESEARCH.md` 尝试 #2 + §4。

---

## 模块职责

```
            +---------------------+
            |    arc_agi.Env      |   <- ARC SDK,黑盒
            +----------+----------+
                       | obs, reward, done, info
                       v
+----------------------+----------------------+
|              ObservationParser              |   把 grid 序列化成 (1) text/ASCII (2) 张量 (3) 帧 diff
+----------------------+----------------------+
                       |
        +--------------+--------------+
        v              v              v
   +---------+   +-----------+   +-----------+
   | Memory  |<->| WorldModel|<->|  Planner  |
   | (per-ep |   | 规则归纳, |   | BFS/MCTS, |
   | + cross)|   | 转移函数  |   | LLM proposer
   +---------+   +-----------+   +-----------+
                       |              |
                       v              v
                  +------------------+
                  |  ActionSelector  |   选 ACTION1–5 / UNDO / 坐标
                  +--------+---------+
                           |
                           v   action
                       arc_agi.Env.step
```

### ObservationParser
- 输入:原始 grid (np.array, H×W=64×64, dtype=int, 值域 0–15)
- 输出:多视图
  - `text_view`:每行 64 字符,16 色 → 16 个字符的字符表(兼容 Phase 1 LLMAgent)
  - `image_view`:PIL Image(512×512,ARC 标准色),供 Qwen2.5-VL vision encoder
  - `diff_view`:与上一帧的 diff cell 列表 `[(x,y,old,new), ...]`(给规则归纳用)

### Memory
- **per-episode**:状态-动作-后果三元组缓存,辅助前向搜索
- **cross-episode**(同一 game_id 跨 levels):本场已学到的规则、win condition 候选
- **cross-game**:**禁止**(比赛禁止 task-specific 优化,但可以保留通用先验)

### WorldModel
- 输入:per-episode 三元组流
- 输出:候选转移规则集合,例如 "ACTION1 → 颜色 7 的连通块整体向上移 1 cell"
- 实现路线:Phase 2 先用 LLM 直接归纳,Phase 3 加 symbolic 验证器

### Planner
- 用 WorldModel 在动作树里做前向搜索;UNDO 是天然的回退操作
- 评估函数:**没有真 reward**(reward 几乎只在过关时给),所以用 LLM 当 heuristic 评估"这个状态像不像在朝目标走"
- 预算:5h 是硬上限,**留 30% buffer 给规划失败的纠错**

### ActionSelector
- 综合 Planner 输出 + 探索 ε(早期高、后期低)
- 坐标选择动作:Planner 必须给出 (x, y),不能是随机猜

---

## 数据流时间尺度

| 尺度 | 跨度 | 例子 |
|------|------|------|
| Step | 单次 env.step | 选一个动作 |
| Episode | 一个 level | 通关或被 5h 截断 |
| Environment | 一个 game_id 的所有 levels | 共享 WorldModel、规则集 |
| Run | 一次评估 | 跑全部 25 / 55 个 game,生成 scorecard |

---

## 模型架构:Qwen2.5-VL-3B-Instruct(2026-04-28 更新)

**架构决策变更(2026-04-28)**:原"三层混合"方案(CNN encoder + Qwen3-0.6B text + 结构化槽位 + cross-attention 融合)被 **Qwen2.5-VL-3B-Instruct** 单模型方案替代。核心原因:Qwen3-0.6B 是纯文本模型,读取 hex 文本网格时按 1D token 序列处理,无法感知行间/列间的 2D 空间关联(相邻行的同一列 cell 在序列中相距 ~64 token)。VLM 的 vision encoder 用 2D patch attention 直接在像素级捕获空间关联,彻底解决这个根本缺陷,同时消除了 CNN+槽位+融合三个独立组件的实现负担。

```
  64×64 grid (numpy int array)
         │
         ▼  grid_to_image()  (arc_agent/observation.py)
  512×512 PNG (ARC 标准 16 色)
         │
         ▼
  ┌──────────────────────────────────────────┐
  │  Qwen2.5-VL-3B-Instruct                 │
  │                                          │
  │  ┌──────────────────┐                   │
  │  │ Vision Encoder   │  2D patch attention│
  │  │ (ViT-style)      │  捕获空间关联      │
  │  └────────┬─────────┘                   │
  │           │ visual tokens                │
  │           ▼                              │
  │  ┌──────────────────┐                   │
  │  │  LLM Backbone    │  + text tokens     │
  │  │  (3B Qwen2.5)    │  (state / history) │
  │  └────────┬─────────┘                   │
  └───────────┼──────────────────────────────┘
              │
              ▼  HEI 三段式文本输出
       ACTION: ACTION3   (或 ACTION6 x=.. y=..)
```

### 组件职责

**grid_to_image()**(`arc_agent/observation.py`)
- 输入:64×64 numpy int 数组(值域 0–15)
- 输出:PIL Image,默认 scale=8 → 512×512 像素
- 使用 ARC 标准 16 色调色板(与 ARC 官方网站一致)
- 输出图像供 VLM vision encoder 处理;不做 ImageNet 归一化(颜色本身是语义)

**Qwen2.5-VL-3B-Instruct vision encoder**
- ViT-style 2D patch attention,每个 patch = 14×14 像素
- 512×512 图像 → 约 (512/14)² ≈ 1337 visual tokens
- 直接在 2D 空间捕获颜色块的形状、相对位置、连通性 — 解决了 hex 文本的根本缺陷
- **不从零预训练**:使用 Qwen2.5-VL-3B 的预训练视觉权重(已见过大量图表/网格类图像)

**Qwen2.5-VL-3B LLM backbone**
- 输入:visual tokens + text prompt(游戏状态、可用动作、HEI 历史)
- 输出:HEI 三段式文本(Hypothesis / Execute / Iterate)+ ACTION 指令行
- 与 Phase 1 LLMAgent 的 prompt 格式完全兼容,Phase 1 积累的数据可直接复用
- LoRA 目标:text backbone(rank 8–16)+ 可选 vision encoder 后几层

**LoRA 配置(在线适应)**
- base = Qwen2.5-VL-3B 预训练权重(冻结)
- LoRA 层:text backbone 的 q/v projection(~10M 可训参数)
- 每个新游戏单独开一个 LoRA,episode 结束清空
- rank 8–16,前 30% 步微调,后 70% 只推理

### 内存估算(RTX A4500 20GB)

| 阶段 | 占用 |
|------|------|
| 模型权重 BF16 | ~6 GB |
| LoRA 参数 | ~0.1 GB |
| 512×512 图像 KV cache(推理) | ~2 GB |
| 文本 context(~2K token) KV cache | ~1 GB |
| 剩余 buffer | ~10 GB |
| **合计(推理)** | **~9 GB / 20 GB** ✓ |

BC 训练(gradient checkpointing + 4-bit QLoRA):~16–18 GB — 在 A4500 边缘但可行。Kaggle T4(16GB)推理 OK;训练用 QLoRA 4-bit。

### 与原三层方案的对照

| 维度 | 旧方案(CNN+Qwen3-0.6B+槽位) | 新方案(Qwen2.5-VL-3B) |
|------|-------------------------------|------------------------|
| 空间感知 | CNN 感知但与 LLM 割裂(cross-attention) | vision encoder 与 LLM backbone 端到端联合训练 |
| 开发量 | 5 个独立组件需分别实现 + 调试 | 1 个预训练模型,接口标准化 |
| 文本推理 | Qwen3-0.6B(0.6B 弱) | Qwen2.5(3B,强) |
| Kaggle T4 推理 | 需拼 3 个组件的内存 | 单模型 ~6GB BF16 |
| Phase 1 数据复用 | 需转换格式 | 完全兼容(HEI prompt 不变) |
| 风险 | 融合层不稳定;槽位模板设计复杂 | 单点失败;VLM 在纯颜色网格上表现待验证 |

---

## 训练范式:离线 BC → 离线 RL → 在线 LoRA(2026-04-27 新增)

三阶段,前两阶段离线一次性做完,第三阶段每个游戏现做。

### Stage A — 离线 BC 预训练(Phase 2)

**数据**:
- 458 名参与者的 ARC-AGI-3 人类 trace(2,893 attempts,**已开源**,见 `RESEARCH.md` §1 + 读后感 #5)
- demo 25 游戏的 RandomAgent rollouts(对照 + 数据增广)
- 训练时增广:颜色 shuffle、grid 旋转/翻转(ARC-AGI 颜色无语义,这种增广合规且强力)

**损失**:
- 主损失:动作 cross-entropy(模仿人类动作)
- 辅助损失 1:下一帧预测 MSE(强化世界模型表征)
- 辅助损失 2:规则槽位自监督 — 槽位预测的"动作 → 帧变化"应该匹配实际 diff
- 辅助损失 3(可选):规则文本自监督 — 用离线 LLM(开发期可调 Claude API)给 trace 标注 silver-label 规则文本,LoRA 学习生成

**预算估算**:Qwen3-0.6B + CNN + 槽位 transformer ≈ 0.65B 参数;BC 数据 ~22K 轨迹 × ~50 step = ~1.1M (s,a,s') 三元组;单 A100 估 50–100 GPU-h(几天内能跑完一版)。

### Stage B — 离线 RL 微调(Phase 3)

**算法**:PPO(成熟、好调)+ KL 正则到 Stage A 的 BC 策略(防遗忘人类先验)

**Reward**:
- 稀疏:RHAE-style 过关奖励(只在 win 给)
- 密集 intrinsic:① 帧变化量(borrowed from StochasticGoose);② 规则集 entropy 减少量(信息增益);③ 新规则槽位数量增长(鼓励探索)

**采样**:demo 25 游戏自玩 rollouts,跨游戏混采(防过拟合单个游戏)

**预算估算**:100–200 GPU-h on A100 / 4090

### Stage C — 在线 LoRA 微调(部署时,每游戏)

每个新游戏(包括 semi-private / private)开始时:
1. 加载冷冻的 base + Stage B 的策略权重
2. 启用独立 LoRA(rank 8–16,~5–10M 可训参数)
3. 前 30% 探索预算用作微调:BC-style 损失,target 是当前游戏内已积累的高 reward / 信息增益高的轨迹
4. 后 70% 预算只前向推理
5. **游戏结束清空 LoRA**,跨游戏不共享(比赛禁止 task-specific 优化)

**为什么这样切**:test-time training 在 ARC-AGI-1/2 上验证可让性能翻倍(`RESEARCH.md` 读后感 #2);70/30 切分待 Phase 3 实验调,初版按这个跑。

### 训练范式选择的 HEI 对齐

- **Hypothesize**:三阶段训练 + 在线 LoRA 比纯 StochasticGoose 路线(无离线、无知识迁移)在 OOD 上至少抬 10× 分数(详见 `RESEARCH.md` §尝试 #3)
- **Execute**:Stage A 先跑通,出 BC-only 基线;Stage B 增量加 RL;Stage C 增量加 LoRA — 每加一层做对照实验
- **Iterate**:每加一层都看 demo 与 semi-private 的 gap,gap 不收窄就回头调架构而非加数据

---

## 已经做出的架构决策

| 决策 | 时间 | 理由 |
|------|------|------|
| 用 Protocol 而不是 ABC 定义 Agent 接口 | 2026-04-27 | 鸭子类型在 Python 里更简单,不强制继承 |
| 跨 game_id 不共享 WorldModel | 2026-04-27 | 比赛禁止 task-specific 优化 |
| **本地权重而非 API LLM 作为最终系统** | 2026-04-27 | Kaggle 沙盒无网,API 不可用(详见 RESEARCH.md §4 闭合问题) |
| **三阶段训练:BC→RL→在线 LoRA** | 2026-04-27 | BC 引入人类先验(StochasticGoose 缺这个);RL 优化稀疏 reward;LoRA 应对 OOD |
| **放弃 CNN+Qwen3-0.6B+槽位三层方案** | 2026-04-28 | Qwen3-0.6B 为纯文本模型,读 hex 网格按 1D 序列处理,行间空间关联(相距 ~64 token)难以捕获;三层独立组件开发量大且融合层不稳定 |
| **Backbone 改用 Qwen2.5-VL-3B-Instruct** | 2026-04-28 | VLM vision encoder 用 2D patch attention 直接感知空间关联;预训练权重含网格/图表知识;单模型端到端,消除 CNN+融合层开发量;T4 16GB BF16 可装 |
| **grid_to_image():scale=8 → 512×512** | 2026-04-28 | 64×64 原始尺寸低于 VLM patch 精度下限;×8 = 512×512 匹配 ViT patch 14px 标准尺寸(1337 tokens);ARC 标准 16 色不做 ImageNet 归一化 |

---

## 待决定

- LoRA 目标层:仅 text backbone q/v,还是包含 vision encoder 后几层(Phase 2 实测)
- BC 训练时图像分辨率:512×512(高精度,内存重)vs 256×256(快但 patch 粗)(待测)
- 是否在 LLM 推理上加 KV cache 跨 step 复用(vision token 不变时可节省 ~50% 计算)
- QLoRA 4-bit 对 BC 精度的影响(Kaggle T4 训练必须用 4-bit)
