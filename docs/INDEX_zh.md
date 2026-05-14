# 文档索引

最近更新: 2026-05-14

新接手代码 / 不知道从哪看起 → **按下面顺序读前 3 篇就够**。

---

## 📐 文件命名规范(2026-05-14 起执行)

```
{prefix}_{name}_{version}_{lang}.md

prefix:  arch_   = 架构 / 设计文档(描述 HOW)
         ref_    = 参考 / 细节文档(描述 WHAT,例如 prompt 详情)
         INDEX   = 索引(特殊,大写,无 prefix)

name:    小写,下划线分词

version: v1 / v2 / v3 / v3_2 ...(下划线分隔,不用点;无版本时省略)

lang:    zh = 中文,en = 英文

status:  不进文件名 — 状态在本 INDEX 写(状态会变,文件名稳定)
```

**例子**: `arch_v3_zh.md`(v3 架构,中文)、`ref_v3_prompt_zh.md`(v3 的 prompt 参考)、`arch_agents_v2_zh.md`(agents 第 2 版)。

新文档遵循这条规则。旧文档(`ARCHITECTURE_RL.md` 等大写下划线)已在 2026-05-14 重命名。

---

## 🟢 当前在用(必读)

### 1. [arch_v3_2_zh.md](./arch_v3_2_zh.md) ⭐ **最新主设计**(v3.2 双 Agent)

> v3.2:**Action Agent + Reflection Agent + 跨 round Knowledge 持久化**。这是当前正在实施的方向 —— 把"决策"和"反思"拆开,Knowledge 跨 episode 累积。包含双 agent 架构、Knowledge schema、多轮调度、评估方法。

### 2. [arch_v3_zh.md](./arch_v3_zh.md) ⭐ **v3 单 Agent 基线设计**

> v3 单 Agent 的完整架构 + 概念表 + 数据流 + 实施清单 + 评估方法。**当前线上代码的实现蓝本** —— scipy 视觉感知 + Qwen text-only 决策。v3.2 在它的基础上拆 agent。

### 3. [ref_v3_prompt_zh.md](./ref_v3_prompt_zh.md) ⭐ **prompt 实战参考**

> 给 Qwen 的 prompt **逐块拆解**(SYSTEM + 8 个 USER 块,每块标来源 + 实战例子)。包含从真实 trace 发现的问题(ACTION 格式滥用 68%、卡死检测不到、state 重访严重)+ P0-A/P0-B/P1 改造记录。**改 prompt 前先读这个**。

### 4. [ref_v3_2_dataflow_zh.md](./ref_v3_2_dataflow_zh.md) ⭐ **v3.2 三板块真实 I/O 走查**

> 拿 2026-05-14 真跑的 `outputs/v3_2_ar25_3x30/round_00/step=6` 做参照,把 v3.2 三个板块(perception、reflection、action)的**输入来源、SYSTEM/USER prompt、实际输出**逐字展开 + 关键文件索引 + 暴露的问题。**理解 v3.2 数据流 / 改板块行为前必读**。

### 5. [ref_v3_2_hardrules_results_zh.md](./ref_v3_2_hardrules_results_zh.md) ⭐ **v3.2 硬规则实测对比**

> commit `1bac4be` 引入的 R1+R2+R3(Knowledge sentinel / action mask / stuck untried)在 ar25 3 round × 30 step 上的对比实测。change_rate 从 23/17/17% 升到 60/100/97%,no_op_streak 16→0,ACTION6 spam 22+23 次被 R2 全拦。**理解硬规则必要性、找下一步攻击方向前读这个**。

### 6. [ref_object_pipeline_zh.md](./ref_object_pipeline_zh.md)

> 视觉感知层的设计 + 评测。**结论**:Qwen-VL 对单帧 / 对齐都不可靠,改用 `scipy.ndimage.label` + Hungarian。包含 ar25 上 scipy 100% vs Qwen 0% 的实测对比。

---

## 🟡 主要参考(挑着看)

### 7. [arch_rl_v0_zh.md](./arch_rl_v0_zh.md)

> 最早的 RL 路线设计(intrinsic F1 reward + GRPO),带 2026-05-12 实测红字批注。**v3 已偏离这条路线** —— 现在没在跑训练,只在 prompt + 检测器层迭代。看这个主要是了解为什么不走 RL。**留在 docs/ 因为 CLAUDE.md / README.md 多处引用。**

---

## ⚫ 历史 / 已归档(在 archive/docs_2026-05-14/)

以下文档**已被 v3 取代**,2026-05-14 移到 `archive/docs_2026-05-14/`。**除非考古否则别看**。

| 归档路径 | 说明 |
|---|---|
| `archive/docs_2026-05-14/arch_agents_v1_en.md` | 4 个 agent 架构(A1 / A2 / A3 / A4)的最初英文设计 + 消融矩阵。**R0/A1/A2/A3/A4 命名定义在这**,看老 ablation 报告时偶尔需要回查。 |
| `archive/docs_2026-05-14/arch_agents_v1_zh.md` | 上一份的中文版,带 2026-05-13 红字批注。 |
| `archive/docs_2026-05-14/arch_agents_v2_zh.md` | v2 修订(三条硬约束 + LearnedActionMap)。被 v3 用 `OutcomeLog` 替代。 |

更早的归档在 `archive/docs_2026-05-11/`(BC pipeline 时代的文档)。

---

## 🔵 数据 / 实验产出(不是文档但常用)

| 路径 | 内容 |
|---|---|
| `outputs/v3_p0b_p1_full/report.md` | v3.1 (P0-A + P0-B + P1) 5 game × 80 step 最新结果(带 GIF 嵌入) |
| `outputs/v3_p0b_p1_full/<game>/trace_view.md` | 每游戏 80 步表格(action / changed / response) |
| `outputs/v3_p0b_p1_full/<game>/play.gif` | 每游戏 GIF 可视化播放 |
| `outputs/v3_p0b_p1_full/trace_audit.json` | 跨游戏审计数据 |
| `outputs/ablation_overnight_/` | v1 旧 ablation 6 agent × 5 game(R0/A1/A2/A3/A4 baseline) |
| `outputs/scipy_object_diag/` | scipy 对象提取的人工审 markdown 报告 |
| `outputs/qwen_object_diag/` | Qwen-VL 对象提取的对照报告(对比 scipy 用) |

---

## 📋 读图推荐路径

**我刚加入 / 想搞清楚现状**:
1. `arch_v3_2_zh.md`(知道当前正在做什么)
2. `arch_v3_zh.md`(知道线上代码长啥样)
3. `ref_v3_prompt_zh.md`(知道已发现的问题 + 优先级)
4. `outputs/v3_p0b_p1_full/report.md`(知道当前实测效果)

**我要改 prompt**:
1. `ref_v3_prompt_zh.md`(看每个 prompt 块来源 + 已知问题)
2. 改完跑 canary(见 `ref_v3_prompt_zh.md` §10 工作流)

**我要加新的检测器 / 模块**:
1. `arch_v3_zh.md` §5 看模块清单
2. `ref_object_pipeline_zh.md` §9 看 v3 的 perception 设计哲学

**我想知道为什么不走 RL**:
→ `arch_rl_v0_zh.md` + 2026-05-12 红字批注

---

## 📝 文档状态约定

- **🟢 当前在用** = 代码 ground truth 在这里,改代码前看这个
- **🟡 主要参考** = 历史决策 / 命名约定,需要时查
- **⚫ 历史** = 已被取代,只在考古时看
- 任何 v 前缀(v1 / v2 / v3 / v3_2)= 设计版本,**新的取代旧的**
- 任何 `_zh.md` 后缀 = 中文版,通常是主要版本(英文版可能过期)

---

## 🔁 改名 + 归档记录(2026-05-14)

### 改名(旧 ALL-CAPS → 新规范)

| 旧名 | 新名 |
|---|---|
| ARCHITECTURE_RL.md | arch_rl_v0_zh.md(留在 docs/) |
| ARCHITECTURE_AGENTS.md | arch_agents_v1_en.md(已归档) |
| ARCHITECTURE_AGENTS_zh.md | arch_agents_v1_zh.md(已归档) |
| ARCHITECTURE_AGENTS_v2_zh.md | arch_agents_v2_zh.md(已归档) |
| ARCHITECTURE_v3_zh.md | arch_v3_zh.md(留在 docs/) |
| ARCHITECTURE_v3.2_zh.md | arch_v3_2_zh.md(留在 docs/) |
| OBJECT_PIPELINE_DESIGN_zh.md | ref_object_pipeline_zh.md(留在 docs/) |
| V3_PROMPT_REFERENCE_zh.md | ref_v3_prompt_zh.md(留在 docs/) |
| INDEX_zh.md | (保留) |

### 归档(被 v3 取代)

3 份旧 agent 设计文档移到 `archive/docs_2026-05-14/`。完整列表在上面 ⚫ 区块。

### 当前 docs/ 文件清单(6 个)

```
docs/
├── INDEX_zh.md                 ← 本文件
├── arch_v3_2_zh.md            ← 🟢 v3.2 双 agent 最新设计
├── arch_v3_zh.md              ← 🟢 v3 单 agent 基线
├── ref_v3_prompt_zh.md        ← 🟢 prompt 实战参考
├── ref_object_pipeline_zh.md  ← 🟢 视觉感知层评测
└── arch_rl_v0_zh.md           ← 🟡 RL 老设计(留作 CLAUDE.md 引用)
```

---

*这个索引每加一份新文档就更新一行。新文档默认写 🟢,被取代时移到 archive/。*
