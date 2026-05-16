# 文档规范 (CONVENTIONS_zh.md)

> 所有 `docs/` 和 `outputs/reports/` 里的文档都遵守这份规范。Claude / 协作者写新文档前先读。

最近更新: 2026-05-16

---

## 0. 一句话核心

> **架构文档说「做什么 + 为什么」,实验报告说「这次跑了什么 + 结果如何」,词汇表是唯一的术语字典,索引是导航。每个文档专门负责一个回答,不跨界。**

---

## 1. 目录结构(强制)

```
docs/
├── README.md                ← 项目总入口:目标 + 边界 + 当前进展 + 版本历史
├── CONVENTIONS_zh.md        ← 本文件,文档规范
├── GLOSSARY_zh.md           ← 唯一的术语字典(R1/R2/BUG-X/Knowledge 等)
├── data/
│   └── data.md              ← 数据来源 + 内容 + 用法
└── project/                  ← 每个 project = 1 git 分支 = 1 个版本
    └── <project_name>/
        ├── architecture.md   ← 人工核心管理(设计意图、决策门、风险)
        ├── reference_*.md    ← 人工核心管理(prompt / 数据流 / 评测细节)
        ├── report*.md        ← Claude auto 写(实验结果、人工审)
        └── figures/          ← 报告引用的 PNG / GIF

outputs/                       不动,仍是实验产物 dir
└── <tag>_<YYYYMMDD-HHMMSS>/   ← 跑实验的原始 trace / step PNG / play.gif

archive/                       ← 单向门:被取代的旧文档/旧实验
├── docs_<YYYY-MM-DD>/
└── reports_<YYYY-MM-DD>/
```

**规则**:
- `archive/` 是单向的,东西进得去不能直接出来,要复用必须先 **promote** 回 docs/
- **1 个 project = 1 个 git 分支**(命名 `feat-<name>` 或 `<name>-v0` 等);项目完结合主线
- 每个 `docs/project/<X>/architecture.md` 都应有至少一份 `report*.md`(可以是空骨架,等实验跑完填)
- `figures/` 子目录只放报告引用的图,不放代码、不放原始 trace

---

## 2. 命名规范

### 2.1 Project 文件夹名

```
docs/project/<name>_<version?>/
```

例:`v3`、`v3_2`、`predictor_v0`、`grpo_v0`、`action_proposer_v0`、`v2_canary_verify`。

- 小写、下划线分词
- 版本号 `v0`/`v1`/`v2`(可省略,例如 `v3_2`)
- 1 个 folder = 1 个 git 分支(命名 `feat-<name>` 或 `<name>-v0` 等)

### 2.2 Project 内的文件名

| 文件 | 必须? | 内容 |
|---|---|---|
| `architecture.md` | ✅ 必须 | 设计意图(套 §3 模板) |
| `reference_<aspect>.md` | 可选 | prompt / 数据流 / 评测细节,每个 aspect 一个文件 |
| `report.md` | ✅ 实验跑完必须 | 主实验报告(套 §4 模板) |
| `report_<experiment>.md` | 可选 | 多个子实验时,每个一个 |
| `figures/<name>.png` | 可选 | 报告引用的图 |

### 2.3 实验产物目录名(`outputs/`)

```
outputs/<tag>_<YYYYMMDD-HHMMSS>/
```
例:`outputs/mask_revive_3x200_20260516-014356/`

每个实验产物目录至少含:
- `run_meta.json`(git_commit + args + 开始时间)
- `report.md`(automated,简短)
- 数据(trace.jsonl / metrics.json / 等)

实验跑完,把人审报告写到 `docs/project/<project>/report.md`(套 §4 模板),把关键图复制到 `docs/project/<project>/figures/`。原始数据**留在 `outputs/`** 不动(gitignore)。

---

## 3. 架构文档模板(`docs/project/<name>/architecture.md`)

每个架构设计文档**必须**有这 10 节,顺序固定:

````markdown
# ARCHITECTURE — <Name> (v<version>)

日期: YYYY-MM-DD
状态: <设计 | 实施中 | 实施完 | parked | 取代>
取代关系: [前置](./prev.md) → 本文 → [后续](./next.md)(只有改 / 替代时填)
前置阅读: [`other.md`](./other.md)

---

## 0. 为什么做这个

(2-5 句话。给陌生人讲:**当前痛点是什么** + **本设计要解决什么**。
 必须引用一个具体数据或现象,例:「ar25 change_rate 3-4%,0 levels won」)

## 1. 范围

### 1.1 In-scope

(本文档承诺要做的事,bullet list)

### 1.2 Out-of-scope

(本文档**不**承诺的事 — 防止 scope creep)

### 1.3 规则合规

(如果有比赛 / 部署约束,在这里说明本设计如何不违反)

## 2. 概念表

(本文档里出现但还没进 GLOSSARY 的新概念,2-列表格:概念名 | 一句话定义。
 进了 GLOSSARY 之后这里删掉,只留 [[link]] 到 GLOSSARY 即可)

## 3. 数据流 / 系统图

(必填。ASCII 图或 mermaid。读完图应该能猜出代码会怎么写)

## 4. 关键决策

(本设计的判断和取舍。每条:**什么决策 + 为什么 + 替代方案为什么不选**。
 例:「Reflection per-step 而不是 per-round,因为 episode 内反馈延迟太长」)

## 5. 模块 / 文件清单

| 文件 | 状态(🆕 NEW / 🟡 改 / ⚫ deprecate)| 责任 |
|---|---|---|
| `arc_agent/...py` | 🆕 | one-line |

## 6. 评估方法

### 6.1 指标

(每个指标:**指标名 | 含义 | 期望值/范围**)

### 6.2 决策门(Gate)

| 门 | 条件 | 失败应对 |
|---|---|---|
| G1 | ... | ... |

## 7. 已知风险

| 风险 | 概率 | 缓解 |
|---|---|---|


## 8. 跟其它路线的关系

(本设计跟其它 architecture 文档的依赖、平行、替代关系。链接出去)

## 9. 文档历史

- YYYY-MM-DD 初稿
- YYYY-MM-DD <修改原因>

````

---

## 4. 实验报告模板(`docs/project/<name>/report.md` 或 `report_<exp>.md`)

每个实验报告**必须**有这 8 节,顺序固定:

````markdown
# <Title> — <一句话定位>

生成时间: YYYY-MM-DD HH:MM
对应架构: [`architecture.md`](./architecture.md)
源数据: `outputs/<run_dir>/`
状态: <进行中 | 已完成 | 失败 | 部分>

## TL;DR

(3-5 个 bullet。直接给结论。任何 5 秒读完就能知道:
 ✅ 通过的门 / ❌ 没通过的门 / 关键数字 / 下一步)

## 1. 实验目的

**验证什么**: (一句话假设)
**为什么这么做**: (给陌生人讲:**当前数据 / 情况**,**这个改动应该如何改善**)
**对照基线**: (跟谁比?用什么数据集?)

## 2. Setup

| 项 | 值 |
|---|---|
| 模型 / 代码版本 | git commit hash + 简述 |
| 数据 | ... |
| 关键参数 | ... |
| 总耗时 | 预估 / 实际 |

## 3. 结果(表 + 图)

(主指标表格 + 至少 1 张图。每张图都要有 caption 说明看什么)

![标题](./<name>/figure1.png)
**图 1**: caption,看 X 轴 vs Y 轴,期望趋势,实际趋势。

## 4. 分析

(分析为什么是这个结果。**坦诚**,不要美化。
 - 哪些假设被证实
 - 哪些被推翻
 - 有没有意外发现
 - 数据有没有 caveat)

## 5. 决策门判定

(表格:门 | 条件 | 实测 | 通过?)

## 6. 下一步建议

(优先级排序的 3-5 条具体动作。每条说明:**做什么 / 工作量 / 预期收益**)

## 7. 复现命令

```bash
.venv/Scripts/python.exe scripts/...
```

## 8. 文件清单

```
outputs/<run_dir>/
├── ...
```
````

---

## 5. 词汇表条目格式(`docs/GLOSSARY_zh.md`)

```markdown
### `<term>`

**定义**: 一句话定义,中性、精确。

**出处**: 首次出现的文档 / commit / 实验报告。

**相关**: [[other_term]]、[[file_path]]

**例**: (可选,1-2 行)
```

---

## 6. README 里的版本历史条目格式

每个 project 一段(顺序按时间):

```markdown
### <project_name> — `docs/project/<name>/`

- **路径**: [`docs/project/<name>/architecture.md`](./project/<name>/architecture.md) + 列其它关键文件
- **状态**: 🟢 当前活 / 🟡 参考 / ⚫ 历史
- **关键 commits**: `<hash1>` → `<hash2>` → `<hash3>`(或 commit 范围)
- **一句话**: 项目本质(< 80 字符) + 最新实测数字
```

`🟢` = 当前活(代码 ground truth);`🟡` = 参考(留着但不主动改);`⚫` = 历史(被取代)

---

## 7. 写作规范

- **中文为主版本**,英文版可选。代码注释用英文(跟着代码风格走)。
- 一段不超过 4 行屏幕。长段拆短。
- 直接给数字,不要「显著提升」「大幅下降」这种空话。
- **不要美化失败**。实验跑挂了就写「FAIL,因为 X」,不写「未达预期」。
- 命令带 `.venv\Scripts\python.exe` 或 `bash` 前缀,可复制粘贴。
- 路径用反引号 `` `arc_agent/foo.py` ``,行号 `arc_agent/foo.py:123`。
- 表头加单位,例:`| change_rate | overrides | wall_clock_s |`。
- 图必须有 caption(图 N: 看什么 vs 什么)。
- 链接相对路径,不写绝对路径。
- emoji 只在状态标记(🟢🟡⚫✅❌⚠️)和我们已经在用的少量地方用。

---

## 8. 流程:什么时候写什么

1. **想做一个新方向** → 开 git 分支 `feat-<name>` → 起 `docs/project/<name>/` 目录 → 写 `architecture.md` 草稿(至少 §0、§1、§3、§6)
   → 跟用户对齐 → 补完 §4-§9 → commit
2. **架构跑出实验数据** → 同 project 目录写 `report.md`(套 §4 模板)
   → 把图放 `figures/` 子目录 → commit + 在 `README.md` §5 加版本历史条目
3. **架构被取代** → 旧 project 状态改成 ⚫,在 README §5 标记;不删
4. **项目失败** → 用户判断;在该分支留报告 + 经验,主分支不合并代码,只合并经验文档
5. **发现新概念** → 写到 `GLOSSARY_zh.md`;在使用它的文档里用 `[[term]]` 链接

---

## 9. 提交规范

commit message 模板(配合本文档):

```
<scope>(<area>): <action>

<context: 一段为什么>

<changes: 具体改了什么>

<results: 如果是实验类提交,给数字>

Co-Authored-By: ...
```

`scope` 用其中之一:`feat | fix | docs | test | refactor | perf | exp(实验)`
`area` 用其中之一:`arch | predictor | mask | grpo | reflect | knowledge | prompt | sft | infra`

---

## 10. 维护

- 本规范每 4 周或大改一次后回看一遍。
- 凡是 INDEX / GLOSSARY 没及时更新的,在下次 commit 里补一行就行,**不要单开一个 PR 只更新索引**。
- 如果发现某条规范在实际工作中被反复违反,优先考虑改规范,而不是责怪不遵守。

---

*文档历史:*
- *2026-05-16 初稿*
