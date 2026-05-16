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
├── INDEX_zh.md              ← 总入口:项目目标 + 全部文档一句话 + 进展
├── CONVENTIONS_zh.md        ← 本文件,文档规范
├── GLOSSARY_zh.md           ← 唯一的术语字典(R1/R2/BUG-X/Knowledge 等)
├── architecture/            ← 架构设计文档,每篇一个版本
│   ├── v3_zh.md
│   ├── v3_2_zh.md
│   ├── predictor_v0_zh.md
│   ├── grpo_v0_zh.md
│   ├── sft_tier1_zh.md
│   └── rl_v0_zh.md          (parked 老路线,留作引用)
└── reference/               ← 参考资料(prompt 详情、数据流、底层评测)
    ├── v3_prompt_zh.md
    ├── v3_2_dataflow_zh.md
    ├── v3_2_hardrules_results_zh.md
    └── object_pipeline_zh.md

outputs/
└── reports/
    ├── INDEX_zh.md          ← 报告总入口:每份报告一句话总结
    ├── <experiment>.md      ← 实验报告(一对一对应某个 architecture)
    └── <experiment>/         ← 该报告的图(PNG/GIF)放在这子目录
        ├── ...png
        └── ...

archive/                     ← 单向门:被取代的旧文档/旧实验,只做考古
└── docs_<YYYY-MM-DD>/
```

**规则**:
- `archive/` 是单向的,东西进得去不能直接出来,要复用必须先 **promote** 回 docs/ 或 outputs/reports/
- 任何一个 `docs/architecture/X_zh.md` 都应该至少有一份对应的 `outputs/reports/X_*.md`(可以是空骨架,等实验跑完填)
- `outputs/reports/<experiment>/` 子目录只放该报告引用的图,不放代码

---

## 2. 命名规范

### 2.1 文件名

```
{prefix}_{name}_{version?}_{lang}.md
```

| 元素 | 取值 | 例 |
|---|---|---|
| `prefix` | `arch` = 架构设计;`ref` = 参考资料;`INDEX`/`GLOSSARY`/`CONVENTIONS` 例外 | `arch_v3_2_zh.md` |
| `name` | 小写,下划线分词 | `predictor`、`v3_2`、`hardrules_results` |
| `version` | `v1`/`v2`/`v3`/`v3_2`/`v0`...(可省略) | `v0` |
| `lang` | `zh` 中文(默认主版本)/`en` 英文 | `zh` |

**报告文件名**(`outputs/reports/`):
```
{architecture_name}[_<detail>].md
```
例:`predictor_v0.md`、`mask_revive_3x200.md`、`trace_balance.md`

### 2.2 目录名(实验产物)

```
outputs/<tag>_<YYYYMMDD-HHMMSS>/
```
例:`outputs/mask_revive_3x200_20260516-014356/`

每个实验产物目录至少含:
- `run_meta.json`(git_commit + args + 开始时间)
- `report.md`(automated,简短)
- 数据(trace.jsonl / metrics.json / 等)

---

## 3. 架构文档模板(`docs/architecture/<name>_zh.md`)

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

## 8. 实施顺序

| # | 步骤 | 文件 | 验证 |
|---|---|---|---|
| 1 | ... | ... | ... |

## 9. 跟其它路线的关系

(本设计跟其它 architecture 文档的依赖、平行、替代关系。链接出去)

## 10. 文档历史

- YYYY-MM-DD 初稿
- YYYY-MM-DD <修改原因>

````

---

## 4. 实验报告模板(`outputs/reports/<name>.md`)

每个实验报告**必须**有这 8 节,顺序固定:

````markdown
# <Title> — <一句话定位>

生成时间: YYYY-MM-DD HH:MM
对应架构: [`docs/architecture/X_zh.md`](../../docs/architecture/X_zh.md)
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

## 6. 索引条目格式

### `docs/INDEX_zh.md` 里的条目

```markdown
- 🟢/🟡/⚫ [`name`](./architecture/name_zh.md) — 一句话定位(< 80 字符)
```

`🟢` = 当前活的(代码 ground truth);
`🟡` = 参考(留着但不主动改);
`⚫` = 历史(被取代,留考古)

### `outputs/reports/INDEX_zh.md` 里的条目

```markdown
- [`name`](./name.md) (YYYY-MM-DD) — 对应 [arch X](../../docs/architecture/X_zh.md) — 一句话结论。关键指标 N%。
```

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

1. **想做一个新方向** → 先写 `docs/architecture/<name>_zh.md` 草稿(至少 §0、§1、§3、§6)
   → 跟用户对齐 → 补完 §4-§10 → commit
2. **架构跑出实验数据** → 同步写 `outputs/reports/<name>.md`
   → 把图放进 `outputs/reports/<name>/` → commit + 更新两个 INDEX
3. **架构被取代** → 旧文档状态改成 `⚫`,在 INDEX 移动到 archive 区段;不删
4. **发现新概念** → 写到 `GLOSSARY_zh.md`;在使用它的文档里用 `[[term]]` 链接

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
