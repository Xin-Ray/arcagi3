# docs/ 总索引

这七份文档构成项目的"工作记忆"和"对外作品"。**任何会话开工前必须先读相关文档,收工前必须把变化写回去。** CLAUDE.md 里有强制规则。

## 七份文档

前六份是**内部跟踪**(我们怎么干活),`PAPER.md` 是**对外作品**(怎么把成果讲给别人):

| 文档 | 角色 | 何时读 | 何时写 |
|------|------|--------|--------|
| [ROADMAP.md](ROADMAP.md) | 任务清单 + 时间推进 | 每次会话开始,确认当前阶段、下一步任务 | 完成里程碑、调整阶段、更新日期时 |
| [LIBRARY.md](LIBRARY.md) | 函数索引(API 级,`arc_agent/` 包) | **每次写代码前**,先搜本文件看有无现成函数 | 新增/修改/废弃任何库函数时 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 架构逻辑(模块职责、数据流、接口契约) | 设计新模块、改接口、做架构决策时 | 接口或职责发生变化时 |
| [RESEARCH.md](RESEARCH.md) | 文献 + 已尝试方案 + 当前缺陷 | 设计新方法前,看有没有踩过的坑 | 读了相关论文、试了新方法(无论成败)、发现新缺陷时 |
| [EXPERIMENTS.md](EXPERIMENTS.md) | 关键实验结果(数字和结论) | 比较方案、写报告、判断是否回归时 | 每次跑完有结论的实验后 |
| [PAPER.md](PAPER.md) | **活论文**(Background/Related Work/Method/Experiments/Discussion/References) | 阶段汇报、Milestone 提交前、需要把项目讲给外人时 | 实验出新结论、读了新文献、改了方法、阶段切换时 |

`PAPER.md` 的存在意义:强迫我们时不时回到"对外讲清楚"的视角,顺便为 Milestone #1(2026-06-30)/ #2(2026-09-30)的开源提交准备好材料。它**不引入新事实**,只把别处的事实重组成论文叙述 —— 数字必须能在 EXPERIMENTS 找到出处,引用必须有 URL。

## 维护规则

1. **HEI 强制**(开发流程,见 CLAUDE.md):每个实验/方案先写 **Hypothesis(预测数字)**,跑完写 **Iteration trigger(下一步)**。没有这两条的实验/方案不算完成。
2. **不要重复**:同一信息只在一个文档落地。例如分数曲线在 EXPERIMENTS,方法论判断("这条路通不通")在 RESEARCH。
3. **保留最新值,不删除历史**:实验结果、尝试过的方案永远 append,不覆盖,这样后人能看到走过的路。
4. **失败比成功更要写**:负面结果是 RESEARCH.md 的核心价值。
5. **日期都用绝对日期**(YYYY-MM-DD),不要 "上周" "前两天"。
6. **每份文档顶部维护"最近一次更新"日期**,方便判断陈旧度。

## 附:操作手册(非七件套)

- [HUMAN_TRACE_GUIDE.md](HUMAN_TRACE_GUIDE.md) — 人工 think-aloud 标注操作手册(Phase 1 对照实验:人工 trace vs Claude silver-label trace)。**操作流程文档,不引入方法论判断**;hypothesis 仍在 RESEARCH.md 提案 #4,实验登记仍在 EXPERIMENTS.md。

## 修改本目录的流程

- 加新文档:先在这里加一行说明,再写文档本身。
- 重命名/删除:同步改 CLAUDE.md 里的引用。

---
最近一次更新:2026-05-08(CODE_MAP.md + EXPERIMENTS.md 从 archive/ 迁回 docs/,并追加 Phase 2 / vlm_test 内容)
