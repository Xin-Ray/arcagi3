# 实验报告总入口 (outputs/reports/INDEX_zh.md)

> 所有实验报告的索引。每条 ≤ 120 字符。最新在最上面。
>
> 报告写作模板见 [`../../docs/CONVENTIONS_zh.md`](../../docs/CONVENTIONS_zh.md) §4。

最近更新: 2026-05-16

---

## 🟢 当前活的(代码 ground truth 在这里)

| 报告 | 日期 | 对应架构 | 一句话结论 | 关键指标 |
|---|---|---|---|---|
| [`mask_revive_3x200.md`](./mask_revive_3x200.md) | 2026-05-16 | [v3_2](../../docs/architecture/v3_2_zh.md) | R2 mask 重启 + 修归因 bug;change_rate 5-8% < v2 canary 60-100%;ACTION1 过度承诺成新病灶 | change_rate 5.5/7.0/8.5%; overrides 1/82/1 |
| [`predictor_v0.md`](./predictor_v0.md) | 2026-05-16 | [predictor_v0](../../docs/architecture/predictor_v0_zh.md) | 4 架构对比:LogReg/MLP-S/MLP-L/CNN-small。CNN +6pp 但 val n=14 太小,不接入 prompt | val AUC 0.802/0.836/0.838/**0.894** |
| [`trace_balance.md`](./trace_balance.md) | 2026-05-16 | [predictor_v0](../../docs/architecture/predictor_v0_zh.md) | 6132 trace step 整体 change_rate 42%,**驳斥「都是失败数据」担心**;5 demo game 覆盖 | 2599 changed / 3533 no-op |
| [`grpo_v0_ar25_plan.md`](./grpo_v0_ar25_plan.md) | 2026-05-16 | [grpo_v0](../../docs/architecture/grpo_v0_zh.md) | GRPO 单游戏诊断报告 6 张图布局;Phase 0 未开工 | — (skeleton) |
| [`../v3_p0b_p1_full/report.md`](../v3_p0b_p1_full/report.md) | 2026-05-14 | [v3](../../docs/architecture/v3_zh.md) | v3.1 (P0-A + P0-B + P1) 5 game × 80 step 终验收。带 5 GIF。**v3 时代的最新基线** | RHAE 0 (across all 5 games) |
| [`../goal_inference/report.md`](../goal_inference/report.md) | 2026-05-13 | [v3](../../docs/architecture/v3_zh.md) | VL+image vs Text-3B+scipy 猜每个游戏目标;**Text-3B 给具体假设,VL 给套话** | qualitative |
| [`../spatial_bench_combined/report.md`](../spatial_bench_combined/report.md) | 2026-05-13 | [v3](../../docs/architecture/v3_zh.md) | VL-text vs Pure-text 17 空间题;**两模型 enriched 都 12/17 打平** | 12/17 各 |
| [`../scipy_object_diag/SUMMARY.md`](../scipy_object_diag/SUMMARY.md) | 2026-05-13 | [object_pipeline](../../docs/reference/object_pipeline_zh.md) | scipy 对象提取人审 4 game × 5 帧;**ar25 100% 正确**;切 scipy 的依据 | 100% on ar25 |
| [`../qwen_object_diag/SUMMARY.md`](../qwen_object_diag/SUMMARY.md) | 2026-05-13 | [object_pipeline](../../docs/reference/object_pipeline_zh.md) | Qwen-VL 提取同 4×5 帧;**严格内容正确率 0%**;反向支持 scipy 路线 | 0% strict |
| [`../ablation_overnight_/report.md`](../ablation_overnight_/report.md) | 2026-05-13 | [v3](../../docs/architecture/v3_zh.md) | v1 6 agent × 5 G_base × 80 step ablation;**全 RHAE = 0** → 决定推 v3 | RHAE all 0 |

---

## ⚫ 已归档(被新版取代,本地副本在 `outputs/reports/`,原始资源在 `archive/outputs_2026-05-14/`)

| 报告(本地副本) | 日期 | 取代它的 | 一句话 |
|---|---|---|---|
| [`2026-05-14_v3_visual_full.md`](./2026-05-14_v3_visual_full.md) | 05-14 | `v3_p0b_p1_full` | v3.0 单 agent 全 5 game × 80 步 + 5 GIF。P0-A/B/P1 改进的「基线」 |
| [`2026-05-14_v3_visual_ar25.md`](./2026-05-14_v3_visual_ar25.md) | 05-14 | `v3_p0b_p1_full` | v3.0 ar25 80 步 + GIF 单游戏验证 |
| [`2026-05-14_v3_eval_full_comparison.md`](./2026-05-14_v3_eval_full_comparison.md) | 05-14 | — | v3.0 vs v1 各 agent 横向对比表,**v3 entropy +67% vs A3** |
| [`2026-05-13_spatial_bench_v1.md`](./2026-05-13_spatial_bench_v1.md) | 05-13 | `spatial_bench_combined` | VL with image vs text-only 6 空间题 |
| [`2026-05-13_spatial_bench_v2.md`](./2026-05-13_spatial_bench_v2.md) | 05-13 | `spatial_bench_combined` | enriched prompt 后 17 题扩展 bench |

---

## 📋 怎么用这个索引

- 找**当前 baseline**:看 🟢 表第 1 行 — `mask_revive_3x200`(当前线上)
- 找**某次实验的依据**:对应架构链接 → docs/architecture/ → 看 §6.2 决策门
- 找**老的归档资源**:🟢 表 links 是原位,⚫ 表 links 是本地副本(GIF/PNG 路径已改写指向 archive/)

---

## 🆕 加新实验报告时

1. 跑实验。若长期、长跑,在 `outputs/<experiment>/report.md` 自动生成(自动 report 用模板)。
2. 写人审报告 `outputs/reports/<name>.md`,套 [CONVENTIONS §4](../../docs/CONVENTIONS_zh.md#4-实验报告模板outputsreportsnamemd) 8 节模板。
3. 图放 `outputs/reports/<name>/`,markdown 用相对链接 `./<name>/figure.png`。
4. **回到本文加一行**到 🟢 表(顶部),并把被取代的旧报告降到 ⚫ 表。
5. 在 `docs/INDEX_zh.md` §5 也加一行(同步)。

## 📦 存储规则

- 🟢 活跃:report 留在 `outputs/<exp>/report.md`(原位),本 INDEX 链接过去,**不复制**
- ⚫ 归档:复制 report 到本目录,改写图链接指回 `archive/outputs_<date>/<exp>/`;原 dir 移到 archive。用 `scripts/copy_archived_reports.py` 自动化

---

## 报告 ↔ 架构 配对

```
docs/architecture/<X>_zh.md   ←—对应—→   outputs/reports/<X>*.md
    (写设计意图、决策门、风险)              (写实测结果、判定门、下一步)
                  ↓
        共享 docs/GLOSSARY_zh.md 的术语
```

每个 architecture 至少配一个 report(可以是 skeleton)。每个 report 必须指回它对应的 architecture。

---

*历史:*
- *2026-05-16 按新 conventions 重构,加 active / archived 双区表 + 写作流程*
- *2026-05-14 初始版*
