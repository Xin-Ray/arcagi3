# v4 Phase 4 — Module 1 + 4 人工标注请求

> **作用**: Module 1 (Goal Generation 是否对) + Module 4 (Force-reject 是否对) 我无法从 trace 客观验证(game 真实 win condition 隐藏 + reject 是否正确没有 GT)。请用户标注后,合并到 [`module_validation.md`](./module_validation.md) 给出完整 6 模块 PASS/FAIL 结论。
> **作量**: 5 game,每 game 3-4 个 hypothesis + 2-8 个 rejected_goals。**总约 25-35 个标注**,~15 分钟。

## 0. 怎么标注

### 标注前需要做的(看清楚 game 状态)

每 game 看一眼 `play.gif`(就在 outputs 目录下),知道游戏在干啥:

| Game | GIF 路径 |
|---|---|
| **ar25** | `outputs/v4_phase4_g1_ar25_s42_20260519-053015/round_00/play.gif` |
| **bp35** | `outputs/v4_phase4_g2_bp35_s42_20260519-062440/round_00/play.gif` |
| **cd82** | `outputs/v4_phase4_g3_cd82_s42_20260519-065756/round_00/play.gif` |
| **cn04** | `outputs/v4_phase4_g4_cn04_s42_20260519-080429/round_00/play.gif` |
| **dc22** | `outputs/v4_phase4_g5_dc22_s42_20260519-084820/round_00/play.gif` |

(看不出来通关条件也没关系 — 标 "不确定")

### 标注规则

**Module 1 (Goal Generation 是否对)**:
- 判断 Reflection 写的 hypothesis **是否描述了游戏的真实 win 条件**
- 选项: `YES` / `NO` / `PARTIAL` / `UNSURE`
  - `YES`: hypothesis 描述的状态等于(或非常接近)游戏 win 时的状态
  - `NO`: hypothesis 描述的是无关的事,跟 win 没关系
  - `PARTIAL`: hypothesis 命中了 win 条件的一部分,但缺信息 / 有错误命名
  - `UNSURE`: 看不懂游戏 win 条件,标这个就行

**Module 4 (Force-reject 是否对)**:
- 这些 hypothesis **被 orchestrator 拒绝了**(parser parse 通过但 verdict=None,或 achieved 但 env!=WIN)。判断:**这个 reject 是否正确**?
- 选项: `CORRECT` / `WRONG` / `UNSURE`
  - `CORRECT`: 这个 hypothesis 本来就错,reject 是对的
  - `WRONG`: 这个 hypothesis 其实是对的,我们误伤了
  - `UNSURE`: 看不出

## 1. ar25 标注

> game 状态参考: `outputs/v4_phase4_g1_ar25_s42_20260519-053015/round_00/step_0000.png` (起始) → `step_0099.png` (末) → 看 `play.gif`

### 1.1 Module 1 — hypothesis 是否对 (3 条)

| # | step | round | hypothesis | 你标注 (YES/NO/PARTIAL/UNSURE) |
|---|---:|---|---|---|
| 1 | 69 | round_01 | "match the moving purple 1x1 (obj_008) to the static purple target" | **____** |
| 2 | 81 | round_01 | "match the moving purple 1x1s to the static purple targets" | **____** |
| 3 | 83 | round_01 | "match the moving purple 1x1 (obj_018) with the static purple target" | **____** |

### 1.2 Module 4 — 这些 hypothesis 被 reject 是否对 (8 条)

| # | round | rejected hypothesis | 你标注 (CORRECT/WRONG/UNSURE) |
|---|---|---|---|
| 1 | round_00 | "match the moving blue square to the static blue target" | **____** |
| 2 | round_00 | "match the moving yellow square to the static yellow target" | **____** |
| 3 | round_00 | "match the moving yellow 1x1 (obj_000) to the static yellow target" | **____** |
| 4 | round_00 | "match the moving yellow size=45 object to the static yellow target" | **____** |
| 5 | round_00 | "match the moving yellow size=45 object (obj_0) to the static yellow target" | **____** |
| 6 | round_01 | "match the moving purple 1x1 (obj_008) to the static purple target" | **____** |
| 7 | round_01 | "match the moving purple 1x1 (obj_010) to the static purple target" | **____** |
| 8 | round_01 | "match the moving yellow size=18 object (obj_017) to the static yellow target" | **____** |

## 2. bp35 标注

> GIF: `outputs/v4_phase4_g2_bp35_s42_20260519-062440/round_00/play.gif`

### 2.1 Module 1

| # | step | round | hypothesis | 标注 |
|---|---:|---|---|---|
| 1 | 2 | round_00 | "match every gray, green, maroon, navy, and purple object with a corresponding static target" | **____** |
| 2 | 7 | round_00 | "match every purple object with a corresponding static target" | **____** |
| 3 | 1 | round_01 | "match every rose with a corresponding static target" | **____** |

### 2.2 Module 4

| # | round | rejected hypothesis | 标注 |
|---|---|---|---|
| 1 | round_00 | "match every red dot with a red target square" | **____** |
| 2 | round_00 | "match the moving blue square to the static blue target" | **____** |
| 3 | round_00 | "match every gray, green, maroon, navy, and purple object with a corresponding static target" | **____** |
| 4 | round_00 | "match every purple object with a corresponding static target" | **____** |
| 5 | round_00 | "match the moving blue square (obj_174) with the static blue target" | **____** |

## 3. cd82 标注

> GIF: `outputs/v4_phase4_g3_cd82_s42_20260519-065756/round_00/play.gif`

### 3.1 Module 1

| # | step | round | hypothesis | 标注 |
|---|---:|---|---|---|
| 1 | 1 | round_00 | "align the two yellow 1x1s (obj_001 and obj_000) vertically in the left column" | **____** |
| 2 | 4 | round_00 | "align the two yellow 1x1s vertically in the left column" | **____** |
| 3 | 37 | round_00 | "align the two yellow 1x1s (obj_000 and obj_001) vertically in the left column" | **____** |
| 4 | 0 | round_01 | "align the two yellow 1x1s (obj_000 and obj_001) vertically in the left column" | **____** |

### 3.2 Module 4

| # | round | rejected hypothesis | 标注 |
|---|---|---|---|
| 1 | round_00 | "align the two yellow 1x1s (obj_001 and obj_000) vertically in the left column" | **____** |
| 2 | round_00 | "align the two yellow 1x1s vertically in the left column" | **____** |

## 4. cn04 标注

> GIF: `outputs/v4_phase4_g4_cn04_s42_20260519-080429/round_00/play.gif`

### 4.1 Module 1

| # | step | round | hypothesis | 标注 |
|---|---:|---|---|---|
| 1 | 0 | round_00 | "move the yellow 1x1 (obj_002) to the bottom edge" | **____** |

### 4.2 Module 4

| # | round | rejected hypothesis | 标注 |
|---|---|---|---|
| 1 | round_00 | "move the yellow 1x1 (obj_002) to the bottom edge" | **____** |
| 2 | round_00 | "match the moving blue square to the static blue target" | **____** |
| 3 | round_01 | "align the two red squares vertically in the left column" | **____** |

## 5. dc22 标注

> GIF: `outputs/v4_phase4_g5_dc22_s42_20260519-084820/round_00/play.gif`

### 5.1 Module 1

| # | step | round | hypothesis | 标注 |
|---|---:|---|---|---|
| 1 | 0 | round_00 | "align the two red squares vertically in the left column" | **____** |
| 2 | 0 | round_01 | "align the two red squares vertically in the left column" | **____** |

### 5.2 Module 4

(dc22 没有 rejected_goals,跳过)

## 6. 总分计算 (你标完后我会自动算)

标完后我会:
1. 把你填的标注汇总到 `module_validation.md`
2. 算 Module 1 PASS rate = (#YES) / total
3. 算 Module 4 PASS rate = (#CORRECT) / total
4. 加到 6 模块 PASS/FAIL 表里
5. 跟 Phase 4 0/5 wins 关联,定位真实 bottleneck

## 7. 如果觉得规则不清

跟我说,我再细化。**最重要的就是 Module 1 — 你判断这个 hypothesis 是不是描述了 game 真实的 win condition**。如果 5 个 game 大部分都是 UNSURE / NO,那说明 **Reflection 在猜目标的能力本来就不行**,这就是 bottleneck;如果都 YES 那说明 hypothesis 对,但 Action 没能力执行,这是另一个 bottleneck。
