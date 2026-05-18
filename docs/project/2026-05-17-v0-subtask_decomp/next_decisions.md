# 2026-05-17 v0 subtask_decomp — Next Decisions (force_cot 后续动作)

> **写于**: 2026-05-18 force_cot batch 跑到 60% 进度时 (T-SEL-1 in progress)。
> **目的**: 用户睡前授权 autonomous 执行,这是 batch 完成后的决策树 + 安全约束。

## 0. 已知部分结果 (此刻)

| Subtask | Baseline default | force_cot 进度 |
|---|---:|---:|
| T-NAV-1 | 52% | **71.0%** done (+19pp, FAIL) |
| T-NAV-2 | 95% PASS | **93.0%** done (-2pp, 仍 PASS) |
| T-NAV-3 | 67% FAIL | **97.0%** done (**+30pp, PASS**) |
| T-SEL-1 | 78% FAIL | 跑中 (running_acc 60% @ 10/100) |
| T-GOAL | 30% SEVERE FAIL | 未开始 |

## 1. 流程

batch 完成 (`outputs/bench_subtask_batch_v2_forcecot.log` 出现 "batch done") 后:

1. 读 5 个 `outputs/subtask_<T-*>_<ts>/metrics.json` + `per_probe_smollm3-cot.jsonl`
2. 算 `long_acc`/`short_acc`/CoT 激活率 (每个 subtask)
3. 写 `docs/project/2026-05-18-v0-force_cot/report.md` (A/B 对照)
4. 同步 `docs/README.md` 版本历史 + 分支表 + 验证汇总
5. 同步 `docs/GLOSSARY_zh.md`: 加 `force_cot` 词条,更新 5 个 T-* 条目数字
6. Commit 在 `feat-2026-05-17-v0-subtask-T-NAV-1` 分支
7. **Push 到 origin** (用户授权)
8. 按 §2 表选 next 分支
9. 在新分支上跑 1 个实验 (不 push 新分支)

## 2. 按 T-GOAL force_cot 结果选下一步分支

| T-GOAL force_cot acc | 含义 | 新分支 | 实验 |
|---|---|---|---|
| **≥ 80%** | force_cot 单 prompt 改动就能让 LLM 做 goal 识别 | `feat-2026-05-18-v0-force_cot_production` | 把 force_cot 写进 `prompts_v3_2.py` (Action + Reflection),跑 `run_v3_multi_round.py --game ar25 --rounds 2 --max-actions 100`,看 win 率 |
| **40-80%** | force_cot 有帮助但不够 (典型) | `feat-2026-05-18-v0-deterministic_goal_plus_force_cot` | (a) 建 `arc_agent/goal_evaluator.py` 解析 hypothesis + 读 ObjectMemory 做 deterministic 判断 (b) Reflection Agent 用 evaluator OVERRIDE LLM goal signal (c) 同时 force_cot 进 prompts。跑 1×2×100 ar25 |
| **< 40%** | LLM 做不了 goal recognition | `feat-2026-05-18-v0-deterministic_goal` | 只做 (a)+(b),不动 prompts |

## 3. 其他 4 subtask 的解读 (不影响 branch 选择)

| 结果 | 行动 |
|---|---|
| T-NAV-2 < 90% | 异常,force_cot 破坏 PASS task — 报告 §6 标 BUG-14 |
| T-NAV-2 ≥ 90% | 维持 PASS (✅ 已知 93%) |
| T-NAV-1 ≥ 90% | bonus,但不改 branch |
| T-NAV-1 70-90% (✅ 已知 71%) | 模型对 letter-shuffle attention 弱 — 报告 §7 P2 (Qwen3-4B/SmolLM3-7B) |
| T-NAV-1 < 70% | force_cot 比 default 差 → 异常,标重跑 |
| T-SEL-1 ≥ 90% | P1 备注: 坐标约定不是真瓶颈,prompt 解决 |
| T-SEL-1 70-90% | 报告 P1 fix: 把题面 (x,y) 改 (row,col) |
| T-SEL-1 < 70% | force_cot 没救 → P2: click_targets bandit + 容错 |

## 4. 新分支实验 spec

### Scenario A (T-GOAL ≥ 80%) — force_cot 进 production

```powershell
.venv/Scripts/python.exe scripts/run_v3_multi_round.py `
  --game ar25 --rounds 2 --max-actions 100 --seed 42 `
  --mask strict --propose on `
  --backbone HuggingFaceTB/SmolLM3-3B --reasoning-mode cot `
  --max-new-tokens-action 1024 `
  --tag force_cot_prod_ar25_2x100 `
  > outputs/force_cot_prod_ar25_2x100.log 2>&1
```

需先改 `arc_agent/prompts_v3_2.py:build_action_user_prompt` 末尾从默认 "Output reasoning + action" 改成 "Solve step by step. Compute Δrow/Δcol if relevant. Show your work, then output Action: ACTION_X"。

### Scenario B (40-80%) — deterministic goal + force_cot

1. **`arc_agent/goal_evaluator.py`** (新):
   ```python
   def evaluate_goal(goal_hypothesis: str,
                    objects: dict[str, ObjectRecord]) -> bool:
       """Parse a natural-language goal and check if it's achieved.

       Handles patterns like:
       - "align two yellow squares vertically in left column" → all yellow.col == 0
       - "move red square to col=5"                          → red.col == 5
       - "stack blue on top of green"                        → blue.row == green.row - 1
       Returns True/False; None if can't parse.
       """
   ```
2. **单测**: `tests/test_goal_evaluator.py` 至少 4 case (vertical-align / row-align / col-target / unknown-pattern)
3. **接入**: `arc_agent/agents/reflection_agent.py`,在算 `reflection.delta` 后,如果 evaluator returns True,把 delta 改成 `{"goal_achieved": True}`。
4. **跑 1×2×100 ar25**:
   ```powershell
   .venv/Scripts/python.exe scripts/run_v3_multi_round.py `
     --game ar25 --rounds 2 --max-actions 100 --seed 42 `
     --mask strict --propose on `
     --backbone HuggingFaceTB/SmolLM3-3B --reasoning-mode cot `
     --max-new-tokens-action 1024 `
     --tag det_goal_ar25_2x100 `
     > outputs/det_goal_ar25_2x100.log 2>&1
   ```

### Scenario C (< 40%) — 只做 deterministic goal

同 B 但跳过 force_cot prompt 改动。

## 5. 安全约束 (用户睡时)

- **不删任何文件**,新建为主
- **不在新分支上 push** — 只 push subtask-T-NAV-1 一次
- **不动 main**
- 新实验中途 crash:log 留 outputs/,标 failed,继续等用户醒
- 磁盘/GPU OOM/Python 异常:保存 partial,诚实标"未跑完"
- **不开第二个 GPU 实验** 避免争抢
- 任何 git push 失败:记录到下条 chat,不 retry-force

## 6. 失败兜底

如果 batch 中途 crash (例如 OOM):
- 读已有 metrics.json,标其余 "INTERRUPTED"
- 报告里照写已完成 subtask
- 不开新实验,让用户决定

如果 push 失败 (网络/auth):
- 提交保留,新分支不开
- 在 chat 报告 push 失败 + error
- 等用户处理 auth

如果新分支实验启动失败:
- 文件全保留
- 报告 INIT_FAIL 不继续
