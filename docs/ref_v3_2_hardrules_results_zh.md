# v3.2 硬规则实测结果 —— ar25 3 round × 30 step

日期: 2026-05-14
状态: 🟢 reference
前置阅读: [`arch_v3_2_zh.md`](./arch_v3_2_zh.md) 设计, [`ref_v3_2_dataflow_zh.md`](./ref_v3_2_dataflow_zh.md) 数据流

> 这份文档是 R1 + R2 + R3 三条 orchestrator-level 硬规则（commit `1bac4be`）实施前后,在 ar25 上跑 3 round × 30 step 的真实对比。把"prompt 只能劝、orchestrator 才能管"这条原则的有效性量化了。

---

## 0. 三条硬规则一句话回顾

- **R1** —— `Knowledge.merged_with_delta` 拒绝 `"unknown" / "none" / "n/a" / "tbd"` 等 sentinel 字符串作为 `goal_hypothesis_update`,避免 Reflection 违规字面量污染 Knowledge。`arc_agent/knowledge.py`。
- **R2** —— `compute_action_mask` 在 orchestrator 调 `env.step()` 之前介入,把"≥5 次全 no-op"或"Knowledge 规则/失败策略中被显式标记"的 action 屏蔽掉,自动替换为 untried > known-good > random non-blocked。`arc_agent/action_mask.py` + `scripts/run_v3_multi_round.py`。
- **R3** —— `ActionAgent.choose` 末尾的兜底:`no_op_streak ≥ 5` 或 `state_revisit_count ≥ 5` 时强制选 untried legal action。`arc_agent/agents/action_agent.py`。

---

## 1. 实验设置

| 项 | 值 |
|---|---|
| 游戏 | `ar25-0c556536` |
| 轮数 / 步数 | 3 round × 30 step |
| 后端 | Qwen2.5-VL-3B-Instruct (text-only, 4-bit, max_new_tokens action=96 reflection=250) |
| 随机种子 | 42 |
| v1 输出目录 | `outputs/v3_2_ar25_3x30/` |
| v2 输出目录 | `outputs/v3_2_ar25_3x30_v2/` |

两次 run 唯一差异 = `1bac4be` 引入的 R1 + R2 + R3。其它代码 (prompt、perception、Knowledge schema) 完全一致。

---

## 2. 核心指标对比

| 指标 | v1 (无硬规则) | v2 (R1+R2+R3) | 变化 |
|---|---:|---:|---|
| **change_rate** round 0 | 23% | **60%** | +37 pp |
| **change_rate** round 1 | 17% | **100%** | +83 pp |
| **change_rate** round 2 | 17% | **97%** | +80 pp |
| max `no_op_streak` round 0 | 9 | 3 | -6 |
| max `no_op_streak` round 1 | **16** | **0** | -16 ✅ |
| max `no_op_streak` round 2 | 16 | 1 | -15 ✅ |
| ACTION 多样性 round 1 (distinct picks 送 env) | 2 种 (ACTION1/6) | **5 种** (1/2/3/4/5) | +3 |
| `ACTION6` 真正送 env round 1 | 21 次 | **0 次** | R2 mask 全拦 |
| `R2 override` 次数 (round 0/1/2) | N/A | 18 / 22 / 23 | 大量生效 |
| `R3 forced_explore` 标记 | N/A | 0 | R2 已经治本,R3 没用上 |
| `goal_hypothesis = "unknown"` 字符串 | 永远 | 不再出现 | R1 ✅ |
| levels_completed | 0 | 0 | (ar25 还没破) |

**结论**:Knowledge 累积到了行为上 —— change_rate 三倍翻,no_op_streak 几乎归零,ACTION 多样性翻倍。

---

## 3. 哪条规则起的作用

### R2 是主力(treat-the-cause)

ar25 的 LLM 行为问题不是"卡住后没法出来",而是"明知 ACTION6 没用还反复选"。R2 在因果链最早一环介入:

```
LLM choose ACTION6 (5,60)              ← LLM 决策(无视 failed_strategies)
        ↓
Orchestrator: outcome_log.n_tried("ACTION6") >= 5 AND n_changed == 0
        ↓
mask = {"ACTION6"}, 选 untried = ACTION3
        ↓
env.step(ACTION3)  -- LLM 选的 ACTION6 永远到不了 env
```

v1 trace 显示 round 1 LLM 自相矛盾选 ACTION6 共 21 次,**v2 trace 显示 LLM 还在选 ACTION6 共 22 次,但全部被 R2 替换,真正送 env 的 ACTION6 = 0**。

### R3 没必要触发

v2 max `no_op_streak` 是 3 / 0 / 1,远低于 R3 阈值 5。R2 在源头解决了"反复选无效 action"问题,no-op streak 根本没机会累积。R3 作为最后一道防线还在,这局 ar25 没用上。

### R1 工作但暴露了新问题

- v1: `goal_hypothesis` 永远卡在字符串 `"unknown"` (因为 Reflection 违规)
- v2: R1 过滤生效,`"unknown"` 不再渗入。**但 Reflection 又出了个新错** —— round 1 step 8 / round 2 step 0 把 `failed_strategies` 的内容 `"ACTION6 anywhere in the right half."` 写到了 `goal_hypothesis_update`。R1 的 sentinel 集合不拦这条,所以这条假目标渗进 Knowledge。

```json
// v2 final knowledge — note goal_hypothesis is failed_strategies content
{
  "goal_hypothesis": "ACTION6 anywhere in the right half.",
  "goal_confidence": "low",
  "failed_strategies": ["ACTION6 anywhere in the right half."]
}
```

---

## 4. R2 mask 的工作记录(示例)

`outputs/v3_2_ar25_3x30_v2/round_01/trace.jsonl` 的前几步,可以看到 R2 怎么拦的:

```text
step 0: action=ACTION1                                          ← LLM 选 ACTION6 → mask 替换为 ACTION1
        orch_override: "untried ACTION1 over masked ACTION6"
step 1: action=ACTION1                                          ← LLM 又选 ACTION6 → 又替换
        orch_override: "untried ACTION1 over masked ACTION6"
step 2: action=ACTION2                                          ← 替换为 ACTION2(此时 ACTION1 已 tried,ACTION2 untried)
        orch_override: "untried ACTION2 over masked ACTION6"
...
```

22 次替换全部记进 trace,可审计、可分析、可回滚(只要不传 mask 参数就回到 v1 行为)。

---

## 5. 还剩两个明显问题

### 问题 1 —— Reflection 把字段串错

Reflection 把 failed_strategies 内容写到 goal_hypothesis_update,R1 的 sentinel 集合管不到。这是模型字段对应错误,不是字面量违规。

**候选 R5 修复方向**(放后面做):
- 在 `Knowledge.merged_with_delta` 检测 `goal_hypothesis_update` 字符串是否同时出现在 `failed_strategies_append` / 已有 `failed_strategies`,如果是,拒绝
- 或检测 goal_hypothesis_update 是否以 "ACTION_X" / "anywhere" 等失败策略词开头

### 问题 2 —— ar25 还没通关

3 round × 30 step 的预算下没有任何 round 通关。注意:

- v2 的 change_rate ~100% 说明 agent **每步都在让画面变**,但**没有进入 WIN 状态**
- 这意味着 ar25 的过关条件不只是"让对象动",而是某种特定 spatial 配置或目标位置
- Knowledge 当前没推出来这个目标(`goal_hypothesis` 还是被错填的内容)
- Reflection 拿到的 outcome 信息够不够推目标? `primary_direction + distance + frame_changed` 只够推单步效果,不够推"哪个对象走到哪里 = 赢"

**下一步候选**(优先级未定):
- 拉长 max_actions 到 80(原设计预算)再跑
- 改 Reflection prompt 引导它推目标(给具体例子: "if a colored object reaches the same color target, that's the win condition")
- 在 step_summary 里加更结构化的 spatial 信息(每个 active object 的位置变化)
- 看 ar25 这个 game 类的实现源码,确认真实的 win condition

---

## 6. 复现命令

```bash
# v1 复现(rollback hard rules):
git checkout c00a825
.venv/Scripts/python.exe scripts/run_v3_multi_round.py \
    --game ar25 --rounds 3 --max-actions 30 \
    --output outputs/v3_2_ar25_3x30 --tag v3_2_ar25_3x30

# v2 (当前 main):
.venv/Scripts/python.exe scripts/run_v3_multi_round.py \
    --game ar25 --rounds 3 --max-actions 30 \
    --output outputs/v3_2_ar25_3x30_v2 --tag v3_2_ar25_3x30_v2
```

每次 run 大约 13 分钟(Qwen2.5-VL-3B 4-bit 加载 + 90 步 × ~8s/step)。

---

## 7. 结论

**硬规则路线得到验证**。Prompt 改 5 次都拦不住 Qwen-3B 自相矛盾选 ACTION6,加 R2 之后**一次性归零**。设计文档 [`arch_v3_2_zh.md`](./arch_v3_2_zh.md) §0 的判断 "v3 单 agent 隐含完成历史总结、推断、决策,后果是模型不积累" 在 ar25 上得到反向印证:**只要把"基于历史的决策"从模型手里拿走,放到 orchestrator,模型就能做好它擅长的(选未试过的、给单步语义)**。

下一步要解决的是"如何让 Reflection 真的推出 win condition",而不是"如何让 Action 听 Reflection 话"——后者已经被硬规则拍死了。

---

*此文档量化的是 commit `1bac4be` 前后的对比。如果 R5 (goal_hypothesis 字段交叉污染过滤)实施,在这份文档加 §3.X 即可。*
