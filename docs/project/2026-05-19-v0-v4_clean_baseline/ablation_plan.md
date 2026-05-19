# 2026-05-19 v4 — Ablation Plan (Phase 3 spec)

## 实验设计

每 ablation 在 V4 baseline 上加**单一**旧模块,跟 baseline 直接对照。

**控制变量**:
- 同 game: ar25
- 同 max-actions-total: 100 (短 ablation,Phase 4 才 200)
- 同 seed: 42
- 同 backbone + reasoning_mode (由 Phase 1 决定)
- 同 token budgets

**唯一变量**: 一次只开一个 flag。

## Ablation 列表

| # | Tag | 新加模块 | CLI 与 V4 baseline 差异 | 预期 |
|---:|---|---|---|---|
| 0 | `v4_baseline` | (none) | 全砍 (`--click-targets off --propose off --action-semantics-from-llm off --hard-rules off`) | reference |
| 1 | `v4_plus_click` | click_targets bandit | `--click-targets on` 其余同 baseline | 无效 / 更差 (cross-validation 0/5) |
| 2 | `v4_plus_propose` | action_proposer K=3 | `--propose on` | 更差 (ACTION1 bias 重现) |
| 3 | `v4_plus_semantics` | action_semantics from LLM | `--action-semantics-from-llm on` | 更差 (known-good prior 重现) |
| 4 | `v4_plus_hard_rules` | R1/R4/R5/R6/R7 | `--hard-rules on` | 中性 — 这批没单独测过 |

总 wall: 5 × ~30-60 min = 2.5-5 小时。

## 命令模板

```powershell
# V4 baseline (ab #0)
.venv\Scripts\python.exe scripts\run_v3_multi_round.py `
  --game ar25 --max-actions-total 100 --max-actions 100 --rounds 1 `
  --seed 42 --mask off --propose off `
  --click-targets off --action-semantics-from-llm off `
  --hard-rules off --validate-hypothesis-schema wide `
  --backbone HuggingFaceTB/SmolLM3-3B `
  --reasoning-mode {phase1_winner} `
  --max-new-tokens-action {tbd} --max-new-tokens-reflection 2048 `
  --tag v4_ablate_baseline_s42 `
  > outputs\v4_ablate_baseline_s42.log 2>&1
```

每个 ablation 替换相应 flag(`--click-targets on` / `--propose on` / 等)+ 改 tag。

## 评估指标 (per ablation)

| 指标 | 提取自 | 解读 |
|---|---|---|
| `change_rate` | trace.jsonl: changed/total | 行为多样性总指标 |
| `levels_won` | summary.json | 通关核心目标 |
| `parser_triggered` | trace: goal_pred_kind != '' count | 反思闭环是否激活 |
| `parser_verdict_non_None` | trace: goal_achieved_det != None count | 闭环是否真 evaluate |
| `hypothesis_change_count` | knowledge_per_step diff | Reflection 是否真改目标 |
| `force_reject_count` | grep "HALLUCINATED" or "HYPOTHESIS REJECTED" in action_raw or current_alert | force-reject 是否触发 |
| `action_diversity` | unique(actions) / total | ACTION1 spam 检测 |
| `reasoning_empty_rate` | trace: reasoning='' / total | parsing 失败率 |
| `wall_time_sec` | log timestamps | 跑得快还是慢 |

## 比较表 (Phase 3 完成后生成)

| ab | change | won | parser_trig | parser_eval | hyp_chg | force_rej | act_div | rsn_empty | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 baseline | _ | _ | _ | _ | _ | _ | _ | _ | _ |
| +click | Δ | Δ | Δ | Δ | Δ | Δ | Δ | Δ | _ |
| +propose | Δ | Δ | … | … | … | … | … | … | _ |
| +semantics | … | … | … | … | … | … | … | … | _ |
| +hard_rules | … | … | … | … | … | … | … | … | _ |

(Δ vs baseline)

## 决策 (after all 4 done)

- 找出 **每个 ablation 是否使 baseline 改善 / 退化 / 中性**
- 选出 "有用" 集合 → Phase 4 用 V4 baseline + 该集合
- 如果**全部退化**,只跑 V4 baseline 进 Phase 4
- 如果**全部改善**(不太可能),最终配置 = full v3.2

## 失败 fallback

- 任何 ablation crash → 跳过,继续下一个,日志记录
- 全部跑完 wall > 5 小时 → 跳 Phase 4,直接报告
- 某个 ablation 在 < 20 step 就崩了 → 可能是新模块跟 v4 不兼容,标"crash" 不计 metric

## 引用

- `architecture.md` §3 (CLI flag 一览)
- `decision_tree.md` Phase 3 (顶层决策)
- 上游证据:
  - click_targets 0/5: `docs/project/2026-05-18-v0-det_goal_plus_force_cot/report.md` §5.4
  - action_proposer + semantics bias: `docs/project/2026-05-18-v0-det_goal_v3_extended_parser/report.md` (smoke 3-4 raw)
  - hard_rules: 各 commit 单测,没合并 production ablation
