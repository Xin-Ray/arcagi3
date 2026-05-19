# verify.md — 模块验证总览

> **作用**: 项目方法学心脏。每个子模块的输入 / 输出 / PASS 定义 / bench 验证方式 / **production 集成后如何验证它仍然 work**。
> **触发**: 2026-05-19 用户提的方法学修正 — 之前的 "bench PASS" 在 production 不一定 PASS,**必须有 production-level 验证手段**。
> **配套**: `docs/README.md` §2 (顶层版),`docs/project/2026-05-19-v0-v4_clean_baseline/module_validation.md` (Phase 4 实测)。

---

## 1. 顶层目标 → 7 模块拆解(0-6,沿数据流)

```
通关 1 game (env.state == WIN)
   ↓ 必要条件
推断 win 条件 + 高效执行到该 state
   ↓ 沿数据流拆 7 个模块

帧 (raw 64×64 grid)
   ↓
0. Perception            scipy.ndimage.label → ObjectRecord list (色/bbox/center/cells)
   ↓ + history (前帧 + outcome_log)
1. Goal Generation       Reflection 从结构化对象 + history 推断 win 条件,写 goal_hypothesis
   ↓
2. Goal Recognition      parser/judge 判 "hypothesis 是否已达成"
   ↓ (if achieved=True 但 env!=WIN)
3. Reflection Loop       reject hypothesis → push rejected_goals → 强迫 Reflection 重写
   ↓ (else hypothesis 仍 active)
4. Action Selection      给 hypothesis,选朝 hypothesis 方向的 action
   ↓
5. force_cot Prompt      Action prompt 加 step-by-step + goal-target 措辞,引导推理
   ↓
6. K=3 Candidates        action_proposer 提供 explore 多样性,防 ACTION1 死循环
   ↓
env.step(action)
   ↓
帧 t+1 → 回 0. Perception ...
```

**核心原则**: 视觉用算法,推理用 LLM,两者通过 ObjectRecord 结构化数据连接,LLM 永远不看像素。

---

## 2. 每模块详细规格

### Module 0: Perception (scipy)

| 项 | 内容 |
|---|---|
| **Input** | `grid: np.ndarray (64×64, dtype=int8)` — color index per cell, 0=background |
| **Output** | `list[ObjectRecord]` 每个含 `{id, color, color_name, cells, bbox, center, size, shape_signature}` |
| **代码** | `arc_agent/object_extractor.py:extract_objects` |
| **依赖** | `scipy.ndimage.label`(determined,microseconds) |
| **PASS 定义** | 帧每个 4-连通的非背景同色区域 = 一个 ObjectRecord,**bbox / center / cells 必须完全正确** |
| **Bench 验证** | 跟手工标的 ground truth 对照,在 ar25 / G_base 抽样 N 帧,**100% 一致** |
| **Bench 结果** | scipy 100% vs Qwen-VL 0% (`ref_object_pipeline_zh.md`) |
| **Production 验证** | (a) 跨帧稳定性: 同色同形 obj 出现 / 消失必须有 action 触发,**不能凭空蹦** (b) 在 step PNG 上覆盖 bbox,人工 spot check (c) cells.size = bbox 范围内 of same color |
| **Production 自动化检查** | 跑完 trace 后,扫每一步: 上一步 frame_changed=False 但 obj 列表变了 → 异常 |

### Module 1: Goal Generation (Reflection Agent)

| 项 | 内容 |
|---|---|
| **Input** | `(Knowledge, StepSummary, frame_objects, outcome_log, object_relations, exploration_hint, legal_actions, step, max_steps)` |
| **Output** | `delta = {"goal_hypothesis_update": str?, "action_semantics_update": dict?, "current_alert": str?}` |
| **代码** | `arc_agent/agents/reflection_agent.py:reflect_after_step` + `prompts_v3_2.py:build_reflection_user_prompt` + system prompt `REFLECTION_SYSTEM` |
| **依赖** | LLM backbone (SmolLM3 /no_think 当前) |
| **PASS 定义** | hypothesis 描述的状态 ≈ game 真实 win 条件(语义,非字面) |
| **Bench 验证(理想)** | T-DISCOVER: 给一帧 + 0 prior,问 model 写 hypothesis,跟 game 真实 win 条件比对 |
| **Bench 现状** | ❌ **没 bench**(game GT 隐藏,需要人工标 win 条件) |
| **Production 验证(已做)** | 人工标 `annotation_request.md` Module 1:每 game 抽 3-4 个 hypothesis 标 YES/NO/PARTIAL/UNSURE |
| **Production 验证(待做)** | 当 env-WIN 触发,看那一刻 knowledge.goal_hypothesis 是什么 — 跟 win precondition 比 |
| **集成后跟 bench 一致性检查** | 同一 frame 输入,bench T-DISCOVER 跟 production Reflection 写出来的 hypothesis 应该 same kind 至少 |

### Module 2: Goal Recognition (parser)

| 项 | 内容 |
|---|---|
| **Input** | `(goal_hypothesis: str, frame_objects: list[ObjectRecord])` |
| **Output** | `(achieved: bool | None, GoalPredicate | None)` |
| **代码** | `arc_agent/goal_evaluator.py:evaluate_goal` + `parse_goal_hypothesis` + `evaluate_predicate` |
| **依赖** | 纯 Python regex + 几何判断,deterministic |
| **PASS 定义** | parser=True **iff** env-WIN(precision=recall=1.0) |
| **Bench 验证** | T-GOAL synthetic probe (100 题 4 选 1,25/75 success/failure):**parser 83%**,LLM judge 68% (`goal_judge_ab/report.md`) |
| **Production 验证(自动)** | 每个 env-WIN step,检查 `goal_achieved_det` 是否在最近 5 步内 True 过 |
| **Production 验证(待做)** | 当 verdict=None 但 hypothesis "看起来对",抽样检查 parser 漏判原因 |
| **集成后跟 bench 一致性检查** | 同一 (hypothesis, objects) 输入,bench 和 production 应得 same verdict (parser 是 deterministic,**必须 100% match**) |

### Module 3: Reflection Loop (orchestrator force-reject)

| 项 | 内容 |
|---|---|
| **Input** | `(goal_achieved_det, env.state, knowledge.goal_hypothesis, knowledge.rejected_goals, frame_objects, goal_pred_kind, goal_pred_colors)` |
| **Output** | 修改后的 `knowledge` (清空 hypothesis / append rejected_goals / set alert),返回给下一步 |
| **代码** | `scripts/run_v3_multi_round.py` line ~700-770(evaluate_goal 后的 if/elif 三路 + force-reject 块) |
| **依赖** | Module 2 输出 + env state |
| **PASS 定义** | 被 reject 的 hypothesis **真的是错的**(无误伤);未 reject 的 hypothesis 不假阳性留着 |
| **Bench 验证(理想)** | T-REJECT: 合成 (hypothesis, outcome) 对,有 GT "该不该 reject",测 reject precision/recall |
| **Bench 现状** | ❌ **没 bench** |
| **Production 验证(已做)** | 人工标 `annotation_request.md` Module 4:每 game rejected_goals 标 CORRECT/WRONG/UNSURE |
| **Production 验证(自动 proxy)** | reject 后 Reflection 重写的 hypothesis 跟旧的 dialect 是否 different (不是简单换 obj_id) |
| **集成一致性检查** | 触发 trigger 1 (achieved+!WIN) / trigger 2 (hallucinated colors) / trigger 3 (parser parse 失败) 的步,在 trace.jsonl 记录 `force_reject_reason` 字段,后查 |

### Module 4: Action Selection (Action Agent)

| 项 | 内容 |
|---|---|
| **Input** | `(latest_frame, history, knowledge, click_targets, candidates_for_prompt)` 经 build_action_user_prompt 拼成 prompt |
| **Output** | `(action: GameAction, reasoning: str)`,action 含 coords if ACTION6 |
| **代码** | `arc_agent/agents/action_agent.py:choose` + `prompts_v3_2.py:build_action_user_prompt` |
| **依赖** | LLM backbone + Module 0 (frame_objects) + Module 1 输出 (hypothesis in knowledge) + Module 6 (K=3 candidates) |
| **PASS 定义** | 当 hypothesis 隐含方向时,action 方向 = hypothesis 方向(e.g., hypothesis = move_to_row 63 且 obj 在 row 20 → action 应为 ACTION2 DOWN) |
| **Bench 验证** | T-NAV-1 (单步) 71% (`/think`force_cot), T-NAV-2 (多步) 95%, T-NAV-3 (L 路径) 97% |
| **Production 验证(已做)** | 重审 Phase 4 trace:把 hypothesis parse 出 direction,跟实际 action 方向比对 → **0/794 步有 directional hypothesis,无法测** |
| **Production 验证(待做)** | 先 fix Module 1 让 hypothesis 有方向 → 再算 alignment rate |
| **集成一致性检查** | 找 Phase 4 trace 中 hypothesis kind ∈ {move_to_row/col/center,stack},把那个 step 当 T-NAV-1 probe 重 evaluate;production 选的 action 应该对应 bench 上 same probe 的 correct answer (允许 ±1 letter) |

### Module 5: force_cot Prompt

| 项 | 内容 |
|---|---|
| **Input** | Action prompt template (`_ACTION_ASK_BLOCK` 或 `_ACTION_ASK_BLOCK_MC`) |
| **Output** | 渲染后的 prompt 文本,末尾要求 model "solve step by step. note position. pick ACTION" |
| **代码** | `arc_agent/prompts_v3_2.py:_ACTION_ASK_BLOCK_MC` (multi-choice 版,V4+propose 用这个) |
| **依赖** | 拼到 build_action_user_prompt 的输出 |
| **PASS 定义** | model 的 reasoning 字符串里提到的 ACTION_X == 最终选的 action |
| **Bench 验证** | T-NAV-3 force_cot vs default: 67% → 97% (+30pp on synthetic) |
| **Production 验证(已做)** | Phase 4 trace 重审: reasoning 末尾 ACTION mention vs trace.action → **51% match,49% LLM letter mapping confusion (orch_override=0)** |
| **Production 验证(待做)** | 拆 49% mismatch:是 model 全程混乱,还是只在 K=3 letter shuffle 时混 |
| **集成一致性检查** | 取 trace 中"choice: A" 但 reasoning 写"ACTION5"的步,把它当 T-MAPPING probe(动态 letter shuffle)直接测 SmolLM3,**production 错误率应 ≈ bench T-MAPPING 错误率** |

### Module 6: K=3 Candidates (action_proposer)

| 项 | 内容 |
|---|---|
| **Input** | `(knowledge, outcome_log, legal_actions, recent_action_names, rng, k=3)` |
| **Output** | `list[Candidate]` 长度 ≤ 3,每个含 `(action_name, coords, source, confidence)` |
| **代码** | `arc_agent/action_proposer.py:propose` |
| **依赖** | outcome_log (untried actions) + knowledge.click_targets (已 V4 砍) + knowledge.action_semantics (已 V4 砍) |
| **PASS 定义** | K=3 始终满 + 至少 1 个候选朝 hypothesis 方向 |
| **Bench 验证** | Phase 3 ablation:加 propose 把 V4 baseline 11% change_rate → 86% (+75pp) |
| **Production 验证(已做)** | Phase 4 trace 重审:"choice: A/B/C" 在 raw 出现 78% (22% 步 K<3 不走 multi-choice 格式) |
| **Production 验证(待做)** | 把 K=3 候选 dump 到 trace.jsonl 每步,事后核 "goal-aligned 候选" 出现率 |
| **集成一致性检查** | 给 V4 baseline (11%) 跟 V4+propose (86%) 同 seed,**单一变量 + 75pp delta 必须可复现** |

---

## 3. 集成测试协议 — 6 个"模块一致性"check

每个 module **bench PASS 不等于 production PASS**。集成后必须有一个 check 把 bench 跟 production 桥接。

### 3.1 Module 0 (Perception) 集成 check

跑完 production,扫所有 frame:
```python
for step in trace:
    if not step["frame_changed"]:
        # 对象列表跟上步比应该一样
        assert objs[step] == objs[step-1], "perception unstable"
```

**当前实现**: 没有,trace 不存 obj 列表。**TODO**: 在 trace.jsonl 加 `n_objects` 字段。

### 3.2 Module 1 (Goal Generation) 集成 check

production 写的 hypothesis 必须**至少 dialect 跟 bench 假设的一致**:
- bench 假设 Reflection 写 "align ... vertically" / "to bottom edge" 等 directional/explicit 语言
- production 实测 hypothesis 应 ≥ 50% 落在 parser 已知 dialect

**当前数据**: Phase 4 直观看,Reflection 全 "match X with Y" → 100% align_any kind,**bench 假设跟 production 不一致**。修补: 加 T-DIALECT bench (Reflection 自由写时偏好分布)。

### 3.3 Module 2 (Parser) 集成 check

parser 是 deterministic,production 调用结果应该 100% 等于 bench 同输入。

**自动化**: 拿 production trace 中的 (hypothesis, frame_objects) 对,重跑 evaluate_goal,比对 verdict — **必须 100% match**。

**当前实现**: 待写 `scripts/v4_parser_consistency_check.py`。

### 3.4 Module 3 (Reflection Loop) 集成 check

每次 force-reject 触发,记录到 trace `force_reject_event` 字段:
```json
{"step": 42, "reject_reason": "hallucinated_colors",
 "old_hypothesis": "match red ...",
 "frame_colors": ["yellow", "gray"]}
```

**事后人工抽查** + 自动: reject 触发后**下一步**Reflection 写的 hypothesis 不应该跟 rejected_goals 字面重复(R5 应拦)。

### 3.5 Module 4 (Action Selection) 集成 check

production trace 取 (hypothesis kind, current obj pos, target),**重新构造一个 T-NAV-1 probe**,跑同样 LLM (SmolLM3 /no_think force_cot prompt),看 production 实际选的 action 是否跟 bench 给同 probe 的答案一致。

**预期**: bench T-NAV-1 force_cot 71%,production 同 step 当作 probe 重测 应也是 ~71%。如果 production **远低**于 71%,说明 prompt 在 production 里被其他 block 干扰。

**待写**: `scripts/v4_action_consistency_check.py`。

### 3.6 Module 5 (force_cot) 集成 check

production "model say A, actually B" 的步,把 prompt 整段 + raw response 摘出来,**单独再喂给 same model**(不带 production 的 history 干扰),看是否还是同 reasoning ≠ same action。

**预期**: 如果干净环境也错,是 model letter mapping 能力问题;如果干净环境对,是 production 累积 state 干扰。

### 3.7 Module 6 (K=3) 集成 check

把 K=3 候选 dump 到 trace 每步,事后:
- K<3 的步占比应 ≤ 10%(当前 22%,FAIL)
- K=3 中包含 goal-aligned action 的步占比应 ≥ 80%(未测)

**待写**: 在 `action_agent.choose` 把 `candidates_for_prompt` 序列化到 `_state.last_candidates`,trace 写出来。

---

## 4. 当前各模块 PASS / FAIL / UNKNOWN 状态(2026-05-19 截止)

| # | Module | Bench | Production | 集成 check | 备注 |
|---:|---|:-:|:-:|:-:|---|
| 0 | Perception | ✅ 100% | ✅ stable | ⏳ 待加 trace.n_objects 字段 | deterministic,信心高 |
| 1 | Goal Generation | ❌ 无 bench | ⏳ 等用户标 | ❌ T-DIALECT 没建 | 是 0/5 wins 主嫌疑 |
| 2 | Goal Recognition (parser) | ✅ 83% | ⏳ 没 wins 没法测 | ⏳ 待写自动 check | 自动 check 简单可加 |
| 3 | Reflection Loop | ❌ 无 bench | ⏳ 等用户标 | ⏳ 需要 trace.force_reject_event 字段 | |
| 4 | Action Selection | ✅ 71-97% | ❌ 0/794 directional | ❌ T-NAV-1 production-probe 没建 | 受 Module 1 阻塞 |
| 5 | force_cot Prompt | ✅ +30pp | ❌ 51% | ⏳ 干净环境 retest 待写 | 49% pure LLM disconnect |
| 6 | K=3 Candidates | ✅ +75pp | ⚠️ 22% K<3 | ❌ candidates dump 待写 | partial PASS |

---

## 5. 优先级跟进

| P | 行动 | 修哪个 module |
|---|---|---|
| **P0** | 用户标 `annotation_request.md` | 1, 3 PASS rate |
| **P0** | 加 `--validate-hypothesis-schema strict` 重跑 ar25 | 1 (强制 directional) |
| **P0** | 固定 action_proposer letter mapping (A→ACTION1 永远) | 5 (消 letter confusion) |
| **P0** | 在 trace.jsonl 加 `n_objects`, `force_reject_event`, `candidates_dump` 字段 | 0/3/6 集成 check 基础 |
| P1 | 写 `scripts/v4_parser_consistency_check.py` | 2 集成 check |
| P1 | 写 `scripts/v4_action_consistency_check.py` (T-NAV-1 production-probe) | 4 集成 check |
| P1 | 建 T-DIALECT bench (Reflection 自由 dialect 分布) | 1 bench |
| P1 | 建 T-REJECT bench | 3 bench |
| P1 | 建 T-MAPPING bench (动态 letter shuffle) | 5 bench |
| P2 | A\* 寻路接管 Module 4 navigation | 4 (production-bypass LLM) |

---

## 6. 引用

- 顶层版: `docs/README.md` §2
- Phase 4 实测重审: `docs/project/2026-05-19-v0-v4_clean_baseline/module_validation.md`
- 人工标注请求: `docs/project/2026-05-19-v0-v4_clean_baseline/annotation_request.md`
- 上游设计: `docs/project/2026-05-11-v3-baseline/`, `docs/project/2026-05-14-v3_2-double_agent/`
- 全局词汇: `docs/GLOSSARY_zh.md`
