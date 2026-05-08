# EXPERIMENTS — 关键实验结果

最近一次更新:2026-04-27(append 第一条实验:RandomAgent baseline on demo 25)

**只记**值得后人参考的实验:有 Hypothesis、有数字、有 Iteration 触发。临时调试不要写。

本文件遵守 **HEI 原则**(Hypothesize → Execute → Iterate,详见 CLAUDE.md):每条实验**必须先写预测值**,再跑;跑完**必须写每种结果分支的下一步**,否则不算结束。

---

## 实验记录模板

每个实验一节,**append-only**,不删除旧条目。

```markdown
## [YYYY-MM-DD] <短标签>
- **Hypothesis(预测,带数字)**:必填。"我预期 X 比 Y 高 ≥0.05 RHAE,因为 ...";没有可证伪的预测就不要开跑。
- **配置**:agent 版本 / commit / game_id 列表 / episode 数 / 关键超参 / 预算上限(token、wall-clock)
- **执行**:实际跑了什么、花了多少
- **结果**:核心数字(平均 RHAE、过关率、step 数、API 成本……)放表格
- **vs 假设**:确认 / 驳斥 / 不确定;偏离方向和幅度
- **Iteration trigger(下一步)**:必填,三种结果分支都要给一个动作
  - 假设确认 → 下一步做 ...(具体到要新建的 ROADMAP 任务或下一组实验)
  - 假设驳斥 → 这驳斥了什么底层信念?新候选是什么?
  - 不确定 → 加什么对照实验能区分?
- **写入其它文档**:已 append RESEARCH.md 第 X 条 / 已更新 ROADMAP Phase Y / 已新增 LIBRARY 的 fn_z
- **日志**:`runs/<file>.jsonl`(如有)
```

---

## 跨实验对照表

每完成一个实验,把代表数字加到这里(便于横向比较)。

| 日期 | Agent | 数据集 | 平均 RHAE | 过关率 | 平均 step / level | API 成本 | 备注 |
|------|-------|--------|-----------|--------|--------------------|----------|------|
| 2026-04-27 | RandomAgent | demo 25, 1 ep × 80 step cap | **0.000** | **0% (level 1)** | 80 (capped) | $0(免费 RPM 内) | 绝对下界,符合 Hypothesis(≤0.10 / ≤30%) |
| 2026-04-28 | LLMAgent/haiku (claude-haiku-4-5) | keyboard 3 (ls20,tr87,wa30), 1 ep × 80 step cap | [pending] | [pending] | [pending] | [pending] | Phase 1 pilot run |
| 2026-04-28 | LLMAgent/opus (claude-opus-4-7) | keyboard 3 (ls20,tr87,wa30), 1 ep × 80 step cap | [pending] | [pending] | [pending] | [pending] | Phase 1 main run |

---

## 实验记录(按时间倒序)

### [2026-04-28] llm_haiku_phase1 — LLMAgent(haiku) on 3 keyboard games

- **Hypothesis(预测)**:claude-haiku-4-5 LLMAgent 在 ls20/tr87/wa30 上平均 RHAE ≥ 0.02,至少 1/3 个游戏完成 level 1。理由:LLM 可以识别网格变化规律、定向移动,哪怕 7B-sized haiku 也远超 random。成本估算 < $2。
- **配置**:
  - agent:`LLMAgent(LLMClient(model="claude-haiku-4-5"))`,HEI 三段式 system prompt,history_limit=3,fallback=random on parse failure
  - 数据集:keyboard 3 games(ls20, tr87, wa30),各 1 episode,共 3 episodes
  - 单局 step 上限:80(同官方 scaffold)
  - 入口:`eval.py --agent llm-haiku --games ls20,tr87,wa30 --episodes 1 --max-actions 80 --tag llm_haiku_phase1 --budget 15`
  - 代码版本:2026-04-28,新增 shared LLMClient + cost tracker
- **执行**:[待填] wall time, 实际 step 数, token 统计
- **结果**:
  - 平均 RHAE:[待填]
  - 过关率:[待填]
  - 总 token:[待填] input / [待填] output / [待填] cache_read
  - 估算成本:[待填]
- **vs 假设**:[待填]
- **Iteration trigger**:
  - 假设确认(RHAE ≥ 0.02,≥1 游戏过关) → 立即跑 llm-opus 对照实验(实验 #3),验证 Opus vs Haiku 差距
  - 假设驳斥(RHAE = 0,0/3 过关) → 检查 HEI prompt 是否在 haiku 上退化为乱输出;考虑 prompt 简化或 max_tokens 削减再跑
  - 不确定(RHAE > 0 但 < 0.02) → 再跑 3 episodes 取均值
- **写入其它文档**:[待填]
- **日志**:`runs/<ts>_llm_haiku_phase1.jsonl`

---

### [2026-04-28] llm_opus_phase1 — LLMAgent(opus) on 3 keyboard games

- **Hypothesis(预测)**:claude-opus-4-7 LLMAgent 在 ls20/tr87/wa30 上平均 RHAE ≥ 0.05(ROADMAP Phase 1 原始 Hypothesis),至少 1/3 个游戏完成 level 1。理由:Opus 推理能力显著强于 haiku,应能在 80 步内完成至少 1 个 keyboard 游戏的第 1 关。
- **配置**:
  - agent:`LLMAgent(LLMClient(model="claude-opus-4-7"))`,与 haiku 实验完全相同 prompt/配置
  - 数据集:ls20, tr87, wa30,各 1 episode
  - 单局 step 上限:80
  - 预算上限:$15(硬截止)
  - 入口:`eval.py --agent llm --games ls20,tr87,wa30 --episodes 1 --max-actions 80 --tag llm_opus_phase1 --budget 15`
- **执行**:[待填]
- **结果**:[待填]
- **vs 假设**:[待填]
- **Iteration trigger**:
  - 假设确认 → 进入 Phase 1.5:下载 human traces,开始 Phase 2 BC 数据准备
  - 假设驳斥(0/3 过关,RHAE ≈ 0) → 分析 LLM 输出日志;可能需要更丰富 observation(颜色映射 + 对象检测)或更少 steps 的 exploration-first prompt
  - 不确定(RHAE > 0 但 < 0.05) → 跑更多 episodes 或手动检查 claude 输出 log
- **写入其它文档**:[待填]
- **日志**:`runs/<ts>_llm_opus_phase1.jsonl`

---

### [2026-04-27] random_baseline_1ep — RandomAgent on demo 25

- **Hypothesis(预测)**:平均 RHAE ≤ 0.10,过关率(完成 level 1)≤ 30%。理由:80 步内随机决策对任何 level 1(human baseline 范围 7–78)的概率上随机走对 22+ 步序列 ≈ 0;若高于此,说明 SDK 给了非预期 reward 或 baseline 太宽松。
- **配置**:
  - agent:`agent_starter.RandomAgent`(均匀随机 over `available_actions`,排除 `RESET`,`ACTION6` 配随机 (x,y))
  - 数据集:demo 25 games(SDK `arc.get_environments()` 返回的全部),1 episode × game,共 25 episodes
  - 单局 step 上限:`MAX_ACTIONS=80`(对齐官方 scaffold)
  - 单 scorecard 跨全部 25 game(`d809e8db-816a-41e4-a4eb-a449adba7715`)
  - 入口:`eval.py --episodes 1 --tag random_baseline_1ep`
- **执行**:wall 17.46 s(初次跑因为下游戏文件偏慢;之后会更快),共 2,000 步 API 调用,**远低于 600 RPM 配额**
- **结果**:
  - 平均 RHAE(per-game `score`)= **0.000**
  - 过关率(`levels_completed > 0`)= **0/25 = 0%**
  - 总动作数 2,000(每局正好 80,无早退)
  - 所有 25 局 final state = `NOT_FINISHED`(因 80 步 cap 触发,无 GAME_OVER)
- **vs 假设**:**确认**(实测在预测下界,且实际是 0 不是接近 0.10)。这意味着 RandomAgent 在 demo 上几乎是绝对下界。
- **Iteration trigger**:
  - 假设确认 → **下一步:进入 Phase 1**,实现 LLMAgent(Claude API,prompt-level HEI)。新假设:LLMAgent 平均 RHAE ≥ 0.05、过关率 ≥ 40%(详见 ROADMAP Phase 1)
  - (假设若被驳斥,即 RandomAgent 居然 > 0.10) → 已不适用,不必触发分支
- **写入其它文档**:
  - PAPER.md §5.2 Random 行已填(0.000 / 0% / 80 / $0)
  - RESEARCH.md §尝试 #1 RandomAgent **结果/结论已闭合**
  - ROADMAP Phase 0 最后一项已勾 ✅
- **日志**:`runs/20260427_211515_random_baseline_1ep.jsonl`(25 行 + 1 行 summary)

#### 顺手的 SDK 元数据发现

`arc.get_environments()` 返回的 `EnvironmentInfo` 已经携带 `baseline_actions`(每个 level 的 `h`)和 `tags`(`keyboard` / `click` / `keyboard_click`),**不用跑就能拿**。25 个 demo 游戏的总 level 数 ≈ 180,total `h` 之和 = 我们 RHAE 评估的"理想步数总预算"。
按 tag 切分(对未来设计有用):
- 仅 `keyboard`:ls20、tr87、wa30(3 个)
- 仅 `click`:vc33、lp85、s5i5、tn36、lf52、su15、r11l(7 个)
- `keyboard_click`(混合):剩余 15 个
**含义**:Phase 1 LLMAgent 可以先专攻 keyboard 类游戏(动作空间小),再扩到 click 和混合;这能把"每步候选动作"从 ~4096 减少到 ~5,显著降低 LLM 推理成本。
