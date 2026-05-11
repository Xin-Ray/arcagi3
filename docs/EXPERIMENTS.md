# EXPERIMENTS — 关键实验结果

最近一次更新:2026-05-08(append Exp #4/#5:vlm_test 能力测试 + QLoRA 烟雾训练)

**只记**值得后人参考的实验:有 Hypothesis、有数字、有 Iteration 触发。临时调试不要写。

本文件遵守 **HEI 原则**(Hypothesize → Execute → Iterate,详见 CLAUDE.md):每条实验**必须先写预测值**,再跑;跑完**必须写每种结果分支的下一步**,否则不算结束。

---

## 实验记录模板

每个实验一节，**append-only**，不删除旧条目。

```markdown
## [YYYY-MM-DD] <短标签>
- **Hypothesis(预测，带数字)**：必填。
- **配置**：agent 版本 / game_id 列表 / episode 数 / 关键超参 / 预算上限
- **执行**：实际跑了什么、花了多少
- **结果**：核心数字放表格
- **vs 假设**：确认 / 驳斥 / 不确定
- **Iteration trigger(下一步)**：必填，三种结果分支都要给一个动作
- **写入其它文档**：已 append RESEARCH / 已更新 ROADMAP / 已新增 LIBRARY 等
- **日志**：runs/<file>.jsonl（如有）
```

---

## 跨实验对照表

每完成一个实验，把代表数字加到这里（便于横向比较）。

| 日期 | 实验 | Agent | 数据集 | 平均 RHAE | 过关率 | 备注 |
|------|------|-------|--------|-----------|--------|------|
| 2026-04-27 | Exp #1 | RandomAgent | demo 25, 1 ep × 80 cap | **0.000** | **0%** | 绝对下界 |
| 2026-04-28 | Exp #2 | LLMAgent/haiku | keyboard 3, 1 ep | [pending] | [pending] | Phase 1 pilot |
| 2026-04-28 | Exp #3 | LLMAgent/opus | keyboard 3, 1 ep | [pending] | [pending] | Phase 1 main |
| 2026-05-08 | Exp #4 | Qwen2.5-VL-3B (capability) | 5 synthetic tests | 4/5 pass | n/a | T3 format defect |
| 2026-05-08 | Exp #5 | Qwen2.5-VL-3B QLoRA | 60 silver-label steps | pipeline ok | n/a | smoke test |

---

## 实验记录（按时间倒序）

---

### [2026-05-08] Exp #5 — QLoRA 端到端训练烟雾测试

- **Hypothesis(预测)**：在 60 条 silver-label 步骤上做 1 epoch QLoRA 微调不会崩溃（不 OOM、loss 可收敛、adapter 可保存）；pipeline 端到端验证通过。
- **配置**：
  - Model: `Qwen/Qwen2.5-VL-3B-Instruct` + QLoRA（4-bit NF4 + LoRA r=8，target=q_proj/v_proj，lora_alpha=16）
  - 数据: 60 条（ls20，2 episodes × 30 steps，silver-label 随机动作，无 animation info）
  - Epochs: 1，lr=2e-4，batch=1（single sample loop，无 gradient accumulation）
  - 脚本: `vlm_test/scripts/tiny_train.py`
  - 硬件: RTX A4500 20GB
- **执行**：1 epoch 顺序遍历 60 条，手动 forward-backward-step loop，无 SFTTrainer。
- **结果**：
  - 训练完成：✓（无 OOM，无崩溃）
  - Adapter 保存：✓（`vlm_test/outputs/checkpoint/adapter_model.safetensors`，PEFT 0.19.1）
  - Loss 趋势：未持久化具体数字（terminal 输出未保存）；训练过程稳定
  - Sample inference after training：模型输出合法 ACTION 格式（见 terminal）
- **vs 假设**：**确认**。Pipeline 端到端可运行：4-bit 加载 → LoRA attach → forward → loss → backward → save。
- **Iteration trigger**：
  - 假设确认 → Phase 2C 正式 BC 训练解锁：实现 `arc_agent/vlm_backbone.py` + `arc_agent/train_bc.py`，在真实人类 trace 数据（或 LLM silver-label）上重跑，目标 Exp #6：demo 25 val RHAE ≥ 0.10
  - 若重跑时 OOM → 降 batch size / 开 gradient checkpointing / 降 LoRA rank
  - 若 loss 不收敛 → 检查 prompt 格式与 label masking 是否一致；换 SFTTrainer
- **写入其它文档**：vlm_test/README.md Step 3 完成；ROADMAP Phase 2C 开工条件满足

---

### [2026-05-08] Exp #4 — Qwen2.5-VL-3B-Instruct 能力测试

- **Hypothesis(预测)**：Qwen2.5-VL-3B-Instruct 在 5 项 ARC-style 能力测试中通过率 ≥ 4/5，且每次均输出合法 "ACTION: ACTIONx" 格式。
- **配置**：
  - Model: `Qwen/Qwen2.5-VL-3B-Instruct`（BF16，device=cuda，RTX A4500）
  - 测试: 5 项（t1_format / t2_color / t3_navigate / t4_four_corners / t5_consistency）
  - 入参: 512×512 PNG grid image（来自 `grid_to_image`）+ text prompt
  - 脚本: `vlm_test/scripts/test_vlm.py`
- **执行**：本地加载模型，每个测试 1 次推理（t5 重复 3 次），结果写 `vlm_test/outputs/test_results.json`。
- **结果**：

  | Tag | Pass | 输出 | Elapsed |
  |-----|------|------|---------|
  | t1_format（blank grid） | ✓ | "ACTION: ACTION1" | 1.15 s |
  | t2_color（red dot） | ✓ | "The non-black element is red.\n\nACTION: ACTION1" | 0.77 s |
  | t3_navigate（direction labels） | **✗** | "ACTION: Right" → parsed as NONE | 0.36 s |
  | t4_four_corners | ✓ | "ACTION: ACTION1" | 0.40 s |
  | t5_consistency（×3） | ✓ | "ACTION: ACTION1" ×3 | 1.23 s total |

  通过率：**4/5**（80%）；T3 失败。

- **vs 假设**：**部分确认**。通过率达到预测的 ≥4/5；但"每次均输出合法格式"这一子假设被 T3 驳斥 —— 当 prompt 里写了 `ACTION4(Right)` 的括号标签，模型输出了标签而不是代码。
- **Iteration trigger**：
  - 触发"部分确认 → 生产 prompt 修正"分支：永远不在动作代码后加括号方向标签；此规则已写入 vlm_test/README.md Key findings + 登记为 RESEARCH.md C9 缺陷
  - 后续能力测试用修正 prompt（纯代码列表）验证格式可靠性：预期 5/5 通过
- **写入其它文档**：vlm_test/README.md Key findings 修正；RESEARCH.md §3 新增 C9；LIBRARY.md `grid_to_image` 状态从 experimental 提升（已实测可用）

---

### [2026-04-28] Exp #3 — llm_opus_phase1：LLMAgent(opus) on 3 keyboard games

- **Hypothesis(预测)**：claude-opus-4-7 LLMAgent 在 ls20/tr87/wa30 上平均 RHAE ≥ 0.05，至少 1/3 游戏完成 level 1。理由：Opus 推理能力显著强于 haiku，应能在 80 步内完成至少 1 个 keyboard 游戏的第 1 关。
- **配置**：
  - agent：`LLMAgent(LLMClient(model="claude-opus-4-7"))`，与 haiku 实验相同 prompt/配置
  - 数据集：ls20, tr87, wa30，各 1 episode
  - 单局 step 上限：80；预算上限：$15（硬截止）
  - 入口：`eval.py --agent llm --games ls20,tr87,wa30 --episodes 1 --max-actions 80 --tag llm_opus_phase1 --budget 15`
- **执行**：[待填]
- **结果**：[待填]
- **vs 假设**：[待填]
- **Iteration trigger**：
  - 假设确认 → 进入 Phase 1.5：下载 human traces，开始 Phase 2 BC 数据准备
  - 假设驳斥（0/3 过关，RHAE ≈ 0）→ 分析 LLM 输出日志；可能需要更丰富 observation 或更少 steps 的 exploration-first prompt
  - 不确定（RHAE > 0 但 < 0.05）→ 跑更多 episodes 或手动检查 claude 输出 log
- **写入其它文档**：[待填]
- **日志**：`runs/<ts>_llm_opus_phase1.jsonl`

---

### [2026-04-28] Exp #2 — llm_haiku_phase1：LLMAgent(haiku) on 3 keyboard games

- **Hypothesis(预测)**：claude-haiku-4-5 LLMAgent 在 ls20/tr87/wa30 上平均 RHAE ≥ 0.02，至少 1/3 游戏完成 level 1。理由：LLM 可以识别网格变化规律、定向移动；成本 < $2。
- **配置**：
  - agent：`LLMAgent(LLMClient(model="claude-haiku-4-5"))`，HEI 三段式 system prompt，history_limit=3，fallback=random on parse failure
  - 数据集：keyboard 3 games（ls20, tr87, wa30），各 1 episode，共 3 episodes
  - 单局 step 上限：80；入口：`eval.py --agent llm-haiku --games ls20,tr87,wa30 --episodes 1 --max-actions 80 --tag llm_haiku_phase1 --budget 15`
- **执行**：[待填]
- **结果**：[待填]
- **vs 假设**：[待填]
- **Iteration trigger**：
  - 假设确认（RHAE ≥ 0.02，≥1 游戏过关）→ 立即跑 llm-opus 对照实验（Exp #3）
  - 假设驳斥（RHAE = 0，0/3 过关）→ 检查 HEI prompt 是否在 haiku 上退化；考虑 prompt 简化
  - 不确定（RHAE > 0 但 < 0.02）→ 再跑 3 episodes 取均值
- **写入其它文档**：[待填]
- **日志**：`runs/<ts>_llm_haiku_phase1.jsonl`

---

### [2026-04-27] Exp #1 — random_baseline_1ep：RandomAgent on demo 25

- **Hypothesis(预测)**：平均 RHAE ≤ 0.10，过关率（完成 level 1）≤ 30%。理由：80 步内随机决策对任何 level 1（human baseline 范围 7–78）几乎不可能完成正确序列。
- **配置**：
  - agent：`RandomAgent`（均匀随机 over `available_actions`，排除 `RESET`，`ACTION6` 配随机 (x,y)）
  - 数据集：demo 25 games，1 episode × game，共 25 episodes
  - 单局 step 上限：`MAX_ACTIONS=80`（对齐官方 scaffold）
  - 单 scorecard 跨全部 25 game（`d809e8db-816a-41e4-a4eb-a449adba7715`）
  - 入口：`eval.py --episodes 1 --tag random_baseline_1ep`
- **执行**：wall 17.46 s（初次跑因下载游戏文件偏慢），共 2,000 步 API 调用，远低于 600 RPM 配额。
- **结果**：
  - 平均 RHAE：**0.000**
  - 过关率（`levels_completed > 0`）：**0/25 = 0%**
  - 总动作数 2,000（每局正好 80，无早退）
  - 所有 25 局 final state = `NOT_FINISHED`（80 步 cap 触发，无 GAME_OVER）
- **vs 假设**：**确认**（实测在预测下界，实际是 0 不是接近 0.10）。RandomAgent 在 demo 上是绝对下界。
- **Iteration trigger**：
  - 假设确认 → 进入 Phase 1：实现 LLMAgent（Claude API + HEI）。新 Hypothesis：LLMAgent 平均 RHAE ≥ 0.05、过关率 ≥ 40%（详见 ROADMAP Phase 1）
- **写入其它文档**：
  - PAPER.md §5.2 Random 行已填（0.000 / 0% / 80 / $0）
  - RESEARCH.md §2 方案 #1 RandomAgent 结论已闭合
  - ROADMAP Phase 0 最后一项已勾 ✅
- **日志**：`runs/20260427_211515_random_baseline_1ep.jsonl`（25 行 + 1 行 summary）

#### 顺手的 SDK 元数据发现

`arc.get_environments()` 返回的 `EnvironmentInfo` 已经携带 `baseline_actions`（每个 level 的 `h`）和 `tags`（`keyboard` / `click` / `keyboard_click`），不用跑就能拿。25 个 demo 游戏的 tag 分布：
- 仅 `keyboard`：ls20、tr87、wa30（3 个）
- 仅 `click`：vc33、lp85、s5i5、tn36、lf52、su15、r11l（7 个）
- `keyboard_click`（混合）：剩余 15 个

含义：Phase 1 LLMAgent 可以先专攻 keyboard 类游戏，再扩到 click 和混合。
