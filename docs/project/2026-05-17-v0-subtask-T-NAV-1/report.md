# 2026-05-17 v0 subtask T-NAV-1 — Verification Report

> **作用**: 验证 SmolLM3-3B CoT 在 "单步方向选择" 子任务上的准确率。
> **状态**: 完成,FAIL (acc = 48.0%, 远低于目标 ≥ 90%)。
> **决策**: T-NAV-1 不通过 → 不能进入 T-NAV-2 / T-NAV-3 链条的"假定 T-NAV-1 OK"前提;需先修复 CoT 注入逻辑。

## 0. 原始数据 (outputs/ 引用)

**v0 (无 /think 注入,baseline)**

| 文件 | 说明 |
|---|---|
| `outputs/subtask_T-NAV-1_20260517-161834/metrics.json` | 汇总: 48/100, total 511.4s |
| `outputs/subtask_T-NAV-1_20260517-161834/per_probe_smollm3-cot.jsonl` | 100 条 per-probe |
| `outputs/bench_subtask_T-NAV-1_v2.log` | v0 运行日志 |

**v1 (/think 注入 + 1024 tokens,batch 第 1 个 subtask)**

| 文件 | 说明 |
|---|---|
| `outputs/subtask_T-NAV-1_20260517-163018/metrics.json` | 汇总: 52/100, total 555.3s |
| `outputs/subtask_T-NAV-1_20260517-163018/per_probe_smollm3-cot.jsonl` | 100 条 per-probe |
| `outputs/subtask_batch_20260517-163018/{metrics.json,summary.md}` | 跨 5 个 subtask 汇总 |
| `outputs/bench_subtask_batch_v1_think.log` | batch 运行日志 |

## 1. 任务定义

来源: `docs/project/2026-05-17-v0-subtask_decomp/architecture.md` T-NAV-1。

**Probe 形态** (`arc_agent/subtask_probes/__init__.py:gen_T_NAV_1`):
- 给定 object 当前位置 `(r1, c1)` 和 target 位置 `(r2, c2)`,target 与 object 共行或共列,delta ∈ ±[1..6]。
- 4 选 1 多选: 一个正确方向 (ACTION1=UP / ACTION2=DOWN / ACTION3=LEFT / ACTION4=RIGHT),三个干扰。
- n = 100 probes,seed = 42。

**目标**: ≥ 90% (单步、无歧义,模型必须能稳定通过才能进入更复杂的多步推理子任务)。

## 2. 实验配置

| 项 | 值 |
|---|---|
| 模型 | SmolLM3-3B (`HuggingFaceTB/SmolLM3-3B`) |
| reasoning_mode | `cot` (in registry `scripts/bench_subtask.py:30-49`) |
| 量化 | 4-bit nf4, bf16 compute dtype |
| max_new_tokens | 512 |
| sampling | greedy (do_sample=False) |
| prompt 模板 | system + user multi-choice,要求 `Answer: X` 单行 (`scripts/bench_subtask.py:52-83`) |
| GPU | RTX (单卡,与另一 ar25 run 竞争) |

## 3. 结果

**Final accuracy: 48.0% (48/100)**,total 511.4s,平均每 probe ~5.1s。

进度记录 (来自 `outputs/bench_subtask_T-NAV-1_v2.log`):

```
[10/100]  running_acc=40.0%  last_elapsed=0.7s
[20/100]  running_acc=35.0%  last_elapsed=1.4s
[30/100]  running_acc=30.0%  last_elapsed=0.7s
[40/100]  running_acc=37.5%  last_elapsed=38.6s
[50/100]  running_acc=40.0%  last_elapsed=0.7s
[60/100]  running_acc=46.7%  last_elapsed=38.4s
[70/100]  running_acc=44.3%  last_elapsed=1.4s
[80/100]  running_acc=43.8%  last_elapsed=1.0s
[90/100]  running_acc=47.8%  last_elapsed=30.2s
[100/100] running_acc=48.0%  last_elapsed=1.4s
```

## 4. 关键观察 — CoT chain 被绕过

**只有 11/100 probes 真正激活了 CoT 链** (elapsed > 10s),其余 89/100 是短路径直接给 `Answer: X`。

| 子集 | n | accuracy |
|---|---:|---:|
| CoT activated (elapsed > 10s) | 11 | 54.5% |
| No CoT (elapsed ≤ 10s) | 89 | 47.2% |

→ 即使 CoT 激活,accuracy 也只到 54.5%,远低于 model_bench T1 (`outputs/spatial_bench_combined/.../metrics.json` 100%)。

**根因**: `scripts/bench_subtask.py:121-147 generate()` 只对 `no_think` 模式注入 `/no_think` 标签,**没有对 `cot` 模式显式注入 `/think`**。SmolLM3 的双模式推理需要显式 flag 来稳定开启 CoT 链;默认行为不可控,导致 89% probe 走短路径。

```python
# 当前代码 — 不对称
if "SmolLM3" in entry["hf_id"]:
    want_no_think = rm in ("no_think", "auto")
    if want_no_think and "/no_think" not in sys_text:
        sys_text = (system + "\n/no_think").strip()
```

应改为 (跟进项,不在本子任务范围内):

```python
if "SmolLM3" in entry["hf_id"]:
    if rm == "no_think" and "/no_think" not in sys_text:
        sys_text = (system + "\n/no_think").strip()
    elif rm == "cot" and "/think" not in sys_text:
        sys_text = (system + "\n/think").strip()
```

## 5. 错误分布

来自 `outputs/subtask_T-NAV-1_20260517-161834/per_probe_smollm3-cot.jsonl`。

| 项 | A | B | C | D | (None) |
|---|---:|---:|---:|---:|---:|
| guess 分布 | 9 | 33 | 36 | 17 | 5 |
| correct 分布 | 19 | 29 | 33 | 19 | - |
| 错误数 (按 correct) | 13 | 12 | 13 | 14 | - |

- **A 严重欠选** (9 实际 / 19 应有) — 模型有 letter-bias。
- 错误在四个正确答案上分布近似均匀 → 不是单一方向的认知盲点,而是整体推理能力薄弱。
- 5 个 parse fail (没有 `Answer: X` 模式) — 都计为错。

## 6. 与 model_bench T1 (100%) 的差距来源

| 项 | model_bench T1 | T-NAV-1 |
|---|---|---|
| n_probes | 8 | 100 |
| CoT 实际激活 | (推测) 全部 | 11/100 |
| 题型 | 单步方向 | 单步方向 |
| 报告 acc | 100% | 48.0% |

T-NAV-1 是 T1 的 100× 扩展放大版,样本量大后 CoT 激活率的真实分布暴露出来。

## 7. 通过/失败判定

### v0 (no /think): **FAIL** acc = 48% < target 90%

### v1 (/think injected,2026-05-17 16:30 重跑): **STILL FAIL** acc = 52.0%

| 指标 | v0 (no /think) | v1 (/think + 1024 tokens) |
|---|---:|---:|
| 整体 accuracy | 48.0% | **52.0%** |
| CoT activation 率 | 11/100 | 11/100(没变化!) |
| CoT 激活时 accuracy | 54.5% | **90.9%** ✅ |
| 非激活时 accuracy | 47.2% | 47.2% |
| total wall | 511.4s | 555.3s |

**关键发现**:
- `/think` system flag **没有提升 CoT activation 率**(仍 11/100),仅在已激活情况下让 CoT chain 不被 1024 token 限制截断,把激活 acc 从 54.5% → **90.9%**(只要思考就基本对)。
- 整体 acc 仅 +4pp 是因为大头(89/100)的"短路径不思考"分支没变。
- **真问题**: SmolLM3 在 T-NAV-1 这种"看起来简单"的题上自己决定不启动 `<think>` chain → 拿短路径输出 47% 的猜。

## 8. 跟进项

| 优先级 | 行动 | 验证目标 |
|---|---|---|
| ~~P0~~ ✅ done | 修 `/think` 注入 | 部分缓解 (CoT-when-fired 54.5%→90.9%) |
| P0 | **在 user prompt 前缀加 "Let's think step by step." 强制 CoT** | T-NAV-1 acc 应 ≥ 85%(已经证明 CoT 链 90.9% 准) |
| P1 | 试 `tokenizer.apply_chat_template(messages, enable_thinking=True)` 看是否能强制激活 | 如比 user prompt 干净,优先 |
| P2 | 升级到 SmolLM3-7B 或 Qwen2.5-7B 看 base CoT 激活率 | - |

## 文件清单

```
docs/project/2026-05-17-v0-subtask-T-NAV-1/
├── report.md                                          (本文件)

outputs/subtask_T-NAV-1_20260517-161834/
├── metrics.json
├── per_probe_smollm3-cot.jsonl
└── summary.md

outputs/
└── bench_subtask_T-NAV-1_v2.log

arc_agent/subtask_probes/__init__.py                   (gen_T_NAV_1)
scripts/bench_subtask.py                               (runner,/think 注入缺失点)
tests/test_subtask_probes.py                           (probe sanity tests)
```

## 引用

- 主架构: `docs/project/2026-05-17-v0-subtask_decomp/architecture.md` (7 子任务 DAG)
- model_bench 5×2×300 报告: `docs/project/2026-05-17-v0-model_bench/report_5game.md`
- 全局词汇: `docs/GLOSSARY_zh.md` (subtask probe、CoT、/no_think 条目)
- 仓库总入口: `docs/README.md`
