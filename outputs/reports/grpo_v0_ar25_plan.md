# GRPO v0 ar25 — single-game training plan

**Status**: design only; training not started (16-20 h GPU + 3-day dev work; gated on predictor v0 + mask verification finishing first).

Full design: [`docs/arch_grpo_v0_zh.md`](../../docs/arch_grpo_v0_zh.md)

## What this report will contain when training is done

Layout reserved here so the visualization script (`scripts/plot_grpo.py`, TBD) has a known target. Each `.png` will be ~120 dpi, 800-1200 px wide.

1. **`train_curves.png`** — 4 subplots: loss / mean reward / KL-to-ref / entropy vs step. Should see loss ↓, reward ↑, KL bounded < 0.5, entropy slowly ↓ without collapse.
2. **`val_metrics.png`** — 3 subplots: pass_rate @10ep / mean_change_rate / mean_unique_states vs rollout idx. Baseline = pre-training values.
3. **`action_distribution_evolution.png`** — 3 bar charts side-by-side: action distribution at rollouts 0, 500, 1000. Watch for mode collapse (all picks → one action).
4. **`reward_decomposition.png`** — Stacked area chart: contribution of each reward term (R_WIN, R_F1, R_PARSE_FAIL, R_ILLEGAL, R_ENTITY) over time.
5. **`a_b_comparison.png`** — Bar chart: baseline vs trained-LoRA pass_rate on 5 demo games. Only ar25 was trained; bp35/cd82/cn04/dc22 measure transfer.
6. **`failure_episode.gif`** ×3 — Three failed post-training episodes spliced into a 256×256 GIF each so we can eyeball where the LoRA still breaks.

## Decision gates

| Gate | Condition | Action |
|---|---|---|
| **G1** | Loss drops ≥ 30 % in first 200 steps; mean reward > 0 | Continue |
| **G2** | Trained val pass_rate ≥ 30 % on ar25 | LLM-RL path viable for *some* games |
| **G3** | Trained LoRA pass_rate ≥ 5 % on untrained game | Some generalisation |

If **G1 fails** → tune GRPO hyperparams (lr, KL β, group size); 1 retry.
If **G2 fails** → LLM-RL ceiling on ar25 is < 30 %; pivot to non-LLM path (StochasticGoose CNN+RL or Symbolica orchestrator with large model).

## Phase 0 (no GPU; can start any time)

1. `arc_agent/rollout_wrapper.py` — wrap v3.2 ActionAgent for `trl` GRPO API.
2. `arc_agent/prompts_v3_grpo.py` — JSON prompt that forces `{"action": "...", "predicted_changes": [...]}` so the F1 reward has signal.
3. `scripts/run_grpo.py` — finish the GRPO main loop (currently `--dry-run` only).
4. `scripts/plot_grpo.py` — read `train_log.jsonl` + `val_*.json` → emit the 6 PNGs above.
5. `tests/test_train_grpo_full.py` — mock env, 10 GRPO steps, assert loss ↓.

Estimated 3 work days for Phase 0. After that 16-20 h background training launches.

## Why not start training tonight

Three blockers:

- The predictor v0 report (`outputs/reports/predictor_v0.md`) shows hand features can't discriminate per-action well. Before committing 16 h of GPU to GRPO we want to know if the issue is the model's decision policy (GRPO fixes that) or the model's perception (GRPO doesn't). Predictor v0 doesn't fully isolate this — the next experiment is to feed reward signal back into v3.2 prompt and see if the LLM responds.
- Phase 0 plumbing (rollout_wrapper / JSON prompts) is ~3 days of work; not in tonight's scope.
- `trl` 0.x has had GRPO API churn; verifying it works with the current install is its own task.

## Re-engagement criteria

Start GRPO Phase 0 implementation when:

- v3.2 + mask + click_targets bandit verified change_rate ≥ 30 % on ar25 (current mask run will tell us).
- Predictor v0 either deprecated (current path) or extended to M4 CNN with raw grid.
- The user explicitly authorises the GPU budget for 1-day training.

---

*Doc skeleton 2026-05-16. Fill in when training runs.*
