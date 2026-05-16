<!-- COPIED from archive/outputs_2026-05-14/v3_eval_full/comparison.md; paths rewritten to point back to archive/outputs_2026-05-14/v3_eval_full/ -->
<!-- DO NOT EDIT; edit the original to refresh this copy. -->

# v3 TextAgent vs v1 ablation — G_base × 80 steps

v3 run: `outputs\v3_eval_full`
v1 baselines: `outputs/ablation_overnight_/`

## Aggregate (mean over 5 games)

| Agent | n games | action entropy | uniq actions | no-op rate | max levels | mean RHAE | s/step |
|---|---|---:|---:|---:|---:|---:|---:|
| R0 random | 5 | 1.926 | 7.4 | 17.8% | 0 | 0.000 | 0.01s |
| A1 lite | 5 | 0.273 | 2.6 | 19.5% | 0 | 0.000 | 0.59s |
| A2 full | 5 | 0.261 | 2.0 | 60.0% | 0 | 0.000 | 43.23s |
| A3 reflect | 5 | 0.588 | 3.4 | 34.2% | 0 | 0.000 | 16.95s |
| A4 reflect+m | 5 | 0.465 | 3.0 | 30.8% | 0 | 0.000 | 16.01s |
| **v3 TextAgent** | 5 | 0.984 | 6.0 | 31.0% | 0 | 0.000 | 0.84s |

## Per-game breakdown (v3 only)

| Game | entropy | uniq actions | no-op | levels | RHAE |
|---|---:|---:|---:|---:|---:|
| ar25 | 0.836 | 7 | 43.8% | 0 | 0.000 |
| bp35 | 1.164 | 5 | 5.0% | 0 | 0.000 |
| cd82 | 0.802 | 6 | 35.0% | 0 | 0.000 |
| cn04 | 1.096 | 7 | 30.0% | 0 | 0.000 |
| dc22 | 1.019 | 5 | 41.2% | 0 | 0.000 |

## Decision gate verdict

- ⚠️ action_entropy gate (≥ 1.5) **NOT MET** (got 0.984) — exploration still has bias
- ❌ levels_completed gate **NOT MET** — 80 actions not enough OR strategy needed

- v3 entropy vs best v1 LLM baseline (A3 reflect, 0.588): **+0.396** (+67%)
- v3 entropy vs random (1.926): 51% of uniform-action ceiling

Random has max entropy by definition (uniform sampling). The actionable comparison is **v3 vs LLM-based v1 agents**, where v3 is the clear winner.