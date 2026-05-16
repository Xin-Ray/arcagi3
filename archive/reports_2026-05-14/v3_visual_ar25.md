<!-- COPIED from archive/outputs_2026-05-14/v3_visual_ar25/report.md; paths rewritten to point back to archive/outputs_2026-05-14/v3_visual_ar25/ -->
<!-- DO NOT EDIT; edit the original to refresh this copy. -->

# v3 TextAgent — visual report

Run: `outputs\v3_visual_ar25`. 5 games × 80 steps with **play.gif + per-step trace** for each game.

## Aggregate (mean over 5 games)

| Agent | entropy | uniq actions | no-op | levels | RHAE | s/step |
|---|---:|---:|---:|---:|---:|---:|
| R0 random | 1.926 | 7.4 | 17.8% | 0 | 0.000 | 0.01s |
| A1 lite | 0.273 | 2.6 | 19.5% | 0 | 0.000 | 0.59s |
| A2 full | 0.261 | 2.0 | 60.0% | 0 | 0.000 | 43.23s |
| A3 reflect | 0.588 | 3.4 | 34.2% | 0 | 0.000 | 16.95s |
| A4 reflect+m | 0.465 | 3.0 | 30.8% | 0 | 0.000 | 16.01s |
| **v3 TextAgent** | **0.836** | **7.0** | **43.8%** | 0 | 0.000 | **0.69s** |

## Per-game visual + trace

### ar25 (ar25-0c556536)

- entropy: **0.836**, unique actions: **7/7**, no-op rate: **43.8%**, levels: **0/8**, RHAE: **0.000**

![ar25 play.gif](../../archive/outputs_2026-05-14/v3_visual_ar25/ar25-0c556536/play.gif)

Full step-by-step trace: [ar25-0c556536/trace_view.md](../../archive/outputs_2026-05-14/v3_visual_ar25/ar25-0c556536/trace_view.md)

---
