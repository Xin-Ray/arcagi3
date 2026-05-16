<!-- COPIED from archive/outputs_2026-05-14/v3_visual_full/report.md; paths rewritten to point back to archive/outputs_2026-05-14/v3_visual_full/ -->
<!-- DO NOT EDIT; edit the original to refresh this copy. -->

# v3 TextAgent — visual report

Run: `outputs\v3_visual_full`. 5 games × 80 steps with **play.gif + per-step trace** for each game.

## Aggregate (mean over 5 games)

| Agent | entropy | uniq actions | no-op | levels | RHAE | s/step |
|---|---:|---:|---:|---:|---:|---:|
| R0 random | 1.926 | 7.4 | 17.8% | 0 | 0.000 | 0.01s |
| A1 lite | 0.273 | 2.6 | 19.5% | 0 | 0.000 | 0.59s |
| A2 full | 0.261 | 2.0 | 60.0% | 0 | 0.000 | 43.23s |
| A3 reflect | 0.588 | 3.4 | 34.2% | 0 | 0.000 | 16.95s |
| A4 reflect+m | 0.465 | 3.0 | 30.8% | 0 | 0.000 | 16.01s |
| **v3 TextAgent** | **0.984** | **6.0** | **31.0%** | 0 | 0.000 | **0.78s** |

## Per-game visual + trace

### ar25 (ar25-0c556536)

- entropy: **0.836**, unique actions: **7/7**, no-op rate: **43.8%**, levels: **0/8**, RHAE: **0.000**

![ar25 play.gif](../../archive/outputs_2026-05-14/v3_visual_full/ar25-0c556536/play.gif)

Full step-by-step trace: [ar25-0c556536/trace_view.md](../../archive/outputs_2026-05-14/v3_visual_full/ar25-0c556536/trace_view.md)

---

### bp35 (bp35-0a0ad940)

- entropy: **1.164**, unique actions: **5/7**, no-op rate: **5.0%**, levels: **0/9**, RHAE: **0.000**

![bp35 play.gif](../../archive/outputs_2026-05-14/v3_visual_full/bp35-0a0ad940/play.gif)

Full step-by-step trace: [bp35-0a0ad940/trace_view.md](../../archive/outputs_2026-05-14/v3_visual_full/bp35-0a0ad940/trace_view.md)

---

### cd82 (cd82-fb555c5d)

- entropy: **0.802**, unique actions: **6/7**, no-op rate: **35.0%**, levels: **0/6**, RHAE: **0.000**

![cd82 play.gif](../../archive/outputs_2026-05-14/v3_visual_full/cd82-fb555c5d/play.gif)

Full step-by-step trace: [cd82-fb555c5d/trace_view.md](../../archive/outputs_2026-05-14/v3_visual_full/cd82-fb555c5d/trace_view.md)

---

### cn04 (cn04-2fe56bfb)

- entropy: **1.096**, unique actions: **7/7**, no-op rate: **30.0%**, levels: **0/6**, RHAE: **0.000**

![cn04 play.gif](../../archive/outputs_2026-05-14/v3_visual_full/cn04-2fe56bfb/play.gif)

Full step-by-step trace: [cn04-2fe56bfb/trace_view.md](../../archive/outputs_2026-05-14/v3_visual_full/cn04-2fe56bfb/trace_view.md)

---

### dc22 (dc22-fdcac232)

- entropy: **1.019**, unique actions: **5/7**, no-op rate: **41.2%**, levels: **0/6**, RHAE: **0.000**

![dc22 play.gif](../../archive/outputs_2026-05-14/v3_visual_full/dc22-fdcac232/play.gif)

Full step-by-step trace: [dc22-fdcac232/trace_view.md](../../archive/outputs_2026-05-14/v3_visual_full/dc22-fdcac232/trace_view.md)

---
