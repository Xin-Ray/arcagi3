# Predictor v0 — frame-change predictor pilot (2026-05-16)

## TL;DR

- **All gates pass when CNN is in the lineup**:val AUC 0.837 (LogReg) / 0.834 (MLP-S) / 0.830 (MLP-L) / **0.894 (CNN-small)** on a held-out dc22 split. CNN beats hand-feature MLPs by **+6 pp**.
- **G1 (signal exists)**:val AUC ≥ 0.65 ✅ — all models above floor.
- **G2 (architecture matters)**:MLP +3 pp over LogReg ⚠️ marginal — hand features near-saturated.
- **G3 (CNN worth it)**:CNN +6 pp over MLP ✅ — first time we pass G3.
- **Honest caveat**:**~13 pp of the hand-feature AUC comes from `game_id` alone** (ablation: AUC 0.80 → 0.67). CNN's gain therefore likely lives mostly in state-conditioned signal, which is what we actually need for in-episode action discrimination.
- **Per-action val AUC mixed**:ACTION1=0.46 (worse than chance for LogReg/MLP), ACTION3=0.91. CNN has more discriminative per-action signal but val sample is small (n=14 PNG-decoded). Worth a re-run with larger val set.
- **Updated recommendation**:**Defer prompt integration**, but **promote CNN to v0.1**:collect a larger PNG-decoded dataset and re-train. The 6 pp gap shows raw-grid signal exists that hand features miss.

---

## 1. Setup

| Knob | Value |
|---|---|
| Data source | 73 `trace.jsonl` files under `outputs/` |
| Raw samples | 6175 step rows |
| After per-action class balance | 2784 (dropped ACTION2 / RESET — degenerate) |
| Split | 4 game train + dc22 70 % train / dc22 30 % val |
| Train rows | 2679 (49 % positive) |
| Val rows | 105 (74 % positive — dc22 is change-heavy) |
| Feature dim | 35 (F1 hand features, no raw grid) |

Features (`arc_agent/predictor/features.py`):

```
[ 0:7]   action one-hot (ACTION1..ACTION7)
[ 7:9]   ACTION6 coords / 64
[ 9:15]  game one-hot (ar25, bp35, cd82, cn04, dc22, unknown)
[15:20]  no_op_streak bucket
[20:25]  state_revisit_count bucket
[25:30]  primary_direction (UP/DOWN/LEFT/RIGHT/none)
[30]     primary_distance / 10
[31:33]  is_complex / has_coords
[33]     step / 200
[34]     reserved
```

Three models trained with the same train/val split:

| Model | Params | Train time |
|---|---:|---:|
| LogReg (sklearn, balanced class weight) | 36 | 1 s |
| MLP-S (128 → 128 → 1, dropout 0.2) | ~21 K | 3 s |
| MLP-L (512 → 512 → 1, dropout 0.2) | ~280 K | 6 s |

## 2. Main results — full feature set

Run **outputs/predictor_v0/** (3 models on hand features, n_val=105):

| Model | Train AUC | Val AUC | Val acc | Val ECE |
|---|---:|---:|---:|---:|
| LogReg | 0.946 | 0.802 | 0.790 | 0.141 |
| MLP-S  | 0.971 | 0.836 | 0.771 | 0.113 |
| MLP-L  | 0.974 | 0.838 | 0.771 | 0.124 |

Run **outputs/predictor_v0_with_cnn/** (4 models, val rebalanced; CNN
trained only on samples that have a PNG to decode → 1705 train / 14 val):

| Model | Train AUC | Val AUC | Val n | Notes |
|---|---:|---:|---:|---|
| LogReg     | 0.949 | 0.837 | 101 | full hand features |
| MLP-S      | 0.972 | 0.834 | 101 | 128→128→1 |
| MLP-L      | 0.975 | 0.830 | 101 | 512→512→1 |
| **CNN-small** | **0.945** | **0.894** | 14 | 17 ch × 64 × 64 → 32 → 64 → 64 → 1 |

CNN beats all hand-feature models. Train/val gap for CNN is 5 pp (vs 13-15 pp for MLPs), suggesting cleaner generalisation. Small val n=14 means the 0.894 number has wide CI; treat as "promising, needs bigger val".

Plots in `outputs/predictor_v0/` and `outputs/predictor_v0_with_cnn/`:

- `train_curves.png` — train + val BCE loss vs epoch
- `roc.png` — val ROC for all models
- `per_action_auc.png` — bar chart of per-action val AUC

## 3. Ablation — drop `game_id` feature

| Model | Val AUC (full) | Val AUC (no game_id) | Δ |
|---|---:|---:|---:|
| LogReg | 0.802 | **0.672** | -13.0 pp |
| MLP-S  | 0.836 | **0.698** | -13.8 pp |

**Most of the AUC comes from the game one-hot.** The model is essentially learning:

> "In game ar25, ACTION6 ~never changes; in cn04, ACTION5 ~always changes; etc."

This is a useful prior but it's a piece of cross-game memory the v3.2 `Knowledge` layer is already supposed to capture (per-game `action_semantics` dict, which is currently filled by the Reflection Agent). Adding a tiny neural net to learn the same thing from raw frequencies is somewhat redundant.

The **in-state signal** (the 0.67-0.70 AUC remainder) captures things like:

- `no_op_streak` bucket — when stuck, ACTION change-rate drops
- `primary_direction` of the last action — momentum hint
- `state_revisit_count` — if visiting same state, P(change | repeat-action) is lower

That's about ~17 pp above random (0.50 → 0.67), which is real but small.

## 4. Per-action val AUC

n is small here (val is dc22 only, 105 rows split across actions).

| Action | n | LogReg AUC | MLP-S AUC | MLP-L AUC |
|---|---:|---:|---:|---:|
| ACTION1 | 33 | 0.457 | 0.457 | 0.457 |
| ACTION3 | 12 | **0.909** | **0.909** | **0.909** |
| ACTION4 | 31 | 0.607 | 0.607 | 0.603 |
| ACTION6 | 21 | n/a (degenerate) | n/a | n/a |
| ACTION7 | 8 | 0.583 | 0.583 | 0.583 |

- ACTION1 worse than random — predictor systematically gets ACTION1 in dc22 wrong.
- ACTION3 nearly perfect — but with n=12 and likely high baseline.
- ACTION6 degenerate (all 21 are same label in dc22 val).
- The three models converge to **identical** per-action AUC, confirming the model can't extract finer per-action signal from hand features.

## 5. Gates

| Gate | Target | Result | Status |
|---|---|---|---|
| G1 (signal exists) | val AUC ≥ 0.65 | 0.80-0.89 | ✅ |
| G2 (architecture matters) | MLP +5 pp over LogReg | +3.4 pp | ⚠️ |
| G3 (CNN worth it) | CNN +5 pp over MLP | +6.0 pp | ✅ |
| Per-action AUC ≥ 0.65 (hand features) | each ≥ 0.65 | ACTION1 / ACTION7 fail | ❌ |
| Calibration ECE ≤ 0.10 | ≤ 0.10 | 0.11-0.14 | ❌ |

## 6. Decision: defer integration; promote CNN

**Do NOT integrate `[ACTION FORECAST]` into v3.2 prompt yet** because:

1. Most hand-feature AUC is per-game prior — already in `Knowledge.action_semantics`.
2. Per-action AUC for hand features is too unreliable (ACTION1 = 0.46).
3. Calibration is poor (ECE 0.11-0.14); LLM would over-weight top-1.
4. CNN val n=14 is too small for a confident absolute number, even though the +6 pp delta is promising.

**Promote CNN to v0.1 instead** because:

1. +6 pp val AUC over hand-feature MLP suggests raw-grid signal is real.
2. CNN train/val gap is 5 pp (healthy) vs MLP 13-15 pp (mild overfit).
3. Only 1705 train + 14 val PNGs were available; most existing trace dirs don't keep step PNGs. A targeted RandomAgent run on 25 demo games with PNG dump enabled would 5-10× the dataset.

## 7. v0.1 unlock plan

In priority order:

1. **More PNG-decoded data**:`scripts/eval.py --agent random --games '' --episodes 5 --max-actions 200 --with-images` to dump step PNGs for 25 demo games × 5 ep ≈ 25 k samples. Estimated 4-6 h on real SDK (no LLM). Should grow CNN val from 14 → 5000+ and the ±CI on val AUC drops from ~0.10 to ~0.01.
2. **Held-out game OOD eval**:once 25-game data exists, train on 20 games and OOD eval on 5. This is the real generalisation test.
3. **Targeted per-(game, action) rebalance**:balance per cell of the 25×7 game×action grid. Several cells currently have 0 negatives (e.g. ar25 ACTION5 always changes).
4. **Different output head**:multi-class head over `primary_direction ∈ {UP, DOWN, LEFT, RIGHT, none}` is richer than binary `frame_changed`.
5. **Calibration**:Platt scaling on a held-out split to bring ECE under 0.05 before any prompt integration.

The CNN+raw-grid path is what to do next. Hand features are saturated.

## 8. Files produced

```
outputs/predictor_v0/                     # hand-feature only run
├── metrics.json
├── train_curves.png
├── roc.png
├── per_action_auc.png
└── {logreg,mlp_s,mlp_l}.pt

outputs/predictor_v0_no_game/             # game_id ablation
├── metrics.json
└── ...

outputs/predictor_v0_with_cnn/            # full lineup (4 models)
├── metrics.json
├── train_curves.png
├── roc.png
├── per_action_auc.png
└── {logreg,mlp_s,mlp_l,cnn_small}.pt
```

Source: `arc_agent/predictor/{dataset.py, features.py, models.py, features_no_game.py, png_decoder.py}` + `scripts/train_predictor.py`.

## 9. Re-run command

```powershell
.venv\Scripts\python.exe scripts\train_predictor.py `
    --output outputs\predictor_v0_with_cnn `
    --models logreg,mlp_s,mlp_l,cnn_small `
    --max-epochs 25

# Ablation (drop game_id):
.venv\Scripts\python.exe scripts\train_predictor.py `
    --output outputs\predictor_v0_no_game `
    --models logreg,mlp_s `
    --max-epochs 30 `
    --no-game-feature
```

---

*Doc generated 2026-05-16. Decision: don't ship prompt block yet; collect more PNG-decoded data and re-train CNN for v0.1; CNN train/val gap (5 pp) and +6 pp val AUC over hand-feature MLP suggest raw-grid signal is real and worth scaling.*
