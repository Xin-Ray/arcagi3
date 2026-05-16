# Predictor v0 — frame-change predictor pilot (2026-05-16)

## TL;DR

- **G1 PASS**:val AUC 0.802 (LogReg) / 0.836 (MLP-128) / 0.838 (MLP-512). Above 0.65 floor.
- **G2 MARGINAL**:MLP gives +3.4 pp over LogReg (target +5 pp). Hand features close to saturated.
- **Honest caveat**:**~13 pp of the AUC comes from `game_id` alone**. Ablation without game_id drops AUC from 0.80 → 0.67. The predictor is mostly learning a per-game prior, not state-conditioned action discrimination.
- **Per-action val AUC weak**:ACTION1=0.46 (worse than chance), ACTION4=0.61, ACTION7=0.58. ACTION3=0.91 is the only strong one. ACTION6 val too degenerate to score.
- **Recommendation**:**Do NOT integrate `[ACTION FORECAST]` into the v3.2 prompt yet**. Cross-game prior is useful but the model's per-action ranking is too noisy for the LLM to trust. Next step: CNN on raw grid OR collect targeted "discriminative" episodes.

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

| Model | Train AUC | Val AUC | Val acc | Val ECE |
|---|---:|---:|---:|---:|
| LogReg | 0.946 | **0.802** | 0.790 | 0.141 |
| MLP-S  | 0.971 | **0.836** | 0.771 | 0.113 |
| MLP-L  | 0.974 | **0.838** | 0.771 | 0.124 |

Train/val gap is 14-15 pp → mild overfit but not catastrophic on n=2679. ECE (calibration) is poor (0.11-0.14) — predictions are over-confident in val regime.

Plots in `outputs/predictor_v0/`:

- `train_curves.png` — train + val BCE loss vs epoch
- `roc.png` — val ROC for all 3 models
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

## 5. What didn't pass

| Gate | Target | Result | Status |
|---|---|---|---|
| G1 (signal exists) | val AUC ≥ 0.65 | 0.80 | ✅ |
| G2 (architecture matters) | MLP +5 pp over LogReg | +3.4 pp | ⚠️ |
| Per-action AUC ≥ 0.65 for all | each ≥ 0.65 | ACTION1 / ACTION7 fail | ❌ |
| Calibration ECE ≤ 0.10 | ≤ 0.10 | 0.11-0.14 | ❌ |

## 6. Honest decision: do NOT ship

Adding a `[ACTION FORECAST]` block right now would be a **net negative** because:

1. Most of the AUC signal is per-game prior, which `Knowledge.action_semantics` already covers and the LLM already sees.
2. The novel in-state signal (no_op_streak / state_revisit / direction) is already in the prompt as explicit text, not as a probability.
3. ACTION1's worse-than-random AUC means the LLM, if it trusts the predictor, would actually de-prioritise ACTION1 in dc22 by a wrong amount.
4. Calibration is poor — 0.85 probabilities mean ~70 % actual frequency. LLM would over-weight the top-1.

## 7. What would unlock the next step

In rough priority order:

1. **More games in train**:re-run RandomAgent on 10+ demo games to get cross-game diversity in val. Currently ar25 / bp35 / cd82 / cn04 are all in train, dc22 alone in val. With 10 games we can hold 3 out and get a real OOD AUC number.
2. **Raw-grid CNN (M4)**:hand features can't represent "is the active object next to a wall" — which is what determines whether ACTION1 will work in this specific frame of ar25. A 64×64 CNN with action conditioning might break the 0.84 ceiling.
3. **Targeted dataset rebalancing**:instead of overall 50/50, balance per-(game, action) so we have at least 30 (changed=0, changed=1) pairs in each cell of the 5×7 game×action grid. Currently several cells have 0 negatives.
4. **Different output**:rather than predicting `frame_changed` (single bit), predict `primary_direction` or `primary_distance` (richer signal — encodes both whether and how it changed).
5. **Calibration fix**:Platt scaling or isotonic on a held-out split to bring ECE under 0.05.

## 8. Files produced

```
outputs/predictor_v0/
├── metrics.json               # full numerical results
├── train_curves.png           # train+val loss vs epoch (3 models)
├── roc.png                    # val ROC for all 3 models
├── per_action_auc.png         # per-action AUC bar chart
├── logreg.pt
├── mlp_s.pt
└── mlp_l.pt

outputs/predictor_v0_no_game/  # ablation
├── metrics.json
├── ...
```

Source: `arc_agent/predictor/{dataset.py, features.py, models.py, features_no_game.py}` + `scripts/train_predictor.py`.

## 9. Re-run command

```powershell
.venv\Scripts\python.exe scripts\train_predictor.py `
    --output outputs\predictor_v0 `
    --models logreg,mlp_s,mlp_l `
    --max-epochs 30

# Ablation:
.venv\Scripts\python.exe scripts\train_predictor.py `
    --output outputs\predictor_v0_no_game `
    --models logreg,mlp_s `
    --max-epochs 30 `
    --no-game-feature
```

---

*Doc generated 2026-05-16. Decision: park predictor v0; if state-level signal beyond `game_id` is needed, jump to M4 CNN with raw grid, not deeper MLP on hand features.*
