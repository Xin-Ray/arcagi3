"""Frame-change predictor (v0). Per docs/arch_predictor_v0_zh.md.

Three things live here:

  dataset.py:   Build (state, action, label) tuples from existing trace.jsonl
                files. Self-supervised - no human annotation. Handles both the
                new v3.2 schema (`action` + `frame_changed`) and the older
                v3 / ablation schema (`chosen_action` + `real_diff`).

  features.py:  Hand-engineered feature extractor (F1, ~32 dims).
                Inputs: game_id, action name, optional coords, optional grid.
                Output: numpy float32 vector. Stateless, easy to mock.

  inference.py: At inference time, given a current state + the set of legal
                actions, return `dict[action, float]` of P(frame_change).
                Models load lazily; absent model = no-op pass-through.

  models.py:    Three small torch models (LogReg, MLP-128, CNN-small) sharing
                a `BasePredictor` interface. All output sigmoid logits.

The training script lives in `scripts/train_predictor.py`.
"""
