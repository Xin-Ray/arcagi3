# vlm_test — Qwen2.5-VL-3B-Instruct Capability & Training Smoke Test

Self-contained test project to verify the VLM works for ARC-AGI-3 before
committing to full training.

## Folder layout

```
vlm_test/
├── README.md               this file
├── data/
│   ├── inputs/             saved PNG images + prompt .txt files (one per test step)
│   └── train/              tiny training dataset (JSONL + PNG images)
│       ├── dataset.jsonl   one record per step: {image_path, prompt, action}
│       └── images/         PNG files referenced by dataset.jsonl
├── outputs/
│   ├── test_results.json   inference outputs from test_vlm.py
│   └── checkpoint/         LoRA weights from tiny_train.py (if training ran)
└── scripts/
    ├── collect_data.py     step 1 — run SDK games, save (image, prompt, action) triples
    ├── test_vlm.py         step 2 — load model, run capability tests, save outputs
    └── tiny_train.py       step 3 — QLoRA fine-tune on the tiny dataset (smoke test)
```

## How to run (in order)

### Step 1 — collect training data from the SDK

Runs a few live game episodes via the ARC API, records every step as an image + prompt + action label.
Requires a valid `ARC_API_KEY` in `.env` at the repo root.

```bash
.venv\Scripts\python.exe vlm_test\scripts\collect_data.py
```

Output: `vlm_test/data/train/dataset.jsonl` + PNG images in `vlm_test/data/train/images/`

### Step 2 — capability test (saves inputs + outputs)

Loads Qwen2.5-VL-3B-Instruct, runs 5 capability tests, saves every input image/prompt
and all model outputs to disk.

```bash
.venv\Scripts\python.exe vlm_test\scripts\test_vlm.py
```

Output:
- `vlm_test/data/inputs/t<N>_image.png`      — the PNG fed to the model
- `vlm_test/data/inputs/t<N>_prompt.txt`     — the text prompt
- `vlm_test/outputs/test_results.json`        — model responses + pass/fail

### Step 3 — tiny QLoRA training smoke test

Loads the dataset from Step 1, runs 1 epoch of QLoRA fine-tuning, saves the LoRA adapter.
This verifies the full training pipeline works before committing GPU-hours.

```bash
.venv\Scripts\python.exe vlm_test\scripts\tiny_train.py
```

Output: `vlm_test/outputs/checkpoint/` (LoRA adapter weights)

## What the model sees (input format)

```
[system]
You are an AI agent playing a turn-based puzzle game.
Each color represents a different element. No instructions are given —
you must infer the rules by observing how your actions change the grid.
...

[user]
<image: 512x512 PNG of the current 64x64 game grid>
State: NOT_FINISHED | Level: 1/3
Available: ACTION1 ACTION2 ACTION3 ACTION4 ACTION5
Last actions: ACTION3, ACTION1
Hypothesis: ...
What is your next action? Reply with: ACTION: <name>

[assistant — target during training]
ACTION: ACTION4
```

## Data format (dataset.jsonl)

Each line is one training step:
```json
{"step": 0, "game_id": "ls20", "level": 1, "state": "NOT_FINISHED",
 "image_path": "images/ls20_ep0_step000.png",
 "prompt": "State: NOT_FINISHED | Level: 1/3 ...",
 "action": "ACTION3", "action_data": null}
```

## Key findings after running

See `vlm_test/outputs/test_results.json` for measured results.

Known limitations found during testing (2026-05-08):
- Color identification: model skips description and goes straight to action (fine for our use — we only need the action)
- Format compliance: reliable, always outputs "ACTION: ACTIONx"
- Consistency: identical outputs across 3 runs with same input
- Inference speed: ~0.4s/step on RTX A4500
