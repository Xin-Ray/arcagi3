# ARC-AGI-3 Agent — ARC Prize 2026

An autonomous agent that plays ARC-AGI-3 turn-based puzzle games with no instructions,
infers object movement rules through a Hypothesize → Execute → Iterate (HEI) loop,
and is fine-tuned on human gameplay traces via QLoRA.

Competition: [ARC Prize 2026 on Kaggle](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)

---

## Current Status (2026-05-08)

| Phase | Status | Result |
|-------|--------|--------|
| Phase 0 — SDK + baseline | Done | RandomAgent mean RHAE 0.000 on demo 25 |
| Phase 1 — Environment | Done | Python 3.12, venv rebuilt, 47 tests passing |
| Phase 2 — VLM test | In progress | Qwen2.5-VL-3B loads, QLoRA pipeline verified |
| Phase 2 — Data pipeline | In progress | Animation analysis implemented and tested |
| Phase 2 — BC training | Pending | Awaiting full human trace dataset |

---

## Architecture

```
Game env (ARC-AGI-3 SDK)
        |
        v  FrameDataRaw (grid frames + state)
        |
   VLMAgent.choose()
        |
        +-- latest_grid()       64x64 numpy array
        |       |
        |       v
        |   grid_to_image()     512x512 RGB PNG  (each cell = 8x8 px, ARC 16-color palette)
        |
        +-- animation_to_text() per-object motion description:
        |       "color 1 (blue): (row~32,col~10) -> (row~32,col~18) [RIGHT 8 cells, 10 frames]"
        |       "color 4 (yellow): APPEARED at row~5 col~50"
        |
        +-- grid_diff()         changed cells since previous step
        |
        v
   Qwen2.5-VL-3B-Instruct  (QLoRA fine-tuned)
        |  image + text prompt
        v
   "ACTION: ACTION4"
        |
        v
   env.step(ACTION4)
```

**Why image input?**
64x64 hex text puts vertically adjacent cells ~64 tokens apart — a text model has no
2D inductive bias. Qwen2.5-VL's ViT vision encoder uses 14x14px patches + 2D-RoPE,
giving direct 2D spatial understanding.

**Why animation path?**
Before/after comparison alone cannot distinguish "slides until wall" from "jumps 8 cells".
`animation_to_text()` tracks each color's centroid across all intermediate frames,
detecting direction, distance, multi-segment paths, and appear/disappear events.

---

## Prompt Structure (per game step)

```
[system]
You are an AI agent playing a turn-based puzzle game. You see the current game
grid as an image. The text tells you what happened when you took the last action:
which objects moved, direction, distance, animation frames. Use this to build
hypotheses about each object's movement rule.
Always end your reply with: ACTION: <action_name>

[user]
<image: 512x512 PNG of current grid>
State: NOT_FINISHED | Level: 2/5
Available: ACTION1 ACTION2 ACTION3 ACTION4
Last action: ACTION4
Result of last action:
  animation: 8 frames
    color 1 (blue): (row~32, col~10) -> (row~32, col~18)  [RIGHT 8 cells, 8 frames]
    color 5 (gray): APPEARED at row~20 col~18
  Changed cells: 9
What is your next action? Reply with: ACTION: <name>

[assistant]
ACTION: ACTION3
```

---

## Repository Structure

```
arc_agent/                  core library (imported by all agents + eval)
  observation.py            grid_to_image, grid_diff, animation_to_text, analyze_animation
  llm.py                    Anthropic API wrapper with cost tracking
  runner.py                 play_one() loop + Agent protocol
  agents/
    random.py               RandomAgent (baseline)
    llm.py                  LLMAgent (Claude API, dev/silver-label use)

vlm_test/                   self-contained VLM capability + training smoke test
  README.md                 detailed instructions for this sub-project
  scripts/
    collect_data.py         run SDK games -> save (image, animation_text, diff, action) JSONL
    test_vlm.py             5 capability tests, saves inputs+outputs to disk
    tiny_train.py           QLoRA smoke test on collected data
  data/
    inputs/                 PNG + prompt.txt for each capability test
    train/
      dataset.jsonl         collected training steps (image_path + prompt + action)
  outputs/
    test_results.json       capability test results
    checkpoint/             trained LoRA adapter weights  [gitignored]

docs/
  ROADMAP.md                phase plan with architecture details and references
  ARCHITECTURE.md           module responsibilities and data flow
  LIBRARY.md                function index (every reusable function registered here)
  RESEARCH.md               past attempts, findings, known issues
  PAPER.md                  living paper draft

tests/                      pytest suite (47 tests)
eval.py                     batch evaluator: --agent {random,llm}  --budget $N
agent_starter.py            single-game demo
```

---

## Quick Start

```bash
# 1. install dependencies
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. set API keys in .env
ARC_API_KEY=<from arcprize.org/api-keys>
ANTHROPIC_API_KEY=<from console.anthropic.com>

# 3. run tests
.venv\Scripts\python.exe -m pytest tests/ -q

# 4. collect training data (requires ARC_API_KEY)
.venv\Scripts\python.exe vlm_test/scripts/collect_data.py --games ls20 --episodes 2

# 5. VLM capability test (downloads ~7GB model on first run)
.venv\Scripts\python.exe vlm_test/scripts/test_vlm.py

# 6. tiny QLoRA training smoke test
.venv\Scripts\python.exe vlm_test/scripts/tiny_train.py

# 7. batch eval with RandomAgent
.venv\Scripts\python.exe eval.py --agent random
```

---

## Key Findings

| Test | Result | Notes |
|------|--------|-------|
| Format compliance | PASS | Reliably outputs `ACTION: ACTIONx` |
| Color identification | PASS | Identifies colors from rendered grid |
| Spatial navigation | FAIL (zero-shot) | Outputs `ACTION: Right` not `ACTION: ACTION4` — format learned through fine-tuning |
| Consistency | PASS | 3/3 identical outputs for same input |
| QLoRA training | PASS | Loss 0.38→0.21 over 20 steps, checkpoint saves correctly |
| Trainable params | — | 1.84M / 3.76B (0.049%) — only LoRA adapters |
| Inference speed | — | ~0.4s/step on RTX A4500 |

**T3 failure explanation**: Zero-shot Qwen2.5-VL says `ACTION: Right` (natural language direction)
instead of `ACTION: ACTION4` (game enum). This is the primary thing BC fine-tuning fixes —
the model learns the game's action vocabulary from human traces.

---

## References

| # | Paper / Resource | Link |
|---|-----------------|------|
| 1 | Qwen2.5-VL (ViT + 2D-RoPE) | https://arxiv.org/abs/2502.13923 |
| 2 | QLoRA (4-bit NF4 + LoRA) | https://arxiv.org/abs/2305.14314 |
| 3 | LoRA | https://arxiv.org/abs/2106.09685 |
| 4 | PEFT library | https://huggingface.co/docs/peft/conceptual_guides/lora |
| 5 | BitsAndBytes | https://huggingface.co/docs/bitsandbytes/main/en/index |
| 6 | TRL SFTTrainer | https://huggingface.co/docs/trl/sft_trainer |
| 7 | Qwen2.5-VL-3B model | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct |
| 8 | ARC-AGI-3 Human Dataset | https://arcprize.org/blog/arc-agi-3-human-dataset |
| 9 | ARC Prize 2026 Kaggle | https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3 |
