"""
Step 2 — Qwen2.5-VL-3B capability test. Saves ALL inputs and outputs to disk.

Inputs  saved to: vlm_test/data/inputs/t<N>_image.png + t<N>_prompt.txt
Outputs saved to: vlm_test/outputs/test_results.json

Usage:
    .venv\\Scripts\\python.exe vlm_test/scripts/test_vlm.py
"""
from __future__ import annotations

import json
import re
import sys
import time
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from arc_agent.observation import grid_to_image

MODEL_ID   = "Qwen/Qwen2.5-VL-3B-Instruct"
INPUT_DIR  = ROOT / "vlm_test" / "data" / "inputs"
OUTPUT_DIR = ROOT / "vlm_test" / "outputs"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── ARC 16-color names (for prompts) ─────────────────────────────────────────
COLOR_KEY = "0=black 1=blue 2=red 3=green 4=yellow 5=gray 6=magenta 7=orange 8=lightblue 9=maroon"

SYSTEM = textwrap.dedent("""\
    You are an AI agent playing a turn-based puzzle game.
    You see the game grid as an image. Each color represents a different game element.
    No instructions are given — you must infer the rules by observing the grid.
    Always end your response with exactly one line:  ACTION: <action_name>
    Example: ACTION: ACTION4
""")


# ── model loading ─────────────────────────────────────────────────────────────

def load_model():
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Loading {MODEL_ID}  device={device}  dtype={dtype} ...")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"  loaded in {time.time()-t0:.1f}s")
    return model, processor, device


def run_vlm(model, processor, device, messages, max_new_tokens=128):
    import torch
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0].strip()


def save_input(tag: str, image, prompt: str):
    """Save PNG + prompt text for one test case."""
    image.save(INPUT_DIR / f"{tag}_image.png")
    (INPUT_DIR / f"{tag}_prompt.txt").write_text(prompt, encoding="utf-8")


# ── test grid factories ───────────────────────────────────────────────────────

def make_grid(h=64, w=64):
    return np.zeros((h, w), dtype=np.int32)

def grid_dot(color, row, col, size=4):
    g = make_grid(); g[row:row+size, col:col+size] = color; return g

def grid_navigate():
    g = make_grid()
    g[28:36, 10:14] = 1   # blue character (left)
    g[20:44, 30:34] = 5   # gray wall (center)
    g[28:36, 50:54] = 4   # yellow target (right)
    return g

def grid_four_corners():
    g = make_grid()
    g[4:10,  4:10]  = 2; g[4:10,  54:60] = 3
    g[54:60, 4:10]  = 6; g[54:60, 54:60] = 4
    return g


# ── individual tests ──────────────────────────────────────────────────────────

def run_test(tag, description, grid, prompt_text, model, processor, device,
             max_new_tokens=128, n_runs=1):
    image = grid_to_image(grid, scale=8)
    save_input(tag, image, prompt_text)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": prompt_text},
        ]},
    ]

    outputs = []
    t0 = time.time()
    for _ in range(n_runs):
        outputs.append(run_vlm(model, processor, device, messages, max_new_tokens))
    elapsed = time.time() - t0

    actions = []
    for o in outputs:
        m = re.search(r"ACTION:\s*(ACTION\d+)", o)
        actions.append(m.group(1) if m else "NONE")

    has_action = all(a != "NONE" for a in actions)
    consistent = len(set(actions)) == 1

    print(f"\n[{tag}] {description}")
    print(f"  input image : vlm_test/data/inputs/{tag}_image.png")
    print(f"  input prompt: vlm_test/data/inputs/{tag}_prompt.txt")
    for i, (o, a) in enumerate(zip(outputs, actions)):
        print(f"  run {i+1} output: {o!r}  -> action={a}")
    print(f"  has_action={has_action}  consistent={consistent}  time={elapsed:.1f}s")

    return {
        "tag": tag, "description": description,
        "image_path": f"data/inputs/{tag}_image.png",
        "prompt": prompt_text,
        "outputs": outputs,
        "actions": actions,
        "has_action": has_action,
        "consistent": consistent,
        "elapsed_s": round(elapsed, 2),
        "pass": has_action,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Qwen2.5-VL-3B  ARC capability test")
    print("="*60)

    model, processor, device = load_model()
    results = []

    # T1 — blank grid, just format check
    results.append(run_test(
        "t1_format", "blank grid — format compliance",
        make_grid(),
        "State: NOT_FINISHED | Level: 1/3\nAvailable: ACTION1 ACTION2 ACTION3 ACTION4\n"
        "What is your next action? Reply with ACTION: <name>",
        model, processor, device,
    ))

    # T2 — single red dot, ask for color + position
    results.append(run_test(
        "t2_color", "single red dot top-left — color identification",
        grid_dot(color=2, row=4, col=4),
        f"State: NOT_FINISHED | Level: 1/3\n{COLOR_KEY}\n"
        "Available: ACTION1 ACTION2 ACTION3 ACTION4 ACTION5\n"
        "Describe the non-black element briefly, then pick an action. ACTION: <name>",
        model, processor, device, max_new_tokens=150,
    ))

    # T3 — navigate: blue left, gray wall center, yellow target right
    results.append(run_test(
        "t3_navigate", "blue(you) left | gray wall center | yellow target right",
        grid_navigate(),
        f"State: NOT_FINISHED | Level: 1/3\n{COLOR_KEY}\n"
        "Available: ACTION1(Up) ACTION2(Down) ACTION3(Left) ACTION4(Right)\n"
        "Hypothesis: I need to reach the yellow object.\n"
        "What is your next action? Reply with ACTION: <name>",
        model, processor, device, max_new_tokens=150,
    ))

    # T4 — four colored corners
    results.append(run_test(
        "t4_four_corners", "4 colored objects at corners",
        grid_four_corners(),
        f"State: NOT_FINISHED | Level: 1/3\n{COLOR_KEY}\n"
        "Available: ACTION1 ACTION2 ACTION3 ACTION4 ACTION5\n"
        "How many non-black objects do you see? Then: ACTION: <name>",
        model, processor, device, max_new_tokens=150,
    ))

    # T5 — consistency: same navigate grid, 3 runs
    results.append(run_test(
        "t5_consistency", "navigate grid x3 runs — consistency check",
        grid_navigate(),
        f"State: NOT_FINISHED | Level: 1/3\n{COLOR_KEY}\n"
        "Available: ACTION1 ACTION2 ACTION3 ACTION4\n"
        "What is your next action? Reply with ACTION: <name>",
        model, processor, device, max_new_tokens=64, n_runs=3,
    ))

    # ── save outputs ──────────────────────────────────────────────────────────
    out_file = OUTPUT_DIR / "test_results.json"
    summary = {
        "model":    MODEL_ID,
        "device":   "cuda",
        "date":     "2026-05-08",
        "tests":    results,
        "passed":   sum(1 for r in results if r["pass"]),
        "total":    len(results),
    }
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  [{status}]  {r['tag']:<20} {r['description']}")
    print(f"\n  {summary['passed']}/{summary['total']} passed")
    print(f"\n  All inputs  -> vlm_test/data/inputs/")
    print(f"  All outputs -> {out_file}")


if __name__ == "__main__":
    main()
