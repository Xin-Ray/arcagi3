"""
Qwen2.5-VL-3B-Instruct capability test on ARC-style game inputs.

Tests:
  T1  Model loads and runs             — basic sanity
  T2  Format compliance                — does it output "ACTION: ACTIONx"?
  T3  Color identification             — can it read colors from the rendered grid?
  T4  Spatial localization             — can it find WHERE an object is?
  T5  Full game prompt                 — behaves like the real VLMAgent would use
  T6  Output consistency               — same input × 3 runs, how stable?

Run:
    .venv\\Scripts\\python.exe scripts/test_vlm.py

First run downloads ~7 GB from HuggingFace. Set HF_HOME to control cache location.
GPU required for practical speed (CPU works but each call takes ~60s).
"""
from __future__ import annotations

import sys
import time
import textwrap
from pathlib import Path

import numpy as np

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arc_agent.observation import grid_to_image

# ── model id ─────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# ── helpers ───────────────────────────────────────────────────────────────────

def load_model():
    """Load Qwen2.5-VL-3B-Instruct in BF16. Falls back to float32 if no CUDA."""
    print(f"Loading {MODEL_ID} ...")
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"  device={device}  dtype={dtype}")

    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print(f"  loaded in {time.time()-t0:.1f}s\n")
    return model, processor, device


def run_vlm(model, processor, device, messages: list[dict], max_new_tokens: int = 64) -> str:
    """Run one inference call; return the assistant text."""
    import torch
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def make_grid(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w), dtype=np.int32)


def section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def result(label: str, output: str, expected_hint: str = ""):
    print(f"\n[{label}]")
    if expected_hint:
        print(f"  expected hint : {expected_hint}")
    print(f"  model output  : {output!r}")


# ── test grids ────────────────────────────────────────────────────────────────

def grid_single_dot(color: int, row: int, col: int) -> np.ndarray:
    """64×64 black grid with one 4×4 colored square."""
    g = make_grid()
    g[row:row+4, col:col+4] = color
    return g


def grid_navigate() -> np.ndarray:
    """
    Puzzle: blue character (color 1) on left, yellow target (color 4) on right,
    gray wall (color 5) in the center. Agent must go around the wall.

    Layout (rows 28-36, cols 10-54):
      col 10-14 : blue character  (color 1)
      col 30-34 : gray wall rows 20-44  (color 5)
      col 50-54 : yellow target  (color 4)
    """
    g = make_grid()
    g[28:36, 10:14] = 1   # blue character
    g[20:44, 30:34] = 5   # gray wall
    g[28:36, 50:54] = 4   # yellow target
    return g


def grid_colored_objects() -> np.ndarray:
    """Four colored blobs at four corners — tests color + position reading."""
    g = make_grid()
    g[4:10,  4:10]  = 2   # red    top-left
    g[4:10,  54:60] = 3   # green  top-right
    g[54:60, 4:10]  = 6   # magenta bottom-left
    g[54:60, 54:60] = 4   # yellow bottom-right
    return g


# ── system prompt (same one VLMAgent will use) ────────────────────────────────

SYSTEM = textwrap.dedent("""\
    You are an AI agent playing a turn-based puzzle game.
    You see the game grid as an image. Each color represents a different game element.
    No instructions are given — you must infer the rules by observing how your actions change the grid.

    Available actions and their typical meanings:
      ACTION1 = Move Up
      ACTION2 = Move Down
      ACTION3 = Move Left
      ACTION4 = Move Right
      ACTION5 = Primary interact (game-specific: select, rotate, push, etc.)
      ACTION6 = Click coordinate (needs x,y)
      ACTION7 = Undo last action

    Reply format — always end your response with exactly one line:
      ACTION: <action_name>
    Example: ACTION: ACTION4
""")


# ── tests ─────────────────────────────────────────────────────────────────────

def t1_basic_format(model, processor, device):
    section("T1 — Basic format compliance")
    print("Blank grid. Ask for any action. Does the model follow the format?")

    grid  = make_grid()
    image = grid_to_image(grid, scale=8)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": (
                "State: NOT_FINISHED | Level: 1/3\n"
                "Available: ACTION1 ACTION2 ACTION3 ACTION4\n"
                "What is your next action?"
            )},
        ]},
    ]
    out = run_vlm(model, processor, device, messages)
    result("T1", out, "must contain 'ACTION: ACTION<n>'")
    ok = "ACTION:" in out
    print(f"  PASS: {ok}" if ok else "  FAIL: no ACTION: token found")
    return ok


def t2_color_reading(model, processor, device):
    section("T2 — Color identification")
    print("Grid with a single RED square (color 2) at top-left area (rows 4-8, cols 4-8).")
    print("Ask what color is there.")

    grid  = grid_single_dot(color=2, row=4, col=4)
    image = grid_to_image(grid, scale=8)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": (
                "State: NOT_FINISHED | Level: 1/3\n"
                "Color key: 0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=gray\n\n"
                "Question: What non-black color do you see, and roughly where is it?\n"
                "Then pick an action. Reply with ACTION: <name>"
            )},
        ]},
    ]
    out = run_vlm(model, processor, device, messages, max_new_tokens=128)
    result("T2", out, "should mention red / color 2 / top-left area")
    ok = any(w in out.lower() for w in ["red", "color 2", "top-left", "upper-left", "upper left"])
    print(f"  PASS: {ok}" if ok else "  FAIL: didn't identify red / top-left")
    return ok


def t3_spatial_localization(model, processor, device):
    section("T3 — Spatial localization")
    print("Blue character (left), yellow target (right), gray wall (center).")
    print("Does the model understand the layout and reason about navigation?")

    grid  = grid_navigate()
    image = grid_to_image(grid, scale=8)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": (
                "State: NOT_FINISHED | Level: 1/3\n"
                "Available: ACTION1 ACTION2 ACTION3 ACTION4 ACTION5\n"
                "Color key: 0=black(empty), 1=blue(you/character), 4=yellow(target/goal), 5=gray(wall)\n\n"
                "Hypothesis: I need to reach the yellow target.\n"
                "Describe what you see, then pick your next action. Reply with ACTION: <name>"
            )},
        ]},
    ]
    out = run_vlm(model, processor, device, messages, max_new_tokens=200)
    result("T3", out, "should mention left/right/wall and ACTION1-4")
    ok = "ACTION:" in out
    print(f"  PASS (has action): {ok}")
    # also check if it described the layout
    has_spatial = any(w in out.lower() for w in ["left", "right", "wall", "target", "character", "blue", "yellow"])
    print(f"  Spatial reasoning words found: {has_spatial}")
    return ok


def t4_four_objects(model, processor, device):
    section("T4 — Multi-object scene understanding")
    print("Four colored blobs at four corners. Tests if model can describe all objects.")

    grid  = grid_colored_objects()
    image = grid_to_image(grid, scale=8)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": (
                "State: NOT_FINISHED | Level: 1/3\n"
                "Available: ACTION1 ACTION2 ACTION3 ACTION4 ACTION5\n"
                "Color key: 0=black(empty), 2=red, 3=green, 4=yellow, 6=magenta\n\n"
                "How many distinct non-black colored objects do you see, and where are they?\n"
                "Then pick an action. Reply with ACTION: <name>"
            )},
        ]},
    ]
    out = run_vlm(model, processor, device, messages, max_new_tokens=200)
    result("T4", out, "should mention ~4 objects at corners")
    # rough check: mentions multiple objects / corners
    ok = "ACTION:" in out
    corners_mentioned = sum(
        1 for w in ["corner", "top", "bottom", "left", "right"] if w in out.lower()
    )
    print(f"  PASS (has action): {ok}  |  corner-related words: {corners_mentioned}")
    return ok


def t5_consistency(model, processor, device):
    section("T5 — Output consistency (same prompt × 3 runs)")
    print("Identical grid + prompt, three separate inference calls.")
    print("How stable is the chosen action?")

    grid  = grid_navigate()
    image = grid_to_image(grid, scale=8)

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": (
                "State: NOT_FINISHED | Level: 1/3\n"
                "Available: ACTION1 ACTION2 ACTION3 ACTION4\n"
                "Color key: 0=black, 1=blue(you), 4=yellow(target), 5=gray(wall)\n"
                "What is your next action? Reply with ACTION: <name>"
            )},
        ]},
    ]

    outputs = []
    for i in range(3):
        out = run_vlm(model, processor, device, messages, max_new_tokens=64)
        outputs.append(out)
        print(f"  run {i+1}: {out!r}")

    # extract action tokens
    import re
    actions = []
    for o in outputs:
        m = re.search(r"ACTION:\s*(ACTION\d+)", o)
        actions.append(m.group(1) if m else "NONE")
    unique = set(actions)
    print(f"\n  Actions: {actions}")
    print(f"  Unique:  {unique}  ({'consistent' if len(unique)==1 else 'INCONSISTENT'})")
    return len(unique) == 1


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "#" * 60)
    print("  Qwen2.5-VL-3B-Instruct — ARC game capability test")
    print("#" * 60)
    print(f"  Model : {MODEL_ID}")
    print(f"  Scale : 64x64 grid -> 512x512 PNG (scale=8)")

    try:
        model, processor, device = load_model()
    except ImportError as e:
        print(f"\nERROR: {e}")
        print("Install: pip install transformers accelerate qwen-vl-utils torch")
        sys.exit(1)

    results = {}
    for test_fn in [t1_basic_format, t2_color_reading, t3_spatial_localization,
                    t4_four_objects, t5_consistency]:
        t0 = time.time()
        try:
            ok = test_fn(model, processor, device)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            ok = False
        results[test_fn.__name__] = (ok, time.time() - t0)

    section("SUMMARY")
    for name, (ok, elapsed) in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {name:<35}  ({elapsed:.1f}s)")

    passed = sum(1 for ok, _ in results.values() if ok)
    print(f"\n  {passed}/{len(results)} tests passed")
    print("\nKey findings to record in RESEARCH.md:")
    print("  - Does the model reliably output 'ACTION: ACTIONx'?")
    print("  - Can it identify colors from the rendered image?")
    print("  - Does it demonstrate spatial reasoning?")
    print("  - Is output consistent across runs?")


if __name__ == "__main__":
    main()
