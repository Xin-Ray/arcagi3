"""VLM-mode planning probe.

Same SYSTEM prompt as test_clean_qwen_planning.py, but ALSO pass a
rendered 64x64 RGB image of the board so the multimodal Qwen can use
its vision encoder. Goal: see whether visual grounding fixes the y-axis
inversion + wrong count failures.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


SYSTEM_PROMPT = """You are playing a turn-based 64x64 grid game.

You must output the action chain needed to move the yellow object to the green target.

Legal actions:
ACTION1 = move UP 3 cells
ACTION2 = move DOWN 3 cells
ACTION3 = move RIGHT 3 cells
ACTION4 = move LEFT 3 cells
ACTION5 = Perform Action. Use it exactly once only after the yellow object reaches the green target. Extra ACTION5 is invalid and does not help.
ACTION6 = AVOID
ACTION7 = UNDO / AVOID

Coordinate system:
- The board is 64x64.
- row = y, from top to bottom.
- col = x, from left to right.
- bbox format is [top_row, left_col, bottom_row, right_col].

Objects:
1. Yellow movable object:
   bbox = [15,45,23,53]
   center = x=49, y=19

2. Green target object:
   bbox = [45,51,53,59]
   center = x=55, y=49

Goal:
Move the yellow object so that it reaches the green target object.

Movement:
- ACTION1 moves the yellow object UP by 3 cells.
- ACTION2 moves the yellow object DOWN by 3 cells.
- ACTION3 moves the yellow object RIGHT by 3 cells.
- ACTION4 moves the yellow object LEFT by 3 cells.
- ACTION5 performs the final action only when the yellow object is already at the green target.
- ACTION6 is not used.
- ACTION7 undoes the previous action.

The image attached shows the current board: black is background, yellow is the movable object, green is the target.

Output rules:
- Output only the final answer.
- Do not explain.
- Do not output intermediate coordinates.
- Do not write natural language movement descriptions.
- Do not output numbered steps.
- Do not output markdown.
- Do not output more than two lines.
- The action chain must contain only ACTION1, ACTION2, ACTION3, ACTION4, ACTION5.
- The final action must be ACTION5.
- ACTION5 appears exactly once.
- ACTION5 must be the last action.

Output format exactly:
TOTAL_ACTIONS=<number>
ACTION_CHAIN=<comma-separated action tokens>"""

USER_PROMPT = "Plan now."


def _render_board(yellow_bbox: tuple[int, int, int, int],
                  green_bbox: tuple[int, int, int, int],
                  scale: int = 8):
    """Build a (64*scale, 64*scale, 3) uint8 RGB image. Black bg, yellow obj, green target."""
    import numpy as np
    grid = np.zeros((64, 64), dtype=np.int32)
    r0, c0, r1, c1 = yellow_bbox
    grid[r0:r1 + 1, c0:c1 + 1] = 4   # yellow
    r0, c0, r1, c1 = green_bbox
    grid[r0:r1 + 1, c0:c1 + 1] = 3   # green
    # Palette (subset)
    palette = np.array([
        [0, 0, 0],          # 0 black
        [0, 116, 217],      # 1 blue
        [255, 65, 54],      # 2 red
        [46, 204, 64],      # 3 green
        [255, 220, 0],      # 4 yellow
    ], dtype=np.uint8)
    rgb = palette[grid]                                # (64, 64, 3)
    rgb = np.repeat(np.repeat(rgb, scale, 0), scale, 1)  # upscale
    from PIL import Image
    return Image.fromarray(rgb)


def main() -> None:
    from arc_agent.vlm_backbone import HFBackbone

    print("Building 64x64 board image (yellow at [15,45,23,53], green at [45,51,53,59])...",
          flush=True)
    img = _render_board(
        yellow_bbox=(15, 45, 23, 53),
        green_bbox=(45, 51, 53, 59),
        scale=8,
    )
    out_png = REPO_ROOT / "outputs" / "vlm_planning_test_board.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    print(f"  saved preview: {out_png}", flush=True)

    print("Loading Qwen2.5-VL-3B (4-bit)...", flush=True)
    bb = HFBackbone.load()
    print("Loaded. Generating with image content block...", flush=True)

    raw = bb.generate(
        img,
        USER_PROMPT,
        system=SYSTEM_PROMPT,
        max_new_tokens=256,
        temperature=0.0,
    )

    print("\n" + "=" * 72)
    print("### USER PROMPT")
    print("=" * 72)
    print(USER_PROMPT)
    print()
    print("### IMAGE: see", out_png)
    print()
    print("=" * 72)
    print("### RAW MODEL OUTPUT (with image + text)")
    print("=" * 72)
    print(raw)
    print("=" * 72)

    print()
    print("=" * 72)
    print("### REFERENCE (deterministic)")
    print("=" * 72)
    yc = (49, 19)
    gc = (55, 49)
    dx = gc[0] - yc[0]
    dy = gc[1] - yc[1]
    n_right = abs(dx) // 3
    n_down = abs(dy) // 3
    print(f"yellow center: x={yc[0]}, y={yc[1]}")
    print(f"green  center: x={gc[0]}, y={gc[1]}")
    print(f"dx = +{dx} -> {n_right} x ACTION3 (RIGHT 3)")
    print(f"dy = +{dy} -> {n_down} x ACTION2 (DOWN 3)")
    print(f"plus 1 x ACTION5")
    print(f"OPTIMAL TOTAL_ACTIONS = {n_right + n_down + 1}")


if __name__ == "__main__":
    main()
