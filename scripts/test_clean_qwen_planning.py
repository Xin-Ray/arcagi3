"""Clean planning-only Qwen test.

Loads the bare Qwen backbone (no perception, no Knowledge, no Reflection)
and sends one SYSTEM prompt. Dumps the raw model output verbatim. Goal:
see whether the model can plan a multi-step action chain from a clean
description of state + actions, isolating the LLM's planning ability
from the rest of the v3.2 pipeline.
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


def main() -> None:
    from arc_agent.vlm_backbone import HFBackbone

    print("Loading Qwen2.5-VL-3B (4-bit)...", flush=True)
    bb = HFBackbone.load()
    print("Loaded. Generating...", flush=True)

    raw = bb.generate(
        None,
        USER_PROMPT,
        system=SYSTEM_PROMPT,
        max_new_tokens=512,
        temperature=0.0,
    )

    print("\n" + "=" * 72)
    print("### SYSTEM PROMPT (sent verbatim)")
    print("=" * 72)
    print(SYSTEM_PROMPT)
    print()
    print("=" * 72)
    print(f"### USER PROMPT")
    print("=" * 72)
    print(USER_PROMPT)
    print()
    print("=" * 72)
    print("### RAW MODEL OUTPUT")
    print("=" * 72)
    print(raw)
    print("=" * 72)

    # Math reference
    print()
    print("=" * 72)
    print("### REFERENCE (deterministic)")
    print("=" * 72)
    yc = (49, 19)
    gc = (55, 49)
    dx = gc[0] - yc[0]   # +15 cols -> RIGHT
    dy = gc[1] - yc[1]   # +30 rows -> DOWN
    n_right = abs(dx) // 3
    n_down = abs(dy) // 3
    print(f"yellow center: x={yc[0]}, y={yc[1]}")
    print(f"green  center: x={gc[0]}, y={gc[1]}")
    print(f"dx = +{dx} -> {n_right} x ACTION3 (RIGHT 3)")
    print(f"dy = +{dy} -> {n_down} x ACTION2 (DOWN 3)")
    print(f"plus 1 x ACTION5 at the end")
    print(f"OPTIMAL TOTAL_ACTIONS = {n_right + n_down + 1}")


if __name__ == "__main__":
    main()
