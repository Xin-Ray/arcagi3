"""Minimal natural-language spatial reasoning probe.

One Chinese sentence describing the scene; ask which direction to move.
No coordinates, no math, no action map. Tests whether the LLM can do
elementary spatial reasoning ("top-left of X => move down-right").
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# One-sentence prompt. No system prompt, no coordinates, no action enum.
USER_PROMPT = (
    "黄色L方块在绿色L方块的左上角，你可以移动黄色方块，请问往哪个方向移动，输出方向"
)


def main() -> None:
    # Force UTF-8 stdout so Chinese can be printed under cp1252 consoles
    import io
    sys.stdout.reconfigure(encoding="utf-8")

    from arc_agent.vlm_backbone import HFBackbone

    print("Loading Qwen2.5-VL-3B (4-bit)...", flush=True)
    bb = HFBackbone.load()
    print("Loaded. Generating...", flush=True)

    raw = bb.generate(
        None,
        USER_PROMPT,
        system="",
        max_new_tokens=128,
        temperature=0.0,
    )

    out = []
    out.append("=" * 72)
    out.append("### USER PROMPT (no system, no coords, no action map)")
    out.append("=" * 72)
    out.append(USER_PROMPT)
    out.append("")
    out.append("=" * 72)
    out.append("### RAW MODEL OUTPUT")
    out.append("=" * 72)
    out.append(raw)
    out.append("=" * 72)
    out.append("")
    out.append("### REFERENCE")
    out.append("  Yellow is at top-left of green, so to reach green move DOWN+RIGHT")
    out.append("  (= 下右 / 右下 / move down and right)")
    body = "\n".join(out)
    print(body)
    out_path = REPO_ROOT / "outputs" / "qwen_simple_spatial_probe.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
