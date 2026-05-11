"""Compose step_*.png files in a run folder into play.gif.

Usage:
    .venv/Scripts/python.exe scripts/make_gif.py outputs/baseline_<ts>/<game_id>/

Reads `step_*.png` files in numeric order from the given folder, calls
`arc_agent.viz.write_gif`, and writes `play.gif` in the same folder.

Standalone tool — useful when you have step PNGs from a previous run and
want a GIF without re-running the agent. `run_baseline.py` will normally
write the GIF inline, so this script is a fallback / utility.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arc_agent.viz import write_gif


def _step_index(path: Path) -> int:
    m = re.search(r"step_(\d+)", path.name)
    return int(m.group(1)) if m else -1


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: python {sys.argv[0]} <folder_with_step_NNN_png_files>")
        sys.exit(2)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        sys.exit(f"not a folder: {folder}")

    step_files = sorted(folder.glob("step_*.png"), key=_step_index)
    if not step_files:
        sys.exit(f"no step_*.png files in {folder}")

    from PIL import Image  # local import: only needed when actually running

    print(f"Loading {len(step_files)} frames from {folder} ...")
    frames = [Image.open(p) for p in step_files]
    out = write_gif(frames, folder / "play.gif", fps=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
