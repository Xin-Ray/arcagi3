"""Run Qwen2.5-VL extract + align on the clean PNGs from render_clean_frames.

For every `outputs/qwen_object_diag/<game>/frame_<NN>.png` saved by
`scripts/render_clean_frames.py`, run two diagnostic Qwen calls:

  EXTRACT:  one call per frame -> list of objects (per-frame JSON)
  ALIGN:    one call per adjacent (frame_NN, frame_NN+1) pair
            -> match list (per-pair JSON)

Outputs (JSON, raw model text saved verbatim plus parsed dict if parseable):
  outputs/qwen_object_diag/<game>/extract/frame_<NN>.json
  outputs/qwen_object_diag/<game>/align/pair_<NN>_<NN+1>.json

Prompts come from `arc_agent/prompts.py` (single source of truth so the
design doc stays in sync). Model is loaded ONCE (4-bit, shared across all
36 calls) so wall-clock cost is roughly load + 36 inference passes.

This script does not score Qwen's output -- that's `build_diag_report.py`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arc_agent.prompts import (   # noqa: E402
    ALIGN_SYSTEM,
    ALIGN_USER,
    EXTRACT_SYSTEM,
    EXTRACT_USER,
)

DIAG_ROOT = REPO_ROOT / "outputs" / "qwen_object_diag"


def _load_backbone():
    """Load Qwen2.5-VL-3B 4-bit; return (model, processor)."""
    from arc_agent.vlm_backbone import load_model
    return load_model(quantize="4bit")


def _generate_with_images(
    model: Any, processor: Any, images: list, system: str, user: str,
    *, max_new_tokens: int = 1024, temperature: float = 0.0,
) -> str:
    """Multi-image chat call. images = [PIL.Image, ...] (1 or 2)."""
    import torch
    from qwen_vl_utils import process_vision_info

    content: list[dict] = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": user})
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature,
        )
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    return decoded[0]


_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def _tolerant_parse(text: str) -> dict | None:
    """Pull the first balanced {...} substring; tolerant of fences and prose."""
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        m = _JSON_OBJ_RE.search(text)
        candidate = m.group(0) if m else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _save_call(out_path: Path, *, raw: str, parsed: dict | None, meta: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**meta, "raw": raw, "parsed": parsed, "parse_ok": parsed is not None}
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")


def _run_one_game(model, processor, game_dir: Path) -> None:
    from PIL import Image

    frames = sorted(game_dir.glob("frame_*.png"))
    if not frames:
        print(f"  [{game_dir.name}] no frame PNGs -- skipping")
        return

    extract_dir = game_dir / "extract"
    align_dir = game_dir / "align"

    # EXTRACT pass -----------------------------------------------------------
    extract_results: list[tuple[Path, dict | None]] = []
    for fp in frames:
        t0 = time.time()
        img = Image.open(fp).convert("RGB")
        raw = _generate_with_images(
            model, processor, [img],
            system=EXTRACT_SYSTEM, user=EXTRACT_USER,
            max_new_tokens=1024,
        )
        parsed = _tolerant_parse(raw)
        out_path = extract_dir / (fp.stem + ".json")
        _save_call(
            out_path, raw=raw, parsed=parsed,
            meta={"kind": "extract", "frame": fp.name,
                  "wall_s": round(time.time() - t0, 2)},
        )
        n_objects = len(parsed["objects"]) if parsed and "objects" in parsed else "n/a"
        print(f"  [{game_dir.name}] EXTRACT {fp.name} -> {out_path.name} "
              f"parse_ok={parsed is not None}  n_objects={n_objects}  "
              f"({round(time.time() - t0, 1)}s)")
        extract_results.append((fp, parsed))

    # ALIGN pass -------------------------------------------------------------
    for i in range(len(frames) - 1):
        before_fp, after_fp = frames[i], frames[i + 1]
        t0 = time.time()
        before_img = Image.open(before_fp).convert("RGB")
        after_img = Image.open(after_fp).convert("RGB")
        raw = _generate_with_images(
            model, processor, [before_img, after_img],
            system=ALIGN_SYSTEM, user=ALIGN_USER,
            max_new_tokens=1024,
        )
        parsed = _tolerant_parse(raw)
        pair_name = f"pair_{before_fp.stem.split('_')[1]}_{after_fp.stem.split('_')[1]}"
        out_path = align_dir / f"{pair_name}.json"
        _save_call(
            out_path, raw=raw, parsed=parsed,
            meta={"kind": "align", "before": before_fp.name,
                  "after": after_fp.name,
                  "wall_s": round(time.time() - t0, 2)},
        )
        n_matches = len(parsed["matches"]) if parsed and "matches" in parsed else "n/a"
        print(f"  [{game_dir.name}] ALIGN  {pair_name} -> {out_path.name} "
              f"parse_ok={parsed is not None}  n_matches={n_matches}  "
              f"({round(time.time() - t0, 1)}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DIAG_ROOT),
                        help="root containing per-game folders with frame_*.png")
    args = parser.parse_args()

    root = Path(args.root)
    game_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not game_dirs:
        raise SystemExit(f"no game folders found under {root}")
    print(f"Found {len(game_dirs)} game folders. Loading Qwen backbone...")

    t_load = time.time()
    model, processor = _load_backbone()
    print(f"Backbone loaded in {round(time.time() - t_load, 1)}s.\n")

    t_all = time.time()
    for gdir in game_dirs:
        print(f"=== {gdir.name} ===")
        _run_one_game(model, processor, gdir)
    print(f"\nTotal Qwen wall-clock: {round(time.time() - t_all, 1)}s")


if __name__ == "__main__":
    main()
