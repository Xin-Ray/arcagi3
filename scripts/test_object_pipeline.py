"""Run scipy extractor + Hungarian aligner on the same 5-frame PNG set
that was previously fed to Qwen for diag, and emit a human-review report
with the exact same shape as `qwen_object_diag/<game>/report.md`.

The 5 PNGs per game were rendered by `scripts/render_clean_frames.py`
using `grid_to_image(scale=8)`. Each cell is an 8x8 block of pixels with
one of the 16 ARC palette colors -- so the grid is exactly recoverable
by sampling the centre pixel of each 8x8 block.

Outputs:
  outputs/scipy_object_diag/<game>/frame_<NN>.png   (copied)
  outputs/scipy_object_diag/<game>/extract/...json
  outputs/scipy_object_diag/<game>/align/...json
  outputs/scipy_object_diag/<game>/report.md
  outputs/scipy_object_diag/SUMMARY.md
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arc_agent.object_aligner import align_objects, matches_to_dict   # noqa: E402
from arc_agent.object_extractor import extract_objects, objects_to_dict   # noqa: E402

# Palette from arc_agent/observation.py:_ARC_PALETTE (kept in sync).
_ARC_PALETTE = np.array([
    (0,   0,   0),    # 0 black
    (0,   116, 217),  # 1 blue
    (255, 65,  54),   # 2 red
    (46,  204, 64),   # 3 green
    (255, 220, 0),    # 4 yellow
    (170, 170, 170),  # 5 gray
    (240, 18,  190),  # 6 magenta
    (255, 133, 27),   # 7 orange
    (127, 219, 255),  # 8 light blue
    (135, 12,  37),   # 9 maroon
    (84,  13,  110),  # 10 purple
    (230, 190, 120),  # 11 tan
    (40,  160, 140),  # 12 teal
    (200, 230, 50),   # 13 lime
    (180, 80,  80),   # 14 rose
    (5,   100, 180),  # 15 navy
], dtype=np.int32)


SOURCE_ROOT = REPO_ROOT / "outputs" / "qwen_object_diag"
OUT_ROOT = REPO_ROOT / "outputs" / "scipy_object_diag"


def decode_png_to_grid(png_path: Path, *, scale: int = 8) -> np.ndarray:
    """Reverse-decode a clean grid PNG back to a (H, W) int grid.

    Samples the centre pixel of each scale x scale block and maps to the
    nearest palette color (Euclidean RGB).
    """
    from PIL import Image

    img = Image.open(png_path).convert("RGB")
    arr = np.asarray(img, dtype=np.int32)
    h, w = arr.shape[:2]
    rows = h // scale
    cols = w // scale
    half = scale // 2
    # Sample centre pixel of each cell
    samples = arr[half::scale, half::scale, :]   # (rows, cols, 3)
    samples = samples[:rows, :cols, :]
    flat = samples.reshape(-1, 3)
    # Distance to each palette color (sq L2 is enough)
    palette = _ARC_PALETTE   # (16, 3)
    diff = flat[:, None, :] - palette[None, :, :]   # (N, 16, 3)
    dist2 = (diff * diff).sum(axis=2)                # (N, 16)
    nearest = dist2.argmin(axis=1).reshape(rows, cols)
    return nearest.astype(np.int64)


def _run_one_game(src_game_dir: Path, dst_game_dir: Path) -> dict:
    """Process one game: copy PNGs, run extract + align, write JSONs."""
    dst_game_dir.mkdir(parents=True, exist_ok=True)
    frames_dst = sorted(src_game_dir.glob("frame_*.png"))
    if not frames_dst:
        print(f"  [{src_game_dir.name}] no frame PNGs -- skipping")
        return {"game": src_game_dir.name, "n_frames": 0}

    extract_dir = dst_game_dir / "extract"
    align_dir = dst_game_dir / "align"
    extract_dir.mkdir(parents=True, exist_ok=True)
    align_dir.mkdir(parents=True, exist_ok=True)

    grids: list[tuple[Path, np.ndarray]] = []
    for fp in frames_dst:
        dst_png = dst_game_dir / fp.name
        if not dst_png.exists() or dst_png.stat().st_mtime < fp.stat().st_mtime:
            shutil.copy2(fp, dst_png)
        grid = decode_png_to_grid(dst_png, scale=8)
        grids.append((dst_png, grid))

    # EXTRACT
    extract_results: list[tuple[Path, list]] = []
    for fp, grid in grids:
        t0 = time.time()
        objs = extract_objects(grid)
        dt_ms = round((time.time() - t0) * 1000, 2)
        payload = {
            "kind": "extract", "frame": fp.name, "wall_ms": dt_ms,
            "raw": None, "parsed": objects_to_dict(objs), "parse_ok": True,
        }
        (extract_dir / (fp.stem + ".json")).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  [{src_game_dir.name}] EXTRACT {fp.name} -> n_objects={len(objs)}  ({dt_ms} ms)")
        extract_results.append((fp, objs))

    # ALIGN (adjacent pairs)
    for i in range(len(extract_results) - 1):
        before_fp, before_objs = extract_results[i]
        after_fp, after_objs = extract_results[i + 1]
        t0 = time.time()
        matches = align_objects(before_objs, after_objs)
        dt_ms = round((time.time() - t0) * 1000, 2)
        pair_name = (
            f"pair_{before_fp.stem.split('_')[1]}_"
            f"{after_fp.stem.split('_')[1]}"
        )
        payload = {
            "kind": "align", "before": before_fp.name, "after": after_fp.name,
            "wall_ms": dt_ms,
            "raw": None, "parsed": matches_to_dict(matches), "parse_ok": True,
        }
        (align_dir / f"{pair_name}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        n_chg = sum(1 for m in matches if m.type != "unchanged")
        print(f"  [{src_game_dir.name}] ALIGN  {pair_name} -> n_matches={len(matches)} "
              f"n_changed={n_chg}  ({dt_ms} ms)")

    return {"game": src_game_dir.name, "n_frames": len(frames_dst)}


def _build_one_report(dst_game_dir: Path) -> Path:
    frames = sorted(dst_game_dir.glob("frame_*.png"))
    extract_dir = dst_game_dir / "extract"
    align_dir = dst_game_dir / "align"

    lines: list[str] = [
        f"# scipy_object_diag -- {dst_game_dir.name}",
        "",
        "Deterministic pipeline: `scipy.ndimage.label` (extract) + Hungarian (align).",
        f"Frames analysed: **{len(frames)}**.",
        "",
        "---",
        "",
    ]

    for fp in frames:
        json_path = extract_dir / (fp.stem + ".json")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        parsed = payload["parsed"]
        n_obj = len(parsed["objects"])
        wall_ms = payload["wall_ms"]
        lines += [
            f"## {fp.stem} -- extract",
            "",
            f"![{fp.name}]({fp.name})",
            "",
            f"- wall_clock: {wall_ms} ms   |   objects: {n_obj}",
            "",
            "```json",
            json.dumps(parsed, indent=2, ensure_ascii=False),
            "```",
            "",
            "**Human check:**",
            "- [ ] all objects found?",
            "- [ ] colors correct?",
            "- [ ] bboxes plausible?",
            "- [ ] sizes plausible?",
            "",
            "---",
            "",
        ]

    for i in range(len(frames) - 1):
        before_fp, after_fp = frames[i], frames[i + 1]
        pair_name = (
            f"pair_{before_fp.stem.split('_')[1]}_"
            f"{after_fp.stem.split('_')[1]}"
        )
        json_path = align_dir / f"{pair_name}.json"
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        parsed = payload["parsed"]
        n_matches = len(parsed["matches"])
        n_chg = sum(1 for m in parsed["matches"] if m["type"] != "unchanged")
        wall_ms = payload["wall_ms"]
        lines += [
            f"## {pair_name} -- align",
            "",
            f"BEFORE: ![{before_fp.name}]({before_fp.name})",
            "",
            f"AFTER:  ![{after_fp.name}]({after_fp.name})",
            "",
            f"- wall_clock: {wall_ms} ms   |   matches: {n_matches}   |   changed: {n_chg}",
            "",
            "```json",
            json.dumps(parsed, indent=2, ensure_ascii=False),
            "```",
            "",
            "**Human check:**",
            "- [ ] all BEFORE objects accounted for?",
            "- [ ] match types correct?",
            "- [ ] deltas (dy/dx, from/to) correct?",
            "- [ ] no spurious appears / disappears?",
            "",
            "---",
            "",
        ]

    out = dst_game_dir / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _build_summary(root: Path, stats: list[dict]) -> Path:
    lines = [
        "# scipy_object_diag -- SUMMARY",
        "",
        "Deterministic baseline using `arc_agent.object_extractor` (scipy 4-connected)",
        "+ `arc_agent.object_aligner` (Hungarian). Compare to `qwen_object_diag/`",
        "for the same 4 games x 5 frames.",
        "",
        "## Reports",
        "",
    ]
    for s in stats:
        lines.append(f"- [{s['game']}/report.md]({s['game']}/report.md)  ({s['n_frames']} frames)")
    lines += [
        "",
        "## Manual review",
        "",
        "Open each per-game `report.md` and verify objects + matches against the",
        "PNG. Since this is deterministic (not LLM), the EXPECTED result on these",
        "demo games is 100% correctness IF the 'object = same-color 4-connected'",
        "definition matches your intuition. If an extracted object disagrees with",
        "what you expected, that's a signal that the object definition needs to",
        "change (e.g., cross-color compound), not that the implementation is buggy.",
        "",
    ]
    p = root / "SUMMARY.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(SOURCE_ROOT))
    parser.add_argument("--output", default=str(OUT_ROOT))
    args = parser.parse_args()

    src = Path(args.source)
    dst = Path(args.output)
    dst.mkdir(parents=True, exist_ok=True)

    game_dirs = sorted(
        d for d in src.iterdir() if d.is_dir() and any(d.glob("frame_*.png"))
    )
    if not game_dirs:
        raise SystemExit(f"no game frames found under {src}")
    print(f"Processing {len(game_dirs)} game(s):")
    for d in game_dirs:
        print(f"  - {d.name}")
    print()

    stats: list[dict] = []
    for src_gdir in game_dirs:
        print(f"=== {src_gdir.name} ===")
        s = _run_one_game(src_gdir, dst / src_gdir.name)
        stats.append(s)
        _build_one_report(dst / src_gdir.name)

    summary = _build_summary(dst, stats)
    print(f"\nWrote {summary}")
    print(f"Total wall: scipy extracts + aligns done in microseconds per call.")


if __name__ == "__main__":
    main()
