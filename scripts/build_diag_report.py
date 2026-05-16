"""Build human-review Markdown reports from qwen_object_diag outputs.

Reads `outputs/qwen_object_diag/<game>/extract/*.json` and `.../align/*.json`
written by `scripts/qwen_object_diag.py`, plus the frame PNGs, and produces:

  outputs/qwen_object_diag/<game>/report.md   (per game, with images + JSON inline)
  outputs/qwen_object_diag/SUMMARY.md         (one-page index + parse-rate stats)

The reports do NOT score Qwen's output — they format raw output side-by-side
with the frames so the user can manually mark right/wrong. Empty checkbox
prompts (`- [ ] all objects found?`) follow each block so the file doubles
as a review sheet.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DIAG_ROOT = REPO_ROOT / "outputs" / "qwen_object_diag"


def _load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _format_extract_block(parsed: dict | None, raw: str) -> str:
    if parsed is None:
        return (
            "_(parse failed — raw model output below)_\n\n"
            "```\n" + (raw or "(empty)") + "\n```"
        )
    return "```json\n" + json.dumps(parsed, indent=2, ensure_ascii=False) + "\n```"


def _format_align_block(parsed: dict | None, raw: str) -> str:
    return _format_extract_block(parsed, raw)


def _build_one_report(game_dir: Path) -> tuple[Path, dict[str, Any]]:
    frames = sorted(game_dir.glob("frame_*.png"))
    extract_dir = game_dir / "extract"
    align_dir = game_dir / "align"

    lines: list[str] = [
        f"# qwen_object_diag — {game_dir.name}",
        "",
        f"Frames analysed: **{len(frames)}**. ",
        "Each section shows the clean grid PNG followed by the raw Qwen output. ",
        "Tick the checkboxes after manual review.",
        "",
        "---",
        "",
    ]

    extract_total = 0
    extract_parse_ok = 0
    align_total = 0
    align_parse_ok = 0

    # Per-frame extract sections
    for fp in frames:
        json_path = extract_dir / (fp.stem + ".json")
        payload = _load_json(json_path)
        extract_total += 1
        if payload and payload.get("parse_ok"):
            extract_parse_ok += 1
        parsed = (payload or {}).get("parsed")
        raw = (payload or {}).get("raw", "")
        wall = (payload or {}).get("wall_s")
        n_obj = len(parsed["objects"]) if parsed and "objects" in parsed else None

        lines += [
            f"## {fp.stem} — extract",
            "",
            f"![{fp.name}]({fp.name})",
            "",
            f"- wall_clock: {wall}s   |   objects reported: {n_obj if n_obj is not None else 'n/a'}   |   parse_ok: {payload.get('parse_ok') if payload else 'missing'}",
            "",
            _format_extract_block(parsed, raw),
            "",
            "**Human check:**",
            "- [ ] all objects found?",
            "- [ ] colors correct?",
            "- [ ] bboxes plausible?",
            "- [ ] sizes plausible?",
            "- [ ] descriptions reasonable?",
            "",
            "---",
            "",
        ]

    # Adjacent-pair align sections
    for i in range(len(frames) - 1):
        before_fp, after_fp = frames[i], frames[i + 1]
        pair_name = f"pair_{before_fp.stem.split('_')[1]}_{after_fp.stem.split('_')[1]}"
        json_path = align_dir / f"{pair_name}.json"
        payload = _load_json(json_path)
        align_total += 1
        if payload and payload.get("parse_ok"):
            align_parse_ok += 1
        parsed = (payload or {}).get("parsed")
        raw = (payload or {}).get("raw", "")
        wall = (payload or {}).get("wall_s")
        n_matches = len(parsed["matches"]) if parsed and "matches" in parsed else None

        lines += [
            f"## {pair_name} — align",
            "",
            f"BEFORE: ![{before_fp.name}]({before_fp.name})",
            "",
            f"AFTER:  ![{after_fp.name}]({after_fp.name})",
            "",
            f"- wall_clock: {wall}s   |   matches reported: {n_matches if n_matches is not None else 'n/a'}   |   parse_ok: {payload.get('parse_ok') if payload else 'missing'}",
            "",
            _format_align_block(parsed, raw),
            "",
            "**Human check:**",
            "- [ ] all BEFORE objects accounted for?",
            "- [ ] match types correct (moved / recolored / disappeared / appeared)?",
            "- [ ] deltas (dy/dx, from/to) correct?",
            "- [ ] no spurious appears / disappears?",
            "",
            "---",
            "",
        ]

    out_path = game_dir / "report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path, {
        "game": game_dir.name,
        "n_frames": len(frames),
        "extract_parse_rate": round(extract_parse_ok / extract_total, 3) if extract_total else 0.0,
        "align_parse_rate": round(align_parse_ok / align_total, 3) if align_total else 0.0,
    }


def _build_summary(root: Path, stats: list[dict]) -> Path:
    # Diagnose failure modes per game by sampling raw outputs
    failure_modes: dict[str, str] = {}
    for s in stats:
        game_dir = root / s["game"]
        modes: list[str] = []
        for fp in sorted((game_dir / "extract").glob("*.json")):
            payload = _load_json(fp)
            if payload is None:
                continue
            raw = (payload.get("raw") or "")
            parsed = payload.get("parsed")
            if not payload.get("parse_ok"):
                # Diagnose: truncation vs other
                if len(raw) > 1500 and not raw.rstrip().endswith("```") and not raw.rstrip().endswith("}"):
                    modes.append("truncated (hallucination loop)")
                else:
                    modes.append("malformed JSON")
            else:
                n = len((parsed or {}).get("objects", []))
                if n > 10:
                    modes.append(f"over-segmented (n={n})")
        failure_modes[s["game"]] = ", ".join(sorted(set(modes))) if modes else "ok"

    lines = [
        "# qwen_object_diag — SUMMARY",
        "",
        "Quick index across all games. Open each per-game `report.md` for the",
        "human-review checklists with PNG + raw Qwen JSON side by side.",
        "",
        "## Parse rates + failure modes",
        "",
        "| Game | Frames | Extract parse_ok | Align parse_ok | Extract failure mode |",
        "|---|---|---|---|---|",
    ]
    for s in stats:
        lines.append(
            f"| {s['game']} | {s['n_frames']} | "
            f"{s['extract_parse_rate']:.1%} | {s['align_parse_rate']:.1%} | "
            f"{failure_modes.get(s['game'], '?')} |"
        )
    lines += [
        "",
        "## Observed Qwen failure patterns (read these BEFORE opening reports)",
        "",
        "1. **Format drift**: on simple frames Qwen returns a bare JSON array `[{...}, {...}]` ",
        "   instead of the requested `{\"objects\": [...]}`. The tolerant re-parser ",
        "   (`scripts/reparse_diag.py`) wraps these as `{\"objects\": [...]}` automatically. ",
        "   This is a prompt issue, not a Qwen reasoning issue.",
        "2. **Hallucination loop**: on more complex frames Qwen wraps correctly but ",
        "   then emits 15-20 duplicate \"L-shape\"/\"3x1 bar\" objects with incrementing ",
        "   ids until the 1024-token budget cuts it off. Increasing tokens won't help — ",
        "   the loop is intrinsic. **Look at the truncated raw output in the per-game ",
        "   reports for `bp35` and `cd82` to see this.**",
        "3. **Align task is unaffected**: 100% parse rate across all games and pairs. ",
        "   The output shape `{\"matches\": [...]}` apparently anchors Qwen better than ",
        "   `{\"objects\": [...]}`.",
        "",
        "## What to do with this",
        "",
        "- Open the per-game `report.md` (links below) and **look at each PNG + JSON** ",
        "  to see if Qwen's outputs are useful, even where parse_ok=True.",
        "- For pairs where parse_ok=False the raw text is still shown verbatim in the ",
        "  report — you can manually judge quality from the truncated text.",
        "- After review, decide: is Qwen-3B usable as an object extractor (with a ",
        "  better prompt), or do we need a bigger model / a deterministic extractor.",
        "",
        "## Reports",
        "",
    ]
    for s in stats:
        lines.append(f"- [{s['game']}/report.md]({s['game']}/report.md)")
    lines += [
        "",
        "## Manual review checklist (when reading per-game reports)",
        "",
        "1. Open in a Markdown viewer that renders images (VS Code preview, Typora, GitHub).",
        "2. For every section, look at the PNG + JSON output.",
        "3. Tick the checkboxes that pass, leave blank the ones that fail.",
        "4. Send the marked-up file back so we can quantify Qwen's error modes.",
        "",
    ]
    p = root / "SUMMARY.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DIAG_ROOT))
    args = parser.parse_args()

    root = Path(args.root)
    game_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not game_dirs:
        raise SystemExit(f"no game folders under {root}")

    stats: list[dict] = []
    for gdir in game_dirs:
        out, s = _build_one_report(gdir)
        stats.append(s)
        print(f"wrote {out}")

    summary = _build_summary(root, stats)
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
