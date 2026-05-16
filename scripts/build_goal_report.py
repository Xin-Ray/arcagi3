"""Combine the per-game vl_image.json + text_3b.json into a side-by-side
markdown report for human review.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "outputs" / "goal_inference"


def _load(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _format_payload(p: dict | None) -> tuple[str, str]:
    """Return (header_line, body_block) for one model's output."""
    if p is None:
        return "(missing)", "_(no file)_"
    header = (
        f"wall_clock: {p.get('wall_s', '?')}s, "
        f"parse_ok: {p.get('parse_ok')}"
    )
    parsed = p.get("parsed")
    if parsed:
        body = "```json\n" + json.dumps(parsed, indent=2, ensure_ascii=False) + "\n```"
    else:
        raw = (p.get("raw") or "")[:1500]
        body = "_(parse failed — raw output below)_\n\n```\n" + raw + "\n```"
    return header, body


def main() -> None:
    games = sorted(d for d in OUT_ROOT.iterdir() if d.is_dir())
    if not games:
        raise SystemExit(f"no per-game dirs under {OUT_ROOT}")

    lines = [
        "# Goal-inference bench — VL+image vs Text-3B+scipy",
        "",
        "Same SYSTEM prompt asking for JSON {primary_goal, evidence, alternatives, "
        "invariants, dynamics}. **No automatic scoring** — read both columns, "
        "decide which is more plausible / useful.",
        "",
        "| Game | VL+image | Text-3B+scipy |",
        "|---|---|---|",
    ]
    # Pre-pass for the index table
    for gdir in games:
        v = _load(gdir / "vl_image.json")
        t = _load(gdir / "text_3b.json")
        v_goal = (v or {}).get("parsed", {}).get("primary_goal", "(none)") if v else "(missing)"
        t_goal = (t or {}).get("parsed", {}).get("primary_goal", "(none)") if t else "(missing)"
        lines.append(f"| {gdir.name} | {v_goal[:60]} | {t_goal[:60]} |")
    lines.append("")

    for gdir in games:
        v = _load(gdir / "vl_image.json")
        t = _load(gdir / "text_3b.json")
        actions = (v or {}).get("actions") or (t or {}).get("actions") or []
        scipy_dir = REPO_ROOT / "outputs" / "scipy_object_diag" / gdir.name
        rel_dir = f"../scipy_object_diag/{gdir.name}"
        lines += [
            f"## {gdir.name}",
            "",
            f"Actions taken: `{','.join(actions)}`",
            "",
            f"frame_00:  ![]({rel_dir}/frame_00.png)",
            "",
            f"frame_04:  ![]({rel_dir}/frame_04.png)",
            "",
        ]
        v_hdr, v_body = _format_payload(v)
        t_hdr, t_body = _format_payload(t)
        lines += [
            "### A) Qwen2.5-VL-3B + 2 images",
            "",
            v_hdr,
            "",
            v_body,
            "",
            "### B) Qwen2.5-3B-Instruct + scipy text only",
            "",
            t_hdr,
            "",
            t_body,
            "",
            "**Human judgment:**",
            "- [ ] A primary_goal is plausible",
            "- [ ] B primary_goal is plausible",
            "- [ ] A or B more useful (circle one)",
            "",
            "---",
            "",
        ]

    (OUT_ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_ROOT / 'report.md'}")


if __name__ == "__main__":
    main()
