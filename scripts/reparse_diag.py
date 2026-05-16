"""Re-parse already-saved raw Qwen outputs with a tolerant parser.

The original `qwen_object_diag.py` parser only accepted `{...}` payloads,
but on the EXTRACT task Qwen frequently returns a bare list `[{...}, {...}]`.
This script walks every saved `*.json` under `outputs/qwen_object_diag/`,
re-runs a tolerant parser on the `raw` field, and updates `parsed` /
`parse_ok` in place — no GPU needed, no API calls.

After running this, rerun `scripts/build_diag_report.py` to refresh the
human-review reports.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DIAG_ROOT = REPO_ROOT / "outputs" / "qwen_object_diag"

# ----- tolerant parser ------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```")


def _extract_first_balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    start = text.find(open_ch)
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def tolerant_parse(text: str, *, kind: str) -> Any:
    """Return a dict (canonicalized) or None.

    - For kind="extract": if the model returned `[...]`, wrap as `{"objects": [...]}`.
    - For kind="align":   if the model returned `[...]`, wrap as `{"matches": [...]}`.
    - For both: if the model returned `{...}`, return as-is.

    Strips ```json``` fences, picks first balanced bracket.
    """
    if not isinstance(text, str):
        return None

    candidate = None

    # 1) Try ``` fenced ``` first
    fenced = _FENCE_RE.search(text)
    if fenced:
        candidate = fenced.group(1).strip()

    # 2) Otherwise, find first balanced { or [ in the whole text
    if candidate is None:
        obj = _extract_first_balanced(text, "{", "}")
        arr = _extract_first_balanced(text, "[", "]")
        # Prefer whichever appears first
        positions = []
        if obj is not None:
            positions.append((text.find(obj), obj))
        if arr is not None:
            positions.append((text.find(arr), arr))
        if not positions:
            return None
        candidate = min(positions, key=lambda p: p[0])[1]

    # 3) Even inside a fence, candidate might start with [ or {. Try both.
    if candidate.lstrip().startswith("["):
        try:
            arr = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if not isinstance(arr, list):
            return None
        return {"objects": arr} if kind == "extract" else {"matches": arr}

    if candidate.lstrip().startswith("{"):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            # Try extracting innermost balanced object inside the fence
            inner = _extract_first_balanced(candidate, "{", "}")
            if inner is None:
                return None
            try:
                obj = json.loads(inner)
            except json.JSONDecodeError:
                return None
        if not isinstance(obj, dict):
            return None
        # If keys are missing, but we see top-level "objects" / "matches" we
        # accept as-is. Otherwise leave the dict untouched.
        return obj

    return None


# ----- file walk ------------------------------------------------------------


def _reparse_file(p: Path, kind: str) -> tuple[bool, bool]:
    """Return (changed, new_parse_ok)."""
    payload = json.loads(p.read_text(encoding="utf-8"))
    raw = payload.get("raw", "")
    new_parsed = tolerant_parse(raw, kind=kind)
    new_ok = new_parsed is not None
    old_ok = payload.get("parse_ok", False)
    changed = (new_ok != old_ok) or (new_parsed != payload.get("parsed"))
    if changed:
        payload["parsed"] = new_parsed
        payload["parse_ok"] = new_ok
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                     encoding="utf-8")
    return changed, new_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DIAG_ROOT))
    args = parser.parse_args()

    root = Path(args.root)
    games = sorted(d for d in root.iterdir() if d.is_dir())
    if not games:
        raise SystemExit(f"no games under {root}")

    total_ex = 0
    total_ex_ok = 0
    total_al = 0
    total_al_ok = 0
    for gdir in games:
        for kind, subdir in (("extract", "extract"), ("align", "align")):
            d = gdir / subdir
            if not d.exists():
                continue
            for fp in sorted(d.glob("*.json")):
                changed, ok = _reparse_file(fp, kind=kind)
                if kind == "extract":
                    total_ex += 1
                    total_ex_ok += int(ok)
                else:
                    total_al += 1
                    total_al_ok += int(ok)
                tag = "[UPDATED]" if changed else "[same]   "
                print(f"  {tag} {fp.relative_to(root)}  -> parse_ok={ok}")

    print()
    print(f"Extract: {total_ex_ok}/{total_ex} parsed "
          f"({(100*total_ex_ok/total_ex if total_ex else 0):.1f}%)")
    print(f"Align:   {total_al_ok}/{total_al} parsed "
          f"({(100*total_al_ok/total_al if total_al else 0):.1f}%)")


if __name__ == "__main__":
    main()
