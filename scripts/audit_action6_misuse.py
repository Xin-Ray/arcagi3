"""Deeper audit of ACTION6 and trailing-coords misuse across all games."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN = REPO_ROOT / "outputs" / "v3_visual_full"

ACTION_RE = re.compile(r"\bACTION([1-7])\b", re.IGNORECASE)
TRAILING_COORDS = re.compile(r"\bACTION([1-7])\b\s+\d+\s+\d+")
ACTION6_NO_COORDS = re.compile(r"\bACTION6\b(?!\s+\d+\s+\d+)")


def main() -> None:
    rows_total = 0
    trailing_on_noncoord = Counter()
    action6_no_coords = Counter()
    response_patterns = Counter()

    for gdir in sorted(d for d in RUN.iterdir() if d.is_dir()):
        trace = gdir / "trace.jsonl"
        if not trace.exists():
            continue
        with trace.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line.strip())
                except Exception:
                    continue
                rows_total += 1
                raw = (row.get("response_raw") or "").strip()

                # detect "ACTIONx 12 30" where x is NOT 6
                for m in ACTION_RE.finditer(raw):
                    full_pattern = TRAILING_COORDS.search(raw)
                    if full_pattern and m.group(1) != "6":
                        trailing_on_noncoord[f"ACTION{m.group(1)}"] += 1
                        break

                # ACTION6 without coords
                m = ACTION_RE.search(raw)
                if m and m.group(1) == "6":
                    if not re.search(r"\bACTION6\b\s+\d+\s+\d+", raw, re.IGNORECASE):
                        action6_no_coords[gdir.name] += 1

                # General response shape
                # Normalize: classify into "ACTIONx", "ACTIONx N N", or other
                if m:
                    if re.search(rf"\bACTION{m.group(1)}\b\s+\d+\s+\d+", raw, re.IGNORECASE):
                        response_patterns[f"ACTION{m.group(1)} <coords>"] += 1
                    else:
                        response_patterns[f"ACTION{m.group(1)} only"] += 1
                else:
                    response_patterns["(no ACTION token)"] += 1

    print(f"Total steps audited: {rows_total}")
    print()
    print("=== Response patterns (Qwen's literal output shape) ===")
    for pat, n in response_patterns.most_common():
        print(f"  {pat:<30}  {n:>4}  ({100*n/rows_total:.1f}%)")
    print()
    print("=== ACTION(non-6) WITH trailing coords (illegal format) ===")
    if not trailing_on_noncoord:
        print("  (none — no non-6 actions had coords appended)")
    else:
        for action, n in trailing_on_noncoord.most_common():
            print(f"  {action}: {n} times had trailing coords (should NOT)")
    print()
    print("=== ACTION6 WITHOUT coords (model forgot to add x y) ===")
    if not action6_no_coords:
        print("  (none)")
    else:
        for game, n in action6_no_coords.most_common():
            print(f"  {game}: {n} ACTION6 picks lacked coords -> agent used random (x,y)")


if __name__ == "__main__":
    main()
