"""One-shot: copy archived reports into outputs/reports/ with rewritten paths.

Reads markdown reports from `archive/outputs_2026-05-14/<dir>/...md`, copies
to `outputs/reports/<date>_<name>.md`, and rewrites all relative
`](path)` references to point back to the original asset location
(`../../archive/outputs_2026-05-14/<dir>/path`).

This is a one-shot script, not part of the live pipeline.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# (source_path, dest_filename, base_dir_inside_archive)
COPIES = [
    ("archive/outputs_2026-05-14/v3_visual_full/report.md",
     "2026-05-14_v3_visual_full.md", "v3_visual_full"),
    ("archive/outputs_2026-05-14/v3_visual_ar25/report.md",
     "2026-05-14_v3_visual_ar25.md", "v3_visual_ar25"),
    ("archive/outputs_2026-05-14/v3_eval_full/comparison.md",
     "2026-05-14_v3_eval_full_comparison.md", "v3_eval_full"),
    ("archive/outputs_2026-05-14/spatial_bench/results.md",
     "2026-05-13_spatial_bench_v1.md", "spatial_bench"),
    ("archive/outputs_2026-05-14/spatial_bench_v2/results.md",
     "2026-05-13_spatial_bench_v2.md", "spatial_bench_v2"),
]

DEST_DIR = REPO_ROOT / "outputs" / "reports"


def rewrite_paths(content: str, base_dir: str) -> str:
    """Rewrite ](relative_path) -> ](../../archive/outputs_2026-05-14/base_dir/relative_path)
    Skip absolute URLs (http, https, /) and paths that already start with ../"""
    prefix = f"../../archive/outputs_2026-05-14/{base_dir}/"

    def repl(m: re.Match) -> str:
        path = m.group(1).strip()
        if path.startswith(("http://", "https://", "/", "../", "#")):
            return m.group(0)
        return f"]({prefix}{path})"

    # Match ](path) markdown link content; skip URLs
    return re.sub(r"\]\(([^)\s]+)\)", repl, content)


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    for src_rel, dst_name, base_dir in COPIES:
        src = REPO_ROOT / src_rel
        if not src.exists():
            print(f"  MISSING: {src_rel} -- skip")
            continue
        content = src.read_text(encoding="utf-8")
        rewritten = rewrite_paths(content, base_dir)
        # Add a header note so reader knows this is a snapshot
        header = (
            f"<!-- COPIED from {src_rel}; paths rewritten to point back to "
            f"archive/outputs_2026-05-14/{base_dir}/ -->\n"
            f"<!-- DO NOT EDIT; edit the original to refresh this copy. -->\n\n"
        )
        dst = DEST_DIR / dst_name
        dst.write_text(header + rewritten, encoding="utf-8")
        print(f"  wrote {dst.relative_to(REPO_ROOT)}")
    print(f"\nDone. {len(COPIES)} reports copied to {DEST_DIR.relative_to(REPO_ROOT)}.")


if __name__ == "__main__":
    main()
