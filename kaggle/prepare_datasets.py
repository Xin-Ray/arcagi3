"""Helpers to prepare the two Kaggle datasets the submission notebook
needs:

  1. xinxiang000/arcagi3-code     — the agent library (this repo, no outputs)
  2. xinxiang000/smollm3-3b-4bit  — pre-quantised SmolLM3-3B weights

Usage:
  .venv/Scripts/python.exe kaggle/prepare_datasets.py code
  .venv/Scripts/python.exe kaggle/prepare_datasets.py model
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACK = REPO / "kaggle" / "_pack"


# ─── code dataset ─────────────────────────────────────────────────────

CODE_INCLUDE = [
    "arc_agent",
    "scripts/run_v3_multi_round.py",
    "tests/test_goal_evaluator.py",
    "requirements.txt",
]
CODE_EXCLUDE_PATTERNS = (
    "__pycache__", ".pytest_cache", ".egg-info",
)


def pack_code():
    out = PACK / "arcagi3-code"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for rel in CODE_INCLUDE:
        src = REPO / rel
        if not src.exists():
            print(f"  skip missing: {rel}")
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(
                src, dst,
                ignore=shutil.ignore_patterns(*CODE_EXCLUDE_PATTERNS),
            )
        else:
            shutil.copy2(src, dst)
        n = sum(1 for _ in dst.rglob("*")) if src.is_dir() else 1
        print(f"  packed: {rel} ({n} files)")

    # Generate Kaggle dataset-metadata.json
    meta = {
        "title": "arcagi3-code",
        "id": "xinxiang000/arcagi3-code",
        "licenses": [{"name": "Apache-2.0"}],
    }
    import json
    (out / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n[done] {out}")
    print("Upload via:")
    print(f"  cd {out}")
    print("  kaggle datasets create -p . --dir-mode tar       # first time, REQUIRED --dir-mode for folders")
    print("  kaggle datasets version -p . -m 'note' --dir-mode tar  # subsequent")


# ─── model dataset ────────────────────────────────────────────────────

def pack_model():
    """Download SmolLM3-3B to local cache, then point user to where to
    copy / upload it. We do NOT pre-quantise here -- Kaggle notebook will
    apply bitsandbytes 4-bit at load time. The dataset just needs the
    raw HF weights.
    """
    out = PACK / "smollm3-3b"
    if out.exists():
        print(f"  {out} already exists -- skipping download")
    else:
        out.mkdir(parents=True)
        print("[init] downloading HuggingFaceTB/SmolLM3-3B ...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="HuggingFaceTB/SmolLM3-3B",
            local_dir=str(out),
            local_dir_use_symlinks=False,
        )

    import json
    meta = {
        "title": "smollm3-3b",
        "id": "xinxiang000/smollm3-3b-4bit",
        "licenses": [{"name": "Apache-2.0"}],
    }
    (out / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n[done] {out}")
    print("Upload via:")
    print(f"  cd {out}")
    print("  kaggle datasets create -p . --dir-mode tar    # large, prefer tar")
    print("  kaggle datasets version -p . -m 'note' --dir-mode tar")


# ─── main ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["code", "model"])
    args = p.parse_args()

    PACK.mkdir(exist_ok=True)
    if args.what == "code":
        pack_code()
    else:
        pack_model()


if __name__ == "__main__":
    main()
