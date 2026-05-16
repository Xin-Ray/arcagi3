"""Build the 3 regression mini-sets (committed once, used forever).

Per docs/arch_sft_tier1_zh.md §5.1.1. Pulls public benchmarks from HF
datasets, samples 150 rows each with seed=42, normalizes to a single
`{prompt, answer, source}` schema, writes to data/regression/.

Run once, then commit the 3 jsonl files. After that, eval_tier1.py reads
them directly without ever touching the network.

Usage:
    .venv/Scripts/python.exe scripts/build_regression_mini.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "data" / "regression"
SEED = 42
N_PER_SET = 150


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def _mmlu_prompt(q: str, choices: list[str]) -> str:
    return (
        f"Question: {q}\n"
        f"A) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\n"
        f"Answer with a single letter (A/B/C/D)."
    )


def build_mmlu_mini(rng: random.Random) -> list[dict]:
    """5 subjects × 30 rows each = 150. Uses cais/mmlu test split."""
    from datasets import load_dataset
    subjects = [
        "abstract_algebra",
        "high_school_world_history",
        "professional_psychology",
        "elementary_mathematics",
        "global_facts",
    ]
    rows: list[dict] = []
    for subj in subjects:
        ds = load_dataset("cais/mmlu", subj, split="test")
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        for i in idx[:30]:
            ex = ds[i]
            ans_letter = "ABCD"[int(ex["answer"])]
            rows.append({
                "prompt": _mmlu_prompt(ex["question"], list(ex["choices"])),
                "answer": ans_letter,
                "source": f"mmlu/{subj}",
            })
    rng.shuffle(rows)
    return rows[:N_PER_SET]


def build_gsm8k_mini(rng: random.Random) -> list[dict]:
    """Pull from openai/gsm8k 'main' test split, keep integer answers."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    rows: list[dict] = []
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    for i in idx:
        ex = ds[i]
        # answer field is "<reasoning>\n#### <number>"
        if "####" not in ex["answer"]:
            continue
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        if not gold.lstrip("-").isdigit():
            continue
        prompt = (
            f"Question: {ex['question']}\n"
            f"Answer with a single integer enclosed in \\boxed{{...}}."
        )
        rows.append({
            "prompt": prompt,
            "answer": gold,
            "source": "gsm8k/main",
        })
        if len(rows) >= N_PER_SET:
            break
    return rows


def build_ceval_mini(rng: random.Random) -> list[dict]:
    """Pull 5 subjects × 30 from ceval/ceval-exam val split.

    C-Eval is the standard Chinese MMLU alternative on HF (CMMLU has a
    dataset script that newer `datasets` rejects). Test split is unlabeled
    (held out by authors), so use `val` which has gold answers.
    """
    from datasets import load_dataset
    subjects = [
        "chinese_language_and_literature",
        "high_school_mathematics",
        "modern_chinese_history",
        "ideological_and_moral_cultivation",
        "high_school_geography",
    ]
    rows: list[dict] = []
    for subj in subjects:
        ds = load_dataset("ceval/ceval-exam", subj, split="val")
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        for i in idx[:30]:
            ex = ds[i]
            prompt = (
                f"问题:{ex['question']}\n"
                f"A) {ex['A']}\nB) {ex['B']}\nC) {ex['C']}\nD) {ex['D']}\n"
                f"用一个字母回答(A/B/C/D)。"
            )
            rows.append({
                "prompt": prompt,
                "answer": ex["answer"].strip().upper(),
                "source": f"ceval/{subj}",
            })
    rng.shuffle(rows)
    return rows[:N_PER_SET]


def main() -> None:
    rng = random.Random(SEED)
    print(f"[regression] writing 3 mini sets to {OUT_DIR}/ ...")
    print("[regression] mmlu_mini ...")
    mmlu = build_mmlu_mini(rng)
    _write_jsonl(mmlu, OUT_DIR / "mmlu_mini.jsonl")
    print(f"  wrote {len(mmlu)} rows")
    print("[regression] gsm8k_mini ...")
    gsm = build_gsm8k_mini(rng)
    _write_jsonl(gsm, OUT_DIR / "gsm8k_mini.jsonl")
    print(f"  wrote {len(gsm)} rows")
    print("[regression] zh_qa_mini (ceval) ...")
    zh = build_ceval_mini(rng)
    _write_jsonl(zh, OUT_DIR / "zh_qa_mini.jsonl")
    print(f"  wrote {len(zh)} rows")
    print(f"[regression] done. Commit {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
