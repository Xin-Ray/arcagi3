"""Run subtask probes on one or more models.

Per docs/project/2026-05-17-v0-subtask_decomp/architecture.md.

Outputs to outputs/subtask_<subtask>_<ts>/.

Usage:
    .venv/Scripts/python.exe scripts/bench_subtask.py
        --subtask T-NAV-1
        --models qwen,smollm3-cot
        --n-probes 100
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from arc_agent.subtask_probes import GENERATORS


MODEL_REGISTRY: dict[str, dict] = {
    "qwen": {
        "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "loader": "qwen_vl",
        "label": "Qwen2.5-VL-3B",
        "reasoning_mode": "auto",
    },
    "smollm3-cot": {
        "hf_id": "HuggingFaceTB/SmolLM3-3B",
        "loader": "causal_lm",
        "label": "SmolLM3-3B (CoT)",
        "reasoning_mode": "cot",
    },
    "smollm3-nothink": {
        "hf_id": "HuggingFaceTB/SmolLM3-3B",
        "loader": "causal_lm",
        "label": "SmolLM3-3B (/no_think)",
        "reasoning_mode": "no_think",
    },
}


SYSTEM_PROMPT = (
    "You are a careful problem-solver. Read the multiple-choice question, "
    "think it through, and end your reply with EXACTLY ONE LINE: "
    "'Answer: X' where X is A, B, C, or D."
)

_ANSWER_RE = re.compile(r"answer[:\s]*([ABCDabcd])", re.IGNORECASE)
_LETTER_RE = re.compile(r"\b([ABCD])\b")


def parse_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    tail = text[-200:]
    m = _LETTER_RE.search(tail)
    if m:
        return m.group(1).upper()
    return None


def format_probe(probe: dict) -> str:
    lines = [probe["q"], ""]
    for k in ["A", "B", "C", "D"]:
        if k in probe["options"]:
            lines.append(f"{k}) {probe['options'][k]}")
    lines.append("")
    lines.append("Answer with one letter (A, B, C, or D).")
    return "\n".join(lines)


def load_model(entry: dict):
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    hf_id = entry["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

    if entry["loader"] == "qwen_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_id, quantization_config=bnb_cfg,
            device_map="auto", trust_remote_code=True,
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=bnb_cfg,
            device_map="auto", trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer


def free_model(model):
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate(model, tokenizer, system: str, user: str, entry: dict,
             max_new_tokens: int = 512) -> str:
    import torch
    sys_text = system
    rm = entry.get("reasoning_mode", "auto")
    if "SmolLM3" in entry["hf_id"]:
        if rm == "no_think" and "/no_think" not in sys_text:
            sys_text = (system + "\n/no_think").strip() if system else "/no_think"
        elif rm == "cot" and "/think" not in sys_text:
            sys_text = (system + "\n/think").strip() if system else "/think"
        elif rm == "auto" and "/no_think" not in sys_text:
            sys_text = (system + "\n/no_think").strip() if system else "/no_think"
    messages = [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": user},
    ]
    try:
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        prompt_str = f"{sys_text}\n\nUser: {user}\nAssistant:"
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def run_one_model(model_key: str, probes: list[dict], max_new_tokens: int) -> dict:
    entry = MODEL_REGISTRY[model_key]
    print(f"\n=== {entry['label']} ({entry['hf_id']}) "
          f"reasoning={entry['reasoning_mode']} ===", flush=True)
    t0 = time.time()
    try:
        model, tokenizer = load_model(entry)
    except Exception as e:
        print(f"  LOAD FAILED: {e}", flush=True)
        return {"model_key": model_key, "loaded": False,
                "error": str(e), "results": []}
    load_s = time.time() - t0
    print(f"  loaded in {load_s:.1f}s", flush=True)

    results = []
    for i, p in enumerate(probes):
        t = time.time()
        try:
            raw = generate(model, tokenizer, SYSTEM_PROMPT, format_probe(p),
                           entry, max_new_tokens=max_new_tokens)
            elapsed = time.time() - t
        except Exception as e:
            raw = ""
            elapsed = time.time() - t
            print(f"  GEN ERROR on {p['id']}: {e}", flush=True)
        guessed = parse_answer(raw)
        ok = guessed == p["correct"]
        results.append({
            "id": p["id"], "correct": p["correct"], "guessed": guessed,
            "ok": ok, "elapsed_s": round(elapsed, 2),
            "raw_tail": raw[-200:] if raw else "",
        })
        if (i + 1) % 10 == 0 or i + 1 == len(probes):
            n_ok = sum(1 for r in results if r["ok"])
            print(f"  [{i+1}/{len(probes)}] "
                  f"running_acc={100*n_ok/(i+1):.1f}% "
                  f"last_elapsed={elapsed:.1f}s", flush=True)

    free_model(model)
    total_s = time.time() - t0
    n_correct = sum(1 for r in results if r["ok"])
    return {
        "model_key": model_key, "label": entry["label"],
        "loaded": True, "results": results,
        "load_s": round(load_s, 1), "total_s": round(total_s, 1),
        "n_correct": n_correct, "n_probes": len(probes),
        "accuracy": n_correct / len(probes) if probes else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", required=True,
                        choices=list(GENERATORS.keys()))
    parser.add_argument("--models", default="smollm3-cot")
    parser.add_argument("--n-probes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output) if args.output else (
        REPO / f"outputs/subtask_{args.subtask}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    probes = GENERATORS[args.subtask](n=args.n_probes, seed=args.seed)
    print(f"[bench] subtask={args.subtask} n_probes={len(probes)}", flush=True)
    print(f"[bench] output: {out_dir}", flush=True)

    requested = args.models.split(",")
    all_results = []
    for m in requested:
        m = m.strip()
        if m not in MODEL_REGISTRY:
            print(f"  unknown model key: {m}", flush=True)
            continue
        r = run_one_model(m, probes, args.max_new_tokens)
        all_results.append(r)
        with (out_dir / f"per_probe_{m}.jsonl").open("w", encoding="utf-8") as f:
            for x in r.get("results", []):
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    summary = {"subtask": args.subtask, "n_probes": len(probes), "models": {}}
    for r in all_results:
        if not r.get("loaded"):
            summary["models"][r["model_key"]] = {"loaded": False,
                                                  "error": r.get("error")}
            continue
        summary["models"][r["model_key"]] = {
            "label": r["label"], "n_correct": r["n_correct"],
            "n_probes": r["n_probes"], "accuracy": r["accuracy"],
            "load_s": r["load_s"], "total_s": r["total_s"],
        }
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [f"# Subtask bench — {args.subtask}", ""]
    lines.append(f"n_probes: {len(probes)} (seed={args.seed})")
    lines.append("")
    lines.append("| Model | accuracy | n_correct | load(s) | total(s) |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in sorted(all_results, key=lambda x: -x.get("accuracy", 0)):
        if not r.get("loaded"):
            continue
        lines.append(f"| {r['label']} | **{100*r['accuracy']:.1f}%** | "
                     f"{r['n_correct']}/{r['n_probes']} | "
                     f"{r['load_s']} | {r['total_s']} |")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[done] {out_dir}", flush=True)
    for r in all_results:
        if r.get("loaded"):
            print(f"  {r['label']}: {100*r['accuracy']:.1f}% accuracy",
                  flush=True)


if __name__ == "__main__":
    main()
