"""Bench multiple small open-source LLMs on game-content-based spatial probes.

Per docs/project/2026-05-17-v0-model_bench/architecture.md.

Each model:
  1. Loads via HF transformers AutoModelForCausalLM + AutoTokenizer
     (Qwen2.5-VL uses Qwen2_5_VLForConditionalGeneration; we hit the
     LM head text-only).
  2. Applies its tokenizer's chat template.
  3. For each probe: generates ~50 tokens, parses A/B/C/D, scores.

Sequential GPU: load model -> bench -> free -> next model. 4-bit
quantization (bnb-nf4) so even 7B fits T4 16GB.

Output:
  outputs/model_bench_<ts>/
    metrics.json
    per_probe.jsonl
    summary.md
    figures/
      accuracy_by_model.png
      accuracy_by_category.png
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

from arc_agent.bench_probes import get_probes, CATEGORIES


# Model registry. Each entry:
#   id: short key for filenames
#   hf_id: huggingface repo id
#   loader: "qwen_vl" | "causal_lm" — picks the right HF class
#   chat_role: how the system prompt is provided
MODELS: list[dict] = [
    {
        "id": "qwen2_5_vl_3b",
        "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "loader": "qwen_vl",
        "label": "Qwen2.5-VL-3B (current)",
    },
    {
        "id": "phi4_mini_reasoning",
        "hf_id": "microsoft/Phi-4-mini-reasoning",
        "loader": "causal_lm",
        "label": "Phi-4-mini-reasoning (3.8B)",
    },
    {
        "id": "smollm3_3b",
        "hf_id": "HuggingFaceTB/SmolLM3-3B",
        "loader": "causal_lm",
        "label": "SmolLM3-3B",
    },
]


SYSTEM_PROMPT = (
    "You are a careful problem-solver. Read the multiple-choice question, "
    "think step by step internally, and output EXACTLY ONE LINE: 'Answer: X' "
    "where X is A, B, C, or D. Do not output anything else."
)


_ANSWER_RE = re.compile(r"answer[:\s]*([ABCDabcd])", re.IGNORECASE)
_LETTER_RE = re.compile(r"\b([ABCD])\b")


def parse_answer(text: str) -> Optional[str]:
    """Robust answer parser: looks for 'Answer: X' first, then any standalone
    A/B/C/D in the first few lines."""
    if not isinstance(text, str):
        return None
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    # Fallback: first standalone letter in the first 200 chars
    head = text[:200]
    m = _LETTER_RE.search(head)
    if m:
        return m.group(1).upper()
    return None


def format_probe(probe: dict) -> str:
    """User-message text for one probe."""
    lines = [probe["q"], ""]
    for k in ["A", "B", "C", "D"]:
        if k in probe["options"]:
            lines.append(f"{k}) {probe['options'][k]}")
    lines.append("")
    lines.append("Answer with one letter (A, B, C, or D).")
    return "\n".join(lines)


def load_model(entry: dict, dtype: str = "bfloat16"):
    """Returns (model, tokenizer) ready for generate. 4-bit quantized."""
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    hf_id = entry["hf_id"]
    print(f"  loading tokenizer for {hf_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

    if entry["loader"] == "qwen_vl":
        # Use the multimodal class but feed text-only
        from transformers import Qwen2_5_VLForConditionalGeneration
        print(f"  loading Qwen2.5-VL model {hf_id}", flush=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            hf_id, quantization_config=bnb_cfg,
            device_map="auto", trust_remote_code=True,
        )
    else:
        from transformers import AutoModelForCausalLM
        print(f"  loading CausalLM model {hf_id}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=bnb_cfg,
            device_map="auto", trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, system: str, user: str,
                    max_new_tokens: int = 96) -> str:
    """Apply chat template, generate text-only, decode."""
    import torch
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        # Fallback for models without chat template
        prompt_str = f"{system}\n\nUser: {user}\nAssistant:"

    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated part
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def free_model(model) -> None:
    """Best-effort GPU memory free."""
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_bench_one_model(entry: dict, probes: list[dict],
                        max_new_tokens: int = 96) -> dict:
    """Returns a dict with per-probe results + summary."""
    print(f"\n=== {entry['label']} ({entry['hf_id']}) ===", flush=True)
    t0 = time.time()
    try:
        model, tokenizer = load_model(entry)
    except Exception as e:
        print(f"  LOAD FAILED: {e}", flush=True)
        return {
            "model_id": entry["id"], "label": entry["label"],
            "loaded": False, "error": str(e),
            "results": [], "elapsed_s": time.time() - t0,
        }
    load_s = time.time() - t0

    results = []
    for i, p in enumerate(probes):
        user_msg = format_probe(p)
        t_start = time.time()
        try:
            raw = generate_answer(model, tokenizer, SYSTEM_PROMPT, user_msg,
                                  max_new_tokens=max_new_tokens)
            elapsed = time.time() - t_start
        except Exception as e:
            raw = ""
            elapsed = time.time() - t_start
            print(f"  probe {p['id']}: GEN ERROR {e}", flush=True)
        guessed = parse_answer(raw)
        ok = guessed == p["correct"]
        results.append({
            "id": p["id"], "cat": p["cat"],
            "correct": p["correct"],
            "guessed": guessed,
            "ok": ok,
            "raw": raw[:300],
            "elapsed_s": round(elapsed, 2),
        })
        marker = "OK" if ok else "XX"
        print(f"  [{i+1}/{len(probes)}] {p['id']} {marker} "
              f"correct={p['correct']} guessed={guessed} ({elapsed:.1f}s)",
              flush=True)

    free_model(model)
    total = time.time() - t0
    return {
        "model_id": entry["id"], "label": entry["label"],
        "loaded": True, "results": results,
        "load_s": round(load_s, 1),
        "total_s": round(total, 1),
    }


def aggregate(results: list[dict]) -> dict:
    """Compute per-model + per-category accuracy."""
    summary = {"models": {}}
    for r in results:
        mid = r["model_id"]
        if not r.get("loaded"):
            summary["models"][mid] = {
                "label": r["label"], "loaded": False,
                "error": r.get("error", "?"),
            }
            continue
        n = len(r["results"])
        n_correct = sum(1 for x in r["results"] if x["ok"])
        acc = n_correct / max(n, 1)
        by_cat: dict[str, dict] = {}
        for cat in CATEGORIES:
            cat_rows = [x for x in r["results"] if x["cat"] == cat]
            if not cat_rows:
                continue
            by_cat[cat] = {
                "n": len(cat_rows),
                "correct": sum(1 for x in cat_rows if x["ok"]),
                "acc": sum(1 for x in cat_rows if x["ok"]) / len(cat_rows),
            }
        summary["models"][mid] = {
            "label": r["label"], "loaded": True,
            "n_probes": n, "n_correct": n_correct, "accuracy": acc,
            "by_category": by_cat, "load_s": r.get("load_s"),
            "total_s": r.get("total_s"),
        }
    return summary


def write_summary_md(summary: dict, out_path: Path) -> None:
    lines = ["# Model bench summary — game-content-based spatial probes", ""]
    lines.append("## Per-model accuracy")
    lines.append("")
    lines.append("| Model | n_probes | n_correct | accuracy | load (s) | total (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    rows = []
    for mid, m in summary["models"].items():
        if not m.get("loaded"):
            lines.append(f"| {m['label']} | - | - | LOAD FAILED ({m.get('error','?')[:40]}) | - | - |")
            continue
        rows.append((m["label"], m["n_probes"], m["n_correct"], m["accuracy"],
                     m["load_s"], m["total_s"]))
    rows.sort(key=lambda r: -r[3])
    for label, n, nc, acc, ls, ts in rows:
        lines.append(f"| {label} | {n} | {nc} | **{100*acc:.1f}%** | {ls} | {ts} |")
    lines.append("")
    lines.append("## Per-category accuracy (cross-model)")
    lines.append("")
    cats_present = sorted(set(
        c for m in summary["models"].values() if m.get("loaded")
        for c in m["by_category"]
    ))
    header = ["Model"] + cats_present
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for mid, m in summary["models"].items():
        if not m.get("loaded"):
            continue
        row = [m["label"]]
        for c in cats_present:
            d = m["by_category"].get(c, {})
            if not d:
                row.append("-")
            else:
                row.append(f"{int(100*d['acc'])}% ({d['correct']}/{d['n']})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_summary(summary: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. accuracy_by_model bar chart
    loaded_models = [m for m in summary["models"].values() if m.get("loaded")]
    if not loaded_models:
        return
    loaded_models.sort(key=lambda m: -m["accuracy"])
    labels = [m["label"] for m in loaded_models]
    accs = [100 * m["accuracy"] for m in loaded_models]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(labels)), accs, color="#4caf50", alpha=0.9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, fontsize=9)
    ax.set_ylim(0, 100); ax.set_ylabel("Probe accuracy (%)")
    ax.set_title("Model accuracy on game-content-based spatial probes")
    ax.axhline(25, color="r", ls="--", lw=0.5, label="random (25%)")
    ax.legend()
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.0f}%",
                ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_model.png", dpi=120)
    plt.close(fig)

    # 2. per-category heatmap
    cats = sorted(set(c for m in loaded_models for c in m["by_category"]))
    mat = np.full((len(loaded_models), len(cats)), np.nan)
    for i, m in enumerate(loaded_models):
        for j, c in enumerate(cats):
            d = m["by_category"].get(c)
            if d:
                mat[i, j] = 100 * d["acc"]
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats)
    ax.set_yticks(range(len(loaded_models)))
    ax.set_yticklabels([m["label"] for m in loaded_models], fontsize=9)
    for i in range(len(loaded_models)):
        for j in range(len(cats)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{int(mat[i,j])}", ha="center", va="center",
                        color="black", fontsize=9)
    plt.colorbar(im, ax=ax, label="accuracy %")
    ax.set_title("Per-category accuracy by model")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_category.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--models", default="all",
                        help="Comma list of model ids (qwen2_5_vl_3b,phi4_mini_reasoning,smollm3_3b) or 'all'")
    args = parser.parse_args()

    if args.models == "all":
        chosen = MODELS
    else:
        ids = set(args.models.split(","))
        chosen = [m for m in MODELS if m["id"] in ids]
    if not chosen:
        print(f"no models selected from {[m['id'] for m in MODELS]}",
              file=sys.stderr)
        sys.exit(1)

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output) if args.output else (
        REPO / f"outputs/model_bench_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    probes = get_probes()
    print(f"[bench] {len(chosen)} models × {len(probes)} probes", flush=True)
    print(f"[bench] output: {out_dir}", flush=True)

    all_results = []
    for entry in chosen:
        result = run_bench_one_model(entry, probes,
                                      max_new_tokens=args.max_new_tokens)
        all_results.append(result)
        # Dump per-model probe log right away
        per_model_path = out_dir / f"per_probe_{entry['id']}.jsonl"
        with per_model_path.open("w", encoding="utf-8") as f:
            for r in result.get("results", []):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # Save running summary too (in case we crash later)
        summary_so_far = aggregate(all_results)
        (out_dir / "metrics.json").write_text(
            json.dumps(summary_so_far, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    summary = aggregate(all_results)
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_summary_md(summary, out_dir / "summary.md")
    plot_summary(summary, fig_dir)

    print(f"\n[done] {out_dir}", flush=True)
    print(f"  summary: {out_dir / 'summary.md'}", flush=True)
    for mid, m in summary["models"].items():
        if m.get("loaded"):
            print(f"  {m['label']}: {100*m['accuracy']:.1f}% accuracy",
                  flush=True)
        else:
            print(f"  {m['label']}: LOAD FAILED", flush=True)


if __name__ == "__main__":
    main()
