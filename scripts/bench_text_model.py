"""Same 17 questions as bench_spatial_v2.py, but on a PURE-TEXT model
(Qwen2.5-3B-Instruct) instead of Qwen2.5-VL-3B in text-only mode.

This isolates one variable: same family, same size, only difference is
the VL model has a vision tower bolted on. Fair head-to-head on text-only
spatial reasoning.

After this finishes, the merged comparison report combines both runs:
  outputs/spatial_bench_v2/results.json     (VL-3B text-only)
  outputs/spatial_bench_text/results.json   (this run, pure text 3B)
  outputs/spatial_bench_combined/report.md  (merged side-by-side)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Reuse the question set + prompt builders from v2.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from bench_spatial_v2 import (   # noqa: E402
    QUESTIONS, SYSTEM_RAW, SYSTEM_ENRICHED, build_user,
)

TEXT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def _load_text_model():
    """Load Qwen2.5-3B-Instruct in 4-bit. Downloads on first call."""
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading {TEXT_MODEL_ID} (4-bit)...  "
          f"(first run downloads ~6 GB, then quantizes; subsequent runs are fast)")
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        TEXT_MODEL_ID,
        quantization_config=bnb,
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tok


def _generate(model, tok, *, system: str, user: str,
              max_new_tokens: int = 48) -> tuple[str, float]:
    import torch
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tok([text], return_tensors="pt", padding=True).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    dt = time.time() - t0
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    decoded = tok.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip(), dt


def main() -> None:
    t0 = time.time()
    model, tok = _load_text_model()
    print(f"Loaded in {round(time.time()-t0,1)}s. "
          f"Running {len(QUESTIONS)} questions x 2 prompt modes.\n")

    results: list[dict] = []
    for q in QUESTIONS:
        row = {"id": q["id"], "category": q["category"],
               "question": q["question"], "expected": q["expected"]}
        for mode in ("raw", "enriched"):
            user = build_user(q, mode)
            system = SYSTEM_RAW if mode == "raw" else SYSTEM_ENRICHED
            ans, dt = _generate(model, tok, system=system, user=user)
            ok = q["checker"](ans)
            row[mode] = {"answer": ans, "correct": ok, "seconds": round(dt, 2)}
        results.append(row)
        print(f"[{q['id']}:{q['category']}]")
        print(f"  raw      ({row['raw']['seconds']}s): {row['raw']['answer']!r}  "
              f"-> {'OK' if row['raw']['correct'] else 'WRONG'}")
        print(f"  enriched ({row['enriched']['seconds']}s): {row['enriched']['answer']!r}  "
              f"-> {'OK' if row['enriched']['correct'] else 'WRONG'}")

    out_dir = REPO_ROOT / "outputs" / "spatial_bench_text"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    cats = sorted({r["category"] for r in results})
    print("\n" + "=" * 60)
    for cat in cats:
        n = sum(1 for r in results if r["category"] == cat)
        raw_ok = sum(1 for r in results if r["category"] == cat and r["raw"]["correct"])
        enr_ok = sum(1 for r in results if r["category"] == cat and r["enriched"]["correct"])
        print(f"  {cat:12s}  raw={raw_ok}/{n}  enriched={enr_ok}/{n}")
    raw_total = sum(1 for r in results if r["raw"]["correct"])
    enr_total = sum(1 for r in results if r["enriched"]["correct"])
    raw_time = sum(r["raw"]["seconds"] for r in results)
    enr_time = sum(r["enriched"]["seconds"] for r in results)
    print(f"\nOVERALL  raw      = {raw_total}/{len(results)}  total {raw_time:.1f}s")
    print(f"OVERALL  enriched = {enr_total}/{len(results)}  total {enr_time:.1f}s")
    print(f"\nWrote {out_dir / 'results.json'}")
    print(f"Run `scripts/build_text_vs_vlm_report.py` to build the merged comparison.")


if __name__ == "__main__":
    main()
