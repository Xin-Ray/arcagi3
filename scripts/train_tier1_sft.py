"""LoRA SFT trainer for Tier 1 (per docs/arch_sft_tier1_zh.md §4).

Loads Qwen2.5-VL-3B in 4-bit + bf16 LoRA on q/k/v/o, trains 2 epochs on
the 190k synthetic Tier-1 mix, saves to outputs/finetune/qwen3b-tier1-lora.

Heavy imports (torch / transformers / trl / peft / datasets) are at module
top — this script is **only** invoked when training is actually intended;
no other code should import it.

Usage:
    .venv/Scripts/python.exe scripts/train_tier1_sft.py
    .venv/Scripts/python.exe scripts/train_tier1_sft.py --dry-run     # build trainer, skip train()
    .venv/Scripts/python.exe scripts/train_tier1_sft.py --epochs 1 --lr 1e-4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ── Windows compatibility patches (must run before heavy imports) ─────────
#
# 1) trl 1.3.0 reads bundled jinja chat templates with `Path.read_text()` and
#    no explicit encoding. On Windows that defaults to cp1252 and dies on the
#    UTF-8 chars in `deepseekv3.jinja` etc. (see trl/chat_template_utils.py:309).
#    Force read_text() to UTF-8 if no encoding given.
_orig_read_text = Path.read_text
def _read_text_utf8(self, encoding=None, errors=None, *a, **kw):
    return _orig_read_text(self, encoding=encoding or "utf-8", errors=errors, *a, **kw)
Path.read_text = _read_text_utf8  # type: ignore[assignment]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train-data",
                   default="outputs/finetune/tier1_train.jsonl", type=Path)
    p.add_argument("--out-dir",
                   default="outputs/finetune/qwen3b-tier1-lora", type=Path)
    p.add_argument("--model-path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--per-device-batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="build everything but skip trainer.train()")
    args = p.parse_args()

    train_data = args.train_data if args.train_data.is_absolute() else REPO_ROOT / args.train_data
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    if not train_data.exists():
        raise FileNotFoundError(
            f"train data not found at {train_data}. Run "
            "`python scripts/gen_tier1_data.py` first."
        )

    # ── Lazy heavy imports (only when actually training) ───────────────
    # IMPORTANT: datasets MUST import before torch on Windows. Torch's
    # CUDA init clashes with pyarrow's lazy-loader and causes an access-
    # violation segfault during `from datasets import Dataset` if torch
    # was loaded first. Verified locally 2026-05-15.
    from datasets import Dataset
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )
    from trl import SFTConfig, SFTTrainer

    print(f"[train_tier1] model={args.model_path} train_data={train_data}", flush=True)
    print(f"[train_tier1] out_dir={out_dir}  dry_run={args.dry_run}", flush=True)

    # ── load tokenizer/processor and pre-format the chat string ─────────
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def _to_text(row: dict) -> dict:
        msgs = []
        if row.get("system"):
            msgs.append({"role": "system", "content": row["system"]})
        msgs.append({"role": "user", "content": row["user"]})
        msgs.append({"role": "assistant", "content": row["assistant"]})
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    print(f"[train_tier1] loading + formatting jsonl ...", flush=True)
    rows: list[dict] = []
    with train_data.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    ds = Dataset.from_list(rows)
    ds = ds.map(_to_text, remove_columns=ds.column_names, num_proc=1)
    print(f"[train_tier1] dataset: {len(ds)} rows; sample[0][:200]:",
          ds[0]["text"][:200].replace("\n", " | "), flush=True)

    # ── load 4-bit base + LoRA wrap ────────────────────────────────────
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(f"[train_tier1] loading Qwen2.5-VL in 4-bit ...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Restrict to the language-model attention to avoid touching the
        # frozen vision encoder; the qwen2_5_vl module path always
        # contains `.language_model.` for the LM stack.
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train_tier1] trainable params: {trainable:,} / {total:,} "
          f"({trainable / total:.2%})", flush=True)

    # ── SFTConfig ─────────────────────────────────────────────────────
    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=args.logging_steps,
        max_length=args.max_seq_length,
        report_to=[],
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    eff_batch = args.per_device_batch * args.grad_accum
    print(f"[train_tier1] trainer built. effective_batch={eff_batch} "
          f"steps_per_epoch~={len(ds) // eff_batch}",
          flush=True)

    if args.dry_run:
        print("[train_tier1] --dry-run: skipping trainer.train() and save", flush=True)
        return

    print(f"[train_tier1] starting trainer.train() ...", flush=True)
    trainer.train()
    print(f"[train_tier1] saving LoRA adapter to {out_dir}", flush=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[train_tier1] done.", flush=True)


if __name__ == "__main__":
    main()
