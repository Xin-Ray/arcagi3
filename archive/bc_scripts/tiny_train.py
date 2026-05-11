"""
Step 3 — QLoRA smoke test with animation-aware prompts.

Training prompt now includes:
  - 512x512 PNG image of current grid state
  - Animation path text (direction, distance, n_frames per object)
  - Changed cells count + list (for small diffs)

Usage:
    .venv\\Scripts\\python.exe vlm_test/scripts/tiny_train.py
    .venv\\Scripts\\python.exe vlm_test/scripts/tiny_train.py --epochs 2 --max-samples 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TRAIN_DIR  = ROOT / "vlm_test" / "data" / "train"
JSONL_FILE = TRAIN_DIR / "dataset.jsonl"
OUTPUT_DIR = ROOT / "vlm_test" / "outputs" / "checkpoint"

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

SYSTEM = (
    "You are an AI agent playing a turn-based puzzle game. "
    "You see the current game grid as an image. "
    "The text below the image tells you what happened when you took the last action: "
    "which objects moved, in which direction, how far, and how many animation frames it took. "
    "Use this to build hypotheses about each object's movement rule. "
    "Always end your reply with exactly: ACTION: <action_name>"
)


def load_dataset(max_samples: int) -> list[dict]:
    from PIL import Image
    if not JSONL_FILE.exists():
        raise FileNotFoundError(
            f"{JSONL_FILE} not found.\nRun collect_data.py first."
        )
    rows = []
    with open(JSONL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            img_path = TRAIN_DIR / r["image_path"]
            if not img_path.exists():
                print(f"  warning: missing image {img_path}, skipping")
                continue
            r["pil_image"] = Image.open(img_path).convert("RGB")
            rows.append(r)
            if len(rows) >= max_samples:
                break

    print(f"Loaded {len(rows)} samples from {JSONL_FILE}")

    # print a sample prompt so we can verify animation info is there
    if rows:
        sample = rows[min(5, len(rows)-1)]   # pick a mid-episode step
        print("\n--- sample prompt (step with animation) ---")
        print(sample["prompt"])
        print(f"--- target action: {sample['action']} ---\n")

    return rows


def build_messages(row: dict) -> list[dict]:
    return [
        {"role": "system",    "content": SYSTEM},
        {"role": "user",      "content": [
            {"type": "image", "image": row["pil_image"]},
            {"type": "text",  "text":  row["prompt"]},
        ]},
        {"role": "assistant", "content": f"ACTION: {row['action']}"},
    ]


def train(rows: list[dict], epochs: int, lr: float) -> float:
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from qwen_vl_utils import process_vision_info

    print(f"Loading {MODEL_ID} in 4-bit QLoRA ...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    model.train()
    total_loss = 0.0

    print(f"\nTraining {epochs} epoch(s) on {len(rows)} samples ...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, row in enumerate(rows):
            messages     = build_messages(row)
            input_msgs   = messages[:-1]                        # system + user
            target_text  = messages[-1]["content"]             # "ACTION: ACTION3"

            full_text   = processor.apply_chat_template(
                input_msgs, tokenize=False, add_generation_prompt=True
            ) + target_text

            image_inputs, _ = process_vision_info(input_msgs)
            inputs = processor(
                text=[full_text], images=image_inputs,
                padding=True, return_tensors="pt"
            ).to("cuda")

            # mask prompt tokens from loss — only train on action token
            prompt_text = processor.apply_chat_template(
                input_msgs, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = processor(
                text=[prompt_text], images=image_inputs,
                padding=True, return_tensors="pt"
            ).input_ids
            labels = inputs["input_ids"].clone()
            labels[:, :prompt_ids.shape[1]] = -100

            loss = model(**inputs, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            if (i + 1) % 5 == 0 or (i + 1) == len(rows):
                print(f"  epoch {epoch+1}  step {i+1}/{len(rows)}  loss={loss.item():.4f}")

        avg = epoch_loss / len(rows)
        total_loss += avg
        print(f"  epoch {epoch+1} avg loss: {avg:.4f}")

    # save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"\nLoRA adapter saved -> {OUTPUT_DIR}")

    # quick inference check
    model.eval()
    row = rows[min(5, len(rows)-1)]
    msgs = build_messages(row)[:-1]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs)
    inp  = processor(text=[text], images=imgs, padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=32)
    out = processor.decode(gen[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"\nSample inference check:")
    print(f"  animation in prompt : {'animation' in row['prompt']}")
    print(f"  label               : ACTION: {row['action']}")
    print(f"  model output        : {out!r}")

    return total_loss / epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",      type=int,   default=1)
    ap.add_argument("--max-samples", type=int,   default=60)
    ap.add_argument("--lr",          type=float, default=2e-4)
    args = ap.parse_args()

    rows = load_dataset(args.max_samples)
    if not rows:
        print("No data. Run collect_data.py first."); sys.exit(1)

    # count how many samples actually have animation info
    with_anim = sum(1 for r in rows if r.get("animation_text"))
    print(f"Samples with animation info: {with_anim}/{len(rows)}")

    final_loss = train(rows, args.epochs, args.lr)
    print(f"\nFinal avg loss: {final_loss:.4f}")
    print("Pipeline complete. Scale up with more data + epochs for real training.")


if __name__ == "__main__":
    main()
