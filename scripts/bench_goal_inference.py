"""Goal-inference benchmark: VL-3B with 2 images vs Text-3B with scipy text.

For each game in `outputs/scipy_object_diag/`, build:
- The action sequence taken in the first 4 steps (from baseline trace.jsonl)
- The 5 scipy object extracts (frame_00 ... frame_04)
- The 4 alignment dicts (pair_00_01 ... pair_03_04)
- 2 PNGs: frame_00 (start) and frame_04 (after 4 actions)

Run two models on each game:
  Mode A: Qwen2.5-VL-3B  + [start image, end image] + action text
  Mode B: Qwen2.5-3B-Instruct  + scipy text only (no images)

Both are asked the same question, expecting strict JSON with:
  primary_goal, evidence, alternatives[2], invariants, dynamics

Outputs:
  outputs/goal_inference/<game>/vl_image.json
  outputs/goal_inference/<game>/text_3b.json
  outputs/goal_inference/report.md       (side-by-side for human review)
"""
from __future__ import annotations

import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


SCIPY_ROOT = REPO_ROOT / "outputs" / "scipy_object_diag"
BASELINE_ROOT = REPO_ROOT / "outputs" / "baseline_20260511_200835"
OUT_ROOT = REPO_ROOT / "outputs" / "goal_inference"
VL_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
TEXT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


# ─── Prompt templates ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are watching the first few frames of an UNFAMILIAR
turn-based grid game. Your job is to HYPOTHESIZE what the goal of the game is.

You DO NOT know:
- What each ACTION does (different games map ACTION1..7 differently)
- What objects mean (player / wall / goal / enemy / etc)
- The win condition

You see:
- The initial frame state
- A sequence of 4 actions taken
- The resulting state after those 4 actions
- (sometimes) Structured object info extracted by a deterministic detector

Output STRICT JSON, no prose, no markdown fences:

{
  "primary_goal": "one sentence describing the most likely goal",
  "evidence": "1-2 sentences citing specific objects or transitions",
  "alternatives": [
    {"hypothesis": "...", "confidence": "low"},
    {"hypothesis": "...", "confidence": "low"}
  ],
  "invariants": ["objects/regions that did not change"],
  "dynamics": ["what changed across the 4 steps"]
}"""


# ─── Data builders ──────────────────────────────────────────────────────────

def _read_actions(game_id: str, n: int = 4) -> list[str]:
    """Read first n chosen_action names from the baseline trace."""
    path = BASELINE_ROOT / game_id / "trace.jsonl"
    actions: list[str] = []
    if not path.exists():
        return actions
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            ca = row.get("chosen_action")
            if ca:
                actions.append(ca)
            if len(actions) >= n:
                break
    return actions


def _read_extract(game_dir: Path, frame_idx: int) -> list[dict]:
    fp = game_dir / "extract" / f"frame_{frame_idx:02d}.json"
    if not fp.exists():
        return []
    return json.loads(fp.read_text(encoding="utf-8"))["parsed"]["objects"]


def _read_align(game_dir: Path, i: int) -> list[dict]:
    fp = game_dir / "align" / f"pair_{i:02d}_{i+1:02d}.json"
    if not fp.exists():
        return []
    return json.loads(fp.read_text(encoding="utf-8"))["parsed"]["matches"]


def _fmt_obj(o: dict) -> str:
    r0, c0, r1, c1 = o["bbox"]
    return (f"id={o['id']} {o['color_name']} size={o['size']} "
            f"bbox=[{r0},{c0},{r1},{c1}] desc={o.get('description','?')}")


def _fmt_align_match(m: dict) -> str:
    t = m["type"]
    if t == "unchanged":
        return f"id={m['before_id']} ({m.get('color','?')}): unchanged"
    if t == "moved":
        d = m.get("delta") or {}
        dy = d.get("dy", 0); dx = d.get("dx", 0)
        dirs = []
        if dy < 0: dirs.append("UP")
        elif dy > 0: dirs.append("DOWN")
        if dx < 0: dirs.append("LEFT")
        elif dx > 0: dirs.append("RIGHT")
        dist = max(abs(dy), abs(dx))
        return (f"id={m['before_id']} -> id={m['after_id']} ({m.get('color','?')}): "
                f"moved {dist} cells {'+'.join(dirs)} (dy={dy:+d}, dx={dx:+d})")
    if t == "reshaped":
        d = m.get("delta") or {}
        return (f"id={m['before_id']} -> id={m['after_id']} ({m.get('color','?')}): "
                f"reshaped (added={d.get('cells_added',0)}, removed={d.get('cells_removed',0)})")
    if t == "recolored":
        d = m.get("delta") or {}
        return (f"id={m['before_id']} -> id={m['after_id']}: recolored "
                f"{d.get('from','?')} -> {d.get('to','?')}")
    if t == "disappeared":
        return f"id={m['before_id']} ({m.get('color','?')}): DISAPPEARED"
    if t == "appeared":
        return f"new id={m['after_id']} ({m.get('color','?')}): APPEARED"
    return f"({t})"


def build_text_only_prompt(game_dir: Path, actions: list[str]) -> str:
    parts = [f"[GAME] {game_dir.name}",
             "[ACTIONS taken in this episode]",
             "  " + ",".join(actions),
             ""]
    for i in range(5):
        objs = _read_extract(game_dir, i)
        parts.append(f"[FRAME {i:02d} objects]")
        if not objs:
            parts.append("  (none extracted)")
        else:
            for o in objs:
                parts.append("  " + _fmt_obj(o))
        parts.append("")
        if i < 4:
            matches = _read_align(game_dir, i)
            parts.append(f"[TRANSITION {i:02d} -> {i+1:02d}  via {actions[i] if i < len(actions) else '?'}]")
            changed = [m for m in matches if m["type"] != "unchanged"]
            if not changed:
                parts.append("  no objects changed (frame stationary)")
            else:
                for m in changed:
                    parts.append("  " + _fmt_align_match(m))
            parts.append("")
    parts.append("[QUESTION]")
    parts.append("Based on these 5 frames and 4 actions, what is the most "
                 "likely GOAL of this game? Use the JSON schema above.")
    return "\n".join(parts)


def build_image_user_text(game_dir: Path, actions: list[str]) -> str:
    """Compact text companion sent alongside the 2 images."""
    parts = [f"[GAME] {game_dir.name}",
             "[ACTIONS taken between the two attached images]",
             "  " + ",".join(actions),
             "",
             "Image 1 = frame_00 (initial state).",
             "Image 2 = frame_04 (after the 4 actions above).",
             "",
             "[QUESTION]",
             "Based on the two images and the 4 actions between them, what is "
             "the most likely GOAL of this game? Use the JSON schema above."]
    return "\n".join(parts)


# ─── Inference helpers ──────────────────────────────────────────────────────

_JSON_OBJ_RE = re.compile(r"\{[\s\S]+\}")


def _tolerant_parse(text: str) -> Any:
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    candidate = fenced.group(1).strip() if fenced else None
    if candidate is None:
        m = _JSON_OBJ_RE.search(text)
        candidate = m.group(0) if m else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _vl_generate(model, processor, *, system: str, user: str,
                 images: list, max_new_tokens: int = 512) -> tuple[str, float]:
    import torch
    from qwen_vl_utils import process_vision_info

    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": user})
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    dt = time.time() - t0
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    return processor.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0].strip(), dt


def _text_generate(model, tok, *, system: str, user: str,
                   max_new_tokens: int = 512) -> tuple[str, float]:
    import torch
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt", padding=True).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    dt = time.time() - t0
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    return tok.batch_decode(trimmed, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)[0].strip(), dt


def _unload_model(*objs):
    import torch
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─── Main run ───────────────────────────────────────────────────────────────


def main() -> None:
    from PIL import Image

    games = sorted(d for d in SCIPY_ROOT.iterdir()
                   if d.is_dir() and (d / "frame_00.png").exists())
    if not games:
        raise SystemExit(f"no game folders under {SCIPY_ROOT}")
    print(f"Games to bench: {[g.name for g in games]}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ───── Pass 1: VL-3B with image ─────────────────────────────────────────
    print("\n[Pass 1] Loading Qwen2.5-VL-3B (4-bit)...")
    from arc_agent.vlm_backbone import load_model
    t0 = time.time()
    vl_model, vl_processor = load_model(quantize="4bit")
    print(f"  loaded in {round(time.time()-t0,1)}s")

    vl_results: dict[str, dict] = {}
    for gdir in games:
        actions = _read_actions(gdir.name)
        img0 = Image.open(gdir / "frame_00.png").convert("RGB")
        img4 = Image.open(gdir / "frame_04.png").convert("RGB")
        user = build_image_user_text(gdir, actions)
        raw, dt = _vl_generate(vl_model, vl_processor,
                               system=SYSTEM_PROMPT, user=user,
                               images=[img0, img4])
        parsed = _tolerant_parse(raw)
        out = {"game": gdir.name, "model": "Qwen2.5-VL-3B + 2 images",
               "actions": actions, "wall_s": round(dt, 2),
               "raw": raw, "parsed": parsed,
               "parse_ok": parsed is not None}
        (OUT_ROOT / gdir.name).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / gdir.name / "vl_image.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        vl_results[gdir.name] = out
        print(f"  [{gdir.name}] VL+image done in {dt:.1f}s, parse_ok={parsed is not None}")

    _unload_model(vl_model, vl_processor)
    print("  Unloaded VL.")

    # ───── Pass 2: Pure-text 3B with scipy text ─────────────────────────────
    print("\n[Pass 2] Loading Qwen2.5-3B-Instruct (4-bit, text)...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
    txt_model = AutoModelForCausalLM.from_pretrained(
        TEXT_MODEL_ID, quantization_config=bnb, torch_dtype=torch.float16)
    txt_model.eval()
    print(f"  loaded in {round(time.time()-t0,1)}s")

    txt_results: dict[str, dict] = {}
    for gdir in games:
        actions = _read_actions(gdir.name)
        user = build_text_only_prompt(gdir, actions)
        raw, dt = _text_generate(txt_model, tok,
                                 system=SYSTEM_PROMPT, user=user)
        parsed = _tolerant_parse(raw)
        out = {"game": gdir.name, "model": "Qwen2.5-3B-Instruct + scipy text",
               "actions": actions, "wall_s": round(dt, 2),
               "raw": raw, "parsed": parsed,
               "parse_ok": parsed is not None}
        (OUT_ROOT / gdir.name / "text_3b.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        txt_results[gdir.name] = out
        print(f"  [{gdir.name}] Text-3B done in {dt:.1f}s, parse_ok={parsed is not None}")

    _unload_model(txt_model, tok)

    # ───── Build report ─────────────────────────────────────────────────────
    print("\n[Pass 3] Building Markdown report...")
    lines = [
        "# Goal-inference bench — VL+image vs Text-3B+scipy",
        "",
        "Same SYSTEM prompt; same 4-action episode prefix. Both asked to output",
        "structured JSON with primary_goal / evidence / alternatives /",
        "invariants / dynamics. **No automatic scoring** — read each side, judge",
        "plausibility yourself.",
        "",
    ]
    for gdir in games:
        v = vl_results[gdir.name]
        t = txt_results[gdir.name]
        lines += [
            f"## {gdir.name}",
            "",
            f"Actions taken: `{','.join(v['actions'])}`",
            "",
            f"Frame_00:  ![frame_00]({gdir.name}/../scipy_object_diag/{gdir.name}/frame_00.png)",
            "",
            f"Frame_04:  ![frame_04]({gdir.name}/../scipy_object_diag/{gdir.name}/frame_04.png)",
            "",
            "### A) Qwen2.5-VL-3B + 2 images",
            "",
            f"wall_clock: {v['wall_s']}s, parse_ok: {v['parse_ok']}",
            "",
        ]
        if v["parsed"]:
            lines.append("```json")
            lines.append(json.dumps(v["parsed"], indent=2, ensure_ascii=False))
            lines.append("```")
        else:
            lines.append("```")
            lines.append(v["raw"][:1500])
            lines.append("```")
        lines += [
            "",
            "### B) Qwen2.5-3B-Instruct + scipy text only",
            "",
            f"wall_clock: {t['wall_s']}s, parse_ok: {t['parse_ok']}",
            "",
        ]
        if t["parsed"]:
            lines.append("```json")
            lines.append(json.dumps(t["parsed"], indent=2, ensure_ascii=False))
            lines.append("```")
        else:
            lines.append("```")
            lines.append(t["raw"][:1500])
            lines.append("```")
        lines += [
            "",
            "**Human judgment:**",
            "- [ ] A's primary_goal is plausible?",
            "- [ ] B's primary_goal is plausible?",
            "- [ ] Which is more useful: A or B?",
            "",
            "---",
            "",
        ]
    (OUT_ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {OUT_ROOT / 'report.md'}")


if __name__ == "__main__":
    main()
