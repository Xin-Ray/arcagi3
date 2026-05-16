"""Resume the text-3B half of bench_goal_inference for games that OOM'd.

Cap object listings per frame at K largest objects + summary line for the
rest. This handles bp35's 191-object pathology without losing relevant
signal — large objects are the load-bearing ones.

Only runs games where outputs/goal_inference/<game>/text_3b.json is
missing or didn't parse.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Reuse infrastructure from the main bench.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from bench_goal_inference import (    # noqa: E402
    SYSTEM_PROMPT, SCIPY_ROOT, OUT_ROOT, TEXT_MODEL_ID,
    _read_actions, _read_extract, _read_align,
    _fmt_obj, _fmt_align_match,
    _tolerant_parse, _text_generate, _unload_model,
)

OBJ_CAP = 12


def build_text_prompt_capped(game_dir: Path, actions: list[str]) -> str:
    parts = [f"[GAME] {game_dir.name}",
             "[ACTIONS taken in this episode]",
             "  " + ",".join(actions),
             ""]
    for i in range(5):
        objs = _read_extract(game_dir, i)
        # Sort by size descending; keep largest OBJ_CAP, summarize rest
        objs_sorted = sorted(objs, key=lambda o: -o["size"])
        kept = objs_sorted[:OBJ_CAP]
        rest = objs_sorted[OBJ_CAP:]
        parts.append(f"[FRAME {i:02d} objects]  (showing {len(kept)} largest "
                     f"of {len(objs)} total)")
        for o in kept:
            parts.append("  " + _fmt_obj(o))
        if rest:
            # group rest by color and size buckets
            by_color: dict[str, list[dict]] = {}
            for o in rest:
                by_color.setdefault(o.get("color_name", "?"), []).append(o)
            for color, items in by_color.items():
                total_cells = sum(o["size"] for o in items)
                parts.append(f"  ... and {len(items)} more {color} objects "
                             f"(total {total_cells} cells, mostly background)")
        parts.append("")
        if i < 4:
            matches = _read_align(game_dir, i)
            changed = [m for m in matches if m["type"] != "unchanged"]
            parts.append(f"[TRANSITION {i:02d} -> {i+1:02d}  via "
                         f"{actions[i] if i < len(actions) else '?'}]")
            if not changed:
                parts.append("  no objects changed (frame stationary)")
            else:
                for m in changed[:OBJ_CAP]:
                    parts.append("  " + _fmt_align_match(m))
                if len(changed) > OBJ_CAP:
                    parts.append(f"  ... and {len(changed)-OBJ_CAP} more changes")
            parts.append("")
    parts.append("[QUESTION]")
    parts.append("Based on these 5 frames and 4 actions, what is the most "
                 "likely GOAL of this game? Use the JSON schema above.")
    return "\n".join(parts)


def main() -> None:
    games = sorted(d for d in SCIPY_ROOT.iterdir() if d.is_dir())
    todo = []
    for gdir in games:
        out_path = OUT_ROOT / gdir.name / "text_3b.json"
        if out_path.exists():
            try:
                payload = json.loads(out_path.read_text(encoding="utf-8"))
                if payload.get("parse_ok"):
                    print(f"  [{gdir.name}] already done, skipping")
                    continue
            except Exception:
                pass
        todo.append(gdir)
    if not todo:
        print("nothing to resume")
        return
    print(f"resuming: {[g.name for g in todo]}")

    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    )
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        TEXT_MODEL_ID, quantization_config=bnb, torch_dtype=torch.float16)
    model.eval()
    print(f"loaded in {round(time.time()-t0,1)}s")

    for gdir in todo:
        actions = _read_actions(gdir.name)
        user = build_text_prompt_capped(gdir, actions)
        n_chars = len(user)
        print(f"  [{gdir.name}] prompt size = {n_chars} chars (~{n_chars//4} tokens)")
        raw, dt = _text_generate(model, tok, system=SYSTEM_PROMPT, user=user)
        parsed = _tolerant_parse(raw)
        out = {"game": gdir.name,
               "model": "Qwen2.5-3B-Instruct + scipy text (capped)",
               "actions": actions, "wall_s": round(dt, 2),
               "raw": raw, "parsed": parsed,
               "parse_ok": parsed is not None}
        (OUT_ROOT / gdir.name).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / gdir.name / "text_3b.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  [{gdir.name}] done in {dt:.1f}s, parse_ok={parsed is not None}")
    _unload_model(model, tok)


if __name__ == "__main__":
    main()
