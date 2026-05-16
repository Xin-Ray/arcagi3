"""Larger spatial-reasoning bench: text-only mode only, compare RAW prompt
vs ENRICHED prompt (with human-friendly direction / description fields).

Tests the hypothesis: "adding direction labels to the prompt fixes the
row-decrement = up failure we saw in bench_spatial_reasoning.py Q4."

Both configs run on the same Qwen2.5-VL-3B model in text-only mode (no
image content block). Larger question set (~15+) spans three games
(ar25, cd82, cn04) and three question categories:
  - DIRECTION:    "what direction did X move" — RAW vs ENRICHED should differ here
  - POSITION:     left/right/between/adjacent — both modes have all the info
  - ACTION_PICK:  from observed evidence, pick next action — both modes have it
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ─── Per-game scipy extracts (from the prior runs) ──────────────────────────

AR25_FRAME_01 = [
    {"id": 0, "color": "yellow", "size": 45, "bbox": [15, 36, 23, 44], "center": [19.0, 40.0]},
    {"id": 1, "color": "gray",   "size": 40, "bbox": [15, 18, 23, 26], "center": [19.0, 22.0]},
    {"id": 2, "color": "gray",   "size": 63, "bbox": [63,  0, 63, 62], "center": [63.0, 31.0]},
    {"id": 3, "color": "maroon", "size": 1845,"bbox": [0,  0, 62, 29], "center": [31.0, 14.5]},
    {"id": 4, "color": "maroon", "size": 1800,"bbox": [0, 33, 62, 62], "center": [31.0, 47.5]},
    {"id": 5, "color": "purple", "size": 189, "bbox": [0, 30, 62, 32], "center": [31.0, 31.0]},
    {"id": 6, "color": "tan",    "size": 64,  "bbox": [0, 63, 63, 63], "center": [31.5, 63.0]},
    {"id": 7, "color": "tan",    "size": 45,  "bbox": [45, 51, 53, 59], "center": [49.0, 55.0]},
]

CD82_FRAME_00 = [
    {"id": 0, "color": "red",   "size": 5, "bbox": [10, 3, 12, 5], "center": [11.0, 4.0]},
    {"id": 1, "color": "blue",  "size": 4, "bbox": [20, 30, 21, 31], "center": [20.5, 30.5]},
    {"id": 2, "color": "green", "size": 9, "bbox": [40, 40, 42, 42], "center": [41.0, 41.0]},
    {"id": 3, "color": "yellow","size": 6, "bbox": [5, 50, 7, 52], "center": [6.0, 51.0]},
]

CN04_FRAME_00 = [
    {"id": 0, "color": "light blue", "size": 25, "bbox": [11, 8, 15, 12], "center": [13.0, 10.0]},
    {"id": 1, "color": "light blue", "size": 25, "bbox": [11, 14, 15, 18], "center": [13.0, 16.0]},
    {"id": 2, "color": "purple", "size": 15, "bbox": [30, 20, 34, 22], "center": [32.0, 21.0]},
    {"id": 3, "color": "red", "size": 1, "bbox": [40, 40, 40, 40], "center": [40.0, 40.0]},
]


def _direction_from_delta(dy: int, dx: int) -> str:
    """Human-friendly direction label."""
    parts = []
    if dy < 0:
        parts.append("up")
    elif dy > 0:
        parts.append("down")
    if dx < 0:
        parts.append("left")
    elif dx > 0:
        parts.append("right")
    if not parts:
        return "stationary"
    return "+".join(parts)


# ─── Prompt builders (RAW vs ENRICHED) ──────────────────────────────────────

def fmt_objects_raw(objs: list[dict]) -> str:
    return "\n".join(
        f"  id={o['id']}: {o['color']} (size={o['size']}, "
        f"bbox=[{o['bbox'][0]},{o['bbox'][1]},{o['bbox'][2]},{o['bbox'][3]}], "
        f"center=[{o['center'][0]},{o['center'][1]}])"
        for o in objs
    )


def fmt_objects_enriched(objs: list[dict]) -> str:
    out = []
    for o in objs:
        r0, c0, r1, c1 = o["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        out.append(
            f"  id={o['id']}: color={o['color']}, size={o['size']} cells, "
            f"shape={h}x{w} box, "
            f"top-left=({r0},{c0}), bottom-right=({r1},{c1}), "
            f"center=({o['center'][0]:.1f},{o['center'][1]:.1f})"
        )
    return "\n".join(out)


def fmt_move_raw(before_center: tuple[float, float],
                 after_center: tuple[float, float]) -> str:
    dy = after_center[0] - before_center[0]
    dx = after_center[1] - before_center[1]
    return (f"object moved: center {before_center} -> {after_center}, "
            f"delta dy={dy:+.1f}, dx={dx:+.1f}")


def fmt_move_enriched(before_center: tuple[float, float],
                      after_center: tuple[float, float]) -> str:
    dy = int(round(after_center[0] - before_center[0]))
    dx = int(round(after_center[1] - before_center[1]))
    direction = _direction_from_delta(dy, dx)
    distance = max(abs(dy), abs(dx))
    return (f"object moved {distance} cell(s) {direction.upper()} "
            f"(dy={dy:+d}, dx={dx:+d}; "
            f"center {before_center} -> {after_center})")


SYSTEM_RAW = """You are answering questions about a 64x64 grid game frame.
You are given a structured list of objects with their colors, sizes, bounding
boxes, and centers. The coordinate system is (row, col), with row 0 at the
TOP of the grid and increasing DOWNWARD. col 0 is the LEFT and increases
RIGHTWARD.

Answer each question CONCISELY and DIRECTLY. No prose, no explanation."""

SYSTEM_ENRICHED = """You are answering questions about a 64x64 grid game frame.
You are given a structured list of objects with their colors, sizes, bounding
boxes, and centers. Coordinates are (row, col), row 0 = TOP, col 0 = LEFT.

When a movement is described, the relative direction has ALREADY been pre-computed
for you (UP/DOWN/LEFT/RIGHT). Trust the labeled direction; do not re-derive it
from dy/dx yourself.

Answer each question CONCISELY and DIRECTLY. No prose."""


# ─── Question set ───────────────────────────────────────────────────────────

def _ck(needles):
    def f(reply): return all(n.lower() in reply.lower() for n in needles)
    return f


def _ck_any(needles):
    def f(reply): return any(n.lower() in reply.lower() for n in needles)
    return f


QUESTIONS = [
    # ─── DIRECTION questions (the v2 fix should help here) ──
    {
        "id": "D1_up",
        "category": "DIRECTION",
        "objects": AR25_FRAME_01,
        "context_raw": fmt_move_raw((22.0, 40.0), (19.0, 40.0)),
        "context_enriched": fmt_move_enriched((22.0, 40.0), (19.0, 40.0)),
        "question": "What direction did the object move? Answer one word only: up, down, left, or right.",
        "expected": "up", "checker": _ck(["up"]),
    },
    {
        "id": "D2_down",
        "category": "DIRECTION",
        "objects": AR25_FRAME_01,
        "context_raw": fmt_move_raw((10.0, 10.0), (15.0, 10.0)),
        "context_enriched": fmt_move_enriched((10.0, 10.0), (15.0, 10.0)),
        "question": "What direction did the object move? Answer one word only.",
        "expected": "down", "checker": _ck(["down"]),
    },
    {
        "id": "D3_left",
        "category": "DIRECTION",
        "objects": AR25_FRAME_01,
        "context_raw": fmt_move_raw((30.0, 50.0), (30.0, 45.0)),
        "context_enriched": fmt_move_enriched((30.0, 50.0), (30.0, 45.0)),
        "question": "What direction did the object move? Answer one word only.",
        "expected": "left", "checker": _ck(["left"]),
    },
    {
        "id": "D4_right",
        "category": "DIRECTION",
        "objects": AR25_FRAME_01,
        "context_raw": fmt_move_raw((20.0, 5.0), (20.0, 12.0)),
        "context_enriched": fmt_move_enriched((20.0, 5.0), (20.0, 12.0)),
        "question": "What direction did the object move? Answer one word only.",
        "expected": "right", "checker": _ck(["right"]),
    },
    {
        "id": "D5_compound_up_left",
        "category": "DIRECTION",
        "objects": AR25_FRAME_01,
        "context_raw": fmt_move_raw((30.0, 30.0), (25.0, 27.0)),
        "context_enriched": fmt_move_enriched((30.0, 30.0), (25.0, 27.0)),
        "question": "List all directions that describe this move (e.g. 'up and left'). Answer concisely.",
        "expected": "up + left", "checker": _ck(["up", "left"]),
    },

    # ─── ACTION_PICK questions (combine direction + action history) ──
    {
        "id": "A1_pick_up",
        "category": "ACTION",
        "objects": AR25_FRAME_01,
        "context_raw": "Last step: ACTION1 was taken. Object center went (22.0, 40.0) -> (19.0, 40.0).",
        "context_enriched": ("Last step: ACTION1 was taken. The yellow object moved 3 cells UP "
                             "(dy=-3, dx=0; center (22.0,40.0)->(19.0,40.0))."),
        "question": "You want to move the yellow object further UP by another 3 cells. Which action token?",
        "expected": "ACTION1", "checker": _ck(["action1"]),
    },
    {
        "id": "A2_pick_opposite",
        "category": "ACTION",
        "objects": AR25_FRAME_01,
        "context_raw": "Observation: ACTION3 caused object center (20,20) -> (20,15).",
        "context_enriched": "Observation: ACTION3 caused the object to move 5 cells LEFT (dy=0, dx=-5).",
        "question": "Which action is likely OPPOSITE to ACTION3 (i.e. would move objects right)? Answer in the form ACTION_.",
        "expected": "ACTION4 (or other)",
        "checker": _ck_any(["action4", "action 4", "right"]),
    },
    {
        "id": "A3_dont_repeat_failed",
        "category": "ACTION",
        "objects": AR25_FRAME_01,
        "context_raw": ("History: ACTION2 was tried 3 times in this state; "
                        "frame did not change any of the 3 times."),
        "context_enriched": ("History: ACTION2 was tried 3 times in this state; "
                             "frame did NOT change any of the 3 times (no-op streak)."),
        "question": ("From legal actions {ACTION1, ACTION2, ACTION3, ACTION4}, "
                     "which one should you try NEXT to gather new evidence? Answer one token."),
        "expected": "any of ACTION1/3/4",
        "checker": _ck_any(["action1", "action3", "action4"]),
    },

    # ─── POSITION questions (test left/right/between, both prompts have full data) ──
    {
        "id": "P1_leftmost",
        "category": "POSITION",
        "objects": AR25_FRAME_01,
        "context_raw": "",
        "context_enriched": "",
        "question": ("Among id=0 (yellow), id=1 (gray), id=5 (purple), which is "
                     "the LEFTMOST? Answer with the color name only."),
        "expected": "gray", "checker": _ck(["gray"]),
    },
    {
        "id": "P2_topmost",
        "category": "POSITION",
        "objects": CD82_FRAME_00,
        "context_raw": "",
        "context_enriched": "",
        "question": "Which color is the TOPMOST object (smallest row)? Answer one color name.",
        "expected": "yellow", "checker": _ck(["yellow"]),
    },
    {
        "id": "P3_largest",
        "category": "POSITION",
        "objects": AR25_FRAME_01,
        "context_raw": "",
        "context_enriched": "",
        "question": "Which id has the LARGEST size? Answer with the id number only.",
        "expected": "3", "checker": _ck_any(["id 3", "id=3", " 3"]),
    },
    {
        "id": "P4_count_purple",
        "category": "POSITION",
        "objects": AR25_FRAME_01,
        "context_raw": "",
        "context_enriched": "",
        "question": "How many PURPLE objects are in the frame? Answer one integer.",
        "expected": "1", "checker": _ck_any([" 1", "one"]),
    },
    {
        "id": "P5_between",
        "category": "POSITION",
        "objects": AR25_FRAME_01,
        "context_raw": "",
        "context_enriched": "",
        "question": ("Considering only id=0 (yellow), id=1 (gray), and id=5 (purple), "
                     "is purple horizontally between gray and yellow? yes or no."),
        "expected": "yes", "checker": _ck(["yes"]),
    },
    {
        "id": "P6_grid_quadrant_cd82",
        "category": "POSITION",
        "objects": CD82_FRAME_00,
        "context_raw": "",
        "context_enriched": "",
        "question": ("Which quadrant of the 64x64 grid is the GREEN object in? "
                     "Choose one: top-left, top-right, bottom-left, bottom-right. (Quadrants split at row 32 and col 32.)"),
        "expected": "bottom-right",
        "checker": _ck(["bottom", "right"]),
    },
    {
        "id": "P7_count_objects_cn04",
        "category": "POSITION",
        "objects": CN04_FRAME_00,
        "context_raw": "",
        "context_enriched": "",
        "question": "How many DISTINCT objects are in the frame? Answer one integer.",
        "expected": "4", "checker": _ck_any([" 4", "four"]),
    },
    {
        "id": "P8_adjacent_color",
        "category": "POSITION",
        "objects": AR25_FRAME_01,
        "context_raw": "",
        "context_enriched": "",
        "question": ("Considering only id=0 (yellow), id=1 (gray), id=5 (purple): "
                     "which has its center closest to row 19?"
                     " Answer with the color name."),
        "expected": "yellow or gray (both at row 19)",
        "checker": _ck_any(["yellow", "gray"]),
    },

    # ─── COMPOUND questions (combine position + action) ──
    {
        "id": "C1_push_yellow_to_wall",
        "category": "COMPOUND",
        "objects": AR25_FRAME_01,
        "context_raw": ("History: ACTION3 moves objects LEFT by 3 cells per step "
                        "(observed once); ACTION4 moves them RIGHT."),
        "context_enriched": ("History: ACTION3 moves objects LEFT by 3 cells per step "
                             "(observed once); ACTION4 moves them RIGHT."),
        "question": ("Yellow id=0 is at center col 40. Purple wall id=5 is "
                     "at center col 31. Which action token will push yellow "
                     "TOWARD the purple wall?"),
        "expected": "ACTION3",
        "checker": _ck(["action3"]),
    },
]


def _load_backbone():
    from arc_agent.vlm_backbone import load_model
    return load_model(quantize="4bit")


def _generate_text(model, processor, *, system: str, user: str,
                   max_new_tokens: int = 48) -> tuple[str, float]:
    import torch
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(text=[text], padding=True,
                       return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False)
    dt = time.time() - t0
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip(), dt


def build_user(q: dict, mode: str) -> str:
    fmt = fmt_objects_raw if mode == "raw" else fmt_objects_enriched
    obj_block = fmt(q["objects"])
    ctx_field = "context_raw" if mode == "raw" else "context_enriched"
    ctx = q.get(ctx_field, "") or ""
    parts = [f"[OBJECTS]\n{obj_block}"]
    if ctx:
        parts.append(f"[OBSERVATION]\n{ctx}")
    parts.append(f"[QUESTION]\n{q['question']}")
    return "\n\n".join(parts)


def main() -> None:
    print("Loading Qwen2.5-VL-3B (4-bit) for text-only inference...")
    t0 = time.time()
    model, processor = _load_backbone()
    print(f"Loaded in {round(time.time()-t0,1)}s. Running {len(QUESTIONS)} questions x 2 modes.\n")

    results: list[dict] = []
    for q in QUESTIONS:
        row = {"id": q["id"], "category": q["category"],
               "question": q["question"], "expected": q["expected"]}
        for mode in ("raw", "enriched"):
            user = build_user(q, mode)
            system = SYSTEM_RAW if mode == "raw" else SYSTEM_ENRICHED
            ans, dt = _generate_text(model, processor, system=system, user=user)
            ok = q["checker"](ans)
            row[mode] = {"answer": ans, "correct": ok, "seconds": round(dt, 2)}
        results.append(row)
        print(f"[{q['id']}:{q['category']}]")
        print(f"  raw      ({row['raw']['seconds']}s): {row['raw']['answer']!r}  -> {'OK' if row['raw']['correct'] else 'WRONG'}")
        print(f"  enriched ({row['enriched']['seconds']}s): {row['enriched']['answer']!r}  -> {'OK' if row['enriched']['correct'] else 'WRONG'}")

    # Aggregate by category
    cats = sorted({r["category"] for r in results})
    summary_lines = ["", "=" * 60, "Summary per category:"]
    for cat in cats:
        n = sum(1 for r in results if r["category"] == cat)
        raw_ok = sum(1 for r in results if r["category"] == cat and r["raw"]["correct"])
        enr_ok = sum(1 for r in results if r["category"] == cat and r["enriched"]["correct"])
        summary_lines.append(
            f"  {cat:12s}  raw={raw_ok}/{n}  enriched={enr_ok}/{n}"
        )

    raw_total = sum(1 for r in results if r["raw"]["correct"])
    enr_total = sum(1 for r in results if r["enriched"]["correct"])
    raw_time = sum(r["raw"]["seconds"] for r in results)
    enr_time = sum(r["enriched"]["seconds"] for r in results)
    summary_lines.append("")
    summary_lines.append(f"OVERALL  raw      = {raw_total}/{len(results)}  total {raw_time:.1f}s")
    summary_lines.append(f"OVERALL  enriched = {enr_total}/{len(results)}  total {enr_time:.1f}s")
    print("\n".join(summary_lines))

    out_dir = REPO_ROOT / "outputs" / "spatial_bench_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Markdown report
    lines = [
        "# Spatial reasoning bench v2 — RAW vs ENRICHED prompt (text-only)",
        "",
        f"Model: Qwen2.5-VL-3B (4-bit), text-only mode.  Questions: {len(QUESTIONS)}.",
        "Both configs share the same model and questions; only the prompt text differs.",
        "",
        "## Summary by category",
        "",
        "| Category | n | raw correct | enriched correct |",
        "|---|---|---|---|",
    ]
    for cat in cats:
        n = sum(1 for r in results if r["category"] == cat)
        raw_ok = sum(1 for r in results if r["category"] == cat and r["raw"]["correct"])
        enr_ok = sum(1 for r in results if r["category"] == cat and r["enriched"]["correct"])
        lines.append(f"| **{cat}** | {n} | {raw_ok}/{n} | **{enr_ok}/{n}** |")
    lines += [
        "",
        f"| **OVERALL** | {len(results)} | {raw_total}/{len(results)} | **{enr_total}/{len(results)}** |",
        "",
        f"Wall-clock: raw {raw_time:.1f}s, enriched {enr_time:.1f}s.",
        "",
        "## Per-question",
        "",
    ]
    for r in results:
        lines += [
            f"### {r['id']} ({r['category']})",
            "",
            f"**Q:** {r['question']}",
            "",
            f"**Expected:** `{r['expected']}`",
            "",
            "| Mode | Answer | Correct | Latency |",
            "|---|---|---|---|",
            f"| raw      | `{r['raw']['answer']}` | {'✅' if r['raw']['correct'] else '❌'} | {r['raw']['seconds']}s |",
            f"| enriched | `{r['enriched']['answer']}` | {'✅' if r['enriched']['correct'] else '❌'} | {r['enriched']['seconds']}s |",
            "",
        ]
    (out_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_dir / 'results.md'}")


if __name__ == "__main__":
    main()
