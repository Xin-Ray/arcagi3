"""Evaluate Qwen2.5-VL-3B (base or LoRA) against the Tier 1 SFT suite.

Per docs/arch_sft_tier1_zh.md §5. Runs every suite required by the §6.1
decision gate (T1/T2/T3/T4/T8 holdout + planning probes + regression),
plus T1/T3 OOD as bonus. Output is a single metrics.json per backbone;
`scripts/report_tier1.py` diffs base vs lora to produce the markdown report.

Usage:
    .venv/Scripts/python.exe scripts/eval_tier1.py --backbone base \\
        --out outputs/finetune/base_metrics.json
    .venv/Scripts/python.exe scripts/eval_tier1.py --backbone lora \\
        --lora-path outputs/finetune/qwen3b-tier1-lora \\
        --out outputs/finetune/lora_metrics.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ── planning probes (hard-coded; the 5 cases that motivated this whole tier)

@dataclasses.dataclass(frozen=True)
class PlanningProbe:
    name: str
    src_x: int
    src_y: int
    dst_x: int
    dst_y: int
    step: int = 3


PLANNING_PROBES: tuple[PlanningProbe, ...] = (
    PlanningProbe("probe_dx15_dy30",  src_x=40, src_y=19, dst_x=55, dst_y=49),  # doc probe 2
    PlanningProbe("probe_dx6_dy30",   src_x=49, src_y=19, dst_x=55, dst_y=49),  # doc probe 3
    PlanningProbe("probe_y_only_dy30",src_x=10, src_y=10, dst_x=10, dst_y=40),
    PlanningProbe("probe_x_only_dx30",src_x=10, src_y=10, dst_x=40, dst_y=10),
    PlanningProbe("probe_rightup",    src_x=30, src_y=30, dst_x=45, dst_y=15),
)


def _planning_prompt(p: PlanningProbe) -> tuple[str, str]:
    """Return (system, user) prompts matching test_clean_qwen_planning.py."""
    system = f"""You are playing a turn-based 64x64 grid game.

You must output the action chain needed to move the yellow object to the green target.

Legal actions:
ACTION1 = move UP {p.step} cells
ACTION2 = move DOWN {p.step} cells
ACTION3 = move RIGHT {p.step} cells
ACTION4 = move LEFT {p.step} cells
ACTION5 = Perform Action. Use it exactly once only after the yellow object reaches the green target.

Coordinate system:
- The board is 64x64.
- row = y, from top to bottom.
- col = x, from left to right.

Objects:
1. Yellow movable object center = x={p.src_x}, y={p.src_y}
2. Green target object center = x={p.dst_x}, y={p.dst_y}

Goal:
Move the yellow object so that it reaches the green target object.

Output rules:
- Output only the final answer. Do not explain.
- Do not output more than two lines.

Output format exactly:
TOTAL_ACTIONS=<number>
ACTION_CHAIN=<comma-separated action tokens>"""
    return system, "Plan now."


# ── per-task scoring ─────────────────────────────────────────────────────

_INT_RE = re.compile(r"-?\d+")


def _norm(s: str) -> str:
    return s.strip().splitlines()[0].strip() if s.strip() else ""


def _score_T1(prediction: str, label: str) -> dict[str, float]:
    p = _norm(prediction).upper().rstrip(".!?")
    return {"exact_match": float(p == label.strip().upper())}


def _score_T2(prediction: str, label: str) -> dict[str, float]:
    nums = _INT_RE.findall(prediction)
    if not nums:
        return {"exact_match": 0.0, "abs_err": float("inf")}
    pred = int(nums[0])
    gold = int(label.strip())
    return {
        "exact_match": float(pred == gold),
        "abs_err": float(abs(pred - gold)),
    }


def _score_T2_remainder(prediction: str, label: str) -> dict[str, float]:
    """Label is `完整 Q 余 R`. We check both Q and R appear in prediction."""
    nums = _INT_RE.findall(prediction)
    gold_nums = _INT_RE.findall(label)
    if len(gold_nums) < 2:
        return {"exact_match": 0.0}
    match = len(nums) >= 2 and nums[:2] == gold_nums[:2]
    return {"exact_match": float(match)}


def _score_T2_multi(prediction: str, label: str) -> dict[str, float]:
    """Label `x:nx y:ny`. Predict both numbers in same order."""
    nums = _INT_RE.findall(prediction)
    gold_nums = _INT_RE.findall(label)
    if len(gold_nums) < 2:
        return {"exact_match": 0.0}
    match = len(nums) >= 2 and nums[:2] == gold_nums[:2]
    return {"exact_match": float(match)}


_DIR8 = {"up", "down", "left", "right",
         "left-up", "right-up", "left-down", "right-down"}


def _score_T3(prediction: str, label: str) -> dict[str, float]:
    p = _norm(prediction).lower().rstrip(".!?")
    return {"exact_match": float(p == label.strip().lower() and p in _DIR8)}


def _score_T4(prediction: str, label: str) -> dict[str, float]:
    """Exact match on the assistant string (trimmed)."""
    return {"exact_match": float(prediction.strip() == label.strip())}


_ACTION_TOKEN_RE = re.compile(r"ACTION[1-7]")
_DIR_WORDS = ("up", "down", "left", "right")


def _score_T8(prediction: str, user_prompt: str, label: str) -> dict[str, float]:
    """T8: reasoning direction must match user-prompt's binding AND action token."""
    # ground truth direction from user prompt: "moves objects DIR"
    m = re.search(r"moves objects (UP|DOWN|LEFT|RIGHT)", user_prompt)
    if m is None:
        # Chinese prompt fallback
        for d_en, d_zh in (("UP", "上"), ("DOWN", "下"),
                           ("LEFT", "左"), ("RIGHT", "右")):
            if f"向{d_zh}移动" in user_prompt:
                gold_dir = d_en
                break
        else:
            return {
                "exact_match": 0.0, "reasoning_dir_ok": 0.0,
                "action_token_ok": 0.0, "consistency": 0.0,
            }
    else:
        gold_dir = m.group(1)
    gold_token = next(iter(_ACTION_TOKEN_RE.findall(user_prompt)), None)

    pred_tokens = _ACTION_TOKEN_RE.findall(prediction)
    pred_action_token = pred_tokens[0] if pred_tokens else ""

    # reasoning direction: scan the first line for any direction word
    first_line = prediction.splitlines()[0].lower() if prediction.strip() else ""
    pred_reasoning_dir = ""
    for d in _DIR_WORDS:
        if re.search(rf"\b{d}\b", first_line):
            pred_reasoning_dir = d.upper()
            break

    reasoning_ok = pred_reasoning_dir == gold_dir
    action_ok = pred_action_token == gold_token
    return {
        "exact_match": float(prediction.strip() == label.strip()),
        "reasoning_dir_ok": float(reasoning_ok),
        "action_token_ok": float(action_ok),
        "consistency": float(reasoning_ok and action_ok),
    }


def _score_planning(prediction: str, probe: PlanningProbe) -> dict[str, float]:
    """7 binary metrics per §5.2 + final Manhattan distance."""
    dx = probe.dst_x - probe.src_x
    dy = probe.dst_y - probe.src_y
    n_x = abs(dx) // probe.step
    n_y = abs(dy) // probe.step
    expect_dir_x = "ACTION3" if dx > 0 else "ACTION4"
    expect_dir_y = "ACTION2" if dy > 0 else "ACTION1"  # y down = DOWN = ACTION2

    pred_actions = _ACTION_TOKEN_RE.findall(prediction)
    cnt = Counter(pred_actions)

    format_valid = float("TOTAL_ACTIONS=" in prediction and "ACTION_CHAIN=" in prediction)

    has_x_dir = cnt[expect_dir_x] > 0 if n_x > 0 else (
        cnt["ACTION3"] == 0 and cnt["ACTION4"] == 0
    )
    has_y_dir = cnt[expect_dir_y] > 0 if n_y > 0 else (
        cnt["ACTION1"] == 0 and cnt["ACTION2"] == 0
    )

    cnt_x_ok = cnt[expect_dir_x] == n_x if n_x > 0 else (
        cnt["ACTION3"] == 0 and cnt["ACTION4"] == 0
    )
    cnt_y_ok = cnt[expect_dir_y] == n_y if n_y > 0 else (
        cnt["ACTION1"] == 0 and cnt["ACTION2"] == 0
    )

    terminator_ok = pred_actions and pred_actions[-1] == "ACTION5"

    # final Manhattan distance after replaying the chain (excluding ACTION5)
    cur_x, cur_y = probe.src_x, probe.src_y
    for a in pred_actions:
        if a == "ACTION1": cur_y -= probe.step
        elif a == "ACTION2": cur_y += probe.step
        elif a == "ACTION3": cur_x += probe.step
        elif a == "ACTION4": cur_x -= probe.step
        # ignore ACTION5/6/7 for replay
    final_distance = abs(cur_x - probe.dst_x) + abs(cur_y - probe.dst_y)

    return {
        "format_valid":       format_valid,
        "direction_x_correct": float(has_x_dir),
        "direction_y_correct": float(has_y_dir),
        "count_x_correct":    float(cnt_x_ok),
        "count_y_correct":    float(cnt_y_ok),
        "terminator_correct": float(terminator_ok),
        "final_distance":     float(final_distance),
    }


# ── regression eval helpers ───────────────────────────────────────────────

_CHOICE_RE = re.compile(r"\b([ABCD])\b")
_BOXED_RE = re.compile(r"\\boxed\{(-?\d+)\}")


def _score_choice(prediction: str, gold: str) -> float:
    """MMLU / CMMLU: pull the first standalone A/B/C/D from output."""
    m = _CHOICE_RE.search(prediction.upper())
    if m is None:
        return 0.0
    return float(m.group(1) == gold.strip().upper())


def _score_gsm8k(prediction: str, gold: str) -> float:
    """GSM8K: prefer \\boxed{N}, else last integer in output."""
    m = _BOXED_RE.search(prediction)
    if m is not None:
        return float(m.group(1) == gold.strip())
    nums = _INT_RE.findall(prediction)
    if not nums:
        return 0.0
    return float(nums[-1] == gold.strip())


# ── max_new_tokens per task ───────────────────────────────────────────────

_MAX_TOKENS: dict[str, int] = {
    "T1": 8, "T2": 16, "T2_remainder": 24, "T2_multi": 24,
    "T3": 16, "T4": 96, "T8": 64,
    "T1_ood": 8, "T2_ood": 16, "T3_ood": 16, "T4_ood": 32, "T8_ood": 64,
    "planning": 256,
    "regression": 256,
}


# ── eval driver ──────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _bucket_by_task(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["task"]].append(r)
    return out


def _run_holdout_suite(
    backbone: Any,
    rows: list[dict],
    task_tag: str,
    *,
    cap: int,
    scorer: Callable[..., dict[str, float]],
    needs_user_prompt: bool = False,
) -> dict[str, Any]:
    rows = rows[:cap]
    scores: list[dict[str, float]] = []
    for r in rows:
        out = backbone.generate(
            None, r["user"], system=r.get("system", ""),
            max_new_tokens=_MAX_TOKENS.get(task_tag, 64),
            temperature=0.0,
        )
        if needs_user_prompt:
            scores.append(scorer(out, r["user"], r["assistant"]))
        else:
            scores.append(scorer(out, r["assistant"]))
    agg: dict[str, float] = {}
    if scores:
        keys = set().union(*(s.keys() for s in scores))
        for k in keys:
            vals = [s[k] for s in scores if k in s and s[k] != float("inf")]
            agg[k] = sum(vals) / len(vals) if vals else 0.0
    return {"n": len(rows), "metrics": agg}


def _run_planning_probes(backbone: Any) -> dict[str, Any]:
    results: dict[str, dict[str, float]] = {}
    for probe in PLANNING_PROBES:
        system, user = _planning_prompt(probe)
        out = backbone.generate(
            None, user, system=system,
            max_new_tokens=_MAX_TOKENS["planning"], temperature=0.0,
        )
        results[probe.name] = _score_planning(out, probe)
        results[probe.name]["_raw"] = out[:500]  # keep a snippet for debugging
    # aggregate
    binaries = ("format_valid", "direction_x_correct", "direction_y_correct",
                "count_x_correct", "count_y_correct", "terminator_correct")
    agg = {
        k: sum(r[k] for r in results.values()) / len(results)
        for k in binaries
    }
    agg["mean_final_distance"] = (
        sum(r["final_distance"] for r in results.values()) / len(results)
    )
    return {"per_probe": results, "aggregate": agg}


def _run_regression(backbone: Any, reg_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, scorer in (
        ("mmlu_mini",   _score_choice),
        ("gsm8k_mini",  _score_gsm8k),
        ("zh_qa_mini",  _score_choice),
    ):
        path = reg_dir / f"{name}.jsonl"
        if not path.exists():
            out[name] = {"n": 0, "accuracy": None, "note": f"missing: {path}"}
            continue
        rows = _load_jsonl(path)
        scores: list[float] = []
        for r in rows:
            gen = backbone.generate(
                None, r["prompt"], system="",
                max_new_tokens=_MAX_TOKENS["regression"], temperature=0.0,
            )
            scores.append(scorer(gen, r["answer"]))
        out[name] = {
            "n": len(rows),
            "accuracy": sum(scores) / len(scores) if scores else 0.0,
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backbone", choices=["base", "lora"], required=True)
    p.add_argument("--lora-path", default=None,
                   help="required when --backbone lora")
    p.add_argument("--model-path", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--quantize", default="4bit",
                   choices=["4bit", "8bit", "none"])
    p.add_argument("--holdout", default="outputs/finetune/tier1_holdout.jsonl",
                   type=Path)
    p.add_argument("--ood",     default="outputs/finetune/tier1_ood.jsonl",
                   type=Path)
    p.add_argument("--regression-dir", default="data/regression", type=Path)
    p.add_argument("--cap-per-suite", type=int, default=500,
                   help="max samples per holdout/OOD suite (full = 0)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--skip-planning", action="store_true")
    p.add_argument("--skip-regression", action="store_true")
    args = p.parse_args()

    if args.backbone == "lora" and not args.lora_path:
        p.error("--lora-path required when --backbone lora")

    holdout_path = args.holdout if args.holdout.is_absolute() else REPO_ROOT / args.holdout
    ood_path = args.ood if args.ood.is_absolute() else REPO_ROOT / args.ood
    reg_dir = args.regression_dir if args.regression_dir.is_absolute() else REPO_ROOT / args.regression_dir
    out_path = args.out if args.out.is_absolute() else REPO_ROOT / args.out

    print(f"[eval_tier1] loading backbone={args.backbone} ...", flush=True)
    from arc_agent.vlm_backbone import HFBackbone
    quantize = None if args.quantize == "none" else args.quantize
    backbone = HFBackbone.load(
        model_path=args.model_path,
        quantize=quantize,
        lora_path=args.lora_path,
    )
    print(f"[eval_tier1] backbone loaded", flush=True)

    cap = args.cap_per_suite if args.cap_per_suite > 0 else 10**9

    print(f"[eval_tier1] loading holdout from {holdout_path}", flush=True)
    holdout = _bucket_by_task(_load_jsonl(holdout_path))
    print(f"[eval_tier1] loading ood from {ood_path}", flush=True)
    ood = _bucket_by_task(_load_jsonl(ood_path))

    suites: dict[str, Any] = {}

    # In-distribution holdout
    suite_specs = [
        ("T1",            holdout.get("T1", []),            _score_T1, False),
        ("T2",            holdout.get("T2", []),            _score_T2, False),
        ("T2_remainder",  holdout.get("T2_remainder", []),  _score_T2_remainder, False),
        ("T2_multi",      holdout.get("T2_multi", []),      _score_T2_multi, False),
        ("T3",            holdout.get("T3", []),            _score_T3, False),
        ("T4",            holdout.get("T4", []),            _score_T4, False),
        ("T8",            holdout.get("T8", []),            _score_T8, True),
        ("T1_ood",        ood.get("T1_ood", []),            _score_T1, False),
        ("T2_ood",        ood.get("T2_ood", []),            _score_T2, False),
        ("T3_ood",        ood.get("T3_ood", []),            _score_T3, False),
        ("T4_ood",        ood.get("T4_ood", []),            _score_T4, False),
        ("T8_ood",        ood.get("T8_ood", []),            _score_T8, True),
    ]
    for tag, rows, scorer, needs_user in suite_specs:
        if not rows:
            print(f"  [skip] {tag}: no rows", flush=True)
            continue
        t0 = time.time()
        result = _run_holdout_suite(
            backbone, rows, tag, cap=cap, scorer=scorer,
            needs_user_prompt=needs_user,
        )
        dt = time.time() - t0
        em = result["metrics"].get("exact_match", float("nan"))
        print(f"  [{tag}] n={result['n']} exact_match={em:.3f} ({dt:.1f}s)",
              flush=True)
        suites[tag] = result

    if not args.skip_planning:
        print("[eval_tier1] planning probes...", flush=True)
        t0 = time.time()
        suites["planning_probes"] = _run_planning_probes(backbone)
        agg = suites["planning_probes"]["aggregate"]
        print(f"  [planning] y_dir={agg['direction_y_correct']:.2f} "
              f"y_cnt={agg['count_y_correct']:.2f} "
              f"format={agg['format_valid']:.2f} "
              f"final_dist={agg['mean_final_distance']:.1f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    if not args.skip_regression:
        print(f"[eval_tier1] regression from {reg_dir} ...", flush=True)
        t0 = time.time()
        suites["regression"] = _run_regression(backbone, reg_dir)
        for name, res in suites["regression"].items():
            acc = res.get("accuracy")
            print(f"  [{name}] n={res['n']} acc={acc}", flush=True)
        print(f"  regression total {time.time()-t0:.1f}s", flush=True)

    payload: dict[str, Any] = {
        "backbone": args.backbone,
        "lora_path": args.lora_path,
        "model_path": args.model_path,
        "quantize": args.quantize,
        "cap_per_suite": args.cap_per_suite,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "suites": suites,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[eval_tier1] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
