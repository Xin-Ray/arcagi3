"""A/B bench: deterministic parser vs LLM judge for goal achievement.

Two datasets:
  1. T-GOAL synthetic probes (100 probes, ground truth known).
     Tests accuracy when both methods are applicable.
  2. v2 round 0 production trace (100 step hypotheses, no labels but
     measurable coverage rate for the parser).

Outputs:
  outputs/bench_goal_judges_<ts>/
    metrics.json     summary
    per_probe.jsonl  per-probe results (parser + judge side by side)
    summary.md       comparison table

Usage:
    .venv/Scripts/python.exe scripts/bench_goal_judges.py
        [--n-probes 100] [--max-new-tokens 1024]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from arc_agent.goal_evaluator import evaluate_goal as parser_eval
from arc_agent.llm_goal_judge import judge_goal as llm_judge
from arc_agent.object_extractor import ObjectRecord
from arc_agent.subtask_probes import gen_T_GOAL


# ── T-GOAL probe -> (hypothesis, objects, ground_truth) ─────────────────

_POS_RE = re.compile(r"row=(\d+),\s*col=(\d+)", re.IGNORECASE)


def _t_goal_probe_to_inputs(probe: dict) -> tuple[str, list[ObjectRecord], bool]:
    """T-GOAL probes are 4-option multi-choice. Extract the canonical
    (hypothesis, current_objects, achieved?) trio from `_meta` + `q`."""
    meta = probe["_meta"]
    pos_A = meta["pos_A"]
    pos_B = meta["pos_B"]
    objs = [
        ObjectRecord(id=1, color=4, color_name="yellow",
                     cells=[(pos_A[0], pos_A[1])],
                     bbox=(pos_A[0], pos_A[1], pos_A[0], pos_A[1]),
                     center=(float(pos_A[0]), float(pos_A[1])), size=1),
        ObjectRecord(id=2, color=4, color_name="yellow",
                     cells=[(pos_B[0], pos_B[1])],
                     bbox=(pos_B[0], pos_B[1], pos_B[0], pos_B[1]),
                     center=(float(pos_B[0]), float(pos_B[1])), size=1),
    ]
    hypothesis = "align the two yellow squares vertically in the left column"
    return hypothesis, objs, bool(meta["success"])


# ── v2 production trace -> (step, hypothesis, objects) ─────────────────

def _load_v2_trace_pairs(round_dir: Path) -> list[tuple[int, str, list[ObjectRecord]]]:
    """Read v2 round 0 knowledge_per_step.jsonl to get (step, hypothesis)
    pairs. We don't have ObjectRecord snapshots in the trace, so we
    approximate by using empty object list — only the PARSER coverage
    rate is measurable here (whether the hypothesis can be parsed at all
    without needing objects)."""
    kp = round_dir / "knowledge_per_step.jsonl"
    if not kp.exists():
        return []
    rows = []
    for line in kp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        h = r["knowledge_after"].get("goal_hypothesis", "")
        if h:
            rows.append((r["step"], h, []))
    return rows


# ── benchmark loop ─────────────────────────────────────────────────────

@dataclass
class _Row:
    source: str         # "t_goal" or "v2_prod"
    probe_id: str
    hypothesis: str
    ground_truth: Optional[bool]
    parser_verdict: Optional[bool]
    parser_pred_kind: str
    parser_elapsed_s: float
    judge_verdict: Optional[bool]
    judge_elapsed_s: float
    judge_raw_tail: str


def _run_parser(hyp: str, objs: list[ObjectRecord]) -> tuple[Optional[bool], str, float]:
    t = time.time()
    verdict, pred = parser_eval(hyp, objs)
    return verdict, (pred.kind if pred is not None else ""), time.time() - t


def _run_judge(hyp: str, objs: list[ObjectRecord], backbone: Any, max_new_tokens: int):
    t = time.time()
    try:
        verdict, raw = llm_judge(hyp, objs, backbone=backbone,
                                 max_new_tokens=max_new_tokens)
    except Exception as e:
        return None, time.time() - t, f"ERROR: {e}"
    return verdict, time.time() - t, raw[-200:] if raw else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-probes", type=int, default=100,
                        help="T-GOAL probes to bench")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--v2-trace", default="outputs/det_goal_force_cot_v2_ar25_2x100_20260518-090129/round_00",
                        help="Round directory with knowledge_per_step.jsonl")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Only run deterministic parser, skip LLM (debug)")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = REPO / f"outputs/bench_goal_judges_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bench] out: {out_dir}", flush=True)

    # ── load LLM backbone (skip if --skip-judge) ───────────────────────
    backbone = None
    if not args.skip_judge:
        from arc_agent.vlm_backbone import make_backbone
        print("[bench] loading SmolLM3-3B (CoT)...", flush=True)
        t0 = time.time()
        backbone = make_backbone("HuggingFaceTB/SmolLM3-3B",
                                  reasoning_mode="cot")
        print(f"[bench]   loaded in {time.time()-t0:.1f}s", flush=True)

    rows: list[_Row] = []

    # ── 1. T-GOAL synthetic probes (ground truth known) ────────────────
    print(f"\n=== T-GOAL probes (n={args.n_probes}) ===", flush=True)
    probes = gen_T_GOAL(n=args.n_probes, seed=42)
    for i, p in enumerate(probes):
        hyp, objs, gt = _t_goal_probe_to_inputs(p)
        p_v, p_kind, p_t = _run_parser(hyp, objs)
        if backbone is not None:
            j_v, j_t, j_raw = _run_judge(hyp, objs, backbone, args.max_new_tokens)
        else:
            j_v, j_t, j_raw = None, 0.0, ""
        rows.append(_Row(
            source="t_goal", probe_id=p["id"], hypothesis=hyp,
            ground_truth=gt,
            parser_verdict=p_v, parser_pred_kind=p_kind, parser_elapsed_s=p_t,
            judge_verdict=j_v, judge_elapsed_s=j_t, judge_raw_tail=j_raw,
        ))
        if (i + 1) % 10 == 0 or i + 1 == len(probes):
            p_correct = sum(1 for r in rows if r.source=="t_goal" and r.parser_verdict==r.ground_truth)
            j_correct = sum(1 for r in rows if r.source=="t_goal" and r.judge_verdict==r.ground_truth)
            print(f"  [{i+1}/{len(probes)}] parser_acc={100*p_correct/(i+1):.0f}% "
                  f"judge_acc={100*j_correct/(i+1):.0f}% "
                  f"last p_t={p_t*1000:.1f}ms j_t={j_t:.1f}s", flush=True)

    # ── 2. v2 production trace (parser coverage only) ──────────────────
    prod_dir = REPO / args.v2_trace
    if prod_dir.exists():
        print(f"\n=== v2 production trace ({prod_dir.name}) ===", flush=True)
        prod_pairs = _load_v2_trace_pairs(prod_dir)
        # dedupe identical hypothesis (we already know there are ~11 distinct
        # transitions across 100 steps)
        seen = set()
        for step, hyp, objs in prod_pairs:
            if hyp in seen:
                continue
            seen.add(hyp)
            p_v, p_kind, p_t = _run_parser(hyp, objs)
            rows.append(_Row(
                source="v2_prod", probe_id=f"v2-step-{step}", hypothesis=hyp,
                ground_truth=None,
                parser_verdict=p_v, parser_pred_kind=p_kind, parser_elapsed_s=p_t,
                judge_verdict=None, judge_elapsed_s=0.0, judge_raw_tail="",
            ))
            print(f"  step {step}: kind={p_kind!r}  verdict={p_v}  "
                  f"hyp={hyp[:60]!r}", flush=True)

    # ── persist + summarize ────────────────────────────────────────────
    with (out_dir / "per_probe.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")

    # Metrics
    tg = [r for r in rows if r.source == "t_goal"]
    vp = [r for r in rows if r.source == "v2_prod"]

    def _acc(rows_, key):
        n = len(rows_); c = sum(1 for r in rows_ if getattr(r, key) == r.ground_truth)
        return (n, c, 100*c/n if n else 0.0)

    def _coverage(rows_, key):
        n = len(rows_); c = sum(1 for r in rows_ if getattr(r, key) is not None)
        return (n, c, 100*c/n if n else 0.0)

    p_n, p_c, p_acc = _acc(tg, "parser_verdict")
    j_n, j_c, j_acc = _acc(tg, "judge_verdict")
    p_n2, p_c2, p_cov = _coverage(tg, "parser_verdict")
    j_n2, j_c2, j_cov = _coverage(tg, "judge_verdict")
    parser_t_avg = sum(r.parser_elapsed_s for r in tg) / max(1, len(tg))
    judge_t_avg = sum(r.judge_elapsed_s for r in tg) / max(1, len(tg))

    metrics = {
        "t_goal": {
            "n": len(tg),
            "parser_acc": p_acc, "parser_n_correct": p_c,
            "judge_acc": j_acc, "judge_n_correct": j_c,
            "parser_coverage": p_cov, "judge_coverage": j_cov,
            "parser_t_avg_ms": round(parser_t_avg * 1000, 2),
            "judge_t_avg_s": round(judge_t_avg, 2),
        },
        "v2_prod": {
            "n_unique_hyps": len(vp),
            "parser_parse_rate": (
                100 * sum(1 for r in vp if r.parser_pred_kind) / max(1, len(vp))
            ),
        }
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Goal Judge A/B Bench",
        "",
        "## T-GOAL synthetic probes (ground truth known)",
        "",
        f"n_probes: {len(tg)}",
        "",
        "| Method | Accuracy | Coverage | Avg time |",
        "|---|---:|---:|---:|",
        f"| Deterministic Python parser | **{p_acc:.0f}%** ({p_c}/{p_n}) | {p_cov:.0f}% | {parser_t_avg*1000:.2f} ms |",
        f"| LLM judge (SmolLM3 CoT)     | **{j_acc:.0f}%** ({j_c}/{j_n}) | {j_cov:.0f}% | {judge_t_avg:.1f} s |",
        "",
        f"Speed ratio: {judge_t_avg / max(parser_t_avg, 1e-6):.0f}x slower (LLM vs Python)",
        "",
        "## v2 production trace (unique hypotheses)",
        "",
        f"n_unique_hypotheses: {len(vp)}",
        f"Parser parse rate: "
        f"{100 * sum(1 for r in vp if r.parser_pred_kind) / max(1, len(vp)):.0f}% "
        f"({sum(1 for r in vp if r.parser_pred_kind)}/{len(vp)} kinds parsed)",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[done] {out_dir}", flush=True)
    print(f"  T-GOAL: parser {p_acc:.0f}% ({p_c}/{p_n})  vs  "
          f"judge {j_acc:.0f}% ({j_c}/{j_n})", flush=True)
    print(f"  v2_prod parser coverage: {_coverage(vp, 'parser_verdict')[2]:.0f}%", flush=True)


if __name__ == "__main__":
    main()
