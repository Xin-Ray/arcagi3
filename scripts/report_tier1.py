"""Diff base_metrics.json vs lora_metrics.json into tier1_eval_report.md.

Format follows docs/arch_sft_tier1_zh.md §5.4. Also computes the §6.1
decision-gate verdict at the bottom so a glance at the file tells you
whether the LoRA passed.

Usage:
    .venv/Scripts/python.exe scripts/report_tier1.py \\
        outputs/finetune/base_metrics.json \\
        outputs/finetune/lora_metrics.json \\
        --out outputs/finetune/tier1_eval_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pp(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%"


def _delta_pp(base: float | None, lora: float | None) -> str:
    if base is None or lora is None:
        return "n/a"
    return f"{(lora - base) * 100:+.1f} pp"


def _metric(d: dict, key: str) -> float | None:
    if key in d:
        return d[key]
    if "metrics" in d:
        return d["metrics"].get(key)
    return None


# ── decision gate per §6.1 ────────────────────────────────────────────────

def _gate(base: dict[str, Any], lora: dict[str, Any]) -> dict[str, Any]:
    verdict: dict[str, Any] = {"checks": [], "pass": True}

    def _suite_em(payload: dict, tag: str) -> float | None:
        s = payload.get("suites", {}).get(tag, {})
        return _metric(s, "exact_match")

    # 1. T1 ≥ 50 pp
    base_t1 = _suite_em(base, "T1")
    lora_t1 = _suite_em(lora, "T1")
    ok_t1 = (
        base_t1 is not None and lora_t1 is not None
        and (lora_t1 - base_t1) >= 0.50
    )
    verdict["checks"].append({
        "name": "T1 holdout ≥ +50 pp",
        "base": base_t1, "lora": lora_t1, "pass": ok_t1,
    })
    verdict["pass"] &= ok_t1

    # 2. T2/T3/T4/T8 ≥ 30 pp each
    for tag in ("T2", "T3", "T4", "T8"):
        b = _suite_em(base, tag); l = _suite_em(lora, tag)
        ok = b is not None and l is not None and (l - b) >= 0.30
        verdict["checks"].append({
            "name": f"{tag} holdout ≥ +30 pp",
            "base": b, "lora": l, "pass": ok,
        })
        verdict["pass"] &= ok

    # 3. planning probes y direction ≥ 4/5
    def _probe_agg(payload: dict, key: str) -> float | None:
        agg = payload.get("suites", {}).get("planning_probes", {}).get("aggregate", {})
        return agg.get(key)
    b_ydir, l_ydir = _probe_agg(base, "direction_y_correct"), _probe_agg(lora, "direction_y_correct")
    ok_ydir = l_ydir is not None and l_ydir >= 0.8
    verdict["checks"].append({
        "name": "planning y_dir ≥ 4/5",
        "base": b_ydir, "lora": l_ydir, "pass": ok_ydir,
    })
    verdict["pass"] &= ok_ydir

    # 4. planning probes y count ≥ 3/5
    b_ycnt, l_ycnt = _probe_agg(base, "count_y_correct"), _probe_agg(lora, "count_y_correct")
    ok_ycnt = l_ycnt is not None and l_ycnt >= 0.6
    verdict["checks"].append({
        "name": "planning y_cnt ≥ 3/5",
        "base": b_ycnt, "lora": l_ycnt, "pass": ok_ycnt,
    })
    verdict["pass"] &= ok_ycnt

    # 5. regression: all 3 sets within -5 pp
    for name in ("mmlu_mini", "gsm8k_mini", "zh_qa_mini"):
        b = base.get("suites", {}).get("regression", {}).get(name, {}).get("accuracy")
        l = lora.get("suites", {}).get("regression", {}).get(name, {}).get("accuracy")
        ok = b is not None and l is not None and (l - b) >= -0.05
        verdict["checks"].append({
            "name": f"regression {name} Δ ≥ -5 pp",
            "base": b, "lora": l, "pass": ok,
        })
        verdict["pass"] &= ok

    return verdict


# ── markdown rendering ────────────────────────────────────────────────────

def _render(base: dict, lora: dict) -> str:
    out = ["# Tier 1 SFT Evaluation Report", ""]
    out.append(f"- base: `{base.get('model_path')}` ({base.get('quantize')}, "
               f"cap={base.get('cap_per_suite')})")
    lora_path = lora.get("lora_path") or "(none)"
    out.append(f"- lora: `{lora_path}` ({lora.get('quantize')}, "
               f"cap={lora.get('cap_per_suite')})")
    out.append(f"- timestamps: base {base.get('timestamp')} / lora {lora.get('timestamp')}")
    out.append("")

    # In-distribution holdout
    out.append("## In-distribution holdout")
    out.append("")
    out.append("| Suite | N | Base | LoRA | Δ |")
    out.append("|---|---:|---:|---:|---|")
    for tag in ("T1", "T2", "T2_remainder", "T2_multi", "T3", "T4", "T8"):
        b = base.get("suites", {}).get(tag, {})
        l = lora.get("suites", {}).get(tag, {})
        n = l.get("n") or b.get("n", 0)
        em_b = _metric(b, "exact_match")
        em_l = _metric(l, "exact_match")
        out.append(f"| {tag} | {n} | {_pp(em_b)} | {_pp(em_l)} | "
                   f"{_delta_pp(em_b, em_l)} |")
    out.append("")

    # OOD
    out.append("## OOD generalization")
    out.append("")
    out.append("| Suite | N | Base | LoRA | Δ |")
    out.append("|---|---:|---:|---:|---|")
    for tag in ("T1_ood", "T2_ood", "T3_ood", "T4_ood", "T8_ood"):
        b = base.get("suites", {}).get(tag, {})
        l = lora.get("suites", {}).get(tag, {})
        n = l.get("n") or b.get("n", 0)
        em_b = _metric(b, "exact_match")
        em_l = _metric(l, "exact_match")
        out.append(f"| {tag} | {n} | {_pp(em_b)} | {_pp(em_l)} | "
                   f"{_delta_pp(em_b, em_l)} |")
    out.append("")

    # Planning probes
    out.append("## Planning probes (the real test)")
    out.append("")
    out.append("| Probe | format | x_dir | y_dir | x_cnt | y_cnt | term | final |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    base_pp = base.get("suites", {}).get("planning_probes", {}).get("per_probe", {})
    lora_pp = lora.get("suites", {}).get("planning_probes", {}).get("per_probe", {})
    for name in sorted(set(base_pp) | set(lora_pp)):
        b = base_pp.get(name, {}); l = lora_pp.get(name, {})
        out.append(f"| {name} (base) | "
                   f"{b.get('format_valid', 0):.0f} | "
                   f"{b.get('direction_x_correct', 0):.0f} | "
                   f"{b.get('direction_y_correct', 0):.0f} | "
                   f"{b.get('count_x_correct', 0):.0f} | "
                   f"{b.get('count_y_correct', 0):.0f} | "
                   f"{b.get('terminator_correct', 0):.0f} | "
                   f"{b.get('final_distance', float('nan')):.1f} |")
        out.append(f"| {name} (lora) | "
                   f"{l.get('format_valid', 0):.0f} | "
                   f"{l.get('direction_x_correct', 0):.0f} | "
                   f"{l.get('direction_y_correct', 0):.0f} | "
                   f"{l.get('count_x_correct', 0):.0f} | "
                   f"{l.get('count_y_correct', 0):.0f} | "
                   f"{l.get('terminator_correct', 0):.0f} | "
                   f"{l.get('final_distance', float('nan')):.1f} |")
    out.append("")

    # Regression
    out.append("## Regression (LoRA must not break general ability)")
    out.append("")
    out.append("| Suite | N | Base | LoRA | Δ |")
    out.append("|---|---:|---:|---:|---|")
    for name in ("mmlu_mini", "gsm8k_mini", "zh_qa_mini"):
        b = base.get("suites", {}).get("regression", {}).get(name, {})
        l = lora.get("suites", {}).get("regression", {}).get(name, {})
        n = l.get("n") or b.get("n", 0)
        out.append(f"| {name} | {n} | {_pp(b.get('accuracy'))} | "
                   f"{_pp(l.get('accuracy'))} | "
                   f"{_delta_pp(b.get('accuracy'), l.get('accuracy'))} |")
    out.append("")

    # Gate verdict
    gate = _gate(base, lora)
    out.append("## §6.1 decision gate")
    out.append("")
    out.append(f"**OVERALL: {'PASS ✅' if gate['pass'] else 'FAIL ❌'}**")
    out.append("")
    out.append("| Check | Base | LoRA | Pass |")
    out.append("|---|---:|---:|:---:|")
    for c in gate["checks"]:
        out.append(f"| {c['name']} | {_pp(c['base'])} | {_pp(c['lora'])} | "
                   f"{'✅' if c['pass'] else '❌'} |")
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("base", type=Path, help="base_metrics.json path")
    p.add_argument("lora", type=Path, help="lora_metrics.json path")
    p.add_argument("--out",
                   default="outputs/finetune/tier1_eval_report.md", type=Path)
    args = p.parse_args()

    base = json.loads(args.base.read_text(encoding="utf-8"))
    lora = json.loads(args.lora.read_text(encoding="utf-8"))
    out_path = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(base, lora), encoding="utf-8")
    gate = _gate(base, lora)
    print(f"[report_tier1] wrote {out_path}")
    print(f"[report_tier1] gate verdict: {'PASS' if gate['pass'] else 'FAIL'}")
    for c in gate["checks"]:
        mark = "OK " if c["pass"] else "FAIL"
        print(f"  [{mark}] {c['name']:40s}  base={_pp(c['base'])}  lora={_pp(c['lora'])}")


if __name__ == "__main__":
    main()
