"""Merge spatial_bench_v2 (VL-3B text-only) and spatial_bench_text (Qwen2.5-3B
pure text) results into one side-by-side comparison report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

VL_RESULTS = REPO_ROOT / "outputs" / "spatial_bench_v2" / "results.json"
TEXT_RESULTS = REPO_ROOT / "outputs" / "spatial_bench_text" / "results.json"
OUT_DIR = REPO_ROOT / "outputs" / "spatial_bench_combined"


def main() -> None:
    if not VL_RESULTS.exists() or not TEXT_RESULTS.exists():
        raise SystemExit(
            f"missing one of:\n  {VL_RESULTS}\n  {TEXT_RESULTS}\n"
            "run bench_spatial_v2.py and bench_text_model.py first."
        )
    vl = {r["id"]: r for r in json.loads(VL_RESULTS.read_text(encoding="utf-8"))}
    txt = {r["id"]: r for r in json.loads(TEXT_RESULTS.read_text(encoding="utf-8"))}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ids = list(vl.keys())
    cats = sorted({vl[i]["category"] for i in ids})

    def agg(results: dict, mode: str, cat: str | None = None) -> tuple[int, int, float]:
        items = [r for r in results.values()
                 if cat is None or r["category"] == cat]
        ok = sum(1 for r in items if r[mode]["correct"])
        sec = sum(r[mode]["seconds"] for r in items)
        return ok, len(items), sec

    lines = [
        "# Spatial reasoning — VLM (text-only) vs Pure-text model",
        "",
        "Same 17 questions, same prompts. Compared two models:",
        "",
        "- **VL-text**: Qwen2.5-VL-3B-Instruct in TEXT-ONLY mode (no image input)",
        "- **Text-3B**: Qwen2.5-3B-Instruct (pure text model, no vision tower)",
        "",
        "Both at 4-bit, both running on the same GPU.",
        "",
        "## Overall accuracy",
        "",
        "| Model | Prompt | Correct | Total seconds | Avg s/q |",
        "|---|---|---|---|---|",
    ]
    for label, results in (("VL-text", vl), ("Text-3B", txt)):
        for mode in ("raw", "enriched"):
            ok, n, sec = agg(results, mode)
            lines.append(
                f"| {label} | {mode} | **{ok}/{n}** ({100*ok/n:.0f}%) | "
                f"{sec:.1f}s | {sec/n:.2f}s |"
            )
    lines += ["", "## Per category", "",
              "| Category | n | VL-text raw | VL-text enriched | Text-3B raw | Text-3B enriched |",
              "|---|---|---|---|---|---|"]
    for cat in cats:
        n = sum(1 for r in vl.values() if r["category"] == cat)
        vl_r = agg(vl, "raw", cat)[0]
        vl_e = agg(vl, "enriched", cat)[0]
        tx_r = agg(txt, "raw", cat)[0]
        tx_e = agg(txt, "enriched", cat)[0]
        lines.append(
            f"| **{cat}** | {n} | {vl_r}/{n} | {vl_e}/{n} | {tx_r}/{n} | {tx_e}/{n} |"
        )

    lines += ["", "## Per question (enriched prompt)", "",
              "| ID | Category | Expected | VL-text answer | OK | Text-3B answer | OK |",
              "|---|---|---|---|---|---|---|"]
    for qid in ids:
        v, t = vl[qid], txt[qid]
        ve = v["enriched"]
        te = t["enriched"]
        lines.append(
            f"| {qid} | {v['category']} | `{v['expected'][:30]}` | "
            f"`{ve['answer'][:30]}` | "
            f"{'✅' if ve['correct'] else '❌'} | "
            f"`{te['answer'][:30]}` | "
            f"{'✅' if te['correct'] else '❌'} |"
        )

    lines += ["", "## Per question (raw prompt)", "",
              "| ID | Category | Expected | VL-text answer | OK | Text-3B answer | OK |",
              "|---|---|---|---|---|---|---|"]
    for qid in ids:
        v, t = vl[qid], txt[qid]
        vr = v["raw"]
        tr = t["raw"]
        lines.append(
            f"| {qid} | {v['category']} | `{v['expected'][:30]}` | "
            f"`{vr['answer'][:30]}` | "
            f"{'✅' if vr['correct'] else '❌'} | "
            f"`{tr['answer'][:30]}` | "
            f"{'✅' if tr['correct'] else '❌'} |"
        )

    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'report.md'}")
    # Also dump as console summary
    print("\n=== Quick console summary ===")
    for label, results in (("VL-text", vl), ("Text-3B", txt)):
        for mode in ("raw", "enriched"):
            ok, n, sec = agg(results, mode)
            print(f"  {label:8s} {mode:8s}  {ok}/{n} correct, {sec/n:.2f}s/q avg")


if __name__ == "__main__":
    main()
