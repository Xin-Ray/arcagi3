"""Re-enrich annotation_form.md with Module 0 outputs.

For each sample in samples.jsonl we replay the env to step_n, extract
objects + relations + Knowledge, and rewrite annotation_form.md so the
user can see EXACTLY what the LLM saw at hypothesis-generation time.

Usage:
    .venv/Scripts/python.exe scripts/enrich_annotation_form.py \
        --bench-dir docs/project/2026-05-28-v0-goal_gen_validation/bench_run_20260529-141709
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO / ".env")

from arcengine import GameAction, GameState  # noqa: E402

from arc_agent.knowledge import Knowledge  # noqa: E402
from arc_agent.object_extractor import extract_objects, ObjectRecord  # noqa: E402
from arc_agent.object_relations import compute_relations  # noqa: E402
from arc_agent.observation import latest_grid  # noqa: E402


PHASE4_DIRS = {
    "ar25": REPO / "outputs/v4_phase4_g1_ar25_s42_20260519-053015",
    "bp35": REPO / "outputs/v4_phase4_g2_bp35_s42_20260519-062440",
    "cd82": REPO / "outputs/v4_phase4_g3_cd82_s42_20260519-065756",
    "cn04": REPO / "outputs/v4_phase4_g4_cn04_s42_20260519-080429",
    "dc22": REPO / "outputs/v4_phase4_g5_dc22_s42_20260519-084820",
}


def load_action_sequence(phase4_dir: Path
                          ) -> list[tuple[str, Optional[tuple[int, int]]]]:
    tr = phase4_dir / "round_00" / "trace.jsonl"
    rows = [json.loads(l) for l in tr.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    seq = []
    for r in rows:
        coords = r.get("action_coords")
        if coords is not None:
            coords = (int(coords[0]), int(coords[1]))
        seq.append((r["action"], coords))
    return seq


def step_env(env, action_name: str, coords: Optional[tuple[int, int]]):
    act = GameAction[action_name]
    if act.is_complex() and coords is not None:
        act.set_data({"x": coords[0], "y": coords[1]})
    data = act.action_data.model_dump() if act.is_complex() else {}
    try:
        return env.step(act, data=data)
    except TypeError:
        return env.step(act)


def knowledge_at_step(phase4_dir: Path, step_n: int) -> Knowledge:
    kp = phase4_dir / "round_00" / "knowledge_per_step.jsonl"
    for line in kp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("step") == step_n:
            try:
                return Knowledge.from_dict(r["knowledge_after"])
            except Exception:
                return Knowledge.empty()
    return Knowledge.empty()


def render_objects(objs: list[ObjectRecord]) -> str:
    if not objs:
        return "  (no objects)"
    lines = []
    for o in objs[:30]:  # cap for readability
        bbox = o.bbox
        lines.append(
            f"  - obj_{o.id:03d}  color={o.color_name:<10s} "
            f"shape={bbox[2]-bbox[0]+1}x{bbox[3]-bbox[1]+1}  "
            f"center=({int(round(o.center[0]))},{int(round(o.center[1]))})  "
            f"bbox=({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})  size={o.size}"
        )
    if len(objs) > 30:
        lines.append(f"  ... ({len(objs) - 30} more)")
    return "\n".join(lines)


def render_relations(relations) -> str:
    try:
        from arc_agent.object_relations import render_relations_block
        block = render_relations_block(relations)
        # Trim to first ~20 lines
        lines = block.splitlines()
        if len(lines) > 25:
            lines = lines[:25] + [f"  ... ({len(lines)-25} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"  (could not render: {e})"


def render_knowledge_prior(k: Knowledge) -> str:
    out = []
    out.append(f"  rounds_played: {k.rounds_played}, rounds_won: {k.rounds_won}")
    out.append(f"  goal_hypothesis: {k.goal_hypothesis!r}")
    out.append(f"  goal_confidence: {k.goal_confidence}")
    if k.action_semantics:
        out.append("  action_semantics:")
        for a, sem in k.action_semantics.items():
            out.append(f"    {a}: {sem[:120]}")
    else:
        out.append("  action_semantics: (empty)")
    if k.rejected_goals:
        out.append("  rejected_goals:")
        for rg in k.rejected_goals[:5]:
            out.append(f"    - {rg[:120]}")
        if len(k.rejected_goals) > 5:
            out.append(f"    ... ({len(k.rejected_goals)-5} more)")
    else:
        out.append("  rejected_goals: (empty)")
    if k.rules:
        out.append(f"  rules ({len(k.rules)}):")
        for r in k.rules[:3]:
            out.append(f"    - {r[:120]}")
    if k.failed_strategies:
        out.append(f"  failed_strategies ({len(k.failed_strategies)}):")
        for fs in k.failed_strategies[:3]:
            out.append(f"    - {fs[:120]}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench-dir", required=True)
    args = p.parse_args()
    bench = Path(args.bench_dir)

    samples_path = bench / "samples.jsonl"
    if not samples_path.exists():
        samples_path = bench / "samples.json"
    samples = [json.loads(l) for l in samples_path.read_text(encoding="utf-8")
                .splitlines() if l.strip()]
    print(f"[enrich] {len(samples)} samples loaded from {samples_path}")

    # Replay each frame to extract objects + relations
    from arc_agi import Arcade
    arc = Arcade()
    env_infos = arc.get_environments() or []

    enriched = []
    for i, s in enumerate(samples):
        gid = s["game_id"]
        ft = s["frame_type"]
        step_n = s["step_n"]
        phase4 = PHASE4_DIRS.get(gid)
        cands = [e.game_id for e in env_infos if e.game_id.startswith(gid)]
        if not cands or not phase4:
            print(f"  [{i+1}/{len(samples)}] {gid} {ft} — skip (no SDK/phase4)")
            enriched.append(s); continue
        full_id = cands[0]
        card = arc.open_scorecard(tags=["enrich_annotation"])
        env = arc.make(full_id, scorecard_id=card)
        latest = env.reset()
        if ft.startswith("B"):
            seq = load_action_sequence(phase4)
            for j in range(step_n):
                an, cc = seq[j]
                if an == "RESET":
                    latest = env.reset(); continue
                try:
                    latest = step_env(env, an, cc)
                except Exception:
                    break
                if latest.state in (GameState.WIN, GameState.GAME_OVER):
                    break
        grid = latest_grid(latest)
        objs = extract_objects(grid)
        rels = compute_relations(objs, grid_shape=grid.shape,
                                  layer_by_id={}, skip_texture=True)
        s["module0_objects_text"] = render_objects(objs)
        s["module0_relations_text"] = render_relations(rels)
        if ft.startswith("B"):
            k = knowledge_at_step(phase4, step_n)
            s["module1_prior_text"] = render_knowledge_prior(k)
        else:
            s["module1_prior_text"] = render_knowledge_prior(Knowledge.empty(game_id=full_id))
        arc.close_scorecard(card)
        enriched.append(s)
        print(f"  [{i+1}/{len(samples)}] {gid} {ft} step={step_n} — {len(objs)} objects")

    # Save enriched samples.jsonl back
    samples_path.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in enriched) + "\n",
        encoding="utf-8")
    print(f"[enrich] wrote enriched samples back to {samples_path}")

    # Rebuild annotation_form.md with embedded Module 0 / Knowledge blocks
    header = open(bench / "annotation_form.md", encoding="utf-8").read().split("---", 1)[0]

    parts = [header.rstrip(), ""]
    for s in enriched:
        gid = s["game_id"]; ft = s["frame_type"]; step_n = s["step_n"]
        png = Path(s["png_path"]).name if s.get("png_path") else ""
        gt = s["proposed_ground_truth"]
        parts.append("\n---\n")
        parts.append(f"## {gid} — frame {ft} (step {step_n})\n")
        if png:
            parts.append(f"![{gid} {ft}](./{png})\n")
        parts.append("### Module 0 output — objects extracted by scipy\n")
        parts.append("```\n" + s["module0_objects_text"] + "\n```\n")
        parts.append("### Module 0 output — object relations\n")
        parts.append("```\n" + s["module0_relations_text"] + "\n```\n")
        parts.append("### Module 1 prior — Knowledge state going INTO this hypothesis call\n")
        parts.append("```\n" + s["module1_prior_text"] + "\n```\n")
        parts.append("### Proposed ground truth (override below if you disagree)\n")
        parts.append(f"> {gt}\n")
        parts.append("```\nyour_corrected_ground_truth: \n```\n")
        parts.append("### K=5 hypotheses produced by Reflection Agent\n")
        parts.append("| k | hypothesis | label (YES/PARTIAL/NO/UNSURE) |")
        parts.append("|---:|---|---|")
        for h in s["hypotheses"]:
            hyp = h.get("hypothesis")
            hyp_str = hyp if hyp else "(parse failed)"
            parts.append(f"| {h['k_index']} | {hyp_str} | **____** |")
        parts.append("")

    form = bench / "annotation_form.md"
    form.write_text("\n".join(parts), encoding="utf-8")
    print(f"[enrich] rewrote {form} ({form.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
