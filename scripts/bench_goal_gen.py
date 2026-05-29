"""T-DISCOVER bench: Module 1 (Goal Generation) on 5 G_base games.

Setup
-----
For each of 5 games, we test Module 1 (Reflection Agent writing
goal_hypothesis) on TWO frames:
  A. step 0, zero prior  — empty Knowledge, fresh frame.
  B. step ~mid, with prior — accumulated Knowledge replayed from the
     Phase 4 trace's action sequence.

Per frame we sample K=5 hypotheses by running Reflection with
temperature=0.7 (so each call samples differently). reasoning_mode=cot
(/think) is used so the model can reason explicitly before answering.

Output
------
outputs/bench_goal_gen_<ts>/
  samples.jsonl                   one row per (game, frame, k) = 50 rows
  per_frame_summary.md            quick overview (5 game × 2 frame)
  step_<game>_<frame>_*.png       copied from Phase 4 PNGs for the form
  annotation_form.md              what the user fills

Usage
-----
.venv/Scripts/python.exe scripts/bench_goal_gen.py
    [--k 5] [--temperature 0.7] [--max-new-tokens 2048] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO / ".env")

import numpy as np  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

from arc_agent.agents.reflection_agent import (  # noqa: E402
    ReflectionAgent, parse_reflection_output,
)
from arc_agent.knowledge import Knowledge  # noqa: E402
from arc_agent.object_extractor import extract_objects  # noqa: E402
from arc_agent.object_relations import compute_relations  # noqa: E402
from arc_agent.observation import available_action_names, latest_grid  # noqa: E402
from arc_agent.step_summary import StepSummary  # noqa: E402
from arc_agent.vlm_backbone import make_backbone  # noqa: E402


PHASE4_DIRS = {
    "ar25": REPO / "outputs/v4_phase4_g1_ar25_s42_20260519-053015",
    "bp35": REPO / "outputs/v4_phase4_g2_bp35_s42_20260519-062440",
    "cd82": REPO / "outputs/v4_phase4_g3_cd82_s42_20260519-065756",
    "cn04": REPO / "outputs/v4_phase4_g4_cn04_s42_20260519-080429",
    "dc22": REPO / "outputs/v4_phase4_g5_dc22_s42_20260519-084820",
}


# ─── frame retrieval ───────────────────────────────────────────────────

def load_action_sequence(phase4_dir: Path, round_idx: int = 0
                          ) -> list[tuple[str, Optional[tuple[int, int]]]]:
    """Pull (action_name, coords) sequence from Phase 4 trace.jsonl."""
    tr = phase4_dir / f"round_{round_idx:02d}" / "trace.jsonl"
    if not tr.exists():
        return []
    rows = [json.loads(l) for l in tr.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    seq = []
    for r in rows:
        coords = r.get("action_coords")
        if coords is not None:
            coords = (int(coords[0]), int(coords[1]))
        seq.append((r["action"], coords))
    return seq


def make_env_for(game_id_prefix: str):
    from arc_agi import Arcade
    arc = Arcade()
    env_infos = arc.get_environments() or []
    candidates = [e.game_id for e in env_infos
                  if e.game_id.startswith(game_id_prefix)]
    if not candidates:
        raise RuntimeError(f"no game starting with {game_id_prefix!r}")
    return arc, candidates[0]


def step_env(env, action_name: str, coords: Optional[tuple[int, int]]):
    act = GameAction[action_name]
    if act.is_complex() and coords is not None:
        act.set_data({"x": coords[0], "y": coords[1]})
    data = act.action_data.model_dump() if act.is_complex() else {}
    try:
        return env.step(act, data=data)
    except TypeError:
        return env.step(act)


# ─── knowledge state retrieval ─────────────────────────────────────────

def knowledge_at_step(phase4_dir: Path, step_n: int, round_idx: int = 0
                       ) -> Knowledge:
    """Load Knowledge snapshot from Phase 4 knowledge_per_step.jsonl at
    the given step; fall back to empty if not found."""
    kp = phase4_dir / f"round_{round_idx:02d}" / "knowledge_per_step.jsonl"
    if not kp.exists():
        return Knowledge.empty()
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


# ─── one reflection call ───────────────────────────────────────────────

def build_step_summary_dummy(latest_frame) -> StepSummary:
    """Minimal StepSummary for the prompt builder. For step 0 the agent
    hasn't acted yet -- we pass a sentinel that the renderer accepts."""
    return StepSummary(
        step=0, action="RESET", action_coords=None,
        reasoning="(round start)", frame_changed=False,
        primary_direction=None, primary_distance=None,
        object_deltas=[], no_op_streak=0, state_revisit_count=1,
        matches_reasoning="N/A", recent_steps=[],
    )


def run_reflection_k_times(refl: ReflectionAgent, *,
                            knowledge: Knowledge,
                            step_summary: StepSummary,
                            latest, frame_objects, layer_by_id,
                            object_relations, exploration_hint: str,
                            k: int) -> list[dict]:
    """Run reflect_after_step K times. Each call gets the same input but
    sampling is stochastic when temperature > 0 so we get K diverse
    hypotheses."""
    out: list[dict] = []
    legal_actions = available_action_names(latest)
    for ki in range(k):
        t = time.time()
        try:
            delta, raw = refl.reflect_after_step(
                knowledge=knowledge,
                step_summary=step_summary,
                step=step_summary.step,
                max_steps=200,
                level=latest.levels_completed + 1,
                total_levels=latest.win_levels,
                state_name=latest.state.name,
                legal_actions=legal_actions,
                frame_objects=frame_objects,
                layer_by_id=layer_by_id,
                object_memory=None,  # not loaded; relations cover the use case
                outcome_log=None,
                object_relations=object_relations,
                exploration_hint=exploration_hint or None,
            )
            elapsed = time.time() - t
            parsed_hyp = delta.get("goal_hypothesis_update") if isinstance(delta, dict) else None
        except Exception as e:
            delta, raw, elapsed, parsed_hyp = {"_error": str(e)}, "", time.time()-t, None

        out.append({
            "k_index": ki,
            "hypothesis": parsed_hyp,
            "raw_tail": (raw or "")[-400:],
            "raw_len": len(raw or ""),
            "elapsed_s": round(elapsed, 1),
            "delta_keys": list(delta.keys()) if isinstance(delta, dict) else [],
        })
        print(f"    k={ki+1}/{k}: hyp={parsed_hyp!r} ({elapsed:.1f}s)", flush=True)
    return out


# ─── ground-truth proposal ─────────────────────────────────────────────

def propose_ground_truth(game_id: str, frame_objects: list,
                          relations: Any) -> str:
    """Heuristic 'best guess' goal hypothesis from the frame structure.
    The user judges the K=5 model outputs against this proposal AND
    overrides it freely.

    Heuristics, in priority order:
      - If 2+ same-color non-static objects with a static same-color
        target nearby: "match X with the static X target".
      - If 2+ same-color objects spread across rows/cols: "align the
        X objects".
      - If many distinct colors + symmetric layout: "match each colored
        piece with its same-color target".
    """
    from collections import Counter
    if not frame_objects:
        return "(no objects extracted — game-specific)"
    color_counts = Counter(o.color_name for o in frame_objects)
    most_common_color, most_common_n = color_counts.most_common(1)[0]
    if most_common_n >= 2:
        return (f"match / align the {most_common_color} objects with their "
                f"same-color targets (heuristic best-guess — verify by play)")
    if len(color_counts) >= 4:
        return ("match each non-background object with its same-color target "
                "across the grid")
    return "explore — frame structure does not suggest a clear goal"


# ─── annotation form builder ───────────────────────────────────────────

ANNOTATION_HEADER = """# T-DISCOVER bench — Annotation Form

> Generated by `scripts/bench_goal_gen.py` on {ts}. Edit this file in
> place: for every (game, frame, hypothesis_k) row, fill in the
> **label** column with one of `YES`, `PARTIAL`, `NO`, or `UNSURE`.
> Optional: edit `proposed_ground_truth` to your own best guess.
>
> When done, run:
>
>     .venv/Scripts/python.exe scripts/bench_goal_gen_score.py
>
> to produce charts + summary.

## Labels meaning

- **YES** — hypothesis describes the real win condition (or close enough
  that an agent acting on it would tend to win).
- **PARTIAL** — captures part of the goal but misses details / names
  wrong objects / is too vague.
- **NO** — wrong target, hallucinated entities, or action-described
  rather than state-described.
- **UNSURE** — you cannot judge without more info.
"""


def write_annotation_form(out_dir: Path, samples: list[dict],
                           ts: str) -> Path:
    form = [ANNOTATION_HEADER.format(ts=ts)]
    for s in samples:
        gid = s["game_id"]
        ft = s["frame_type"]
        step_n = s["step_n"]
        png = s["png_path"]
        gt = s["proposed_ground_truth"]
        form.append(f"\n---\n\n## {gid} — frame {ft} (step {step_n})")
        form.append(f"\n![{gid} {ft}](./{Path(png).name})\n")
        form.append(f"**Proposed ground truth (you can override below)**:")
        form.append(f"\n> {gt}\n")
        form.append(f"```\nyour_corrected_ground_truth: \n```\n")
        form.append("\n### K=5 hypotheses\n")
        form.append("| k | hypothesis | label (YES/PARTIAL/NO/UNSURE) |")
        form.append("|---:|---|---|")
        for h in s["hypotheses"]:
            hyp = h.get("hypothesis")
            hyp_str = hyp if hyp else "(parse failed)"
            form.append(f"| {h['k_index']} | {hyp_str} | **____** |")
    out = out_dir / "annotation_form.md"
    out.write_text("\n".join(form), encoding="utf-8")
    return out


# ─── main ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--games", default="ar25,bp35,cd82,cn04,dc22")
    p.add_argument("--mid-step", type=int, default=30,
                    help="step index for the 'with-prior' frame B")
    args = p.parse_args()

    rng = random.Random(args.seed)
    games = [g.strip() for g in args.games.split(",") if g.strip()]

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = REPO / f"outputs/bench_goal_gen_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bench] out: {out_dir}", flush=True)

    print("[bench] loading SmolLM3-3B (cot)...", flush=True)
    t0 = time.time()
    backbone = make_backbone("HuggingFaceTB/SmolLM3-3B", reasoning_mode="cot")
    print(f"[bench]   loaded in {time.time()-t0:.1f}s", flush=True)

    refl = ReflectionAgent(
        backbone=backbone,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    samples: list[dict] = []

    print("[bench] connecting to Arcade SDK...", flush=True)
    from arc_agi import Arcade
    arc = Arcade()
    env_infos = arc.get_environments() or []

    for gid in games:
        candidates = [e.game_id for e in env_infos if e.game_id.startswith(gid)]
        if not candidates:
            print(f"  {gid}: no match in SDK; skipping", flush=True)
            continue
        full_game_id = candidates[0]
        phase4_dir = PHASE4_DIRS.get(gid)
        if phase4_dir is None or not phase4_dir.exists():
            print(f"  {gid}: no Phase 4 dir; skipping", flush=True)
            continue

        # Frame A: step 0, zero prior
        print(f"\n[bench] {gid} — Frame A (step 0, zero prior)", flush=True)
        card = arc.open_scorecard(tags=["bench_goal_gen", "frame_A"])
        env = arc.make(full_game_id, scorecard_id=card)
        latest = env.reset()
        grid = latest_grid(latest)
        objs = extract_objects(grid)
        relations = compute_relations(objs, grid_shape=grid.shape,
                                       layer_by_id={}, skip_texture=True)
        layer_by_id = {o.id: "ACTIVE" for o in objs}  # placeholder

        png_a = out_dir / f"frame_A_{gid}_step0.png"
        src_png = phase4_dir / "round_00" / "step_0000.png"
        if src_png.exists():
            shutil.copy(src_png, png_a)

        gt_a = propose_ground_truth(gid, objs, relations)
        print(f"  proposed GT: {gt_a}", flush=True)
        print(f"  running K={args.k} hypotheses...", flush=True)
        hyps_a = run_reflection_k_times(
            refl,
            knowledge=Knowledge.empty(game_id=full_game_id),
            step_summary=build_step_summary_dummy(latest),
            latest=latest,
            frame_objects=objs,
            layer_by_id=layer_by_id,
            object_relations=relations,
            exploration_hint="",
            k=args.k,
        )
        arc.close_scorecard(card)
        samples.append({
            "game_id": gid,
            "full_game_id": full_game_id,
            "frame_type": "A_zero_prior",
            "step_n": 0,
            "n_objects": len(objs),
            "object_colors": sorted({o.color_name for o in objs}),
            "proposed_ground_truth": gt_a,
            "png_path": str(png_a.relative_to(out_dir)) if png_a.exists() else "",
            "hypotheses": hyps_a,
        })

        # Frame B: replay to mid_step, accumulated Knowledge from Phase 4
        target_step = min(args.mid_step,
                          len(list((phase4_dir / "round_00").glob("step_*.png"))) - 1)
        print(f"\n[bench] {gid} — Frame B (step {target_step}, with prior)", flush=True)
        action_seq = load_action_sequence(phase4_dir, round_idx=0)
        if len(action_seq) <= target_step:
            print(f"  Phase 4 trace too short ({len(action_seq)} steps); skip Frame B", flush=True)
            continue
        card_b = arc.open_scorecard(tags=["bench_goal_gen", "frame_B"])
        env_b = arc.make(full_game_id, scorecard_id=card_b)
        latest_b = env_b.reset()
        for i in range(target_step):
            action_name, coords = action_seq[i]
            if action_name == "RESET":
                latest_b = env_b.reset()
                continue
            try:
                latest_b = step_env(env_b, action_name, coords)
            except Exception as e:
                print(f"  replay error at step {i}: {e}", flush=True)
                break
            if latest_b.state in (GameState.WIN, GameState.GAME_OVER):
                break

        grid_b = latest_grid(latest_b)
        objs_b = extract_objects(grid_b)
        relations_b = compute_relations(objs_b, grid_shape=grid_b.shape,
                                          layer_by_id={}, skip_texture=True)
        layer_by_id_b = {o.id: "ACTIVE" for o in objs_b}
        knowledge_b = knowledge_at_step(phase4_dir, target_step, round_idx=0)

        png_b = out_dir / f"frame_B_{gid}_step{target_step}.png"
        src_png_b = phase4_dir / "round_00" / f"step_{target_step:04d}.png"
        if src_png_b.exists():
            shutil.copy(src_png_b, png_b)

        gt_b = propose_ground_truth(gid, objs_b, relations_b)
        print(f"  proposed GT: {gt_b}", flush=True)
        print(f"  loaded Knowledge with {len(knowledge_b.action_semantics)} action_semantics, "
              f"{len(knowledge_b.rejected_goals)} rejected_goals, hyp={knowledge_b.goal_hypothesis!r}",
              flush=True)
        print(f"  running K={args.k} hypotheses...", flush=True)
        hyps_b = run_reflection_k_times(
            refl,
            knowledge=knowledge_b,
            step_summary=build_step_summary_dummy(latest_b),
            latest=latest_b,
            frame_objects=objs_b,
            layer_by_id=layer_by_id_b,
            object_relations=relations_b,
            exploration_hint="",
            k=args.k,
        )
        arc.close_scorecard(card_b)
        samples.append({
            "game_id": gid,
            "full_game_id": full_game_id,
            "frame_type": "B_with_prior",
            "step_n": target_step,
            "n_objects": len(objs_b),
            "object_colors": sorted({o.color_name for o in objs_b}),
            "prior_hypothesis": knowledge_b.goal_hypothesis,
            "prior_action_semantics_count": len(knowledge_b.action_semantics),
            "prior_rejected_goals_count": len(knowledge_b.rejected_goals),
            "proposed_ground_truth": gt_b,
            "png_path": str(png_b.relative_to(out_dir)) if png_b.exists() else "",
            "hypotheses": hyps_b,
        })

    # Save samples.jsonl
    out_jsonl = out_dir / "samples.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Summary
    summary_lines = [
        "# T-DISCOVER bench — Per-frame summary",
        "",
        f"Generated {ts}; model SmolLM3-3B /think; K={args.k}; temp={args.temperature}",
        "",
        "| Game | Frame | Step | #Objects | Colors | #Distinct hypotheses (parsed) | Avg elapsed |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for s in samples:
        parsed = [h["hypothesis"] for h in s["hypotheses"] if h["hypothesis"]]
        distinct = len(set(parsed))
        avg_t = (sum(h["elapsed_s"] for h in s["hypotheses"]) /
                  max(1, len(s["hypotheses"])))
        summary_lines.append(
            f"| {s['game_id']} | {s['frame_type'].split('_')[0]} | {s['step_n']} | "
            f"{s['n_objects']} | {', '.join(s['object_colors'][:5])} | "
            f"{distinct} | {avg_t:.1f}s |"
        )
    (out_dir / "per_frame_summary.md").write_text("\n".join(summary_lines),
                                                   encoding="utf-8")

    # Annotation form
    form_path = write_annotation_form(out_dir, samples, ts)

    print(f"\n[done] {out_dir}", flush=True)
    print(f"  samples.jsonl       — {len(samples)} frames", flush=True)
    print(f"  per_frame_summary.md", flush=True)
    print(f"  annotation_form.md  — {sum(len(s['hypotheses']) for s in samples)} hypotheses to label",
          flush=True)


if __name__ == "__main__":
    main()
