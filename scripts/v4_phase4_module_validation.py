"""v4 Phase 4 per-module re-validation.

Per user 2026-05-19 ("评估方式应该记录下来"), the 6 "validated" modules
were measured by proxy metrics (change_rate / diversity / count), not
by their per-module PASS definitions. This script re-evaluates the
THREE modules whose PASS definition is checkable from existing trace
data (no human labels needed):

  Module 3: Action Selection — when hypothesis carries a direction
            target, is the chosen action aligned with that direction?
  Module 5: force_cot consistency — does reasoning's mentioned ACTION
            match the actual chosen action?
  Module 6: K=3 candidate quality — does the candidate list contain a
            goal-aligned option? (proxy: did model say "I pick X
            because X moves toward goal" in reasoning?)

Modules 1 (Goal Generation) and 4 (Force-reject correctness) need
human labels and are NOT covered here.

Reads:
  outputs/v4_phase4_g{1..5}_*/round_{00,01}/trace.jsonl
  outputs/v4_phase4_g{1..5}_*/round_{00,01}/action_raw.txt
  outputs/v4_phase4_g{1..5}_*/round_{00,01}/knowledge_per_step.jsonl

Writes:
  docs/project/2026-05-19-v0-v4_clean_baseline/module_validation.md
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from arc_agent.goal_evaluator import parse_goal_hypothesis


PHASE4_DIRS = {
    "ar25": "v4_phase4_g1_ar25_s42_*",
    "bp35": "v4_phase4_g2_bp35_s42_*",
    "cd82": "v4_phase4_g3_cd82_s42_*",
    "cn04": "v4_phase4_g4_cn04_s42_*",
    "dc22": "v4_phase4_g5_dc22_s42_*",
}


# ── helpers ────────────────────────────────────────────────────────────

ACTION_DIRECTION = {
    "ACTION1": "up",
    "ACTION2": "down",
    "ACTION3": "left",
    "ACTION4": "right",
}


def hypothesis_direction(hyp_text: str, current_objs_centers: list[tuple[int, int]]
                         ) -> tuple[str | None, str]:
    """Given a hypothesis text + current object centers, return the
    direction the agent SHOULD move to satisfy the hypothesis.

    Returns (direction, target_kind) or (None, "") if not determinable.
    direction is one of: up / down / left / right.
    """
    pred = parse_goal_hypothesis(hyp_text)
    if pred is None:
        return None, ""

    if pred.kind == "move_to_row" and pred.target is not None:
        # Need to know current avg row of relevant objs
        if not current_objs_centers:
            return None, pred.kind
        avg_r = sum(r for r, _ in current_objs_centers) / len(current_objs_centers)
        if pred.target > avg_r:
            return "down", pred.kind
        if pred.target < avg_r:
            return "up", pred.kind
        return None, pred.kind  # already there

    if pred.kind == "move_to_col" and pred.target is not None:
        if not current_objs_centers:
            return None, pred.kind
        avg_c = sum(c for _, c in current_objs_centers) / len(current_objs_centers)
        if pred.target > avg_c:
            return "right", pred.kind
        if pred.target < avg_c:
            return "left", pred.kind
        return None, pred.kind

    if pred.kind == "move_to_center":
        if not current_objs_centers:
            return None, pred.kind
        avg_r = sum(r for r, _ in current_objs_centers) / len(current_objs_centers)
        avg_c = sum(c for _, c in current_objs_centers) / len(current_objs_centers)
        # Whichever axis is further from center is the priority
        dr, dc = 31 - avg_r, 31 - avg_c
        if abs(dr) > abs(dc):
            return ("down" if dr > 0 else "up"), pred.kind
        else:
            return ("right" if dc > 0 else "left"), pred.kind

    # align_any / stack / etc don't have single-axis direction
    return None, pred.kind


_ACTION_MENTION_RE = re.compile(r"\bACTION([1-7])\b")


def extract_reasoning_action_mention(reasoning: str) -> str | None:
    """Pull the LAST ACTION mention from reasoning text (typically the
    chosen one). Returns 'ACTION1'..'ACTION7' or None."""
    if not reasoning:
        return None
    matches = _ACTION_MENTION_RE.findall(reasoning)
    if not matches:
        return None
    return f"ACTION{matches[-1]}"


def get_centers_from_action_raw_or_knowledge(rd: Path, step: int) -> list[tuple[int, int]]:
    """Pull active object centers from knowledge_per_step.jsonl at given
    step. We use click_targets as a proxy since they capture the
    currently-tracked objects.
    """
    kp = rd / "knowledge_per_step.jsonl"
    if not kp.exists():
        return []
    for line in kp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("step") == step:
            cts = r.get("knowledge_after", {}).get("click_targets", [])
            return [(c.get("coords", [0, 0])[0], c.get("coords", [0, 0])[1])
                    for c in cts if c.get("alive", False)]
    return []


# ── module analyses ────────────────────────────────────────────────────

def analyze_game(game: str, glob_pattern: str) -> dict:
    """Run all 3 module checks for one game."""
    dirs = sorted(Path("outputs").glob(glob_pattern))
    if not dirs:
        return {"game": game, "error": "no output dir"}
    d = dirs[-1]

    # Module 3 counters
    m3_total = 0
    m3_directional = 0  # hypothesis has clear direction
    m3_action_dir = 0   # chosen action is directional (1-4)
    m3_aligned = 0      # action direction matches hypothesis direction
    m3_misaligned = 0   # opposite or wrong direction
    m3_examples_misalign = []

    # Module 5 counters
    m5_total = 0
    m5_has_reasoning = 0
    m5_reasoning_mention = 0
    m5_match = 0
    m5_mismatch = 0
    m5_examples_mismatch = []

    # Module 6 (proxy from response): did model invoke "candidate
    # letter A/B/C" reasoning? (Indicates K=3 was offered.)
    m6_total = 0
    m6_letter_choice_mentioned = 0

    for round_dir in sorted(d.glob("round_*")):
        tr = round_dir / "trace.jsonl"
        if not tr.exists():
            continue
        rows = [json.loads(l) for l in tr.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            step = r["step"]
            action_chosen = r["action"]

            # Get the hypothesis at this step
            kp = round_dir / "knowledge_per_step.jsonl"
            hyp = ""
            if kp.exists():
                for kline in kp.read_text(encoding="utf-8").splitlines():
                    if not kline.strip():
                        continue
                    ks = json.loads(kline)
                    if ks.get("step") == step:
                        hyp = ks.get("knowledge_after", {}).get("goal_hypothesis", "")
                        break

            centers = get_centers_from_action_raw_or_knowledge(round_dir, step)

            # ── Module 3: Action direction alignment ───────────────
            m3_total += 1
            hyp_dir, kind = hypothesis_direction(hyp, centers)
            if hyp_dir is not None:
                m3_directional += 1
                act_dir = ACTION_DIRECTION.get(action_chosen)
                if act_dir is not None:
                    m3_action_dir += 1
                    if act_dir == hyp_dir:
                        m3_aligned += 1
                    else:
                        m3_misaligned += 1
                        if len(m3_examples_misalign) < 5:
                            m3_examples_misalign.append({
                                "step": step,
                                "hyp": hyp[:70],
                                "kind": kind,
                                "expected_dir": hyp_dir,
                                "action": action_chosen,
                                "actual_dir": act_dir,
                                "centers": centers[:3],
                            })

            # ── Module 5: reasoning ↔ action consistency ───────────
            reasoning = r.get("reasoning") or ""
            m5_total += 1
            if reasoning:
                m5_has_reasoning += 1
                mention = extract_reasoning_action_mention(reasoning)
                if mention:
                    m5_reasoning_mention += 1
                    if mention == action_chosen:
                        m5_match += 1
                    else:
                        m5_mismatch += 1
                        if len(m5_examples_mismatch) < 5:
                            m5_examples_mismatch.append({
                                "step": step,
                                "reasoning": reasoning[:120],
                                "reasoning_mention": mention,
                                "action_chosen": action_chosen,
                            })

            # ── Module 6 (proxy): letter-choice presence ───────────
            ar = round_dir / "action_raw.txt"
            # crude: search for "choice: A/B/C" in this step block
            m6_total += 1

    # Crude m6: read action_raw and count "choice: " lines
    for round_dir in sorted(d.glob("round_*")):
        ar = round_dir / "action_raw.txt"
        if not ar.exists():
            continue
        raw = ar.read_text(encoding="utf-8", errors="replace")
        m6_letter_choice_mentioned += len(re.findall(r"\bchoice\s*:\s*[ABC]\b",
                                                     raw, re.IGNORECASE))

    return {
        "game": game,
        "dir": d.name,
        "module_3": {
            "total_steps": m3_total,
            "n_directional_hypothesis": m3_directional,
            "n_action_directional": m3_action_dir,
            "n_aligned": m3_aligned,
            "n_misaligned": m3_misaligned,
            "alignment_rate": (m3_aligned / m3_action_dir
                                if m3_action_dir else 0.0),
            "examples_misaligned": m3_examples_misalign,
        },
        "module_5": {
            "total_steps": m5_total,
            "n_has_reasoning": m5_has_reasoning,
            "n_reasoning_mentions_action": m5_reasoning_mention,
            "n_match": m5_match,
            "n_mismatch": m5_mismatch,
            "match_rate": (m5_match / m5_reasoning_mention
                            if m5_reasoning_mention else 0.0),
            "examples_mismatch": m5_examples_mismatch,
        },
        "module_6": {
            "total_steps": m6_total,
            "n_letter_choice": m6_letter_choice_mentioned,
            "letter_choice_rate": (m6_letter_choice_mentioned / m6_total
                                    if m6_total else 0.0),
        },
    }


# ── main ───────────────────────────────────────────────────────────────

def main():
    results = {}
    for game, glob in PHASE4_DIRS.items():
        print(f"=== {game} ===", flush=True)
        r = analyze_game(game, glob)
        results[game] = r
        m3 = r.get("module_3", {})
        m5 = r.get("module_5", {})
        m6 = r.get("module_6", {})
        print(f"  M3 directional steps: {m3.get('n_directional_hypothesis')}/"
              f"{m3.get('total_steps')}  "
              f"aligned: {m3.get('n_aligned')}/{m3.get('n_action_directional')} = "
              f"{100*m3.get('alignment_rate',0):.0f}%")
        print(f"  M5 reasoning mentions: "
              f"{m5.get('n_reasoning_mentions_action')}/{m5.get('total_steps')}  "
              f"match action: {m5.get('n_match')}/{m5.get('n_reasoning_mentions_action')}"
              f" = {100*m5.get('match_rate',0):.0f}%")
        print(f"  M6 letter-choice mentions: {m6.get('n_letter_choice')}/"
              f"{m6.get('total_steps')} = "
              f"{100*m6.get('letter_choice_rate',0):.0f}%")

    # Aggregate
    print(f"\n=== AGGREGATE ===")
    m3_total_dir = sum(r.get("module_3",{}).get("n_action_directional",0)
                       for r in results.values())
    m3_total_aligned = sum(r.get("module_3",{}).get("n_aligned",0)
                            for r in results.values())
    m5_total_mention = sum(r.get("module_5",{}).get("n_reasoning_mentions_action",0)
                           for r in results.values())
    m5_total_match = sum(r.get("module_5",{}).get("n_match",0)
                        for r in results.values())
    print(f"M3 alignment: {m3_total_aligned}/{m3_total_dir} = "
          f"{100*m3_total_aligned/max(1,m3_total_dir):.0f}%")
    print(f"M5 consistency: {m5_total_match}/{m5_total_mention} = "
          f"{100*m5_total_match/max(1,m5_total_mention):.0f}%")

    # Write per-game JSON
    out = REPO / "docs/project/2026-05-19-v0-v4_clean_baseline/module_validation.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n[wrote] {out}")

    return results


if __name__ == "__main__":
    main()
