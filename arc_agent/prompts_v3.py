"""v3 prompt builders for the TextAgent.

Per `docs/ARCHITECTURE_v3_zh.md` §2, the Play prompt has these blocks:

  [SYSTEM]  - static, neutral, no action semantics
  [STATUS]  - step / level / legal / topmost / largest (env-derived facts)
  [ACTIVE]  - non-texture, non-background objects with per-uid history
  [TEXTURE] - one-line summary of texture cells (so LLM knows what was filtered)
  [ACTION]  - LearnedActionMap: what each ACTION did when tried
  [UNTRIED] - explicit list of untried legal actions
  [HISTORY] - last 5 (action, frame_changed) tuples
  [GOAL]    - current hypothesis (initially "unknown")
  [ASK]     - "pick next action. prefer untried."

All non-trivial reasoning (direction names, topmost/largest, action effects)
is pre-computed by upstream modules and inserted as text labels here.
"""
from __future__ import annotations

from typing import Any, Optional

from arc_agent.action_inference import (
    OutcomeLog,
    render_action_block,
    render_history_tail,
    render_untried_block,
)
from arc_agent.click_candidates import (
    ClickCandidate,
    render_click_candidates_block,
)
from arc_agent.object_extractor import ObjectRecord
from arc_agent.object_relations import ObjectRelations, render_relations_block
from arc_agent.object_tracker import ObjectMemory, TrackedObject
from arc_agent.temporal_classifier import Layer, texture_summary


PLAY_SYSTEM = """You play an unfamiliar turn-based grid game on a 64x64 grid.
You do not know in advance what each action does. Different games map
ACTION1..ACTION7 to different effects. Discover by trying actions and
observing how the grid changes.

The user prompt summarizes what has already been computed for you:
  [STATUS]   current step, level, and pre-computed aggregates
  [ACTIVE]   moving / interactive objects with movement history
  [TEXTURE]  static background pattern (filtered for you)
  [ACTION]   what each action has done so far this episode
  [UNTRIED]  legal actions you have NOT tried yet
  [HISTORY]  the last few action outcomes
  [GOAL]     a hypothesis you may refine

Your priorities, in order:
  1. If [UNTRIED] is non-empty, try one of those actions
  2. Otherwise, pick an action whose history is most consistent with
     making progress (frame changes, level advances)
  3. Never repeat the same action 3+ times in a row unless evidence
     proves it advances toward the goal

OUTPUT FORMAT (strict):
  ACTION1..ACTION5 and ACTION7 take NO parameters. Output the bare token.
  ACTION6 is the ONLY action that takes coordinates (x, y) in 0..63.

Valid token shapes:
  ACTION1                      (no params)
  ACTION3                      (no params)
  ACTION7                      (no params)
  ACTION6 <x> <y>              where x, y are integers in 0..63 that
                               YOU pick based on the [ACTIVE] objects
                               or [CLICK CANDIDATES] in the user prompt.

INVALID (do NOT output these):
  ACTION1 10 20                (ACTION1 takes no params)
  ACTION3 5 5                  (ACTION3 takes no params)
  ACTION6                      (ACTION6 needs x and y)

Do NOT copy any specific (x, y) pair verbatim from this prompt. Pick
coordinates that match an object's center or bbox shown in [ACTIVE].

Output ONE token only. No JSON, no prose, no explanation, no trailing
numbers on non-ACTION6 actions."""


REFLECT_SYSTEM = """You are the reflection step for an unfamiliar grid game.
Look at the structured observations of the last K steps, and update your
GOAL hypothesis.

Output strict JSON: {"goal": "...", "confidence": "low/medium/high"}.
Keep "goal" under 120 chars. Use evidence from [ACTIVE] / [ACTION], not
priors about other games.
"""


# ─── User prompt blocks ─────────────────────────────────────────────────────


def _format_status(step: int, max_steps: int,
                   level: int, total_levels: int,
                   state: str,
                   legal_actions: list[str],
                   active_objects: list[TrackedObject]) -> str:
    """Compact STATUS block."""
    # Mark each legal action with its parameter signature
    def _sig(a: str) -> str:
        return "(x, y in 0..63)" if a == "ACTION6" else "(no params)"
    actions_annotated = ", ".join(f"{a} {_sig(a)}" for a in legal_actions)
    lines = [
        f"step: {step} / {max_steps}",
        f"level: {level} / {total_levels}",
        f"game state: {state}",
        f"legal actions: {actions_annotated}",
    ]
    if active_objects:
        # topmost / largest pre-computed
        topmost = min(active_objects,
                      key=lambda t: t.history[-1].bbox[0] if t.history else 99)
        largest = max(active_objects,
                      key=lambda t: t.history[-1].size if t.history else 0)
        lines += [
            f"topmost active: {topmost.uid} ({topmost.history[-1].color_name})",
            f"largest active: {largest.uid} "
            f"({largest.history[-1].color_name}, size={largest.history[-1].size})",
        ]
    return "[STATUS]\n" + "\n".join("  " + ln for ln in lines)


def _format_active_block(active_objects: list[TrackedObject]) -> str:
    """[ACTIVE] block: one line per uid with last position + movement summary."""
    if not active_objects:
        return "[ACTIVE]\n  (no active objects yet — every cell looks static)"
    lines = []
    for t in active_objects:
        last = t.history[-1]
        line = (f"  {t.uid}: {last.color_name} "
                f"(size={last.size}, "
                f"bbox=[{last.bbox[0]},{last.bbox[1]},{last.bbox[2]},{last.bbox[3]}])")
        if len(t.history) >= 2:
            prev = t.history[-2]
            dy = int(round(last.center[0] - prev.center[0]))
            dx = int(round(last.center[1] - prev.center[1]))
            if dy or dx:
                parts = []
                if dy < 0: parts.append("UP")
                elif dy > 0: parts.append("DOWN")
                if dx < 0: parts.append("LEFT")
                elif dx > 0: parts.append("RIGHT")
                dist = max(abs(dy), abs(dx))
                line += (f"\n      last step: moved {dist} cell(s) "
                         f"{'+'.join(parts)} (dy={dy:+d}, dx={dx:+d})")
        lines.append(line)
    return "[ACTIVE]\n" + "\n".join(lines)


def _format_texture_block(frame_objects: list[ObjectRecord],
                          layer_by_id: dict[int, Layer]) -> str:
    summary = texture_summary(frame_objects, layer_by_id)
    if summary["texture_cells_total"] == 0:
        return "[TEXTURE]\n  (none)"
    parts = [f"{n} {c} cells" for c, n in summary["by_color"].items()]
    return ("[TEXTURE] (treated as background, filtered out)\n"
            f"  total: {summary['texture_cells_total']} cells "
            f"({', '.join(parts)})")


def _format_action_block(log: OutcomeLog, legal_actions: list[str]) -> str:
    return "[ACTION effects observed]\n" + render_action_block(log, legal_actions)


def _format_untried_block(log: OutcomeLog, legal_actions: list[str]) -> str:
    return "[UNTRIED legal actions]\n  " + render_untried_block(log, legal_actions)


def _format_history_block(log: OutcomeLog) -> str:
    return "[HISTORY last 5 steps]\n" + render_history_tail(log, n=5)


def _format_goal_block(goal: str, confidence: str = "low") -> str:
    if not goal:
        return "[GOAL hypothesis]\n  (unknown — still exploring)"
    return f"[GOAL hypothesis]\n  {goal}\n  confidence: {confidence}"


# ─── Top-level builders ─────────────────────────────────────────────────────


def build_play_user_prompt(
    *,
    step: int,
    max_steps: int,
    level: int,
    total_levels: int,
    state: str,
    legal_actions: list[str],
    frame_objects: list[ObjectRecord],
    layer_by_id: dict[int, Layer],
    object_memory: ObjectMemory,
    outcome_log: OutcomeLog,
    goal_hypothesis: str = "",
    goal_confidence: str = "low",
    diversification_hint: Optional[str] = None,
    stuck_reason: Optional[str] = None,
    click_candidates: Optional[list[ClickCandidate]] = None,
    object_relations: Optional[ObjectRelations] = None,
) -> str:
    """Construct the full user prompt for one Play step.

    `object_relations` is the optional output of
    `arc_agent.object_relations.compute_relations`. When passed, a
    `[OBJECT RELATIONS]` block (same-color groups, same-shape groups,
    closest pairs, edge clearances) is inserted between `[ACTIVE]` and
    `[TEXTURE]`. None preserves the v3 baseline behavior.
    """
    active = object_memory.alive_tracked()
    blocks = [
        _format_status(step, max_steps, level, total_levels, state,
                       legal_actions, active),
        _format_active_block(active),
    ]
    if object_relations is not None:
        blocks.append(render_relations_block(object_relations))
    blocks.extend([
        _format_texture_block(frame_objects, layer_by_id),
        _format_action_block(outcome_log, legal_actions),
        _format_untried_block(outcome_log, legal_actions),
        _format_history_block(outcome_log),
        _format_goal_block(goal_hypothesis, goal_confidence),
    ])
    # P1: CLICK CANDIDATES only when ACTION6 is legal
    if "ACTION6" in legal_actions and click_candidates is not None:
        blocks.append(render_click_candidates_block(click_candidates))
    if stuck_reason:
        blocks.append(f"[STUCK SIGNALS]\n  {stuck_reason}\n  "
                      "Pick a DIFFERENT action or a DIFFERENT (x, y) — "
                      "what you are doing is not working.")
    if diversification_hint:
        blocks.append(f"[ALERT] {diversification_hint}")
    blocks.append("[ASK]\n  Output ONE action token now.")
    return "\n\n".join(blocks)


def build_reflect_user_prompt(
    *,
    legal_actions: list[str],
    outcome_log: OutcomeLog,
    object_memory: ObjectMemory,
    current_goal: str,
) -> str:
    """Reflection prompt — only used every K steps."""
    active = object_memory.alive_tracked()
    blocks = [
        f"[CURRENT GOAL]\n  {current_goal or '(none)'}",
        _format_active_block(active),
        _format_action_block(outcome_log, legal_actions),
        f"[STEPS OBSERVED] {len(outcome_log.all_steps)}",
        ('[ASK]\n  Given the observations above, output strict JSON:\n'
         '  {"goal": "<one sentence>", "confidence": "low|medium|high"}'),
    ]
    return "\n\n".join(blocks)


__all__ = [
    "PLAY_SYSTEM",
    "REFLECT_SYSTEM",
    "build_play_user_prompt",
    "build_reflect_user_prompt",
]
