"""v3.2 prompts for the Action Agent and Reflection Agent.

Per `docs/arch_v3_2_zh.md` §4.2 / §4.3 (Reflection) and §5.2 / §5.3
(Action). Layout:

  ACTION USER prompt = [REFLECTION ALERT] (when non-empty)
                       + [KNOWLEDGE]
                       + <v3 blocks: STATUS / ACTIVE / TEXTURE / ACTION /
                          UNTRIED / HISTORY / GOAL / [CLICK CANDIDATES] /
                          [STUCK SIGNALS]>
                       + [ASK]  (changed: "reasoning + action" two lines)

  REFLECTION USER prompt = [CURRENT KNOWLEDGE]
                           + step_summary.render()  (action + reasoning
                             + outcome + matches_reasoning + last 3)
                           + [ASK]  (strict JSON delta)

The v3 blocks are produced by `prompts_v3.build_play_user_prompt`. To
stay non-invasive, this module calls that builder and then splices in
the v3.2 additions; v3 prompts.py is unchanged.
"""
from __future__ import annotations

from typing import Any, Optional

from arc_agent.action_inference import render_action_block
from arc_agent.click_targets import ClickTarget, render_click_targets_block
from arc_agent.knowledge import Knowledge
from arc_agent.object_relations import render_relations_block
from arc_agent.step_summary import StepSummary


# ─── SYSTEM constants ───────────────────────────────────────────────────────

ACTION_SYSTEM = """You are the Action Agent for a turn-based 64x64 grid game.

Each step you receive accumulated KNOWLEDGE (action_semantics,
goal_hypothesis, rejected_goals, rules, failed_strategies) plus
per-step perception. Trust the KNOWLEDGE block. Do NOT re-explore
things in failed_strategies or re-propose goals in rejected_goals.

If an [ALERT] block is at the top, the orchestrator or Reflection has
flagged that your previous mental model was wrong. Read it FIRST and
change behavior.

REASONING REQUIREMENTS (the Reflection Agent uses these to update
KNOWLEDGE; vague reasoning starves the loop):
  - Name a SUBJECT: an obj_id (e.g. obj_007) or a color+shape
    (e.g. "the cyan 1x1"). Reasonings like "try edge" / "try a
    different location" / "try another action" without a subject
    are NOT acceptable.
  - State the EXPECTED EFFECT: a direction (UP/DOWN/LEFT/RIGHT) or
    an outcome ("places a marker", "advances the level").

ACTION6 (takes x, y parameters -- effect game-specific, NOT a generic click):
  - The action name is "ACTION6", not "click". Its game-specific effect
    is UNKNOWN until you observe it via frame_changed=True. Do NOT
    assume ACTION6 "clicks", "places markers", "selects", or "matches
    targets" -- those are priors from OTHER games, not this one.
  - PICK an obj_id from [CLICK TARGETS] and use ITS coords; DISAPPEARED
    or WRITTEN OFF entries (low confidence, all-no-op) are evidence
    that ACTION6 does nothing on those objects.
  - Do NOT invent (x, y). Do NOT copy any specific (x, y) pair verbatim
    from this system prompt or any block in the user prompt.
  - HARD RULE: If KNOWLEDGE.rules or KNOWLEDGE.failed_strategies say
    ACTION6 is no-op on tested coords (auto-derived from 5+ tries),
    STOP picking ACTION6 entirely. Pick a different ACTION1..ACTION5
    or ACTION7. The game may not respond to coordinate input at all.

OUTPUT FORMAT (strict, two lines, no JSON, no markdown):
  reasoning: <subject + expected effect, one sentence>
  action: <ACTION1..ACTION5 or ACTION7>         (no params)
       or <ACTION6 x y>                         (x, y in 0..63)

Examples (placeholders -- do NOT copy the numbers):
  reasoning: ACTION1 moves the maroon 1x1 (obj_002) UP by 3 cells; goal is top edge
  action: ACTION1

  reasoning: clicking obj_007 (cyan 1x1 in [CLICK TARGETS]) to test if it advances level
  action: ACTION6 <x_of_obj_007> <y_of_obj_007>

INVALID:
  - Coordinates on ACTION1..5 or ACTION7
  - JSON / markdown / extra prose
  - Reasonings without a subject ("try edge" / "explore" alone)
"""


REFLECTION_SYSTEM = """You are the Reflection Agent. You update Knowledge
based on what just happened.

Per-step inputs you see:
  - Current KNOWLEDGE (action_semantics, goal_hypothesis, rejected_goals,
    rules, failed_strategies, round_history)
  - State context (active objects with positions, same-color groups, etc.)
  - This step's action + reasoning + outcome + matches_reasoning verdict
  - The last few steps for short context
  - [EXPLORATION HINT] showing untried actions + uninteracted objects

You output STRICT JSON with EXACTLY THREE fields. The orchestrator
deterministically computes everything else (rules, failed_strategies,
goal_confidence, stuck alerts) from OutcomeLog -- you do NOT write them.

OUTPUT SCHEMA (no other keys, no prose, no markdown fences):

{
  "goal_hypothesis_update": "..." or null,
  "action_semantics_update": {"ACTION_X": "..."},
  "current_alert": ""
}

(1) goal_hypothesis_update
    What's the win condition? Look at active objects' configuration
    (same-color groups -> matching or alignment; objects near an edge
    -> targets; mover + matching static -> bring them together).

    Rules:
      - Describe the TARGET STATE in plain language, not an action.
        Good:  "match every red dot with a red target square"
        Bad:   "ACTION1 should move up"
      - Do NOT re-propose a goal already in `rejected_goals` -- it has
        been tried and disproved. The orchestrator will drop such
        updates anyway.
      - IF YOUR PROPOSED GOAL IS THE SAME AS THE ONE ALREADY IN KNOWLEDGE,
        WRITE null. Do NOT restate the existing goal verbatim every step
        -- that wastes tokens. Only write a string when the goal CHANGES.
      - If you don't have a real guess, set this to null. Do NOT write
        "unknown" / "none" / "" as a string -- only null.

(2) action_semantics_update -- BE EAGER, NOT CAUTIOUS

    HARD CONSTRAINT: If THIS step's OUTCOME shows frame_changed=True
    with primary_direction set (UP/DOWN/LEFT/RIGHT/null+nonzero distance)
    AND the executed ACTION has no entry in current KNOWLEDGE.action_semantics,
    you MUST write action_semantics_update for that ACTION this turn.
    Empty action_semantics across 5+ frame-change steps is a
    CORRECTNESS FAILURE -- the Action Agent can't learn what each action
    does without this.

    Write a one-line description naming the SUBJECT and the effect.
    Required subject forms:
      - color + shape  (e.g. "the red 1x1", "the yellow square")
      - obj_id         (e.g. "obj_002")
      - shape only     (e.g. "the 2x2 block", "the L-shape")
    "an active object" / "the object" / "a tracked object" are NOT
    valid subjects -- the orchestrator drops such updates.

    For non-positional changes (primary_direction=null but frame_changed=True),
    describe what visually changed instead, e.g. "rotates the red square 90 degrees"
    or "toggles the cyan 1x1's color".

    If you previously wrote an entry for ACTION_X and a NEW observation
    shows a DIFFERENT effect, write a CONDITIONAL clause instead of
    clobbering:
      "ACTION7: reshapes the red square when adjacent to a target;
                moves the yellow 1x1 DOWN by 3 cells otherwise"

    Use {} ONLY when (a) frame_changed=False, or (b) you already have a
    correct entry for the executed ACTION in KNOWLEDGE.action_semantics
    AND this step's outcome matches that entry. Otherwise, write the
    update.

(3) current_alert
    Only fill this when matches_reasoning == "NO" (the Action Agent's
    reasoning predicted X but the outcome was the opposite). The
    orchestrator already handles stuck loops, no_op streaks, and state
    revisits -- you do NOT need to alert on those.

    Even for matches_reasoning == "NO", you may leave this "" -- the
    orchestrator writes a fallback alert in that case.

    Make it SHORT (<140 chars) and SPECIFIC (name the action and the
    contradicted expectation). Use "" for "no alert".

WORKED EXAMPLE. step 4 outcome:
  action=ACTION1, frame_changed=True, primary_direction=UP, distance=3,
  moved object: obj_002 (color=yellow, shape=1x1)
Correct response (subject named, only the relevant field written):
{
  "goal_hypothesis_update": null,
  "action_semantics_update": {"ACTION1": "moves the yellow 1x1 (obj_002) UP by 3 cells"},
  "current_alert": ""
}

WRONG: writing rules_append / failed_strategies_append / goal_confidence_update.
Those are orchestrator-owned; including them is harmless but wastes tokens.
"""


# ─── Action USER prompt ─────────────────────────────────────────────────────


def _format_state_block(
    *,
    step: int,
    max_steps: int,
    level: int,
    total_levels: int,
    state: str,
    legal_actions: list[str],
    object_memory: Any,
    object_relations: Optional[Any] = None,
) -> str:
    """Compact [STATE] block: status header + active object list + relations.

    Replaces three v3 blocks ([STATUS], [ACTIVE], [OBJECT RELATIONS]) with
    one to cut redundancy and prompt length. Texture is dropped entirely --
    it was almost always "(none)" anyway, and the LLM doesn't need it.
    """
    def _sig(a: str) -> str:
        return "(x, y in 0..63)" if a == "ACTION6" else ""
    actions_annotated = ", ".join(
        (a + " " + _sig(a)).strip() for a in legal_actions
    )

    lines: list[str] = ["[STATE]"]
    lines.append(f"  step: {step} / {max_steps}    level: {level} / {total_levels}    game: {state}")
    lines.append(f"  legal actions: {actions_annotated}")

    active = object_memory.alive_tracked() if object_memory is not None else []
    if not active:
        lines.append("  active objects: (none yet -- every cell looks static)")
    else:
        lines.append("  active objects:")
        for t in active:
            if not t.history:
                continue
            last = t.history[-1]
            r0, c0, r1, c1 = last.bbox
            line = (
                f"    {t.uid}: {last.color_name} "
                f"size={last.size} bbox=[{r0},{c0},{r1},{c1}]"
            )
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
                    line += f"  (last step: moved {dist} cell(s) {'+'.join(parts)})"
            lines.append(line)

    # Object relations folded inline so we don't need another top-level block
    if object_relations is not None:
        try:
            rel_text = render_relations_block(object_relations)
            # render_relations_block returns "[OBJECT RELATIONS]\n  ..." —
            # strip its header and indent under [STATE]
            stripped = rel_text.split("\n", 1)[1] if "\n" in rel_text else ""
            if stripped.strip():
                lines.append("  relations:")
                for ln in stripped.splitlines():
                    # demote indentation by one level
                    lines.append("  " + ln)
        except Exception:
            pass

    return "\n".join(lines)


def build_action_user_prompt(
    *,
    knowledge: Knowledge,
    step: int,
    max_steps: int,
    level: int,
    total_levels: int,
    state: str,
    legal_actions: list[str],
    frame_objects: list[Any],
    layer_by_id: dict[int, Any],
    object_memory: Any,
    outcome_log: Any,
    # These four are kept for backward-compat with existing callers but are
    # no longer rendered as separate blocks in v3.2 -- they're either folded
    # into other blocks ([KNOWLEDGE].goal_hypothesis) or dropped as
    # redundant (v3's [TEXTURE] / [HISTORY] / [CLICK CANDIDATES] / [GOAL]
    # / [STUCK SIGNALS] / [UNTRIED] / [LOW-PRIORITY ACTIONS]).
    goal_hypothesis: str = "",
    goal_confidence: str = "low",
    diversification_hint: Optional[str] = None,
    stuck_reason: Optional[str] = None,
    click_candidates: Optional[list[Any]] = None,
    blocked_actions: Optional[set[str]] = None,
    object_relations: Optional[Any] = None,
    exploration_hint: Optional[str] = None,
    click_targets: Optional[list[ClickTarget]] = None,
) -> str:
    """Compose the v3.2 Action Agent USER prompt in 7 blocks max.

    Order (top -> bottom):
        [ALERT]            knowledge.current_alert when non-empty
        [KNOWLEDGE]        cross-round persistent learning
        [EXPLORATION HINT] untried actions + uninteracted objects + stuck
        [CLICK TARGETS]    ACTION6 bandit (when populated and ACTION6 legal)
        [STATE]            status + active objects + relations (merged)
        [ACTION stats]     per-action outcome stats from OutcomeLog
        [ASK]              two-line reasoning+action format

    Dropped vs v3.2-original (information preserved elsewhere):
        [REFLECTION ALERT] -> [ALERT]                (single alert channel)
        [LOW-PRIORITY ACTIONS] -> folded into [EXPLORATION HINT]
        [STATUS] / [ACTIVE] / [OBJECT RELATIONS] -> merged into [STATE]
        [TEXTURE]    -> dropped (almost always "(none)")
        [UNTRIED]    -> subsumed by [EXPLORATION HINT].untried
        [HISTORY]    -> subsumed by [ACTION stats]
        [GOAL]       -> subsumed by [KNOWLEDGE].goal_hypothesis
        [CLICK CANDIDATES] -> superseded by [CLICK TARGETS]
        [STUCK SIGNALS] -> subsumed by [ALERT] / [EXPLORATION HINT].STUCK

    `blocked_actions`, `goal_hypothesis`, `goal_confidence`,
    `diversification_hint`, `stuck_reason`, `click_candidates`,
    `frame_objects`, `layer_by_id` are accepted for API stability but no
    longer rendered as their own blocks. Their content reaches the LLM via
    [KNOWLEDGE] / [EXPLORATION HINT] / [ALERT] / [CLICK TARGETS] instead.
    """
    blocks: list[str] = []

    if knowledge.current_alert:
        blocks.append("[ALERT]\n" + knowledge.render_alert())

    blocks.append("[KNOWLEDGE - accumulated across rounds]\n" + knowledge.render())

    if exploration_hint:
        blocks.append(exploration_hint)

    if click_targets:
        ct_block = render_click_targets_block(click_targets)
        if ct_block:
            blocks.append(ct_block)

    blocks.append(_format_state_block(
        step=step, max_steps=max_steps,
        level=level, total_levels=total_levels,
        state=state, legal_actions=legal_actions,
        object_memory=object_memory,
        object_relations=object_relations,
    ))

    if outcome_log is not None:
        blocks.append(
            "[ACTION stats]\n" + render_action_block(outcome_log, legal_actions)
        )

    blocks.append(_ACTION_ASK_BLOCK)
    return "\n\n".join(blocks)


_ACTION_ASK_BLOCK = """[ASK]
  Output TWO lines (no JSON, no markdown):
    reasoning: <one sentence, mention the expected effect>
    action: <ACTION1..ACTION7 -- only ACTION6 takes x y>"""


# ─── Reflection USER prompt ─────────────────────────────────────────────────


def build_reflection_user_prompt(
    *,
    knowledge: Knowledge,
    step_summary: StepSummary,
    # New: full state context so Reflection can infer goals from object
    # configuration, not just step-level signals.
    step: Optional[int] = None,
    max_steps: Optional[int] = None,
    level: Optional[int] = None,
    total_levels: Optional[int] = None,
    state_name: Optional[str] = None,
    legal_actions: Optional[list[str]] = None,
    frame_objects: Optional[list[Any]] = None,
    layer_by_id: Optional[dict[int, Any]] = None,
    object_memory: Optional[Any] = None,
    outcome_log: Optional[Any] = None,
    object_relations: Optional[Any] = None,
    exploration_hint: Optional[str] = None,
) -> str:
    """Compose the Reflection USER prompt.

    Per `docs/ref_v3_2_dataflow_zh.md` updates 2026-05-14: Reflection now
    sees the same perception context as the Action Agent (active objects,
    object_relations, outcome log per-action stats, history) so it can
    infer goals from object configuration -- the previous version only
    saw a single-step summary which was too narrow to support real
    goal_hypothesis reasoning.

    All state-context kwargs are optional for backward compatibility with
    older callers / tests; when None, the corresponding block is omitted.
    """
    blocks: list[str] = []
    blocks.append("[CURRENT KNOWLEDGE before this step]\n" + knowledge.render())

    if exploration_hint:
        # Same deterministic block the Action Agent will see on the NEXT
        # step. Putting it here lets Reflection write a current_alert that
        # references unexplored actions / objects when it judges the agent
        # has been ignoring them.
        blocks.append(exploration_hint)

    # Status block when env-context provided
    if step is not None and max_steps is not None:
        status_lines = [
            f"  step: {step} / {max_steps}",
        ]
        if level is not None and total_levels is not None:
            status_lines.append(f"  level: {level} / {total_levels}")
        if state_name:
            status_lines.append(f"  game state: {state_name}")
        if legal_actions:
            status_lines.append(f"  legal actions: {', '.join(legal_actions)}")
        blocks.append("[STATUS]\n" + "\n".join(status_lines))

    # Active object snapshot (reuses prompts_v3 formatter)
    if object_memory is not None:
        try:
            from arc_agent.prompts_v3 import _format_active_block
            active = object_memory.alive_tracked()
            blocks.append(_format_active_block(active))
        except Exception:
            pass

    # Object relations (same-color/shape/distances) -- the key piece for
    # goal inference. "If two objects share a color, the goal often is
    # to bring them together / align them" -- see REFLECTION_SYSTEM.
    if object_relations is not None:
        try:
            from arc_agent.object_relations import render_relations_block
            blocks.append(render_relations_block(object_relations))
        except Exception:
            pass

    # Per-action effects observed (so Reflection sees empirical truth,
    # not just the single-step outcome)
    if outcome_log is not None and legal_actions:
        try:
            from arc_agent.action_inference import render_action_block
            blocks.append(
                "[ACTION effects observed]\n"
                + render_action_block(outcome_log, legal_actions)
            )
        except Exception:
            pass

    # Step summary stays as the focal "this step" detail
    blocks.append(step_summary.render())
    blocks.append(_REFLECTION_ASK_BLOCK)
    return "\n\n".join(blocks)


_REFLECTION_ASK_BLOCK = """[ASK]
Output STRICT JSON with all six fields. Use {}/[]/""/null for "no update".

For goal_hypothesis_update specifically:
  - You see the FULL current state above (active objects, their colors,
    shapes, positions, same-color groups, same-shape groups, distances).
  - A good goal describes the WIN STATE -- what the player needs to
    achieve, in plain language. Examples:
      "align the two red squares vertically in the left column"
      "every yellow object must reach the bottom edge"
      "match the moving blue square to the static blue target"
  - Patterns to USE the state info:
      * Same-color groups -> goal often involves bringing them together
      * Same-shape groups -> goal might be pairing or alignment
      * Static objects near one edge -> they may be targets / goals
      * Active object + same-color static object -> match them
  - DO NOT write action-style goals ("ACTION_X should ...") -- that's
    action_semantics, not a goal.
  - DO NOT re-propose any goal that already appears in `rejected_goals`
    in the KNOWLEDGE block above. Those have been tried and disproved;
    the orchestrator will silently drop them.
  - If you have insufficient evidence, set goal_hypothesis_update to
    null. Do NOT write "unknown" / "none" / "" as a string.

Write current_alert ONLY when the Action Agent's mental model is wrong
or the agent is stuck (no_op_streak >= 3 or state_revisit_count >= 3)."""


__all__ = [
    "ACTION_SYSTEM",
    "REFLECTION_SYSTEM",
    "build_action_user_prompt",
    "build_reflection_user_prompt",
]
