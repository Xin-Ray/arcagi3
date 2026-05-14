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

from arc_agent.knowledge import Knowledge
from arc_agent.prompts_v3 import build_play_user_prompt
from arc_agent.step_summary import StepSummary


# ─── SYSTEM constants ───────────────────────────────────────────────────────

ACTION_SYSTEM = """You are the Action Agent for a turn-based 64x64 grid game.

The Reflection Agent has provided KNOWLEDGE accumulated across previous
rounds (and the previous steps of this round):
  - action_semantics: what each ACTION does in this game (when known)
  - goal_hypothesis: the most likely goal so far
  - rules: patterns observed across previous rounds
  - failed_strategies: strategies that were tried and did NOT work

Trust the KNOWLEDGE block. Do NOT re-explore things already documented
as failed_strategies. If a [REFLECTION ALERT] block is present at the
top of the prompt, the Reflection Agent has flagged that your previous
mental model was wrong -- read it FIRST and change behavior accordingly.

The user prompt also gives you v3 enriched context:
  [STATUS]   step, level, legal actions (with parameter signatures)
  [ACTIVE]   tracked objects with movement history
  [TEXTURE]  static cells filtered out
  [ACTION]   observed effects of each action THIS round
  [UNTRIED]  legal actions you have not tried yet
  [HISTORY]  the last 5 (action, frame_changed) tuples
  [GOAL]     current hypothesis from v3 (the [KNOWLEDGE] goal is preferred
             when they disagree)

OUTPUT FORMAT (strict, two lines, no JSON, no markdown):
  reasoning: <one sentence explaining your choice; mention the expected
             effect so the Reflection Agent can judge it>
  action: ACTION1..ACTION5 / ACTION7  (no params)
          ACTION6 <x> <y>             (x, y in 0..63 -- ACTION6 ONLY)

Valid:
  reasoning: knowledge says ACTION1 moves the player up; goal is the top
  action: ACTION1

  reasoning: failed_strategies says clicking near (32,32) doesn't help; try edge
  action: ACTION6 5 60

INVALID:
  Do NOT add coordinates to ACTION1..5 or ACTION7.
  Do NOT output JSON. Just two plain lines: reasoning and action.
"""


REFLECTION_SYSTEM = """You are the Reflection Agent for an in-episode learning loop.

After EACH step you see:
  - The current KNOWLEDGE (what you've learned across all prior steps of
    this round + all prior rounds of this game)
  - The Action Agent's REASONING from this step (what it expected)
  - This step's actual OUTCOME: action taken, what changed in the grid,
    what moved in ObjectMemory, no_op_streak, state_revisit_count, and
    a pre-computed matches_reasoning verdict (YES / PARTIAL / NO / N/A)
  - The last 3 steps for short context

Your two main jobs:

  (A) UPDATE KNOWLEDGE -- be EAGER, not cautious.
      - action_semantics[ACTION_X]: as soon as you see ONE specific
        effect (e.g. "frame_changed=True, primary_direction=UP, distance=3"),
        WRITE the entry. You can REFINE it next time you observe the
        same action. An empty action_semantics after 5+ steps is a FAILURE.
        Concrete rule: if primary_direction is not null, the entry MUST
        name the direction + distance + what kind of object moved.
        If frame_changed=False, write "ACTION_X at <coords>: no observable
        effect" so the agent stops trying it.
      - rules: append a one-line pattern when 2+ steps agree
        (e.g. "ACTION6 has no effect on any tested coord").
      - failed_strategies: append when a strategy or coord region has
        clearly failed 3+ times. Keep these high-level (not "ACTION6"
        bare -- say "ACTION6 anywhere in the right half").
      - goal_hypothesis_update: ONLY when you have a real guess. If you
        don't, set this to null (NOT the literal string "unknown").
        Never write "unknown" / "none" / "" as a string -- use null.
      - goal_confidence_update: raise/lower per evidence; null if no change.

  (B) WRITE current_alert when the Action Agent's mental model is wrong.
      Trigger an alert when ANY of these holds:
        - matches_reasoning == "NO"
        - no_op_streak >= 3
        - state_revisit_count >= 3 (the agent is in a loop)
        - same action chosen 5+ times in a row with no progress
        - reasoning is generic/non-committal for 3+ steps
      Make the alert SHORT (<140 chars) and SPECIFIC (name the action,
      the wrong expectation, and what to try). Otherwise leave it "".

Output STRICT JSON only -- no prose, no markdown fences:

{
  "action_semantics_update": {"ACTION3": "..."},
  "goal_hypothesis_update": "..." or null,
  "goal_confidence_update": "low" or "medium" or "high" or null,
  "rules_append": ["..."],
  "failed_strategies_append": ["..."],
  "current_alert": ""
}

Concrete worked example. Suppose step 4 outcome is:
  action=ACTION1, frame_changed=True, primary_direction=UP, distance=3
Correct response:
{
  "action_semantics_update": {"ACTION1": "moves an active object UP by 3 cells"},
  "goal_hypothesis_update": null,
  "goal_confidence_update": null,
  "rules_append": [],
  "failed_strategies_append": [],
  "current_alert": ""
}

Every field is REQUIRED. Use {} / [] / "" / null for "no update".
"""


# ─── Action USER prompt ─────────────────────────────────────────────────────


def build_action_user_prompt(
    *,
    knowledge: Knowledge,
    # v3 blocks — passed through to build_play_user_prompt
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
    goal_hypothesis: str = "",
    goal_confidence: str = "low",
    diversification_hint: Optional[str] = None,
    stuck_reason: Optional[str] = None,
    click_candidates: Optional[list[Any]] = None,
    blocked_actions: Optional[set[str]] = None,
    object_relations: Optional[Any] = None,
) -> str:
    """Compose the Action Agent USER prompt.

    Three new blocks above v3's:
      - [REFLECTION ALERT]  (only if knowledge.current_alert is non-empty)
      - [BLOCKED ACTIONS]   (R7; only if blocked_actions is non-empty)
      - [KNOWLEDGE]
    The v3 [ASK] block at the end is replaced with one demanding the
    reasoning + action two-line format.

    `blocked_actions` is the same set the orchestrator computes via
    `action_mask.compute_action_mask`. Showing it in the prompt lets the
    LLM avoid wasting picks on actions that will be silently replaced.
    """
    v3_body = build_play_user_prompt(
        step=step, max_steps=max_steps,
        level=level, total_levels=total_levels,
        state=state, legal_actions=legal_actions,
        frame_objects=frame_objects, layer_by_id=layer_by_id,
        object_memory=object_memory, outcome_log=outcome_log,
        goal_hypothesis=goal_hypothesis, goal_confidence=goal_confidence,
        diversification_hint=diversification_hint,
        stuck_reason=stuck_reason,
        click_candidates=click_candidates,
        object_relations=object_relations,
    )

    # Strip the trailing v3 [ASK] block — we'll append the v3.2 version.
    sep = "\n\n[ASK]"
    if sep in v3_body:
        v3_body_no_ask = v3_body.rsplit(sep, 1)[0]
    else:
        v3_body_no_ask = v3_body

    blocks: list[str] = []
    if knowledge.current_alert:
        blocks.append("[REFLECTION ALERT]\n" + knowledge.render_alert())
    if blocked_actions:
        # Sort for deterministic output; orchestrator will reject these picks
        sorted_blocked = sorted(blocked_actions)
        blocks.append(
            "[BLOCKED ACTIONS - orchestrator will REPLACE these picks]\n"
            "  " + ", ".join(sorted_blocked) + "\n"
            "  Picking any of these wastes your turn -- the orchestrator\n"
            "  will silently substitute an untried or known-good action.\n"
            "  Choose from the remaining legal actions instead."
        )
    blocks.append("[KNOWLEDGE - accumulated across rounds]\n" + knowledge.render())
    blocks.append(v3_body_no_ask)
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
) -> str:
    """Compose the Reflection USER prompt for ONE step (per §4.3)."""
    blocks: list[str] = []
    blocks.append("[CURRENT KNOWLEDGE before this step]\n" + knowledge.render())
    blocks.append(step_summary.render())
    blocks.append(_REFLECTION_ASK_BLOCK)
    return "\n\n".join(blocks)


_REFLECTION_ASK_BLOCK = """[ASK]
Output STRICT JSON with all six fields. Use {}/[]/""/null for "no update".
Write current_alert ONLY if the Action Agent's mental model is wrong,
the agent is stuck, or revisits keep increasing."""


__all__ = [
    "ACTION_SYSTEM",
    "REFLECTION_SYSTEM",
    "build_action_user_prompt",
    "build_reflection_user_prompt",
]
