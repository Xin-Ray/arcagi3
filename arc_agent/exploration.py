"""Exploration aids for the v3.2 Action / Reflection prompts.

The user's observation (2026-05-14): once Reflection writes a few
action_semantics entries, the Action Agent converges to those actions and
stops trying the rest. v2 (commit 1bac4be) accidentally avoided this via
a hard mask-override; ABCD removed the override and entropy collapsed.

Rather than re-adding the override, we surface the unexplored set as a
deterministic, orchestrator-computed prompt block. The LLM keeps choice
authority but sees an explicit list of:

  - actions it has never tried this round
  - tracked objects that have never reacted to any action

Plus an optional stuck signal (from `stuck_detector.detect_repeat_stuck`)
that explicitly tells the agent the masked frame hash has repeated.

All functions are pure: no Knowledge mutation, no file I/O. The orchestrator
calls them every step and threads results into the prompts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class UninteractedObject:
    """Lightweight prompt-facing summary of an object that hasn't moved."""
    uid: str
    color_name: str
    bbox: tuple[int, int, int, int]      # (r0, c0, r1, c1)
    center: tuple[int, int]              # (row, col), rounded ints
    seen_steps: int

    def render(self) -> str:
        r0, c0, r1, c1 = self.bbox
        return (
            f"- {self.uid} ({self.color_name}, "
            f"bbox=[{r0},{c0},{r1},{c1}], center~({self.center[0]},{self.center[1]}), "
            f"seen for {self.seen_steps} steps)"
        )


def compute_untried_actions(
    outcome_log: Any,
    legal_actions: list[str],
) -> list[str]:
    """Wrap `OutcomeLog.untried` so callers in the orchestrator don't depend
    on the action_inference module directly."""
    if outcome_log is None:
        return list(legal_actions)
    return list(outcome_log.untried(legal_actions))


def compute_uninteracted_objects(
    object_memory: Any,
    *,
    min_seen_steps: int = 5,
) -> list[UninteractedObject]:
    """Tracked objects that have been alive for `min_seen_steps`+ but whose
    bbox center has never moved.

    Returns at most a few per call (orchestrator typically caps the prompt
    block to ~5 entries; we don't cap here -- caller decides).

    Notes:
      - `object_memory` is `ObjectMemory` from `arc_agent.object_tracker`.
      - We treat *no movement of bbox center* as "uninteracted". This is a
        proxy: a static decoration counts as uninteracted but so does a
        button no one has clicked. That's exactly what we want -- both
        deserve a try.
      - Rounded int centers so the prompt is short.
    """
    if object_memory is None:
        return []
    out: list[UninteractedObject] = []
    for tracked in object_memory.alive_tracked():
        history = tracked.history
        if len(history) < min_seen_steps:
            continue
        centers = {(int(round(snap.center[0])), int(round(snap.center[1])))
                   for snap in history}
        if len(centers) > 1:
            continue   # the object has moved at least once
        last = history[-1]
        out.append(UninteractedObject(
            uid=tracked.uid,
            color_name=last.color_name,
            bbox=tuple(last.bbox),
            center=(int(round(last.center[0])), int(round(last.center[1]))),
            seen_steps=len(history),
        ))
    return out


def render_exploration_hint(
    untried_actions: list[str],
    uninteracted_objects: list[UninteractedObject],
    stuck_reason: Optional[str] = None,
    *,
    max_objects: int = 5,
) -> str:
    """Build the [EXPLORATION HINT] prompt block.

    Returns an empty string when there's nothing to suggest -- caller should
    omit the block entirely in that case.
    """
    has_untried = bool(untried_actions)
    has_objects = bool(uninteracted_objects)
    has_stuck = bool(stuck_reason)
    if not (has_untried or has_objects or has_stuck):
        return ""

    lines: list[str] = ["[EXPLORATION HINT]"]
    if has_stuck:
        # Stuck reason goes FIRST so it's the most visible -- the LLM sees
        # the loop signal before the candidate actions.
        lines.append(f"  STUCK: {stuck_reason}")
    if has_untried:
        lines.append(
            "  Actions you have NOT tried this round: "
            + ", ".join(untried_actions)
        )
    if has_objects:
        lines.append(
            "  Objects that have NEVER reacted to any action "
            "(consider ACTION6 at their center):"
        )
        for obj in uninteracted_objects[:max_objects]:
            lines.append("    " + obj.render())
        if len(uninteracted_objects) > max_objects:
            lines.append(
                f"    ...and {len(uninteracted_objects) - max_objects} more"
            )
    # Closing nudge so the LLM knows what to do with the info.
    # NOTE: avoid the literal substring "[REFLECTION ALERT]" here -- some
    # tests grep the rendered prompt for that exact tag to detect whether
    # the alert block was rendered, and we don't want a false positive
    # from this hint block.
    lines.append(
        "  Unless the reflection-alert block above says otherwise, prefer "
        "an unexplored action OR direct ACTION6 at one of these objects'\n"
        "  centers before repeating an action that has already been tried."
    )
    return "\n".join(lines)


__all__ = [
    "UninteractedObject",
    "compute_untried_actions",
    "compute_uninteracted_objects",
    "render_exploration_hint",
]
