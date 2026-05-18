"""LLM-based goal-achievement judge (counterpart to goal_evaluator.py).

Given (goal_hypothesis, current objects), asks an LLM to answer YES/NO
"is the goal achieved?". Designed as an A/B comparison target for the
deterministic Python parser.

Expected to be SLOWER and LESS RELIABLE than the deterministic path
(T-GOAL bench showed SmolLM3-3B long_acc 33.7%), but LARGER coverage
(handles arbitrary free-form hypotheses).

The natural production pattern is HYBRID: try deterministic first; if
returns None, fall back to LLM. This module enables measuring that
tradeoff.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from arc_agent.object_extractor import ObjectRecord


_SYSTEM = (
    "You judge whether a stated goal is achieved given current state. "
    "Respond with EXACTLY ONE LINE: 'Answer: YES' or 'Answer: NO'."
)

_YESNO_RE = re.compile(r"answer[:\s]*(yes|no)", re.IGNORECASE)


def _format_objects(objects: Sequence[ObjectRecord]) -> str:
    if not objects:
        return "  (no visible objects)"
    lines = []
    for o in objects[:20]:  # cap to keep prompt bounded
        r = int(round(o.center[0]))
        c = int(round(o.center[1]))
        lines.append(f"  - {o.color_name} object at (row={r}, col={c}), "
                     f"size={o.size}")
    return "\n".join(lines)


def build_prompt(hypothesis: str, objects: Sequence[ObjectRecord]) -> str:
    """Compose the user prompt for the LLM judge."""
    obj_block = _format_objects(objects)
    return (
        f"Goal hypothesis: {hypothesis!r}\n\n"
        f"Current objects on the 64x64 grid:\n{obj_block}\n\n"
        f"Solve step by step. First identify what the hypothesis "
        f"requires structurally (positions, alignment, etc). Then "
        f"check whether the current objects satisfy that. End with: "
        f"Answer: YES or Answer: NO"
    )


def parse_yesno(text: str) -> Optional[bool]:
    """Pull YES/NO from raw LLM output. None on parse failure."""
    if not isinstance(text, str):
        return None
    m = _YESNO_RE.search(text)
    if not m:
        return None
    return m.group(1).lower() == "yes"


def judge_goal(
    hypothesis: str,
    objects: Sequence[ObjectRecord],
    *,
    backbone: Any,
    max_new_tokens: int = 1024,
) -> tuple[Optional[bool], str]:
    """Ask the LLM whether `hypothesis` is achieved given `objects`.

    Returns (verdict, raw_text). `verdict` is True/False if parseable,
    None on parse failure or empty hypothesis.
    """
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        return None, ""

    user_prompt = build_prompt(hypothesis, objects)
    try:
        raw = backbone.generate(
            None, user_prompt,
            system=_SYSTEM,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
    except TypeError:
        # Older fake backbones reject image=None
        class _Img:
            size = (1, 1); mode = "RGB"
            def convert(self, _m): return self
        raw = backbone.generate(
            _Img(), user_prompt,
            system=_SYSTEM,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )

    raw = raw if isinstance(raw, str) else ""
    verdict = parse_yesno(raw)
    return verdict, raw


__all__ = ["build_prompt", "parse_yesno", "judge_goal"]
