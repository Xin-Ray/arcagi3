"""ReflectionAgent (v3.2) -- per-step JSON delta producer.

Per `docs/arch_v3_2_zh.md` §4. After every env.step the orchestrator calls
`ReflectionAgent.reflect_after_step(knowledge, step_summary)`. The agent:

  1. Builds the Reflection USER prompt (prompts_v3_2.build_reflection_user_prompt).
  2. Generates with the Qwen backbone in text-only mode.
  3. Parses the response as strict JSON (tolerating ```json fences and
     leading prose); on parse failure returns an EMPTY delta so the
     orchestrator can keep going without losing accumulated Knowledge.

Returns (delta_dict, raw_text). The orchestrator merges the delta into
Knowledge via `knowledge.merged_with_delta(delta)`.

This agent stays minimal -- it owns no episode state of its own; all
context is passed in per call. The `_state` surface mirrors v3 TextAgent
so baseline trace capture sees `last_prompt` / `last_response_raw` /
`last_parse_ok`.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from arc_agent.knowledge import Knowledge
from arc_agent.prompts_v3_2 import (
    REFLECTION_SYSTEM,
    build_reflection_user_prompt,
)
from arc_agent.step_summary import StepSummary

logger = logging.getLogger(__name__)


_ALLOWED_DELTA_KEYS = {
    "action_semantics_update",
    "goal_hypothesis_update",
    "goal_confidence_update",
    "rules_append",
    "failed_strategies_append",
    "current_alert",
}

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class _ReflectionState:
    """Mirrors TextAgent._state for trace-capture compatibility."""
    last_prompt: str = ""
    last_response_raw: str = ""
    last_parse_ok: bool = False
    last_delta: dict[str, Any] = field(default_factory=dict)
    call_count: int = 0
    parse_failures: int = 0


def parse_reflection_output(text: Any) -> tuple[dict[str, Any], bool]:
    """Extract a delta dict from the Reflection raw output.

    Returns `({}, False)` on any failure -- the orchestrator interprets
    that as "no change". Tolerates:
      - bare JSON object
      - JSON in ```json ... ``` fence
      - JSON with leading/trailing prose (first balanced `{...}` wins)
    Bad inner types are NOT scrubbed here -- `Knowledge.merged_with_delta`
    is tolerant of bad keys/values, so we let it filter.
    """
    if not isinstance(text, str) or not text.strip():
        return {}, False

    candidate = text.strip()

    # Strip ```json ... ``` fence if present
    fence_match = _JSON_FENCE_RE.search(candidate)
    if fence_match:
        candidate = fence_match.group(1).strip()

    # Fast path -- the whole thing is a JSON object
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        parsed = _extract_first_json_object(candidate)

    if not isinstance(parsed, dict):
        return {}, False

    # Drop unknown keys to keep merge clean; merge tolerates this anyway.
    cleaned = {k: v for k, v in parsed.items() if k in _ALLOWED_DELTA_KEYS}
    # If literally nothing recognized, treat as parse failure
    if not cleaned and not any(k in parsed for k in _ALLOWED_DELTA_KEYS):
        return {}, False
    return cleaned, True


def _extract_first_json_object(text: str) -> Optional[dict]:
    """Find the first balanced `{...}` substring and json.loads it."""
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    snippet = text[start:i + 1]
                    try:
                        return json.loads(snippet)
                    except (json.JSONDecodeError, ValueError):
                        # Continue scanning for the next balanced block
                        start = -1
    return None


class ReflectionAgent:
    """v3.2 Reflection Agent."""

    DEFAULT_MAX_NEW_TOKENS = 250

    def __init__(
        self,
        *,
        backbone: Any = None,
        model_path: Optional[str] = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.0,
    ) -> None:
        self._backbone = backbone
        self._model_path = model_path
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._state = _ReflectionState()

    def reset(self) -> None:
        self._state = _ReflectionState()

    def reflect_after_step(
        self,
        *,
        knowledge: Knowledge,
        step_summary: StepSummary,
    ) -> tuple[dict[str, Any], str]:
        """One reflection turn. Returns (delta, raw_response).

        On any failure (backbone error, parse error, empty response) the
        delta is {} and raw_response holds whatever we got (possibly "").
        The orchestrator should still call `knowledge.merged_with_delta(
        delta)` -- empty dict means "no change".
        """
        user_prompt = build_reflection_user_prompt(
            knowledge=knowledge,
            step_summary=step_summary,
        )
        self._state.last_prompt = REFLECTION_SYSTEM + "\n\n" + user_prompt
        self._state.call_count += 1

        backbone = self._ensure_backbone()
        try:
            raw = backbone.generate(
                None, user_prompt,
                system=REFLECTION_SYSTEM,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
            )
        except TypeError:
            # Older fake backbones reject image=None
            raw = backbone.generate(
                _PlaceholderImage(), user_prompt,
                system=REFLECTION_SYSTEM,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
            )
        except Exception as e:
            logger.warning("reflection backbone failed (%s) -- empty delta", e)
            self._state.last_response_raw = ""
            self._state.last_parse_ok = False
            self._state.last_delta = {}
            self._state.parse_failures += 1
            return {}, ""

        self._state.last_response_raw = raw if isinstance(raw, str) else ""
        delta, ok = parse_reflection_output(raw)
        self._state.last_parse_ok = ok
        self._state.last_delta = delta
        if not ok:
            self._state.parse_failures += 1
        return delta, self._state.last_response_raw

    # ── helpers ─────────────────────────────────────────────────────────

    def _ensure_backbone(self) -> Any:
        if self._backbone is not None:
            return self._backbone
        if self._model_path is not None:
            from arc_agent.vlm_backbone import HFBackbone
            self._backbone = HFBackbone.load(model_path=self._model_path)
            return self._backbone
        raise RuntimeError(
            "ReflectionAgent has no backbone -- pass `backbone=` or `model_path=`")


class _PlaceholderImage:
    """Stub for fake backbones that require a positional image."""
    size = (1, 1)
    mode = "RGB"

    def convert(self, _mode):
        return self


__all__ = [
    "ReflectionAgent",
    "parse_reflection_output",
]
