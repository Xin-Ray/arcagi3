"""VLMAgentLite (A1) and A1+h — minimal image -> action probe.

Per `docs/ARCHITECTURE_AGENTS.md` §1.A1, this agent strips everything except
visual action grounding: one image, one short ask, an `ACTIONx` line (or
`ACTIONx <x> <y>` for ACTION6) ≤ 8 tokens. No JSON, no entities, no
predicted_diff, no F1.

§3 Step 5 adds the A1+h variant via `history: int`. If history > 0, the last
N action names are appended to the prompt — that's the only difference.

The agent satisfies `arc_agent.runner.Agent` Protocol and exposes the same
`_state.last_prompt` / `last_response_raw` / `last_parse_ok` surface the
baseline runner reads, so it drops into `play_one_with_trace` unchanged.
Predicted_diff is permanently None (A1 has no F1 path).
"""
from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from arcengine import FrameDataRaw, GameAction, GameState

from arc_agent.observation import (
    available_action_names,
    grid_to_image,
    latest_grid,
)

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You play a turn-based grid game. Look at the image, then output exactly "
    "one action from {ACTION1..ACTION7}. ACTION1=up, ACTION2=down, "
    "ACTION3=left, ACTION4=right, ACTION5=interact, ACTION6=coordinate "
    "(needs x y in 0..63), ACTION7=undo. Output ONLY the action token."
)


@dataclass
class _LiteState:
    """Mirror of `_AgentState` so the baseline runner's `_extract_trace` works."""

    last_prompt: str = ""
    last_response_raw: str = ""
    last_parse_ok: bool = False
    last_predicted_diff: None = None   # A1 has no F1 path
    last_chosen_action: Optional[str] = None
    action_history: list[str] = field(default_factory=list)
    step_count: int = 0
    parse_failures: int = 0


_ACTION_RE = re.compile(r"\bACTION([1-7])\b", re.IGNORECASE)
_COORD_RE = re.compile(r"\bACTION6\b[^\d-]*?(\d+)\D+?(\d+)", re.IGNORECASE)


class VLMAgentLite:
    """Image-only Qwen2.5-VL agent: 1 image -> 1 action token."""

    DEFAULT_MAX_NEW_TOKENS = 8

    def __init__(
        self,
        *,
        backbone: Any = None,
        model_path: Optional[str] = None,
        seed: Optional[int] = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        history: int = 0,
    ) -> None:
        """Construct an A1 (or A1+h) agent.

        Args:
            backbone: object exposing `.generate(image, prompt, system=...)`.
                Inject a fake in tests; pass `HFBackbone.load(...)` in prod.
            model_path: lazy backbone load if `backbone is None`.
            seed: rng seed for the fallback-random branch.
            max_new_tokens: kept tiny (8) so a single action token fits.
            history: A1+h. If > 0, append the last N action names to the
                prompt. 0 = pure A1.
        """
        if history < 0:
            raise ValueError(f"history must be >= 0, got {history}")
        self._backbone = backbone
        self._model_path = model_path
        self._max_new_tokens = max_new_tokens
        self._history = history
        self._rng = random.Random(seed)
        self._state = _LiteState()

    # ── public API ────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._state = _LiteState()

    def choose(
        self, latest: FrameDataRaw, history: list[FrameDataRaw]
    ) -> GameAction:
        if latest.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        if not latest.frame:
            logger.warning("FrameDataRaw.frame empty — fallback random")
            return self._fallback_random(latest)

        s_t = latest_grid(latest)
        image = grid_to_image(s_t, scale=8)
        prompt = self._build_prompt(latest)
        self._state.last_prompt = _SYSTEM_PROMPT + "\n\n" + prompt

        backbone = self._ensure_backbone()
        try:
            response_raw = backbone.generate(
                image,
                prompt,
                system=_SYSTEM_PROMPT,
                max_new_tokens=self._max_new_tokens,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning("backbone.generate failed (%s) — fallback random", e)
            self._state.parse_failures += 1
            self._state.last_response_raw = ""
            self._state.last_parse_ok = False
            return self._fallback_random(latest)

        self._state.last_response_raw = response_raw
        action = self._coerce_action(response_raw, latest)

        if action is None:
            self._state.parse_failures += 1
            self._state.last_chosen_action = None
            self._state.last_parse_ok = False
            self._state.step_count += 1
            return self._fallback_random(latest)

        self._state.last_chosen_action = action.name
        self._state.last_parse_ok = True
        self._state.action_history.append(action.name)
        # Cap history buffer at a generous bound; the prompt uses last N.
        if len(self._state.action_history) > max(50, self._history * 2):
            self._state.action_history = self._state.action_history[-50:]
        self._state.step_count += 1
        action.reasoning = "vlm_lite"
        return action

    # ── prompt building ───────────────────────────────────────────────────

    def _build_prompt(self, latest: FrameDataRaw) -> str:
        legal = ", ".join(available_action_names(latest))
        lines = [
            f"Legal actions: {legal}",
        ]
        if self._history > 0 and self._state.action_history:
            tail = self._state.action_history[-self._history:]
            lines.append(f"Last {len(tail)} actions: {', '.join(tail)}")
        lines.append("Reply with one action token only (e.g. ACTION3, or 'ACTION6 12 30').")
        return "\n".join(lines)

    # ── response coercion ─────────────────────────────────────────────────

    def _coerce_action(
        self, text: Any, latest: FrameDataRaw
    ) -> Optional[GameAction]:
        if not isinstance(text, str):
            return None
        m = _ACTION_RE.search(text)
        if not m:
            return None
        try:
            action = GameAction[f"ACTION{m.group(1)}"]
        except KeyError:
            return None
        if action.value not in latest.available_actions:
            return None

        if action.is_complex():
            x = y = None
            cm = _COORD_RE.search(text)
            if cm:
                x = int(cm.group(1))
                y = int(cm.group(2))
            if x is None or y is None or not (0 <= x <= 63 and 0 <= y <= 63):
                # ACTION6 without parseable coords → invalid for A1 (fall back).
                return None
            action.set_data({"x": x, "y": y})

        return action

    # ── helpers ───────────────────────────────────────────────────────────

    def _ensure_backbone(self) -> Any:
        if self._backbone is not None:
            return self._backbone
        if self._model_path is not None:
            from arc_agent.vlm_backbone import HFBackbone
            self._backbone = HFBackbone.load(model_path=self._model_path)
            return self._backbone
        raise RuntimeError(
            "VLMAgentLite has no backbone — pass `backbone=` or `model_path=`"
        )

    def _fallback_random(self, latest: FrameDataRaw) -> GameAction:
        legal = [v for v in latest.available_actions if v != GameAction.RESET.value]
        if not legal:
            return GameAction.RESET
        action = GameAction.from_id(self._rng.choice(legal))
        if action.is_complex():
            action.set_data({
                "x": self._rng.randint(0, 63),
                "y": self._rng.randint(0, 63),
            })
        action.reasoning = "fallback: random over legal actions"
        return action
