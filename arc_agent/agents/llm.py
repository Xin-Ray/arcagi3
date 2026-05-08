"""LLMAgent — prompt-level HEI on Claude.

Each step the agent sends a (cached) system prompt + dynamic user message
describing the current frame, then parses the LLM reply for an action.

The system prompt is the cacheable prefix; the user message is fully dynamic.
With the static system being the bulk of input tokens, prompt cache should
amortize most cost across an episode.
"""
from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Optional

from arcengine import FrameDataRaw, GameAction, GameState

from arc_agent.llm import LLMClient
from arc_agent.observation import grid_diff, latest_grid, summarize_frame

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an agent playing an ARC-AGI-3 game (an interactive turn-based grid game).
You have NO instructions about the game's rules, goal, or what 'winning' means — you must figure them out by interacting.

Each turn you observe a 64x64 grid (16 colors, encoded as hex chars 0..F per cell), the legal-actions set, and (when applicable) what changed since your last action.

Your job is to pick the action that either:
  (a) makes progress toward what you believe is the win condition, OR
  (b) maximizes information gain about how the game works.

You will reason in three explicit blocks every turn (Hypothesize, Execute, Iterate — the scientific method, since with no reward signal it's the only sound way to learn):

HYPOTHESIS: One sentence about what you currently believe about the rules / goal.
EXECUTE: One sentence stating which action you'll take and why (V or IG dominant).
ITERATE: One sentence on what you learned from the previous step's outcome (skip on the first step).

Then END your reply with EXACTLY one line of this form:

ACTION: ACTION1                       # for simple actions
ACTION: ACTION6 x=32 y=10             # for the coordinate action (must have x and y)

Action semantics (likely; verify by experiment):
- ACTION1=Up, ACTION2=Down, ACTION3=Left, ACTION4=Right (4 directional keys)
- ACTION5=Primary interact (game-specific: rotate, select, etc.)
- ACTION6=Coordinate click at (x, y), x and y in [0, 63]
- ACTION7=Undo last action

Rules:
- ONLY pick an action listed in `available_actions` for the current step.
- For ACTION6, ALWAYS provide x and y in the [0, 63] range.
- Do not output anything after the ACTION: line.
- Be concise. Total response < 200 words."""


@dataclass
class _AgentState:
    """Per-episode mutable state — last action and last grid (for diff in next step)."""

    last_action: Optional[GameAction] = None
    last_grid: Optional[object] = None  # numpy array; typed loosely to avoid import here
    step_count: int = 0
    parse_failures: int = 0
    history: list[str] = field(default_factory=list)  # last few HEI summaries (LLM responses)


class LLMAgent:
    """Picks actions via Claude. Falls back to random if parsing fails."""

    HISTORY_LIMIT = 3  # how many of the agent's own past HEI replies to include

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        *,
        seed: Optional[int] = None,
    ) -> None:
        self.llm = llm or LLMClient()  # default model = claude-opus-4-7 per claude-api skill
        self._rng = random.Random(seed)
        self._state = _AgentState()

    def reset(self) -> None:
        """Clear per-episode state. Call between episodes if reusing the agent."""
        self._state = _AgentState()

    def choose(
        self, latest: FrameDataRaw, history: list[FrameDataRaw]
    ) -> GameAction:
        # RESET handles state transitions; LLM doesn't need to think about it.
        if latest.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        user_msg = self._build_user_message(latest)
        try:
            resp = self.llm.complete(system=SYSTEM_PROMPT, user=user_msg)
        except Exception as e:
            logger.warning("LLM call failed (%s) — falling back to random", e)
            return self._fallback_random(latest)

        action = self._parse_action(resp.text, latest)
        if action is None:
            self._state.parse_failures += 1
            logger.warning(
                "Could not parse action from LLM reply (failure %d). Reply: %r",
                self._state.parse_failures,
                resp.text[-200:],
            )
            return self._fallback_random(latest)

        # Stash for next step's "what changed" + history pruning
        self._state.last_action = action
        if latest.frame:
            self._state.last_grid = latest_grid(latest)
        self._state.step_count += 1
        # Keep just enough history to give Claude continuity without ballooning tokens
        self._state.history.append(self._terse_response(resp.text))
        if len(self._state.history) > self.HISTORY_LIMIT:
            self._state.history = self._state.history[-self.HISTORY_LIMIT:]
        return action

    # --- internal -------------------------------------------------------------

    def _build_user_message(self, latest: FrameDataRaw) -> str:
        parts = [
            f"Game: {latest.game_id}",
            f"Step: {self._state.step_count + 1}",
            f"Levels completed: {latest.levels_completed}/{latest.win_levels}",
        ]
        if self._state.last_action is not None:
            parts.append(f"Last action you took: {self._state.last_action.name}")

        # Frame summary (state, available actions, full grid, optional diff)
        parts.append("")
        parts.append(summarize_frame(latest, diff_with=self._state.last_grid))

        if self._state.history:
            parts.append("")
            parts.append("Your recent reasoning (most recent last):")
            for i, h in enumerate(self._state.history, 1):
                parts.append(f"--- step {self._state.step_count - len(self._state.history) + i} ---")
                parts.append(h)

        return "\n".join(parts)

    def _parse_action(
        self, text: str, latest: FrameDataRaw
    ) -> Optional[GameAction]:
        # Find the LAST occurrence of "ACTION: ..." — the LLM may use the word
        # "ACTION" earlier in its reasoning, but the directive line is the last.
        m = list(re.finditer(
            r"^\s*ACTION:\s*(ACTION[1-7])(?:\s+x=(-?\d+))?(?:\s+y=(-?\d+))?\s*$",
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        ))
        if not m:
            return None
        match = m[-1]
        name = match.group(1).upper()
        try:
            action = GameAction[name]
        except KeyError:
            return None

        # Action must be in available_actions
        if action.value not in latest.available_actions:
            logger.warning(
                "LLM picked %s but only %s are legal — falling back to random",
                name, latest.available_actions,
            )
            return None

        if action.is_complex():
            x_str, y_str = match.group(2), match.group(3)
            if x_str is None or y_str is None:
                return None
            x, y = int(x_str), int(y_str)
            if not (0 <= x <= 63 and 0 <= y <= 63):
                return None
            action.set_data({"x": x, "y": y})

        action.reasoning = self._terse_response(text)
        return action

    def _fallback_random(self, latest: FrameDataRaw) -> GameAction:
        # Random over the actually-legal set (not all GameAction enum values)
        legal_ids = [
            v for v in latest.available_actions if v != GameAction.RESET.value
        ]
        if not legal_ids:
            return GameAction.RESET
        action = GameAction.from_id(self._rng.choice(legal_ids))
        if action.is_complex():
            action.set_data(
                {"x": self._rng.randint(0, 63), "y": self._rng.randint(0, 63)}
            )
        action.reasoning = "fallback: random over legal actions"
        return action

    @staticmethod
    def _terse_response(text: str, *, max_chars: int = 400) -> str:
        # Compact the LLM reply for the next-step history; drop trailing noise after ACTION:
        cleaned = text.strip()
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars] + "...[truncated]"
        return cleaned
