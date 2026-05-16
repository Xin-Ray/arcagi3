"""PlayReflectAgent (A3) + mistake-aware PlayReflectMistakesAgent (A4).

Implements `docs/ARCHITECTURE_AGENTS.md` §1 A3/A4:

- Play call every step: image + WorldModel + (A4 only) mistakes → ACTIONx token.
- Reflection call every K=5 steps OR on stuck-trigger: rewrites WorldModel
  as a typed JSON object (schema in `arc_agent.schemas.WORLD_MODEL_SCHEMA`).

Both agents:
- Reuse `VLMAgentLite._coerce_action` parsing (single ACTIONx token, no JSON
  in the play path).
- Implement the `Agent` Protocol from `arc_agent.runner` and surface
  `_state.last_prompt` / `last_response_raw` / `last_parse_ok` so the
  baseline runner can capture them.
- Take a single backbone — same Qwen instance is shared between Play and
  Reflect; the doc explicitly does NOT call for a second model.

Stuck-trigger: 3 consecutive unchanged frames OR an illegal action force a
Reflection call regardless of K.
"""
from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from arc_agent.mistakes import (
    MAX_MISTAKES,
    MistakeBuffer,
    StepRecord,
    frame_hash,
    update_mistakes,
)
from arc_agent.observation import (
    available_action_names,
    grid_to_image,
    latest_grid,
)
from arc_agent.schemas import WORLD_MODEL_SCHEMA
from arc_agent.world_model import WorldModel

logger = logging.getLogger(__name__)


REFLECT_EVERY = 5
NO_OP_TRIGGER = 3   # stuck-trigger threshold (frames unchanged in a row)


_PLAY_SYSTEM = (
    "You play a turn-based grid game. Use the WORLD MODEL block as context, "
    "then output ONE action token from {ACTION1..ACTION7}. ACTION1=up, "
    "ACTION2=down, ACTION3=left, ACTION4=right, ACTION5=interact, "
    "ACTION6=coordinate (needs x y in 0..63), ACTION7=undo. Output ONLY "
    "the action token."
)


_REFLECT_SYSTEM = (
    "You are the reflection step for a small VLM game agent. Look at the two "
    "images (BEFORE and AFTER recent play), the last few action outcomes, "
    "and the current WORLD MODEL. Return an updated WORLD MODEL as strict "
    "JSON with these keys: "
    '{"rules": [...], "entities": [...], "goal": "..."}. '
    "Cap rules at 10 short bullets, entities at 8, goal at one sentence. "
    "Rewrite from scratch when needed; do not echo the input verbatim."
)


_ACTION_RE = re.compile(r"\bACTION([1-7])\b", re.IGNORECASE)
_COORD_RE = re.compile(r"\bACTION6\b[^\d-]*?(\d+)\D+?(\d+)", re.IGNORECASE)


@dataclass
class _ReflectState:
    """Per-episode state shared by A3 / A4 (mistakes are A4-only)."""

    world_model: WorldModel = field(default_factory=WorldModel)
    mistakes_buf: MistakeBuffer = field(default_factory=MistakeBuffer)
    last_grid: Optional[np.ndarray] = None
    prev_reflect_grid: Optional[np.ndarray] = None
    last_prompt: str = ""
    last_response_raw: str = ""
    last_parse_ok: bool = False
    last_predicted_diff: None = None    # A3/A4 have no F1 path
    last_chosen_action: Optional[str] = None
    step_count: int = 0
    parse_failures: int = 0
    reflections_run: int = 0


class PlayReflectAgent:
    """A3 — Play (action only) + Reflection every K steps."""

    USE_MISTAKES = False  # toggled True by the A4 subclass

    def __init__(
        self,
        *,
        backbone: Any = None,
        model_path: Optional[str] = None,
        seed: Optional[int] = None,
        play_max_new_tokens: int = 8,
        reflect_max_new_tokens: int = 384,
        reflect_every: int = REFLECT_EVERY,
    ) -> None:
        if reflect_every < 1:
            raise ValueError(f"reflect_every must be >= 1, got {reflect_every}")
        self._backbone = backbone
        self._model_path = model_path
        self._play_max_new_tokens = play_max_new_tokens
        self._reflect_max_new_tokens = reflect_max_new_tokens
        self._reflect_every = reflect_every
        self._rng = random.Random(seed)
        self._state = _ReflectState()

    # ── public API ────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._state = _ReflectState()

    def choose(
        self, latest: FrameDataRaw, history: list[FrameDataRaw]
    ) -> GameAction:
        if latest.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET
        if not latest.frame:
            return self._fallback_random(latest)

        s_t = latest_grid(latest)

        # --- A: stash mistake record for the *previous* step's outcome ---
        if self._state.last_grid is not None:
            self._record_step_outcome(s_t, latest)

        # --- B: maybe run a Reflection pass ---
        if self._should_reflect():
            self._run_reflection(s_t, latest)

        # --- C: Play forward pass ---
        action = self._play(s_t, latest)
        return action

    # ── reflection scheduling ─────────────────────────────────────────────

    def _should_reflect(self) -> bool:
        s = self._state
        if s.step_count == 0:
            return False   # nothing to reflect on yet
        # Cadence
        if s.step_count % self._reflect_every == 0:
            return True
        # Stuck-trigger: NO_OP_TRIGGER consecutive unchanged frames
        recs = s.mistakes_buf.records
        if len(recs) >= NO_OP_TRIGGER:
            if all(not r.frame_changed for r in recs[-NO_OP_TRIGGER:]):
                return True
        # Illegal-action trigger
        if recs and not recs[-1].legal:
            return True
        return False

    # ── play call ─────────────────────────────────────────────────────────

    def _play(self, s_t: np.ndarray, latest: FrameDataRaw) -> GameAction:
        image = grid_to_image(s_t, scale=8)
        prompt = self._build_play_prompt(latest)
        self._state.last_prompt = _PLAY_SYSTEM + "\n\n" + prompt

        backbone = self._ensure_backbone()
        try:
            response_raw = backbone.generate(
                image, prompt,
                system=_PLAY_SYSTEM,
                max_new_tokens=self._play_max_new_tokens,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning("play backbone failed (%s) — fallback random", e)
            self._state.parse_failures += 1
            self._state.last_response_raw = ""
            self._state.last_parse_ok = False
            self._state.last_grid = s_t.copy()
            self._state.step_count += 1
            return self._fallback_random(latest)

        self._state.last_response_raw = response_raw
        action = self._coerce_action(response_raw, latest)
        if action is None:
            self._state.parse_failures += 1
            self._state.last_parse_ok = False
            self._state.last_chosen_action = None
            self._state.last_grid = s_t.copy()
            self._state.step_count += 1
            return self._fallback_random(latest)

        self._state.last_parse_ok = True
        self._state.last_chosen_action = action.name
        self._state.last_grid = s_t.copy()
        self._state.step_count += 1
        action.reasoning = f"a3:{action.name}"
        return action

    def _build_play_prompt(self, latest: FrameDataRaw) -> str:
        legal = ", ".join(available_action_names(latest))
        parts = [self._state.world_model.render()]
        if self.USE_MISTAKES and self._state.mistakes_buf.mistakes:
            parts.append("[RECENT MISTAKES]")
            for m in self._state.mistakes_buf.mistakes:
                parts.append(f"  - {m}")
        parts.append(f"Legal actions: {legal}")
        parts.append("Reply with one action token only.")
        return "\n".join(parts)

    # ── reflection call ───────────────────────────────────────────────────

    def _run_reflection(self, s_t: np.ndarray, latest: FrameDataRaw) -> None:
        backbone = self._ensure_backbone()
        # Two-image input — BEFORE is prev_reflect_grid (or first observed),
        # AFTER is current. Backbones with single-image protocols can collapse
        # to AFTER only; we still send the prompt with both labelled.
        after_img = grid_to_image(s_t, scale=8)
        prompt = self._build_reflect_prompt(latest)
        try:
            response_raw = backbone.generate(
                after_img, prompt,
                system=_REFLECT_SYSTEM,
                max_new_tokens=self._reflect_max_new_tokens,
                temperature=0.0,
                constrained_schema=WORLD_MODEL_SCHEMA,
            )
        except TypeError:
            # Older backbones don't accept constrained_schema; retry without.
            response_raw = backbone.generate(
                after_img, prompt,
                system=_REFLECT_SYSTEM,
                max_new_tokens=self._reflect_max_new_tokens,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning("reflection backbone failed (%s) — keeping prev WM", e)
            return

        self._state.reflections_run += 1
        parsed = self._parse_world_model(response_raw)
        if parsed is not None:
            self._state.world_model = parsed
            self._state.prev_reflect_grid = s_t.copy()

    def _build_reflect_prompt(self, latest: FrameDataRaw) -> str:
        s = self._state
        recs = s.mistakes_buf.records[-5:]
        recent = "\n".join(
            f"  step {r.step}: {r.action} "
            f"({'changed' if r.frame_changed else 'no_op'}"
            f"{', illegal' if not r.legal else ''})"
            for r in recs
        ) or "  (none)"
        parts = [
            "[CURRENT WORLD MODEL]",
            s.world_model.render(),
            "[RECENT OUTCOMES (last 5)]",
            recent,
        ]
        if self.USE_MISTAKES and s.mistakes_buf.mistakes:
            parts.append("[OPEN MISTAKES]")
            for m in s.mistakes_buf.mistakes:
                parts.append(f"  - {m}")
            parts.append(
                "Above is the auto-detected mistake list. You MAY dedup or "
                "compress these into the WORLD MODEL but do NOT invent new "
                "mistakes."
            )
        parts.append(
            "Return JSON: "
            '{"rules": [...≤10], "entities": [...≤8], "goal": "..."}'
        )
        return "\n\n".join(parts)

    @staticmethod
    def _parse_world_model(text: str) -> Optional[WorldModel]:
        if not isinstance(text, str):
            return None
        # Find first balanced {...}
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        end = -1
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end < 0:
            return None
        try:
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
        return WorldModel.from_dict(payload)

    # ── step-outcome recording (for mistakes + reflect trigger) ───────────

    def _record_step_outcome(
        self, s_t: np.ndarray, latest: FrameDataRaw
    ) -> None:
        s = self._state
        prev_grid = s.last_grid
        if prev_grid is None:
            return
        changed = not np.array_equal(prev_grid, s_t)
        rec = StepRecord(
            step=s.step_count - 1,
            action=s.last_chosen_action or "FALLBACK",
            legal=bool(s.last_parse_ok),
            frame_changed=changed,
            frame_hash=frame_hash(s_t),
            levels_completed=int(latest.levels_completed),
        )
        update_mistakes(s.mistakes_buf, rec)

    # ── action coercion (mirrors VLMAgentLite) ────────────────────────────

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
            cm = _COORD_RE.search(text)
            if not cm:
                return None
            x, y = int(cm.group(1)), int(cm.group(2))
            if not (0 <= x <= 63 and 0 <= y <= 63):
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
            f"{type(self).__name__} has no backbone — pass `backbone=` or `model_path=`"
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


class PlayReflectMistakesAgent(PlayReflectAgent):
    """A4 — A3 with auto-detected mistakes plumbed into Play + Reflect prompts.

    The only behavioural delta is `USE_MISTAKES = True`. The mistake list
    itself is already maintained by `_record_step_outcome` in the parent
    (so we can read it post-hoc for diagnostics regardless of agent mode).
    """

    USE_MISTAKES = True
