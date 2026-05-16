"""TextAgent (v3) — the main ARC-AGI-3 agent built on scipy perception
+ text-only Qwen reasoning.

End-to-end per `docs/ARCHITECTURE_v3_zh.md`:

  perception:  scipy extract -> temporal_classifier -> Hungarian align
  memory:      ObjectMemory (UID history) + OutcomeLog (action stats)
  reasoner:    Qwen2.5-VL-3B in text-only mode (no image content block)
               4-bit, max_new_tokens=24, greedy
  postprocess: anti-collapse (force diversification when last 3 same)

The agent satisfies `arc_agent.runner.Agent` and exposes the standard
`_state.last_prompt` / `last_response_raw` / `last_parse_ok` surface so
the baseline runner trace.jsonl captures everything.
"""
from __future__ import annotations

import hashlib
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from arc_agent.action_inference import (
    OutcomeLog,
    StepOutcome,
    detect_collapse,
    detect_stuck,
)
from arc_agent.click_candidates import (
    list_click_candidates,
    pick_default_action6_coords,
)
from arc_agent.object_aligner import align_objects
from arc_agent.object_extractor import extract_objects
from arc_agent.object_tracker import ObjectMemory
from arc_agent.observation import (
    available_action_names,
    latest_grid,
)
from arc_agent.prompts_v3 import PLAY_SYSTEM, build_play_user_prompt
from arc_agent.temporal_classifier import (
    Layer,
    classify_frame,
    filter_active,
    update_history,
)

logger = logging.getLogger(__name__)


_ACTION_RE = re.compile(r"\bACTION([1-7])\b", re.IGNORECASE)
_COORD_RE = re.compile(r"\bACTION6\b[^\d-]*?(\d+)\D+?(\d+)", re.IGNORECASE)


@dataclass
class _TextAgentState:
    """Per-episode mutable state."""
    object_memory: ObjectMemory = field(default_factory=ObjectMemory)
    outcome_log: OutcomeLog = field(default_factory=OutcomeLog)
    history_per_sig: dict = field(default_factory=dict)
    prev_active_objects: list = field(default_factory=list)
    prev_grid: Optional[np.ndarray] = None
    prev_action_name: Optional[str] = None
    prev_legal_set: set = field(default_factory=set)
    last_prompt: str = ""
    last_response_raw: str = ""
    last_parse_ok: bool = False
    last_predicted_diff: None = None     # v3 has no F1 path
    last_chosen_action: Optional[str] = None
    step_count: int = 0
    parse_failures: int = 0
    goal_hypothesis: str = ""
    goal_confidence: str = "low"
    frame_hashes: list = field(default_factory=list)  # P0-B: state revisit detection
    tried_action6_coords: list = field(default_factory=list)  # P1: rotate candidates


class TextAgent:
    """The v3 agent. Text-only Qwen + scipy perception."""

    DEFAULT_MAX_NEW_TOKENS = 24
    COLLAPSE_WINDOW = 3
    MAX_ACTIONS_DEFAULT = 80

    def __init__(
        self,
        *,
        backbone: Any = None,
        model_path: Optional[str] = None,
        seed: Optional[int] = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_actions: int = MAX_ACTIONS_DEFAULT,
    ) -> None:
        self._backbone = backbone
        self._model_path = model_path
        self._max_new_tokens = max_new_tokens
        self._max_actions = max_actions
        self._rng = random.Random(seed)
        self._state = _TextAgentState()

    # ── public API ────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._state = _TextAgentState()

    def choose(self, latest: FrameDataRaw,
               history: list[FrameDataRaw]) -> GameAction:
        if latest.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET
        if not latest.frame:
            return self._fallback_random(latest)

        grid = latest_grid(latest)
        legal_names = available_action_names(latest)
        legal_set = set(latest.available_actions)

        # ── 1) RECORD outcome of the PREVIOUS step (if any) ──────────────
        if self._state.prev_grid is not None and self._state.prev_action_name:
            self._record_previous_outcome(grid, latest, legal_set)

        # ── 2) PERCEPTION on current frame ───────────────────────────────
        current_objs = extract_objects(grid)
        update_history(self._state.history_per_sig, current_objs)
        layer_by_id = classify_frame(current_objs, self._state.history_per_sig)
        current_active = filter_active(current_objs, layer_by_id)

        # ── 3) ALIGN previous ACTIVE -> current ACTIVE ───────────────────
        matches = align_objects(self._state.prev_active_objects, current_active)
        self._state.object_memory.update(
            step=self._state.step_count,
            current_active=current_active,
            matches=matches,
        )

        # Append current frame hash for state-revisit detection (P0-B)
        self._state.frame_hashes.append(hash(grid.tobytes()))

        # ── 4) BUILD prompt ──────────────────────────────────────────────
        diversification = None
        if detect_collapse(self._state.outcome_log, self.COLLAPSE_WINDOW):
            last = self._state.outcome_log.all_steps[-1].action
            diversification = (
                f"You have picked {last} {self.COLLAPSE_WINDOW} times in a row "
                f"without progress. STOP repeating it. Pick a different action."
            )

        # P0-B: multi-condition stuck detection
        is_stuck, stuck_reason = detect_stuck(
            self._state.outcome_log,
            self._state.frame_hashes,
        )

        # P1: pre-compute click candidates if ACTION6 is legal
        click_cands = None
        if "ACTION6" in legal_names:
            click_cands = list_click_candidates(current_objs, layer_by_id)

        user_prompt = build_play_user_prompt(
            step=self._state.step_count,
            max_steps=self._max_actions,
            level=latest.levels_completed + 1,
            total_levels=latest.win_levels,
            state=latest.state.name,
            legal_actions=legal_names,
            frame_objects=current_objs,
            layer_by_id=layer_by_id,
            object_memory=self._state.object_memory,
            outcome_log=self._state.outcome_log,
            goal_hypothesis=self._state.goal_hypothesis,
            goal_confidence=self._state.goal_confidence,
            diversification_hint=diversification,
            stuck_reason=stuck_reason if is_stuck else None,
            click_candidates=click_cands,
        )
        self._state.last_prompt = PLAY_SYSTEM + "\n\n" + user_prompt

        # ── 5) GENERATE ──────────────────────────────────────────────────
        backbone = self._ensure_backbone()
        try:
            response_raw = backbone.generate(
                None, user_prompt,
                system=PLAY_SYSTEM,
                max_new_tokens=self._max_new_tokens,
                temperature=0.0,
            )
        except TypeError:
            # Older fake backbones don't allow `image=None`. Pass a tiny stub.
            response_raw = backbone.generate(
                _PlaceholderImage(), user_prompt,
                system=PLAY_SYSTEM,
                max_new_tokens=self._max_new_tokens,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning("text backbone failed (%s) — fallback random", e)
            self._state.parse_failures += 1
            self._state.last_response_raw = ""
            self._state.last_parse_ok = False
            return self._stash_and_return_fallback(grid, current_active, latest)

        self._state.last_response_raw = response_raw
        action = self._coerce_action(response_raw, latest,
                                     frame_objects=current_objs,
                                     layer_by_id=layer_by_id)

        # ── 6) Anti-collapse postprocess ─────────────────────────────────
        if action is not None and diversification is not None:
            # If we're under a diversification alert, REJECT any choice that
            # matches the recent collapsed action.
            last_choice = self._state.outcome_log.all_steps[-1].action
            if action.name == last_choice:
                untried = self._state.outcome_log.untried(legal_names)
                if untried:
                    name = untried[0]
                    try:
                        action = GameAction[name]
                        if action.is_complex():
                            action.set_data({
                                "x": self._rng.randint(0, 63),
                                "y": self._rng.randint(0, 63),
                            })
                    except KeyError:
                        action = None

        if action is None:
            self._state.parse_failures += 1
            self._state.last_parse_ok = False
            return self._stash_and_return_fallback(grid, current_active, latest)

        self._state.last_parse_ok = True
        self._state.last_chosen_action = action.name
        # Stash for next step's outcome computation
        self._state.prev_grid = grid.copy()
        self._state.prev_action_name = action.name
        self._state.prev_legal_set = legal_set
        self._state.prev_active_objects = current_active
        self._state.step_count += 1
        action.reasoning = "v3"
        return action

    # ── helpers ───────────────────────────────────────────────────────────

    def _record_previous_outcome(self, current_grid: np.ndarray,
                                 latest: FrameDataRaw,
                                 legal_set_now: set) -> None:
        s = self._state
        changed = not np.array_equal(s.prev_grid, current_grid)
        # Primary direction: dominant move in ACTIVE objects
        primary_dir = None
        primary_dist = 0
        if changed and s.prev_active_objects:
            # Use the FIRST moved active object's delta as the primary signal
            from arc_agent.object_aligner import align_objects as _aln
            current_objs = extract_objects(current_grid)
            # We can re-derive matches now; cheap.
            new_active = current_objs   # crude; ignores re-classification
            try:
                tmp_matches = _aln(s.prev_active_objects, new_active)
                for m in tmp_matches:
                    if m.type == "moved" and m.delta:
                        dy = m.delta.get("dy", 0)
                        dx = m.delta.get("dx", 0)
                        primary_dist = max(abs(dy), abs(dx))
                        parts = []
                        if dy < 0: parts.append("UP")
                        elif dy > 0: parts.append("DOWN")
                        if dx < 0: parts.append("LEFT")
                        elif dx > 0: parts.append("RIGHT")
                        primary_dir = "+".join(parts) if parts else None
                        break
            except Exception:
                pass

        legal_at_time = s.prev_legal_set
        last_action_name = s.prev_action_name
        try:
            action_value = GameAction[last_action_name].value
            legal = action_value in legal_at_time
        except KeyError:
            legal = False

        s.outcome_log.record(StepOutcome(
            step=s.step_count - 1 if s.step_count > 0 else 0,
            action=last_action_name or "FALLBACK",
            legal=legal,
            frame_changed=changed,
            n_active_changed=sum(1 for _ in s.prev_active_objects) if changed else 0,
            primary_direction=primary_dir,
            primary_distance=primary_dist,
        ))

    def _coerce_action(self, text: Any,
                       latest: FrameDataRaw,
                       *,
                       frame_objects=None,
                       layer_by_id=None) -> Optional[GameAction]:
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
            if cm:
                x = int(cm.group(1)); y = int(cm.group(2))
                if 0 <= x <= 63 and 0 <= y <= 63:
                    action.set_data({"x": x, "y": y})
                    self._state.tried_action6_coords.append((x, y))
                    return action
            # P1: ACTION6 without parseable coords -> click candidate, not random
            chosen = None
            if frame_objects is not None and layer_by_id is not None:
                chosen = pick_default_action6_coords(
                    frame_objects, layer_by_id,
                    tried_coords=self._state.tried_action6_coords,
                )
            if chosen is None:
                chosen = (self._rng.randint(0, 63), self._rng.randint(0, 63))
            action.set_data({"x": chosen[0], "y": chosen[1]})
            self._state.tried_action6_coords.append(chosen)
        return action

    def _ensure_backbone(self) -> Any:
        if self._backbone is not None:
            return self._backbone
        if self._model_path is not None:
            from arc_agent.vlm_backbone import HFBackbone
            self._backbone = HFBackbone.load(model_path=self._model_path)
            return self._backbone
        raise RuntimeError(
            "TextAgent has no backbone — pass `backbone=` or `model_path=`")

    def _stash_and_return_fallback(self, grid, current_active,
                                   latest: FrameDataRaw) -> GameAction:
        action = self._fallback_random(latest)
        self._state.last_chosen_action = None
        self._state.prev_grid = grid.copy()
        self._state.prev_action_name = action.name
        self._state.prev_legal_set = set(latest.available_actions)
        self._state.prev_active_objects = current_active
        self._state.step_count += 1
        return action

    def _fallback_random(self, latest: FrameDataRaw) -> GameAction:
        legal = [v for v in latest.available_actions
                 if v != GameAction.RESET.value]
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


class _PlaceholderImage:
    """Stub PIL-like object for fake backbones that require positional image."""
    size = (1, 1)
    mode = "RGB"
    def convert(self, _mode): return self
