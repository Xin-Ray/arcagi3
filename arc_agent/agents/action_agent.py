"""ActionAgent (v3.2) -- the main per-step decision agent.

Derived from v3 `TextAgent` (`docs/arch_v3_zh.md`), with these differences
per `docs/arch_v3_2_zh.md` §5.1:

  - Reads a shared `Knowledge` object (attach_knowledge before each choose)
    that persists across rounds within one game.
  - Prompt is built by `prompts_v3_2.build_action_user_prompt` which
    prepends [REFLECTION ALERT] + [KNOWLEDGE] above v3 blocks and replaces
    v3's [ASK] with a two-line `reasoning + action` ASK.
  - choose() returns (GameAction, reasoning_text). Reasoning is captured
    into `_state.last_reasoning` so the orchestrator can package a
    StepSummary for the Reflection Agent.

The perception / memory / anti-collapse machinery is INHERITED from v3
verbatim -- this class only changes the prompt and output format.
"""
from __future__ import annotations

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
from arc_agent.knowledge import Knowledge
from arc_agent.object_aligner import align_objects
from arc_agent.object_extractor import extract_objects
from arc_agent.object_tracker import ObjectMemory
from arc_agent.observation import available_action_names, latest_grid
from arc_agent.prompts_v3_2 import ACTION_SYSTEM, build_action_user_prompt
from arc_agent.temporal_classifier import (
    classify_frame,
    filter_active,
    update_history,
)

logger = logging.getLogger(__name__)


_ACTION_RE = re.compile(r"\bACTION([1-7])\b", re.IGNORECASE)
_COORD_RE = re.compile(r"\bACTION6\b[^\d-]*?(\d+)\D+?(\d+)", re.IGNORECASE)
_REASONING_RE = re.compile(r"reasoning\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_ACTION_LINE_RE = re.compile(r"action\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)


def parse_reasoning_and_action(text: str) -> tuple[str, str]:
    """Pull 'reasoning:' and 'action:' lines out of the response.

    Tolerant of order, case, and missing lines. When `reasoning:` is
    missing we return empty string; when `action:` is missing we fall
    back to the raw text so `_coerce_action`'s regex can still try.
    """
    if not isinstance(text, str):
        return "", ""
    r_match = _REASONING_RE.search(text)
    a_match = _ACTION_LINE_RE.search(text)
    reasoning = r_match.group(1).strip() if r_match else ""
    action_text = a_match.group(1).strip() if a_match else text.strip()
    return reasoning, action_text


@dataclass
class _ActionAgentState:
    """Per-episode mutable state. Cleared on reset_episode_state."""
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
    last_reasoning: str = ""
    last_chosen_action: Optional[str] = None
    step_count: int = 0
    parse_failures: int = 0
    frame_hashes: list = field(default_factory=list)
    tried_action6_coords: list = field(default_factory=list)


class ActionAgent:
    """v3.2 Action Agent. Same perception + memory as v3, plus Knowledge."""

    DEFAULT_MAX_NEW_TOKENS = 96   # bigger than v3 (24) -- reasoning is verbose
    COLLAPSE_WINDOW = 3
    MAX_ACTIONS_DEFAULT = 80

    # R3 hard rule: when no-op streak or state-revisit count crosses these
    # thresholds, the orchestrator forces an untried legal action regardless
    # of what the LLM picked. See docs/ref_v3_2_dataflow_zh.md.
    STUCK_NO_OP_THRESHOLD = 5
    STUCK_STATE_REVISIT_THRESHOLD = 5

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
        self._state = _ActionAgentState()
        self._knowledge: Knowledge = Knowledge.empty()

    # ── public API ────────────────────────────────────────────────────────

    def attach_knowledge(self, knowledge: Knowledge) -> None:
        """Plug in the shared Knowledge object. Called once per step (or
        once per round if the orchestrator prefers) before `choose()`."""
        self._knowledge = knowledge

    def reset_episode_state(self, *, knowledge: Optional[Knowledge] = None) -> None:
        """Wipe per-episode state (ObjectMemory, OutcomeLog, ...). Knowledge
        is NOT cleared -- it persists across rounds. If `knowledge=` is
        supplied it replaces the attached one (orchestrator uses this to
        re-attach after a round boundary)."""
        self._state = _ActionAgentState()
        if knowledge is not None:
            self._knowledge = knowledge

    def choose(self, latest: FrameDataRaw,
               history: Optional[list[FrameDataRaw]] = None
               ) -> tuple[GameAction, str]:
        """Returns (action, reasoning). reasoning is "" on parse failure
        / fallback (orchestrator should still feed "" to step_summary)."""
        if latest.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET, ""
        if not latest.frame:
            return self._fallback_random(latest), ""

        grid = latest_grid(latest)
        legal_names = available_action_names(latest)
        legal_set = set(latest.available_actions)

        # 1) record outcome of previous step
        if self._state.prev_grid is not None and self._state.prev_action_name:
            self._record_previous_outcome(grid, legal_set)

        # 2) perception
        current_objs = extract_objects(grid)
        update_history(self._state.history_per_sig, current_objs)
        layer_by_id = classify_frame(current_objs, self._state.history_per_sig)
        current_active = filter_active(current_objs, layer_by_id)

        # 3) align + memory update
        matches = align_objects(self._state.prev_active_objects, current_active)
        self._state.object_memory.update(
            step=self._state.step_count,
            current_active=current_active,
            matches=matches,
        )
        self._state.frame_hashes.append(hash(grid.tobytes()))

        # 4) build v3.2 user prompt (knowledge + alert prepended)
        diversification = None
        if detect_collapse(self._state.outcome_log, self.COLLAPSE_WINDOW):
            last = self._state.outcome_log.all_steps[-1].action
            diversification = (
                f"You have picked {last} {self.COLLAPSE_WINDOW} times in a row "
                f"without progress. STOP repeating it. Pick a different action."
            )

        is_stuck, stuck_reason = detect_stuck(
            self._state.outcome_log, self._state.frame_hashes,
        )

        click_cands = None
        if "ACTION6" in legal_names:
            click_cands = list_click_candidates(current_objs, layer_by_id)

        user_prompt = build_action_user_prompt(
            knowledge=self._knowledge,
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
            goal_hypothesis=self._knowledge.goal_hypothesis,
            goal_confidence=self._knowledge.goal_confidence,
            diversification_hint=diversification,
            stuck_reason=stuck_reason if is_stuck else None,
            click_candidates=click_cands,
        )
        self._state.last_prompt = ACTION_SYSTEM + "\n\n" + user_prompt

        # 5) generate
        backbone = self._ensure_backbone()
        try:
            response_raw = backbone.generate(
                None, user_prompt,
                system=ACTION_SYSTEM,
                max_new_tokens=self._max_new_tokens,
                temperature=0.0,
            )
        except TypeError:
            response_raw = backbone.generate(
                _PlaceholderImage(), user_prompt,
                system=ACTION_SYSTEM,
                max_new_tokens=self._max_new_tokens,
                temperature=0.0,
            )
        except Exception as e:
            logger.warning("action backbone failed (%s) -- fallback random", e)
            self._state.parse_failures += 1
            self._state.last_response_raw = ""
            self._state.last_parse_ok = False
            self._state.last_reasoning = ""
            return self._stash_and_return_fallback(grid, current_active, latest), ""

        self._state.last_response_raw = response_raw if isinstance(response_raw, str) else ""

        reasoning, action_text = parse_reasoning_and_action(self._state.last_response_raw)
        self._state.last_reasoning = reasoning

        action = self._coerce_action(action_text, latest,
                                     frame_objects=current_objs,
                                     layer_by_id=layer_by_id)

        # 6) anti-collapse postprocess (reject repeat when in diversification)
        if action is not None and diversification is not None:
            last_choice = self._state.outcome_log.all_steps[-1].action
            if action.name == last_choice:
                untried = self._state.outcome_log.untried(legal_names)
                if untried:
                    try:
                        action = GameAction[untried[0]]
                        if action.is_complex():
                            action.set_data({
                                "x": self._rng.randint(0, 63),
                                "y": self._rng.randint(0, 63),
                            })
                    except KeyError:
                        action = None

        # 6b) R3 hard rule: forced exploration on persistent stuck states.
        # If the last N steps were all no-op OR the agent is revisiting the
        # same state too often, override the LLM's choice with an untried
        # legal action (orchestrator-level guard against Qwen-3B ignoring
        # its own "this is ineffective" reasoning). See R3 in
        # docs/ref_v3_2_dataflow_zh.md.
        if action is not None:
            current_streak = self.no_op_streak()
            try:
                current_revisit = self._state.frame_hashes.count(
                    hash(grid.tobytes())
                )
            except Exception:
                current_revisit = 1
            stuck = (current_streak >= self.STUCK_NO_OP_THRESHOLD
                     or current_revisit >= self.STUCK_STATE_REVISIT_THRESHOLD)
            if stuck:
                untried = self._state.outcome_log.untried(legal_names)
                if untried and action.name not in untried:
                    try:
                        forced = GameAction[untried[0]]
                        if forced.is_complex():
                            forced.set_data({
                                "x": self._rng.randint(0, 63),
                                "y": self._rng.randint(0, 63),
                            })
                        # Record what got overridden so traces can audit it
                        self._state.last_response_raw = (
                            f"{self._state.last_response_raw}\n"
                            f"[R3 forced_explore: streak={current_streak} "
                            f"revisit={current_revisit} "
                            f"chosen={action.name} -> forced={forced.name}]"
                        )
                        action = forced
                    except KeyError:
                        pass

        if action is None:
            self._state.parse_failures += 1
            self._state.last_parse_ok = False
            return self._stash_and_return_fallback(grid, current_active, latest), reasoning

        self._state.last_parse_ok = True
        self._state.last_chosen_action = action.name
        self._state.prev_grid = grid.copy()
        self._state.prev_action_name = action.name
        self._state.prev_legal_set = legal_set
        self._state.prev_active_objects = current_active
        self._state.step_count += 1
        action.reasoning = reasoning or "v3.2"
        return action, reasoning

    # ── orchestrator helpers (exposed for step_summary build) ────────────

    def get_outcome_log(self) -> OutcomeLog:
        """Expose OutcomeLog for orchestrator-level R2 action masking."""
        return self._state.outcome_log

    def no_op_streak(self) -> int:
        streak = 0
        for o in reversed(self._state.outcome_log.all_steps):
            if not o.frame_changed:
                streak += 1
            else:
                break
        return streak

    def state_revisit_count(self, grid_or_hash: Any) -> int:
        """How many times this state hash has appeared this episode."""
        if isinstance(grid_or_hash, int):
            target = grid_or_hash
        else:
            try:
                target = hash(grid_or_hash.tobytes())
            except AttributeError:
                return 1
        return self._state.frame_hashes.count(target)

    def recent_step_records(self, n: int = 3) -> list[tuple[str, bool, Optional[str]]]:
        tail = self._state.outcome_log.all_steps[-n:]
        return [(o.action, o.frame_changed, o.primary_direction) for o in tail]

    # ── internals (parallel to TextAgent helpers) ────────────────────────

    def _record_previous_outcome(self, current_grid: np.ndarray,
                                 legal_set_now: set) -> None:
        s = self._state
        changed = not np.array_equal(s.prev_grid, current_grid)
        primary_dir, primary_dist = None, 0
        if changed and s.prev_active_objects:
            try:
                new_active = extract_objects(current_grid)
                tmp_matches = align_objects(s.prev_active_objects, new_active)
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

        try:
            action_value = GameAction[s.prev_action_name].value
            legal = action_value in s.prev_legal_set
        except KeyError:
            legal = False

        s.outcome_log.record(StepOutcome(
            step=s.step_count - 1 if s.step_count > 0 else 0,
            action=s.prev_action_name or "FALLBACK",
            legal=legal,
            frame_changed=changed,
            n_active_changed=len(s.prev_active_objects) if changed else 0,
            primary_direction=primary_dir,
            primary_distance=primary_dist,
        ))

    def _coerce_action(self, text: Any, latest: FrameDataRaw,
                       *, frame_objects=None, layer_by_id=None
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
            if cm:
                x = int(cm.group(1)); y = int(cm.group(2))
                if 0 <= x <= 63 and 0 <= y <= 63:
                    action.set_data({"x": x, "y": y})
                    self._state.tried_action6_coords.append((x, y))
                    return action
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
            "ActionAgent has no backbone -- pass `backbone=` or `model_path=`")

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
    """Stub for fake backbones that require a positional image."""
    size = (1, 1)
    mode = "RGB"

    def convert(self, _mode):
        return self


__all__ = ["ActionAgent", "parse_reasoning_and_action"]
