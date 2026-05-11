"""VLMAgent — Qwen2.5-VL agent with intrinsic-F1-aware prompt + rule table.

End-to-end inference loop per `docs/ARCHITECTURE_RL.md` §4. Each `choose()`:

1.  If state ∈ {NOT_PLAYED, GAME_OVER} → return RESET.
2.  Compute reflection: f1 between *previous* step's predicted_diff and the
    actual real_diff (s_{t-1} → s_t). Update rule_table.
3.  Render current grid → PIL image.
4.  Build 6-段 prompt (system + 5 user sections — §1).
5.  Backbone generate → parse JSON: entities / reflection / predicted_diff /
    chosen_action / new_rule.
6.  On parse fail or illegal action → uniform-random fallback over legal ids.
7.  Stash (grid, predicted_diff, action) for next step's reflection.

The backbone is injected (matches `LLMAgent`/`LLMClient` pattern) so unit
tests can pass a deterministic fake and never touch torch.
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

from arc_agent.observation import (
    available_action_names,
    grid_to_image,
    latest_grid,
)
from arc_agent.rewards import (
    ChangeSet,
    changes_to_set,
    real_changes,
    verify_prediction_f1,
)

logger = logging.getLogger(__name__)


# ── Prompt section headings (kept as named constants so tests can pin order) ──

SECTION_SYSTEM   = "[SYSTEM]"
SECTION_1_META   = "【段 1: 场景元信息】"
SECTION_2_HIST   = "【段 2: 历史动作】"
SECTION_3_REFL   = "【段 3: 上一轮反思】"
SECTION_4_ENTITY = "【段 4: 实体识别请求】"
SECTION_5_FORMAT = "【段 5: 输出格式】"

PROMPT_SECTIONS_IN_ORDER: tuple[str, ...] = (
    SECTION_SYSTEM,
    SECTION_1_META,
    SECTION_2_HIST,
    SECTION_3_REFL,
    SECTION_4_ENTITY,
    SECTION_5_FORMAT,
)


_SYSTEM_PREAMBLE = """你是 ARC-AGI-3 游戏 agent。每步需要:
(1) 识别画面中的实体
(2) 反思上一步预测的对错
(3) 预测这一步选某动作后会发生什么
(4) 选一个动作
(5) 如果发现新规则,输出新规则
"""


_JSON_SPEC = """请只输出一个 JSON 对象,不要有任何前后缀:
{
  "entities":       [{"shape": ..., "color": int, "count": int,
                      "type": str, "function": str,
                      "position": [row, col]}, ...],
  "reflection":     "上轮推断(自然语言一句)",
  "predicted_diff": [{"row": int, "col": int, "to_color": int}, ...],
  "chosen_action":  "ACTIONx",
  "coords":         {"x": int 0..63, "y": int 0..63}   // only for ACTION6
  "new_rule":       null | {"trigger_action": str, "subject_color": int,
                             "effect": str, "confidence": float 0..1}
}"""


@dataclass
class _AgentState:
    """Per-episode mutable state. Cleared on `reset()`."""

    last_grid: Optional[np.ndarray] = None
    last_predicted_diff: Optional[ChangeSet] = None
    last_chosen_action: Optional[str] = None
    last_f1: Optional[float] = None
    last_real_diff: Optional[ChangeSet] = None
    last_reflection: str = ""
    action_history: list[str] = field(default_factory=list)
    rule_table: list[dict] = field(default_factory=list)
    step_count: int = 0
    parse_failures: int = 0


class VLMAgent:
    """Qwen2.5-VL agent with reflection + entity-recognition prompt."""

    HISTORY_LIMIT = 3
    DEFAULT_CONFIDENCE = 0.5
    EVICT_BELOW = 0.30   # confidence floor before drop
    REINFORCE_F1 = 0.80  # f1 >= → reinforce existing rules
    ADD_RULE_F1  = 0.50  # f1 <  → consider adding new_rule from model

    def __init__(
        self,
        *,
        backbone: Any = None,
        model_path: Optional[str] = None,
        max_rules: int = 20,
        seed: Optional[int] = None,
        max_new_tokens: int = 512,
    ) -> None:
        """Construct a VLM agent.

        Args:
            backbone: object with `.generate(image, prompt, system=...)`.
                Pass a fake in tests; in production use
                `arc_agent.vlm_backbone.HFBackbone.load(...)`.
            model_path: if given and `backbone is None`, load the HFBackbone
                lazily on first use. Mutually optional with `backbone`.
            max_rules: rule_table cap; lowest-confidence entries are evicted
                when over.
            seed: rng seed for the fallback-random branch.
            max_new_tokens: forwarded to backbone.generate when supported.
        """
        self._backbone = backbone
        self._model_path = model_path
        self._max_rules = max_rules
        self._max_new_tokens = max_new_tokens
        self._rng = random.Random(seed)
        self._state = _AgentState()

    # ── public API ───────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear ALL per-episode state including the rule table.

        Each game has different rules, so we start fresh between episodes.
        """
        self._state = _AgentState()

    def choose(
        self, latest: FrameDataRaw, history: list[FrameDataRaw]
    ) -> GameAction:
        if latest.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        if not latest.frame:
            logger.warning("FrameDataRaw.frame empty — fallback random")
            return self._fallback_random(latest)

        s_t = latest_grid(latest)

        # --- Step A: reflection on previous step ---
        if self._state.last_grid is not None:
            real_diff = real_changes(self._state.last_grid, s_t)
            f1 = verify_prediction_f1(
                self._state.last_predicted_diff or set(), real_diff
            )
            self._state.last_real_diff = real_diff
            self._state.last_f1 = f1
            # rule table updates use this turn's predicted_new_rule, which we
            # don't have yet — see Step C below.

        # --- Step B: build prompt + image, call backbone ---
        image = grid_to_image(s_t, scale=8)
        system, user = self._build_prompt(latest)

        backbone = self._ensure_backbone()
        try:
            response_raw = backbone.generate(image, user, system=system)
        except Exception as e:
            logger.warning("backbone.generate failed (%s) — fallback random", e)
            self._state.parse_failures += 1
            self._state.last_predicted_diff = None
            return self._fallback_random(latest)

        parsed = self._parse_response(response_raw)

        # --- Step C: rule-table update using parsed new_rule + last F1 ---
        if self._state.last_f1 is not None:
            self._update_rule_table(self._state.last_f1, parsed.get("new_rule"))

        # --- Step D: action validation ---
        action = self._coerce_action(parsed, latest)
        if action is None:
            self._state.parse_failures += 1
            logger.warning(
                "VLM response unparseable or illegal (failure %d). raw=%r",
                self._state.parse_failures, response_raw[:200],
            )
            self._state.last_predicted_diff = None
            self._state.last_chosen_action = None
            self._state.step_count += 1
            return self._fallback_random(latest)

        # --- Step E: stash for next reflection ---
        self._state.last_grid = s_t.copy()
        self._state.last_predicted_diff = changes_to_set(parsed.get("predicted_diff"))
        self._state.last_chosen_action = action.name
        self._state.last_reflection = str(parsed.get("reflection") or "")
        self._state.action_history.append(action.name)
        if len(self._state.action_history) > self.HISTORY_LIMIT:
            self._state.action_history = self._state.action_history[-self.HISTORY_LIMIT:]
        self._state.step_count += 1
        action.reasoning = self._state.last_reflection or "vlm"
        return action

    # ── prompt building (kept testable) ──────────────────────────────────

    def _build_prompt(self, latest: FrameDataRaw) -> tuple[str, str]:
        """Return (system, user) prompt strings. 6-段 markers must appear in
        the order defined by `PROMPT_SECTIONS_IN_ORDER`."""
        rule_json = json.dumps(self._state.rule_table, ensure_ascii=False)
        system = (
            f"{SECTION_SYSTEM}\n"
            f"{_SYSTEM_PREAMBLE}\n"
            f"已知规则 (rule_table, JSON):\n{rule_json}"
        )

        # 段 1: meta
        meta = "\n".join([
            SECTION_1_META,
            f"游戏: {latest.game_id}",
            f"关卡: {latest.levels_completed + 1} / {latest.win_levels}",
            f"状态: {latest.state.name}",
            f"可用动作: {', '.join(available_action_names(latest))}",
        ])

        # 段 2: history
        if self._state.action_history:
            hist_str = ", ".join(self._state.action_history)
        else:
            hist_str = "(start)"
        hist = f"{SECTION_2_HIST}\n最近 {self.HISTORY_LIMIT} 步: {hist_str}"

        # 段 3: reflection
        if self._state.last_f1 is None:
            refl_body = "(首步,无反思)"
        else:
            pred = sorted(self._state.last_predicted_diff or [])
            real = sorted(self._state.last_real_diff or [])
            refl_body = (
                f"上次动作: {self._state.last_chosen_action}\n"
                f"上次 predicted_diff: {pred}\n"
                f"上次 real_diff:      {real}\n"
                f"F1 = {self._state.last_f1:.2f}\n"
                f"模型上轮自述: {self._state.last_reflection or '(无)'}"
            )
        refl = f"{SECTION_3_REFL}\n{refl_body}"

        # 段 4: entity request
        entity = (
            f"{SECTION_4_ENTITY}\n"
            "请识别画面中所有实体,每个实体给出:\n"
            "- shape: 占据哪几个 cell (相对坐标或描述)\n"
            "- color: 颜色编号 0..15\n"
            "- count: 该色块出现几次\n"
            "- type: 推测属于哪类 (player / wall / goal / enemy / movable_obj / ...)\n"
            "- function: 你认为它的作用\n"
            "- position: 中心 cell 坐标 [row, col]"
        )

        # 段 5: output format
        fmt = f"{SECTION_5_FORMAT}\n{_JSON_SPEC}"

        user = "\n\n".join([meta, hist, refl, entity, fmt])
        return system, user

    # ── response parsing (tolerant) ──────────────────────────────────────

    @staticmethod
    def _parse_response(text: str) -> dict:
        """Best-effort parse of model output.

        Strips ```json fences```, takes the FIRST balanced `{...}` substring,
        and returns whatever loads. On any failure returns `{}` — never raises.
        """
        if not isinstance(text, str):
            return {}
        # Strip ``` fences if present
        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if fenced:
            candidate = fenced.group(1)
        else:
            # Find first { … balanced …}
            start = text.find("{")
            if start < 0:
                return {}
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
                return {}
            candidate = text[start:end + 1]

        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            return {}
        return obj if isinstance(obj, dict) else {}

    # ── action coercion ──────────────────────────────────────────────────

    def _coerce_action(
        self, parsed: dict, latest: FrameDataRaw
    ) -> Optional[GameAction]:
        name = parsed.get("chosen_action")
        if not isinstance(name, str):
            return None
        m = re.match(r"\s*(ACTION[1-7])\b", name.upper())
        if not m:
            return None
        try:
            action = GameAction[m.group(1)]
        except KeyError:
            return None

        if action.value not in latest.available_actions:
            return None

        if action.is_complex():
            coords = parsed.get("coords")
            x = y = None
            if isinstance(coords, dict):
                x, y = coords.get("x"), coords.get("y")
            if x is None or y is None:
                x = parsed.get("x")
                y = parsed.get("y")
            try:
                xi, yi = int(x), int(y)
            except (TypeError, ValueError):
                return None
            if not (0 <= xi <= 63 and 0 <= yi <= 63):
                return None
            action.set_data({"x": xi, "y": yi})

        return action

    # ── rule table update (§3.2) ──────────────────────────────────────────

    def _update_rule_table(self, f1: float, new_rule: Any) -> None:
        """Reinforce on high F1; add proposed rule on low F1; evict + cap."""
        table = self._state.rule_table

        if f1 >= self.REINFORCE_F1:
            for r in table:
                r["evidence_count"] = int(r.get("evidence_count", 0)) + 1
                r["confidence"] = min(
                    1.0, float(r.get("confidence", self.DEFAULT_CONFIDENCE)) + 0.05
                )

        if f1 < self.ADD_RULE_F1 and isinstance(new_rule, dict) and new_rule:
            rule = dict(new_rule)  # copy
            rule.setdefault("confidence", self.DEFAULT_CONFIDENCE)
            rule.setdefault("evidence_count", 1)
            try:
                rule["confidence"] = float(rule["confidence"])
            except (TypeError, ValueError):
                rule["confidence"] = self.DEFAULT_CONFIDENCE
            table.append(rule)

        # Evict low-confidence rules first
        self._state.rule_table = [
            r for r in table
            if float(r.get("confidence", self.DEFAULT_CONFIDENCE)) >= self.EVICT_BELOW
        ]

        # Cap by descending confidence
        if len(self._state.rule_table) > self._max_rules:
            self._state.rule_table.sort(
                key=lambda r: float(r.get("confidence", self.DEFAULT_CONFIDENCE)),
                reverse=True,
            )
            self._state.rule_table = self._state.rule_table[: self._max_rules]

    # ── backbone helpers ─────────────────────────────────────────────────

    def _ensure_backbone(self) -> Any:
        if self._backbone is not None:
            return self._backbone
        if self._model_path is not None:
            from arc_agent.vlm_backbone import HFBackbone
            self._backbone = HFBackbone.load(model_path=self._model_path)
            return self._backbone
        raise RuntimeError(
            "VLMAgent has no backbone — pass `backbone=` or `model_path=`"
        )

    # ── fallback (mirrors LLMAgent) ──────────────────────────────────────

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
