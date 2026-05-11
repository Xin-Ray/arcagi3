"""Per-game baseline runner with per-step trace + visualization.

Sits between `runner.play_one` (too generic — no trace capture) and the
orchestration in `scripts/run_baseline.py` (no reusable logic by policy).

Single entry point: `play_one_with_trace(env, agent, *, run_dir, game_id,
max_actions=80, fps=2)`. The function:

1. Loops `agent.choose() → env.step()` up to `max_actions` or WIN.
2. After each step, computes real_diff from `(s_t, s_{t+1})` and reads
   `predicted_diff` from `agent._state.last_predicted_diff` (when the agent
   surfaces one; not every Agent does — see `_extract_trace`).
3. Saves a 4-quadrant `step_<n>.png` (via `viz.compose_step_image`) and
   appends one trace.jsonl row (via `observation.serialize_step`).
4. End-of-episode: writes `play.gif` from the saved frames.
5. Returns a metrics dict suitable for aggregation in summary.json.

The agent doesn't need to be a `VLMAgent` — any Agent works. But trace
fields like `prompt`, `response_raw`, `predicted_diff` are pulled from a
`_state` attribute when present and recorded as `None` otherwise. That
keeps random / LLM agents usable for plumbing tests.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
from arcengine import FrameDataRaw, GameAction, GameState

from arc_agent.observation import latest_grid, serialize_step
from arc_agent.rewards import ChangeSet, real_changes, verify_prediction_f1
from arc_agent.viz import compose_step_image, write_gif

logger = logging.getLogger(__name__)


@dataclass
class GameMetrics:
    """Per-game summary returned by `play_one_with_trace`."""

    game_id: str
    actions: int
    final_state: str
    levels_completed: int
    total_levels: int
    n_parseable: int
    parse_rate: float          # n_parseable / actions
    mean_f1: float             # parse-fail steps counted as 0
    mean_f1_when_parsed: float # over parseable steps only (NaN-ish → 0 if none)
    trace_path: str
    gif_path: Optional[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "actions": self.actions,
            "final_state": self.final_state,
            "levels_completed": self.levels_completed,
            "total_levels": self.total_levels,
            "n_parseable": self.n_parseable,
            "parse_rate": round(self.parse_rate, 4),
            "mean_f1": round(self.mean_f1, 4),
            "mean_f1_when_parsed": round(self.mean_f1_when_parsed, 4),
            "trace_path": self.trace_path,
            "gif_path": self.gif_path,
        }


class _Env(Protocol):
    """Subset of `EnvironmentWrapper` we depend on (kept narrow for mocking)."""

    def reset(self) -> FrameDataRaw: ...
    def step(self, action: GameAction, *, data: Any = None, reasoning: Any = None) -> FrameDataRaw: ...


def _extract_trace(agent: Any) -> tuple[str, str, Optional[ChangeSet], bool]:
    """Pull (prompt, response_raw, predicted_diff, parse_ok) from agent state.

    Returns blanks/None when the agent doesn't expose them — keeps the runner
    usable with non-VLM agents for plumbing tests.
    """
    state = getattr(agent, "_state", None)
    prompt = getattr(state, "last_prompt", "") or ""
    response_raw = getattr(state, "last_response_raw", "") or ""
    predicted_diff = getattr(state, "last_predicted_diff", None)
    parse_ok = bool(getattr(state, "last_parse_ok", False))
    return prompt, response_raw, predicted_diff, parse_ok


def play_one_with_trace(
    env: _Env,
    agent: Any,
    *,
    run_dir: Path | str,
    game_id: str,
    max_actions: int = 80,
    history_limit: int = 8,
    fps: int = 2,
    write_images: bool = True,
) -> GameMetrics:
    """Play one episode, capturing per-step trace + image + final GIF.

    Args:
        env: live or mocked env exposing `.reset()` and `.step(action, data=..., reasoning=...)`.
        agent: any object with `.choose(latest, history) -> GameAction`. If it
            also has `_state` with `last_prompt` / `last_response_raw` /
            `last_predicted_diff` / `last_parse_ok` (i.e. it's a VLMAgent),
            those land in the trace; otherwise they appear as empty/None.
        run_dir: where step_<n>.png, trace.jsonl, play.gif land. Created if missing.
        game_id: identifier carried into every trace row.
        max_actions: per-episode safety cap (matches `runner.play_one`).
        history_limit: how many tail-frames to pass to `agent.choose`.
        fps: GIF frame rate.
        write_images: set False to skip PNG + GIF (faster for plumbing tests).

    Returns:
        `GameMetrics` with aggregate per-game numbers + paths.
    """
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace_path = out / "trace.jsonl"

    latest = env.reset()
    history: list[FrameDataRaw] = [latest]

    f1_values: list[float] = []
    parseable_f1_values: list[float] = []
    n_parseable = 0
    frames_for_gif: list[Any] = []
    actions = 0

    with trace_path.open("w", encoding="utf-8") as fp:
        while latest.state is not GameState.WIN and actions < max_actions:
            if not latest.frame:
                logger.warning("FrameDataRaw.frame empty at step %d — stopping", actions)
                break
            s_t = latest_grid(latest)

            action = agent.choose(latest, history[-history_limit:])
            prompt, response_raw, predicted_diff, parse_ok = _extract_trace(agent)

            next_frame = env.step(
                action,
                data=action.action_data.model_dump(),
                reasoning=getattr(action, "reasoning", None),
            )

            s_tp1 = latest_grid(next_frame) if next_frame.frame else s_t
            r_diff = real_changes(s_t, s_tp1)

            if parse_ok and predicted_diff is not None:
                f1 = verify_prediction_f1(predicted_diff, r_diff)
                parseable_f1_values.append(f1)
                n_parseable += 1
                f1_for_agg = f1
            else:
                f1 = None
                f1_for_agg = 0.0
            f1_values.append(f1_for_agg)

            image_path: Optional[str] = None
            if write_images:
                fname = f"step_{actions:04d}.png"
                image_path = fname
                composite = compose_step_image(
                    s_t,
                    predicted_diff,
                    s_tp1,
                    json_text=response_raw or "(no response)",
                    header=f"{game_id}  step={actions}  action={action.name}  "
                           f"f1={'NaN' if f1 is None else f'{f1:.2f}'}  "
                           f"parse_ok={parse_ok}",
                )
                composite.save(out / fname)
                frames_for_gif.append(composite)

            row = serialize_step(
                step=actions,
                game_id=game_id,
                level=latest.levels_completed + 1,
                state=latest.state.name,
                image_path=image_path,
                prompt=prompt,
                response_raw=response_raw,
                parse_ok=parse_ok,
                predicted_diff=predicted_diff if parse_ok else None,
                chosen_action=action.name,
                real_diff=r_diff,
                f1=f1,
            )
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()

            history.append(next_frame)
            latest = next_frame
            actions += 1

    gif_path: Optional[str] = None
    if write_images and frames_for_gif:
        gif_full = write_gif(frames_for_gif, out / "play.gif", fps=fps)
        gif_path = gif_full.name

    parse_rate = n_parseable / actions if actions else 0.0
    mean_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
    mean_f1_when_parsed = (
        sum(parseable_f1_values) / len(parseable_f1_values)
        if parseable_f1_values else 0.0
    )

    return GameMetrics(
        game_id=game_id,
        actions=actions,
        final_state=latest.state.name,
        levels_completed=latest.levels_completed,
        total_levels=latest.win_levels,
        n_parseable=n_parseable,
        parse_rate=parse_rate,
        mean_f1=mean_f1,
        mean_f1_when_parsed=mean_f1_when_parsed,
        trace_path=str(trace_path),
        gif_path=gif_path,
    )
