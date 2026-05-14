"""v3.2 multi-round orchestrator -- per docs/arch_v3_2_zh.md S7.

Runs N rounds of one game with the v3.2 ActionAgent + ReflectionAgent +
persistent Knowledge. Each round resets ObjectMemory / OutcomeLog but
keeps Knowledge so the agents learn across rounds (and across steps).

Outputs:
    outputs/v3_2_<ts>/
        knowledge_history.jsonl                 one row per round end
        report.md                               summary
        round_<k>/
            trace.jsonl                         one row per step
            knowledge_per_step.jsonl            one row per step (snapshot)
            step_0000.png ... step_NNNN.png     viz_v3_2 PNG per step
            play.gif                            episode playback
            reflection_raw.txt                  raw reflection outputs (debug)

CLI:
    .venv/Scripts/python.exe scripts/run_v3_multi_round.py \
        --game ar25 --rounds 3 --max-actions 20 --dry-run

    # Real run (needs ARC_API_KEY + Qwen weights + transformers stack)
    .venv/Scripts/python.exe scripts/run_v3_multi_round.py \
        --game ar25 --rounds 5 --max-actions 80
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

import numpy as np  # noqa: E402

from arcengine import GameAction, GameState  # noqa: E402

from arc_agent.agents.action_agent import ActionAgent  # noqa: E402
from arc_agent.agents.reflection_agent import ReflectionAgent  # noqa: E402
from arc_agent.knowledge import Knowledge  # noqa: E402
from arc_agent.object_aligner import align_objects  # noqa: E402
from arc_agent.object_extractor import extract_objects  # noqa: E402
from arc_agent.observation import latest_grid  # noqa: E402
from arc_agent.step_summary import (  # noqa: E402
    StepSummary,
    compute_matches_reasoning,
    object_delta_lines,
)
from arc_agent.viz_v3_2 import compose_step_image_v32, write_gif_v3_2  # noqa: E402

OUTPUTS_ROOT = REPO_ROOT / "outputs"


# ─── helpers ────────────────────────────────────────────────────────────


def _check_key() -> None:
    key = os.getenv("ARC_API_KEY", "")
    if not key or key.startswith("your_"):
        raise RuntimeError(
            "ARC_API_KEY missing or still the .env.example placeholder. "
            "Put a real key in .env (https://arcprize.org/api-keys)."
        )


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ─── DryRunAdapter: returns deterministic-random actions + reasoning ────


class _DryRunActionAgent:
    """Stand-in for ActionAgent when --dry-run is set. Does not load Qwen.

    Returns one of the legal actions at random, with a stock reasoning
    string. Exposes the same surface (attach_knowledge / reset / choose /
    no_op_streak / state_revisit_count / recent_step_records) the
    orchestrator needs.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._knowledge = Knowledge.empty()
        self._step = 0
        self._frame_hashes: list[int] = []
        self._recent: list[tuple[str, bool, Optional[str]]] = []
        self._state = type("S", (), {"last_prompt": "", "last_response_raw": "",
                                     "last_parse_ok": True, "last_reasoning": "",
                                     "last_chosen_action": None,
                                     "step_count": 0, "parse_failures": 0})()

    def attach_knowledge(self, knowledge: Knowledge) -> None:
        self._knowledge = knowledge

    def reset_episode_state(self, *, knowledge: Optional[Knowledge] = None) -> None:
        self._step = 0
        self._frame_hashes = []
        self._recent = []
        if knowledge is not None:
            self._knowledge = knowledge

    def choose(self, latest, history=None):
        if latest.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET, "(dry-run) reset"
        legal = [v for v in latest.available_actions
                 if v != GameAction.RESET.value]
        if not legal:
            return GameAction.RESET, "(dry-run) no legal actions"
        action = GameAction.from_id(self._rng.choice(legal))
        if action.is_complex():
            action.set_data({
                "x": self._rng.randint(0, 63),
                "y": self._rng.randint(0, 63),
            })
        reasoning = f"(dry-run) random pick: {action.name}"
        try:
            grid = latest_grid(latest)
            self._frame_hashes.append(hash(grid.tobytes()))
        except Exception:
            pass
        self._step += 1
        self._state.step_count = self._step
        self._state.last_reasoning = reasoning
        self._state.last_chosen_action = action.name
        return action, reasoning

    def no_op_streak(self) -> int:
        s = 0
        for _, ch, _ in reversed(self._recent):
            if not ch:
                s += 1
            else:
                break
        return s

    def state_revisit_count(self, grid_or_hash) -> int:
        if isinstance(grid_or_hash, int):
            target = grid_or_hash
        else:
            try:
                target = hash(grid_or_hash.tobytes())
            except AttributeError:
                return 1
        return self._frame_hashes.count(target)

    def recent_step_records(self, n: int = 3):
        return self._recent[-n:]

    def record_outcome(self, action_name: str, changed: bool,
                       direction: Optional[str]) -> None:
        """The orchestrator calls this after env.step so we can return the
        right no_op_streak / recent_step_records on subsequent steps."""
        self._recent.append((action_name, changed, direction))


class _DryRunReflectionAgent:
    """Stand-in for ReflectionAgent when --dry-run is set. Never loads Qwen.
    Returns a tiny canned delta every step so the orchestrator's merge path
    is exercised end-to-end without needing a model."""

    def __init__(self) -> None:
        self._state = type("S", (), {"last_prompt": "", "last_response_raw": "{}",
                                     "last_parse_ok": True,
                                     "last_delta": {}, "call_count": 0,
                                     "parse_failures": 0})()

    def reset(self) -> None:
        self._state.call_count = 0

    def reflect_after_step(self, *, knowledge, step_summary):
        delta = {
            "action_semantics_update": {},
            "current_alert": "",
        }
        # Occasionally update an action_semantics entry so the GIF/PNG show
        # something happening; deterministic on step number.
        if step_summary.step % 5 == 0 and step_summary.frame_changed:
            delta["action_semantics_update"] = {
                step_summary.action: f"observed {step_summary.primary_direction or 'change'} "
                                     f"after step {step_summary.step}"
            }
        self._state.call_count += 1
        self._state.last_delta = delta
        return delta, json.dumps(delta)


# ─── agent factory ──────────────────────────────────────────────────────


def _make_agents(
    *, dry_run: bool, seed: int, max_new_tokens_action: int,
    max_new_tokens_reflection: int,
):
    if dry_run:
        return _DryRunActionAgent(seed=seed), _DryRunReflectionAgent()

    from arc_agent.vlm_backbone import HFBackbone
    backbone = HFBackbone.load()
    action_agent = ActionAgent(
        backbone=backbone, seed=seed,
        max_new_tokens=max_new_tokens_action,
    )
    reflection_agent = ReflectionAgent(
        backbone=backbone,
        max_new_tokens=max_new_tokens_reflection,
        temperature=0.0,
    )
    return action_agent, reflection_agent


# ─── per-step bookkeeping helpers ───────────────────────────────────────


def _compute_primary_change(
    prev_grid: Optional[np.ndarray],
    curr_grid: Optional[np.ndarray],
) -> tuple[bool, Optional[str], int, list[str]]:
    """Compute (frame_changed, primary_direction, primary_distance, object_delta_lines)
    for one (s_t, s_{t+1}) pair."""
    if prev_grid is None or curr_grid is None:
        return False, None, 0, []
    if np.array_equal(prev_grid, curr_grid):
        return False, None, 0, []
    primary_dir: Optional[str] = None
    primary_dist = 0
    deltas: list[str] = []
    try:
        prev_objs = extract_objects(prev_grid)
        curr_objs = extract_objects(curr_grid)
        matches = align_objects(prev_objs, curr_objs)
        deltas = object_delta_lines(matches, max_lines=5)
        for m in matches:
            if m.type == "moved" and m.delta:
                dy = int(m.delta.get("dy", 0))
                dx = int(m.delta.get("dx", 0))
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
    return True, primary_dir, primary_dist, deltas


def _safe_grid(frame) -> Optional[np.ndarray]:
    try:
        return latest_grid(frame)
    except Exception:
        return None


# ─── main loop ──────────────────────────────────────────────────────────


def run_one_game(
    *,
    arc,
    card_id,
    game_id_full: str,
    n_rounds: int,
    max_actions: int,
    out_dir: Path,
    action_agent,
    reflection_agent,
    fps: int = 2,
    save_images: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    knowledge_history_path = out_dir / "knowledge_history.jsonl"

    knowledge = Knowledge.empty(game_id=game_id_full)

    per_round_metrics: list[dict[str, Any]] = []

    for r in range(n_rounds):
        env = arc.make(game_id_full, scorecard_id=card_id)
        action_agent.reset_episode_state(knowledge=knowledge)
        if hasattr(reflection_agent, "reset"):
            reflection_agent.reset()

        round_dir = out_dir / f"round_{r:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        trace_path = round_dir / "trace.jsonl"
        knowledge_step_path = round_dir / "knowledge_per_step.jsonl"
        reflection_raw_path = round_dir / "reflection_raw.txt"

        frames_for_gif: list[Any] = []
        levels_completed_start: Optional[int] = None
        latest = env.reset()
        prev_grid: Optional[np.ndarray] = _safe_grid(latest)
        levels_completed_start = latest.levels_completed

        n_changed = 0
        n_no_op = 0
        n_parse_failures = 0

        for step in range(max_actions):
            if latest.state == GameState.WIN:
                break

            alert_active = knowledge.current_alert

            # 1) Action Agent choice
            action_agent.attach_knowledge(knowledge)
            try:
                action, reasoning = action_agent.choose(latest, history=[])
            except Exception as e:
                print(f"[round {r}] action_agent failed at step {step}: {e}",
                      file=sys.stderr)
                break

            # 2) env.step
            grid_before = prev_grid
            try:
                latest = env.step(
                    action,
                    data=action.action_data.model_dump(),
                    reasoning=getattr(action, "reasoning", None),
                )
            except Exception as e:
                print(f"[round {r}] env.step failed at step {step}: {e}",
                      file=sys.stderr)
                break
            grid_after = _safe_grid(latest)

            # 3) outcome computation
            changed, primary_dir, primary_dist, obj_deltas = _compute_primary_change(
                grid_before, grid_after,
            )
            if changed:
                n_changed += 1
            else:
                n_no_op += 1

            # tell the dry-run agent (real ActionAgent records this internally)
            if hasattr(action_agent, "record_outcome"):
                action_agent.record_outcome(action.name, changed, primary_dir)

            matches_reasoning = compute_matches_reasoning(
                reasoning,
                frame_changed=changed,
                primary_direction=primary_dir,
            )

            no_op_streak = action_agent.no_op_streak() if hasattr(action_agent, "no_op_streak") else 0
            try:
                state_revisit = action_agent.state_revisit_count(grid_after) if grid_after is not None else 1
            except Exception:
                state_revisit = 1
            recent = action_agent.recent_step_records(n=3) if hasattr(action_agent, "recent_step_records") else []

            coords = None
            if action.is_complex():
                d = action.action_data.model_dump()
                coords = (int(d.get("x", 0)), int(d.get("y", 0)))

            summary = StepSummary(
                step=step,
                action=action.name,
                action_coords=coords,
                reasoning=reasoning or "",
                frame_changed=changed,
                primary_direction=primary_dir,
                primary_distance=primary_dist,
                object_deltas=obj_deltas,
                no_op_streak=no_op_streak,
                state_revisit_count=state_revisit,
                matches_reasoning=matches_reasoning,
                recent_steps=recent,
            )

            # 4) Reflection Agent
            try:
                delta, refl_raw = reflection_agent.reflect_after_step(
                    knowledge=knowledge, step_summary=summary,
                )
            except Exception as e:
                print(f"[round {r}] reflection failed at step {step}: {e}",
                      file=sys.stderr)
                delta, refl_raw = {}, ""

            knowledge = knowledge.merged_with_delta(delta)

            # 5) viz
            if save_images and grid_after is not None:
                try:
                    png = compose_step_image_v32(
                        grid_after,
                        action=action.name + (f" ({coords[0]},{coords[1]})" if coords else ""),
                        reasoning=reasoning or "",
                        reflection_delta=delta,
                        alert=alert_active,
                        header=f"{game_id_full} round={r} step={step} "
                               f"lvl={latest.levels_completed}",
                        matches_reasoning=matches_reasoning,
                    )
                    png.save(round_dir / f"step_{step:04d}.png")
                    frames_for_gif.append(png)
                except Exception as e:
                    print(f"[round {r}] viz failed at step {step}: {e}",
                          file=sys.stderr)

            # 6) trace + knowledge snapshot
            _append_jsonl(trace_path, {
                "step": step,
                "action": action.name,
                "action_coords": coords,
                "reasoning": reasoning,
                "frame_changed": changed,
                "primary_direction": primary_dir,
                "primary_distance": primary_dist,
                "matches_reasoning": matches_reasoning,
                "current_alert_active": alert_active,
                "reflection_delta": delta,
                "no_op_streak": no_op_streak,
                "state_revisit_count": state_revisit,
            })
            _append_jsonl(knowledge_step_path, {
                "step": step,
                "knowledge_after": knowledge.to_dict(),
            })
            try:
                with reflection_raw_path.open("a", encoding="utf-8") as f:
                    f.write(f"--- step {step} ---\n")
                    f.write(refl_raw or "")
                    f.write("\n")
            except OSError:
                pass

            prev_grid = grid_after

        # end of round
        if save_images and frames_for_gif:
            try:
                write_gif_v3_2(frames_for_gif, round_dir / "play.gif", fps=fps)
            except Exception as e:
                print(f"[round {r}] gif failed: {e}", file=sys.stderr)

        levels_completed_end = latest.levels_completed if latest else 0
        levels_gained = max(0, levels_completed_end - (levels_completed_start or 0))
        round_won = latest.state == GameState.WIN if latest else False

        knowledge.rounds_played += 1
        if round_won:
            knowledge.rounds_won += 1
        n_steps = n_changed + n_no_op
        change_rate = (n_changed / n_steps) if n_steps else 0.0
        knowledge.append_round_summary(
            f"round {r}: {n_steps} steps, {n_changed} changed "
            f"({change_rate:.0%}), levels+{levels_gained}, "
            f"{'WIN' if round_won else 'incomplete'}"
        )

        per_round_metrics.append({
            "round": r,
            "n_steps": n_steps,
            "n_changed": n_changed,
            "n_no_op": n_no_op,
            "change_rate": change_rate,
            "levels_gained": levels_gained,
            "won": round_won,
            "knowledge_action_semantics_size": len(knowledge.action_semantics),
            "knowledge_rules_size": len(knowledge.rules),
            "knowledge_failed_strategies_size": len(knowledge.failed_strategies),
        })
        _append_jsonl(knowledge_history_path, {
            "round": r,
            "knowledge_at_round_end": knowledge.to_dict(),
            "metrics": per_round_metrics[-1],
        })

    return {
        "game_id": game_id_full,
        "n_rounds": n_rounds,
        "rounds_played": knowledge.rounds_played,
        "rounds_won": knowledge.rounds_won,
        "final_knowledge": knowledge.to_dict(),
        "per_round": per_round_metrics,
    }


# ─── report ─────────────────────────────────────────────────────────────


def write_report(out_dir: Path, run_summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append(f"# v3.2 multi-round run: {run_summary['game_id']}")
    lines.append("")
    lines.append(f"rounds: {run_summary['rounds_played']} played, "
                 f"{run_summary['rounds_won']} won")
    lines.append("")
    lines.append("## Per-round metrics")
    lines.append("")
    lines.append("| round | steps | changed | no-op | change_rate | levels+ | won | |semantics| | |rules| | |failed| |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for m in run_summary["per_round"]:
        lines.append(
            f"| {m['round']} | {m['n_steps']} | {m['n_changed']} | "
            f"{m['n_no_op']} | {m['change_rate']:.0%} | "
            f"{m['levels_gained']} | {'YES' if m['won'] else ''} | "
            f"{m['knowledge_action_semantics_size']} | "
            f"{m['knowledge_rules_size']} | "
            f"{m['knowledge_failed_strategies_size']} |"
        )
    lines.append("")
    lines.append("## Final knowledge")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(run_summary["final_knowledge"], indent=2,
                            ensure_ascii=False))
    lines.append("```")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ─── CLI ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=False, default="ar25",
                        help="Game id prefix (default: ar25). Resolved against "
                             "SDK get_environments() to the full id.")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-actions", type=int, default=80)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="",
                        help="Run dir (default: outputs/v3_2_<ts>)")
    parser.add_argument("--tag", default="v3_2")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use random/stub agents -- no Qwen load, no ARC key needed for plumbing.")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip PNG + GIF (trace.jsonl still written).")
    parser.add_argument("--max-new-tokens-action", type=int, default=96)
    parser.add_argument("--max-new-tokens-reflection", type=int, default=250)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output) if args.output else (
        OUTPUTS_ROOT / f"{args.tag}_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "started_at": ts,
        "git_commit": _git_commit(),
        "args": vars(args),
        "doc": "docs/arch_v3_2_zh.md",
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    action_agent, reflection_agent = _make_agents(
        dry_run=args.dry_run, seed=args.seed,
        max_new_tokens_action=args.max_new_tokens_action,
        max_new_tokens_reflection=args.max_new_tokens_reflection,
    )

    if args.dry_run:
        # Bypass SDK entirely -- use a stub arc / env so plumbing is testable
        # without an ARC_API_KEY. Real runs go through Arcade below.
        print("[dry-run] using local stub Arcade -- no network calls")
        from arc_agent.runner import play_one  # noqa: F401  (sanity import)
        run_summary = _dry_run_loop(
            game_id_full=args.game,
            n_rounds=args.rounds, max_actions=args.max_actions,
            out_dir=out_dir, action_agent=action_agent,
            reflection_agent=reflection_agent, fps=args.fps,
            save_images=not args.no_images,
        )
    else:
        _check_key()
        from arc_agi import Arcade  # noqa: E402
        arc = Arcade()
        env_infos = arc.get_environments() or []
        candidates = [e.game_id for e in env_infos
                      if e.game_id.startswith(args.game)]
        if not candidates:
            raise RuntimeError(f"no game starting with {args.game!r} in SDK")
        game_id_full = candidates[0]
        card_id = arc.open_scorecard(tags=[args.tag, "v3_2"])
        try:
            run_summary = run_one_game(
                arc=arc, card_id=card_id, game_id_full=game_id_full,
                n_rounds=args.rounds, max_actions=args.max_actions,
                out_dir=out_dir, action_agent=action_agent,
                reflection_agent=reflection_agent, fps=args.fps,
                save_images=not args.no_images,
            )
        finally:
            try:
                arc.close_scorecard(card_id)
            except Exception:
                pass

    (out_dir / "summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(out_dir, run_summary)
    print(f"[done] {out_dir}")


# ─── dry-run loop (no SDK) ──────────────────────────────────────────────


def _dry_run_loop(
    *, game_id_full: str, n_rounds: int, max_actions: int,
    out_dir: Path, action_agent, reflection_agent,
    fps: int, save_images: bool,
) -> dict[str, Any]:
    """Plumbing test: simulate `arc.make/env.reset/env.step` with a tiny
    deterministic stub. Exercises every code path in run_one_game without
    touching the network."""

    class _StubFrame:
        def __init__(self, state, grid, lvl=0):
            self.state = state
            self.frame = [grid]
            self.available_actions = [1, 2, 3, 4, 5, 7]
            self.levels_completed = lvl
            self.win_levels = 3
            self.game_id = game_id_full
            self.guid = "stub"

    class _StubEnv:
        def __init__(self):
            self._step_n = 0

        def reset(self):
            return _StubFrame(GameState.NOT_FINISHED, np.zeros((8, 8), dtype=int))

        def step(self, action, data=None, reasoning=None):
            self._step_n += 1
            g = np.zeros((8, 8), dtype=int)
            g[self._step_n % 8, self._step_n % 8] = (self._step_n % 15) + 1
            state = GameState.NOT_FINISHED
            return _StubFrame(state, g)

    class _StubArcade:
        def make(self, gid, scorecard_id=None):
            return _StubEnv()
        def open_scorecard(self, tags=None):
            return "stub-card"
        def close_scorecard(self, card_id):
            return {}

    arc = _StubArcade()
    card_id = arc.open_scorecard(tags=["dry"])
    return run_one_game(
        arc=arc, card_id=card_id, game_id_full=game_id_full,
        n_rounds=n_rounds, max_actions=max_actions,
        out_dir=out_dir, action_agent=action_agent,
        reflection_agent=reflection_agent, fps=fps,
        save_images=save_images,
    )


if __name__ == "__main__":
    main()
