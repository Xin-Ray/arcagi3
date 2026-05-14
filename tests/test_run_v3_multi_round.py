"""Integration smoke test for scripts/run_v3_multi_round.py.

We do NOT touch the SDK -- the script ships an internal _StubArcade /
_StubEnv path activated by --dry-run. This test invokes the module's
functions directly so it runs in pytest in <2s, with no network or GPU.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# `run_v3_multi_round` imports dotenv etc. -- we don't care, but importing
# is enough to expose the helpers.
run_module = importlib.import_module("run_v3_multi_round")


# ── pure helpers ──────────────────────────────────────────────────────────


def test_compute_primary_change_no_change() -> None:
    g = np.zeros((4, 4), dtype=int)
    changed, dir_, dist, deltas = run_module._compute_primary_change(g, g.copy())
    assert changed is False
    assert dir_ is None
    assert dist == 0
    assert deltas == []


def test_compute_primary_change_movement() -> None:
    """A 1x1 red blob shifts right by 1 -- aligner reports RIGHT delta."""
    before = np.zeros((6, 6), dtype=int)
    after = np.zeros((6, 6), dtype=int)
    before[2, 1] = 2
    after[2, 2] = 2
    changed, dir_, dist, deltas = run_module._compute_primary_change(before, after)
    assert changed is True
    assert dir_ == "RIGHT"
    assert dist == 1
    assert deltas  # at least one line


def test_compute_primary_change_missing_grid_returns_no_change() -> None:
    changed, *_ = run_module._compute_primary_change(None, None)
    assert changed is False


# ── DryRun adapters ──────────────────────────────────────────────────────


def test_dry_run_action_agent_returns_legal() -> None:
    agent = run_module._DryRunActionAgent(seed=0)

    class _F:
        from arcengine import GameState as _GS
        state = _GS.NOT_FINISHED
        available_actions = [1, 2, 3]
        frame = [np.zeros((4, 4), dtype=int)]
    a, r = agent.choose(_F(), history=[])
    assert a.value in [1, 2, 3]
    assert "dry-run" in r


def test_dry_run_reflection_returns_empty_delta_then_periodic_update() -> None:
    agent = run_module._DryRunReflectionAgent()
    from arc_agent.knowledge import Knowledge
    from arc_agent.step_summary import StepSummary
    k = Knowledge.empty("ar25")

    # step 0 is a multiple of 5 -> action_semantics update fires
    summ = StepSummary(step=0, action="ACTION1", reasoning="r",
                       frame_changed=True, primary_direction="UP")
    delta, raw = agent.reflect_after_step(knowledge=k, step_summary=summ)
    assert "action_semantics_update" in delta
    assert delta["action_semantics_update"]   # non-empty on step 0
    # step 1 -> no update
    summ2 = StepSummary(step=1, action="ACTION2", reasoning="r",
                        frame_changed=True, primary_direction="UP")
    delta2, _ = agent.reflect_after_step(knowledge=k, step_summary=summ2)
    assert delta2["action_semantics_update"] == {}


# ── End-to-end dry-run integration ────────────────────────────────────────


def test_end_to_end_dry_run(tmp_path) -> None:
    """Run 2 rounds × 3 steps with stub agents + stub Arcade; verify
    artifact tree and Knowledge persistence."""
    action_agent = run_module._DryRunActionAgent(seed=0)
    reflection_agent = run_module._DryRunReflectionAgent()

    summary = run_module._dry_run_loop(
        game_id_full="ar25",
        n_rounds=2,
        max_actions=3,
        out_dir=tmp_path,
        action_agent=action_agent,
        reflection_agent=reflection_agent,
        fps=2,
        save_images=True,
    )

    # Summary shape
    assert summary["game_id"] == "ar25"
    assert summary["rounds_played"] == 2
    assert len(summary["per_round"]) == 2

    # Artifact tree per round
    for r in (0, 1):
        rd = tmp_path / f"round_{r:02d}"
        assert (rd / "trace.jsonl").exists()
        assert (rd / "knowledge_per_step.jsonl").exists()
        assert (rd / "play.gif").exists()
        pngs = sorted(rd.glob("step_*.png"))
        assert len(pngs) == 3

    # Cross-round Knowledge history file
    knowledge_history = tmp_path / "knowledge_history.jsonl"
    assert knowledge_history.exists()
    lines = knowledge_history.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # one row per round
    # Round 1's knowledge should have rounds_played=2 (cumulative)
    k1 = json.loads(lines[1])["knowledge_at_round_end"]
    assert k1["rounds_played"] == 2


def test_report_md_includes_per_round_table(tmp_path) -> None:
    action_agent = run_module._DryRunActionAgent(seed=0)
    reflection_agent = run_module._DryRunReflectionAgent()
    summary = run_module._dry_run_loop(
        game_id_full="ar25", n_rounds=1, max_actions=2,
        out_dir=tmp_path, action_agent=action_agent,
        reflection_agent=reflection_agent, fps=2, save_images=False,
    )
    run_module.write_report(tmp_path, summary)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "v3.2 multi-round run" in report
    assert "Per-round metrics" in report
    assert "Final knowledge" in report


# ── R2 hard rule: action mask integration ────────────────────────────────


def test_B_orch_override_never_fires_now(tmp_path) -> None:
    """B (2026-05-14): R2 mask replacement was removed. Orchestrator no
    longer overrides the LLM choice; orch_override stays empty in every
    row of trace.jsonl regardless of Knowledge flags or OutcomeLog state."""
    import json
    action_agent = run_module._DryRunActionAgent(seed=0)
    reflection_agent = run_module._DryRunReflectionAgent()
    run_module._dry_run_loop(
        game_id_full="ar25", n_rounds=1, max_actions=5,
        out_dir=tmp_path,
        action_agent=action_agent,
        reflection_agent=reflection_agent,
        fps=2, save_images=False, seed=0,
    )
    trace = [json.loads(l) for l in
             (tmp_path / "round_00" / "trace.jsonl").read_text().splitlines()]
    overrides = [r.get("orch_override", "") for r in trace]
    assert all(not o for o in overrides), (
        f"B: orchestrator should never override; got: {overrides}"
    )


def test_R2_mask_does_not_fire_without_flag(tmp_path) -> None:
    """Baseline: dry-run agents + empty Knowledge -> no override."""
    import json
    action_agent = run_module._DryRunActionAgent(seed=0)
    reflection_agent = run_module._DryRunReflectionAgent()
    run_module._dry_run_loop(
        game_id_full="ar25", n_rounds=1, max_actions=3,
        out_dir=tmp_path,
        action_agent=action_agent,
        reflection_agent=reflection_agent,
        fps=2, save_images=False, seed=0,
    )
    trace = [json.loads(l) for l in
             (tmp_path / "round_00" / "trace.jsonl").read_text().splitlines()]
    overrides = [r.get("orch_override", "") for r in trace]
    # No Knowledge flags + < 5 attempts per action -> no mask -> all empty
    assert all(not o for o in overrides), (
        f"R2 mask should NOT fire without flag or threshold but did: {overrides}"
    )


# ── C: orchestrator stuck alert ──────────────────────────────────────────


def test_C_build_stuck_alert_returns_empty_when_below_threshold() -> None:
    out = run_module._build_stuck_alert(no_op_streak=0, state_revisit=1,
                                        last_picks=["ACTION1"])
    assert out == ""


def test_C_build_stuck_alert_fires_at_threshold() -> None:
    out = run_module._build_stuck_alert(
        no_op_streak=5, state_revisit=6,
        last_picks=["ACTION1"] * 5,
    )
    assert out != ""
    # Should name the culprit
    assert "ACTION1" in out
    # Should suggest others
    assert "ACTION2" in out or "ACTION3" in out


def test_C_build_stuck_alert_fires_on_state_revisit_alone() -> None:
    """Either condition (no_op_streak OR state_revisit) is enough to fire."""
    out = run_module._build_stuck_alert(
        no_op_streak=0, state_revisit=10,
        last_picks=["ACTION6"] * 3,
    )
    assert out != ""
    assert "ACTION6" in out


# ── D: natural termination on WIN/GAME_OVER ────────────────────────────


def test_D_loop_breaks_on_game_over_state(tmp_path) -> None:
    """A stub env that transitions to GAME_OVER after 3 steps should
    cause the round to end at step 3, not run to max_actions."""
    import numpy as np
    from arcengine import GameState

    class _GameOverEnv:
        def __init__(self):
            self._step = 0
        def reset(self):
            class F:
                state = GameState.NOT_FINISHED
                frame = [np.zeros((8, 8), dtype=int)]
                available_actions = [1, 2, 3, 4, 5, 7]
                levels_completed = 0
                win_levels = 3
                game_id = "stub"
                guid = "g"
            return F()
        def step(self, action, data=None, reasoning=None):
            self._step += 1
            class F:
                state = (GameState.GAME_OVER if self._step >= 3
                         else GameState.NOT_FINISHED)
                frame = [np.zeros((8, 8), dtype=int)]
                available_actions = [1, 2, 3, 4, 5, 7]
                levels_completed = 0
                win_levels = 3
                game_id = "stub"
                guid = "g"
            return F()

    class _Arc:
        def make(self, gid, scorecard_id=None): return _GameOverEnv()
        def open_scorecard(self, tags=None): return "c"
        def close_scorecard(self, card_id): return {}

    summary = run_module.run_one_game(
        arc=_Arc(), card_id="c", game_id_full="stub",
        n_rounds=1, max_actions=100,
        out_dir=tmp_path,
        action_agent=run_module._DryRunActionAgent(seed=0),
        reflection_agent=run_module._DryRunReflectionAgent(),
        fps=2, save_images=False, seed=0,
    )
    # Round should stop at step 3 (GAME_OVER), not at 100
    assert summary["per_round"][0]["n_steps"] <= 3


def test_no_images_flag_skips_pngs(tmp_path) -> None:
    action_agent = run_module._DryRunActionAgent(seed=0)
    reflection_agent = run_module._DryRunReflectionAgent()
    run_module._dry_run_loop(
        game_id_full="ar25", n_rounds=1, max_actions=2,
        out_dir=tmp_path, action_agent=action_agent,
        reflection_agent=reflection_agent, fps=2, save_images=False,
    )
    rd = tmp_path / "round_00"
    assert (rd / "trace.jsonl").exists()
    assert not list(rd.glob("step_*.png"))
    assert not (rd / "play.gif").exists()
