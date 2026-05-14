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
