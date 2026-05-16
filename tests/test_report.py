"""Tests for arc_agent.report — pure aggregation, no I/O surprises."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from arc_agent.report import (
    aggregate,
    decision_gate,
    load_summary,
    render_per_game,
    render_table,
)


def _summary(agent: str, per_game: dict, *, wall: float = 10.0,
             mean_f1: float = 0.2, parse_rate: float = 0.5,
             mean_rhae: float = 0.0) -> dict:
    return {
        "agent": agent,
        "mean_f1": mean_f1,
        "parse_rate": parse_rate,
        "mean_rhae": mean_rhae,
        "wall_clock_seconds": wall,
        "per_game": per_game,
    }


def test_aggregate_groups_by_agent() -> None:
    summaries = [
        _summary("a1", {"g1": {"score": 0.1, "actions": 80, "levels_completed": 0,
                              "parse_rate": 0.9}}),
        _summary("a2", {"g1": {"score": 0.2, "actions": 80, "levels_completed": 1,
                              "parse_rate": 0.6}}),
    ]
    rows = aggregate(summaries)
    by = {r.agent: r for r in rows}
    assert by["a1"].mean_rhae == pytest.approx(0.1)
    assert by["a2"].mean_rhae == pytest.approx(0.2)
    assert by["a1"].mean_levels == 0
    assert by["a2"].mean_levels == 1


def test_aggregate_handles_missing_scores() -> None:
    rows = aggregate([_summary("x", {"g1": {"actions": 10, "levels_completed": 0}})])
    assert rows[0].mean_rhae == 0.0


def test_aggregate_sorts_by_rhae_desc() -> None:
    summaries = [
        _summary("low",  {"g": {"score": 0.05, "actions": 10, "levels_completed": 0}}),
        _summary("high", {"g": {"score": 0.50, "actions": 10, "levels_completed": 1}}),
        _summary("mid",  {"g": {"score": 0.20, "actions": 10, "levels_completed": 0}}),
    ]
    rows = aggregate(summaries)
    assert [r.agent for r in rows] == ["high", "mid", "low"]


def test_aggregate_wall_per_step() -> None:
    s = _summary("x",
                 {"g1": {"score": 0.1, "actions": 40, "levels_completed": 0,
                         "parse_rate": 0.5},
                  "g2": {"score": 0.2, "actions": 60, "levels_completed": 0,
                         "parse_rate": 0.5}},
                 wall=100.0)
    rows = aggregate([s])
    # 100 / (40 + 60) = 1.0 s/step
    assert rows[0].wall_per_step_sec == pytest.approx(1.0)


def test_render_table_contains_all_agents() -> None:
    rows = aggregate([
        _summary("a1", {"g": {"score": 0.1, "actions": 10, "levels_completed": 0}}),
        _summary("a2", {"g": {"score": 0.2, "actions": 10, "levels_completed": 1}}),
    ])
    out = render_table(rows)
    assert "a1" in out and "a2" in out
    assert "RHAE" in out


def test_render_per_game_handles_empty() -> None:
    assert "no per-game" in render_per_game([])


def test_render_per_game_matrix() -> None:
    rows = aggregate([
        _summary("a", {"g1": {"score": 0.1, "actions": 10, "levels_completed": 0},
                       "g2": {"score": 0.2, "actions": 10, "levels_completed": 0}}),
        _summary("b", {"g1": {"score": 0.3, "actions": 10, "levels_completed": 0}}),
    ])
    out = render_per_game(rows)
    assert "g1" in out and "g2" in out


def test_decision_gate_a1_wins() -> None:
    rows = aggregate([
        _summary("vlm_qwen25vl3b:lite_h0",
                 {"g": {"score": 0.2, "actions": 10, "levels_completed": 0}}),
        _summary("vlm_qwen25vl3b:full",
                 {"g": {"score": 0.1, "actions": 10, "levels_completed": 0}}),
    ])
    verdict = decision_gate(rows)
    assert "ship A1" in verdict


def test_decision_gate_scaffold_wins() -> None:
    rows = aggregate([
        _summary("vlm_qwen25vl3b:lite_h0",
                 {"g": {"score": 0.1, "actions": 10, "levels_completed": 0}}),
        _summary("vlm_qwen25vl3b:reflect",
                 {"g": {"score": 0.5, "actions": 10, "levels_completed": 0}}),
    ])
    verdict = decision_gate(rows)
    assert "beats A1" in verdict and "reflect" in verdict


def test_decision_gate_no_a1_row() -> None:
    rows = aggregate([
        _summary("vlm_qwen25vl3b:full",
                 {"g": {"score": 0.1, "actions": 10, "levels_completed": 0}}),
    ])
    assert "can't apply" in decision_gate(rows)


def test_load_summary_reads_json(tmp_path: Path) -> None:
    payload = _summary("a", {"g": {"score": 0.1, "actions": 1, "levels_completed": 0}})
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    assert load_summary(p) == payload
