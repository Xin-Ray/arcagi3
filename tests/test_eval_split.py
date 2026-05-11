"""Tests for arc_agent.eval_split (game splits + run summary writer)."""
from __future__ import annotations

import json

import pytest

from arc_agent.eval_split import demo_555_split, write_summary


# ---- demo_555_split ---------------------------------------------------------


def test_split_demo_25_alphabetical_slices() -> None:
    games = [f"g{i:02d}-abc" for i in range(25)]
    # already sorted, but pass shuffled to prove sort happens
    shuffled = games[::-1]
    split = demo_555_split(shuffled)
    assert split["g_base"]  == games[0:5]
    assert split["g_train"] == games[5:10]
    assert split["g_val"]   == games[10:15]
    assert split["holdout"] == games[15:25]
    assert split["n_input"] == 25


def test_split_dedupes_input() -> None:
    games = [f"g{i:02d}" for i in range(20)] + ["g00", "g05"]  # 2 dups
    split = demo_555_split(games)
    assert split["n_input"] == 20
    assert len(split["g_base"]) == 5
    assert len(split["g_train"]) == 5
    assert len(split["g_val"]) == 5
    assert len(split["holdout"]) == 5  # 20 - 15


def test_split_minimum_15_games_works() -> None:
    games = [f"x{i:02d}" for i in range(15)]
    split = demo_555_split(games)
    assert split["holdout"] == []  # nothing left over
    assert split["g_val"] == games[10:15]


def test_split_raises_on_too_few_games() -> None:
    with pytest.raises(ValueError, match=r"need >= 15"):
        demo_555_split([f"x{i}" for i in range(14)])


def test_split_raises_on_non_list() -> None:
    with pytest.raises(TypeError, match="must be a list"):
        demo_555_split("not a list")  # type: ignore[arg-type]


def test_split_disjoint_groups() -> None:
    games = [f"g{i:03d}" for i in range(25)]
    split = demo_555_split(games)
    base = set(split["g_base"])
    train = set(split["g_train"])
    val = set(split["g_val"])
    holdout = set(split["holdout"])
    # Pairwise disjoint
    assert base & train == set()
    assert base & val == set()
    assert base & holdout == set()
    assert train & val == set()
    assert train & holdout == set()
    assert val & holdout == set()
    # Union covers all 25
    assert base | train | val | holdout == set(games)


# ---- write_summary ----------------------------------------------------------


def _minimal_summary_kwargs() -> dict:
    return dict(
        run_kind="baseline",
        games=["g0", "g1"],
        n_episodes_per_game=1,
        wall_clock_seconds=12.5,
        mean_f1=0.42,
        parse_rate=0.91,
        mean_rhae=0.03,
        per_game={"g0": {"mean_f1": 0.5}, "g1": {"mean_f1": 0.34}},
    )


def test_write_summary_creates_file_with_required_fields(tmp_path) -> None:
    out = tmp_path / "subdir" / "summary.json"  # parent doesn't exist yet
    payload = write_summary(out, **_minimal_summary_kwargs())
    assert out.exists()

    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk == payload
    # auto-filled
    assert on_disk["schema_version"] == "1"
    assert "run_ts" in on_disk and on_disk["run_ts"].endswith("Z")
    # required preserved
    assert on_disk["run_kind"] == "baseline"
    assert on_disk["games"] == ["g0", "g1"]
    assert on_disk["mean_f1"] == 0.42


def test_write_summary_passes_through_optional_fields(tmp_path) -> None:
    out = tmp_path / "summary.json"
    payload = write_summary(
        out,
        **_minimal_summary_kwargs(),
        split="demo_555.json",
        git_commit="abc1234",
        notes="first real run",
    )
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk["split"] == "demo_555.json"
    assert on_disk["git_commit"] == "abc1234"
    assert on_disk["notes"] == "first real run"


def test_write_summary_raises_when_required_field_missing(tmp_path) -> None:
    kwargs = _minimal_summary_kwargs()
    kwargs.pop("mean_f1")
    with pytest.raises(ValueError, match="missing required fields"):
        write_summary(tmp_path / "summary.json", **kwargs)
