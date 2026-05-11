"""Tests for arc_agent.viz (compose_step_image + write_gif)."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from arc_agent.viz import compose_step_image, write_gif


# ---- compose_step_image ------------------------------------------------------


def _two_grids(h: int = 8, w: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Two same-shape grids that differ in one cell (for diff/F1 tests)."""
    g0 = np.zeros((h, w), dtype=int)
    g0[h // 2, w // 2] = 1
    g1 = g0.copy()
    g1[h // 2, w // 2] = 0
    g1[h // 2, (w // 2) - 1] = 1  # in-bounds for any w >= 2
    return g0, g1


def test_compose_step_image_returns_pil_image_with_expected_size() -> None:
    g0, g1 = _two_grids(h=8, w=8)
    img = compose_step_image(
        g0,
        predicted_diff={(2, 4, 1)},
        grid_next=g1,
        json_text='{"chosen_action":"ACTION4"}',
        header="Game: ls20 | Step 5 | F1: 0.62",
        cell_scale=4,
    )
    # 2 columns × (8 cols × 4 px) = 64 wide; header 40 + 2 rows × 32 = 104 tall
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (8 * 4 * 2, 40 + 8 * 4 * 2)


def test_compose_step_image_handles_none_predicted_diff() -> None:
    g0, g1 = _two_grids()
    img = compose_step_image(
        g0, predicted_diff=None, grid_next=g1,
        json_text="parse failed", header="x", cell_scale=2,
    )
    assert isinstance(img, Image.Image)
    # No crash, output produced


def test_compose_step_image_handles_empty_predicted_diff() -> None:
    g0, g1 = _two_grids()
    img = compose_step_image(
        g0, predicted_diff=set(), grid_next=g1,
        json_text="no changes", header="x", cell_scale=2,
    )
    assert isinstance(img, Image.Image)


def test_compose_step_image_rejects_shape_mismatch() -> None:
    g0 = np.zeros((4, 4), dtype=int)
    g1 = np.zeros((4, 5), dtype=int)
    with pytest.raises(ValueError, match="!="):
        compose_step_image(g0, None, g1, "", header="x")


def test_compose_step_image_rejects_zero_scale() -> None:
    g0, g1 = _two_grids()
    with pytest.raises(ValueError, match="cell_scale"):
        compose_step_image(g0, None, g1, "", header="x", cell_scale=0)


def test_compose_step_image_overlay_clips_out_of_range_color() -> None:
    """Overlay color 99 must not crash; clipped to palette[15]."""
    g0, g1 = _two_grids()
    img = compose_step_image(
        g0, predicted_diff={(0, 0, 99)}, grid_next=g1,
        json_text="", header="x", cell_scale=2,
    )
    assert isinstance(img, Image.Image)


def test_compose_step_image_long_json_text_doesnt_crash() -> None:
    g0, g1 = _two_grids(h=4, w=4)
    long_text = "\n".join("a" * 200 for _ in range(50))
    img = compose_step_image(
        g0, None, g1, long_text, header="x", cell_scale=2,
    )
    assert isinstance(img, Image.Image)


# ---- write_gif ---------------------------------------------------------------


def test_write_gif_writes_file_with_nonzero_size(tmp_path) -> None:
    frames = [Image.new("RGB", (32, 32), color=(i * 80, 0, 0)) for i in range(3)]
    out = write_gif(frames, tmp_path / "play.gif", fps=2)
    assert out.exists()
    assert out.stat().st_size > 0


def test_write_gif_creates_parent_dir(tmp_path) -> None:
    frames = [Image.new("RGB", (16, 16))]
    target = tmp_path / "missing" / "subdir" / "x.gif"
    out = write_gif(frames, target, fps=1)
    assert out.exists()


def test_write_gif_returns_path(tmp_path) -> None:
    frames = [Image.new("RGB", (16, 16))]
    out = write_gif(frames, tmp_path / "x.gif", fps=1)
    assert out == tmp_path / "x.gif"


def test_write_gif_raises_on_empty_frames(tmp_path) -> None:
    with pytest.raises(ValueError, match="no frames"):
        write_gif([], tmp_path / "x.gif")


def test_write_gif_raises_on_zero_fps(tmp_path) -> None:
    with pytest.raises(ValueError, match="fps"):
        write_gif([Image.new("RGB", (8, 8))], tmp_path / "x.gif", fps=0)


def test_write_gif_accepts_generator(tmp_path) -> None:
    frames = (Image.new("RGB", (16, 16), color=(i * 50, 0, 0)) for i in range(4))
    out = write_gif(frames, tmp_path / "x.gif", fps=2)
    assert out.exists()
