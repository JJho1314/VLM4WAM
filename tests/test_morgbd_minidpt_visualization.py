from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "qwen3_vl_semantic_planner"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from visualize_morgbd_minidpt_probe import (  # noqa: E402
    EXPECTED_PANEL_NAMES,
    _reference_suptitle,
    _sample_metrics,
    normalize_log_depth_pair,
    save_reference_style_sample,
    unsquish_and_split,
)


def test_unsquish_then_split_preserves_camera_order() -> None:
    composite = torch.zeros(1, 1, 224, 224)
    composite[..., :112] = 1
    cameras = unsquish_and_split(composite)
    assert tuple(cameras["main"].shape) == (1, 1, 224, 224)
    assert tuple(cameras["wrist"].shape) == (1, 1, 224, 224)
    assert float(cameras["main"].mean()) > 0.99
    assert float(cameras["wrist"].mean()) < 0.01


def test_depth_teacher_and_planner_share_one_disparity_range() -> None:
    teacher = torch.zeros(1, 1, 8, 8)
    planner = torch.full((1, 1, 8, 8), torch.log(torch.tensor(2.0)))
    normalized = normalize_log_depth_pair(teacher, planner)
    assert float(normalized["teacher"].max()) == pytest.approx(1.0)
    assert float(normalized["planner"].min()) == pytest.approx(0.0)
    assert float(normalized["teacher"].min()) > float(normalized["planner"].max())


def test_sample_metrics_use_full_feature_and_depth_composites() -> None:
    target_log = torch.linspace(-1, 1, 64).reshape(1, 1, 8, 8)
    metrics = _sample_metrics(
        teacher_dino_current=torch.zeros(1, 4, 3),
        planner_dino_current=torch.ones(1, 4, 3),
        teacher_dino_future=torch.zeros(1, 4, 3),
        planner_dino_future=torch.full((1, 4, 3), 2.0),
        planner_log_current=target_log + 3.0,
        planner_log_future=target_log + 3.0,
        target_log_current=target_log,
        target_log_future=target_log,
        sample_index=0,
    )
    assert metrics["dino_current_mse"] == pytest.approx(1.0)
    assert metrics["dino_future_mse"] == pytest.approx(4.0)
    assert metrics["depth_current_abs_rel"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["depth_future_abs_rel"] == pytest.approx(0.0, abs=1e-6)


def test_reference_suptitle_matches_supplied_style() -> None:
    title = _reference_suptitle(
        camera="main",
        instruction="x" * 90,
        metrics={
            "dino_current_mse": 0.0049,
            "dino_future_mse": 0.0073,
            "depth_current_abs_rel": 0.355,
            "depth_future_abs_rel": 0.387,
        },
    )
    assert title == (
        f"[main cam] {'x' * 80}\n"
        "dino_mse cur=0.0049 fut=0.0073  |  "
        "depth_absrel cur=0.355 fut=0.387"
    )


def test_reference_layout_saves_two_camera_figures_and_individual_panels(
    tmp_path: Path,
) -> None:
    panels = {
        name: Image.new("RGB", (224, 224), color=(index % 255, 20, 30))
        for index, name in enumerate(EXPECTED_PANEL_NAMES)
    }
    paths = save_reference_style_sample(
        output_dir=tmp_path,
        sample_index=0,
        instruction="pick up the black bowl",
        panels=panels,
        metrics={"dino_current_mse": 0.1, "dino_future_mse": 0.2,
                 "depth_current_abs_rel": 0.3, "depth_future_abs_rel": 0.4},
    )
    names = {path.name for path in paths}
    assert {"sample_00_main.png", "sample_00_wrist.png"} <= names
    assert {f"{name}.png" for name in EXPECTED_PANEL_NAMES} <= names
    assert "instruction.txt" in names
    for name in EXPECTED_PANEL_NAMES:
        with Image.open(tmp_path / f"{name}.png") as image:
            assert image.size == (224, 224)
    for camera in ("main", "wrist"):
        with Image.open(tmp_path / f"sample_00_{camera}.png") as image:
            assert image.size == (2181, 1137)
            assert image.mode == "RGBA"
            assert image.getpixel((0, image.height - 1)) == (255, 255, 255, 255)
