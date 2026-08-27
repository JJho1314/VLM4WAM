from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/qwen3_vl_semantic_planner"
    / "evaluate_lingbot_current_future_planner.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("lingbot_planner_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_eval_indices_is_deterministic_and_spans_dataset():
    module = load_module()

    assert module.select_eval_indices(length=10, count=4) == [0, 3, 6, 9]
    assert module.select_eval_indices(length=3, count=8) == [0, 1, 2]


def test_compute_branch_metrics_reports_exact_prediction_and_norms():
    module = load_module()
    target = torch.tensor(
        [[[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]],
        dtype=torch.float32,
    )

    metrics = module.compute_branch_metrics(target.clone(), target)

    assert metrics["mse_per_value"] == pytest.approx(0.0)
    assert metrics["smooth_l1_per_value"] == pytest.approx(0.0)
    assert metrics["mean_cosine"] == pytest.approx(1.0)
    assert metrics["token_retrieval_top1"] == pytest.approx(1.0)
    assert metrics["norm_ratio"] == pytest.approx(1.0)
    assert metrics["dispersion_ratio"] == pytest.approx(1.0)


def test_compute_future_baselines_uses_current_target_as_persistence():
    module = load_module()
    current = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    future = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])

    baselines = module.compute_future_baselines(current, future)

    assert baselines["persistence_mse_per_value"] == pytest.approx(1.0)
    assert baselines["persistence_mean_cosine"] == pytest.approx(0.0)
    assert baselines["collapsed_mean_mse_per_value"] == pytest.approx(0.25)


def test_joint_pca_maps_share_scale_and_return_rgb_grids():
    module = load_module()
    base = torch.arange(16 * 6, dtype=torch.float32).reshape(16, 6)
    maps = module.joint_pca_maps([base, base + 1.0], grid_size=4)

    assert len(maps) == 2
    assert maps[0].shape == (4, 4, 3)
    assert maps[1].shape == (4, 4, 3)
    assert all(torch.isfinite(item).all() for item in maps)
    assert all(0.0 <= float(item.min()) <= float(item.max()) <= 1.0 for item in maps)

