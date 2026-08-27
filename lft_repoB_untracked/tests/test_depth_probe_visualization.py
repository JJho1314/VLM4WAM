from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/qwen3_vl_semantic_planner"
    / "train_depth_probe_visualization.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("depth_probe_visualization", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_disjoint_probe_split_has_requested_counts_without_overlap():
    module = load_module()

    train, evaluation = module.select_disjoint_indices(
        length=100,
        train_count=20,
        eval_count=5,
    )

    assert len(train) == 20
    assert len(evaluation) == 5
    assert set(train).isdisjoint(evaluation)
    assert train == module.select_disjoint_indices(100, 20, 5)[0]


def test_relative_log_depth_is_scale_invariant_and_downsamples():
    module = load_module()
    depth = torch.arange(1, 65, dtype=torch.float32).reshape(1, 8, 8)

    first = module.relative_log_depth(depth, grid_size=4)
    second = module.relative_log_depth(depth * 7.0, grid_size=4)

    assert first.shape == (1, 4, 4)
    assert torch.allclose(first, second, atol=1e-6)
    assert float(first.mean()) == pytest.approx(0.0, abs=1e-6)


def test_linear_depth_probe_maps_tokens_to_grid():
    module = load_module()
    probe = module.LinearDepthProbe(feature_dim=6, grid_size=4)

    output = probe(torch.randn(2, 16, 6))

    assert output.shape == (2, 4, 4)


def test_scale_aligned_decode_and_metrics_are_exact_for_oracle_log_depth():
    module = load_module()
    target = torch.tensor([[[1.0, 2.0], [4.0, 8.0]]])
    relative = target.log() - target.log().mean(dim=(-2, -1), keepdim=True)

    decoded = module.decode_relative_log_depth(relative, target)
    metrics = module.compute_depth_metrics(decoded, target)

    assert torch.allclose(decoded, target, atol=1e-6)
    assert metrics["abs_rel"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["delta1"] == pytest.approx(1.0)


def test_depth_gradient_loss_is_zero_for_identical_maps():
    module = load_module()
    target = torch.randn(2, 4, 4)

    assert float(module.depth_gradient_loss(target, target)) == pytest.approx(0.0)


def test_best_probe_state_tracker_restores_lowest_loss_weights():
    module = load_module()
    tracker = module.BestProbeStateTracker()
    probe = module.LinearDepthProbe(feature_dim=2, grid_size=1)

    with torch.no_grad():
        probe.projection.weight.fill_(1.0)
    tracker.consider(epoch=1, loss=0.4, probe=probe)
    with torch.no_grad():
        probe.projection.weight.fill_(2.0)
    tracker.consider(epoch=2, loss=0.2, probe=probe)
    with torch.no_grad():
        probe.projection.weight.fill_(3.0)
    tracker.consider(epoch=3, loss=0.3, probe=probe)
    tracker.restore(probe)

    assert tracker.best_epoch == 2
    assert tracker.best_loss == pytest.approx(0.2)
    assert torch.equal(probe.projection.weight, torch.full_like(probe.projection.weight, 2.0))
