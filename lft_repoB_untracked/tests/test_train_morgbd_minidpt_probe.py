from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "qwen3_vl_semantic_planner"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_morgbd_minidpt_probe import (  # noqa: E402
    BestValidationState,
    DenseProbeCache,
    compute_log_depth_metrics,
    validate_disjoint_caches,
)


def _cache(records: list[dict[str, object]]) -> DenseProbeCache:
    return DenseProbeCache(
        features=torch.randn(len(records), 16, 8),
        log_depth=torch.randn(len(records), 1, 12, 12),
        records=records,
        grid=4,
        feature_dim=8,
        output_size=12,
    )


def test_dense_probe_cache_validates_shapes_and_records() -> None:
    cache = _cache(
        [
            {"suite": "libero_spatial", "dataset_index": 1, "time": "current"},
            {"suite": "libero_spatial", "dataset_index": 1, "time": "future"},
        ]
    )
    cache.validate()
    payload = cache.to_payload()
    restored = DenseProbeCache.from_payload(payload)
    restored.validate()
    assert restored.records == cache.records
    assert torch.equal(restored.features, cache.features)


def test_dense_probe_cache_rejects_wrong_dense_shape() -> None:
    cache = _cache(
        [{"suite": "libero_goal", "dataset_index": 2, "time": "current"}]
    )
    cache.log_depth = torch.randn(1, 12, 12)
    with pytest.raises(ValueError, match="log_depth"):
        cache.validate()


def test_train_and_eval_cache_records_must_be_disjoint() -> None:
    train = _cache(
        [{"suite": "libero_10", "dataset_index": 4, "time": "current"}]
    )
    evaluation = _cache(
        [{"suite": "libero_10", "dataset_index": 4, "time": "future"}]
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_disjoint_caches(train, evaluation)


def test_best_validation_state_restores_lowest_loss_weights() -> None:
    model = nn.Linear(2, 1, bias=False)
    tracker = BestValidationState()
    with torch.no_grad():
        model.weight.fill_(1.0)
    tracker.consider(step=10, loss=0.3, model=model)
    with torch.no_grad():
        model.weight.fill_(2.0)
    tracker.consider(step=20, loss=0.5, model=model)
    tracker.restore(model)
    assert tracker.best_step == 10
    assert tracker.best_loss == pytest.approx(0.3)
    assert torch.equal(model.weight, torch.ones_like(model.weight))


def test_log_depth_metrics_align_global_scale_and_measure_structure() -> None:
    target = torch.linspace(-1, 1, 64).reshape(1, 1, 8, 8)
    prediction = target + 3.0
    metrics = compute_log_depth_metrics(prediction, target)
    assert metrics["abs_rel"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["delta1"] == pytest.approx(1.0)
    assert metrics["pearson"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["gradient_error"] == pytest.approx(0.0, abs=1e-6)
