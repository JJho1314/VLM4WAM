"""Numerical contract tests for Baton Equation-8 feature regression."""

from __future__ import annotations

import pytest
import torch

from qwen35_baton.losses import compute_baton_planner_loss


def test_baton_loss_is_only_pointwise_continuous_feature_mse() -> None:
    prediction = torch.tensor([0.0, 2.0]).reshape(1, 1, 1, 1, 2)
    target = torch.tensor([1.0, 0.0]).reshape(1, 1, 1, 1, 2)

    loss = compute_baton_planner_loss(prediction, target)

    assert loss.mse.item() == pytest.approx(2.5)
    assert loss.total.item() == pytest.approx(2.5)
    assert set(vars(loss)) == {"mse", "total"}


def test_perfect_prediction_has_zero_loss() -> None:
    target = torch.randn(2, 2, 4, 256, 3)

    loss = compute_baton_planner_loss(target.clone(), target)

    torch.testing.assert_close(loss.total, torch.zeros_like(loss.total))


def test_loss_is_global_pointwise_mean() -> None:
    prediction = torch.tensor([0.0, 0.0, 4.0]).reshape(1, 1, 1, 1, 3)
    target = torch.zeros_like(prediction)

    loss = compute_baton_planner_loss(prediction, target)

    assert loss.total.item() == pytest.approx(16.0 / 3.0)


@pytest.mark.parametrize(
    ("prediction", "target"),
    [
        (
            torch.zeros(1, 1, 1, 1, 2, dtype=torch.float64),
            torch.zeros(1, 1, 1, 1, 2),
        ),
        (
            torch.zeros(1, 1, 1, 1, 2),
            torch.zeros(1, 1, 1, 2, 2),
        ),
        (
            torch.full((1, 1, 1, 1, 2), float("nan")),
            torch.zeros(1, 1, 1, 1, 2),
        ),
    ],
)
def test_loss_rejects_incompatible_or_nonfinite_features(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        compute_baton_planner_loss(prediction, target)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_low_precision_inputs_use_finite_fp32_loss_and_gradients(
    dtype: torch.dtype,
) -> None:
    target = torch.full((1, 1, 1, 2, 2), 65504.0, dtype=dtype)
    prediction = torch.full_like(target, 60000.0, requires_grad=True)

    loss = compute_baton_planner_loss(prediction, target)

    assert loss.mse.dtype == torch.float32
    assert bool(torch.isfinite(loss.total))
    loss.total.backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
    assert bool(prediction.grad.ne(0).any())
