"""Numerical contract tests for continuous Baton Stage-1 objectives."""

from __future__ import annotations

import pytest
import torch

from qwen35_baton.losses import changed_patch_weights, compute_baton_planner_loss


def _features() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current_teacher = torch.zeros(2, 2, 256, 3)
    future_teacher = torch.ones(2, 2, 4, 256, 3)
    future_teacher[:, :, 1] *= 2.0
    future_teacher[:, :, :, 0] *= 3.0
    positive = future_teacher.clone()
    negative = -future_teacher
    return positive, negative, future_teacher, current_teacher


def test_changed_patch_weights_are_per_frame_and_bounded() -> None:
    current_teacher = torch.zeros(2, 2, 256, 3)
    future_teacher = torch.empty(2, 2, 4, 256, 3)
    for frame_index, value in enumerate((1.0, 2.0, 3.0, 4.0)):
        future_teacher[:, :, frame_index].fill_(value)

    weight = changed_patch_weights(future_teacher, current_teacher)

    assert weight.shape == (2, 2, 4, 256)
    assert float(weight.min()) >= 1.0
    assert float(weight.max()) <= 3.0
    torch.testing.assert_close(weight, torch.full_like(weight, 2.0))


def test_perfect_prediction_has_zero_primary_losses() -> None:
    positive, negative, future_teacher, current_teacher = _features()

    loss = compute_baton_planner_loss(
        positive=positive,
        negative=negative,
        future_teacher=future_teacher,
        current_teacher=current_teacher,
    )

    torch.testing.assert_close(loss.mse, torch.zeros_like(loss.mse))
    torch.testing.assert_close(
        loss.cosine, torch.zeros_like(loss.cosine), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(loss.delta, torch.zeros_like(loss.delta))


def test_static_copying_has_nonzero_delta_loss_when_future_changes() -> None:
    _, negative, future_teacher, current_teacher = _features()
    static_copy = current_teacher[:, :, None].expand_as(future_teacher).clone()

    loss = compute_baton_planner_loss(
        static_copy, negative, future_teacher, current_teacher
    )

    assert loss.delta > 0


def test_counterfactual_hinge_requires_correct_instruction_to_rank_better() -> None:
    positive, _, future_teacher, current_teacher = _features()
    good = positive
    bad = -future_teacher

    ranked = compute_baton_planner_loss(good, bad, future_teacher, current_teacher)
    reversed_rank = compute_baton_planner_loss(
        bad, good, future_teacher, current_teacher
    )

    assert ranked.instruction_counterfactual < reversed_rank.instruction_counterfactual


def test_total_is_the_exact_weighted_sum() -> None:
    positive, negative, future_teacher, current_teacher = _features()
    positive = positive + 0.5

    loss = compute_baton_planner_loss(
        positive, negative, future_teacher, current_teacher
    )

    torch.testing.assert_close(
        loss.total,
        loss.mse
        + 0.5 * loss.cosine
        + 0.5 * loss.delta
        + 0.2 * loss.instruction_counterfactual,
    )


def test_loss_reduces_each_sample_before_the_batch_mean() -> None:
    positive, negative, future_teacher, current_teacher = _features()
    positive[0] += 0.5
    positive[1] += 2.0

    batched = compute_baton_planner_loss(
        positive, negative, future_teacher, current_teacher
    )
    per_sample = [
        compute_baton_planner_loss(
            positive[index : index + 1],
            negative[index : index + 1],
            future_teacher[index : index + 1],
            current_teacher[index : index + 1],
        )
        for index in range(2)
    ]

    torch.testing.assert_close(
        batched.mse, torch.stack([item.mse for item in per_sample]).mean()
    )
    torch.testing.assert_close(
        batched.cosine, torch.stack([item.cosine for item in per_sample]).mean()
    )
    torch.testing.assert_close(
        batched.delta, torch.stack([item.delta for item in per_sample]).mean()
    )
    torch.testing.assert_close(
        batched.instruction_counterfactual,
        torch.stack([item.instruction_counterfactual for item in per_sample]).mean(),
    )


@pytest.mark.parametrize(
    ("tensor_name", "replacement"),
    [
        ("positive", lambda values: values[0].double()),
        ("future_teacher", lambda values: values[2][..., :-1, :]),
        ("current_teacher", lambda values: torch.full_like(values[3], float("nan"))),
    ],
)
def test_loss_rejects_incompatible_or_nonfinite_feature_tensors(
    tensor_name: str,
    replacement: object,
) -> None:
    positive, negative, future_teacher, current_teacher = _features()
    values = (positive, negative, future_teacher, current_teacher)
    replacements = {
        tensor_name: replacement(values),  # type: ignore[operator]
    }

    with pytest.raises((TypeError, ValueError)):
        compute_baton_planner_loss(
            replacements.get("positive", positive),
            replacements.get("negative", negative),
            replacements.get("future_teacher", future_teacher),
            replacements.get("current_teacher", current_teacher),
        )


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_low_precision_boundary_inputs_produce_finite_outputs_and_gradients(
    dtype: torch.dtype,
) -> None:
    current_teacher = torch.full((1, 1, 2, 2), -65504.0, dtype=dtype)
    future_teacher = torch.full((1, 1, 1, 2, 2), 65504.0, dtype=dtype)
    positive = torch.full_like(future_teacher, 60000.0, requires_grad=True)
    negative = -future_teacher

    weight = changed_patch_weights(future_teacher, current_teacher)
    loss = compute_baton_planner_loss(
        positive, negative, future_teacher, current_teacher
    )

    assert weight.dtype == torch.float32
    assert bool(torch.isfinite(weight).all())
    for component in (
        loss.mse,
        loss.cosine,
        loss.delta,
        loss.instruction_counterfactual,
        loss.total,
    ):
        assert component.dtype == torch.float32
        assert bool(torch.isfinite(component))
    loss.total.backward()
    assert positive.grad is not None
    assert bool(torch.isfinite(positive.grad).all())
    assert bool(positive.grad.ne(0).any())
