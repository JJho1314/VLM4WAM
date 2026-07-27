"""Stage-1 objectives for continuous Baton feature prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from qwen35_baton.config import BatonLossWeights


@dataclass(frozen=True)
class BatonPlannerLoss:
    """Scalar Stage-1 objective terms and their approved weighted sum."""

    mse: torch.Tensor
    cosine: torch.Tensor
    delta: torch.Tensor
    instruction_counterfactual: torch.Tensor
    total: torch.Tensor


def _validate_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _validate_teacher_features(
    future_teacher: torch.Tensor, current_teacher: torch.Tensor
) -> None:
    _validate_tensor("future_teacher", future_teacher)
    _validate_tensor("current_teacher", current_teacher)
    if future_teacher.ndim != 5:
        raise ValueError(
            "future_teacher must have shape [batch, camera, frame, patch, feature]"
        )
    if current_teacher.ndim != 4:
        raise ValueError(
            "current_teacher must have shape [batch, camera, patch, feature]"
        )
    if any(size <= 0 for size in future_teacher.shape):
        raise ValueError("future_teacher dimensions must all be non-empty")
    expected_current_shape = future_teacher.shape[:2] + future_teacher.shape[3:]
    if current_teacher.shape != expected_current_shape:
        raise ValueError(
            "current_teacher must match future_teacher batch, camera, patch, and feature dimensions"
        )
    if current_teacher.dtype != future_teacher.dtype:
        raise TypeError("current_teacher and future_teacher must have the same dtype")
    if current_teacher.device != future_teacher.device:
        raise ValueError(
            "current_teacher and future_teacher must be on the same device"
        )


def _validate_loss_inputs(
    positive: torch.Tensor,
    negative: torch.Tensor,
    future_teacher: torch.Tensor,
    current_teacher: torch.Tensor,
    loss_weights: BatonLossWeights,
) -> None:
    _validate_teacher_features(future_teacher, current_teacher)
    _validate_tensor("positive", positive)
    _validate_tensor("negative", negative)
    if positive.shape != future_teacher.shape:
        raise ValueError("positive must have the same shape as future_teacher")
    if negative.shape != future_teacher.shape:
        raise ValueError("negative must have the same shape as future_teacher")
    for name, tensor in (("positive", positive), ("negative", negative)):
        if tensor.dtype != future_teacher.dtype:
            raise TypeError(f"{name} and future_teacher must have the same dtype")
        if tensor.device != future_teacher.device:
            raise ValueError(f"{name} and future_teacher must be on the same device")
    if not isinstance(loss_weights, BatonLossWeights):
        raise TypeError("loss_weights must be BatonLossWeights")


def changed_patch_weights(
    future_teacher: torch.Tensor, current_teacher: torch.Tensor
) -> torch.Tensor:
    """Weight patches by normalized within-frame teacher-feature change."""
    _validate_teacher_features(future_teacher, current_teacher)
    change = torch.linalg.vector_norm(
        future_teacher - current_teacher[:, :, None],
        dim=-1,
    )
    mean_change = change.mean(dim=-1, keepdim=True)
    return 1.0 + torch.clamp(
        change / (mean_change + 1e-6),
        min=0.0,
        max=2.0,
    )


def compute_baton_planner_loss(
    positive: torch.Tensor,
    negative: torch.Tensor,
    future_teacher: torch.Tensor,
    current_teacher: torch.Tensor,
    loss_weights: BatonLossWeights = BatonLossWeights(),
) -> BatonPlannerLoss:
    """Compute the approved Stage-1 loss with per-sample-first reductions."""
    _validate_loss_inputs(
        positive, negative, future_teacher, current_teacher, loss_weights
    )
    patch_weight = changed_patch_weights(future_teacher, current_teacher)
    patch_mse = (positive - future_teacher).square().mean(dim=-1)
    mse_per_sample = (patch_mse * patch_weight).flatten(1).sum(
        dim=1
    ) / patch_weight.flatten(1).sum(dim=1).clamp_min(1e-12)
    cosine_per_sample = (
        (1.0 - F.cosine_similarity(positive, future_teacher, dim=-1))
        .flatten(1)
        .mean(dim=1)
    )
    predicted_delta = positive - current_teacher[:, :, None]
    teacher_delta = future_teacher - current_teacher[:, :, None]
    delta_per_sample = (predicted_delta - teacher_delta).square().flatten(1).mean(dim=1)
    correct_distance = (
        (1.0 - F.cosine_similarity(positive, future_teacher, dim=-1))
        .flatten(1)
        .mean(dim=1)
    )
    wrong_distance = (
        (1.0 - F.cosine_similarity(negative, future_teacher, dim=-1))
        .flatten(1)
        .mean(dim=1)
    )
    instruction_cf_per_sample = torch.relu(
        loss_weights.counterfactual_margin + correct_distance - wrong_distance
    )
    mse = mse_per_sample.mean()
    cosine = cosine_per_sample.mean()
    delta = delta_per_sample.mean()
    instruction_cf = instruction_cf_per_sample.mean()
    total = (
        loss_weights.mse * mse
        + loss_weights.cosine * cosine
        + loss_weights.delta * delta
        + loss_weights.instruction_counterfactual * instruction_cf
    )
    return BatonPlannerLoss(
        mse=mse,
        cosine=cosine,
        delta=delta,
        instruction_counterfactual=instruction_cf,
        total=total,
    )
