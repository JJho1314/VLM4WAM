"""Baton Equation-8 continuous visual-feature regression."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BatonPlannerLoss:
    """The pointwise feature MSE and identical Stage-1 total."""

    mse: torch.Tensor
    total: torch.Tensor


def _validate_feature_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if tensor.ndim != 5 or any(size <= 0 for size in tensor.shape):
        raise ValueError(
            f"{name} must be nonempty [batch,camera,frame,patch,feature]"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")


def compute_baton_planner_loss(
    prediction: torch.Tensor,
    future_teacher: torch.Tensor,
) -> BatonPlannerLoss:
    """Compute Baton's pointwise L2 feature loss from Equation 8."""

    _validate_feature_tensor("prediction", prediction)
    _validate_feature_tensor("future_teacher", future_teacher)
    if prediction.shape != future_teacher.shape:
        raise ValueError("prediction and future_teacher must have the same shape")
    if prediction.dtype != future_teacher.dtype:
        raise TypeError("prediction and future_teacher must have the same dtype")
    if prediction.device != future_teacher.device:
        raise ValueError("prediction and future_teacher must be on the same device")

    mse = (prediction.float() - future_teacher.float()).square().mean()
    return BatonPlannerLoss(mse=mse, total=mse)


@dataclass(frozen=True)
class SpatialWSALoss:
    """Weighted per-layer DA3 alignment and its two components."""

    total: torch.Tensor
    cosine: torch.Tensor
    layernorm_mse: torch.Tensor


def compute_spatial_wsa_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    layer_weights: torch.Tensor,
) -> SpatialWSALoss:
    """WSA multi-layer DA3 alignment, identical to the Qwen3-VL planner line.

    ``prediction``/``target`` are ``[..., L, D]``. Per layer the loss is
    ``(1 - cos) + MSE(LayerNorm(pred), LayerNorm(target))``, weighted by
    ``layer_weights`` and averaged over layers.
    """

    if prediction.shape != target.shape or prediction.ndim < 3:
        raise ValueError("spatial prediction and target must share shape [...,L,D]")
    layers, dim = prediction.shape[-2:]
    if tuple(layer_weights.shape) != (layers,):
        raise ValueError(f"layer_weights must have shape ({layers},)")
    prediction = prediction.float().reshape(-1, layers, dim)
    target = target.detach().float().reshape(-1, layers, dim)
    cosine = 1.0 - torch.nn.functional.cosine_similarity(
        prediction, target, dim=-1
    ).mean(dim=0)
    layernorm_mse = (
        torch.nn.functional.layer_norm(prediction, (dim,))
        - torch.nn.functional.layer_norm(target, (dim,))
    ).square().mean(dim=(0, 2))
    weights = layer_weights.to(device=prediction.device, dtype=prediction.dtype)
    return SpatialWSALoss(
        total=((cosine + layernorm_mse) * weights).mean(),
        cosine=cosine.mean(),
        layernorm_mse=layernorm_mse.mean(),
    )
