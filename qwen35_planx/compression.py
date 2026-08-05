"""Relevance-aware compression of dense 27x27 grounded spatial plans."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


_SOURCE_GRID = 27
_SOURCE_TOKENS = _SOURCE_GRID * _SOURCE_GRID
_COVERAGE_GRID = 8
_COVERAGE_TOKENS = _COVERAGE_GRID * _COVERAGE_GRID
_ROLES = 3
_MAX_EXACT_TOKENS = 32


def _build_area_overlap() -> Tensor:
    """Build normalized 2-D area overlaps in target/source raster order."""

    source_start = torch.arange(_SOURCE_GRID, dtype=torch.float64)
    source_end = source_start + 1.0
    target_start = (
        torch.arange(_COVERAGE_GRID, dtype=torch.float64)
        * (_SOURCE_GRID / _COVERAGE_GRID)
    )
    target_end = target_start + (_SOURCE_GRID / _COVERAGE_GRID)
    axis_overlap = (
        torch.minimum(target_end[:, None], source_end[None, :])
        - torch.maximum(target_start[:, None], source_start[None, :])
    ).clamp_min(0)
    overlap = torch.einsum(
        "iy,jx->ijyx",
        axis_overlap,
        axis_overlap,
    ).reshape(_COVERAGE_TOKENS, _SOURCE_TOKENS)
    return overlap / overlap.sum(dim=-1, keepdim=True)


def _build_source_positions() -> Tensor:
    """Build normalized patch-center coordinates in ``(x,y)`` order."""

    indices = torch.arange(_SOURCE_TOKENS, dtype=torch.float64)
    x = (indices.remainder(_SOURCE_GRID) + 0.5) / _SOURCE_GRID
    y = (torch.div(indices, _SOURCE_GRID, rounding_mode="floor") + 0.5) / (
        _SOURCE_GRID
    )
    return torch.stack((x, y), dim=-1)


_AREA_OVERLAP = _build_area_overlap()
_SOURCE_POSITIONS = _build_source_positions()


@dataclass(frozen=True)
class CompressedSemanticPlan:
    """Coverage tokens followed by masked, exact high-relevance tokens.

    ``positions`` contain normalized source-grid patch centers in ``(x,y)``
    order. ``source_indices`` is the source raster index for exact tokens and
    ``-1`` for pooled coverage or padding.
    """

    tokens: Tensor
    positions: Tensor
    mask: Tensor
    relevance: Tensor
    source_indices: Tensor


def _validate_inputs(
    features: Tensor,
    relevance: Tensor,
    *,
    top_k: int,
) -> None:
    if not isinstance(features, Tensor):
        raise TypeError("features must be a torch tensor")
    if not isinstance(relevance, Tensor):
        raise TypeError("relevance must be a torch tensor")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if not 0 <= top_k <= _MAX_EXACT_TOKENS:
        raise ValueError("top_k must be in [0,32]")

    if features.ndim != 5:
        raise ValueError("features must have shape [B,C,F,729,D]")
    if relevance.ndim != 5:
        raise ValueError("relevance must have shape [B,C,F,3,729]")
    if features.shape[-2] != _SOURCE_TOKENS:
        raise ValueError("features must contain exactly 729 source tokens")
    if relevance.shape[-2:] != (_ROLES, _SOURCE_TOKENS):
        raise ValueError("relevance must have shape [B,C,F,3,729]")
    if features.shape[:-2] != relevance.shape[:-2]:
        raise ValueError("features and relevance leading dimensions must match")
    if any(size <= 0 for size in features.shape[:-2]):
        raise ValueError("the three leading dimensions must be nonempty")
    if features.shape[-1] <= 0:
        raise ValueError("feature width must be positive")

    if not features.dtype.is_floating_point:
        raise TypeError("features and relevance must have floating dtypes")
    if not relevance.dtype.is_floating_point:
        raise TypeError("features and relevance must have floating dtypes")
    if features.dtype != relevance.dtype:
        raise TypeError("features and relevance must have the same dtype")
    if features.device != relevance.device:
        raise ValueError("features and relevance must be on the same device")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("features must contain only finite values")
    if not bool(torch.isfinite(relevance).all()):
        raise ValueError("relevance must contain only finite values")
    if bool((relevance < 0).any()):
        raise ValueError("relevance must be non-negative")


def compress_grounded_plan(
    features: Tensor,
    relevance: Tensor,
    *,
    top_k: int = _MAX_EXACT_TOKENS,
) -> CompressedSemanticPlan:
    """Compress dense spatial plans into coverage and exact task tokens.

    Args:
        features: Fused source tokens shaped ``[B,C,F,729,D]``.
        relevance: Source/target/action maps shaped ``[B,C,F,3,729]``.
        top_k: Number of exact-token slots to append, from zero through 32.

    Returns:
        A plan with ``64 + top_k`` slots per frame. All 64 coverage slots are
        valid. Exact slots are valid only for source positions whose maximum
        role relevance is positive; remaining slots are zero padded.
    """

    _validate_inputs(features, relevance, top_k=top_k)
    leading = features.shape[:-2]
    feature_width = features.shape[-1]
    peak_relevance = relevance.amax(dim=-2)

    overlap = _AREA_OVERLAP.to(device=features.device, dtype=features.dtype)
    source_positions = _SOURCE_POSITIONS.to(
        device=features.device,
        dtype=features.dtype,
    )
    overlap = overlap.reshape(
        *((1,) * len(leading)),
        _COVERAGE_TOKENS,
        _SOURCE_TOKENS,
    )
    coverage_weights = overlap * (1.0 + peak_relevance.unsqueeze(-2))
    coverage_weights = coverage_weights / coverage_weights.sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-6)

    coverage_tokens = torch.einsum(
        "...kn,...nd->...kd",
        coverage_weights,
        features,
    )
    coverage_positions = torch.einsum(
        "...kn,nc->...kc",
        coverage_weights,
        source_positions,
    )
    coverage_relevance = torch.einsum(
        "...kn,...n->...k",
        coverage_weights,
        peak_relevance,
    )
    coverage_mask = torch.ones(
        (*leading, _COVERAGE_TOKENS),
        device=features.device,
        dtype=torch.bool,
    )
    coverage_source_indices = torch.full(
        (*leading, _COVERAGE_TOKENS),
        -1,
        device=features.device,
        dtype=torch.long,
    )

    ranked_indices = torch.argsort(
        peak_relevance,
        dim=-1,
        descending=True,
        stable=True,
    )[..., :top_k]
    exact_relevance = torch.gather(peak_relevance, -1, ranked_indices)
    exact_mask = exact_relevance > 0
    exact_tokens = torch.gather(
        features,
        dim=-2,
        index=ranked_indices.unsqueeze(-1).expand(
            *leading,
            top_k,
            feature_width,
        ),
    )
    exact_positions = source_positions[ranked_indices]
    exact_source_indices = torch.where(
        exact_mask,
        ranked_indices,
        torch.full_like(ranked_indices, -1),
    )
    exact_tokens = torch.where(
        exact_mask.unsqueeze(-1),
        exact_tokens,
        torch.zeros_like(exact_tokens),
    )
    exact_positions = torch.where(
        exact_mask.unsqueeze(-1),
        exact_positions,
        torch.zeros_like(exact_positions),
    )
    exact_relevance = torch.where(
        exact_mask,
        exact_relevance,
        torch.zeros_like(exact_relevance),
    )

    return CompressedSemanticPlan(
        tokens=torch.cat((coverage_tokens, exact_tokens), dim=-2),
        positions=torch.cat((coverage_positions, exact_positions), dim=-2),
        mask=torch.cat((coverage_mask, exact_mask), dim=-1),
        relevance=torch.cat((coverage_relevance, exact_relevance), dim=-1),
        source_indices=torch.cat(
            (coverage_source_indices, exact_source_indices),
            dim=-1,
        ),
    )
