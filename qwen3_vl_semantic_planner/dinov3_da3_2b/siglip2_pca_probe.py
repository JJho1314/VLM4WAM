from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def sample_pca_tokens(
    features: torch.Tensor,
    *,
    max_tokens: int,
    seed: int,
) -> torch.Tensor:
    if features.ndim < 2:
        raise ValueError("PCA features must have a token and feature dimension")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    flat = features.detach().float().reshape(-1, features.shape[-1])
    if flat.shape[0] <= max_tokens:
        return flat
    generator = torch.Generator(device=flat.device).manual_seed(seed)
    indices = torch.randperm(
        flat.shape[0],
        generator=generator,
        device=flat.device,
    )[:max_tokens]
    return flat[indices]


def _anchor_component_signs(components: torch.Tensor) -> torch.Tensor:
    anchor_columns = components.abs().argmax(dim=1)
    anchor_values = components[
        torch.arange(components.shape[0], device=components.device),
        anchor_columns,
    ]
    signs = torch.where(anchor_values < 0, -1.0, 1.0)
    return components * signs[:, None]


def fit_fixed_pca(
    features: torch.Tensor,
    *,
    max_tokens: int = 50_000,
    seed: int = 0,
) -> dict[str, Any]:
    sampled = sample_pca_tokens(
        features,
        max_tokens=max_tokens,
        seed=seed,
    )
    if sampled.shape[0] < 4 or sampled.shape[1] < 3:
        raise ValueError("PCA requires at least four tokens and three features")
    mean = sampled.mean(dim=0)
    centered = sampled - mean
    covariance = centered.T @ centered / max(sampled.shape[0] - 1, 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    components = _anchor_component_signs(eigenvectors[:, -3:].T.flip(0))
    projected = centered @ components.T
    display_low = torch.quantile(projected, 0.02, dim=0)
    display_high = torch.quantile(projected, 0.98, dim=0)
    if torch.any(display_high <= display_low):
        raise ValueError("PCA display range is degenerate")
    return {
        "mean": mean,
        "components": components,
        "display_low": display_low,
        "display_high": display_high,
        "feature_dim": sampled.shape[1],
        "component_sign_rule": "largest_absolute_loading_positive",
        "max_tokens": max_tokens,
        "seed": seed,
        "sampled_token_count": sampled.shape[0],
    }


def project_fixed_pca(
    features: torch.Tensor,
    state: Mapping[str, Any],
) -> torch.Tensor:
    feature_dim = int(state["feature_dim"])
    if features.shape[-1] != feature_dim:
        raise ValueError(
            f"expected PCA feature dim {feature_dim}, got {features.shape[-1]}"
        )
    values = features.float()
    mean = torch.as_tensor(state["mean"], device=values.device)
    components = torch.as_tensor(
        state["components"],
        device=values.device,
    )
    low = torch.as_tensor(state["display_low"], device=values.device)
    high = torch.as_tensor(state["display_high"], device=values.device)
    projected = (values - mean) @ components.T
    return ((projected - low) / (high - low)).clamp(0, 1)
