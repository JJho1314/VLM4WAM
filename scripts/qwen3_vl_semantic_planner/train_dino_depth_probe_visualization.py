#!/usr/bin/env python3
"""Fit lightweight DINO/Depth probes and render separate 224px outputs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


TOKEN_GRID_SIZE = 16
TOKEN_COUNT = TOKEN_GRID_SIZE * TOKEN_GRID_SIZE
OUTPUT_SIZE = 224


def validate_token_features(
    features: torch.Tensor,
    *,
    feature_dim: int | None = None,
    name: str = "features",
) -> None:
    if features.ndim != 3 or features.shape[1] != TOKEN_COUNT:
        raise ValueError(
            f"{name} must be [B,{TOKEN_COUNT},D], got {tuple(features.shape)}"
        )
    if feature_dim is not None and features.shape[2] != int(feature_dim):
        raise ValueError(
            f"{name} feature dimension must be {feature_dim}, got {features.shape[2]}"
        )
    if features.shape[2] <= 0:
        raise ValueError(f"{name} feature dimension must be positive")
    if not bool(torch.isfinite(features).all()):
        raise ValueError(f"{name} contains non-finite values")


class DinoPCAProbe(nn.Module):
    def __init__(
        self,
        mean: torch.Tensor,
        basis: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
        *,
        output_size: int = OUTPUT_SIZE,
    ) -> None:
        super().__init__()
        if int(output_size) <= 0:
            raise ValueError("output_size must be positive")
        self.register_buffer("mean", mean.detach().to(torch.float32))
        self.register_buffer("basis", basis.detach().to(torch.float32))
        self.register_buffer("low", low.detach().to(torch.float32))
        self.register_buffer("high", high.detach().to(torch.float32))
        self.output_size = int(output_size)

    @classmethod
    def fit(
        cls,
        features: torch.Tensor,
        *,
        seed: int = 0,
        output_size: int = OUTPUT_SIZE,
    ) -> "DinoPCAProbe":
        validate_token_features(features, name="DINO PCA training features")
        flat = features.detach().to(device="cpu", dtype=torch.float32).flatten(0, 1)
        mean = flat.mean(dim=0)
        centered = flat - mean
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            _u, _s, basis = torch.pca_lowrank(
                centered,
                q=3,
                center=False,
                niter=4,
            )
        basis = basis[:, :3]
        # Remove the arbitrary PCA sign so serialized probes are reproducible.
        pivots = basis.abs().argmax(dim=0)
        signs = basis[pivots, torch.arange(3)].sign()
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        basis = basis * signs
        projected = centered @ basis
        low = torch.quantile(projected, 0.01, dim=0)
        high = torch.quantile(projected, 0.99, dim=0)
        if not all(
            bool(torch.isfinite(value).all())
            for value in (mean, basis, low, high)
        ):
            raise ValueError("DINO PCA probe contains non-finite statistics")
        return cls(
            mean,
            basis,
            low,
            high,
            output_size=output_size,
        )

    def project_224(self, features: torch.Tensor) -> torch.Tensor:
        validate_token_features(
            features,
            feature_dim=self.mean.numel(),
            name="DINO features",
        )
        projected = (features.to(torch.float32) - self.mean) @ self.basis
        projected = (projected - self.low) / (self.high - self.low).clamp_min(1e-6)
        grid = (
            projected.clamp(0.0, 1.0)
            .reshape(-1, TOKEN_GRID_SIZE, TOKEN_GRID_SIZE, 3)
            .permute(0, 3, 1, 2)
        )
        return F.interpolate(
            grid,
            size=(self.output_size, self.output_size),
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)
