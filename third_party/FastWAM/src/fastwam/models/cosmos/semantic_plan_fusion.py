"""Trainable same-position fusion for dense DINO and depth semantic plans."""
from __future__ import annotations

import math

import torch
from torch import nn


class DinoDepthPlanFusion(nn.Module):
    """Fuse aligned DINO and depth tokens without changing token geometry."""

    def __init__(
        self,
        feature_dim: int = 1024,
        max_tokens: int = 1024,
        initial_depth_gate: float = 0.1,
    ):
        super().__init__()
        for name, value in (
            ("feature_dim", feature_dim),
            ("max_tokens", max_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"{name} must be a strictly positive integer, got {value!r}"
                )
        if not 0.0 < initial_depth_gate < 1.0:
            raise ValueError("initial_depth_gate must be strictly between 0 and 1")

        self.feature_dim = feature_dim
        self.max_tokens = max_tokens
        self.dino_norm = nn.LayerNorm(self.feature_dim)
        self.depth_norm = nn.LayerNorm(self.feature_dim)
        self.dino_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.depth_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.depth_gate_logit = nn.Parameter(
            torch.tensor(
                math.log(initial_depth_gate / (1.0 - initial_depth_gate)),
                dtype=torch.float32,
            )
        )
        self.out_norm = nn.LayerNorm(self.feature_dim)

    def _validate(
        self,
        dino_plan: torch.Tensor,
        depth_plan: torch.Tensor,
    ) -> None:
        if dino_plan.shape != depth_plan.shape:
            raise ValueError(
                "DINO and depth plans must have the same shape, got "
                f"{tuple(dino_plan.shape)} and {tuple(depth_plan.shape)}"
            )
        expected = (self.max_tokens, self.feature_dim)
        if dino_plan.ndim != 3 or tuple(dino_plan.shape[1:]) != expected:
            raise ValueError(
                f"each plan must have shape [B, {self.max_tokens}, "
                f"{self.feature_dim}], got {tuple(dino_plan.shape)}"
            )
        if not torch.isfinite(dino_plan).all():
            raise ValueError("DINO plan contains non-finite values")
        if not torch.isfinite(depth_plan).all():
            raise ValueError("depth plan contains non-finite values")

    def forward(
        self,
        dino_plan: torch.Tensor,
        depth_plan: torch.Tensor,
    ) -> torch.Tensor:
        self._validate(dino_plan, depth_plan)
        dino = self.dino_proj(self.dino_norm(dino_plan))
        depth = self.depth_proj(self.depth_norm(depth_plan))
        gate = torch.sigmoid(self.depth_gate_logit).to(dtype=dino.dtype)
        return self.out_norm(dino + gate * depth)
