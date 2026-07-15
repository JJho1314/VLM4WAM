#!/usr/bin/env python3
"""Feature-only MiniDPT decoder for dense MoRGBD depth visualization."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in range(min(32, channels), 0, -1):
        if channels % groups == 0:
            return groups
    raise ValueError(f"cannot choose GroupNorm groups for {channels} channels")


class _ResidualRefine(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class MiniDPTDepthProbe(nn.Module):
    """Decode a square spatial token grid into dense log depth.

    The decoder only consumes feature tokens. It has no RGB or target-depth input,
    so the same weights can be used on frozen teacher and planner-predicted tokens.
    """

    def __init__(
        self,
        *,
        in_dim: int = 1024,
        feat: int = 256,
        grid: int = 16,
        out_ch: int = 1,
        output_size: int = 224,
    ) -> None:
        super().__init__()
        if min(in_dim, feat, grid, out_ch, output_size) <= 0:
            raise ValueError("all MiniDPT dimensions must be positive")
        self.in_dim = int(in_dim)
        self.feat = int(feat)
        self.grid = int(grid)
        self.out_ch = int(out_ch)
        self.output_size = int(output_size)

        self.projection = nn.Conv2d(self.in_dim, self.feat, kernel_size=1)
        self.to_coarse = nn.Conv2d(
            self.feat,
            self.feat,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.to_native = nn.Identity()
        self.to_fine = nn.ConvTranspose2d(
            self.feat,
            self.feat,
            kernel_size=2,
            stride=2,
        )
        self.to_finest = nn.Sequential(
            nn.ConvTranspose2d(
                self.feat,
                self.feat,
                kernel_size=2,
                stride=2,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.feat,
                self.feat,
                kernel_size=2,
                stride=2,
            ),
        )
        self.refine_coarse = _ResidualRefine(self.feat)
        self.refine_native = _ResidualRefine(self.feat)
        self.refine_fine = _ResidualRefine(self.feat)
        self.refine_finest = _ResidualRefine(self.feat)
        self.head = nn.Sequential(
            nn.Conv2d(self.feat, self.feat // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.feat // 2, self.out_ch, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        expected = (self.grid * self.grid, self.in_dim)
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != expected:
            raise ValueError(
                f"tokens must be [B,{expected[0]},{expected[1]}], "
                f"got {tuple(tokens.shape)}"
            )
        batch = tokens.shape[0]
        value = tokens.transpose(1, 2).reshape(
            batch,
            self.in_dim,
            self.grid,
            self.grid,
        ).contiguous()
        value = self.projection(value)
        coarse = self.refine_coarse(self.to_coarse(value))
        native = self.refine_native(self.to_native(value))
        fine = self.refine_fine(self.to_fine(value))
        finest = self.refine_finest(self.to_finest(value))

        fused = F.interpolate(
            coarse,
            size=native.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ) + native
        fused = F.interpolate(
            fused,
            size=fine.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ) + fine
        fused = F.interpolate(
            fused,
            size=finest.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ) + finest
        fused = F.interpolate(
            fused,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        return self.head(fused)

    def config(self) -> dict[str, int]:
        return {
            "in_dim": self.in_dim,
            "feat": self.feat,
            "grid": self.grid,
            "out_ch": self.out_ch,
            "output_size": self.output_size,
        }


def _as_b1hw(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim == 3:
        value = value.unsqueeze(1)
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"{name} must be [B,H,W] or [B,1,H,W], got {tuple(value.shape)}")
    return value


def dense_log_depth_target(
    depth: torch.Tensor,
    *,
    output_size: int = 224,
) -> torch.Tensor:
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    depth = _as_b1hw(depth, name="depth").to(torch.float32)
    depth = torch.nan_to_num(
        depth,
        nan=1e-6,
        posinf=1e6,
        neginf=1e-6,
    ).clamp_min(1e-6)
    if tuple(depth.shape[-2:]) != (output_size, output_size):
        depth = F.interpolate(
            depth,
            size=(output_size, output_size),
            mode="bilinear",
            align_corners=False,
        )
    return depth.clamp_min(1e-6).log()


def silog_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    lambda_scale: float = 0.85,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    residual = prediction.float() - target.float()
    return residual.square().mean() - float(lambda_scale) * residual.mean().square()


def multiscale_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    scales: Iterable[int] = (1, 2, 4),
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must share [B,C,H,W]")
    loss = prediction.new_zeros(())
    used = 0
    for raw_scale in scales:
        scale = int(raw_scale)
        if scale <= 0:
            raise ValueError("gradient scales must be positive")
        if min(prediction.shape[-2:]) < scale:
            continue
        pred = F.avg_pool2d(prediction, scale) if scale > 1 else prediction
        truth = F.avg_pool2d(target, scale) if scale > 1 else target
        pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
        truth_dx = truth[..., :, 1:] - truth[..., :, :-1]
        pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
        truth_dy = truth[..., 1:, :] - truth[..., :-1, :]
        loss = loss + (pred_dx - truth_dx).abs().mean()
        loss = loss + (pred_dy - truth_dy).abs().mean()
        used += 1
    if used == 0:
        raise ValueError("no gradient scale fits the spatial dimensions")
    return loss
