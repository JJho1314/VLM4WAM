#!/usr/bin/env python3
"""Render saved planner probes as aligned main/wrist 224px maps."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_dino_depth_probe_visualization import (  # noqa: E402
    DinoPCAProbe,
    OUTPUT_SIZE,
    TOKEN_GRID_SIZE,
    validate_token_features,
)


CAMERAS = ("main", "wrist")


def _split_width(
    value: torch.Tensor,
    *,
    name: str,
) -> dict[str, torch.Tensor]:
    if value.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")
    if value.shape[-1] % 2:
        raise ValueError(f"{name} width must be even, got {value.shape[-1]}")
    midpoint = value.shape[-1] // 2
    return {
        "main": value[..., :midpoint],
        "wrist": value[..., midpoint:],
    }


def _as_bhw(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    elif value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(
            f"{name} must be [B,H,W] or [B,1,H,W], got {tuple(value.shape)}"
        )
    return value


def split_rgb_cameras_224(value: Any) -> dict[str, Image.Image]:
    if isinstance(value, Image.Image):
        tensor = torch.from_numpy(__import__("numpy").asarray(value.convert("RGB")).copy())
    else:
        tensor = torch.as_tensor(value).detach().cpu()
    if tuple(tensor.shape) != (OUTPUT_SIZE, 2 * OUTPUT_SIZE, 3):
        raise ValueError(
            "RGB composite must be "
            f"[{OUTPUT_SIZE},{2 * OUTPUT_SIZE},3], got {tuple(tensor.shape)}"
        )
    if tensor.dtype != torch.uint8:
        maximum = float(tensor.max()) if tensor.numel() else 0.0
        tensor = tensor.float()
        if maximum <= 1.5:
            tensor = tensor * 255.0
        tensor = tensor.round().clamp(0, 255).to(torch.uint8)
    halves = _split_width(
        tensor.permute(2, 0, 1),
        name="RGB composite",
    )
    return {
        camera: Image.fromarray(half.permute(1, 2, 0).contiguous().numpy())
        for camera, half in halves.items()
    }


def project_dino_cameras_224(
    probe: DinoPCAProbe,
    features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_token_features(
        features,
        feature_dim=probe.mean.numel(),
        name="DINO features",
    )
    features = features.to(device=probe.mean.device, dtype=torch.float32)
    projected = (features - probe.mean) @ probe.basis
    projected = (
        (projected - probe.low) / (probe.high - probe.low).clamp_min(1e-6)
    ).clamp(0.0, 1.0)
    grid = projected.reshape(
        -1,
        TOKEN_GRID_SIZE,
        TOKEN_GRID_SIZE,
        3,
    ).permute(0, 3, 1, 2)
    return {
        camera: F.interpolate(
            half,
            size=(OUTPUT_SIZE, OUTPUT_SIZE),
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)
        for camera, half in _split_width(grid, name="DINO token grid").items()
    }


def resize_depth_target_cameras_224(
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    target = _as_bhw(target, name="target_depth").to(torch.float32)
    target = torch.nan_to_num(
        target,
        nan=1e-6,
        posinf=1e6,
        neginf=1e-6,
    ).clamp_min(1e-6)
    return {
        camera: F.interpolate(
            half.unsqueeze(1),
            size=(OUTPUT_SIZE, OUTPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        for camera, half in _split_width(
            target,
            name="dense Depth target",
        ).items()
    }


def decode_depth_cameras_224(
    relative: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    relative = _as_bhw(
        relative,
        name="relative_log_prediction",
    ).to(torch.float32)
    if not bool(torch.isfinite(relative).all()):
        raise ValueError("relative_log_prediction contains non-finite values")
    target_cameras = resize_depth_target_cameras_224(target)
    result = {}
    for camera, half in _split_width(
        relative,
        name="Depth token grid",
    ).items():
        prediction = F.interpolate(
            half.unsqueeze(1),
            size=(OUTPUT_SIZE, OUTPUT_SIZE),
            mode="bicubic",
            align_corners=False,
        )[:, 0]
        prediction = prediction - prediction.mean(
            dim=(-2, -1),
            keepdim=True,
        )
        truth = target_cameras[camera].to(prediction.device)
        shift = (
            (truth.log() - prediction)
            .flatten(1)
            .median(dim=1)
            .values[:, None, None]
        )
        decoded = (prediction + shift).exp()
        if not bool(torch.isfinite(decoded).all()):
            raise ValueError(f"decoded {camera} depth contains non-finite values")
        result[camera] = decoded
    return result
