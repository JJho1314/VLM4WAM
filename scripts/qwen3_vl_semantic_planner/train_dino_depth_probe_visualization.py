#!/usr/bin/env python3
"""Fit lightweight DINO/Depth probes and render separate 224px outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


TOKEN_GRID_SIZE = 16
TOKEN_COUNT = TOKEN_GRID_SIZE * TOKEN_GRID_SIZE
OUTPUT_SIZE = 224
DINO_OUTPUT_NAMES = (
    "dino_teacher_current_224",
    "dino_planner_current_224",
    "dino_teacher_future_224",
    "dino_planner_future_224",
)
DEPTH_OUTPUT_NAMES = (
    "depth_target_current_224",
    "depth_planner_current_224",
    "depth_target_future_224",
    "depth_planner_future_224",
)
EXPECTED_SAMPLE_FILES = (
    "observation_current.png",
    "observation_future.png",
    "instruction.txt",
    *(f"{name}.png" for name in DINO_OUTPUT_NAMES),
    *(f"{name}.png" for name in DEPTH_OUTPUT_NAMES),
)


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


def _as_bhw(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    elif value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(f"{name} must be [B,H,W] or [B,1,H,W], got {tuple(value.shape)}")
    return value


def resize_depth_target_224(target_depth: torch.Tensor) -> torch.Tensor:
    target = _as_bhw(target_depth, name="target_depth").to(torch.float32)
    target = torch.nan_to_num(
        target,
        nan=1e-6,
        posinf=1e6,
        neginf=1e-6,
    ).clamp_min(1e-6)
    return F.interpolate(
        target.unsqueeze(1),
        size=(OUTPUT_SIZE, OUTPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    )[:, 0]


def decode_depth_224(
    relative_log_prediction: torch.Tensor,
    target_depth: torch.Tensor,
) -> torch.Tensor:
    prediction = _as_bhw(
        relative_log_prediction,
        name="relative_log_prediction",
    ).to(torch.float32)
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("relative_log_prediction contains non-finite values")
    prediction = F.interpolate(
        prediction.unsqueeze(1),
        size=(OUTPUT_SIZE, OUTPUT_SIZE),
        mode="bicubic",
        align_corners=False,
    )[:, 0]
    prediction = prediction - prediction.mean(dim=(-2, -1), keepdim=True)
    target = resize_depth_target_224(target_depth).to(prediction.device)
    target_log = target.log()
    shift = (target_log - prediction).flatten(1).median(dim=1).values.view(-1, 1, 1)
    decoded = (prediction + shift).exp()
    if not bool(torch.isfinite(decoded).all()):
        raise ValueError("decoded depth contains non-finite values")
    return decoded


def _rgb_image_224(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
    else:
        tensor = torch.as_tensor(value).detach().cpu()
        if tensor.ndim != 3:
            raise ValueError(f"RGB observation must be 3-D, got {tuple(tensor.shape)}")
        if tensor.shape[0] == 3 and tensor.shape[-1] != 3:
            tensor = tensor.permute(1, 2, 0)
        if tensor.shape[-1] != 3:
            raise ValueError(f"RGB observation must end in 3 channels, got {tuple(tensor.shape)}")
        if tensor.dtype != torch.uint8:
            maximum = float(tensor.max()) if tensor.numel() else 0.0
            tensor = tensor.float()
            if maximum <= 1.5:
                tensor = tensor * 255.0
            tensor = tensor.round().clamp(0, 255).to(torch.uint8)
        image = Image.fromarray(tensor.numpy())
    if image.height < OUTPUT_SIZE or image.width < OUTPUT_SIZE:
        raise ValueError(
            f"RGB observation must be at least {OUTPUT_SIZE}x{OUTPUT_SIZE}, "
            f"got {image.width}x{image.height}"
        )
    # FastWAM composes the external view on the left and wrist view on the right.
    image = image.crop((0, 0, OUTPUT_SIZE, OUTPUT_SIZE))
    if image.size != (OUTPUT_SIZE, OUTPUT_SIZE):
        raise RuntimeError(f"failed to produce a {OUTPUT_SIZE}x{OUTPUT_SIZE} observation")
    return image


def _dino_image_224(value: torch.Tensor) -> Image.Image:
    tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32)
    if tensor.shape != (3, OUTPUT_SIZE, OUTPUT_SIZE):
        raise ValueError(
            f"DINO map must be [3,{OUTPUT_SIZE},{OUTPUT_SIZE}], got {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("DINO map contains non-finite values")
    array = (
        tensor.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array)


def _depth_image_224(
    value: torch.Tensor,
    *,
    low: float,
    high: float,
) -> Image.Image:
    from matplotlib import colormaps

    tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32)
    if tensor.shape != (OUTPUT_SIZE, OUTPUT_SIZE):
        raise ValueError(
            f"depth map must be [{OUTPUT_SIZE},{OUTPUT_SIZE}], got {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("depth map contains non-finite values")
    normalized = ((tensor - float(low)) / max(float(high) - float(low), 1e-6)).clamp(0, 1)
    rgb = colormaps["viridis"](normalized.numpy(), bytes=True)[..., :3]
    return Image.fromarray(rgb)


def save_sample_outputs(
    *,
    output_dir: Path,
    current_rgb: Any,
    future_rgb: Any,
    instruction: str,
    dino_maps: dict[str, torch.Tensor],
    depth_maps: dict[str, torch.Tensor],
) -> list[Path]:
    output_dir = Path(output_dir)
    if set(dino_maps) != set(DINO_OUTPUT_NAMES):
        raise ValueError(f"DINO output names must be {DINO_OUTPUT_NAMES}")
    if set(depth_maps) != set(DEPTH_OUTPUT_NAMES):
        raise ValueError(f"Depth output names must be {DEPTH_OUTPUT_NAMES}")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for filename, value in (
        ("observation_current.png", current_rgb),
        ("observation_future.png", future_rgb),
    ):
        path = output_dir / filename
        _rgb_image_224(value).save(path)
        paths.append(path)

    instruction_path = output_dir / "instruction.txt"
    instruction_path.write_text(instruction.strip() + "\n", encoding="utf-8")
    paths.append(instruction_path)

    for name in DINO_OUTPUT_NAMES:
        path = output_dir / f"{name}.png"
        _dino_image_224(dino_maps[name]).save(path)
        paths.append(path)

    bounds = {}
    for time_name in ("current", "future"):
        target = torch.as_tensor(
            depth_maps[f"depth_target_{time_name}_224"]
        ).detach().to(device="cpu", dtype=torch.float32)
        quantiles = torch.quantile(target, torch.tensor([0.02, 0.98]))
        bounds[time_name] = (float(quantiles[0]), float(quantiles[1]))
    for name in DEPTH_OUTPUT_NAMES:
        time_name = "current" if "current" in name else "future"
        path = output_dir / f"{name}.png"
        low, high = bounds[time_name]
        _depth_image_224(depth_maps[name], low=low, high=high).save(path)
        paths.append(path)

    if {path.name for path in paths} != set(EXPECTED_SAMPLE_FILES):
        raise RuntimeError("sample output layout is incomplete")
    return paths
