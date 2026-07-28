#!/usr/bin/env python3
"""Export sampled RGB, SigLIP2, and DA3 images from one LIBERO episode."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def sample_frame_indices(num_frames: int, stride: int) -> list[int]:
    """Return stride-spaced indices that always include the episode's last frame."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    indices = list(range(0, num_frames, stride))
    final_index = num_frames - 1
    if indices[-1] != final_index:
        indices.append(final_index)
    return indices


def artifact_paths(
    output_dir: Path,
    camera: str,
    frame_index: int,
) -> dict[str, Path]:
    """Return the three independent image paths for a camera/frame pair."""
    frame_dir = output_dir / camera / f"frame_{frame_index:06d}"
    return {
        "rgb": frame_dir / "rgb.png",
        "siglip_pca": frame_dir / "siglip_pca.png",
        "da3_depth": frame_dir / "da3_depth.png",
    }


def _robust_unit_interval(values: torch.Tensor) -> torch.Tensor:
    """Normalize the final channel dimension with shared 2–98% ranges."""
    flat = values.reshape(-1, values.shape[-1])
    low = torch.quantile(flat, 0.02, dim=0)
    high = torch.quantile(flat, 0.98, dim=0)
    return ((values - low) / (high - low + 1e-6)).clamp(0, 1)


def siglip_pca_images(
    features: torch.Tensor,
    *,
    grid_size: int,
    output_size: int,
) -> np.ndarray:
    """Project all SigLIP grids through one PCA basis and display range."""
    if (
        features.ndim != 3
        or grid_size <= 0
        or features.shape[1] != grid_size * grid_size
        or features.shape[2] < 3
    ):
        raise ValueError(
            "SigLIP features must be [frames, grid_size squared, dim>=3]"
        )
    if output_size <= 0:
        raise ValueError("output_size must be positive")

    features_cpu = features.detach().float().cpu()
    flat = features_cpu.reshape(-1, features_cpu.shape[-1])
    centered = flat - flat.mean(dim=0, keepdim=True)
    _, _, vectors = torch.linalg.svd(centered, full_matrices=False)
    projected = (centered @ vectors[:3].T).reshape(
        features_cpu.shape[0],
        grid_size,
        grid_size,
        3,
    )
    projected = _robust_unit_interval(projected)
    resized = F.interpolate(
        projected.permute(0, 3, 1, 2),
        size=(output_size, output_size),
        mode="nearest",
    ).permute(0, 2, 3, 1)
    return (resized * 255.0).round().to(torch.uint8).numpy()


def da3_depth_images(depth: torch.Tensor) -> np.ndarray:
    """Colorize DA3 depth using one disparity range for the whole episode."""
    depth_cpu = depth.detach().float().cpu()
    if depth_cpu.ndim != 3 or not bool(torch.all(depth_cpu > 0)):
        raise ValueError("DA3 depth must be positive [frames,height,width]")

    disparity = depth_cpu.reciprocal()
    low = torch.quantile(disparity, 0.02)
    high = torch.quantile(disparity, 0.98)
    normalized = ((disparity - low) / (high - low + 1e-6)).clamp(0, 1)

    from matplotlib import colormaps

    rgb = colormaps["turbo"](normalized.numpy())[..., :3]
    return np.rint(rgb * 255.0).astype(np.uint8)
