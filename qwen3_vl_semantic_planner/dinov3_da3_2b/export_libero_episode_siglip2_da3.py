#!/usr/bin/env python3
"""Export sampled RGB, SigLIP2, and DA3 images from one LIBERO episode."""

from __future__ import annotations

from pathlib import Path


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
