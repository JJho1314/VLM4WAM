"""SigLIP2 semantic conditioning utilities for the GE-Act LTX video model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_FUTURE_KEYFRAME_INDICES = (0, 3, 5, 8)


def select_future_keyframes(
    future_video: torch.Tensor,
    indices: Sequence[int] = DEFAULT_FUTURE_KEYFRAME_INDICES,
) -> torch.Tensor:
    """Select canonical future frames from ``[B,V,T,C,H,W]`` video."""

    if future_video.ndim != 6:
        raise ValueError(f"future_video must be [B,V,T,C,H,W], got {tuple(future_video.shape)}")
    if not indices:
        raise ValueError("at least one semantic keyframe is required")
    if min(indices) < 0 or max(indices) >= future_video.shape[2]:
        raise ValueError(
            f"semantic keyframe indices {tuple(indices)} exceed {future_video.shape[2]} future frames"
        )
    index = torch.as_tensor(indices, device=future_video.device, dtype=torch.long)
    return future_video.index_select(2, index)


def build_semantic_plan_times(
    batch_size: int,
    n_view: int,
    n_previous: int,
    num_future_frames: int,
    num_latent_frames: int,
    indices: Sequence[int] = DEFAULT_FUTURE_KEYFRAME_INDICES,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return full-clip-normalized keyframe times in ``[B*V,K]`` order.

    Memory frames occupy integer latent positions ``[0, n_previous-1]`` and the
    complete future clip spans the remaining latent interval continuously.
    """

    if num_future_frames < 2:
        raise ValueError("num_future_frames must be at least two")
    if num_latent_frames <= n_previous:
        raise ValueError("num_latent_frames must include at least one future latent frame")
    if min(indices) < 0 or max(indices) >= num_future_frames:
        raise ValueError(f"invalid semantic keyframe indices {tuple(indices)}")

    future_fraction = torch.as_tensor(indices, device=device, dtype=dtype) / (num_future_frames - 1)
    future_latent_span = num_latent_frames - n_previous - 1
    latent_times = n_previous + future_fraction * future_latent_span
    normalized_times = latent_times / (num_latent_frames - 1)
    return normalized_times.unsqueeze(0).repeat(batch_size * n_view, 1)


class SemanticContextAdapter(nn.Module):
    """Adapt per-camera SigLIP2 grids to LTX tokens with explicit coordinates."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 2048,
        coordinate_dim: int = 256,
        num_views: int = 3,
    ) -> None:
        super().__init__()
        self.num_views = num_views
        self.input_norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coordinate_projection = nn.Sequential(
            nn.Linear(3, coordinate_dim),
            nn.SiLU(),
            nn.Linear(coordinate_dim, hidden_dim),
        )
        self.semantic_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.semantic_view_embedding = nn.Parameter(torch.zeros(num_views, hidden_dim))
        self.output_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def forward(
        self,
        semantic_tokens: torch.Tensor,
        semantic_plan_times: torch.Tensor,
        latent_height: int,
        latent_width: int,
        latent_num_frames: int = 6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if semantic_tokens.ndim != 5:
            raise ValueError(
                "semantic_tokens must be [B,V,K,P,D], "
                f"got {tuple(semantic_tokens.shape)}"
            )
        batch_size, n_view, num_keyframes, num_patches, _ = semantic_tokens.shape
        if n_view > self.num_views:
            raise ValueError(f"received {n_view} views, adapter supports {self.num_views}")
        grid_size = math.isqrt(num_patches)
        if grid_size * grid_size != num_patches:
            raise ValueError(f"SigLIP2 patch count must be square, got {num_patches}")

        if semantic_plan_times.ndim == 3:
            semantic_plan_times = semantic_plan_times.reshape(batch_size * n_view, num_keyframes)
        expected_times = (batch_size * n_view, num_keyframes)
        if tuple(semantic_plan_times.shape) != expected_times:
            raise ValueError(
                f"semantic_plan_times must have shape {expected_times}, got {tuple(semantic_plan_times.shape)}"
            )

        tokens = semantic_tokens.reshape(batch_size * n_view, num_keyframes, num_patches, -1)
        device = tokens.device
        position_dtype = torch.float32
        normalized_times = semantic_plan_times.to(device=device, dtype=position_dtype)

        grid_y = torch.linspace(0, latent_height - 1, grid_size, device=device, dtype=position_dtype)
        grid_x = torch.linspace(0, latent_width - 1, grid_size, device=device, dtype=position_dtype)
        y, x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        y = y.flatten()
        x = x.flatten()

        raw_t = normalized_times * (latent_num_frames - 1)
        raw_positions = torch.stack(
            (
                raw_t[:, :, None].expand(-1, -1, num_patches),
                y[None, None].expand(batch_size * n_view, num_keyframes, -1),
                x[None, None].expand(batch_size * n_view, num_keyframes, -1),
            ),
            dim=-1,
        )

        coord_y = y / max(latent_height - 1, 1) * 2 - 1
        coord_x = x / max(latent_width - 1, 1) * 2 - 1
        normalized_positions = torch.stack(
            (
                (normalized_times * 2 - 1)[:, :, None].expand(-1, -1, num_patches),
                coord_y[None, None].expand(batch_size * n_view, num_keyframes, -1),
                coord_x[None, None].expand(batch_size * n_view, num_keyframes, -1),
            ),
            dim=-1,
        )

        projected = self.feature_projection(self.input_norm(tokens))
        coordinate_embedding = self.coordinate_projection(
            normalized_positions.to(dtype=projected.dtype)
        )
        view_ids = torch.arange(n_view, device=device).repeat(batch_size)
        view_embedding = self.semantic_view_embedding[view_ids, None, None]
        projected = projected + coordinate_embedding + view_embedding + self.semantic_type_embedding
        projected = self.output_norm(projected)
        return (
            projected.flatten(1, 2),
            raw_positions.flatten(1, 2),
        )


class OnlineSiglip2SemanticEncoder:
    """Frozen online SigLIP2 feature extractor kept outside the trainable module."""

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        frame_microbatch_size: int = 32,
        expected_tokens: int = 256,
        expected_feature_dim: int = 1024,
    ) -> None:
        from transformers import AutoModel

        self.device = torch.device(device)
        self.dtype = dtype
        self.frame_microbatch_size = frame_microbatch_size
        self.expected_tokens = expected_tokens
        full_model = AutoModel.from_pretrained(
            str(model_name_or_path),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if not hasattr(full_model, "vision_model"):
            raise RuntimeError("expected a SigLIP2 model with a vision_model attribute")
        self.model = full_model.vision_model.to(self.device)
        del full_model
        self.model.requires_grad_(False)
        self.model.eval()
        model_config = getattr(self.model, "config", None)
        self.feature_dim = int(getattr(model_config, "hidden_size", expected_feature_dim))
        if self.feature_dim != expected_feature_dim:
            raise ValueError(
                f"expected SigLIP2 feature width {expected_feature_dim}, received {self.feature_dim}"
            )
        self.native_size = int(getattr(model_config, "image_size", 256))
        self.interpolate_pos_encoding = self.native_size != 256

    def _penultimate_patch_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Match the planner target: stop after the penultimate vision layer."""

        hidden_states = self.model.embeddings(
            pixel_values,
            interpolate_pos_encoding=self.interpolate_pos_encoding,
        )
        layers = list(self.model.encoder.layers)
        if len(layers) < 2:
            raise RuntimeError(f"expected at least two SigLIP2 vision layers, got {len(layers)}")
        for layer_index, layer in enumerate(layers):
            hidden_states = layer(hidden_states, None)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            if layer_index == len(layers) - 2:
                return hidden_states
        raise RuntimeError("failed to capture penultimate SigLIP2 patch tokens")

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode ``[B,V,K,C,H,W]`` frames and return ``[B,V,K,256,1024]``."""

        if frames.ndim != 6 or frames.shape[3] != 3:
            raise ValueError(f"frames must be [B,V,K,3,H,W], got {tuple(frames.shape)}")
        batch_size, n_view, num_frames, channels, height, width = frames.shape
        flat_frames = frames.reshape(-1, channels, height, width)
        if (height, width) != (256, 256):
            flat_frames = F.interpolate(flat_frames, size=(256, 256), mode="bicubic", align_corners=False)
        flat_frames = flat_frames.to(device=self.device, dtype=self.dtype)

        outputs = []
        for start in range(0, flat_frames.shape[0], self.frame_microbatch_size):
            tokens = self._penultimate_patch_tokens(
                flat_frames[start : start + self.frame_microbatch_size]
            )
            if tokens.shape[1] != self.expected_tokens:
                raise ValueError(
                    f"expected {self.expected_tokens} SigLIP2 tokens, received {tokens.shape[1]}"
                )
            outputs.append(tokens)

        features = torch.cat(outputs, dim=0)
        return features.reshape(batch_size, n_view, num_frames, self.expected_tokens, -1)
