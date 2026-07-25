#!/usr/bin/env python3
"""Visualize dual-camera K4 VLM planner query-to-image attention.

The semantic planner's Perceiver implementation does not return attention
weights.  This module reconstructs the exact trained attention operation from
the forward-hook inputs without changing model code or checkpoint contents.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import TracebackType
from typing import Literal, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


AttentionReduction = Literal["mean", "max"]


def _reshape_attention_heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
    """Convert ``[B, N, heads * dim_head]`` to ``[B, heads, N, dim_head]``."""

    batch, length, width = tensor.shape
    if width % heads:
        raise ValueError(f"attention width {width} is not divisible by {heads} heads")
    return tensor.view(batch, length, heads, -1).transpose(1, 2).contiguous()


def reconstruct_perceiver_attention(
    module: nn.Module,
    x: torch.Tensor,
    latents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct the exact softmax weights and values used by the resampler."""

    x_norm = module.norm1(x)
    latent_norm = module.norm2(latents)
    query = _reshape_attention_heads(module.to_q(latent_norm), module.heads)
    key, value = module.to_kv(torch.cat((x_norm, latent_norm), dim=-2)).chunk(
        2,
        dim=-1,
    )
    key = _reshape_attention_heads(key, module.heads)
    value = _reshape_attention_heads(value, module.heads)
    scale = 1.0 / math.sqrt(math.sqrt(module.dim_head))
    weights = torch.softmax(
        ((query * scale) @ (key * scale).transpose(-2, -1)).float(),
        dim=-1,
    ).to(query.dtype)
    return weights, value


def reduce_image_attention(
    weights: torch.Tensor,
    *,
    image_token_count: int,
    reduction: AttentionReduction = "mean",
) -> torch.Tensor:
    """Reduce heads and output queries while retaining image keys only.

    The key sequence also includes the semantic latent tokens fed to the head
    and the output query tokens appended inside ``PerceiverAttention``.  Qwen
    image tokens are the leading ``image_token_count`` columns.
    """

    if weights.ndim != 4:
        raise ValueError(
            "attention weights must have shape [batch, heads, queries, keys], "
            f"got {tuple(weights.shape)}"
        )
    if image_token_count <= 0 or image_token_count > weights.shape[-1]:
        raise ValueError(
            f"invalid image_token_count={image_token_count} for "
            f"{weights.shape[-1]} attention keys"
        )
    image_weights = weights[..., :image_token_count]
    if reduction == "mean":
        reduced = image_weights.mean(dim=(1, 2))
    elif reduction == "max":
        reduced = image_weights.amax(dim=(1, 2))
    else:
        raise ValueError(f"unsupported query reduction: {reduction!r}")
    if not torch.isfinite(reduced).all():
        raise ValueError("planner attention contains non-finite values")
    return reduced


class PlannerAttentionCapture:
    """Temporarily capture reduced attention from one Perceiver layer."""

    def __init__(
        self,
        module: nn.Module,
        *,
        image_token_count: int,
        reduction: AttentionReduction = "mean",
    ) -> None:
        self.module = module
        self.image_token_count = int(image_token_count)
        self.reduction = reduction
        self.maps: list[torch.Tensor] = []
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def _forward_hook(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        if len(inputs) != 2:
            raise ValueError(
                "expected PerceiverAttention inputs (x, latents), "
                f"received {len(inputs)} tensors"
            )
        with torch.no_grad():
            weights, _ = reconstruct_perceiver_attention(module, inputs[0], inputs[1])
            reduced = reduce_image_attention(
                weights,
                image_token_count=self.image_token_count,
                reduction=self.reduction,
            )
        self.maps.append(reduced.detach().float().cpu())

    def __enter__(self) -> "PlannerAttentionCapture":
        if self._handle is not None:
            raise RuntimeError("attention capture context is already active")
        self.maps.clear()
        self._handle = self.module.register_forward_hook(self._forward_hook)
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def merged_image_grid(
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
    expected_tokens: int,
) -> tuple[int, int]:
    """Restore the post-merge 2D Qwen image-token grid."""

    if image_grid_thw.numel() != 3:
        raise ValueError(
            "image_grid_thw must contain temporal, height, width; "
            f"got shape {tuple(image_grid_thw.shape)}"
        )
    temporal, height, width = (int(value) for value in image_grid_thw.tolist())
    merge = int(spatial_merge_size)
    if temporal != 1 or merge <= 0 or height % merge or width % merge:
        raise ValueError(
            f"unsupported Qwen image grid {(temporal, height, width)} "
            f"with spatial_merge_size={merge}"
        )
    merged = (height // merge, width // merge)
    token_count = merged[0] * merged[1]
    if token_count != int(expected_tokens):
        raise ValueError(
            f"Qwen merged grid has {token_count} tokens, expected "
            f"{int(expected_tokens)}"
        )
    return merged


def normalize_attention_stack(
    maps: torch.Tensor,
    *,
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.98,
) -> torch.Tensor:
    """Jointly normalize a camera's K attention maps to a shared [0, 1] scale."""

    maps = torch.as_tensor(maps).detach().float()
    if maps.ndim != 3:
        raise ValueError(f"attention stack must have shape [K, H, W], got {maps.shape}")
    if not torch.isfinite(maps).all():
        raise ValueError("attention maps contain non-finite values")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError(
            "normalization quantiles must satisfy "
            f"0 <= lower < upper <= 1, got {lower_quantile}, {upper_quantile}"
        )
    lower = torch.quantile(maps, lower_quantile)
    upper = torch.quantile(maps, upper_quantile)
    scale = upper - lower
    if float(scale) <= torch.finfo(maps.dtype).eps:
        return torch.zeros_like(maps)
    return ((maps - lower) / scale).clamp_(0.0, 1.0)


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"RGB image must have shape [H, W, 3], got {array.shape}")
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        upper = float(np.nanmax(array))
        if upper <= 1.0:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def attention_products(
    rgb: np.ndarray,
    normalized_map: torch.Tensor,
    *,
    alpha: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an unblended Turbo heatmap and an RGB overlay."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"overlay alpha must be within [0, 1], got {alpha}")
    rgb_uint8 = _validate_rgb(rgb)
    attention = torch.as_tensor(normalized_map).detach().float()
    if attention.ndim != 2:
        raise ValueError(f"normalized attention must be 2D, got {attention.shape}")
    if not torch.isfinite(attention).all():
        raise ValueError("normalized attention contains non-finite values")
    resized = F.interpolate(
        attention[None, None],
        size=rgb_uint8.shape[:2],
        mode="bilinear",
        align_corners=False,
    )[0, 0].clamp(0.0, 1.0)

    import matplotlib

    matplotlib.use("Agg", force=True)
    colorized = matplotlib.colormaps["turbo"](resized.cpu().numpy())[..., :3]
    heatmap = np.round(colorized * 255.0).astype(np.uint8)
    overlay = np.round(
        alpha * heatmap.astype(np.float32)
        + (1.0 - alpha) * rgb_uint8.astype(np.float32)
    )
    return heatmap, np.clip(overlay, 0, 255).astype(np.uint8)


def render_composite(
    output_path: str | Path,
    *,
    instruction: str,
    observations: Mapping[str, np.ndarray],
    overlays: Mapping[str, Sequence[np.ndarray]],
    offsets: Sequence[int],
) -> None:
    """Render a paper-ready 2x(1+K) dual-camera attention comparison."""

    if tuple(observations) != ("main", "wrist"):
        raise ValueError("observations must contain ordered cameras: main, wrist")
    if not offsets:
        raise ValueError("at least one future offset is required")
    for camera in observations:
        if camera not in overlays or len(overlays[camera]) != len(offsets):
            raise ValueError(
                f"{camera} overlays must contain exactly {len(offsets)} maps"
            )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    columns = 1 + len(offsets)
    figure, axes = plt.subplots(
        2,
        columns,
        figsize=(2.6 * columns, 5.5),
        squeeze=False,
    )
    headers = ["Observation", *[f"Planner t+{offset}" for offset in offsets]]
    row_names = {"main": "Main Camera", "wrist": "Wrist Camera"}

    for row, camera in enumerate(("main", "wrist")):
        panels = [_validate_rgb(observations[camera]), *overlays[camera]]
        for column, (axis, panel) in enumerate(zip(axes[row], panels)):
            axis.imshow(_validate_rgb(panel))
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(headers[column], fontsize=10, weight="semibold", pad=7)
            for spine in axis.spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(
            row_names[camera],
            fontsize=11,
            weight="bold",
            rotation=90,
            labelpad=10,
        )

    compact_instruction = " ".join(str(instruction).split())
    figure.suptitle(
        f'VLM Planner Query-to-Image Attention\nInstruction: "{compact_instruction}"',
        fontsize=13,
        weight="bold",
        y=0.985,
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.055,
        top=0.87,
        hspace=0.32,
        wspace=0.08,
    )
    figure.canvas.draw()
    for row, facecolor in enumerate(("#eef3f8", "#fffbe6")):
        boxes = [axis.get_position() for axis in axes[row]]
        left = min(box.x0 for box in boxes) - 0.018
        bottom = min(box.y0 for box in boxes) - 0.035
        right = max(box.x1 for box in boxes) + 0.012
        top = max(box.y1 for box in boxes) + 0.04
        container = FancyBboxPatch(
            (left, bottom),
            right - left,
            top - bottom,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            transform=figure.transFigure,
            linewidth=1.0,
            edgecolor="#333333",
            facecolor=facecolor,
            zorder=-1,
            clip_on=False,
        )
        figure.add_artist(container)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    with Image.open(destination) as rendered:
        rendered.convert("RGB").save(destination)
