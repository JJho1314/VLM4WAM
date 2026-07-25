#!/usr/bin/env python3
"""Visualize dual-camera K4 VLM planner query-to-image attention.

The semantic planner's Perceiver implementation does not return attention
weights.  This module reconstructs the exact trained attention operation from
the forward-hook inputs without changing model code or checkpoint contents.
"""

from __future__ import annotations

import math
from types import TracebackType
from typing import Literal

import torch
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
