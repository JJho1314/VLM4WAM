"""Full-grid Baton semantic conditioning for the GE-Act LTX model."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.ltx_models.semantic_conditioning import (
    BATON_GRID_SIZE,
    BATON_NUM_KEYFRAMES,
    BATON_NUM_VIEWS,
    DEFAULT_FUTURE_KEYFRAME_INDICES,
    SemanticContext,
    build_patch_center_positions,
    build_semantic_plan_times,
)
from qwen35_baton.provider import BatonSemanticPlan, FrozenBatonPlanner


BATON_TOKENS_PER_FRAME = BATON_GRID_SIZE * BATON_GRID_SIZE
BATON_FEATURE_DIM = 1024
_BATON_TOKEN_TAIL = (
    BATON_NUM_VIEWS,
    BATON_NUM_KEYFRAMES,
    BATON_TOKENS_PER_FRAME,
    BATON_FEATURE_DIM,
)


def _positive_integer(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_latent_shape(
    latent_shape: tuple[int, int, int],
    *,
    n_previous: int,
) -> tuple[int, int, int]:
    if (
        not isinstance(latent_shape, tuple)
        or len(latent_shape) != 3
        or any(type(value) is not int for value in latent_shape)
    ):
        raise ValueError("latent_shape must be a three-integer (frames,height,width) tuple")
    if any(value <= 0 for value in latent_shape):
        raise ValueError("latent_shape dimensions must be positive")
    latent_frames, latent_height, latent_width = latent_shape
    if latent_frames <= n_previous:
        raise ValueError(
            "latent temporal geometry must contain a future latent position"
        )
    return latent_frames, latent_height, latent_width


def _adapter_device_dtype(adapter: nn.Module) -> tuple[torch.device, torch.dtype]:
    value = next(
        (
            tensor
            for tensor in (*adapter.parameters(), *adapter.buffers())
            if tensor.dtype.is_floating_point
        ),
        None,
    )
    if value is None:
        raise ValueError("semantic_adapter has no floating-point parameters or buffers")
    return value.device, value.dtype


def _validate_tokens(
    tokens: Any,
    *,
    adapter: nn.Module,
) -> torch.Tensor:
    if (
        not isinstance(tokens, torch.Tensor)
        or tokens.ndim != 5
        or tuple(tokens.shape[1:]) != _BATON_TOKEN_TAIL
        or tokens.shape[0] <= 0
    ):
        raise ValueError("Baton tokens must have shape [B,2,4,256,1024]")
    if not tokens.dtype.is_floating_point:
        raise TypeError("Baton tokens must have a floating dtype")
    adapter_device, adapter_dtype = _adapter_device_dtype(adapter)
    if tokens.device != adapter_device:
        raise ValueError("Baton token device must match semantic_adapter device")
    if tokens.dtype != adapter_dtype:
        raise ValueError("Baton token dtype must match semantic_adapter dtype")
    if not bool(torch.isfinite(tokens).all()):
        raise ValueError("Baton tokens must contain only finite values")
    return tokens


def _validate_positions(
    positions_xy: Any,
    *,
    tokens: torch.Tensor,
) -> torch.Tensor:
    expected = build_patch_center_positions(
        batch_size=int(tokens.shape[0]),
        num_views=BATON_NUM_VIEWS,
        num_keyframes=BATON_NUM_KEYFRAMES,
        grid_size=BATON_GRID_SIZE,
        device=tokens.device,
    )
    if (
        not isinstance(positions_xy, torch.Tensor)
        or tuple(positions_xy.shape) != tuple(expected.shape)
    ):
        raise ValueError(
            "Baton positions_xy must have shape [B,2,4,256,2]"
        )
    if positions_xy.dtype != torch.float32:
        raise TypeError("Baton positions_xy must have dtype float32")
    if positions_xy.device != tokens.device:
        raise ValueError("Baton positions_xy device must match the token device")
    if not bool(torch.isfinite(positions_xy).all()):
        raise ValueError("Baton positions_xy must contain only finite values")
    if not torch.equal(positions_xy, expected):
        raise ValueError("Baton positions_xy must be the exact normalized patch centers")
    return positions_xy


def _unpack_plan(
    plan: BatonSemanticPlan | torch.Tensor | Any,
    *,
    adapter: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if isinstance(plan, torch.Tensor):
        tokens = _validate_tokens(plan, adapter=adapter)
        positions_xy = build_patch_center_positions(
            batch_size=int(tokens.shape[0]),
            num_views=BATON_NUM_VIEWS,
            num_keyframes=BATON_NUM_KEYFRAMES,
            grid_size=BATON_GRID_SIZE,
            device=tokens.device,
        )
        return tokens, positions_xy, DEFAULT_FUTURE_KEYFRAME_INDICES

    tokens = _validate_tokens(getattr(plan, "tokens", None), adapter=adapter)
    future_indices = getattr(plan, "future_indices", None)
    if future_indices != DEFAULT_FUTURE_KEYFRAME_INDICES:
        raise ValueError("Baton future_indices must be exactly (0, 3, 5, 8)")
    if getattr(plan, "relevance", None) is not None:
        raise ValueError("Baton plans never contain relevance")
    for field in ("mask", "token_mask", "key_mask"):
        if getattr(plan, field, None) is not None:
            raise ValueError("Baton plans never contain token masks")
    positions_xy = _validate_positions(
        getattr(plan, "positions_xy", None),
        tokens=tokens,
    )
    return tokens, positions_xy, future_indices


def build_baton_semantic_context(
    model: nn.Module,
    plan: BatonSemanticPlan | torch.Tensor,
    *,
    n_previous: int,
    num_future_frames: int,
    latent_shape: tuple[int, int, int],
) -> SemanticContext:
    """Adapt a complete Baton plan into same-camera LTX semantic keys.

    ``semantic_plan_times`` are full-clip-normalized only at the adapter
    boundary. The returned ``SemanticContext.positions`` are unambiguously in
    latent-grid ``(t,y,x)`` coordinates.
    """

    if not bool(getattr(model, "semantic_plan_context", False)):
        raise ValueError("model semantic_plan_context must be enabled")
    adapter = getattr(model, "semantic_adapter", None)
    if not isinstance(adapter, nn.Module):
        raise ValueError("model must expose an enabled semantic_adapter")
    n_previous = _positive_integer(n_previous, name="n_previous")
    num_future_frames = _positive_integer(
        num_future_frames,
        name="num_future_frames",
    )
    latent_frames, latent_height, latent_width = _validate_latent_shape(
        latent_shape,
        n_previous=n_previous,
    )
    tokens, positions_xy, future_indices = _unpack_plan(
        plan,
        adapter=adapter,
    )
    if max(future_indices) >= num_future_frames:
        raise ValueError(
            "Baton future_indices exceed the available num_future_frames"
        )

    plan_times = build_semantic_plan_times(
        batch_size=int(tokens.shape[0]),
        n_view=BATON_NUM_VIEWS,
        n_previous=n_previous,
        num_future_frames=num_future_frames,
        num_latent_frames=latent_frames,
        indices=future_indices,
        device=tokens.device,
        dtype=torch.float32,
    )
    context = adapter(
        tokens,
        semantic_plan_times=plan_times,
        latent_height=latent_height,
        latent_width=latent_width,
        latent_num_frames=latent_frames,
        semantic_positions_xy=positions_xy,
    )
    if (
        tuple(context.hidden_states.shape[:2])
        != (int(tokens.shape[0]) * BATON_NUM_VIEWS, 1024)
        or tuple(context.positions.shape)
        != (int(tokens.shape[0]) * BATON_NUM_VIEWS, 1024, 3)
        or context.key_mask is not None
        or context.relevance is not None
    ):
        raise RuntimeError("semantic_adapter violated the full-grid Baton contract")
    return context


class FrozenDualCameraBatonPlanner(nn.Module):
    """Thin frozen GE-Act wrapper around the Stage-1 Baton provider."""

    def __init__(self, provider: FrozenBatonPlanner | nn.Module) -> None:
        super().__init__()
        if not isinstance(provider, nn.Module) or not callable(
            getattr(provider, "predict", None)
        ):
            raise TypeError("provider must be a torch module exposing predict")
        self.provider = provider
        self._freeze_for_inference()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        **kwargs: Any,
    ) -> "FrozenDualCameraBatonPlanner":
        return cls(
            FrozenBatonPlanner.from_checkpoint(
                checkpoint_dir,
                **kwargs,
            )
        )

    def _freeze_for_inference(self) -> None:
        self.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True) -> "FrozenDualCameraBatonPlanner":
        del mode
        self._freeze_for_inference()
        return self

    @torch.no_grad()
    def predict(
        self,
        current_images: torch.Tensor,
        instructions: Sequence[str],
        **kwargs: Any,
    ) -> BatonSemanticPlan:
        self._freeze_for_inference()
        plan = self.provider.predict(
            current_images,
            instructions,
            **kwargs,
        )
        if not isinstance(plan, BatonSemanticPlan):
            raise TypeError("provider.predict must return BatonSemanticPlan")
        return plan


__all__ = [
    "FrozenDualCameraBatonPlanner",
    "build_baton_semantic_context",
    "build_patch_center_positions",
]
