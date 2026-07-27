"""Online, frozen SigLIP2 patch-feature teacher for Baton supervision."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn


class FrozenSiglip2Teacher:
    """Extract detached penultimate SigLIP2 patch grids without caching targets."""

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        frame_microbatch_size: int = 32,
    ) -> None:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(str(model_name_or_path), use_fast=False)
        full_model = AutoModel.from_pretrained(
            str(model_name_or_path),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if not hasattr(full_model, "vision_model"):
            raise RuntimeError("expected a SigLIP2 model with a vision_model attribute")
        self._initialize(
            processor=processor,
            vision_model=full_model.vision_model,
            device=device,
            dtype=dtype,
            frame_microbatch_size=frame_microbatch_size,
        )
        del full_model

    @classmethod
    def from_components(
        cls,
        *,
        processor: Any,
        vision_model: nn.Module,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        frame_microbatch_size: int = 32,
    ) -> "FrozenSiglip2Teacher":
        """Build a teacher from injected components for tests or local deployment."""

        teacher = cls.__new__(cls)
        teacher._initialize(
            processor=processor,
            vision_model=vision_model,
            device=device,
            dtype=dtype,
            frame_microbatch_size=frame_microbatch_size,
        )
        return teacher

    def _initialize(
        self,
        *,
        processor: Any,
        vision_model: nn.Module,
        device: torch.device | str,
        dtype: torch.dtype,
        frame_microbatch_size: int,
    ) -> None:
        if not callable(processor):
            raise TypeError("processor must be callable")
        if not isinstance(vision_model, nn.Module):
            raise TypeError("vision_model must be a torch module")
        if type(frame_microbatch_size) is not int or frame_microbatch_size <= 0:
            raise ValueError("frame_microbatch_size must be a positive integer")
        self.processor = processor
        self.device = torch.device(device)
        self.dtype = dtype
        self.frame_microbatch_size = frame_microbatch_size
        self.model = vision_model.to(device=self.device, dtype=dtype)
        self.model.requires_grad_(False)
        self.model.eval()

    @staticmethod
    def _as_uint8_rgb(frames: torch.Tensor) -> torch.Tensor:
        if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"frames must be [N,3,H,W] RGB, got {tuple(frames.shape)}")
        if frames.dtype == torch.uint8:
            return frames
        if not frames.dtype.is_floating_point:
            raise TypeError("frames must use uint8 or normalized floating RGB values")
        if not bool(torch.isfinite(frames).all()):
            raise ValueError("frames must contain only finite values")
        if bool(((frames < -1) | (frames > 1)).any()):
            raise ValueError("normalized RGB frames must be in [-1,1]")
        return frames.add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8)

    def _pixel_values(self, frames: torch.Tensor) -> torch.Tensor:
        rgb = self._as_uint8_rgb(frames)
        processed = self.processor(images=list(rgb.cpu()), return_tensors="pt")
        if not isinstance(processed, Mapping) or "pixel_values" not in processed:
            raise ValueError("SigLIP2 processor must return pixel_values")
        pixel_values = processed["pixel_values"]
        if (
            not isinstance(pixel_values, torch.Tensor)
            or pixel_values.ndim != 4
            or pixel_values.shape[0] != frames.shape[0]
            or pixel_values.shape[1] != 3
        ):
            raise ValueError("SigLIP2 processor must return [N,3,H,W] pixel_values")
        if not bool(torch.isfinite(pixel_values).all()):
            raise ValueError("SigLIP2 processor returned nonfinite pixel_values")
        return pixel_values.to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode [N,3,H,W] RGB into [N,256,1024] detached patch features."""

        pixel_values = self._pixel_values(frames)
        output_chunks = []
        for start in range(0, pixel_values.shape[0], self.frame_microbatch_size):
            outputs = self.model(
                pixel_values[start : start + self.frame_microbatch_size],
                output_hidden_states=True,
            )
            hidden_states = getattr(outputs, "hidden_states", None)
            if not isinstance(hidden_states, (tuple, list)) or len(hidden_states) < 2:
                raise RuntimeError("SigLIP2 vision model must return at least two hidden states")
            tokens = hidden_states[-2]
            if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
                raise RuntimeError("SigLIP2 penultimate hidden state must be [N,P,D]")
            if tokens.shape[1] == 257:
                tokens = tokens[:, 1:]
            if tuple(tokens.shape[1:]) != (256, 1024):
                raise ValueError(
                    "expected 256 SigLIP2 patch tokens of width 1024, "
                    f"got {tuple(tokens.shape[1:])}"
                )
            if not bool(torch.isfinite(tokens).all()):
                raise ValueError("SigLIP2 teacher returned nonfinite patch features")
            output_chunks.append(tokens)
        if not output_chunks:
            raise ValueError("frames must contain at least one image")
        return torch.cat(output_chunks, dim=0).detach()

    @torch.no_grad()
    def encode_current(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``[B,2,3,H,W]`` RGB into detached ``[B,2,256,1024]`` features."""

        if not isinstance(images, torch.Tensor) or images.ndim != 5 or images.shape[1:3] != (2, 3):
            raise ValueError(f"current images must be [B,2,3,H,W], got {tuple(images.shape)}")
        batch_size = images.shape[0]
        return self._encode_frames(images.reshape(-1, *images.shape[-3:])).reshape(
            batch_size, 2, 256, 1024
        )

    @torch.no_grad()
    def encode_future(self, images: torch.Tensor) -> torch.Tensor:
        """Encode ``[B,2,4,3,H,W]`` RGB into detached future patch features."""

        if (
            not isinstance(images, torch.Tensor)
            or images.ndim != 6
            or images.shape[1:4] != (2, 4, 3)
        ):
            raise ValueError(f"future images must be [B,2,4,3,H,W], got {tuple(images.shape)}")
        batch_size = images.shape[0]
        return self._encode_frames(images.reshape(-1, *images.shape[-3:])).reshape(
            batch_size, 2, 4, 256, 1024
        )
