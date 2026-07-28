"""Text-conditioned SigLIP relevance maps from pairwise image-text scores."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def normalize_relevance(
    cam: torch.Tensor,
    low_q: float = 0.05,
    high_q: float = 0.95,
) -> torch.Tensor:
    """Scale each relevance map between its fixed lower and upper quantiles."""
    flat = cam.flatten(1)
    low = torch.quantile(flat, low_q, dim=1).view(-1, 1, 1)
    high = torch.quantile(flat, high_q, dim=1).view(-1, 1, 1)
    span = high - low
    normalized = torch.where(
        span > 0,
        (cam - low) / span.clamp_min(torch.finfo(cam.dtype).eps),
        torch.zeros_like(cam),
    )
    return normalized.clamp(0, 1)


def token_gradcam(
    tokens: torch.Tensor,
    gradients: torch.Tensor,
    *,
    grid_size: int = 16,
    output_size: int = 256,
) -> torch.Tensor:
    """Return normalized spatial Grad-CAM maps for flattened visual tokens."""
    expected = grid_size * grid_size
    if tokens.shape != gradients.shape or tokens.ndim != 3:
        raise ValueError("tokens and gradients must share [B,N,D]")
    if tokens.shape[1] != expected:
        raise ValueError(f"expected {expected} tokens")
    cam = (tokens.float() * gradients.float()).sum(dim=-1).relu()
    cam = cam.reshape(tokens.shape[0], 1, grid_size, grid_size)
    cam = F.interpolate(
        cam,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    return normalize_relevance(cam)


class SiglipPairGradCAM:
    """Frozen SigLIP Grad-CAM diagnostic for paired image and text inputs."""

    def __init__(
        self,
        model_dir: Path,
        device: torch.device,
        *,
        model: torch.nn.Module | None = None,
        processor: Any | None = None,
    ) -> None:
        self.device = torch.device(device)
        if model is None:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(str(model_dir), local_files_only=True)
        if processor is None:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
        if not hasattr(model, "vision_model") or not hasattr(model, "text_model"):
            raise RuntimeError("Expected a SigLIP model with vision_model and text_model towers.")

        vision_config = model.vision_model.config
        expected_config = {"image_size": 256, "patch_size": 16, "hidden_size": 1024}
        actual_config = {
            name: int(getattr(vision_config, name, -1)) for name in expected_config
        }
        if actual_config != expected_config:
            raise RuntimeError(
                "Expected siglip2-large-patch16-256 vision config "
                f"{expected_config}, got {actual_config}."
            )

        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.processor = processor

    def __call__(
        self,
        images: Sequence[Image.Image],
        phrases: Sequence[str],
    ) -> np.ndarray:
        if len(images) != len(phrases):
            raise ValueError("images and phrases must have the same length")

        inputs = self.processor(
            text=list(phrases),
            images=list(images),
            padding="max_length",
            return_tensors="pt",
        )
        pixel_values = inputs.pop("pixel_values").to(self.device)
        pixel_values.requires_grad_(True)
        outputs = self.model(
            pixel_values=pixel_values,
            **{name: value.to(self.device) for name, value in inputs.items()},
            output_hidden_states=True,
            return_dict=True,
        )
        tokens = outputs.vision_model_output.hidden_states[-2]
        tokens.retain_grad()
        score = outputs.logits_per_image.diagonal().sum()
        score.backward()
        if tokens.grad is None:
            raise RuntimeError("SigLIP penultimate tokens did not receive gradients")
        maps = token_gradcam(tokens.detach(), tokens.grad.detach())
        self.model.zero_grad(set_to_none=True)
        pixel_values.grad = None
        return maps.cpu().numpy().astype(np.float32)
