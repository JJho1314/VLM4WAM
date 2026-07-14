"""Online DINOv3 target encoder for the 2B DINOv3+DA3 planner line.

Drop-in replacement for the 4B line's ``DinoVideoTargetEncoder``: same
``encode_future_keyframes(current, keyframes) -> [B, K*tok, D]`` contract, but the target is
Meta **DINOv3** (ViT-H+/16, HF-native ``DINOv3ViTModel``) run per-keyframe as a plain image ViT
(no 3-frame clip — DINOv3 is a single-image model). For each future keyframe we return that frame's
dense patch tokens; at ``input_size=256`` with patch 16 that is a 16x16 = 256-token grid (dim 1280),
matching ``LingbotDinoPlanHead``'s ``[B, K*256, D]`` output.

The ``last_hidden_state`` layout is ``[CLS, R registers, N patch tokens]`` (R = config
``num_register_tokens`` = 4 for this checkpoint); we strip the leading ``1 + R`` tokens and keep the
patch grid. Preprocessing = RGB -> [0,1] -> resize to ``input_size`` -> ImageNet mean/std. Target is
detached (frozen teacher, no grad).

Weights (HF snapshot dir with config.json + model.safetensors), overridable via env DINOV3_MODEL_DIR:
  /data/LFT-W02_data/junjie/VLA_WM/LAST-ViT/weights/dinov3_vith16plus/facebook/dinov3-vith16plus-pretrain-lvd1689m
Smaller drop-in variants (ViT-B/S, lower dim) live under FastWAM/checkpoints/dinov3.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_DINOV3_DIR = os.environ.get(
    "DINOV3_MODEL_DIR",
    "/data/LFT-W02_data/junjie/VLA_WM/LAST-ViT/weights/dinov3_vith16plus/"
    "facebook/dinov3-vith16plus-pretrain-lvd1689m",
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class Dinov3TargetEncoder(nn.Module):
    def __init__(
        self,
        model_dir: str | Path = _DINOV3_DIR,
        *,
        input_size: int = 256,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        try:
            from transformers import DINOv3ViTModel as _Model  # native class (transformers >= 4.56)
        except ImportError:  # pragma: no cover - fallback for name changes
            from transformers import AutoModel as _Model

        self.input_size = int(input_size)
        self.device = torch.device(device)
        self.model = _Model.from_pretrained(str(model_dir), torch_dtype=dtype).to(self.device).eval()
        self.model.requires_grad_(False)
        # [CLS, R registers, patches]; read R off the config so a variant swap survives.
        self.num_prefix = 1 + int(getattr(self.model.config, "num_register_tokens", 0))
        self.feature_dim = int(self.model.config.hidden_size)
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def _prep(self, frames_b3hw: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) RGB in [0,255] or [0,1] -> (B,3,S,S) float in ImageNet space on the teacher device."""
        x = frames_b3hw.to(self.device).float()
        if x.max() > 1.5:
            x = x / 255.0
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def _patch_tokens(self, images_b3hw: torch.Tensor) -> torch.Tensor:
        """(B,3,S,S) normalized -> (B, N, D) dense patch tokens (CLS + registers stripped)."""
        pv = images_b3hw.to(dtype=next(self.model.parameters()).dtype)
        out = self.model(pixel_values=pv)
        return out.last_hidden_state[:, self.num_prefix:]  # drop CLS + R registers

    @torch.no_grad()
    def encode_future_keyframes(
        self,
        current_b3hw: torch.Tensor,
        keyframes_b3hw: Sequence[torch.Tensor],
        effective_fps=None,  # accepted for lingbot-encoder parity; DINOv3 is per-image, unused
    ) -> torch.Tensor:
        """current: (B,3,H,W) [unused — DINOv3 is per-image; kept for encoder-swap parity].
        keyframes: list of K x (B,3,H,W) future frames.

        Returns (B, K*N, D) DINOv3 targets (detached), N = (input_size/16)^2, D = hidden_size (1280)."""
        prepped = [self._prep(kf) for kf in keyframes_b3hw]  # each (B,3,S,S)
        b = prepped[0].shape[0]
        k = len(prepped)
        batch = torch.cat(prepped, dim=0)  # (K*B, 3, S, S)
        feats = self._patch_tokens(batch)  # (K*B, N, D)
        tok, dim = feats.shape[1], feats.shape[2]
        feats = feats.view(k, b, tok, dim).permute(1, 0, 2, 3).reshape(b, k * tok, dim)
        return feats.detach()

    @torch.no_grad()
    def encode_current_and_future(self, current_b3hw, keyframe_b3hw, effective_fps=None):
        """LingBot current-alignment parity: encode the current frame AND one future keyframe.
        Returns (current [B,N,D], future [B,N,D]) DINOv3 patch tokens (detached). effective_fps unused."""
        cur = self._patch_tokens(self._prep(current_b3hw)).detach()   # (B, N, D)
        fut = self._patch_tokens(self._prep(keyframe_b3hw)).detach()  # (B, N, D)
        return cur, fut
