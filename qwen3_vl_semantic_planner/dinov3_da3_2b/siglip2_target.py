"""Online SigLIP2 target encoder for the 2B SigLIP2+DA3 planner line.

Drop-in sibling of ``Dinov3TargetEncoder``: same
``encode_future_keyframes(current, keyframes) -> [B, K*tok, D]`` contract, but the video target is
**SigLIP2** instead of DINOv3.

Feature definition follows ``build_siglip2_semantic_plan_labels.py`` (the offline semantic_plan
label builder): the **penultimate** vision encoder layer's spatial tokens — NOT last_hidden_state,
NOT the pooled head — then ``F.adaptive_avg_pool2d`` to ``grid_size`` if the native grid differs.
grid=16 -> 256 tokens matches the DA3 depth teacher's 256 (224/14), so the double head keeps a
uniform grid.

**Which checkpoint you pick decides whether this line can drive the WM:**
  * ``siglip2-so400m-patch14-384`` (dim **1152**) is the ONLY one that matches what the Cosmos WM
    consumes as its ``semantic_plan`` cross-attention KV. 384 = native res / patch 14 -> 27x27 = 729
    tokens (``interpolate_pos_encoding=False``), pooled 27->16. Label-exact, but 729 teacher tokens
    cost ~1.97 s/it.
  * ``siglip2-large-patch16-256`` (dim **1024**) is what the current run uses: native 256 / patch 16
    -> 16x16 = 256 tokens exactly, so **no pos-emb interpolation and no pooling**, ~1.47 s/it. It
    **breaks the WM link** — the WM's ``SemanticPlanContextAdapter`` takes 1152-d — making this a
    teacher-comparison experiment (SigLIP2 vs DINOv3), not a WM-closed-loop line.

HAZARD: large-256's 1024 collides with the legacy lingbot ``dino_video`` dim, so such a checkpoint is
dimensionally INDISTINGUISHABLE from a dino_video one — only ``planner_meta.json``'s
``video_target_type`` tells them apart.

Off-native ``input_size`` auto-enables position-embedding interpolation, which drifts the feature away
from the label builder's — prefer a checkpoint whose native res already gives the grid you want.
Native SigLIP2 has no CLS token; ``_infer_square_grid`` still tolerates an ``n^2+1`` layout.
Preprocessing = RGB -> [0,1] -> resize to ``input_size`` -> SigLIP mean/std 0.5 (i.e. [-1,1]). Target
is detached (frozen teacher, no grad).

Weights (HF snapshot dir with config.json + model.safetensors), overridable via env SIGLIP2_MODEL_DIR.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_SIGLIP2_DIR = os.environ.get(
    "SIGLIP2_MODEL_DIR",
    "/data/users/junjie/vlm4wam_2b/weights/siglip2-so400m-patch14-384",
)

# SigLIP normalization is mean=std=0.5 on [0,1] inputs -> [-1,1].
_SIGLIP_MEAN = (0.5, 0.5, 0.5)
_SIGLIP_STD = (0.5, 0.5, 0.5)


def _infer_square_grid(num_tokens: int) -> tuple[int, bool]:
    """(side, has_cls) for an n^2 or n^2+1 token layout."""
    side = int(math.sqrt(num_tokens))
    if side * side == num_tokens:
        return side, False
    side = int(math.sqrt(num_tokens - 1))
    if side * side == num_tokens - 1:
        return side, True
    raise RuntimeError(f"Cannot infer square SigLIP token grid from {num_tokens} tokens.")


class Siglip2TargetEncoder(nn.Module):
    def __init__(
        self,
        model_dir: str | Path = _SIGLIP2_DIR,
        *,
        input_size: int = 384,   # native; keeps interpolate_pos_encoding=False valid
        grid_size: int = 16,     # pool 27x27 -> 16x16 = 256 tok, matching the DA3 teacher
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,  # frozen teacher; bf16 halves the fwd on H100
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.input_size = int(input_size)
        self.grid_size = int(grid_size)
        self.device = torch.device(device)
        full = AutoModel.from_pretrained(str(model_dir), torch_dtype=dtype)
        if not hasattr(full, "vision_model"):
            raise RuntimeError("Expected an AutoModel with a vision_model attribute for SigLIP2.")
        # keep only the vision tower (the text tower is dead weight for a visual target)
        self.model = full.vision_model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.feature_dim = int(self.model.config.hidden_size)
        # The checkpoint's position embeddings are built for config.image_size (384 -> 27x27).
        # Any other input res must interpolate them; at native res we keep them untouched so the
        # feature matches the WM's semantic_plan labels exactly.
        self.native_size = int(getattr(self.model.config, "image_size", 384))
        self.interpolate_pos_encoding = self.input_size != self.native_size
        self.register_buffer("mean", torch.tensor(_SIGLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_SIGLIP_STD).view(1, 3, 1, 1))

    def _prep(self, frames_b3hw: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) RGB in [0,255] or [0,1] -> (B,3,S,S) float in SigLIP space on the teacher device."""
        x = frames_b3hw.to(self.device).float()
        if x.max() > 1.5:
            x = x / 255.0
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def _patch_tokens(self, images_b3hw: torch.Tensor) -> torch.Tensor:
        """(B,3,S,S) normalized -> (B, grid^2, D) penultimate spatial tokens, pooled to grid_size.

        Mirrors build_siglip2_semantic_plan_labels.py::encode_images so the planner target and the
        WM's semantic_plan conditioning are the SAME feature.
        """
        pv = images_b3hw.to(dtype=next(self.model.parameters()).dtype)
        hidden = self.model.embeddings(pv, interpolate_pos_encoding=self.interpolate_pos_encoding)
        layers = list(self.model.encoder.layers)
        if len(layers) < 2:
            raise RuntimeError(f"Expected at least two SigLIP2 vision layers, got {len(layers)}.")
        penultimate = None
        for idx, layer in enumerate(layers):
            hidden = layer(hidden, None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            if idx == len(layers) - 2:
                penultimate = hidden
                break  # nothing after the penultimate layer is needed
        if penultimate is None:
            raise RuntimeError("Failed to capture SigLIP2 penultimate spatial features.")

        side, has_cls = _infer_square_grid(penultimate.shape[1])
        patches = penultimate[:, 1:] if has_cls else penultimate
        if self.grid_size <= 0 or self.grid_size == side:
            return patches
        b, _, d = patches.shape
        grid = patches.reshape(b, side, side, d).permute(0, 3, 1, 2)          # (B,D,side,side)
        pooled = F.adaptive_avg_pool2d(grid.float(), (self.grid_size, self.grid_size))
        return pooled.permute(0, 2, 3, 1).reshape(b, self.grid_size * self.grid_size, d).to(patches.dtype)

    @torch.no_grad()
    def encode_future_keyframes(
        self,
        current_b3hw: torch.Tensor,
        keyframes_b3hw: Sequence[torch.Tensor],
        effective_fps=None,  # accepted for lingbot-encoder parity; SigLIP2 is per-image, unused
    ) -> torch.Tensor:
        """current: (B,3,H,W) [unused — SigLIP2 is per-image; kept for encoder-swap parity].
        keyframes: list of K x (B,3,H,W) future frames.

        Returns (B, K*N, D) SigLIP2 targets (detached), N = grid_size^2 (256), D = 1152."""
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
        Returns (current [B,N,D], future [B,N,D]) SigLIP2 tokens (detached). effective_fps unused."""
        cur = self._patch_tokens(self._prep(current_b3hw)).detach()   # (B, N, D)
        fut = self._patch_tokens(self._prep(keyframe_b3hw)).detach()  # (B, N, D)
        return cur, fut
