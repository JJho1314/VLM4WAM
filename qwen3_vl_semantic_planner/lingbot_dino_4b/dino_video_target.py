"""Online DINO-video target encoder for the 4B lingbot-aligned planner (block C).

Thin adapter around lingbot-vla-v2's ``DinoVideoTeacher`` (lumos_dinov3 NaViT video ViT). Produces the
MSE regression targets: for each of our K future keyframes it runs the teacher on a 3-frame clip
``[warmup(=current), current, keyframe_k]`` and returns that keyframe's 256 patch tokens (dim 1024),
matching ``LingbotDinoPlanHead``'s ``[B, K*256, 1024]`` output.

Faithful to the teacher's contract (see LINGBOT_DINO_SPEC.md): input video ``[B,C,T,H,W]`` bf16, RGB→[0,1]
→ ImageNet mean/std, resized to ``input_size`` (256); ``get_future_feature`` uses the final-block
LayerNorm-normalized patches and returns the LAST frame. Target is detached (no grad to the teacher).

VALIDATION GATE: importing lumos_dinov3 + building from the released ``dino_video/`` ckpt must succeed in
the training env (flex_attention vs sdpa attention_mode). Validate right after the 6b download completes.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# box default; override on other hosts (e.g. HPC3) via env LINGBOT_SRC_ROOT.
_LINGBOT_ROOT = os.environ.get("LINGBOT_SRC_ROOT", "/data/LFT-W02_data/junjie/VLA_WM/lingbot-vla-v2")

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _import_teacher_builder(lingbot_root: str):
    """Import lingbot's DINO-video teacher builder WITHOUT running the heavy ``lingbotvla.models``
    package __init__ (which imports Auto* APIs removed in transformers 5.x). We pre-stub the parent
    packages with the correct ``__path__`` so only the self-contained ``dino_video`` subtree executes.
    """
    if lingbot_root not in sys.path:
        sys.path.insert(0, lingbot_root)
    for pkg, sub in [
        ("lingbotvla", "lingbotvla"),
        ("lingbotvla.models", "lingbotvla/models"),
        ("lingbotvla.models.vla", "lingbotvla/models/vla"),
        ("lingbotvla.models.vla.vision_models", "lingbotvla/models/vla/vision_models"),
    ]:
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [f"{lingbot_root}/{sub}"]
            m.__package__ = pkg
            sys.modules[pkg] = m
    from lingbotvla.models.vla.vision_models.dino_video.teacher import build_dino_video_teacher

    return build_dino_video_teacher


class DinoVideoTargetEncoder(nn.Module):
    def __init__(
        self,
        ckpt_path: str | Path,
        config_path: str | Path,
        *,
        input_size: int = 256,
        attention_mode: str = "flex_block_causal",
        cls_pool: str = "last",
        effective_fps: float = 1.0,
        device: str | torch.device = "cuda",
        lingbot_root: str | Path = _LINGBOT_ROOT,
    ) -> None:
        super().__init__()
        build_dino_video_teacher = _import_teacher_builder(str(lingbot_root))

        self.input_size = int(input_size)
        self.device = torch.device(device)
        self.effective_fps = float(effective_fps)
        self.teacher = build_dino_video_teacher(
            {
                "ckpt_path": str(ckpt_path),
                "config_path": str(config_path),
                "attention_mode": attention_mode,
                "input_size": self.input_size,
                "n_blocks": 1,
                "cls_pool": cls_pool,
                "effective_fps": effective_fps,
                "device": str(device),
            }
        )
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1, 1))

    def _prep(self, frames_b3hw: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) uint8/float RGB in [0,255] or [0,1] -> (B,3,input,input) float in ImageNet space."""
        x = frames_b3hw.float()
        if x.max() > 1.5:
            x = x / 255.0
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return x

    def _normalize_video(self, video: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=video.device, dtype=video.dtype)
        std = self.std.to(device=video.device, dtype=video.dtype)
        return (video - mean) / std

    @staticmethod
    def _validated_effective_fps(
        effective_fps: float | torch.Tensor,
    ) -> float | torch.Tensor:
        fps = torch.as_tensor(effective_fps, dtype=torch.float32)
        if fps.numel() == 0 or not bool(torch.isfinite(fps).all()) or not bool((fps > 0).all()):
            raise ValueError(
                "effective_fps must contain only positive finite values, got "
                f"{fps}"
            )
        if fps.ndim == 0:
            return float(fps.item())
        return fps

    def _flatten_future_fps(
        self,
        effective_fps: float | torch.Tensor | None,
        *,
        batch_size: int,
        num_keyframes: int,
    ) -> float | torch.Tensor:
        if effective_fps is None:
            return self.effective_fps
        fps = self._validated_effective_fps(effective_fps)
        if isinstance(fps, float):
            return fps
        expected = (batch_size, num_keyframes)
        if tuple(fps.shape) != expected:
            raise ValueError(
                "effective_fps for future keyframes must have shape "
                f"{expected}, got {tuple(fps.shape)}"
            )
        # Clips are concatenated keyframe-major: all B samples for k=0, then all B for k=1.
        return fps.transpose(0, 1).reshape(-1)

    @torch.no_grad()
    def encode_future_keyframes(
        self,
        current_b3hw: torch.Tensor,
        keyframes_b3hw: Sequence[torch.Tensor],
        *,
        effective_fps: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """current: (B,3,H,W); keyframes: list of K x (B,3,H,W) future frames.

        Returns (B, K*256, 1024) bf16 DINO-video targets (detached).
        """
        cur = self._prep(current_b3hw)  # (B,3,S,S)
        b = cur.shape[0]
        k = len(keyframes_b3hw)
        clips = []
        for kf in keyframes_b3hw:
            fut = self._prep(kf)  # (B,3,S,S)
            # clip [warmup=current, current, future_k] -> (B,3,T=3,S,S)
            clip = torch.stack([cur, cur, fut], dim=2)
            clips.append(clip)
        video = torch.cat(clips, dim=0)  # (B*K,3,3,S,S)
        video = self._normalize_video(video)
        fps = self._flatten_future_fps(
            effective_fps,
            batch_size=b,
            num_keyframes=k,
        )
        feats = self.teacher.get_future_feature(
            video,
            current_index=1,
            fps=fps,
        )  # (B*K, 256, 1024)
        tok, dim = feats.shape[1], feats.shape[2]
        feats = feats.view(k, b, tok, dim).permute(1, 0, 2, 3).reshape(b, k * tok, dim)
        return feats.detach()

    @torch.no_grad()
    def encode_current_and_future(
        self,
        current_b3hw: torch.Tensor,
        future_b3hw: torch.Tensor,
        *,
        effective_fps: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode LingBot's exact ``[warmup=current,current,future]`` target clip.

        Returns ``(current_patch, future_patch)``, each ``[B,256,1024]``.
        """
        current = self._prep(current_b3hw)
        future = self._prep(future_b3hw)
        video = torch.stack([current, current, future], dim=2)
        video = self._normalize_video(video)
        fps = (
            self.effective_fps
            if effective_fps is None
            else self._validated_effective_fps(effective_fps)
        )
        if isinstance(fps, torch.Tensor) and tuple(fps.shape) != (video.shape[0],):
            raise ValueError(
                "effective_fps for current/future encoding must have shape "
                f"({video.shape[0]},), got {tuple(fps.shape)}"
            )
        future_patch, current_patch = self.teacher.get_future_feature(
            video,
            return_current=True,
            current_index=1,
            fps=fps,
        )
        return current_patch.detach(), future_patch.detach()
