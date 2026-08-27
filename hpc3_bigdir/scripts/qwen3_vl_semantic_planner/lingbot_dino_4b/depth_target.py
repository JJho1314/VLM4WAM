"""Online LingBot-Depth target encoder for the 4B lingbot-aligned planner (future-depth head).

Faithful port of lingbot-vla-v2's depth-distillation target (``module_utils.build_depth_model`` +
``get_depth_target``): a TWO-STAGE frozen teacher

    MoGe-2 (monocular geometry)  RGB -> depth map
    LingBot-Depth / MoRGBD (RGB-D DINOv2 encoder)  (RGB, depth map) -> depth features

The alignment TARGET the head regresses is the **LingBot-Depth features** (README: "distilled from
LingBot-Depth"); MoGe-2 only produces the depth-map input that the RGB-D encoder needs. For each future
keyframe we return that frame's 256 depth-feature tokens (dim 1024), matching ``LingbotDinoPlanHead``'s
``[B, K*256, 1024]`` output.

Both teachers' DINOv2 backbones use xformers ``memory_efficient_attention``; the box's xformers has no
built CUDA op, so we set ``XFORMERS_DISABLED=1`` (their dinov2 falls back to
``F.scaled_dot_product_attention``). The ``moge``/``mdm`` packages are vendored inside the lingbot repo;
``utils3d`` is the MoGe-pinned commit installed into an isolated dir (kept off the starVLA site-packages).

VALIDATION GATE (2026-07-09, box GPU0, starVLA env): build + get_depth_target on a 256x256 sample ->
[B, 256, 1024] bf16. PASS.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# dinov2 (MoGe + MoRGBD) -> F.scaled_dot_product_attention instead of the unbuilt xformers op.
os.environ.setdefault("XFORMERS_DISABLED", "1")

# box defaults; override on other hosts (e.g. HPC3) via env LINGBOT_SRC_ROOT / UTILS3D_MOGE_PATH.
_LINGBOT_ROOT = os.environ.get("LINGBOT_SRC_ROOT", "/data/LFT-W02_data/junjie/VLA_WM/lingbot-vla-v2")
_VISION = "lingbotvla/models/vla/vision_models"
_UTILS3D_PINNED = os.environ.get("UTILS3D_MOGE_PATH", "/data/LFT-W02_data/junjie/weights/py_deps/utils3d_moge")


def _import_depth_builders(lingbot_root: str, utils3d_path: str):
    """Import MoGe-2 (``moge.model.v2.MoGeModel``) and MoRGBD (``mdm.model.v2.MDMModel``) from the
    lingbot-vendored packages, with the MoGe-pinned ``utils3d`` on the path."""
    for p in (utils3d_path, f"{lingbot_root}/{_VISION}/MoGe", f"{lingbot_root}/{_VISION}/lingbot-depth"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from moge.model.v2 import MoGeModel  # noqa: E402
    from mdm.model.v2 import MDMModel  # noqa: E402

    return MoGeModel, MDMModel


class DepthTargetEncoder(nn.Module):
    def __init__(
        self,
        moge_path: str | Path,
        morgbd_path: str | Path,
        *,
        input_size: int = 256,
        num_tokens: int = 256,
        resolution_level: int = 3,
        device: str | torch.device = "cuda",
        lingbot_root: str | Path = _LINGBOT_ROOT,
        utils3d_path: str | Path = _UTILS3D_PINNED,
    ) -> None:
        super().__init__()
        MoGeModel, MDMModel = _import_depth_builders(str(lingbot_root), str(utils3d_path))
        self.input_size = int(input_size)
        self.num_tokens = int(num_tokens)
        self.resolution_level = int(resolution_level)
        self.device = torch.device(device)

        moge = MoGeModel.from_pretrained(str(moge_path))
        for p in moge.parameters():
            p.requires_grad = False
        self.moge = moge.to(self.device).eval()

        morgbd = MDMModel.from_pretrained(str(morgbd_path))
        for p in morgbd.parameters():
            p.requires_grad = False
        self.morgbd = morgbd.to(self.device).eval()

    def _prep(self, frames_b3hw: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) uint8/float RGB in [0,255] or [0,1] -> (B,3,S,S) float in [0,255] (MoGe expects
        [0,255] then divides internally via _rgb_to_unit_float; we keep [0,255] and /255 here)."""
        x = frames_b3hw.float()
        if x.max() <= 1.5:  # already [0,1] -> back to [0,255] scale so the /255 below matches lingbot
            x = x * 255.0
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return x

    def _depth_target(self, images_b3hw: torch.Tensor) -> torch.Tensor:
        """Faithful ``get_depth_target`` for a single (B,3,S,S) RGB[0,255] batch -> (B, num_tokens, C)."""
        input_images = images_b3hw / 255.0  # _rgb_to_unit_float
        out = self.moge.infer(input_images, resolution_level=self.resolution_level,
                              num_tokens=self.num_tokens, apply_mask=False)
        depth_pred = torch.nan_to_num(out["depth"].squeeze().detach().clone(), nan=0.0, posinf=0.0, neginf=0.0)
        feat, _cls = self.morgbd.infer_feat(input_images, depth_pred, depth_down_scale=1,
                                            resolution_level=self.resolution_level,
                                            num_tokens=self.num_tokens, enable_depth_mask=False)
        feat = feat.permute(0, 2, 3, 1)                       # (B, H', W', C)
        feat = feat.reshape(feat.shape[0], -1, feat.shape[-1])  # (B, H'*W', C)
        return feat

    @torch.no_grad()
    def encode_future_keyframes(self, keyframes_b3hw: Sequence[torch.Tensor]) -> torch.Tensor:
        """keyframes: list of K x (B,3,H,W) future frames. Returns (B, K*num_tokens, C) bf16 (detached).

        Depth is monocular per-frame (no warmup clip, unlike DINO-video)."""
        prepped = [self._prep(kf) for kf in keyframes_b3hw]  # each (B,3,S,S)
        b = prepped[0].shape[0]
        k = len(prepped)
        batch = torch.cat(prepped, dim=0)  # (K*B, 3, S, S)
        feats = self._depth_target(batch)  # (K*B, num_tokens, C)
        tok, dim = feats.shape[1], feats.shape[2]
        feats = feats.view(k, b, tok, dim).permute(1, 0, 2, 3).reshape(b, k * tok, dim)
        return feats.detach().to(torch.bfloat16)
