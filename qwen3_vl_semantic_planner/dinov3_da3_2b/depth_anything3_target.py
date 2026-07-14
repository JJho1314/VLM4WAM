"""Online Depth-Anything-3 (DA3) target encoder for the 2B DINOv3+DA3 planner line.

Drop-in replacement for the 4B line's ``DepthTargetEncoder`` (MoGe-2 -> MoRGBD): same
``encode_future_keyframes(keyframes) -> [B, K*tok, D]`` monocular per-frame contract, but the target
is the **DA3 ViT-L encoder's dense patch tokens** (the DINOv2-style backbone features, NOT the scalar
depth map). We align to the geometry-pretrained encoder representation, mirroring how the lingbot line
aligns to LingBot-Depth/MoRGBD features.

DA3's backbone is a DINOv2-style ViT-L (patch 14, embed 1024) with ``cat_token=True`` so each patch
feature is ``concat([local, LayerNorm(global)]) -> dim 2048``. At ``process_res=224`` (14*16) the grid
is 16x16 = 256 tokens/keyframe, matching a grid-16 depth head; bump ``process_res`` (e.g. 504 -> 36x36)
for the native DA3 resolution. Preprocessing = RGB -> [0,1] -> resize to a square multiple of 14 ->
ImageNet mean/std. Target is detached (frozen teacher).

Checkpoint (HF-mixin dir with config.json + model.safetensors), overridable via env DA3_CKPT_DIR:
  /data/LFT-W02_data/junjie/VLA_WM/WSA/checkpoints/DA3-LARGE-1.1
DA3 code root (its ``src`` on sys.path), overridable via env DA3_CODE_ROOT:
  /data/LFT-W02_data/junjie/VLA_WM/Geometric-Action-Model/Depth-Anything-3
"""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_DA3_CKPT = os.environ.get("DA3_CKPT_DIR", "/data/LFT-W02_data/junjie/VLA_WM/WSA/checkpoints/DA3-LARGE-1.1")
_DA3_CODE_ROOT = os.environ.get(
    "DA3_CODE_ROOT", "/data/LFT-W02_data/junjie/VLA_WM/Geometric-Action-Model/Depth-Anything-3"
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# DA3's api.py eagerly imports export utilities (moviepy/pycolmap/trimesh/imageio/evo/plyfile/addict).
# For a frozen feature teacher none are needed; register lightweight stubs so the import succeeds.
_STUB_NAMES = (
    "moviepy", "moviepy.editor", "pycolmap", "trimesh",
    "imageio", "imageio.v2", "imageio.v3", "evo", "evo.core", "evo.core.trajectory", "plyfile",
)


class _StubModule(types.ModuleType):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        # Real dunders so inspect/import machinery (which scans all sys.modules) doesn't choke.
        self.__file__ = f"<stub:{name}>"
        self.__path__ = []

    def __getattr__(self, item):
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)  # let dunder lookups miss cleanly
        return type(f"_stub_{item}", (), {"__init__": lambda self, *a, **k: None})


def _install_da3_optional_stubs() -> None:
    def _real(name: str) -> bool:
        try:
            importlib.import_module(name)
            return True
        except Exception:
            return False

    for name in _STUB_NAMES:
        if name in sys.modules or _real(name):
            continue
        sys.modules[name] = _StubModule(name)
    if "addict" not in sys.modules and not _real("addict"):
        stub = _StubModule("addict")
        stub.Dict = dict  # minimal Dict shim
        sys.modules["addict"] = stub


def _import_da3(code_root: str):
    src = str(Path(code_root) / "src") if (Path(code_root) / "src").exists() else str(code_root)
    if src not in sys.path:
        sys.path.insert(0, src)
    _install_da3_optional_stubs()
    from depth_anything_3.api import DepthAnything3  # noqa: E402

    return DepthAnything3


class DepthAnything3TargetEncoder(nn.Module):
    def __init__(
        self,
        ckpt_dir: str | Path = _DA3_CKPT,
        *,
        process_res: int = 224,          # 224 -> 16x16=256 tok (grid 16); 504 -> 36x36 (native)
        out_layer_index: int = -1,       # which of the backbone's out_layers to align to (-1 = last)
        feature_slice: str = "full",     # "full" (2048), "global" (normed half, 1024), "local" (1024)
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        code_root: str | Path = _DA3_CODE_ROOT,
    ) -> None:
        super().__init__()
        DepthAnything3 = _import_da3(str(code_root))
        if process_res % 14 != 0:
            raise ValueError(f"DA3 process_res must be a multiple of patch=14 (got {process_res})")
        self.process_res = int(process_res)
        self.out_layer_index = int(out_layer_index)
        self.feature_slice = str(feature_slice)
        self.device = torch.device(device)

        wrapper = DepthAnything3.from_pretrained(str(ckpt_dir))
        self.backbone = wrapper.model.backbone.to(self.device, dtype=dtype).eval()
        self.backbone.requires_grad_(False)
        self.out_layers = tuple(int(i) for i in wrapper.config.net.out_layers)
        # concatenated patch-feature dim (cat_token doubles embed); halved when slicing to one half.
        _cat_dim = int(getattr(wrapper.config.head, "dim_in", 0)) or (2 * int(wrapper.config.net.embed_dim))
        self.feature_dim = _cat_dim if self.feature_slice == "full" else _cat_dim // 2
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def _prep(self, frames_b3hw: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) RGB in [0,255] or [0,1] -> (B,3,R,R) ImageNet-normalized square on the teacher device."""
        x = frames_b3hw.to(self.device).float()
        if x.max() > 1.5:
            x = x / 255.0
        if x.shape[-1] != self.process_res or x.shape[-2] != self.process_res:
            x = F.interpolate(x, size=(self.process_res, self.process_res), mode="bilinear", align_corners=False)
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def _slice(self, feats: torch.Tensor) -> torch.Tensor:
        if self.feature_slice == "full":
            return feats
        half = feats.shape[-1] // 2
        return feats[..., :half] if self.feature_slice == "local" else feats[..., half:]

    def _patch_tokens(self, images_b3hw: torch.Tensor) -> torch.Tensor:
        """(N,3,R,R) normalized -> (N, tok, D) DA3 encoder patch tokens for the selected out layer."""
        x = images_b3hw.to(dtype=next(self.backbone.parameters()).dtype).unsqueeze(1)  # [N,V=1,3,R,R]
        feats_per_layer, _aux = self.backbone(x)
        patch_tokens, _cam = feats_per_layer[self.out_layer_index]  # [N, V=1, tok, D]
        n = patch_tokens.shape[0]
        return self._slice(patch_tokens.reshape(n, -1, patch_tokens.shape[-1]))  # [N, tok, D']

    @torch.no_grad()
    def encode_future_keyframes(self, keyframes_b3hw: Sequence[torch.Tensor]) -> torch.Tensor:
        """keyframes: list of K x (B,3,H,W) future frames. Returns (B, K*tok, D) bf16 (detached).

        Monocular per-frame (no warmup clip), mirroring the lingbot depth teacher."""
        prepped = [self._prep(kf) for kf in keyframes_b3hw]  # each (B,3,R,R)
        b = prepped[0].shape[0]
        k = len(prepped)
        batch = torch.cat(prepped, dim=0)  # (K*B, 3, R, R)
        feats = self._patch_tokens(batch)  # (K*B, tok, D)
        tok, dim = feats.shape[1], feats.shape[2]
        feats = feats.view(k, b, tok, dim).permute(1, 0, 2, 3).reshape(b, k * tok, dim)
        return feats.detach().to(torch.bfloat16)

    @torch.no_grad()
    def encode_current_and_future(self, current_b3hw, keyframe_b3hw):
        """LingBot current-alignment parity: DA3 encoder features for the current frame AND one
        future keyframe. Returns (current [B,tok,D], future [B,tok,D]) bf16 (detached)."""
        cur = self._patch_tokens(self._prep(current_b3hw)).detach().to(torch.bfloat16)   # (B, tok, D)
        fut = self._patch_tokens(self._prep(keyframe_b3hw)).detach().to(torch.bfloat16)  # (B, tok, D)
        return cur, fut
