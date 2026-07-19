"""Four-layer DA3 WSA feature decoder used only for planner visualization.

The planner is supervised with per-token cosine and LayerNorm losses over DA3
layers 11/15/19/23.  This probe applies the same scale-invariant normalization
before decoding both teacher and planner features, so it does not prefer the raw
feature scale of either source.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in range(min(32, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _refinement_block(channels: int) -> nn.Sequential:
    groups = _group_count(channels)
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1),
        nn.GroupNorm(groups, channels),
        nn.GELU(),
        nn.Conv2d(channels, channels, 3, padding=1),
    )


class WSAMultiLayerDPTProbe(nn.Module):
    """Decode four same-resolution DA3 feature layers into dense log depth.

    Input order is fixed by ``teacher_layers`` and has shape ``[B,L,N,D]``.
    The four token maps are reassembled to half, native, 2x, and 4x spatial
    scales before coarse-to-fine fusion.
    """

    NORMALIZATION = "per_token_layer_norm_no_affine"

    def __init__(
        self,
        in_dim: int = 2048,
        feat: int = 256,
        grid: int = 16,
        output_size: int = 224,
        teacher_layers: Sequence[int] = (11, 15, 19, 23),
        out_ch: int = 1,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.feat = int(feat)
        self.grid = int(grid)
        self.output_size = int(output_size)
        self.teacher_layers = tuple(int(layer) for layer in teacher_layers)
        self.out_ch = int(out_ch)
        if self.in_dim <= 0 or self.feat <= 0 or self.out_ch <= 0:
            raise ValueError("in_dim, feat, and out_ch must be positive")
        if self.grid < 2 or self.grid % 2:
            raise ValueError("grid must be an even integer >= 2")
        if self.output_size <= 0:
            raise ValueError("output_size must be positive")
        if len(self.teacher_layers) != 4:
            raise ValueError(
                "WSA depth probe requires exactly 4 teacher layers, got "
                f"{self.teacher_layers}"
            )
        if any(
            left >= right
            for left, right in zip(
                self.teacher_layers,
                self.teacher_layers[1:],
            )
        ):
            raise ValueError(
                "teacher_layers must be strictly increasing, got "
                f"{self.teacher_layers}"
            )

        self.projections = nn.ModuleList(
            [nn.Conv2d(self.in_dim, self.feat, 1) for _ in self.teacher_layers]
        )
        self.resamplers = nn.ModuleList(
            [
                nn.Conv2d(self.feat, self.feat, 3, stride=2, padding=1),
                nn.Identity(),
                nn.ConvTranspose2d(self.feat, self.feat, 2, stride=2),
                nn.Sequential(
                    nn.ConvTranspose2d(self.feat, self.feat, 2, stride=2),
                    nn.GELU(),
                    nn.ConvTranspose2d(self.feat, self.feat, 2, stride=2),
                ),
            ]
        )
        self.refinements = nn.ModuleList(
            [_refinement_block(self.feat) for _ in self.teacher_layers]
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(self.feat, max(self.feat // 2, 1), 3, padding=1),
            nn.GELU(),
            nn.Conv2d(max(self.feat // 2, 1), self.out_ch, 1),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "WSAMultiLayerDPTProbe":
        config = dict(config)
        normalization = config.pop("normalization", cls.NORMALIZATION)
        if normalization != cls.NORMALIZATION:
            raise ValueError(
                "unsupported WSA probe normalization: "
                f"{normalization!r}"
            )
        return cls(**config)

    def config(self) -> dict[str, Any]:
        return {
            "in_dim": self.in_dim,
            "feat": self.feat,
            "grid": self.grid,
            "output_size": self.output_size,
            "teacher_layers": list(self.teacher_layers),
            "out_ch": self.out_ch,
            "normalization": self.NORMALIZATION,
        }

    def _validate(self, tokens: torch.Tensor) -> None:
        if tokens.ndim != 4:
            raise ValueError(
                "WSA depth tokens must be [B,L,N,D], got "
                f"{tuple(tokens.shape)}"
            )
        if tokens.shape[1] != len(self.teacher_layers):
            raise ValueError(
                f"WSA depth probe expects 4 layers, got {tokens.shape[1]}"
            )
        expected_tokens = self.grid * self.grid
        if tokens.shape[2] != expected_tokens:
            raise ValueError(
                f"WSA depth probe expects {expected_tokens} tokens, "
                f"got {tokens.shape[2]}"
            )
        if tokens.shape[3] != self.in_dim:
            raise ValueError(
                f"WSA depth probe expects feature width {self.in_dim}, "
                f"got {tokens.shape[3]}"
            )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        self._validate(tokens)
        tokens = F.layer_norm(tokens.float(), (self.in_dim,))
        batch = tokens.shape[0]
        maps = []
        for layer_index, (projection, resampler) in enumerate(
            zip(self.projections, self.resamplers, strict=True)
        ):
            feature_map = tokens[:, layer_index].transpose(1, 2).reshape(
                batch,
                self.in_dim,
                self.grid,
                self.grid,
            )
            maps.append(resampler(projection(feature_map)))

        fused = self.refinements[0](maps[0])
        for feature_map, refinement in zip(
            maps[1:],
            self.refinements[1:],
            strict=True,
        ):
            fused = F.interpolate(
                fused,
                size=feature_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = refinement(feature_map) + fused
        fused = F.interpolate(
            fused,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        return self.depth_head(fused)


__all__ = ["WSAMultiLayerDPTProbe"]
