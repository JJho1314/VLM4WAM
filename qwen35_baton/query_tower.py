"""Baton visual alignment tower over gathered Qwen plan states."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from qwen35_baton.config import BatonGeometry


def _positive_integer(name: str, value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class QueryTowerOutput:
    hidden_states: torch.Tensor
    cross_attention_maps: tuple[torch.Tensor, ...] | None


class BatonVisualAlignmentTower(nn.Module):
    """One learned query per target patch, cross-attending all plan states."""

    def __init__(self, qwen_dim: int) -> None:
        super().__init__()
        geometry = BatonGeometry()
        self._initialize(
            qwen_dim=qwen_dim,
            num_frames=len(geometry.future_indices),
            tokens_per_frame=geometry.tokens_per_frame,
            num_heads=geometry.query_heads,
        )

    @classmethod
    def _from_test_config(
        cls,
        *,
        qwen_dim: int,
        num_frames: int,
        tokens_per_frame: int,
        num_heads: int,
    ) -> BatonVisualAlignmentTower:
        tower = cls.__new__(cls)
        nn.Module.__init__(tower)
        tower._initialize(
            qwen_dim=qwen_dim,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            num_heads=num_heads,
        )
        return tower

    def _initialize(
        self,
        *,
        qwen_dim: int,
        num_frames: int,
        tokens_per_frame: int,
        num_heads: int,
    ) -> None:
        for name, value in (
            ("qwen_dim", qwen_dim),
            ("num_frames", num_frames),
            ("tokens_per_frame", tokens_per_frame),
            ("num_heads", num_heads),
        ):
            _positive_integer(name, value)
        if qwen_dim % num_heads:
            raise ValueError("qwen_dim must be divisible by num_heads")

        self.qwen_dim = qwen_dim
        self.num_frames = num_frames
        self.tokens_per_frame = tokens_per_frame
        self.num_heads = num_heads
        self.learned_queries = nn.Parameter(
            torch.empty(num_frames, tokens_per_frame, qwen_dim)
        )
        self.cross_attention = nn.MultiheadAttention(
            qwen_dim,
            num_heads,
            batch_first=True,
        )
        nn.init.normal_(self.learned_queries, std=0.02)

    def forward(
        self,
        qwen_states: torch.Tensor,
        *,
        return_attention_maps: bool = False,
    ) -> QueryTowerOutput:
        expected_tail = (
            self.num_frames,
            self.tokens_per_frame,
            self.qwen_dim,
        )
        if (
            not isinstance(qwen_states, torch.Tensor)
            or qwen_states.ndim != 4
            or tuple(qwen_states.shape[1:]) != expected_tail
        ):
            raise ValueError(
                "qwen_states must be "
                f"[rows,{self.num_frames},{self.tokens_per_frame},{self.qwen_dim}]"
            )

        rows = qwen_states.shape[0]
        context = qwen_states.reshape(rows, -1, self.qwen_dim)
        queries = self.learned_queries.reshape(-1, self.qwen_dim)
        queries = queries.unsqueeze(0).expand(rows, -1, -1)
        aligned, attention = self.cross_attention(
            queries,
            context,
            context,
            need_weights=return_attention_maps,
            average_attn_weights=True,
        )
        return QueryTowerOutput(
            hidden_states=aligned.reshape(
                rows,
                self.num_frames,
                self.tokens_per_frame,
                self.qwen_dim,
            ),
            cross_attention_maps=(
                (attention.detach(),) if return_attention_maps else None
            ),
        )
