"""Block-causal spatiotemporal queries over gathered Qwen plan states."""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen35_baton.config import BatonGeometry


def _positive_integer(name: str, value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_block_causal_allowed_mask(
    num_frames: int,
    tokens_per_frame: int,
) -> torch.Tensor:
    """Return a public boolean mask where ``True`` means attention is allowed."""

    _positive_integer("num_frames", num_frames)
    _positive_integer("tokens_per_frame", tokens_per_frame)
    frame_ids = torch.arange(num_frames).repeat_interleave(tokens_per_frame)
    return frame_ids[:, None] >= frame_ids[None, :]


def build_spatiotemporal_positions(
    num_frames: int,
    tokens_per_frame: int,
) -> torch.Tensor:
    """Build row-major ``(frame, y, x)`` patch coordinates."""

    _positive_integer("num_frames", num_frames)
    _positive_integer("tokens_per_frame", tokens_per_frame)
    grid_size = isqrt(tokens_per_frame)
    if grid_size * grid_size != tokens_per_frame:
        raise ValueError("tokens_per_frame must form a square spatial grid")
    patch_ids = torch.arange(tokens_per_frame)
    spatial = torch.stack(
        (
            torch.div(patch_ids, grid_size, rounding_mode="floor"),
            patch_ids.remainder(grid_size),
        ),
        dim=-1,
    )
    frames = torch.arange(num_frames)[:, None, None].expand(
        num_frames, tokens_per_frame, 1
    )
    spatial = spatial[None].expand(num_frames, tokens_per_frame, 2)
    return torch.cat((frames, spatial), dim=-1).reshape(-1, 3).to(torch.float32)


def _test_rotary_dimensions(head_dim: int) -> tuple[int, int, int]:
    """Allocate an even rotary width across axes for reduced unit-test towers."""

    pairs, remainder = divmod(head_dim // 2, 3)
    pair_counts = [pairs, pairs, pairs]
    for index in range(remainder):
        pair_counts[index] += 1
    return (
        2 * pair_counts[0],
        2 * pair_counts[1],
        2 * pair_counts[2],
    )


def _rotate_pairs(values: torch.Tensor) -> torch.Tensor:
    paired = values.reshape(*values.shape[:-1], -1, 2)
    return torch.stack((-paired[..., 1], paired[..., 0]), dim=-1).flatten(-2)


class RotaryPosition3D(nn.Module):
    """Apply independent rotary bands for temporal, row, and column positions."""

    def __init__(
        self,
        head_dim: int,
        rotary_dimensions: tuple[int, int, int],
        *,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if (
            len(rotary_dimensions) != 3
            or any(
                type(width) is not int or width < 0 or width % 2
                for width in rotary_dimensions
            )
            or sum(rotary_dimensions) > head_dim
        ):
            raise ValueError(
                "rotary dimensions must be three non-negative even widths "
                "whose sum does not exceed head_dim"
            )
        self.head_dim = head_dim
        self.rotary_dimensions = rotary_dimensions
        for axis, width in enumerate(rotary_dimensions):
            inverse_frequency = 1.0 / (
                base ** (torch.arange(0, width, 2, dtype=torch.float32) / max(width, 1))
            )
            self.register_buffer(
                f"inverse_frequency_{axis}",
                inverse_frequency,
                persistent=False,
            )

    def forward(
        self,
        values: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if values.ndim != 4 or values.shape[-1] != self.head_dim:
            raise ValueError("rotary input must be [batch,heads,tokens,head_dim]")
        if positions.shape != (values.shape[-2], 3):
            raise ValueError("positions must be [tokens,3]")

        rotated_chunks: list[torch.Tensor] = []
        start = 0
        for axis, width in enumerate(self.rotary_dimensions):
            if width == 0:
                continue
            stop = start + width
            chunk = values[..., start:stop]
            inverse_frequency = getattr(self, f"inverse_frequency_{axis}")
            angles = (
                positions[:, axis].to(torch.float32)[:, None]
                * inverse_frequency[None]
            )
            cosines = angles.cos().repeat_interleave(2, dim=-1).to(chunk.dtype)
            sines = angles.sin().repeat_interleave(2, dim=-1).to(chunk.dtype)
            rotated_chunks.append(
                chunk * cosines[None, None]
                + _rotate_pairs(chunk) * sines[None, None]
            )
            start = stop
        rotated_chunks.append(values[..., start:])
        return torch.cat(rotated_chunks, dim=-1)


def _additive_attention_bias(
    allowed_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if allowed_mask.ndim != 2 or allowed_mask.dtype != torch.bool:
        raise TypeError("allowed_mask must be a two-dimensional boolean tensor")
    bias = torch.zeros(allowed_mask.shape, dtype=dtype, device=device)
    return bias.masked_fill(~allowed_mask.to(device=device), float("-inf"))


class BlockCausalAttention(nn.Module):
    """Explicit projected multi-head attention with 3D rotary positions."""

    def __init__(
        self,
        query_dim: int,
        *,
        num_heads: int,
        dropout: float,
        rotary_dimensions: tuple[int, int, int],
    ) -> None:
        super().__init__()
        if query_dim % num_heads:
            raise ValueError("query_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(query_dim, query_dim)
        self.v_proj = nn.Linear(query_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.rotary = RotaryPosition3D(self.head_dim, rotary_dimensions)

    def _heads(self, states: torch.Tensor) -> torch.Tensor:
        batch_size, tokens, width = states.shape
        return states.reshape(
            batch_size, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        query_positions: torch.Tensor,
        allowed_mask: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        context_positions: torch.Tensor | None = None,
        return_attention_map: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if context is None:
            context = query
        if context_positions is None:
            context_positions = query_positions
        if allowed_mask.shape != (query.shape[1], context.shape[1]):
            raise ValueError(
                "allowed_mask shape must match query and context token counts"
            )

        query_heads = self.rotary(self._heads(self.q_proj(query)), query_positions)
        key_heads = self.rotary(self._heads(self.k_proj(context)), context_positions)
        value_heads = self._heads(self.v_proj(context))
        additive_bias = _additive_attention_bias(
            allowed_mask,
            dtype=query_heads.dtype,
            device=query_heads.device,
        )
        attended = F.scaled_dot_product_attention(
            query_heads,
            key_heads,
            value_heads,
            attn_mask=additive_bias[None, None],
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(
            query.shape[0], query.shape[1], -1
        )
        output = self.out_proj(attended)
        if not return_attention_map:
            return output

        with torch.no_grad():
            scale = self.head_dim**-0.5
            scores = torch.matmul(query_heads, key_heads.transpose(-2, -1)) * scale
            probabilities = torch.softmax(
                scores + additive_bias[None, None],
                dim=-1,
            )
            head_mean = probabilities.mean(dim=1)
        return output, head_mean


class _PreNormQueryBlock(nn.Module):
    def __init__(
        self,
        query_dim: int,
        *,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        rotary_dimensions: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(query_dim)
        self.self_attention = BlockCausalAttention(
            query_dim,
            num_heads=num_heads,
            dropout=dropout,
            rotary_dimensions=rotary_dimensions,
        )
        self.cross_norm = nn.LayerNorm(query_dim)
        self.cross_attention = BlockCausalAttention(
            query_dim,
            num_heads=num_heads,
            dropout=dropout,
            rotary_dimensions=rotary_dimensions,
        )
        self.ffn_norm = nn.LayerNorm(query_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(query_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        positions: torch.Tensor,
        allowed_mask: torch.Tensor,
        *,
        return_attention_map: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self_output = self.self_attention(
            self.self_norm(x),
            positions,
            allowed_mask,
        )
        assert isinstance(self_output, torch.Tensor)
        x = x + self_output

        cross_output = self.cross_attention(
            self.cross_norm(x),
            positions,
            allowed_mask,
            context=context,
            context_positions=positions,
            return_attention_map=return_attention_map,
        )
        attention_map = None
        if return_attention_map:
            assert isinstance(cross_output, tuple)
            cross_output, attention_map = cross_output
        assert isinstance(cross_output, torch.Tensor)
        x = x + cross_output
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, attention_map


@dataclass(frozen=True)
class QueryTowerOutput:
    hidden_states: torch.Tensor
    cross_attention_maps: tuple[torch.Tensor, ...] | None


class SpatiotemporalQueryTower(nn.Module):
    """Four-block production Query Tower with a reduced test-only constructor."""

    def __init__(self, qwen_dim: int) -> None:
        super().__init__()
        geometry = BatonGeometry()
        self._initialize(
            qwen_dim=qwen_dim,
            query_dim=geometry.query_dim,
            num_frames=len(geometry.future_indices),
            tokens_per_frame=geometry.tokens_per_frame,
            num_heads=geometry.query_heads,
            ffn_dim=geometry.query_ffn_dim,
            dropout=geometry.query_dropout,
            num_layers=geometry.query_layers,
            num_cameras=len(geometry.camera_names),
            rotary_dimensions=(16, 24, 24),
            test_config=False,
            geometry=geometry,
        )

    @classmethod
    def _from_test_config(
        cls,
        *,
        qwen_dim: int,
        query_dim: int,
        num_frames: int,
        tokens_per_frame: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> SpatiotemporalQueryTower:
        """Construct a reduced tower that is forbidden from checkpoint export."""

        tower = cls.__new__(cls)
        nn.Module.__init__(tower)
        _positive_integer("query_dim", query_dim)
        _positive_integer("num_heads", num_heads)
        if query_dim % num_heads:
            raise ValueError("query_dim must be divisible by num_heads")
        head_dim = query_dim // num_heads
        tower._initialize(
            qwen_dim=qwen_dim,
            query_dim=query_dim,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            num_layers=4,
            num_cameras=2,
            rotary_dimensions=_test_rotary_dimensions(head_dim),
            test_config=True,
            geometry=None,
        )
        return tower

    def _initialize(
        self,
        *,
        qwen_dim: int,
        query_dim: int,
        num_frames: int,
        tokens_per_frame: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        num_layers: int,
        num_cameras: int,
        rotary_dimensions: tuple[int, int, int],
        test_config: bool,
        geometry: BatonGeometry | None,
    ) -> None:
        for name, value in (
            ("qwen_dim", qwen_dim),
            ("query_dim", query_dim),
            ("num_frames", num_frames),
            ("tokens_per_frame", tokens_per_frame),
            ("num_heads", num_heads),
            ("ffn_dim", ffn_dim),
            ("num_layers", num_layers),
            ("num_cameras", num_cameras),
        ):
            _positive_integer(name, value)
        if query_dim % num_heads:
            raise ValueError("query_dim must be divisible by num_heads")
        if not isinstance(dropout, (int, float)) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        grid_size = isqrt(tokens_per_frame)
        if grid_size * grid_size != tokens_per_frame:
            raise ValueError("tokens_per_frame must form a square spatial grid")

        self.qwen_dim = qwen_dim
        self.query_dim = query_dim
        self.num_frames = num_frames
        self.tokens_per_frame = tokens_per_frame
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.num_cameras = num_cameras
        self.rotary_dimensions = rotary_dimensions
        self.geometry = geometry
        self._test_config = test_config

        self.learned_queries = nn.Parameter(
            torch.empty(num_frames, tokens_per_frame, query_dim)
        )
        self.frame_embeddings = nn.Embedding(num_frames, query_dim)
        self.y_embeddings = nn.Embedding(grid_size, query_dim)
        self.x_embeddings = nn.Embedding(grid_size, query_dim)
        self.camera_embeddings = nn.Embedding(num_cameras, query_dim)
        self.context_projection = nn.Linear(qwen_dim, query_dim)
        self.blocks = nn.ModuleList(
            _PreNormQueryBlock(
                query_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=float(dropout),
                rotary_dimensions=rotary_dimensions,
            )
            for _ in range(num_layers)
        )

        positions = build_spatiotemporal_positions(num_frames, tokens_per_frame)
        self.register_buffer("positions", positions, persistent=False)
        self.register_buffer(
            "allowed_mask",
            build_block_causal_allowed_mask(num_frames, tokens_per_frame),
            persistent=False,
        )
        nn.init.normal_(self.learned_queries, std=0.02)

    def _validate_inputs(
        self,
        qwen_states: torch.Tensor,
        camera_ids: torch.Tensor,
    ) -> torch.Tensor:
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
        if (
            not isinstance(camera_ids, torch.Tensor)
            or camera_ids.ndim != 1
            or camera_ids.shape[0] != qwen_states.shape[0]
            or camera_ids.dtype
            not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        ):
            raise ValueError(
                "camera_ids must be a one-dimensional integer tensor per row"
            )
        if bool(((camera_ids < 0) | (camera_ids >= self.num_cameras)).any()):
            raise ValueError(f"camera_ids must be in [0,{self.num_cameras})")
        return camera_ids.to(device=qwen_states.device, dtype=torch.long)

    def _initial_queries(self, camera_ids: torch.Tensor) -> torch.Tensor:
        positions = self.positions.to(device=self.learned_queries.device)
        frame_ids = positions[:, 0].to(torch.long)
        y_ids = positions[:, 1].to(torch.long)
        x_ids = positions[:, 2].to(torch.long)
        base = (
            self.learned_queries.reshape(-1, self.query_dim)
            + self.frame_embeddings(frame_ids)
            + self.y_embeddings(y_ids)
            + self.x_embeddings(x_ids)
        )
        return base[None] + self.camera_embeddings(camera_ids)[:, None]

    def forward(
        self,
        qwen_states: torch.Tensor,
        camera_ids: torch.Tensor,
        *,
        return_attention_maps: bool = False,
    ) -> QueryTowerOutput:
        camera_ids = self._validate_inputs(qwen_states, camera_ids)
        rows = qwen_states.shape[0]
        context = self.context_projection(
            qwen_states.reshape(rows, -1, self.qwen_dim)
        )
        x = self._initial_queries(camera_ids)
        attention_maps: list[torch.Tensor] | None = (
            [] if return_attention_maps else None
        )
        for block in self.blocks:
            x, attention_map = block(
                x,
                context,
                self.positions,
                self.allowed_mask,
                return_attention_map=return_attention_maps,
            )
            if attention_maps is not None:
                assert attention_map is not None
                attention_maps.append(attention_map)
        return QueryTowerOutput(
            hidden_states=x.reshape(
                rows,
                self.num_frames,
                self.tokens_per_frame,
                self.query_dim,
            ),
            cross_attention_maps=(
                tuple(attention_maps) if attention_maps is not None else None
            ),
        )

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        if self._test_config:
            raise RuntimeError(
                "test-config Query Towers cannot be serialized as production checkpoints"
            )
        return super().state_dict(*args, **kwargs)
