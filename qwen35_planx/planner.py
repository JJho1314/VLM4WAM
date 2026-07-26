"""Trainable grounded planner heads on frame-local Qwen3.5 causal states."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from qwen35_planx.losses import (
    _chunked_visual_objective,
    counterfactual_loss,
    dense_feature_loss,
    grounding_loss,
    temporal_loss,
)
from qwen35_planx.planner_dataset import GroundedPlannerBatch


_NUM_KEYFRAMES = 4
_NUM_ROLES = 3
_TOKENS_PER_FRAME = 729
_VISUAL_VOCAB_SIZE = 65_536
_CODEBOOK_DIM = 1_536
_QWEN_HIDDEN_DIM = 2_048
_TEXT_DIM = 1_152
_MAX_CODE_CHUNK_SIZE = 64
_NORMALIZED_TARGET_TIMES = (0.0, 3.0 / 8.0, 5.0 / 8.0, 1.0)


def gather_positions(hidden: Tensor, positions: Tensor) -> Tensor:
    """Gather sequence states at independently aligned batch positions."""

    if hidden.ndim != 3:
        raise ValueError("hidden must have shape [B,L,H]")
    if positions.ndim != 2 or positions.shape[0] != hidden.shape[0]:
        raise ValueError("positions must have shape [B,N]")
    if positions.dtype == torch.bool or positions.dtype.is_floating_point:
        raise TypeError("positions must contain integer sequence indices")
    index = positions.to(device=hidden.device, dtype=torch.long)
    return torch.gather(
        hidden,
        dim=1,
        index=index.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]),
    )


@dataclass(frozen=True)
class GroundedPlannerOutput:
    """Flattened-camera planner predictions and five training objectives."""

    codes: Tensor
    code_embeddings: Tensor
    post_hidden: Tensor
    visual_regression: Tensor
    semantic_features: Tensor
    predicted_phrase_embeddings: Tensor
    relevance_logits: Tensor
    relevance: Tensor
    fusion_gate: Tensor
    times: Tensor
    code_loss: Tensor
    dense_feature_loss: Tensor
    grounding_loss: Tensor
    counterfactual_loss: Tensor
    temporal_loss: Tensor
    total_loss: Tensor
    debug_pre_positions: Tensor
    debug_post_positions: Tensor

    @property
    def loss(self) -> Tensor:
        """Trainer-compatible alias for the weighted planner objective."""

        return self.total_loss

    def unflatten_cameras(self, batch_size: int) -> GroundedPlannerOutput:
        """Split every leading ``B*2`` camera tensor into ``[B,2,...]``."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        camera_batch = batch_size * 2
        per_camera_names = (
            "codes",
            "code_embeddings",
            "post_hidden",
            "visual_regression",
            "semantic_features",
            "predicted_phrase_embeddings",
            "relevance_logits",
            "relevance",
            "fusion_gate",
            "debug_pre_positions",
            "debug_post_positions",
        )
        updates: dict[str, Tensor] = {}
        for name in per_camera_names:
            value = getattr(self, name)
            if value.shape[0] != camera_batch:
                raise ValueError(
                    f"{name} leading dimension must equal batch_size*2"
                )
            updates[name] = value.reshape(batch_size, 2, *value.shape[1:])
        return replace(self, **updates)


class GroundedQwen35Planner(nn.Module):
    """Qwen3.5 backbone with visual-only code prediction and grounding heads."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        visual_embedding_weight: Tensor,
        codebook: Tensor,
        hidden_dim: int,
        text_dim: int,
        code_chunk_size: int = _MAX_CODE_CHUNK_SIZE,
        _enforce_released_geometry: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch module")
        if hidden_dim <= 0 or text_dim <= 0:
            raise ValueError("hidden_dim and text_dim must be positive")
        if (
            not isinstance(visual_embedding_weight, Tensor)
            or visual_embedding_weight.ndim != 2
            or visual_embedding_weight.shape[1] != hidden_dim
        ):
            raise ValueError(
                "visual_embedding_weight must have shape [visual_vocab, hidden_dim]"
            )
        if (
            not isinstance(codebook, Tensor)
            or codebook.ndim != 2
            or codebook.shape[0] != visual_embedding_weight.shape[0]
        ):
            raise ValueError(
                "codebook must have shape [visual_vocab, codebook_dim]"
            )
        if codebook.shape[1] <= 0:
            raise ValueError("codebook width must be positive")
        if not 1 <= code_chunk_size <= _MAX_CODE_CHUNK_SIZE:
            raise ValueError("code_chunk_size must be in [1,64]")
        released_geometry = (
            visual_embedding_weight.shape[0],
            codebook.shape[1],
            hidden_dim,
            text_dim,
        )
        expected_geometry = (
            _VISUAL_VOCAB_SIZE,
            _CODEBOOK_DIM,
            _QWEN_HIDDEN_DIM,
            _TEXT_DIM,
        )
        if _enforce_released_geometry and released_geometry != expected_geometry:
            raise ValueError(
                "released planner geometry requires visual_vocab=65536, "
                "codebook_dim=1536, hidden_dim=2048, and text_dim=1152"
            )

        self.backbone = backbone
        self.visual_embedding_weight = visual_embedding_weight
        self.register_buffer("codebook", codebook.detach(), persistent=True)
        self.hidden_dim = hidden_dim
        self.text_dim = text_dim
        self.code_chunk_size = code_chunk_size
        self._enforce_released_geometry = _enforce_released_geometry

        code_dim = int(codebook.shape[1])
        self.visual_regression = nn.Linear(hidden_dim, code_dim)
        self.semantic_projection = nn.Linear(hidden_dim, text_dim)
        self.phrase_projection = nn.Linear(hidden_dim, text_dim)
        self.grounding_query = nn.Linear(hidden_dim, text_dim, bias=False)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim + _NUM_ROLES, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.register_buffer(
            "target_times",
            torch.tensor(_NORMALIZED_TARGET_TIMES),
            persistent=True,
        )

    @classmethod
    def from_components(
        cls,
        *,
        backbone: nn.Module,
        visual_embedding_weight: Tensor,
        codebook: Tensor,
        hidden_dim: int = 2048,
        text_dim: int = 1152,
        code_chunk_size: int = _MAX_CODE_CHUNK_SIZE,
    ) -> GroundedQwen35Planner:
        """Construct from an already loaded Qwen backbone and released codebook."""

        return cls(
            backbone=backbone,
            visual_embedding_weight=visual_embedding_weight,
            codebook=codebook,
            hidden_dim=hidden_dim,
            text_dim=text_dim,
            code_chunk_size=code_chunk_size,
        )

    @classmethod
    def _from_test_components(
        cls,
        *,
        backbone: nn.Module,
        visual_embedding_weight: Tensor,
        codebook: Tensor,
        hidden_dim: int,
        text_dim: int,
        code_chunk_size: int = _MAX_CODE_CHUNK_SIZE,
    ) -> GroundedQwen35Planner:
        """Build a reduced-geometry unit-test planner, never a checkpoint model."""

        return cls(
            backbone=backbone,
            visual_embedding_weight=visual_embedding_weight,
            codebook=codebook,
            hidden_dim=hidden_dim,
            text_dim=text_dim,
            code_chunk_size=code_chunk_size,
            _enforce_released_geometry=False,
        )

    @staticmethod
    def _last_hidden(output: Any) -> Tensor:
        if isinstance(output, Mapping):
            hidden = output.get("last_hidden_state")
        else:
            hidden = getattr(output, "last_hidden_state", None)
        if not isinstance(hidden, Tensor):
            raise TypeError("Qwen backbone output must expose last_hidden_state")
        return hidden

    def _language_backbone(self) -> nn.Module:
        base_model = getattr(self.backbone, "model", None)
        return base_model if isinstance(base_model, nn.Module) else self.backbone

    def _geometry(self, batch: GroundedPlannerBatch) -> tuple[int, int]:
        if batch.relevance_targets.ndim != 4:
            raise ValueError("relevance targets must have shape [B,4,3,N]")
        camera_batch, frames, roles, tokens = batch.relevance_targets.shape
        if frames != _NUM_KEYFRAMES or roles != _NUM_ROLES or tokens <= 0:
            raise ValueError("relevance targets must have shape [B,4,3,N]")
        if self._enforce_released_geometry and tokens != _TOKENS_PER_FRAME:
            raise ValueError(
                "released planner geometry requires 729 patches per frame"
            )
        if batch.code_targets.shape != (camera_batch, frames * tokens):
            raise ValueError("code targets must align with four future grids")
        if batch.pre_positions.shape != batch.code_targets.shape:
            raise ValueError("pre_positions must align with code targets")
        if batch.post_positions.shape != batch.code_targets.shape:
            raise ValueError("post_positions must align with code targets")
        if batch.field_positions.shape != (camera_batch, _NUM_ROLES):
            raise ValueError("field_positions must have shape [B,3]")
        return camera_batch, tokens

    def forward(self, batch: GroundedPlannerBatch) -> GroundedPlannerOutput:
        if not isinstance(batch, GroundedPlannerBatch):
            raise TypeError("batch must be a GroundedPlannerBatch")
        camera_batch, tokens = self._geometry(batch)
        reserved = {"output_hidden_states", "return_dict"}.intersection(
            batch.qwen_inputs
        )
        if reserved:
            raise ValueError(
                "qwen_inputs must not override planner backbone output options"
            )
        backbone_output = self._language_backbone()(
            **batch.qwen_inputs,
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden = self._last_hidden(backbone_output)
        if last_hidden.shape[0] != camera_batch:
            raise ValueError("Qwen hidden batch must align with planner targets")
        if last_hidden.shape[-1] != self.hidden_dim:
            raise ValueError("Qwen hidden width differs from hidden_dim")

        h_pre = gather_positions(last_hidden, batch.pre_positions)
        h_post_flat = gather_positions(last_hidden, batch.post_positions)
        h_fields = gather_positions(last_hidden, batch.field_positions)

        code_loss_value, flat_codes = _chunked_visual_objective(
            h_pre,
            self.visual_embedding_weight,
            batch.code_targets,
            chunk_size=self.code_chunk_size,
        )
        codes = flat_codes.reshape(camera_batch, _NUM_KEYFRAMES, tokens)
        code_embeddings = F.embedding(codes, self.codebook)
        target_code_embeddings = F.embedding(
            batch.code_targets.to(device=self.codebook.device, dtype=torch.long),
            self.codebook,
        )

        visual_prediction = F.normalize(
            self.visual_regression(h_pre),
            dim=-1,
            eps=1e-12,
        )
        post_hidden = h_post_flat.reshape(
            camera_batch,
            _NUM_KEYFRAMES,
            tokens,
            self.hidden_dim,
        )
        semantic_features = F.normalize(
            self.semantic_projection(h_post_flat),
            dim=-1,
            eps=1e-12,
        ).reshape(
            camera_batch,
            _NUM_KEYFRAMES,
            tokens,
            self.text_dim,
        )
        predicted_phrases = F.normalize(
            self.phrase_projection(h_fields),
            dim=-1,
            eps=1e-12,
        )
        grounding_queries = F.normalize(
            self.grounding_query(post_hidden),
            dim=-1,
            eps=1e-12,
        )
        relevance_logits = torch.einsum(
            "bktd,brd->bkrt",
            grounding_queries,
            predicted_phrases,
        )
        relevance = relevance_logits.softmax(dim=-1)
        token_logits = relevance_logits.permute(0, 1, 3, 2)
        fusion_gate = self.fusion_gate(
            torch.cat((post_hidden, token_logits), dim=-1)
        )

        dense_value = dense_feature_loss(
            visual_regression=visual_prediction,
            target_code_embeddings=target_code_embeddings.to(visual_prediction),
            semantic_features=semantic_features,
            relevance_targets=batch.relevance_targets.to(semantic_features),
            relevance_confidence=batch.relevance_confidence.to(semantic_features),
            predicted_phrase_embeddings=predicted_phrases,
            phrase_embeddings=batch.phrase_embeddings.to(predicted_phrases),
            field_mask=batch.field_mask.to(device=predicted_phrases.device),
        )
        grounding_value = grounding_loss(
            relevance_logits,
            batch.relevance_targets.to(relevance_logits),
            batch.relevance_confidence.to(relevance_logits),
        )
        counterfactual_value = counterfactual_loss(
            semantic_features,
            relevance,
            batch.phrase_embeddings.to(semantic_features),
            batch.counterfactual_embeddings.to(semantic_features),
            batch.counterfactual_mask.to(device=semantic_features.device),
            fusion_gate=fusion_gate,
        )
        temporal_value = temporal_loss(
            relevance,
            batch.flow_targets.to(relevance),
        )
        total = (
            code_loss_value
            + 0.5 * dense_value
            + 0.5 * grounding_value
            + 0.2 * counterfactual_value
            + 0.1 * temporal_value
        )

        return GroundedPlannerOutput(
            codes=codes,
            code_embeddings=code_embeddings,
            post_hidden=post_hidden,
            visual_regression=visual_prediction,
            semantic_features=semantic_features,
            predicted_phrase_embeddings=predicted_phrases,
            relevance_logits=relevance_logits,
            relevance=relevance,
            fusion_gate=fusion_gate,
            times=self.target_times.to(post_hidden),
            code_loss=code_loss_value,
            dense_feature_loss=dense_value,
            grounding_loss=grounding_value,
            counterfactual_loss=counterfactual_value,
            temporal_loss=temporal_value,
            total_loss=total,
            debug_pre_positions=batch.pre_positions,
            debug_post_positions=batch.post_positions,
        )
