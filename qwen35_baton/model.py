"""Qwen3.5 hidden-state extraction and continuous SigLIP-grid prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from qwen35_baton.data import BatonPlannerBatch
from qwen35_baton.query_tower import BatonVisualAlignmentTower, QueryTowerOutput
from qwen35_baton.sequence import ADDED_TOKENS, PLAN_PAD


_NUM_FRAMES = 4
_TOKENS_PER_FRAME = 256
_PLAN_TOKENS = _NUM_FRAMES * _TOKENS_PER_FRAME
_FEATURE_DIM = 1024
_PLAN_PAD_ADAPTER_INDEX = ADDED_TOKENS.index(PLAN_PAD)


class PlanTokenEmbeddingAdapter(nn.Module):
    """Overlay seven separately-owned plan rows on the resized Qwen embedding.

    Constructing the adapter freezes the base embedding so that standalone use
    (provider inference, unit tests) never touches Qwen rows. Stage-1 training
    deliberately overrides this: ``qwen35_baton.ownership`` re-enables gradients
    for the *full* planner, base rows included. The attribute keeps the name
    ``frozen_base_embedding`` because it is part of the on-disk
    ``planner.safetensors`` key contract, not because it stays frozen in training.
    """

    def __init__(
        self,
        frozen_base_embedding: nn.Module,
        added_token_ids: tuple[int, ...],
    ) -> None:
        super().__init__()
        weight = getattr(frozen_base_embedding, "weight", None)
        if (
            not isinstance(frozen_base_embedding, nn.Module)
            or not isinstance(weight, torch.Tensor)
            or weight.ndim != 2
        ):
            raise TypeError(
                "frozen_base_embedding must be a token embedding with rank-2 weight"
            )
        if (
            not isinstance(added_token_ids, tuple)
            or len(added_token_ids) != len(ADDED_TOKENS)
            or any(type(token_id) is not int for token_id in added_token_ids)
            or len(set(added_token_ids)) != len(added_token_ids)
            or min(added_token_ids) < 0
            or max(added_token_ids) >= weight.shape[0]
        ):
            raise ValueError(
                "added_token_ids must contain seven unique in-vocabulary integers"
            )

        frozen_base_embedding.requires_grad_(False)
        # Keep the base as a non-owning reference. BatonQwen35Planner registers it
        # separately, and Stage-1 ownership later re-enables its gradients; this
        # module owns only the seven plan rows.
        object.__setattr__(self, "_frozen_base_embedding", frozen_base_embedding)
        self.register_buffer(
            "added_token_ids",
            torch.tensor(added_token_ids, dtype=torch.long),
        )
        initial_rows = weight.detach()[list(added_token_ids)].clone()
        self.plan_embeddings = nn.Embedding.from_pretrained(
            initial_rows,
            freeze=False,
        )
        self.embedding_dim = int(weight.shape[1])
        self.num_embeddings = int(weight.shape[0])

    @property
    def frozen_base_embedding(self) -> nn.Module:
        return object.__getattribute__(self, "_frozen_base_embedding")

    @property
    def weight(self) -> torch.Tensor:
        """Expose the full base matrix for Qwen weight-tying introspection."""

        weight = getattr(self.frozen_base_embedding, "weight")
        assert isinstance(weight, torch.Tensor)
        return weight

    @property
    def plan_pad_token_id(self) -> int:
        return int(self.added_token_ids[_PLAN_PAD_ADAPTER_INDEX].item())

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.dtype == torch.bool
            or input_ids.dtype.is_floating_point
        ):
            raise TypeError("input_ids must be an integer tensor")
        base = self.frozen_base_embedding(input_ids)
        added_ids = self.added_token_ids.to(device=input_ids.device)
        matches = input_ids.unsqueeze(-1).eq(added_ids)
        replacement_rows = matches.to(torch.long).argmax(dim=-1)
        replacements = self.plan_embeddings(replacement_rows)
        return torch.where(matches.any(dim=-1, keepdim=True), replacements, base)


@dataclass(frozen=True)
class BatonPlannerOutput:
    """Continuous predictions in flat-row and dual-camera layouts."""

    flat: torch.Tensor
    positive: torch.Tensor
    cross_attention_maps: tuple[torch.Tensor, ...] | None


def _multimodal_base_model(backbone: nn.Module) -> nn.Module:
    """Resolve the released Qwen3.5 conditional wrapper's logits-free base."""

    base_model = getattr(backbone, "model", None)
    if not isinstance(base_model, nn.Module):
        raise ValueError(
            "Qwen multimodal base must resolve through explicit path 'model'"
        )
    return base_model


def _last_hidden_state(output: Any) -> torch.Tensor:
    last_hidden_state = getattr(output, "last_hidden_state", None)
    if not isinstance(last_hidden_state, torch.Tensor):
        raise RuntimeError(
            "Qwen multimodal base output must expose last_hidden_state"
        )
    return last_hidden_state


class BatonQwen35Planner(nn.Module):
    """Dense Qwen3.5 backbone followed by the continuous Baton Query Tower."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        added_token_ids: tuple[int, ...],
        query_tower: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch module")
        get_embeddings = getattr(backbone, "get_input_embeddings", None)
        set_embeddings = getattr(backbone, "set_input_embeddings", None)
        if not callable(get_embeddings) or not callable(set_embeddings):
            raise TypeError(
                "backbone must expose get_input_embeddings and set_input_embeddings"
            )
        base_embedding = get_embeddings()
        if not isinstance(base_embedding, nn.Module):
            raise TypeError("Qwen input embedding must be a torch module")

        self.backbone = backbone
        _multimodal_base_model(backbone)
        self.frozen_base_embedding = base_embedding
        adapter = PlanTokenEmbeddingAdapter(base_embedding, added_token_ids)
        set_embeddings(adapter)
        if get_embeddings() is not adapter:
            raise RuntimeError("Qwen did not install the plan-token embedding adapter")

        qwen_dim = adapter.embedding_dim
        if query_tower is None:
            query_tower = BatonVisualAlignmentTower(qwen_dim)
        if not isinstance(query_tower, nn.Module):
            raise TypeError("query_tower must be a torch module")
        expected_geometry = {
            "qwen_dim": qwen_dim,
            "num_frames": _NUM_FRAMES,
            "tokens_per_frame": _TOKENS_PER_FRAME,
        }
        for name, expected in expected_geometry.items():
            if getattr(query_tower, name, None) != expected:
                raise ValueError(f"query_tower {name} must be {expected}")
        self.query_tower = query_tower
        self.sem_mlp = nn.Sequential(
            nn.Linear(qwen_dim, qwen_dim),
            nn.GELU(),
            nn.Linear(qwen_dim, _FEATURE_DIM),
        )

    @property
    def plan_token_adapter(self) -> PlanTokenEmbeddingAdapter:
        embedding = self.backbone.get_input_embeddings()
        if not isinstance(embedding, PlanTokenEmbeddingAdapter):
            raise RuntimeError("Qwen plan-token embedding adapter was replaced")
        return embedding

    def _validate_rows(
        self,
        qwen_inputs: Mapping[str, torch.Tensor],
        plan_positions: torch.Tensor,
        *,
        validate_input_contents: bool,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if type(validate_input_contents) is not bool:
            raise TypeError("validate_input_contents must be boolean")
        if not isinstance(qwen_inputs, Mapping):
            raise TypeError("qwen_inputs must be a tensor mapping")
        input_ids = qwen_inputs.get("input_ids")
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.ndim != 2
            or input_ids.dtype == torch.bool
            or input_ids.dtype.is_floating_point
        ):
            raise ValueError("qwen_inputs input_ids must be a rank-2 integer tensor")
        rows, sequence_length = input_ids.shape
        if (
            not isinstance(plan_positions, torch.Tensor)
            or plan_positions.shape != (rows, _PLAN_TOKENS)
        ):
            raise ValueError(
                "plan_positions must contain exactly 1024 positions per Qwen row"
            )
        if (
            plan_positions.dtype == torch.bool
            or plan_positions.dtype.is_floating_point
        ):
            raise TypeError("plan_positions must contain integer sequence indices")
        positions = plan_positions.to(device=input_ids.device, dtype=torch.long)
        if validate_input_contents:
            if bool(((positions < 0) | (positions >= sequence_length)).any()):
                raise ValueError(
                    "plan_positions contain an out-of-range sequence index"
                )
            plan_pad_id = self.plan_token_adapter.plan_pad_token_id
            pad_mask = input_ids.eq(plan_pad_id)
            if bool(pad_mask.sum(dim=1).ne(_PLAN_TOKENS).any()):
                raise ValueError(
                    "each Qwen row must contain exactly 1024 <PLAN_PAD> tokens"
                )
            gathered_ids = torch.gather(input_ids, 1, positions)
            if not bool(gathered_ids.eq(plan_pad_id).all()):
                raise ValueError(
                    "plan_positions must point only to <PLAN_PAD> tokens"
                )
            expected_positions = torch.stack(
                tuple(
                    torch.nonzero(row, as_tuple=False).flatten()
                    for row in pad_mask
                )
            )
            if not torch.equal(positions, expected_positions):
                raise ValueError(
                    "plan_positions must match all <PLAN_PAD> positions in sequence order"
                )

        forwarded: dict[str, torch.Tensor] = {}
        for name, value in qwen_inputs.items():
            if not isinstance(name, str) or not isinstance(value, torch.Tensor):
                raise TypeError("qwen_inputs must map string names to tensors")
            forwarded[name] = value
        return forwarded, positions

    def forward_rows(
        self,
        qwen_inputs: Mapping[str, torch.Tensor],
        plan_positions: torch.Tensor,
        *,
        return_attention_maps: bool = False,
        validate_input_contents: bool = True,
    ) -> BatonPlannerOutput:
        """Run one causal Qwen pass and predict one continuous grid per row."""

        forwarded, positions = self._validate_rows(
            qwen_inputs,
            plan_positions,
            validate_input_contents=validate_input_contents,
        )
        forwarded["use_cache"] = False
        forwarded["output_hidden_states"] = False
        forwarded["return_dict"] = True
        qwen_output = _multimodal_base_model(self.backbone)(**forwarded)
        last_hidden = _last_hidden_state(qwen_output)
        input_ids = forwarded["input_ids"]
        if (
            last_hidden.ndim != 3
            or tuple(last_hidden.shape[:2]) != tuple(input_ids.shape)
            or last_hidden.shape[-1] != self.plan_token_adapter.embedding_dim
        ):
            raise RuntimeError(
                "Qwen final hidden state must be [rows,sequence,qwen_dim]"
            )
        positions = positions.to(device=last_hidden.device)
        gathered = torch.gather(
            last_hidden,
            dim=1,
            index=positions.unsqueeze(-1).expand(
                -1, -1, last_hidden.shape[-1]
            ),
        )
        qwen_plan_states = gathered.reshape(
            last_hidden.shape[0],
            _NUM_FRAMES,
            _TOKENS_PER_FRAME,
            last_hidden.shape[-1],
        )
        tower_output = self.query_tower(
            qwen_plan_states,
            return_attention_maps=return_attention_maps,
        )
        if not isinstance(tower_output, QueryTowerOutput):
            raise RuntimeError("query_tower must return QueryTowerOutput")
        predictions = self.sem_mlp(tower_output.hidden_states)
        expected_shape = (
            last_hidden.shape[0],
            _NUM_FRAMES,
            _TOKENS_PER_FRAME,
            _FEATURE_DIM,
        )
        if tuple(predictions.shape) != expected_shape:
            raise RuntimeError(
                "Sem-MLP predictions must be [rows,4,256,1024]"
            )
        return BatonPlannerOutput(
            flat=predictions,
            positive=predictions,
            cross_attention_maps=tower_output.cross_attention_maps,
        )

    def forward(
        self,
        batch: BatonPlannerBatch,
        *,
        return_attention_maps: bool = False,
        validate_input_contents: bool = True,
    ) -> BatonPlannerOutput:
        if not isinstance(batch, BatonPlannerBatch):
            raise TypeError("batch must be BatonPlannerBatch")
        batch_size = batch.batch_size
        camera_count = len(batch.camera_names)
        if (
            batch_size <= 0
            or camera_count <= 0
            or len(batch.row_labels) != batch_size * camera_count
        ):
            raise ValueError("Baton batch must contain one Qwen row per camera per sample")
        expected_labels = tuple(
            (sample, camera)
            for sample in range(batch_size)
            for camera in batch.camera_names
        )
        if batch.row_labels != expected_labels:
            raise ValueError("Baton rows must be sample-major in camera_names order")
        row_output = self.forward_rows(
            batch.qwen_inputs,
            batch.plan_positions,
            return_attention_maps=return_attention_maps,
            validate_input_contents=validate_input_contents,
        )
        return BatonPlannerOutput(
            flat=row_output.flat,
            positive=row_output.flat.reshape(
                batch_size,
                camera_count,
                _NUM_FRAMES,
                _TOKENS_PER_FRAME,
                _FEATURE_DIM,
            ),
            cross_attention_maps=row_output.cross_attention_maps,
        )
