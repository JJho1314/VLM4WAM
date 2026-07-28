"""Tiny, executable Baton Stage-1/2/3 and exact-resume verification.

The modules in this file replace expensive live Qwen3.5, SigLIP2, and LTX
compute only. Tensor validation, ownership, conditioning, optimizer grouping,
checkpoint envelopes, strict Stage-3 loading, and distributed synchronization
all go through the production interfaces.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
from safetensors.torch import save_file
import torch
import torch.distributed as dist
import torch.nn as nn

from qwen35_baton.cli.train_semantic_planner import (
    Stage1TrainingConfig,
    build_stage1_optimizer_groups,
)
from qwen35_baton.data import BatonPlannerBatch
from qwen35_baton.hashing import sha256_json
from qwen35_baton.losses import compute_baton_planner_loss
from qwen35_baton.model import BatonQwen35Planner
from qwen35_baton.ownership import configure_stage1_trainable_modules
from qwen35_baton.provider import (
    BatonSemanticPlan,
    build_patch_center_positions as build_provider_positions,
)
from qwen35_baton.query_tower import QueryTowerOutput
from qwen35_baton.teacher import FrozenSiglip2Teacher


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_GE_ACT_ROOT = _REPOSITORY_ROOT / "ge_act"
if str(_GE_ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(_GE_ACT_ROOT))

from models.ltx_models.baton_semantic_planner import (  # noqa: E402
    FrozenDualCameraBatonPlanner,
)
from runner.ge_trainer import (  # noqa: E402
    BATON_PREDICTION_SOURCE,
    BATON_TEACHER_SOURCE,
    BatonConditioningComponents,
    TrainingCursor,
    build_baton_semantic_condition,
    build_optimizer_parameter_groups,
    forward_baton_ge_act,
    load_baton_training_checkpoint,
    save_baton_training_checkpoint,
    strict_load_baton_stage3_diffusion_model,
)


_SAMPLER_SEED = 42
_STEP_SEED = 1729
_MICROBATCHES_PER_EPOCH = 4
_FUTURE_INDICES = (0, 3, 5, 8)


@dataclass(frozen=True)
class StageSmokeResult:
    optimizer_steps: int
    plan_shape: tuple[int, ...]
    condition_source: str
    source_ownership: str
    source_hash_before: str
    source_hash_after: str
    trainable_hash_before: str
    trainable_hash_after: str


@dataclass(frozen=True)
class CheckpointSmokeResult:
    source: str
    cursor: Mapping[str, int]
    envelope_loaded: bool
    strict_stage3_loaded: bool
    stage2_artifact_hash: str
    optimizer_hash: str
    scheduler_hash: str
    rng_hash: str


@dataclass(frozen=True)
class TinyPipelineResult:
    stage1: StageSmokeResult
    stage2: StageSmokeResult
    stage3: StageSmokeResult
    checkpoint: CheckpointSmokeResult
    rank_agreement: bool
    executed_ranks: tuple[int, ...]
    exact_resume: bool | None
    fresh_process_restore: bool | None


@dataclass(frozen=True)
class _Stage1Artifacts:
    result: StageSmokeResult
    predicted_tokens: torch.Tensor
    teacher: FrozenSiglip2Teacher
    rank_agreement: bool


@dataclass(frozen=True)
class _Stage2Artifacts:
    result: StageSmokeResult
    checkpoint: Path
    cursor: TrainingCursor
    artifact_hash: str
    optimizer_hash: str
    scheduler_hash: str
    rng_hash: str
    first_sample: tuple[int, int, int]
    rank_agreement: bool


class _TinyImageProcessor:
    def __call__(
        self,
        *,
        images: Sequence[torch.Tensor],
        return_tensors: str,
    ) -> Mapping[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError("tiny SigLIP2 processor only supports PyTorch tensors")
        return {
            "pixel_values": torch.stack(tuple(images)).to(torch.float32).div(255.0)
        }


class _TinySiglipVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.125))

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        output_hidden_states: bool,
    ) -> Any:
        if output_hidden_states is not True:
            raise ValueError("tiny SigLIP2 must expose hidden states")
        value = pixel_values.mean(dim=(1, 2, 3), keepdim=False)[:, None, None]
        tokens = (value + self.anchor).expand(-1, 257, 1024)
        return SimpleNamespace(
            hidden_states=(tokens * 0.0, tokens, tokens + 0.01)
        )


class _TinyQwenLanguage(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(width, width, bias=False) for _ in range(24)
        )


class _TinyQwenBase(nn.Module):
    def __init__(self, *, vocab_size: int, width: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.language_model = _TinyQwenLanguage(width)
        self.visual = nn.Linear(1, 1, bias=False)

    def forward(self, input_ids: torch.Tensor, **_: Any) -> Any:
        hidden = self.embedding(input_ids)
        hidden = hidden + hidden.mean(dim=1, keepdim=True) * 0.1
        for layer in self.language_model.layers:
            hidden = hidden + torch.tanh(layer(hidden)) * 0.02
        hidden = hidden + self.visual.weight.reshape(1, 1, 1) * 0.01
        return SimpleNamespace(last_hidden_state=hidden)


class _TinyQwenBackbone(nn.Module):
    def __init__(self, *, vocab_size: int = 128, width: int = 8) -> None:
        super().__init__()
        self.model = _TinyQwenBase(vocab_size=vocab_size, width=width)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embedding

    def set_input_embeddings(self, embedding: nn.Module) -> None:
        self.model.embedding = embedding


class _TinyQueryTower(nn.Module):
    qwen_dim = 8
    query_dim = 1024
    num_frames = 4
    tokens_per_frame = 256
    num_cameras = 2

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.75))
        self.camera_embeddings = nn.Embedding(2, 1024)

    def forward(
        self,
        qwen_states: torch.Tensor,
        camera_ids: torch.Tensor,
        *,
        return_attention_maps: bool,
    ) -> QueryTowerOutput:
        hidden = qwen_states.mean(dim=-1, keepdim=True).expand(
            -1, -1, -1, 1024
        )
        camera = self.camera_embeddings(camera_ids)[:, None, None]
        return QueryTowerOutput(
            hidden_states=hidden * self.scale + camera,
            cross_attention_maps=() if return_attention_maps else None,
        )


class _TinySemanticProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.bias = nn.Parameter(torch.tensor(0.01))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.scale + self.bias


class _TinyGeActModel(nn.Module):
    """Three-owner GE-Act module with a real safetensors deployment snapshot."""

    def __init__(self) -> None:
        super().__init__()
        self.video_weight = nn.Parameter(torch.tensor(0.25))
        self.action_expert = nn.Parameter(torch.tensor(-0.4))
        self.semantic_adapter = nn.Parameter(torch.tensor(0.15))

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        semantic_plan: torch.Tensor,
        semantic_condition_mask: torch.Tensor,
        **_: Any,
    ) -> tuple[dict[str, torch.Tensor]]:
        base = hidden_states.float().mean()
        semantic = semantic_plan.float().mean()
        gate = semantic_condition_mask.float().mean()
        video = (
            self.video_weight * (base + 0.5)
            + self.semantic_adapter * semantic * gate
        )
        action = self.action_expert * (semantic + base + 0.25)
        return (
            {
                "video": video.reshape(1, 1, 1),
                "action": action.reshape(1, 1, 1),
            },
        )

    def save_pretrained(
        self,
        output_dir: str | Path,
        *,
        safe_serialization: bool,
    ) -> None:
        if safe_serialization is not True:
            raise ValueError("Baton snapshots require safetensors")
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=False)
        save_file(
            {
                name: value.detach().cpu()
                for name, value in self.state_dict().items()
            },
            str(destination / "diffusion_pytorch_model.safetensors"),
        )


class _TinyPredictedProvider(nn.Module):
    def __init__(self, tokens: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("predicted_tokens", tokens.detach().clone())
        self.anchor = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    @torch.no_grad()
    def predict(
        self,
        current_images: torch.Tensor,
        instructions: Sequence[str],
        **_: Any,
    ) -> BatonSemanticPlan:
        batch_size = int(current_images.shape[0])
        if batch_size != len(instructions):
            raise ValueError("predicted source requires one instruction per sample")
        tokens = self.predicted_tokens[:1].expand(batch_size, -1, -1, -1, -1)
        return BatonSemanticPlan(
            tokens=tokens.detach(),
            future_indices=_FUTURE_INDICES,
            positions_xy=build_provider_positions(
                batch_size,
                device=tokens.device,
            ),
            cross_attention_maps=None,
            instruction_sensitivity=None,
        )


def _canonical(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes().hex()
        return {
            "tensor": {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "bytes": raw,
            }
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "ndarray": {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "bytes": array.tobytes().hex(),
            }
        }
    if isinstance(value, Mapping):
        items = [(_canonical(key), _canonical(child)) for key, child in value.items()]
        items.sort(
            key=lambda item: json.dumps(
                item[0], sort_keys=True, separators=(",", ":")
            )
        )
        return {"mapping": items}
    if isinstance(value, tuple):
        return {"tuple": [_canonical(child) for child in value]}
    if isinstance(value, list):
        return {"list": [_canonical(child) for child in value]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported smoke hash value: {type(value).__name__}")


def _state_hash(value: Any) -> str:
    payload = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _module_hash(module: nn.Module, *, trainable_only: bool = False) -> str:
    if trainable_only:
        state = {
            name: parameter
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
    else:
        state = module.state_dict()
    return _state_hash(state)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _rank_hash_agrees(digest: str) -> bool:
    if not dist.is_initialized():
        return True
    gathered: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, digest)
    return len(set(gathered)) == 1


def _distributed_model(model: nn.Module) -> nn.Module:
    if not dist.is_initialized():
        return model
    return torch.nn.parallel.DistributedDataParallel(model)


def _make_teacher() -> FrozenSiglip2Teacher:
    return FrozenSiglip2Teacher.from_components(
        processor=_TinyImageProcessor(),
        vision_model=_TinySiglipVision(),
        device="cpu",
        dtype=torch.float32,
        frame_microbatch_size=4,
    )


def _stage1_batch() -> BatonPlannerBatch:
    plan_pad_id = 105
    input_ids = torch.full((4, 1025), plan_pad_id, dtype=torch.long)
    input_ids[:, 0] = torch.tensor([10, 11, 12, 13])
    plan_positions = torch.arange(1, 1025).expand(4, -1).clone()
    current = torch.arange(1 * 2 * 3 * 2 * 2, dtype=torch.uint8).reshape(
        1, 2, 3, 2, 2
    )
    future = torch.arange(
        1 * 2 * 4 * 3 * 2 * 2,
        dtype=torch.uint8,
    ).reshape(1, 2, 4, 3, 2, 2)
    return BatonPlannerBatch(
        qwen_inputs={
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        },
        plan_positions=plan_positions,
        current_images=current,
        future_images=future,
        instructions=("pick the red cube",),
        negative_instructions=("place the blue cube",),
        row_labels=(
            ("positive", 0, "main"),
            ("positive", 0, "wrist"),
            ("negative", 0, "main"),
            ("negative", 0, "wrist"),
        ),
    )


def _stage1_config(output_dir: Path) -> Stage1TrainingConfig:
    return Stage1TrainingConfig(
        output_dir=str(output_dir),
        qwen_model_path="tiny-local-qwen",
        qwen_processor_path="tiny-local-qwen-processor",
        qwen_tokenizer_path="tiny-local-qwen-tokenizer",
        siglip2_model_path="tiny-local-siglip2",
        siglip2_config_hash="1" * 64,
        siglip2_artifact_hash="2" * 64,
        hdf5_manifest_path="tiny-local-manifest",
        hdf5_manifest_hash="3" * 64,
        dataset_statistics_path="tiny-local-statistics",
        per_device_batch=1,
        gradient_accumulation_steps=1,
        max_steps=2,
        warmup_steps=0,
        save_every=1,
        mixed_precision="no",
        num_workers=0,
        tiny_test=True,
    )


def _run_stage1(output_dir: Path) -> _Stage1Artifacts:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        planner = BatonQwen35Planner(
            _TinyQwenBackbone(),
            added_token_ids=tuple(range(100, 107)),
            query_tower=_TinyQueryTower(),
        )
    planner.sem_mlp = _TinySemanticProjection()
    ownership = configure_stage1_trainable_modules(planner)
    groups = build_stage1_optimizer_groups(
        planner,
        ownership,
        _stage1_config(output_dir),
    )
    optimizer = torch.optim.SGD(groups)
    teacher = _make_teacher()
    source_before = _module_hash(teacher.model)
    trainable_before = _module_hash(planner, trainable_only=True)
    batch = _stage1_batch()
    wrapped = _distributed_model(planner)
    output = wrapped(batch)
    with torch.no_grad():
        current_teacher = teacher.encode_current(batch.current_images)
        future_teacher = teacher.encode_future(batch.future_images)
    if output.negative is None:
        raise RuntimeError("Stage 1 did not produce the counterfactual plan")
    loss = compute_baton_planner_loss(
        output.positive,
        output.negative,
        future_teacher,
        current_teacher,
    )
    loss.total.backward()
    torch.nn.utils.clip_grad_norm_(
        tuple(
            parameter
            for parameter in planner.parameters()
            if parameter.requires_grad
        ),
        1.0,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    source_after = _module_hash(teacher.model)
    trainable_after = _module_hash(planner, trainable_only=True)
    if source_before != source_after:
        raise RuntimeError("Stage-1 frozen SigLIP2 source changed")
    if trainable_before == trainable_after:
        raise RuntimeError("Stage-1 optimizer did not update trainable parameters")
    rank_agreement = _rank_hash_agrees(trainable_after)
    if not rank_agreement:
        raise RuntimeError("Stage-1 synchronized parameter hashes differ by rank")
    plan = output.positive.detach()
    return _Stage1Artifacts(
        result=StageSmokeResult(
            optimizer_steps=1,
            plan_shape=tuple(plan.shape),
            condition_source="teacher_supervision",
            source_ownership="frozen_siglip2_teacher",
            source_hash_before=source_before,
            source_hash_after=source_after,
            trainable_hash_before=trainable_before,
            trainable_hash_after=trainable_after,
        ),
        predicted_tokens=plan,
        teacher=teacher,
        rank_agreement=rank_agreement,
    )


def _semantic_config(source: str) -> dict[str, Any]:
    semantic: dict[str, Any] = {
        "enabled": True,
        "source": source,
        "tokens_per_frame": 256,
        "feature_dim": 1024,
        "keyframe_indices": [0, 3, 5, 8],
    }
    if source == BATON_TEACHER_SOURCE:
        semantic.update(
            {
                "siglip2_model_path": "tiny-local-siglip2",
                "siglip2_config_hash": "1" * 64,
                "siglip2_artifact_hash": "2" * 64,
                "teacher_preprocessing_hash": "2" * 64,
                "frame_microbatch_size": 4,
            }
        )
    elif source == BATON_PREDICTION_SOURCE:
        semantic.update(
            {
                "planner_checkpoint": "tiny-local-stage1",
                "expected_planner_topology": "tiny-local-topology",
                "qwen_model_path": "tiny-local-qwen",
                "qwen_tokenizer_path": "tiny-local-tokenizer",
                "qwen_processor_path": "tiny-local-processor",
                "siglip2_model_path": "tiny-local-siglip2",
            }
        )
    else:
        raise ValueError(f"unsupported tiny Baton source: {source}")
    return semantic


def _teacher_components(
    teacher: FrozenSiglip2Teacher | None = None,
) -> BatonConditioningComponents:
    return BatonConditioningComponents(
        source=BATON_TEACHER_SOURCE,
        teacher=_make_teacher() if teacher is None else teacher,
        planner=None,
    )


def _prediction_components(
    predicted_tokens: torch.Tensor,
) -> BatonConditioningComponents:
    planner = FrozenDualCameraBatonPlanner(
        _TinyPredictedProvider(predicted_tokens)
    )
    return BatonConditioningComponents(
        source=BATON_PREDICTION_SOURCE,
        teacher=None,
        planner=planner,
    )


def _sample_marker() -> tuple[int, int, int]:
    return (
        random.randrange(1000),
        int(np.random.randint(0, 1000)),
        int(torch.randint(0, 1000, ()).item()),
    )


def _video_for_sample(sample: tuple[int, int, int]) -> torch.Tensor:
    base = (sum(sample) % 41) / 80.0 - 0.25
    timeline = torch.arange(13, dtype=torch.float32).reshape(1, 1, 1, 13, 1, 1)
    views = torch.tensor([0.0, 0.03]).reshape(1, 1, 2, 1, 1, 1)
    return (torch.full((1, 3, 2, 13, 2, 2), base) + timeline / 64 + views).clamp(
        -1.0, 1.0
    )


def _diffusion_kwargs(sample: tuple[int, int, int]) -> dict[str, Any]:
    scalar = (sum(sample) % 17) / 100.0
    return {
        "timesteps": torch.ones(2, dtype=torch.long),
        "noisy_latents": torch.full((2, 1, 1), scalar),
        "prompt_embeds": torch.zeros(1, 1, 1),
        "prompt_attention_mask": torch.ones(1, 1),
        "num_frames": 6,
        "height": 1,
        "width": 1,
        "n_view": 2,
    }


def _ge_optimizer(
    model: _TinyGeActModel,
    source: str,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.StepLR]:
    groups = build_optimizer_parameter_groups(
        model,
        train_mode="all",
        base_lr=2e-5,
        action_lr=1e-4,
        semantic_lr=5e-5,
        baton_source=source,
    )
    optimizer = torch.optim.SGD(groups, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.8,
    )
    return optimizer, scheduler


def _ge_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    components: BatonConditioningComponents,
    source: str,
    *,
    sample: tuple[int, int, int] | None = None,
) -> tuple[int, int, int]:
    selected = _sample_marker() if sample is None else sample
    condition = build_baton_semantic_condition(
        components,
        _semantic_config(source),
        _video_for_sample(selected),
        ("pick the red cube",),
        n_previous=4,
        num_future_frames=9,
        num_latent_frames=6,
        device="cpu",
        dtype=torch.float32,
    )
    output = forward_baton_ge_act(
        model,
        condition,
        semantic_condition_mask=torch.ones(2),
        diffusion_kwargs=_diffusion_kwargs(selected),
    )
    latents = output["latents"]
    loss = latents["video"].square().mean() + latents["action"].square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return selected


def _provenance_hash(label: str) -> str:
    return hashlib.sha256(f"qwen35-baton-smoke:{label}".encode("utf-8")).hexdigest()


def _teacher_training_provenance() -> dict[str, str | int]:
    sampling = {
        "algorithm": "libero_fastwam_hdf5_stateless_sha256",
        "version": 1,
        "seed": _SAMPLER_SEED,
    }
    return {
        "hdf5_manifest_hash": _provenance_hash("hdf5"),
        "siglip2_config_hash": _provenance_hash("siglip2-config"),
        "siglip2_artifact_hash": _provenance_hash("siglip2-artifact"),
        "teacher_preprocessing_hash": _provenance_hash("preprocessing"),
        "window_sampling_algorithm": sampling["algorithm"],
        "window_sampling_version": sampling["version"],
        "window_sampling_seed": sampling["seed"],
        "window_sampling_topology_hash": sha256_json(sampling),
    }


def _training_state_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": _rng_state(),
    }


class _CheckpointStateAdapter:
    """Minimal Accelerate state adapter; the envelope remains production code."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        *,
        rank: int,
        world_size: int,
        distributed_save: bool,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.rank = rank
        self.world_size = world_size
        self.distributed_save = distributed_save
        self.is_main_process = rank == 0

    def wait_for_everyone(self) -> None:
        if self.distributed_save:
            _barrier()

    def save_state(self, output_dir: str) -> None:
        local = _training_state_payload(
            self.model,
            self.optimizer,
            self.scheduler,
        )
        if self.distributed_save:
            states: list[Any] = [None] * self.world_size
            dist.all_gather_object(states, local)
        else:
            states = [local]
        if self.is_main_process:
            torch.save(
                {"format_version": 1, "rank_states": states},
                Path(output_dir) / "accelerator_state.pt",
            )

    def load_state(self, checkpoint_dir: str) -> None:
        payload = torch.load(
            Path(checkpoint_dir) / "accelerator_state.pt",
            weights_only=False,
            map_location="cpu",
        )
        states = payload.get("rank_states")
        if (
            payload.get("format_version") != 1
            or not isinstance(states, list)
            or len(states) != self.world_size
        ):
            raise ValueError("tiny Accelerator state has invalid rank coverage")
        selected = states[self.rank]
        self.model.load_state_dict(selected["model"], strict=True)
        self.optimizer.load_state_dict(selected["optimizer"])
        self.scheduler.load_state_dict(selected["scheduler"])
        random.setstate(selected["rng"]["python"])
        np.random.set_state(selected["rng"]["numpy"])
        torch.random.set_rng_state(selected["rng"]["torch"])

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return model


def _checkpoint_cursor(step: int) -> TrainingCursor:
    return TrainingCursor(
        global_step=step,
        epoch=0,
        consumed_microbatches=step,
        microbatches_per_epoch=_MICROBATCHES_PER_EPOCH,
        sampler_seed=_SAMPLER_SEED,
    )


def _run_stage2(
    output_dir: Path,
    teacher: FrozenSiglip2Teacher,
) -> _Stage2Artifacts:
    _seed_all(_STEP_SEED)
    model = _TinyGeActModel()
    optimizer, scheduler = _ge_optimizer(model, BATON_TEACHER_SOURCE)
    components = _teacher_components(teacher)
    source_before = _module_hash(teacher.model)
    trainable_before = _module_hash(model)
    wrapped = _distributed_model(model)
    first_sample = _ge_optimizer_step(
        wrapped,
        optimizer,
        scheduler,
        components,
        BATON_TEACHER_SOURCE,
    )
    source_after = _module_hash(teacher.model)
    trainable_after = _module_hash(model)
    if source_before != source_after:
        raise RuntimeError("Stage-2 frozen teacher changed")
    if trainable_before == trainable_after:
        raise RuntimeError("Stage-2 optimizer did not update GE-Act")
    rank_agreement = _rank_hash_agrees(trainable_after)
    if not rank_agreement:
        raise RuntimeError("Stage-2 synchronized parameter hashes differ by rank")

    cursor = _checkpoint_cursor(1)
    adapter = _CheckpointStateAdapter(
        model,
        optimizer,
        scheduler,
        rank=_rank(),
        world_size=_world_size(),
        distributed_save=dist.is_initialized(),
    )
    checkpoint = save_baton_training_checkpoint(
        adapter,
        output_dir / "stage2",
        cursor=cursor,
        diffusion_model=model,
        source=BATON_TEACHER_SOURCE,
        training_provenance=_teacher_training_provenance(),
    )
    return _Stage2Artifacts(
        result=StageSmokeResult(
            optimizer_steps=1,
            plan_shape=(1, 2, 4, 256, 1024),
            condition_source="teacher",
            source_ownership="frozen_siglip2_teacher",
            source_hash_before=source_before,
            source_hash_after=source_after,
            trainable_hash_before=trainable_before,
            trainable_hash_after=trainable_after,
        ),
        checkpoint=checkpoint,
        cursor=cursor,
        artifact_hash=trainable_after,
        optimizer_hash=_state_hash(optimizer.state_dict()),
        scheduler_hash=_state_hash(scheduler.state_dict()),
        rng_hash=_state_hash(_rng_state()),
        first_sample=first_sample,
        rank_agreement=rank_agreement,
    )


def _load_stage2_envelope(
    stage2: _Stage2Artifacts,
) -> bool:
    model = _TinyGeActModel()
    optimizer, scheduler = _ge_optimizer(model, BATON_TEACHER_SOURCE)
    adapter = _CheckpointStateAdapter(
        model,
        optimizer,
        scheduler,
        rank=_rank(),
        world_size=_world_size(),
        distributed_save=False,
    )
    restored = load_baton_training_checkpoint(
        adapter,
        stage2.checkpoint,
        diffusion_model=model,
        expected_source=BATON_TEACHER_SOURCE,
        expected_microbatches_per_epoch=_MICROBATCHES_PER_EPOCH,
        expected_sampler_seed=_SAMPLER_SEED,
        expected_training_provenance=_teacher_training_provenance(),
    )
    if restored != stage2.cursor or _module_hash(model) != stage2.artifact_hash:
        raise RuntimeError("Stage-2 envelope restore differs from saved state")
    return True


def _run_stage3(
    stage2: _Stage2Artifacts,
    predicted_tokens: torch.Tensor,
) -> tuple[StageSmokeResult, bool, bool]:
    metadata = json.loads(
        (stage2.checkpoint / "baton_state.json").read_text(encoding="utf-8")
    )
    model = _TinyGeActModel()
    strict_load_baton_stage3_diffusion_model(
        model,
        stage2.checkpoint / "diffusion_model",
        expected_snapshot_topology_hash=metadata["snapshot_topology_hash"],
        expected_diffusion_files=metadata["diffusion_files"],
    )
    trainable_before = _module_hash(model)
    if trainable_before != stage2.artifact_hash:
        raise RuntimeError("Stage-3 strict load differs from Stage-2 artifact")
    components = _prediction_components(predicted_tokens)
    assert components.planner is not None
    source_before = _module_hash(components.planner)
    optimizer, scheduler = _ge_optimizer(model, BATON_PREDICTION_SOURCE)
    wrapped = _distributed_model(model)
    _ge_optimizer_step(
        wrapped,
        optimizer,
        scheduler,
        components,
        BATON_PREDICTION_SOURCE,
        sample=(3, 5, 8),
    )
    source_after = _module_hash(components.planner)
    trainable_after = _module_hash(model)
    if source_before != source_after:
        raise RuntimeError("Stage-3 frozen predicted source changed")
    if trainable_before == trainable_after:
        raise RuntimeError("Stage-3 optimizer did not update GE-Act")
    rank_agreement = _rank_hash_agrees(trainable_after)
    if not rank_agreement:
        raise RuntimeError("Stage-3 synchronized parameter hashes differ by rank")
    return (
        StageSmokeResult(
            optimizer_steps=1,
            plan_shape=(1, 2, 4, 256, 1024),
            condition_source="prediction",
            source_ownership="frozen_baton_prediction",
            source_hash_before=source_before,
            source_hash_after=source_after,
            trainable_hash_before=trainable_before,
            trainable_hash_after=trainable_after,
        ),
        True,
        rank_agreement,
    )


def _resume_worker(
    checkpoint: Path,
    result_path: Path,
    *,
    rank: int,
    world_size: int,
) -> int:
    model = _TinyGeActModel()
    optimizer, scheduler = _ge_optimizer(model, BATON_TEACHER_SOURCE)
    teacher = _make_teacher()
    adapter = _CheckpointStateAdapter(
        model,
        optimizer,
        scheduler,
        rank=rank,
        world_size=world_size,
        distributed_save=False,
    )
    restored = load_baton_training_checkpoint(
        adapter,
        checkpoint,
        diffusion_model=model,
        expected_source=BATON_TEACHER_SOURCE,
        expected_microbatches_per_epoch=_MICROBATCHES_PER_EPOCH,
        expected_sampler_seed=_SAMPLER_SEED,
        expected_training_provenance=_teacher_training_provenance(),
    )
    sample = _ge_optimizer_step(
        model,
        optimizer,
        scheduler,
        _teacher_components(teacher),
        BATON_TEACHER_SOURCE,
    )
    final_cursor = _checkpoint_cursor(2)
    probes = [
        random.random(),
        float(np.random.rand()),
        float(torch.rand(()).item()),
    ]
    payload = {
        "pid": os.getpid(),
        "restored_cursor": restored.to_dict(),
        "final_cursor": final_cursor.to_dict(),
        "sample": list(sample),
        "model_hash": _module_hash(model),
        "optimizer_hash": _state_hash(optimizer.state_dict()),
        "scheduler_hash": _state_hash(scheduler.state_dict()),
        "probes": probes,
        "rng_hash": _state_hash(_rng_state()),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    staging = result_path.with_name(f".{result_path.name}.{os.getpid()}.tmp")
    staging.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, result_path)
    return 0


def _fresh_process_resume(
    checkpoint: Path,
    result_path: Path,
    *,
    rank: int,
    world_size: int,
) -> Mapping[str, Any]:
    environment = dict(os.environ)
    for name in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
    ):
        environment.pop(name, None)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "qwen35_baton.cli.smoke_pipeline",
            "--internal-resume-worker",
            "--checkpoint",
            str(checkpoint),
            "--result-path",
            str(result_path),
            "--rank",
            str(rank),
            "--world-size",
            str(world_size),
        ],
        cwd=str(_REPOSITORY_ROOT),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "fresh-process resume worker failed: "
            f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
        )
    return json.loads(result_path.read_text(encoding="utf-8"))


def _verify_exact_resume(
    stage2: _Stage2Artifacts,
    output_dir: Path,
) -> tuple[bool, bool]:
    if not dist.is_initialized() or dist.get_world_size() != 2:
        raise RuntimeError("exact distributed resume requires a real two-rank group")
    _seed_all(_STEP_SEED)
    model = _TinyGeActModel()
    optimizer, scheduler = _ge_optimizer(model, BATON_TEACHER_SOURCE)
    teacher = _make_teacher()
    wrapped = _distributed_model(model)
    first_sample = _ge_optimizer_step(
        wrapped,
        optimizer,
        scheduler,
        _teacher_components(teacher),
        BATON_TEACHER_SOURCE,
    )
    if (
        first_sample != stage2.first_sample
        or _module_hash(model) != stage2.artifact_hash
    ):
        raise RuntimeError("uninterrupted first update differs from Stage-2 checkpoint")
    second_sample = _ge_optimizer_step(
        wrapped,
        optimizer,
        scheduler,
        _teacher_components(teacher),
        BATON_TEACHER_SOURCE,
    )
    baseline_probes = [
        random.random(),
        float(np.random.rand()),
        float(torch.rand(()).item()),
    ]
    baseline = {
        "final_cursor": _checkpoint_cursor(2).to_dict(),
        "sample": list(second_sample),
        "model_hash": _module_hash(model),
        "optimizer_hash": _state_hash(optimizer.state_dict()),
        "scheduler_hash": _state_hash(scheduler.state_dict()),
        "probes": baseline_probes,
        "rng_hash": _state_hash(_rng_state()),
    }
    result_path = output_dir / "resume-workers" / f"rank-{_rank()}.json"
    resumed = _fresh_process_resume(
        stage2.checkpoint,
        result_path,
        rank=_rank(),
        world_size=_world_size(),
    )
    expected_cursor = stage2.cursor.to_dict()
    exact = (
        resumed["restored_cursor"] == expected_cursor
        and all(resumed[name] == baseline[name] for name in baseline)
    )
    fresh = int(resumed["pid"]) != os.getpid()
    gathered: list[Mapping[str, Any] | None] = [None] * _world_size()
    dist.all_gather_object(
        gathered,
        {
            "rank": _rank(),
            "exact": exact,
            "fresh": fresh,
            "model_hash": resumed["model_hash"],
            "optimizer_hash": resumed["optimizer_hash"],
            "scheduler_hash": resumed["scheduler_hash"],
            "sample": resumed["sample"],
        },
    )
    if any(item is None for item in gathered):
        raise RuntimeError("distributed resume did not report every rank")
    complete = [item for item in gathered if item is not None]
    agreement = (
        all(bool(item["exact"]) and bool(item["fresh"]) for item in complete)
        and len({str(item["model_hash"]) for item in complete}) == 1
        and len({str(item["optimizer_hash"]) for item in complete}) == 1
        and len({str(item["scheduler_hash"]) for item in complete}) == 1
        and len(
            {
                tuple(int(value) for value in item["sample"])
                for item in complete
            }
        )
        == 1
    )
    if not agreement:
        raise RuntimeError("two-rank exact-resume state or sample sequence differs")
    return True, True


def _invocation_directory(output_dir: Path) -> Path:
    root = output_dir.expanduser().resolve()
    if _rank() == 0:
        root.mkdir(parents=True, exist_ok=True)
        invocation = os.environ.get(
            "QWEN35_BATON_SMOKE_INVOCATION_ID",
            f"invocation-{uuid.uuid4().hex}",
        )
        if (
            not invocation.startswith("invocation-")
            or Path(invocation).name != invocation
        ):
            raise ValueError("Baton smoke invocation ID is invalid")
    else:
        invocation = ""
    if dist.is_initialized():
        payload = [invocation]
        dist.broadcast_object_list(payload, src=0)
        invocation = str(payload[0])
    destination = root / invocation
    if _rank() == 0:
        destination.mkdir()
    _barrier()
    return destination


def validate_two_rank_result(path: str | Path) -> None:
    """Fail closed unless one invocation proved the complete two-rank contract."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        payload.get("rank_agreement") is not True
        or payload.get("exact_resume") is not True
        or payload.get("fresh_process_restore") is not True
        or payload.get("executed_ranks") != [0, 1]
        or any(
            payload.get(stage, {}).get("optimizer_steps") != 1
            for stage in ("stage1", "stage2", "stage3")
        )
    ):
        raise RuntimeError("two-rank Baton smoke result is incomplete")


def run_tiny_pipeline(
    output_dir: str | Path,
    *,
    verify_exact_resume: bool = False,
) -> TinyPipelineResult:
    """Run one tiny optimizer step for every Baton curriculum stage."""

    if verify_exact_resume and (
        not dist.is_initialized() or dist.get_world_size() != 2
    ):
        raise RuntimeError(
            "verify_exact_resume requires torchrun with exactly two real ranks"
        )
    invocation = _invocation_directory(Path(output_dir))
    stage1 = _run_stage1(invocation)
    stage2 = _run_stage2(invocation, stage1.teacher)
    envelope_loaded = _load_stage2_envelope(stage2)
    stage3_result, strict_loaded, stage3_rank_agreement = _run_stage3(
        stage2,
        stage1.predicted_tokens,
    )
    exact: bool | None = None
    fresh: bool | None = None
    if verify_exact_resume:
        exact, fresh = _verify_exact_resume(stage2, invocation)

    local_counts = (
        stage1.result.optimizer_steps,
        stage2.result.optimizer_steps,
        stage3_result.optimizer_steps,
    )
    if dist.is_initialized():
        counts: list[tuple[int, int, int] | None] = [None] * _world_size()
        dist.all_gather_object(counts, local_counts)
        if any(value != (1, 1, 1) for value in counts):
            raise RuntimeError("one or more distributed ranks skipped a Baton stage")
        executed_ranks = tuple(range(_world_size()))
    else:
        executed_ranks = (0,)
    rank_agreement = (
        stage1.rank_agreement
        and stage2.rank_agreement
        and stage3_rank_agreement
        and (exact is not False)
    )
    result = TinyPipelineResult(
        stage1=stage1.result,
        stage2=stage2.result,
        stage3=stage3_result,
        checkpoint=CheckpointSmokeResult(
            source=BATON_TEACHER_SOURCE,
            cursor=stage2.cursor.to_dict(),
            envelope_loaded=envelope_loaded,
            strict_stage3_loaded=strict_loaded,
            stage2_artifact_hash=stage2.artifact_hash,
            optimizer_hash=stage2.optimizer_hash,
            scheduler_hash=stage2.scheduler_hash,
            rng_hash=stage2.rng_hash,
        ),
        rank_agreement=rank_agreement,
        executed_ranks=executed_ranks,
        exact_resume=exact,
        fresh_process_restore=fresh,
    )
    if _rank() == 0:
        (invocation / "result.json").write_text(
            json.dumps(
                asdict(result),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    _barrier()
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the tiny production-path Baton curriculum smoke"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--verify-exact-resume", action="store_true")
    parser.add_argument(
        "--internal-resume-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--checkpoint", help=argparse.SUPPRESS)
    parser.add_argument("--result-path", help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--world-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--validate-two-rank-result", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate_two_rank_result:
        validate_two_rank_result(args.validate_two_rank_result)
        return 0
    if args.internal_resume_worker:
        if (
            not args.checkpoint
            or not args.result_path
            or args.rank is None
            or args.world_size is None
        ):
            raise ValueError("internal resume worker arguments are incomplete")
        return _resume_worker(
            Path(args.checkpoint),
            Path(args.result_path),
            rank=args.rank,
            world_size=args.world_size,
        )
    if not args.output_dir:
        raise ValueError("--output-dir is required")

    environment_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    initialized_here = False
    if environment_world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="gloo")
        initialized_here = True
    try:
        result = run_tiny_pipeline(
            args.output_dir,
            verify_exact_resume=args.verify_exact_resume,
        )
        if _rank() == 0:
            print(json.dumps(asdict(result), sort_keys=True, allow_nan=False))
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
