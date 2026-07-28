"""Resumable Accelerate Stage-1 training for the continuous Baton planner."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from qwen35_baton.checkpoint import (
    BatonTrainingCursor,
    capture_rank_rng_state,
    load_baton_checkpoint,
    load_trusted_planner_topology,
    planner_module_topology,
    publish_trusted_planner_topology,
    restore_rank_rng_state,
    save_baton_checkpoint,
)
from qwen35_baton.config import BatonCheckpointMetadata
from qwen35_baton.hashing import sha256_artifact, sha256_file, sha256_json
from qwen35_baton.losses import BatonPlannerLoss, compute_baton_planner_loss
from qwen35_baton.ownership import (
    Stage1Ownership,
    configure_stage1_trainable_modules,
)
from qwen35_baton.sequence import ADDED_TOKENS, build_plan_text


_APPROVED_LR = 1e-5
_METRICS_SCHEMA_VERSION = 1
_METRICS_RECORD_KEYS = frozenset(
    {"schema_version", "step", "metrics", "checksum"}
)
_DURABLE_METRIC_NAMES = frozenset(
    {
        "loss/total",
        "loss/mse",
        "data_time",
        "qwen_time",
        "teacher_time",
        "query_tower_time",
        "backward_time",
        "throughput",
        "microbatches",
        *(
            f"mse/{camera}/frame_{frame}"
            for camera in ("main", "wrist")
            for frame in range(4)
        ),
    }
)


@dataclass(frozen=True)
class Stage1TrainingConfig:
    """Validated Stage-1 schedule and immutable local artifact locations."""

    output_dir: str
    qwen_model_path: str
    qwen_processor_path: str
    qwen_tokenizer_path: str
    siglip2_model_path: str
    siglip2_config_hash: str
    siglip2_artifact_hash: str
    hdf5_manifest_path: str
    hdf5_manifest_hash: str
    dataset_statistics_path: str
    per_device_batch: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 30_000
    warmup_steps: int = 1_000
    save_every: int = 5_000
    log_every: int = 20
    max_consecutive_skipped_updates: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = False
    deepspeed_config_path: str = "qwen35_baton/configs/deepspeed_zero2.json"
    num_workers: int = 4
    seed: int = 42
    resume_from: str | None = None
    tiny_test: bool = False

    def __post_init__(self) -> None:
        for name in (
            "output_dir",
            "qwen_model_path",
            "qwen_processor_path",
            "qwen_tokenizer_path",
            "siglip2_model_path",
            "siglip2_config_hash",
            "siglip2_artifact_hash",
            "hdf5_manifest_path",
            "hdf5_manifest_hash",
            "dataset_statistics_path",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        for name in (
            "per_device_batch",
            "gradient_accumulation_steps",
            "max_steps",
            "save_every",
            "log_every",
            "max_consecutive_skipped_updates",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.warmup_steps) is not int
            or self.warmup_steps < 0
            or self.warmup_steps >= self.max_steps
        ):
            raise ValueError("warmup_steps must be in [0,max_steps)")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be boolean")
        if (
            not isinstance(self.deepspeed_config_path, str)
            or not self.deepspeed_config_path
        ):
            raise ValueError("deepspeed_config_path must be a nonempty string")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.mixed_precision not in ({"no", "bf16"} if self.tiny_test else {"bf16"}):
            raise ValueError("production Stage-1 mixed_precision must be bf16")
        if self.gradient_clip_norm != 1.0:
            raise ValueError("Stage-1 gradient clipping norm must be exactly 1.0")
        if not self.tiny_test and (
            self.max_steps != 30_000 or self.save_every != 5_000
        ):
            raise ValueError(
                "production Stage-1 cadence must be 30000 steps with saves every 5000"
            )
        if self.learning_rate != _APPROVED_LR:
            raise ValueError("Stage-1 learning rate must be exactly 1e-5")
        for name in (
            "hdf5_manifest_hash",
            "siglip2_config_hash",
            "siglip2_artifact_hash",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Stage1TrainingConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("Stage-1 config must contain an object")
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(payload).difference(known))
        if unknown:
            raise ValueError(
                "unknown Stage-1 config fields: " + ", ".join(unknown)
            )
        return cls(**dict(payload))

    @classmethod
    def from_json(cls, path: str | Path) -> "Stage1TrainingConfig":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Stage-1 config is invalid: {path}") from error
        return cls.from_mapping(payload)


@dataclass
class Stage1TrainingArtifacts:
    """Injected or production planner, teacher, data, and mutable optimizer state."""

    planner: nn.Module
    teacher: Any
    train_batches: Iterable[Any]
    optimizer: torch.optim.Optimizer
    scheduler: Any
    metadata: BatonCheckpointMetadata
    ownership: Stage1Ownership


@dataclass(frozen=True)
class Stage1TrainingResult:
    global_step: int
    cursor: BatonTrainingCursor
    checkpoint: Path | None
    last_metrics: Mapping[str, float]


def validate_global_batch(
    *,
    per_device_batch: int,
    world_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """Report the effective batch without silently changing its factors."""

    values = {
        "per_device_batch": per_device_batch,
        "world_size": world_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    if any(type(value) is not int or value <= 0 for value in values.values()):
        raise ValueError("global batch factors must be positive integers")
    effective = per_device_batch * world_size * gradient_accumulation_steps
    return effective


def require_stage1_global_batch(
    *,
    per_device_batch: int,
    world_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """Require the user-selected production Stage-1 global batch."""

    effective = validate_global_batch(
        per_device_batch=per_device_batch,
        world_size=world_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    if effective != 128:
        raise ValueError(
            f"production Stage-1 global batch must be exactly 128, got {effective}"
        )
    return effective


def checkpoint_steps(*, max_steps: int, save_every: int) -> tuple[int, ...]:
    if type(max_steps) is not int or type(save_every) is not int:
        raise TypeError("checkpoint cadence values must be integers")
    if max_steps <= 0 or save_every <= 0 or max_steps % save_every:
        raise ValueError("max_steps must be a positive multiple of save_every")
    return tuple(range(save_every, max_steps + 1, save_every))


def resolve_deepspeed_runtime_config(
    config: Stage1TrainingConfig,
    *,
    world_size: int,
) -> dict[str, Any]:
    """Resolve Accelerate's ``auto`` batch fields before model preparation."""

    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    try:
        payload = json.loads(
            Path(config.deepspeed_config_path).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"DeepSpeed config is invalid: {config.deepspeed_config_path}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("DeepSpeed config must contain an object")
    payload["train_micro_batch_size_per_gpu"] = config.per_device_batch
    payload["gradient_accumulation_steps"] = config.gradient_accumulation_steps
    payload["train_batch_size"] = (
        config.per_device_batch
        * world_size
        * config.gradient_accumulation_steps
    )
    return payload


def _owned_parameters(modules: tuple[nn.Module, ...]) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) in seen:
                raise ValueError("Stage-1 optimizer ownership contains overlap")
            seen.add(id(parameter))
            if not parameter.requires_grad:
                raise ValueError("Stage-1 owned parameter is unexpectedly frozen")
            parameters.append(parameter)
    if not parameters:
        raise ValueError("Stage-1 optimizer groups must be nonempty")
    return parameters


def build_stage1_optimizer_groups(
    planner: nn.Module,
    ownership: Stage1Ownership,
    config: Stage1TrainingConfig,
) -> list[dict[str, Any]]:
    """Build one exhaustive parameter group for the full VA-Planner."""

    if not isinstance(planner, nn.Module):
        raise TypeError("planner must be a torch module")
    if not isinstance(ownership, Stage1Ownership):
        raise TypeError("ownership must be Stage1Ownership")
    groups = [{
        "name": "va_planner",
        "params": _owned_parameters(ownership.trainable_modules),
        "lr": config.learning_rate,
        "initial_lr": config.learning_rate,
    }]
    grouped = [id(parameter) for group in groups for parameter in group["params"]]
    trainable = [
        id(parameter) for parameter in planner.parameters() if parameter.requires_grad
    ]
    if len(grouped) != len(set(grouped)) or set(grouped) != set(trainable):
        raise ValueError(
            "Stage-1 optimizer ownership must be duplicate-free and exhaustive"
        )
    canonical_names = {
        id(parameter): name for name, parameter in planner.named_parameters()
    }
    for group in groups:
        names = [canonical_names.get(id(parameter)) for parameter in group["params"]]
        if any(name is None for name in names):
            raise ValueError(
                "Stage-1 optimizer parameters must have canonical names"
            )
        group["parameter_names"] = names
        group["parameter_shapes"] = [
            list(parameter.shape) for parameter in group["params"]
        ]
        group["parameter_dtypes"] = [
            str(parameter.dtype) for parameter in group["params"]
        ]
    return groups


def build_stage1_optimizer(
    planner: nn.Module,
    ownership: Stage1Ownership,
    config: Stage1TrainingConfig,
) -> torch.optim.AdamW:
    """Construct the Baton AdamW optimizer for the full VA-Planner."""

    return torch.optim.AdamW(
        build_stage1_optimizer_groups(planner, ownership, config),
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
    )


def _cosine_warmup_multiplier(
    step: int, *, warmup_steps: int, max_steps: int
) -> float:
    if warmup_steps and step <= warmup_steps:
        return float(step) / float(warmup_steps)
    progress = min(
        1.0,
        max(0.0, float(step - warmup_steps) / float(max_steps - warmup_steps)),
    )
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Stage1TrainingConfig,
) -> "BatonCosineWarmupScheduler":
    return BatonCosineWarmupScheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        max_consecutive_skipped_updates=config.max_consecutive_skipped_updates,
    )


class BatonCosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    """LambdaLR whose serialized state includes the immutable schedule recipe."""

    SCHEDULE_TYPE = "linear_warmup_cosine_v1"

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        max_steps: int,
        max_consecutive_skipped_updates: int = 8,
    ) -> None:
        self.baton_contract = {
            "schedule_type": self.SCHEDULE_TYPE,
            "warmup_steps": warmup_steps,
            "max_steps": max_steps,
            "max_consecutive_skipped_updates": max_consecutive_skipped_updates,
            "base_lrs": [
                float(group.get("initial_lr", group["lr"]))
                for group in optimizer.param_groups
            ],
        }
        super().__init__(
            optimizer,
            lambda step: _cosine_warmup_multiplier(
                step,
                warmup_steps=warmup_steps,
                max_steps=max_steps,
            ),
        )


class EpochSeededRandomSampler(torch.utils.data.Sampler[int]):
    """Reconstruct every sample permutation from the saved seed and epoch."""

    def __init__(self, data_source: Any, *, seed: int) -> None:
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("sampler epoch must be non-negative")
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.data_source)


def _artifact_hash(path: Path) -> str:
    return sha256_artifact(path)


def _load_transformer_components(config: Stage1TrainingConfig) -> dict[str, Any]:
    from transformers import (
        AutoImageProcessor,
        AutoModel,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.qwen_tokenizer_path, local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(
        config.qwen_processor_path, local_files_only=True
    )
    qwen = AutoModelForImageTextToText.from_pretrained(
        config.qwen_model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    siglip_processor = AutoImageProcessor.from_pretrained(
        config.siglip2_model_path,
        local_files_only=True,
        use_fast=False,
    )
    siglip = AutoModel.from_pretrained(
        config.siglip2_model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    return {
        "tokenizer": tokenizer,
        "processor": processor,
        "qwen": qwen,
        "siglip_processor": siglip_processor,
        "siglip": siglip,
    }


def load_local_artifacts(
    config: Stage1TrainingConfig,
    *,
    models_only: bool = False,
) -> Stage1TrainingArtifacts | Mapping[str, Any]:
    """Load only persisted local artifacts; no Hub/network fallback is permitted."""

    siglip_path = Path(config.siglip2_model_path)
    if sha256_file(siglip_path / "config.json") != config.siglip2_config_hash:
        raise ValueError("SigLIP2 config hash mismatch before model loading")
    if sha256_artifact(siglip_path) != config.siglip2_artifact_hash:
        raise ValueError("SigLIP2 artifact hash mismatch before model loading")
    components = _load_transformer_components(config)
    if models_only:
        return components
    from qwen35_baton.data import BatonLiberoDataset, BatonPlannerCollator
    from qwen35_baton.model import BatonQwen35Planner
    from qwen35_baton.teacher import FrozenSiglip2Teacher
    from ge_act.data.libero_fastwam_hdf5_dataset import (
        LiberoFastWAMHDF5Dataset,
    )

    tokenizer = components["tokenizer"]
    processor = components["processor"]
    try:
        processor.tokenizer = tokenizer
    except (AttributeError, TypeError):
        if getattr(processor, "tokenizer", None) is not tokenizer:
            raise ValueError("persisted Qwen processor did not accept its tokenizer")
    token_ids = tuple(
        int(tokenizer.convert_tokens_to_ids(token)) for token in ADDED_TOKENS
    )
    if len(set(token_ids)) != len(ADDED_TOKENS) or min(token_ids) < 0:
        raise ValueError("persisted Baton tokens do not map to seven unique IDs")
    planner = BatonQwen35Planner(components["qwen"], added_token_ids=token_ids)
    siglip = components["siglip"]
    vision_model = getattr(siglip, "vision_model", None)
    if not isinstance(vision_model, nn.Module):
        raise ValueError("local SigLIP2 artifact does not expose vision_model")
    teacher = FrozenSiglip2Teacher.from_components(
        processor=components["siglip_processor"],
        vision_model=vision_model,
        dtype=torch.bfloat16,
    )
    base_dataset = LiberoFastWAMHDF5Dataset(
        config.hdf5_manifest_path,
        config.dataset_statistics_path,
        train_dataset=True,
    )
    dataset = BatonLiberoDataset(base_dataset, seed=config.seed)
    sampler = EpochSeededRandomSampler(dataset, seed=config.seed)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_batches = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.per_device_batch,
        sampler=sampler,
        collate_fn=BatonPlannerCollator(processor),
        num_workers=config.num_workers,
        drop_last=True,
        generator=generator,
        persistent_workers=config.num_workers > 0,
    )
    if len(train_batches) <= 0:
        raise ValueError("Stage-1 dataset must yield at least one complete microbatch")
    ownership = configure_stage1_trainable_modules(planner)
    optimizer = build_stage1_optimizer(planner, ownership, config)
    scheduler = build_cosine_warmup_scheduler(optimizer, config)
    example = BatonCheckpointMetadata.example()
    metadata = replace(
        example,
        qwen_config_hash=sha256_file(Path(config.qwen_model_path) / "config.json"),
        tokenizer_hash=_artifact_hash(Path(config.qwen_tokenizer_path)),
        processor_hash=_artifact_hash(Path(config.qwen_processor_path)),
        input_template_hash=sha256_json(build_plan_text("{instruction}")),
        added_token_ids=token_ids,
        siglip2_config_hash=config.siglip2_config_hash,
        siglip2_artifact_hash=config.siglip2_artifact_hash,
        teacher_preprocessing_hash=_artifact_hash(Path(config.siglip2_model_path)),
        hdf5_manifest_hash=config.hdf5_manifest_hash,
    )
    return Stage1TrainingArtifacts(
        planner=planner,
        teacher=teacher,
        train_batches=train_batches,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        ownership=ownership,
    )


def _move_batch(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return type(batch)(
            (key, _move_batch(value, device)) for key, value in batch.items()
        )
    if isinstance(batch, tuple) and hasattr(batch, "_fields"):
        return type(batch)(*(_move_batch(value, device) for value in batch))
    if isinstance(batch, tuple):
        return tuple(_move_batch(value, device) for value in batch)
    if isinstance(batch, list):
        return [_move_batch(value, device) for value in batch]
    if hasattr(batch, "__dataclass_fields__"):
        return type(batch)(
            **{
                name: _move_batch(getattr(batch, name), device)
                for name in batch.__dataclass_fields__
            }
        )
    return batch


def _set_dataloader_epoch(batches: Any, *, epoch: int, sampler_seed: int) -> None:
    set_epoch = getattr(batches, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(epoch)
    dataset = getattr(batches, "dataset", None)
    dataset_set_epoch = getattr(dataset, "set_epoch", None)
    if callable(dataset_set_epoch):
        dataset_set_epoch(epoch)
    sampler = getattr(batches, "sampler", None)
    sampler_set_epoch = getattr(sampler, "set_epoch", None)
    if callable(sampler_set_epoch):
        sampler_set_epoch(epoch)
    generator = getattr(batches, "generator", None)
    if isinstance(generator, torch.Generator):
        generator.manual_seed(sampler_seed + epoch)


def _advance_cursor(
    cursor: BatonTrainingCursor,
    *,
    epoch: int,
    consumed_microbatches: int,
    global_step: int,
) -> BatonTrainingCursor:
    if consumed_microbatches == cursor.microbatches_per_epoch:
        epoch += 1
        consumed_microbatches = 0
    return BatonTrainingCursor(
        global_step=global_step,
        epoch=epoch,
        consumed_microbatches=consumed_microbatches,
        microbatches_per_epoch=cursor.microbatches_per_epoch,
        sampler_seed=cursor.sampler_seed,
    )


def _optimizer_for_checkpoint(optimizer: Any) -> torch.optim.Optimizer:
    inner = getattr(optimizer, "optimizer", optimizer)
    if not isinstance(inner, torch.optim.Optimizer):
        raise TypeError("prepared optimizer does not expose a torch optimizer")
    return inner


def _scheduler_for_checkpoint(scheduler: Any) -> Any:
    return getattr(scheduler, "scheduler", scheduler)


def _rank_states(accelerator: Any) -> dict[int, Mapping[str, Any]]:
    local = capture_rank_rng_state(distributed_rank=accelerator.process_index)
    if accelerator.num_processes == 1:
        return {0: local}
    from accelerate.utils import gather_object

    gathered = gather_object([local])
    states = {
        int(state["distributed_rank"]): state
        for state in gathered
        if isinstance(state, Mapping)
    }
    if sorted(states) != list(range(accelerator.num_processes)):
        raise RuntimeError("not every distributed rank published its RNG state")
    return states


def _save_training_checkpoint(
    *,
    accelerator: Any,
    config: Stage1TrainingConfig,
    artifacts: Stage1TrainingArtifacts,
    planner: nn.Module,
    optimizer: Any,
    scheduler: Any,
    cursor: BatonTrainingCursor,
) -> Path:
    accelerator.wait_for_everyone()
    states = _rank_states(accelerator)
    destination = Path(config.output_dir) / f"step_{cursor.global_step:06d}"
    if accelerator.is_main_process:
        save_baton_checkpoint(
            destination,
            planner=accelerator.unwrap_model(planner),
            optimizer=_optimizer_for_checkpoint(optimizer),
            scheduler=_scheduler_for_checkpoint(scheduler),
            scaler=getattr(accelerator, "scaler", None),
            metadata=artifacts.metadata,
            cursor=cursor,
            rank_rng_state=states,
        )
    accelerator.wait_for_everyone()
    return destination


def _loss_metrics(
    losses: BatonPlannerLoss,
    *,
    positive: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    metrics = {
        "loss/total": float(losses.total.detach().float().cpu()),
        "loss/mse": float(losses.mse.detach().float().cpu()),
    }
    positive_math = positive.detach().float()
    target_math = target.detach().float()
    camera_names = ("main", "wrist")
    for camera in range(positive.shape[1]):
        camera_name = (
            camera_names[camera] if camera < len(camera_names) else f"camera_{camera}"
        )
        for frame in range(positive.shape[2]):
            predicted = positive_math[:, camera, frame]
            teacher = target_math[:, camera, frame]
            metrics[f"mse/{camera_name}/frame_{frame}"] = float(
                (predicted - teacher).square().mean().cpu()
            )
    return metrics


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _configure_gradient_checkpointing(
    planner: nn.Module,
    *,
    enabled: bool,
) -> None:
    backbone = getattr(planner, "backbone", None)
    if not isinstance(backbone, nn.Module):
        raise ValueError("Baton planner must expose its Qwen backbone")
    method_name = (
        "gradient_checkpointing_enable"
        if enabled
        else "gradient_checkpointing_disable"
    )
    method = getattr(backbone, method_name, None)
    if not callable(method):
        raise RuntimeError(
            f"Qwen backbone does not expose public {method_name}()"
        )
    method()


def _average_metrics(
    accelerator: Any,
    sums: Mapping[str, float],
    *,
    microbatches: int,
) -> dict[str, float]:
    if microbatches <= 0:
        raise ValueError("metric window must contain at least one microbatch")
    names = tuple(sorted(sums))
    values = torch.tensor(
        [sums[name] / microbatches for name in names],
        dtype=torch.float64,
        device=accelerator.device,
    )
    values = accelerator.reduce(values, reduction="mean")
    return {
        name: float(value)
        for name, value in zip(names, values.detach().cpu().tolist())
    }


def _durable_metrics_record(
    *, step: int, metrics: Mapping[str, float]
) -> dict[str, Any]:
    if type(step) is not int or step <= 0:
        raise ValueError("durable metric step must be a positive integer")
    if not isinstance(metrics, Mapping) or set(metrics) != _DURABLE_METRIC_NAMES:
        raise ValueError("durable metric names differ from the Stage-1 contract")
    if any(
        not isinstance(name, str)
        or type(value) not in (int, float)
        or not math.isfinite(float(value))
        for name, value in metrics.items()
    ):
        raise ValueError("durable metric values must be finite numbers")
    unsigned = {
        "schema_version": _METRICS_SCHEMA_VERSION,
        "step": step,
        "metrics": {
            name: float(metrics[name])
            for name in sorted(_DURABLE_METRIC_NAMES)
        },
    }
    return {
        **unsigned,
        "checksum": sha256_json(unsigned),
    }


def _validated_durable_metrics_record(
    record: Any,
) -> dict[str, Any] | None:
    if not isinstance(record, Mapping) or set(record) != _METRICS_RECORD_KEYS:
        return None
    schema_version = record.get("schema_version")
    step = record.get("step")
    metrics = record.get("metrics")
    checksum = record.get("checksum")
    if (
        type(schema_version) is not int
        or schema_version != _METRICS_SCHEMA_VERSION
        or type(step) is not int
        or step <= 0
        or not isinstance(metrics, Mapping)
        or set(metrics) != _DURABLE_METRIC_NAMES
        or any(
            not isinstance(name, str)
            or type(value) is not float
            or not math.isfinite(value)
            for name, value in metrics.items()
        )
        or not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        return None
    unsigned = {
        "schema_version": schema_version,
        "step": step,
        "metrics": dict(metrics),
    }
    if checksum != sha256_json(unsigned):
        return None
    return {
        **unsigned,
        "checksum": checksum,
    }


def _append_metrics(path: Path, *, step: int, metrics: Mapping[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _durable_metrics_record(step=step, metrics=metrics)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _reconcile_metrics(path: Path, *, completed_step: int) -> None:
    """Atomically retain one valid record per completed checkpoint step."""

    if type(completed_step) is not int or completed_step < 0:
        raise ValueError("completed metric step must be a non-negative integer")
    if not path.exists():
        return
    canonical: dict[int, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Stage-1 metrics JSONL is unreadable: {path}") from error
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        validated = _validated_durable_metrics_record(record)
        if validated is None:
            continue
        step = validated["step"]
        if step > completed_step:
            continue
        previous = canonical.get(step)
        if previous is not None:
            if previous == validated:
                continue
            raise ValueError(
                f"conflicting integrity-valid metrics records for step {step}"
            )
        canonical[step] = validated

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.reconcile-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for step in sorted(canonical):
                stream.write(
                    json.dumps(
                        canonical[step],
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_training(
    config: Stage1TrainingConfig,
    *,
    artifacts: Stage1TrainingArtifacts | None = None,
    stop_at_step: int | None = None,
) -> Stage1TrainingResult:
    """Run optimizer steps without manually dividing the loss under Accelerate."""

    if not isinstance(config, Stage1TrainingConfig):
        raise TypeError("config must be Stage1TrainingConfig")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # Model constructors consume RNG too, so seed before creating any artifact.
    _seed_everything(config.seed)
    if artifacts is None:
        # This CPU-only preflight is deliberately before Accelerator/GPU setup.
        from qwen35_baton.cli.preflight import preflight_stage1

        preflight_stage1(config.to_dict(), world_size=world_size)
        artifacts = load_local_artifacts(config)
        assert isinstance(artifacts, Stage1TrainingArtifacts)
    if not config.tiny_test:
        _configure_gradient_checkpointing(
            artifacts.planner,
            enabled=config.gradient_checkpointing,
        )

    from accelerate import Accelerator
    from accelerate.utils import DeepSpeedPlugin, GradientAccumulationPlugin

    deepspeed_plugin = None
    if not config.tiny_test:
        deepspeed_runtime_config = resolve_deepspeed_runtime_config(
            config,
            world_size=world_size,
        )
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=deepspeed_runtime_config,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            gradient_clipping=config.gradient_clip_norm,
            zero_stage=2,
        )

    accelerator = Accelerator(
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=config.gradient_accumulation_steps,
            sync_with_dataloader=False,
        ),
        mixed_precision=config.mixed_precision,
        cpu=config.tiny_test,
        step_scheduler_with_optimizer=False,
        deepspeed_plugin=deepspeed_plugin,
    )
    if not config.tiny_test:
        require_stage1_global_batch(
            per_device_batch=config.per_device_batch,
            world_size=accelerator.num_processes,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
    planner_topology = planner_module_topology(artifacts.planner)
    planner_topology_hash = sha256_json(planner_topology)
    output_topology_path = Path(config.output_dir) / "planner_topology.json"
    resume_topology_path = (
        None
        if config.resume_from is None
        else Path(config.resume_from).parent / "planner_topology.json"
    )
    if resume_topology_path is not None:
        resumed_topology, resumed_hash = load_trusted_planner_topology(
            resume_topology_path
        )
        if resumed_topology != planner_topology:
            raise ValueError(
                "resume root trusted planner topology differs from runtime planner"
            )
        if resumed_hash != planner_topology_hash:
            raise ValueError("resume root planner topology hash is invalid")
    if accelerator.is_main_process:
        if output_topology_path.is_file():
            output_topology, output_hash = load_trusted_planner_topology(
                output_topology_path
            )
            if (
                output_topology != planner_topology
                or output_hash != planner_topology_hash
            ):
                raise ValueError(
                    "output root trusted planner topology differs from runtime planner"
                )
        else:
            publish_trusted_planner_topology(
                output_topology_path,
                planner_topology,
            )
    accelerator.wait_for_everyone()
    artifacts.metadata = replace(
        artifacts.metadata,
        planner_topology_hash=planner_topology_hash,
    )
    train_batches = artifacts.train_batches
    if isinstance(train_batches, torch.utils.data.DataLoader):
        train_batches = accelerator.prepare_data_loader(train_batches)
    microbatches_per_epoch = len(train_batches)  # type: ignore[arg-type]
    if microbatches_per_epoch <= 0:
        raise ValueError("Stage-1 training requires a nonempty loader")
    cursor = BatonTrainingCursor(
        global_step=0,
        epoch=0,
        consumed_microbatches=0,
        microbatches_per_epoch=microbatches_per_epoch,
        sampler_seed=config.seed,
    )
    metrics_path = Path(config.output_dir) / "training_metrics.jsonl"
    if config.resume_from is not None:
        resumed = load_baton_checkpoint(
            Path(config.resume_from),
            planner=artifacts.planner,
            optimizer=artifacts.optimizer,
            scheduler=artifacts.scheduler,
            scaler=getattr(accelerator, "scaler", None),
            expected_contract=artifacts.metadata,
            expected_sampler_seed=config.seed,
            expected_microbatches_per_epoch=microbatches_per_epoch,
            expected_planner_topology=resume_topology_path,
            distributed_rank=accelerator.process_index,
            world_size=accelerator.num_processes,
        )
        cursor = resumed.cursor
        if accelerator.is_main_process:
            _reconcile_metrics(metrics_path, completed_step=cursor.global_step)
        accelerator.wait_for_everyone()
    elif metrics_path.exists():
        raise FileExistsError(
            f"stale Stage-1 metrics file exists for fresh training: {metrics_path}"
        )
    planner, optimizer = accelerator.prepare(
        artifacts.planner,
        artifacts.optimizer,
    )
    scheduler = artifacts.scheduler
    teacher_model = getattr(artifacts.teacher, "model", None)
    teacher_to = getattr(artifacts.teacher, "to", None)
    if callable(teacher_to):
        teacher_to(accelerator.device)
    elif isinstance(teacher_model, nn.Module):
        teacher_model.to(accelerator.device)
        teacher_model.requires_grad_(False)
        teacher_model.eval()
    target_step = config.max_steps if stop_at_step is None else stop_at_step
    if (
        type(target_step) is not int
        or target_step <= cursor.global_step
        or target_step > config.max_steps
    ):
        raise ValueError("stop_at_step must be above resume and at most max_steps")

    planner.train()
    last_checkpoint: Path | None = None
    last_metrics: dict[str, float] = {}
    last_batch_end = time.perf_counter()
    window_started = last_batch_end
    window_sums: dict[str, float] = {}
    window_microbatches = 0
    consecutive_skipped_updates = 0
    while cursor.global_step < target_step:
        epoch = cursor.epoch
        _set_dataloader_epoch(
            train_batches,
            epoch=epoch,
            sampler_seed=cursor.sampler_seed,
        )
        iterator = iter(train_batches)
        skipped = cursor.consumed_microbatches
        if skipped:
            restored_rng = capture_rank_rng_state(
                distributed_rank=accelerator.process_index
            )
            for _ in range(skipped):
                try:
                    next(iterator)
                except StopIteration as error:
                    raise ValueError("resume cursor exceeds the training loader") from error
            restore_rank_rng_state(restored_rng)
        for offset, raw_batch in enumerate(iterator, start=skipped + 1):
            if cursor.global_step >= target_step:
                break
            data_ready = time.perf_counter()
            batch = _move_batch(raw_batch, accelerator.device)
            data_time = data_ready - last_batch_end
            if window_microbatches == 0:
                window_started = last_batch_end
            with accelerator.accumulate(planner):
                _synchronize_device(accelerator.device)
                planner_start = time.perf_counter()
                query_elapsed = [0.0]
                query_started = [0.0]
                query_tower = getattr(
                    accelerator.unwrap_model(planner), "query_tower", None
                )
                handles = []
                if isinstance(query_tower, nn.Module):
                    def _query_pre_hook(_module: nn.Module, _inputs: Any) -> None:
                        _synchronize_device(accelerator.device)
                        query_started[0] = time.perf_counter()

                    def _query_hook(
                        _module: nn.Module, _inputs: Any, _output: Any
                    ) -> None:
                        _synchronize_device(accelerator.device)
                        query_elapsed[0] = time.perf_counter() - query_started[0]

                    handles = [
                        query_tower.register_forward_pre_hook(_query_pre_hook),
                        query_tower.register_forward_hook(_query_hook),
                    ]
                try:
                    planner_output = planner(batch)
                finally:
                    for handle in handles:
                        handle.remove()
                _synchronize_device(accelerator.device)
                planner_time = time.perf_counter() - planner_start
                teacher_start = time.perf_counter()
                with torch.no_grad():
                    future_teacher = artifacts.teacher.encode_future(
                        batch.future_images
                    )
                _synchronize_device(accelerator.device)
                teacher_time = time.perf_counter() - teacher_start
                if not all(
                    bool(torch.isfinite(value).all())
                    for value in (
                        planner_output.positive,
                        future_teacher,
                    )
                ):
                    raise FloatingPointError(
                        "Stage-1 predictions or teacher targets are nonfinite"
                    )
                # Keep strict standalone loss validation while doing all Stage-1
                # regression math in stable fp32 across bf16 model boundaries.
                positive_for_loss = planner_output.positive.float()
                future_for_loss = future_teacher.float()
                losses = compute_baton_planner_loss(
                    positive_for_loss,
                    future_for_loss,
                )
                if not all(
                    bool(torch.isfinite(value).all())
                    for value in (
                        losses.total,
                        losses.mse,
                    )
                ):
                    raise FloatingPointError("Stage-1 loss is nonfinite")
                backward_start = time.perf_counter()
                # Accelerate performs accumulation normalization exactly once.
                accelerator.backward(losses.total)
                _synchronize_device(accelerator.device)
                backward_time = time.perf_counter() - backward_start
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        planner.parameters(), config.gradient_clip_norm
                    )
                optimizer.step()
                synchronized = bool(accelerator.sync_gradients)
                completed_update = synchronized and not bool(
                    accelerator.optimizer_step_was_skipped
                )
                if completed_update:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            next_step = cursor.global_step + int(completed_update)
            cursor = _advance_cursor(
                cursor,
                epoch=epoch,
                consumed_microbatches=offset,
                global_step=next_step,
            )
            if synchronized:
                if completed_update:
                    consecutive_skipped_updates = 0
                else:
                    consecutive_skipped_updates += 1
                    if (
                        consecutive_skipped_updates
                        >= config.max_consecutive_skipped_updates
                    ):
                        raise FloatingPointError(
                            "Stage-1 aborted after "
                            f"{consecutive_skipped_updates} consecutive synchronized "
                            "optimizer updates were skipped; "
                            f"global_step={cursor.global_step}, epoch={cursor.epoch}, "
                            "consumed_microbatches="
                            f"{cursor.consumed_microbatches}"
                        )
            micro_metrics = _loss_metrics(
                losses,
                positive=planner_output.positive,
                target=future_teacher,
            )
            micro_metrics.update(
                {
                    "data_time": data_time,
                    "qwen_time": max(planner_time - query_elapsed[0], 0.0),
                    "teacher_time": teacher_time,
                    "query_tower_time": query_elapsed[0],
                    "backward_time": backward_time,
                }
            )
            for name, value in micro_metrics.items():
                window_sums[name] = window_sums.get(name, 0.0) + value
            window_microbatches += 1
            if completed_update:
                _synchronize_device(accelerator.device)
                elapsed = max(time.perf_counter() - window_started, 1e-12)
                last_metrics = _average_metrics(
                    accelerator,
                    window_sums,
                    microbatches=window_microbatches,
                )
                gathered_elapsed = accelerator.gather(
                    torch.tensor(
                        [elapsed],
                        dtype=torch.float64,
                        device=accelerator.device,
                    )
                )
                reduced_elapsed = float(
                    gathered_elapsed.detach().max().cpu()
                )
                reduced_microbatches = float(
                    accelerator.reduce(
                        torch.tensor(
                            float(window_microbatches),
                            dtype=torch.float64,
                            device=accelerator.device,
                        ),
                        reduction="mean",
                    )
                    .detach()
                    .cpu()
                )
                last_metrics.update(
                    {
                        "throughput": (
                            config.per_device_batch
                            * accelerator.num_processes
                            * reduced_microbatches
                            / max(reduced_elapsed, 1e-12)
                        ),
                        "microbatches": reduced_microbatches,
                    }
                )
                if cursor.global_step % config.log_every == 0:
                    if accelerator.is_main_process:
                        _append_metrics(
                            metrics_path,
                            step=cursor.global_step,
                            metrics=last_metrics,
                        )
                if cursor.global_step % config.save_every == 0:
                    last_checkpoint = _save_training_checkpoint(
                        accelerator=accelerator,
                        config=config,
                        artifacts=artifacts,
                        planner=planner,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        cursor=cursor,
                    )
            if synchronized:
                window_sums = {}
                window_microbatches = 0
            last_batch_end = time.perf_counter()
            if cursor.global_step >= target_step:
                break
        else:
            continue
    return Stage1TrainingResult(
        global_step=cursor.global_step,
        cursor=cursor,
        checkpoint=last_checkpoint,
        last_metrics=last_metrics,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--per-device-batch", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--deepspeed-config-path", type=str)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--resume-from", type=str)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = Stage1TrainingConfig.from_json(args.config)
    overrides = {}
    if args.per_device_batch is not None:
        overrides["per_device_batch"] = args.per_device_batch
    if args.gradient_accumulation_steps is not None:
        overrides["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.deepspeed_config_path is not None:
        overrides["deepspeed_config_path"] = args.deepspeed_config_path
    if args.gradient_checkpointing is not None:
        overrides["gradient_checkpointing"] = args.gradient_checkpointing
    if args.resume_from is not None:
        overrides["resume_from"] = args.resume_from
    if overrides:
        config = replace(config, **overrides)
    result = run_training(config)
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            json.dumps(
                {
                    "global_step": result.global_step,
                    "checkpoint": (
                        None if result.checkpoint is None else str(result.checkpoint)
                    ),
                    "cursor": result.cursor.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
