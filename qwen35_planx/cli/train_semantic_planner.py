#!/usr/bin/env python3
"""Distributed training and self-contained export for the grounded planner."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any

import numpy as np
from safetensors import safe_open
from safetensors.torch import load_model, save_file, save_model
import torch
from torch import nn
import torch.nn.functional as F

from qwen35_planx.config import GroundedPlannerMetadata
from qwen35_planx.hashing import sha256_file, sha256_json
from qwen35_planx.planner_dataset import GroundedPlannerBatch


_GROUP_LRS = {
    "qwen_language": 1e-5,
    "qwen_vision": 5e-6,
    "visual_vocab_and_prediction_head": 1e-4,
    "semantic_phrase_grounding_fusion_heads": 1e-4,
}
_SEMANTIC_HEAD_NAMES = (
    "semantic_projection",
    "phrase_projection",
    "grounding_query",
    "fusion_gate",
)
_VISION_NAME_PARTS = (
    "vision_model",
    "visual",
    "vision_tower",
    "image_encoder",
)
REQUIRED_CHECKPOINT_ENTRIES = (
    "planner.safetensors",
    "ta_codebook.safetensors",
    "ta_codebook.json",
    "processor",
    "tokenizer",
    "model_config",
    "planner_meta.json",
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "rng_state.pt",
    "trainer_state.json",
)


@dataclass(frozen=True)
class PlannerTrainingConfig:
    """Validated stage-one configuration shared by CLI and preflight."""

    output_dir: str
    base_model: str | None = None
    hdf5_manifest: str | None = None
    hindsight_cache: str | None = None
    ta_codebook: str | None = None
    ta_codebook_metadata: str | None = None
    resume_from: str | None = None
    tiny_smoke: bool = False
    per_device_batch: int = 4
    gradient_accumulation_steps: int = 8
    max_steps: int = 30_000
    warmup_steps: int = 1_000
    save_every: int = 5_000
    validate_every: int = 5_000
    log_every: int = 20
    qwen_language_lr: float = 1e-5
    qwen_vision_lr: float = 5e-6
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    mixed_precision: str = "bf16"
    tf32: bool = True
    activation_checkpointing: bool = True
    num_workers: int = 4
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.output_dir:
            raise ValueError("output_dir must not be empty")
        for name in (
            "per_device_batch",
            "gradient_accumulation_steps",
            "max_steps",
            "save_every",
            "validate_every",
            "log_every",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("warmup_steps must be in [0,max_steps)")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        if self.mixed_precision != "bf16":
            raise ValueError("planner training mixed_precision must be bf16")
        if self.gradient_clip_norm != 1.0:
            raise ValueError("planner gradient clipping norm must be exactly 1.0")
        if not self.tf32:
            raise ValueError("planner training requires TF32")
        for name in ("qwen_language_lr", "qwen_vision_lr", "head_lr"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        configured_lrs = {
            "qwen_language": self.qwen_language_lr,
            "qwen_vision": self.qwen_vision_lr,
            "visual_vocab_and_prediction_head": self.head_lr,
            "semantic_phrase_grounding_fusion_heads": self.head_lr,
        }
        if configured_lrs != _GROUP_LRS:
            raise ValueError("planner optimizer learning rates must match Task 8")
        if not self.tiny_smoke:
            missing = [
                name
                for name in (
                    "base_model",
                    "hdf5_manifest",
                    "hindsight_cache",
                    "ta_codebook",
                    "ta_codebook_metadata",
                )
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError(
                    "production planner config is missing: " + ", ".join(missing)
                )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PlannerTrainingConfig":
        if not isinstance(payload, Mapping):
            raise TypeError("planner training config must contain an object")
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(payload).difference(known))
        if unknown:
            raise ValueError(
                "unknown planner training config fields: " + ", ".join(unknown)
            )
        return cls(**dict(payload))


def cosine_lr_multiplier(
    step: int,
    *,
    warmup_steps: int,
    max_steps: int,
) -> float:
    """Linear warmup followed by a zero-ending cosine multiplier."""

    if type(step) is not int or step < 0:
        raise ValueError("scheduler step must be a non-negative integer")
    if (
        type(warmup_steps) is not int
        or type(max_steps) is not int
        or warmup_steps < 0
        or max_steps <= warmup_steps
    ):
        raise ValueError("scheduler requires 0 <= warmup_steps < max_steps")
    if warmup_steps and step < warmup_steps:
        return float(step) / float(warmup_steps)
    progress = min(
        1.0,
        max(0.0, float(step - warmup_steps) / float(max_steps - warmup_steps)),
    )
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def estimate_per_gpu_batch_candidates(
    *,
    num_processes: int,
    available_bytes: int,
    estimated_bytes_per_sample: int,
) -> tuple[tuple[int, int], ...]:
    """Return stable microbatch/accumulation pairs preserving batch 256."""

    for name, value in {
        "num_processes": num_processes,
        "available_bytes": available_bytes,
        "estimated_bytes_per_sample": estimated_bytes_per_sample,
    }.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    capacity = max(1, available_bytes // estimated_bytes_per_sample)
    candidates = []
    for per_device_batch in range(min(256, capacity), 0, -1):
        denominator = num_processes * per_device_batch
        if 256 % denominator == 0:
            candidates.append((per_device_batch, 256 // denominator))
    if not candidates:
        raise ValueError(
            f"{num_processes} processes cannot form effective global batch 256"
        )
    return tuple(candidates)


def _module_parameter_ids(module: object) -> set[int]:
    if not isinstance(module, nn.Module):
        return set()
    return {id(parameter) for parameter in module.parameters(recurse=True)}


def _input_embedding_parameter(planner: nn.Module) -> nn.Parameter | None:
    backbone = getattr(planner, "backbone", None)
    candidates = (backbone, getattr(backbone, "model", None))
    for candidate in candidates:
        getter = getattr(candidate, "get_input_embeddings", None)
        if callable(getter):
            embedding = getter()
            weight = getattr(embedding, "weight", None)
            if isinstance(weight, nn.Parameter):
                return weight
    value = getattr(planner, "visual_embedding_weight", None)
    return value if isinstance(value, nn.Parameter) else None


def _mask_base_embedding_rows(
    parameter: nn.Parameter,
    *,
    first_trainable_row: int,
) -> None:
    if parameter.ndim != 2 or not 0 <= first_trainable_row < parameter.shape[0]:
        raise ValueError(
            "visual token start must address a row in the Qwen input embedding"
        )
    marker = (id(parameter), first_trainable_row)
    installed = getattr(parameter, "_planx_gradient_masks", set())
    if marker in installed:
        return

    def mask(gradient: torch.Tensor) -> torch.Tensor:
        result = gradient.clone()
        result[:first_trainable_row].zero_()
        return result

    parameter.register_hook(mask)
    setattr(parameter, "_planx_gradient_masks", {*installed, marker})


def build_optimizer_groups(
    planner: nn.Module,
    *,
    visual_token_start_id: int,
    experiment_token_start_id: int | None = None,
    qwen_language_lr: float = 1e-5,
    qwen_vision_lr: float = 5e-6,
    head_lr: float = 1e-4,
) -> list[dict[str, Any]]:
    """Build the exact four exhaustive, duplicate-free Task-8 groups.

    The resized input-embedding parameter owns both base and new rows. It is
    placed in the visual-vocabulary group, while a gradient hook makes the
    group update only experiment-local rows and leaves base Qwen rows intact.
    Weight decay is disabled for this row-partitioned parameter group.
    """

    if not isinstance(planner, nn.Module):
        raise TypeError("planner must be a torch module")
    backbone = getattr(planner, "backbone", None)
    input_embedding = _input_embedding_parameter(planner)
    input_id = id(input_embedding) if input_embedding is not None else None
    if isinstance(backbone, nn.Module):
        forwarded_base = getattr(backbone, "model", None)
        forwarded_ids = _module_parameter_ids(forwarded_base)
        for name, parameter in backbone.named_parameters():
            bypassed_by_forward = (
                bool(forwarded_ids) and id(parameter) not in forwarded_ids
            )
            if (
                ("lm_head" in name or bypassed_by_forward)
                and id(parameter) != input_id
            ):
                parameter.requires_grad_(False)
    learning_rates = {
        "qwen_language": float(qwen_language_lr),
        "qwen_vision": float(qwen_vision_lr),
        "visual_vocab_and_prediction_head": float(head_lr),
        "semantic_phrase_grounding_fusion_heads": float(head_lr),
    }
    if any(not math.isfinite(value) or value <= 0 for value in learning_rates.values()):
        raise ValueError("all optimizer learning rates must be finite and positive")

    semantic_ids: set[int] = set()
    for name in _SEMANTIC_HEAD_NAMES:
        module = getattr(planner, name, None)
        if not isinstance(module, nn.Module):
            raise ValueError(f"planner is missing Task-7 head: {name}")
        semantic_ids.update(_module_parameter_ids(module))
    visual_head = getattr(planner, "visual_regression", None)
    if not isinstance(visual_head, nn.Module):
        raise ValueError("planner is missing Task-7 head: visual_regression")
    visual_ids = _module_parameter_ids(visual_head)
    input_embedding = _input_embedding_parameter(planner)
    if input_embedding is None:
        raise ValueError("planner backbone does not expose Qwen input embeddings")
    visual_ids.add(id(input_embedding))
    first_trainable_row = (
        int(visual_token_start_id)
        if experiment_token_start_id is None
        else int(experiment_token_start_id)
    )
    if first_trainable_row > int(visual_token_start_id):
        raise ValueError(
            "experiment token start cannot follow the visual token start"
        )
    _mask_base_embedding_rows(
        input_embedding,
        first_trainable_row=first_trainable_row,
    )

    grouped: dict[str, list[nn.Parameter]] = {
        name: [] for name in learning_rates
    }
    unclassified: list[str] = []
    for name, parameter in planner.named_parameters():
        if not parameter.requires_grad:
            continue
        identifier = id(parameter)
        if identifier in semantic_ids:
            group_name = "semantic_phrase_grounding_fusion_heads"
        elif identifier in visual_ids:
            group_name = "visual_vocab_and_prediction_head"
        elif name.startswith("backbone.") and any(
            part in name.lower() for part in _VISION_NAME_PARTS
        ):
            group_name = "qwen_vision"
        elif name.startswith("backbone."):
            group_name = "qwen_language"
        else:
            unclassified.append(name)
            continue
        grouped[group_name].append(parameter)
    if unclassified:
        raise ValueError(
            "unclassified trainable planner parameters: " + ", ".join(unclassified)
        )
    if any(not parameters for parameters in grouped.values()):
        empty = [name for name, parameters in grouped.items() if not parameters]
        raise ValueError("optimizer groups must not be empty: " + ", ".join(empty))

    all_identifiers = [
        id(parameter)
        for parameters in grouped.values()
        for parameter in parameters
    ]
    trainable_identifiers = {
        id(parameter)
        for parameter in planner.parameters()
        if parameter.requires_grad
    }
    if len(all_identifiers) != len(set(all_identifiers)):
        raise RuntimeError("optimizer parameter groups contain duplicates")
    if set(all_identifiers) != trainable_identifiers:
        raise RuntimeError("optimizer parameter groups do not cover every trainable parameter")

    result = []
    for name, learning_rate in learning_rates.items():
        group: dict[str, Any] = {
            "name": name,
            "params": grouped[name],
            "lr": learning_rate,
        }
        if name == "visual_vocab_and_prediction_head":
            group["weight_decay"] = 0.0
        result.append(group)
    return result


def validate_effective_global_batch(
    *,
    per_device_batch: int,
    num_processes: int,
    grad_accum: int,
) -> int:
    """Enforce the immutable stage-one effective global batch."""

    for name, value in {
        "per_device_batch": per_device_batch,
        "num_processes": num_processes,
        "grad_accum": grad_accum,
    }.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    global_batch = per_device_batch * num_processes * grad_accum
    if global_batch != 256:
        raise ValueError(f"effective global batch must be 256, got {global_batch}")
    return global_batch


def enable_selective_qwen_activation_checkpointing(planner: nn.Module) -> None:
    """Enable non-reentrant checkpointing on Qwen language blocks only."""

    backbone = getattr(planner, "backbone", None)
    candidates = (
        getattr(backbone, "language_model", None),
        getattr(getattr(backbone, "model", None), "language_model", None),
        getattr(backbone, "model", None),
    )
    language = next(
        (
            candidate
            for candidate in candidates
            if callable(getattr(candidate, "gradient_checkpointing_enable", None))
        ),
        None,
    )
    if language is None:
        raise ValueError(
            "Qwen language backbone does not support activation checkpointing"
        )
    language.gradient_checkpointing_enable(
        {"use_reentrant": False}
    )


def _is_within(path: Path, parent: Path) -> bool:
    resolved = path.resolve()
    base = parent.resolve()
    return resolved == base or base in resolved.parents


def _assert_safe_output(
    output_dir: Path,
    *,
    base_model_dir: Path | str | None,
    released_ta_dir: Path | str | None,
) -> None:
    for label, protected in (
        ("base Qwen", base_model_dir),
        ("released TA-Tok", released_ta_dir),
    ):
        if protected is not None and _is_within(output_dir, Path(protected)):
            raise ValueError(
                f"checkpoint output must not be inside the {label} directory"
            )


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _save_model_config(planner: nn.Module, output_dir: Path) -> None:
    backbone = getattr(planner, "backbone", None)
    config = getattr(backbone, "config", None)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_pretrained = getattr(config, "save_pretrained", None)
    if callable(save_pretrained):
        save_pretrained(output_dir)
        return
    _json_dump(
        output_dir / "config.json",
        {
            "architectures": [type(backbone).__name__],
            "test_only": True,
        },
    )


def _artifact_hash(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    entries = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        entries.append(
            (
                child.relative_to(path).as_posix(),
                child.stat().st_size,
                sha256_file(child),
            )
        )
    if not entries:
        raise ValueError(f"checkpoint artifact directory is empty: {path}")
    return sha256_json(entries)


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(value.view(torch.uint8).numpy())).hexdigest()


def _capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    return {
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "python_version": int(python_state[0]),
        "python_state": torch.tensor(python_state[1], dtype=torch.int64),
        "python_gauss": python_state[2],
        "numpy_bit_generator": str(numpy_state[0]),
        "numpy_state": torch.from_numpy(numpy_state[1].copy()),
        "numpy_position": int(numpy_state[2]),
        "numpy_has_gauss": int(numpy_state[3]),
        "numpy_cached_gaussian": float(numpy_state[4]),
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {
        "torch_cpu",
        "torch_cuda",
        "python_version",
        "python_state",
        "python_gauss",
        "numpy_bit_generator",
        "numpy_state",
        "numpy_position",
        "numpy_has_gauss",
        "numpy_cached_gaussian",
    }
    if set(state) != required:
        raise ValueError("checkpoint RNG state fields are invalid")
    torch.random.set_rng_state(state["torch_cpu"])
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_states)
    random.setstate(
        (
            int(state["python_version"]),
            tuple(int(value) for value in state["python_state"].tolist()),
            state["python_gauss"],
        )
    )
    np.random.set_state(
        (
            str(state["numpy_bit_generator"]),
            state["numpy_state"].cpu().numpy().astype(np.uint32, copy=False),
            int(state["numpy_position"]),
            int(state["numpy_has_gauss"]),
            float(state["numpy_cached_gaussian"]),
        )
    )


def _validate_rng_state(state: Mapping[str, Any]) -> None:
    required = {
        "torch_cpu", "torch_cuda", "python_version", "python_state",
        "python_gauss", "numpy_bit_generator", "numpy_state",
        "numpy_position", "numpy_has_gauss", "numpy_cached_gaussian",
    }
    if not isinstance(state, Mapping) or set(state) != required:
        raise ValueError("checkpoint RNG state fields are invalid")
    torch_cpu = state["torch_cpu"]
    torch_cuda = state["torch_cuda"]
    python_state = state["python_state"]
    numpy_state = state["numpy_state"]
    if (
        not isinstance(torch_cpu, torch.Tensor)
        or torch_cpu.dtype != torch.uint8
        or torch_cpu.ndim != 1
        or torch_cpu.numel() == 0
        or not isinstance(torch_cuda, list)
        or any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.uint8
            or value.ndim != 1
            or value.numel() == 0
            for value in torch_cuda
        )
        or type(state["python_version"]) is not int
        or not isinstance(python_state, torch.Tensor)
        or python_state.dtype != torch.int64
        or python_state.ndim != 1
        or python_state.numel() == 0
        or (
            state["python_gauss"] is not None
            and (
                not isinstance(state["python_gauss"], (int, float))
                or not math.isfinite(float(state["python_gauss"]))
            )
        )
        or not isinstance(state["numpy_bit_generator"], str)
        or not state["numpy_bit_generator"]
        or not isinstance(numpy_state, torch.Tensor)
        or numpy_state.dtype != torch.uint32
        or numpy_state.ndim != 1
        or numpy_state.numel() == 0
        or type(state["numpy_position"]) is not int
        or type(state["numpy_has_gauss"]) is not int
        or state["numpy_has_gauss"] not in (0, 1)
        or not isinstance(state["numpy_cached_gaussian"], (int, float))
        or not math.isfinite(float(state["numpy_cached_gaussian"]))
    ):
        raise ValueError("checkpoint torch RNG state is invalid")


def _rng_state_to_wire(state: Mapping[str, Any]) -> dict[str, Any]:
    """Make RNG state safe for all_gather_object across PyTorch versions."""

    _validate_rng_state(state)
    return {
        **dict(state),
        "torch_cpu": state["torch_cpu"].tolist(),
        "torch_cuda": [value.tolist() for value in state["torch_cuda"]],
        "python_state": state["python_state"].tolist(),
        "numpy_state": state["numpy_state"].tolist(),
    }


def _rng_state_from_wire(state: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise ValueError("gathered RNG state is invalid")
    restored = {
        **dict(state),
        "torch_cpu": torch.tensor(state["torch_cpu"], dtype=torch.uint8),
        "torch_cuda": [
            torch.tensor(value, dtype=torch.uint8)
            for value in state["torch_cuda"]
        ],
        "python_state": torch.tensor(state["python_state"], dtype=torch.int64),
        "numpy_state": torch.tensor(state["numpy_state"], dtype=torch.uint32),
    }
    _validate_rng_state(restored)
    return restored


def _optimizer_step(state_dict: Mapping[str, Any]) -> int:
    steps = []
    for state in state_dict.get("state", {}).values():
        value = state.get("step") if isinstance(state, Mapping) else None
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            steps.append(int(value.item()))
        elif isinstance(value, (int, float)):
            steps.append(int(value))
    return max(steps, default=0)


def _validate_state_topology(
    saved: Any,
    runtime: Any,
    *,
    label: str,
) -> None:
    if isinstance(runtime, Mapping):
        if not isinstance(saved, Mapping) or set(saved) != set(runtime):
            raise ValueError(f"{label} state topology mismatch")
        for name in runtime:
            _validate_state_topology(
                saved[name],
                runtime[name],
                label=f"{label}.{name}",
            )
        return
    if isinstance(runtime, (list, tuple)):
        if not isinstance(saved, type(runtime)) or len(saved) != len(runtime):
            raise ValueError(f"{label} state topology mismatch")
        for index, (saved_value, runtime_value) in enumerate(
            zip(saved, runtime)
        ):
            _validate_state_topology(
                saved_value,
                runtime_value,
                label=f"{label}[{index}]",
            )
        return
    if isinstance(runtime, torch.Tensor):
        if (
            not isinstance(saved, torch.Tensor)
            or saved.dtype != runtime.dtype
            or tuple(saved.shape) != tuple(runtime.shape)
        ):
            raise ValueError(f"{label} state topology mismatch")
        return
    if type(saved) is not type(runtime):
        raise ValueError(f"{label} state topology mismatch")


def _validate_metadata_payload(
    payload: Mapping[str, Any],
    *,
    allow_test_artifacts: bool,
) -> None:
    required = {
        "format_version",
        "planner_backend",
        "visual_vocab_size",
        "visual_token_start_id",
        "visual_token_end_id",
        "hindsight_cache_hash",
        "ta_tok_hash",
        "tokenizer_hash",
        "base_model_hash",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            "planner metadata is missing required fields: " + ", ".join(missing)
        )
    start = payload["visual_token_start_id"]
    end = payload["visual_token_end_id"]
    size = payload["visual_vocab_size"]
    if (
        type(start) is not int
        or type(end) is not int
        or type(size) is not int
        or start < 0
        or end - start != size
    ):
        raise ValueError("planner metadata visual-token range is invalid")
    for name in (
        "hindsight_cache_hash",
        "ta_tok_hash",
        "tokenizer_hash",
        "base_model_hash",
    ):
        if not isinstance(payload[name], str) or not payload[name]:
            raise ValueError(f"planner metadata {name} must be nonempty")
    if not allow_test_artifacts:
        GroundedPlannerMetadata.from_dict(payload)


def save_planner_checkpoint(
    *,
    output_dir: Path | str,
    step: int,
    planner: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    processor: Any,
    tokenizer: Any,
    metadata: GroundedPlannerMetadata | Mapping[str, Any],
    codebook: torch.Tensor,
    scaler: Any,
    optimizer_group_lrs: Mapping[str, float],
    base_model_dir: Path | str | None = None,
    released_ta_dir: Path | str | None = None,
    allow_test_artifacts: bool = False,
    rng_states_by_rank: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """Atomically publish one complete, resumable planner checkpoint."""

    if type(step) is not int or step < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    if not isinstance(planner, nn.Module):
        raise TypeError("planner must be a torch module")
    metadata_payload = (
        metadata.to_dict()
        if isinstance(metadata, GroundedPlannerMetadata)
        else dict(metadata)
    )
    _validate_metadata_payload(
        metadata_payload,
        allow_test_artifacts=allow_test_artifacts,
    )
    expected_groups = set(_GROUP_LRS)
    if set(optimizer_group_lrs) != expected_groups:
        raise ValueError("checkpoint optimizer groups must be exactly the Task-8 groups")
    optimizer_initial_lrs = {
        str(group.get("name")): float(group.get("initial_lr", group["lr"]))
        for group in optimizer.param_groups
    }
    if optimizer_initial_lrs != _GROUP_LRS:
        raise ValueError("checkpoint optimizer group learning rates must match Task 8")
    if (
        not isinstance(codebook, torch.Tensor)
        or codebook.ndim != 2
        or codebook.shape[0] != int(metadata_payload["visual_vocab_size"])
        or not codebook.dtype.is_floating_point
        or not bool(torch.isfinite(codebook).all())
    ):
        raise ValueError("checkpoint codebook is non-finite or has incompatible shape")
    if not allow_test_artifacts and tuple(codebook.shape) != (65_536, 1_536):
        raise ValueError("released checkpoint codebook must have shape [65536,1536]")

    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    optimizer_step = _optimizer_step(optimizer_state)
    scheduler_step = int(scheduler_state.get("last_epoch", -1))
    if step != optimizer_step or step != scheduler_step:
        raise ValueError(
            "current step, optimizer step, and scheduler step must match"
        )

    root = Path(output_dir)
    _assert_safe_output(
        root,
        base_model_dir=base_model_dir,
        released_ta_dir=released_ta_dir,
    )
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"step_{step:06d}"
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.incomplete-",
            dir=root,
        )
    )
    try:
        save_model(
            planner,
            str(staging / "planner.safetensors"),
            metadata={"format": "qwen35_planx_grounded"},
        )
        codebook_cpu = codebook.detach().float().cpu().contiguous()
        save_file(
            {"codebook": codebook_cpu},
            staging / "ta_codebook.safetensors",
        )
        _json_dump(
            staging / "ta_codebook.json",
            {
                "format_version": 1,
                "shape": list(codebook_cpu.shape),
                "dtype": "float32",
                "tensor_sha256": _tensor_hash(codebook_cpu),
                "released_ta_sha256": metadata_payload["ta_tok_hash"],
            },
        )
        processor.save_pretrained(staging / "processor")
        tokenizer.save_pretrained(staging / "tokenizer")
        _save_model_config(planner, staging / "model_config")
        _json_dump(staging / "planner_meta.json", metadata_payload)
        torch.save(optimizer_state, staging / "optimizer.pt")
        torch.save(scheduler_state, staging / "scheduler.pt")
        torch.save(
            {
                "enabled": scaler is not None,
                "state_dict": scaler.state_dict() if scaler is not None else None,
            },
            staging / "scaler.pt",
        )
        rng_states = (
            list(rng_states_by_rank)
            if rng_states_by_rank is not None
            else [_capture_rng_state()]
        )
        for state in rng_states:
            _validate_rng_state(state)
        torch.save(
            {"world_size": len(rng_states), "states": rng_states},
            staging / "rng_state.pt",
        )
        hash_names = tuple(
            name
            for name in REQUIRED_CHECKPOINT_ENTRIES
            if name != "trainer_state.json"
        )
        artifact_hashes = {
            name: _artifact_hash(staging / name)
            for name in hash_names
        }
        _json_dump(
            staging / "trainer_state.json",
            {
                "format_version": 1,
                "current_step": step,
                "optimizer_step": optimizer_step,
                "scheduler_step": scheduler_step,
                "optimizer_groups": {
                    name: optimizer_initial_lrs[name]
                    for name in _GROUP_LRS
                },
                "artifact_hashes": artifact_hashes,
            },
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def validate_planner_checkpoint(
    checkpoint_dir: Path | str,
    *,
    expected_metadata: GroundedPlannerMetadata | Mapping[str, Any] | None = None,
    allow_test_artifacts: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate completeness, hashes, range, and provenance without mutation."""

    checkpoint = Path(checkpoint_dir)
    missing = [
        name
        for name in REQUIRED_CHECKPOINT_ENTRIES
        if not (checkpoint / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"incomplete planner checkpoint {checkpoint}: missing {missing}"
        )
    for directory in ("processor", "tokenizer", "model_config"):
        if not (checkpoint / directory).is_dir():
            raise FileNotFoundError(
                f"incomplete planner checkpoint {checkpoint}: {directory}/ is not a directory"
            )
    try:
        metadata = json.loads(
            (checkpoint / "planner_meta.json").read_text(encoding="utf-8")
        )
        trainer_state = json.loads(
            (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
        )
        codebook_metadata = json.loads(
            (checkpoint / "ta_codebook.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid planner checkpoint JSON: {error}") from error
    if not isinstance(metadata, Mapping):
        raise ValueError("planner metadata must contain an object")
    _validate_metadata_payload(
        metadata,
        allow_test_artifacts=allow_test_artifacts,
    )
    if expected_metadata is not None:
        expected = (
            expected_metadata.to_dict()
            if isinstance(expected_metadata, GroundedPlannerMetadata)
            else dict(expected_metadata)
        )
        _validate_metadata_payload(
            expected,
            allow_test_artifacts=allow_test_artifacts,
        )
        for name, value in expected.items():
            if metadata.get(name) != value:
                raise ValueError(
                    f"checkpoint {name} mismatch: expected {value!r}, "
                    f"got {metadata.get(name)!r}"
                )
    optimizer_groups = (
        trainer_state.get("optimizer_groups", {})
        if isinstance(trainer_state, Mapping)
        else {}
    )
    if (
        not isinstance(trainer_state, Mapping)
        or trainer_state.get("format_version") != 1
        or type(trainer_state.get("current_step")) is not int
        or trainer_state["current_step"] < 0
        or not isinstance(optimizer_groups, Mapping)
        or set(optimizer_groups) != set(_GROUP_LRS)
    ):
        raise ValueError("planner trainer_state contract is invalid")
    if {
        name: float(optimizer_groups[name])
        for name in _GROUP_LRS
    } != _GROUP_LRS:
        raise ValueError(
            "planner optimizer group learning rates differ from Task 8"
        )
    if not (
        trainer_state["current_step"]
        == trainer_state.get("optimizer_step")
        == trainer_state.get("scheduler_step")
    ):
        raise ValueError(
            "current step, optimizer step, and scheduler step must match"
        )
    artifact_hashes = trainer_state.get("artifact_hashes")
    expected_hash_names = set(REQUIRED_CHECKPOINT_ENTRIES) - {"trainer_state.json"}
    if not isinstance(artifact_hashes, Mapping) or set(artifact_hashes) != expected_hash_names:
        raise ValueError("planner checkpoint artifact hash manifest is invalid")
    for name, expected_hash in artifact_hashes.items():
        actual_hash = _artifact_hash(checkpoint / name)
        if actual_hash != expected_hash:
            raise ValueError(f"planner checkpoint artifact hash mismatch: {name}")

    shape = (
        codebook_metadata.get("shape")
        if isinstance(codebook_metadata, Mapping)
        else None
    )
    if (
        not isinstance(codebook_metadata, Mapping)
        or codebook_metadata.get("released_ta_sha256") != metadata["ta_tok_hash"]
        or not isinstance(shape, list)
        or len(shape) != 2
        or shape[0] != int(metadata["visual_vocab_size"])
        or type(shape[1]) is not int
        or shape[1] <= 0
    ):
        raise ValueError("TA codebook metadata is incompatible with planner metadata")
    with safe_open(
        checkpoint / "ta_codebook.safetensors",
        framework="pt",
        device="cpu",
    ) as handle:
        if list(handle.keys()) != ["codebook"]:
            raise ValueError("TA codebook safetensors must contain only codebook")
        codebook = handle.get_tensor("codebook")
    if list(codebook.shape) != codebook_metadata["shape"]:
        raise ValueError("TA codebook tensor shape differs from metadata")
    if _tensor_hash(codebook) != codebook_metadata.get("tensor_sha256"):
        raise ValueError("TA codebook tensor hash differs from metadata")
    if not allow_test_artifacts and tuple(codebook.shape) != (65_536, 1_536):
        raise ValueError("released checkpoint codebook must have shape [65536,1536]")
    return dict(metadata), dict(trainer_state)


def load_planner_checkpoint(
    checkpoint_dir: Path | str,
    *,
    planner: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    expected_metadata: GroundedPlannerMetadata | Mapping[str, Any],
    allow_test_artifacts: bool = False,
    process_index: int = 0,
    world_size: int = 1,
) -> int:
    """Resume only after the complete checkpoint has passed every validation."""

    checkpoint = Path(checkpoint_dir)
    _, trainer_state = validate_planner_checkpoint(
        checkpoint,
        expected_metadata=expected_metadata,
        allow_test_artifacts=allow_test_artifacts,
    )
    optimizer_state = torch.load(
        checkpoint / "optimizer.pt",
        weights_only=True,
        map_location="cpu",
    )
    scheduler_state = torch.load(
        checkpoint / "scheduler.pt",
        weights_only=True,
        map_location="cpu",
    )
    scaler_state = torch.load(
        checkpoint / "scaler.pt",
        weights_only=True,
        map_location="cpu",
    )
    rng_state = torch.load(
        checkpoint / "rng_state.pt",
        weights_only=True,
        map_location="cpu",
    )
    if (
        not isinstance(rng_state, Mapping)
        or rng_state.get("world_size") != world_size
        or not isinstance(rng_state.get("states"), list)
        or len(rng_state["states"]) != world_size
        or not 0 <= process_index < world_size
    ):
        raise ValueError("checkpoint RNG world-size/rank coverage mismatch")
    for state in rng_state["states"]:
        _validate_rng_state(state)
    runtime_state = planner.state_dict()
    with safe_open(
        checkpoint / "planner.safetensors", framework="pt", device="cpu"
    ) as handle:
        file_shapes = {
            name: tuple(handle.get_slice(name).get_shape())
            for name in handle.keys()
        }
        aliases = dict(handle.metadata() or {})
    for name, shape in file_shapes.items():
        if name not in runtime_state or tuple(runtime_state[name].shape) != shape:
            raise ValueError(f"planner state topology mismatch at {name}")
    missing = set(runtime_state).difference(file_shapes)
    alias_names = {
        name for name, target in aliases.items() if target in file_shapes
    }
    if missing.difference(alias_names):
        raise ValueError("planner state keys differ from checkpoint")

    runtime_groups = optimizer.param_groups
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("optimizer state topology mismatch")
    saved_groups = optimizer_state.get("param_groups")
    if (
        not isinstance(saved_groups, list)
        or len(saved_groups) != len(runtime_groups)
    ):
        raise ValueError("optimizer group topology mismatch")
    saved_param_map: dict[int, nn.Parameter] = {}
    for saved_group, runtime_group in zip(saved_groups, runtime_groups):
        if not isinstance(saved_group, Mapping):
            raise ValueError("optimizer group topology mismatch")
        saved_params = saved_group.get("params", [])
        runtime_params = runtime_group.get("params", [])
        if (
            saved_group.get("name") != runtime_group.get("name")
            or set(saved_group) != set(runtime_group)
            or len(saved_params) != len(runtime_params)
        ):
            raise ValueError("optimizer group topology mismatch")
        for name in set(runtime_group).difference({"params"}):
            _validate_state_topology(
                saved_group[name],
                runtime_group[name],
                label=f"optimizer group {saved_group.get('name')}.{name}",
            )
        saved_param_map.update(zip(saved_params, runtime_params))
    saved_optimizer_slots = optimizer_state.get("state")
    if not isinstance(saved_optimizer_slots, Mapping):
        raise ValueError("optimizer state topology mismatch")
    for identifier, state in saved_optimizer_slots.items():
        parameter = saved_param_map.get(identifier)
        if parameter is None or not isinstance(state, Mapping):
            raise ValueError("optimizer state topology mismatch")
        for value in state.values():
            if (
                isinstance(value, torch.Tensor)
                and value.numel() != 1
                and tuple(value.shape) != tuple(parameter.shape)
            ):
                raise ValueError("optimizer state tensor shape mismatch")
    runtime_scheduler = scheduler.state_dict()
    _validate_state_topology(
        scheduler_state,
        runtime_scheduler,
        label="scheduler",
    )
    if len(scheduler_state.get("base_lrs", [])) != len(runtime_groups):
        raise ValueError("scheduler group topology mismatch")
    if not isinstance(scaler_state, Mapping) or set(scaler_state) != {
        "enabled", "state_dict"
    }:
        raise ValueError("checkpoint scaler payload is invalid")
    if _optimizer_step(optimizer_state) != int(trainer_state["optimizer_step"]):
        raise ValueError("optimizer step differs from trainer_state")
    if int(scheduler_state.get("last_epoch", -1)) != int(
        trainer_state["scheduler_step"]
    ):
        raise ValueError("scheduler step differs from trainer_state")
    if bool(scaler_state.get("enabled")) != (scaler is not None):
        raise ValueError("checkpoint scaler mode differs from runtime")
    if scaler is not None:
        _validate_state_topology(
            scaler_state["state_dict"],
            scaler.state_dict(),
            label="scaler",
        )

    load_model(planner, checkpoint / "planner.safetensors", strict=True)
    optimizer.load_state_dict(optimizer_state)
    scheduler.load_state_dict(scheduler_state)
    if scaler is not None:
        scaler.load_state_dict(scaler_state["state_dict"])
    _restore_rng_state(rng_state["states"][process_index])
    return int(trainer_state["current_step"])


def _move_batch(
    batch: GroundedPlannerBatch,
    device: torch.device,
) -> GroundedPlannerBatch:
    values = {}
    for name, value in batch.__dict__.items():
        if name == "qwen_inputs":
            values[name] = {
                key: tensor.to(device=device)
                for key, tensor in value.items()
            }
        elif isinstance(value, torch.Tensor):
            values[name] = value.to(device=device)
        else:
            values[name] = value
    return GroundedPlannerBatch(**values)


class _TinyLanguage(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.gradient_checkpointing_kwargs: Mapping[str, Any] | None = None

    def gradient_checkpointing_enable(
        self,
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.gradient_checkpointing_kwargs = kwargs

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.projection(hidden))


class _TinyBackbone(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.vision_model = nn.Linear(hidden_dim, hidden_dim)
        self.language_model = _TinyLanguage(hidden_dim)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(self, input_ids: torch.Tensor, **_kwargs: Any) -> Mapping[str, torch.Tensor]:
        hidden = self.embed_tokens(input_ids)
        hidden = hidden + 0.01 * self.vision_model(hidden)
        return {"last_hidden_state": self.language_model(hidden)}


class _TinySavedArtifact:
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def save_pretrained(self, output_dir: Path | str) -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        _json_dump(destination / "artifact.json", {"kind": self.kind})


def _tiny_metadata() -> dict[str, Any]:
    return {
        "format_version": 1,
        "planner_backend": "qwen35_planx_grounded",
        "visual_vocab_size": 13,
        "visual_token_start_id": 7,
        "visual_token_end_id": 20,
        "hindsight_cache_hash": "cache-sha256",
        "ta_tok_hash": "released-ta-sha256",
        "tokenizer_hash": "tokenizer-sha256",
        "base_model_hash": "base-model-sha256",
    }


def _tiny_batch(
    *,
    samples: int,
    vocab_size: int = 13,
    hidden_dim: int = 8,
    text_dim: int = 6,
    tokens: int = 4,
) -> GroundedPlannerBatch:
    camera_batch = samples * 2
    code_count = 4 * tokens
    sequence_length = (2 * code_count) + 3
    pre_positions = torch.arange(code_count).expand(camera_batch, -1).clone()
    post_positions = (
        torch.arange(code_count, 2 * code_count)
        .expand(camera_batch, -1)
        .clone()
    )
    field_positions = torch.arange(
        2 * code_count,
        sequence_length,
    ).expand(camera_batch, -1).clone()
    relevance = torch.rand(camera_batch, 4, 3, tokens)
    relevance = relevance / relevance.sum(dim=-1, keepdim=True)
    phrase_embeddings = F.normalize(
        torch.randn(camera_batch, 3, text_dim),
        dim=-1,
    )
    flow = torch.zeros(camera_batch, 3, tokens, 3)
    flow[..., 2] = 1
    return GroundedPlannerBatch(
        qwen_inputs={
            "input_ids": torch.arange(sequence_length)
            .remainder(vocab_size)
            .expand(camera_batch, -1)
            .clone(),
            "attention_mask": torch.ones(
                camera_batch,
                sequence_length,
                dtype=torch.long,
            ),
        },
        code_targets=torch.arange(camera_batch * code_count)
        .reshape(camera_batch, code_count)
        .remainder(vocab_size),
        pre_positions=pre_positions,
        post_positions=post_positions,
        field_positions=field_positions,
        field_mask=torch.ones(camera_batch, 3, dtype=torch.bool),
        relevance_targets=relevance,
        relevance_confidence=torch.ones(camera_batch, 4, 3),
        flow_targets=flow,
        phrase_embeddings=phrase_embeddings,
        counterfactual_embeddings=torch.roll(
            phrase_embeddings,
            shifts=1,
            dims=-1,
        ).unsqueeze(2),
        counterfactual_mask=torch.ones(camera_batch, 3, 1, dtype=torch.bool),
    )


@dataclass
class _TrainingArtifacts:
    planner: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    train_batches: Any
    validation_batches: Any
    processor: Any
    tokenizer: Any
    metadata: GroundedPlannerMetadata | Mapping[str, Any]
    codebook: torch.Tensor
    allow_test_artifacts: bool
    base_model_dir: Path | None = None
    released_ta_dir: Path | None = None
    cache: Any = None


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: PlannerTrainingConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr_multiplier(
            int(step),
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
        ),
    )


def _build_tiny_artifacts(config: PlannerTrainingConfig) -> _TrainingArtifacts:
    from qwen35_planx.planner import GroundedQwen35Planner

    vocab_size = 13
    hidden_dim = 8
    text_dim = 6
    code_dim = 7
    backbone = _TinyBackbone(vocab_size, hidden_dim)
    codebook = torch.randn(vocab_size, code_dim)
    planner = GroundedQwen35Planner._from_test_components(
        backbone=backbone,
        visual_embedding_weight=backbone.embed_tokens.weight,
        codebook=codebook,
        hidden_dim=hidden_dim,
        text_dim=text_dim,
    )
    groups = build_optimizer_groups(
        planner,
        visual_token_start_id=7,
        qwen_language_lr=config.qwen_language_lr,
        qwen_vision_lr=config.qwen_vision_lr,
        head_lr=config.head_lr,
    )
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    scheduler = _make_scheduler(optimizer, config)
    batch = _tiny_batch(samples=config.per_device_batch)
    return _TrainingArtifacts(
        planner=planner,
        optimizer=optimizer,
        scheduler=scheduler,
        train_batches=(batch,),
        validation_batches=(batch,),
        processor=_TinySavedArtifact("processor"),
        tokenizer=_TinySavedArtifact("tokenizer"),
        metadata=_tiny_metadata(),
        codebook=codebook,
        allow_test_artifacts=True,
    )


def _optimizer_group_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    result = {}
    for group in optimizer.param_groups:
        name = group.get("name")
        if not isinstance(name, str) or name in result:
            raise ValueError("optimizer groups must have unique string names")
        result[name] = float(group.get("initial_lr", group["lr"]))
    if set(result) != set(_GROUP_LRS):
        raise ValueError("optimizer does not contain the exact Task-8 groups")
    return result


def _next_batch(iterator: Any, batches: Any) -> tuple[GroundedPlannerBatch, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(batches)
        try:
            return next(iterator), iterator
        except StopIteration as error:
            raise ValueError("planner training requires a nonempty train loader") from error


@torch.no_grad()
def _validate_one_batch(
    planner: nn.Module,
    validation_batches: Any,
    *,
    device: torch.device,
) -> float:
    iterator = iter(validation_batches)
    try:
        batch = next(iterator)
    except StopIteration as error:
        raise ValueError("planner validation requires a nonempty validation loader") from error
    planner.eval()
    output = planner(_move_batch(batch, device))
    value = float(output.total_loss.detach().float().cpu())
    if not math.isfinite(value):
        raise FloatingPointError("planner validation loss is non-finite")
    planner.train()
    return value


def _save_from_training(
    *,
    accelerator: Any,
    artifacts: _TrainingArtifacts,
    config: PlannerTrainingConfig,
    step: int,
) -> Path | None:
    accelerator.wait_for_everyone()
    from accelerate.utils import gather_object

    wire_states = gather_object([_rng_state_to_wire(_capture_rng_state())])
    rng_states = [_rng_state_from_wire(state) for state in wire_states]
    checkpoint = None
    if accelerator.is_main_process:
        checkpoint = save_planner_checkpoint(
            output_dir=Path(config.output_dir),
            step=step,
            planner=accelerator.unwrap_model(artifacts.planner),
            optimizer=artifacts.optimizer,
            scheduler=artifacts.scheduler,
            processor=artifacts.processor,
            tokenizer=artifacts.tokenizer,
            metadata=artifacts.metadata,
            codebook=artifacts.codebook,
            scaler=getattr(accelerator, "scaler", None),
            optimizer_group_lrs=_optimizer_group_lrs(artifacts.optimizer),
            base_model_dir=artifacts.base_model_dir,
            released_ta_dir=artifacts.released_ta_dir,
            allow_test_artifacts=artifacts.allow_test_artifacts,
            rng_states_by_rank=rng_states,
        )
    accelerator.wait_for_everyone()
    return checkpoint


def run_training(config: PlannerTrainingConfig) -> Path | None:
    """Run the bounded Accelerate optimizer-step loop."""

    from accelerate import Accelerator
    from accelerate.utils import GradientAccumulationPlugin, set_seed

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        step_scheduler_with_optimizer=False,
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=config.gradient_accumulation_steps,
            sync_with_dataloader=False,
        ),
    )
    validate_effective_global_batch(
        per_device_batch=config.per_device_batch,
        num_processes=accelerator.num_processes,
        grad_accum=config.gradient_accumulation_steps,
    )
    set_seed(config.seed, device_specific=True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = config.tf32
        torch.backends.cudnn.allow_tf32 = config.tf32

    artifacts = (
        _build_tiny_artifacts(config)
        if config.tiny_smoke
        else _build_production_artifacts(config)
    )
    if config.activation_checkpointing:
        enable_selective_qwen_activation_checkpointing(artifacts.planner)
    artifacts.planner.to(dtype=torch.bfloat16)
    current_step = 0
    if config.resume_from is not None:
        current_step = load_planner_checkpoint(
            config.resume_from,
            planner=artifacts.planner,
            optimizer=artifacts.optimizer,
            scheduler=artifacts.scheduler,
            scaler=getattr(accelerator, "scaler", None),
            expected_metadata=artifacts.metadata,
            allow_test_artifacts=artifacts.allow_test_artifacts,
            process_index=accelerator.process_index,
            world_size=accelerator.num_processes,
        )
        if current_step >= config.max_steps:
            raise ValueError(
                f"resume step {current_step} must be below max_steps {config.max_steps}"
            )

    prepared = accelerator.prepare(
        artifacts.planner,
        artifacts.optimizer,
        artifacts.train_batches,
        artifacts.validation_batches,
        artifacts.scheduler,
    )
    (
        artifacts.planner,
        artifacts.optimizer,
        artifacts.train_batches,
        artifacts.validation_batches,
        artifacts.scheduler,
    ) = prepared
    artifacts.planner.train()
    train_iterator = iter(artifacts.train_batches)
    last_checkpoint: Path | None = None
    while current_step < config.max_steps:
        batch, train_iterator = _next_batch(
            train_iterator,
            artifacts.train_batches,
        )
        batch = _move_batch(batch, accelerator.device)
        with accelerator.accumulate(artifacts.planner):
            output = artifacts.planner(batch)
            accelerator.backward(output.total_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    artifacts.planner.parameters(),
                    config.gradient_clip_norm,
                )
                artifacts.optimizer.step()
                artifacts.scheduler.step()
                artifacts.optimizer.zero_grad(set_to_none=True)
        if not accelerator.sync_gradients:
            continue
        current_step += 1
        if current_step % config.log_every == 0:
            accelerator.print(
                f"step={current_step} loss={float(output.total_loss.detach()):.6f}"
            )
        if current_step % config.validate_every == 0:
            validation_loss = _validate_one_batch(
                artifacts.planner,
                artifacts.validation_batches,
                device=accelerator.device,
            )
            accelerator.print(
                f"validation step={current_step} loss={validation_loss:.6f}"
            )
        if current_step % config.save_every == 0:
            last_checkpoint = _save_from_training(
                accelerator=accelerator,
                artifacts=artifacts,
                config=config,
                step=current_step,
            )

    if current_step % config.validate_every != 0:
        validation_loss = _validate_one_batch(
            artifacts.planner,
            artifacts.validation_batches,
            device=accelerator.device,
        )
        accelerator.print(
            f"validation step={current_step} loss={validation_loss:.6f}"
        )
    if current_step % config.save_every != 0:
        last_checkpoint = _save_from_training(
            accelerator=accelerator,
            artifacts=artifacts,
            config=config,
            step=current_step,
        )
    if accelerator.is_main_process and last_checkpoint is not None:
        validate_planner_checkpoint(
            last_checkpoint,
            expected_metadata=artifacts.metadata,
            allow_test_artifacts=artifacts.allow_test_artifacts,
        )
    accelerator.wait_for_everyone()
    if artifacts.cache is not None:
        artifacts.cache.close()
    accelerator.end_training()
    return last_checkpoint


def _build_production_artifacts(
    config: PlannerTrainingConfig,
) -> _TrainingArtifacts:
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader, Subset
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )

    from qwen35_planx.config import CAMERA_KEYS, CAMERA_NAMES, PlanGeometry
    from qwen35_planx.hindsight_schema import HindsightCache
    from qwen35_planx.planner import GroundedQwen35Planner
    from qwen35_planx.planner_dataset import (
        GroundedPlannerCollator,
        HindsightPlannerDataset,
    )
    from qwen35_planx.vocabulary import install_visual_vocabulary

    base_model_dir = Path(str(config.base_model)).resolve()
    hdf5_manifest = Path(str(config.hdf5_manifest)).resolve()
    cache_dir = Path(str(config.hindsight_cache)).resolve()
    codebook_path = Path(str(config.ta_codebook)).resolve()
    codebook_metadata_path = Path(str(config.ta_codebook_metadata)).resolve()
    from qwen35_planx.cli.preflight import (
        collect_planner_training_preflight_errors,
    )

    preflight_errors = collect_planner_training_preflight_errors(config)
    if preflight_errors:
        raise ValueError(
            "planner training preflight failed:\n- "
            + "\n- ".join(preflight_errors)
        )

    cache = HindsightCache.open(cache_dir)
    try:
        codebook_metadata = json.loads(
            codebook_metadata_path.read_text(encoding="utf-8")
        )
        codebook = load_file(codebook_path, device="cpu")["codebook"].float()
        if tuple(codebook.shape) != (65_536, 1_536):
            raise ValueError("released TA codebook must have shape [65536,1536]")
        if not bool(torch.isfinite(codebook).all()):
            raise ValueError("released TA codebook contains non-finite values")

        processor = AutoProcessor.from_pretrained(
            base_model_dir,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_dir,
            local_files_only=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            base_model_dir,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        layout = install_visual_vocabulary(
            tokenizer,
            model,
            base_model_directory=base_model_dir,
        )
        if hasattr(processor, "tokenizer"):
            processor.tokenizer = tokenizer
        input_embeddings = model.get_input_embeddings()
        if input_embeddings is None:
            raise ValueError("Qwen model does not expose input embeddings")
        visual_weight = input_embeddings.weight[
            layout.visual_start_id : layout.visual_end_id
        ]
        planner = GroundedQwen35Planner.from_components(
            backbone=model,
            visual_embedding_weight=visual_weight,
            codebook=codebook,
        )
        geometry = PlanGeometry()
        base_model_hash = _artifact_hash(base_model_dir)
        released_ta_hash = str(codebook_metadata["checkpoint_sha256"])
        metadata = GroundedPlannerMetadata(
            format_version=GroundedPlannerMetadata.FORMAT_VERSION,
            planner_backend=GroundedPlannerMetadata.BACKEND,
            base_model=str(base_model_dir),
            model_type="qwen3_5",
            camera_names=CAMERA_NAMES,
            camera_keys=CAMERA_KEYS,
            image_size=geometry.image_size,
            num_keyframes=geometry.num_keyframes,
            grid_size=geometry.grid_size,
            visual_vocab_size=geometry.visual_vocab_size,
            future_frame_offsets=geometry.future_frame_offsets,
            ge_act_future_indices=geometry.ge_act_future_indices,
            tokens_per_frame=geometry.tokens_per_frame,
            response_tokens_per_camera=geometry.response_tokens_per_camera,
            visual_token_start_id=layout.visual_start_id,
            visual_token_end_id=layout.visual_end_id,
            structure_token_ids=layout.structure_token_ids,
            loss_weights=(
                ("code", 1.0),
                ("dense_feature", 0.5),
                ("grounding", 0.5),
                ("counterfactual", 0.2),
                ("temporal", 0.1),
            ),
            phrase_roles=("source", "target", "action"),
            hidden_alignment=GroundedPlannerMetadata.HIDDEN_ALIGNMENT,
            qwen_hidden_dim=geometry.qwen_hidden_dim,
            text_align_dim=geometry.text_align_dim,
            tokenizer_hash=layout.tokenizer_hash,
            ta_tok_hash=released_ta_hash,
            base_model_hash=base_model_hash,
            hindsight_cache_hash=cache.cache_hash,
        )
        dataset = HindsightPlannerDataset(
            cache,
            hdf5_manifest,
            metadata=metadata,
        )
        collator = GroundedPlannerCollator(
            processor,
            layout,
            cache_dir=cache_dir,
            dataset=dataset,
            metadata=metadata,
        )
        train_indices = [
            index
            for index, record in enumerate(cache.records)
            if record.split == "train"
        ]
        validation_indices = [
            index
            for index, record in enumerate(cache.records)
            if record.split == "val"
        ]
        if not train_indices or not validation_indices:
            raise ValueError(
                "hindsight cache must contain nonempty train and validation splits"
            )
        train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=config.per_device_batch,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        validation_loader = DataLoader(
            Subset(dataset, validation_indices),
            batch_size=config.per_device_batch,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        groups = build_optimizer_groups(
            planner,
            visual_token_start_id=layout.visual_start_id,
            experiment_token_start_id=layout.original_vocab_size,
            qwen_language_lr=config.qwen_language_lr,
            qwen_vision_lr=config.qwen_vision_lr,
            head_lr=config.head_lr,
        )
        optimizer = torch.optim.AdamW(
            groups,
            weight_decay=config.weight_decay,
        )
        scheduler = _make_scheduler(optimizer, config)
        return _TrainingArtifacts(
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            train_batches=train_loader,
            validation_batches=validation_loader,
            processor=processor,
            tokenizer=tokenizer,
            metadata=metadata,
            codebook=codebook,
            allow_test_artifacts=False,
            base_model_dir=base_model_dir,
            released_ta_dir=codebook_path.parent,
            cache=cache,
        )
    except Exception:
        cache.close()
        raise


def _load_config(path: Path) -> PlannerTrainingConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid planner training config {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("planner training config must contain an object")
    expanded = {
        key: (
            value.replace("${PID}", str(os.getpid()))
            if isinstance(value, str)
            else value
        )
        for key, value in payload.items()
    }
    return PlannerTrainingConfig.from_mapping(expanded)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--per-device-batch", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--qwen-language-lr", type=float)
    parser.add_argument("--qwen-vision-lr", type=float)
    parser.add_argument("--head-lr", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = _load_config(arguments.config)
    updates = {}
    if arguments.max_steps is not None:
        updates["max_steps"] = arguments.max_steps
    if arguments.output_dir is not None:
        updates["output_dir"] = str(arguments.output_dir)
    if arguments.resume_from is not None:
        updates["resume_from"] = str(arguments.resume_from)
    for name in (
        "per_device_batch",
        "gradient_accumulation_steps",
        "warmup_steps",
        "save_every",
        "validate_every",
        "qwen_language_lr",
        "qwen_vision_lr",
        "head_lr",
    ):
        value = getattr(arguments, name)
        if value is not None:
            updates[name] = value
    if updates:
        config = replace(config, **updates)
    run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
