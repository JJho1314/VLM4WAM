"""Atomic, provenance-checked checkpoints for exact Baton Stage-1 resume."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
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
from safetensors.torch import load_model, save_model
import torch
import torch.nn as nn

from qwen35_baton.config import BatonCheckpointMetadata
from qwen35_baton.hashing import sha256_file, sha256_json


_CHECKPOINT_FILES = (
    "planner.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "rank_rng.pt",
    "cursor.json",
    "metadata.json",
)
_RUNTIME_METADATA_FIELDS = {
    "optimizer_topology_hash",
    "scheduler_topology_hash",
    "global_step",
    "distributed_cursor",
    "rng_state_hash",
}
_RNG_FIELDS = {
    "format_version",
    "distributed_rank",
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


@dataclass(frozen=True)
class BatonTrainingCursor:
    """The exact next microbatch position in the deterministic sampler stream."""

    global_step: int
    epoch: int
    consumed_microbatches: int
    microbatches_per_epoch: int
    sampler_seed: int

    def __post_init__(self) -> None:
        for name in (
            "global_step",
            "epoch",
            "consumed_microbatches",
            "microbatches_per_epoch",
            "sampler_seed",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"training cursor {name} must be an integer")
        if self.global_step < 0 or self.epoch < 0:
            raise ValueError("training cursor step and epoch must be non-negative")
        if self.microbatches_per_epoch <= 0:
            raise ValueError("training cursor microbatches_per_epoch must be positive")
        if not 0 <= self.consumed_microbatches < self.microbatches_per_epoch:
            raise ValueError(
                "training cursor consumed_microbatches must identify the next "
                "microbatch inside an epoch"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "consumed_microbatches": self.consumed_microbatches,
            "microbatches_per_epoch": self.microbatches_per_epoch,
            "sampler_seed": self.sampler_seed,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "BatonTrainingCursor":
        fields = {
            "global_step",
            "epoch",
            "consumed_microbatches",
            "microbatches_per_epoch",
            "sampler_seed",
        }
        if not isinstance(payload, Mapping) or set(payload) != fields:
            raise ValueError("checkpoint training cursor fields are invalid")
        return cls(**{name: payload[name] for name in fields})


@dataclass(frozen=True)
class BatonResumeState:
    """Validated cursor, contract, and the selected distributed rank RNG state."""

    metadata: BatonCheckpointMetadata
    cursor: BatonTrainingCursor
    rank_rng_state: Mapping[str, Any]


def capture_rank_rng_state(*, distributed_rank: int) -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and every visible CUDA RNG stream."""

    if type(distributed_rank) is not int or distributed_rank < 0:
        raise ValueError("distributed_rank must be a non-negative integer")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    return {
        "format_version": 1,
        "distributed_rank": distributed_rank,
        "torch_cpu": torch.random.get_rng_state().cpu(),
        "torch_cuda": [
            state.cpu() for state in (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            )
        ],
        "python_version": int(python_state[0]),
        "python_state": torch.tensor(python_state[1], dtype=torch.int64),
        "python_gauss": python_state[2],
        "numpy_bit_generator": str(numpy_state[0]),
        "numpy_state": torch.from_numpy(numpy_state[1].copy()),
        "numpy_position": int(numpy_state[2]),
        "numpy_has_gauss": int(numpy_state[3]),
        "numpy_cached_gaussian": float(numpy_state[4]),
    }


def _validate_rank_rng_state(state: Any, *, expected_rank: int) -> None:
    if not isinstance(state, Mapping) or set(state) != _RNG_FIELDS:
        raise ValueError("checkpoint rank RNG state fields are invalid")
    torch_cpu = state["torch_cpu"]
    torch_cuda = state["torch_cuda"]
    python_state = state["python_state"]
    numpy_state = state["numpy_state"]
    if (
        state["format_version"] != 1
        or state["distributed_rank"] != expected_rank
        or not isinstance(torch_cpu, torch.Tensor)
        or torch_cpu.dtype != torch.uint8
        or torch_cpu.device.type != "cpu"
        or torch_cpu.ndim != 1
        or torch_cpu.numel() == 0
        or not isinstance(torch_cuda, list)
        or any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.uint8
            or value.device.type != "cpu"
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
        raise ValueError("checkpoint rank RNG state is invalid")


def restore_rank_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a rank state only after checkpoint validation has completed."""

    rank = state.get("distributed_rank") if isinstance(state, Mapping) else None
    if type(rank) is not int:
        raise ValueError("checkpoint rank RNG state is invalid")
    _validate_rank_rng_state(state, expected_rank=rank)
    torch.random.set_rng_state(state["torch_cpu"])
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise ValueError(
                "checkpoint contains CUDA RNG streams but CUDA is unavailable"
            )
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG stream count differs from runtime")
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


def _json_write(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _optimizer_topology(
    optimizer_state: Mapping[str, Any],
    *,
    runtime_groups: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    groups = optimizer_state.get("param_groups")
    slots = optimizer_state.get("state")
    if not isinstance(groups, list) or not isinstance(slots, Mapping):
        raise ValueError("optimizer state topology is invalid")
    topology = []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise ValueError("optimizer state topology is invalid")
        runtime_group = None if runtime_groups is None else runtime_groups[index]
        parameter_shapes = []
        for position, identifier in enumerate(group["params"]):
            shape = None
            if runtime_group is not None:
                parameters = runtime_group.get("params")
                if not isinstance(parameters, list) or position >= len(parameters):
                    raise ValueError("optimizer group topology differs from runtime")
                parameter = parameters[position]
                if not isinstance(parameter, torch.Tensor):
                    raise ValueError("optimizer runtime group contains a non-tensor")
                shape = list(parameter.shape)
            slot = slots.get(identifier, {})
            if not isinstance(slot, Mapping):
                raise ValueError("optimizer slot topology is invalid")
            if shape is None:
                for slot_name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    value = slot.get(slot_name)
                    if isinstance(value, torch.Tensor):
                        shape = list(value.shape)
                        break
            slot_shapes = {
                name: (
                    {"shape": list(value.shape), "dtype": str(value.dtype)}
                    if isinstance(value, torch.Tensor)
                    else {"type": type(value).__name__}
                )
                for name, value in sorted(slot.items())
            }
            parameter_shapes.append({"shape": shape, "slots": slot_shapes})
        topology.append(
            {
                "name": group.get("name"),
                "parameter_count": len(group["params"]),
                "initial_lr": float(group.get("initial_lr", group.get("lr", 0.0))),
                "parameters": parameter_shapes,
            }
        )
    return topology


def _scheduler_topology(state: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise ValueError("scheduler state topology is invalid")

    def describe(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return {"tensor_shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, list):
            return [describe(item) for item in value]
        if isinstance(value, tuple):
            return {"tuple": [describe(item) for item in value]}
        if isinstance(value, Mapping):
            return {str(key): describe(item) for key, item in sorted(value.items())}
        return type(value).__name__

    return {str(key): describe(value) for key, value in sorted(state.items())}


def _normalized_rank_states(
    rank_rng_state: Mapping[Any, Any],
) -> dict[int, Mapping[str, Any]]:
    if not isinstance(rank_rng_state, Mapping) or not rank_rng_state:
        raise ValueError("rank RNG publications must be a nonempty mapping")
    states: dict[int, Mapping[str, Any]] = {}
    for raw_rank, state in rank_rng_state.items():
        if type(raw_rank) is int:
            rank = raw_rank
        elif isinstance(raw_rank, str) and raw_rank.isdigit():
            rank = int(raw_rank)
        else:
            raise ValueError("rank RNG publication keys must be integer ranks")
        if rank in states:
            raise ValueError("rank RNG publications contain a duplicate rank")
        states[rank] = state
    if sorted(states) != list(range(len(states))):
        raise ValueError("rank RNG publications must cover contiguous rank IDs")
    for rank, state in states.items():
        _validate_rank_rng_state(state, expected_rank=rank)
    return states


def _metadata_cursor(
    metadata: BatonCheckpointMetadata,
    cursor: BatonTrainingCursor,
    *,
    world_size: int,
    optimizer_hash: str,
    scheduler_hash: str,
    rng_hash: str,
) -> BatonCheckpointMetadata:
    return replace(
        metadata,
        optimizer_topology_hash=optimizer_hash,
        scheduler_topology_hash=scheduler_hash,
        global_step=cursor.global_step,
        distributed_cursor=(
            ("epoch", cursor.epoch),
            ("consumed_microbatches", cursor.consumed_microbatches),
            ("microbatches_per_epoch", cursor.microbatches_per_epoch),
            ("sampler_seed", cursor.sampler_seed),
            ("world_size", world_size),
        ),
        rng_state_hash=rng_hash,
    )


def save_baton_checkpoint(
    checkpoint_dir: Path,
    *,
    planner: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    metadata: BatonCheckpointMetadata,
    cursor: BatonTrainingCursor,
    rank_rng_state: Mapping[Any, Any],
    scaler: Any | None = None,
) -> None:
    """Atomically publish a complete, fsynced, all-rank Stage-1 checkpoint."""

    destination = Path(checkpoint_dir)
    if not isinstance(planner, nn.Module):
        raise TypeError("planner must be a torch module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if not isinstance(metadata, BatonCheckpointMetadata):
        raise TypeError("metadata must be BatonCheckpointMetadata")
    if not isinstance(cursor, BatonTrainingCursor):
        raise TypeError("cursor must be BatonTrainingCursor")
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    states = _normalized_rank_states(rank_rng_state)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.incomplete-",
            dir=destination.parent,
        )
    )
    try:
        save_model(
            planner,
            str(staging / "planner.safetensors"),
            metadata={"format": BatonCheckpointMetadata.ARCHITECTURE_KIND},
        )
        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()
        torch.save(optimizer_state, staging / "optimizer.pt")
        torch.save(scheduler_state, staging / "scheduler.pt")
        torch.save(
            {
                "enabled": scaler is not None,
                "state_dict": None if scaler is None else scaler.state_dict(),
            },
            staging / "scaler.pt",
        )
        torch.save(
            {
                "format_version": 1,
                "world_size": len(states),
                "states": [states[rank] for rank in range(len(states))],
            },
            staging / "rank_rng.pt",
        )
        for name in (
            "planner.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "scaler.pt",
            "rank_rng.pt",
        ):
            _fsync_file(staging / name)
        _json_write(staging / "cursor.json", cursor.to_dict())
        runtime_metadata = _metadata_cursor(
            metadata,
            cursor,
            world_size=len(states),
            optimizer_hash=sha256_json(
                _optimizer_topology(optimizer_state)
            ),
            scheduler_hash=sha256_json(_scheduler_topology(scheduler_state)),
            rng_hash=sha256_file(staging / "rank_rng.pt"),
        )
        _json_write(staging / "metadata.json", runtime_metadata.to_dict())
        hashes = {
            name: sha256_file(staging / name) for name in _CHECKPOINT_FILES
        }
        _json_write(
            staging / "manifest.json",
            {"format_version": 1, "files": hashes},
        )
        _fsync_directory(staging)
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"checkpoint {label} JSON is invalid") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint {label} must contain an object")
    return payload


def _validate_expected_contract(
    actual: BatonCheckpointMetadata,
    expected: BatonCheckpointMetadata,
) -> None:
    actual_payload = actual.to_dict()
    expected_payload = expected.to_dict()
    for name in BatonCheckpointMetadata._REQUIRED_FIELDS:
        if name in _RUNTIME_METADATA_FIELDS:
            continue
        if actual_payload[name] != expected_payload[name]:
            raise ValueError(
                f"checkpoint {name} mismatch: expected {expected_payload[name]!r}, "
                f"got {actual_payload[name]!r}"
            )


def _validate_cursor_metadata(
    cursor: BatonTrainingCursor,
    metadata: BatonCheckpointMetadata,
    *,
    world_size: int,
) -> None:
    expected_cursor = {
        "epoch": cursor.epoch,
        "consumed_microbatches": cursor.consumed_microbatches,
        "microbatches_per_epoch": cursor.microbatches_per_epoch,
        "sampler_seed": cursor.sampler_seed,
        "world_size": world_size,
    }
    if (
        metadata.global_step != cursor.global_step
        or dict(metadata.distributed_cursor) != expected_cursor
    ):
        raise ValueError("checkpoint cursor and metadata are inconsistent")


def _validate_model_topology(checkpoint: Path, planner: nn.Module) -> None:
    runtime = planner.state_dict()
    with safe_open(
        checkpoint / "planner.safetensors", framework="pt", device="cpu"
    ) as handle:
        saved = {
            name: tuple(handle.get_slice(name).get_shape()) for name in handle.keys()
        }
        aliases = dict(handle.metadata() or {})
    for name, shape in saved.items():
        if name not in runtime or tuple(runtime[name].shape) != shape:
            raise ValueError(f"planner state topology mismatch at {name}")
    missing = set(runtime).difference(saved)
    alias_names = {
        name for name, target in aliases.items() if target in saved
    }
    if missing.difference(alias_names):
        raise ValueError("planner state topology keys differ from checkpoint")


def _validate_optimizer_runtime(
    saved: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    expected_hash: str,
) -> None:
    runtime_state = optimizer.state_dict()
    saved_groups = saved.get("param_groups")
    runtime_groups = runtime_state.get("param_groups")
    if (
        not isinstance(saved_groups, list)
        or not isinstance(runtime_groups, list)
        or len(saved_groups) != len(runtime_groups)
        or len(saved_groups) != len(optimizer.param_groups)
    ):
        raise ValueError("optimizer group topology differs from runtime")
    for saved_group, runtime_group, live_group in zip(
        saved_groups, runtime_groups, optimizer.param_groups
    ):
        if (
            saved_group.get("name") != runtime_group.get("name")
            or len(saved_group.get("params", ())) != len(runtime_group.get("params", ()))
            or len(saved_group.get("params", ())) != len(live_group.get("params", ()))
            or float(saved_group.get("initial_lr", saved_group.get("lr", 0.0)))
            != float(runtime_group.get("initial_lr", runtime_group.get("lr", 0.0)))
        ):
            raise ValueError("optimizer group topology differs from runtime")
    actual_hash = sha256_json(_optimizer_topology(saved))
    if actual_hash != expected_hash:
        raise ValueError("optimizer topology hash differs from checkpoint metadata")
    saved_slots = saved.get("state")
    if not isinstance(saved_slots, Mapping):
        raise ValueError("optimizer state topology is invalid")
    for saved_group, live_group in zip(saved_groups, optimizer.param_groups):
        for identifier, parameter in zip(
            saved_group["params"], live_group["params"]
        ):
            slot = saved_slots.get(identifier, {})
            if not isinstance(slot, Mapping):
                raise ValueError("optimizer slot topology is invalid")
            for name, value in slot.items():
                if (
                    name != "step"
                    and isinstance(value, torch.Tensor)
                    and tuple(value.shape) != tuple(parameter.shape)
                ):
                    raise ValueError(
                        "optimizer slot tensor topology differs from runtime"
                    )


def _validate_scheduler_runtime(
    saved: Mapping[str, Any],
    scheduler: Any,
    *,
    expected_hash: str,
) -> None:
    if sha256_json(_scheduler_topology(saved)) != expected_hash:
        raise ValueError("scheduler topology hash differs from checkpoint metadata")
    runtime = scheduler.state_dict()
    if _scheduler_topology(saved) != _scheduler_topology(runtime):
        raise ValueError("scheduler state topology differs from runtime")


def _validate_scaler_runtime(saved: Mapping[str, Any], scaler: Any | None) -> None:
    if not isinstance(saved, Mapping) or set(saved) != {"enabled", "state_dict"}:
        raise ValueError("checkpoint scaler payload is invalid")
    if bool(saved["enabled"]) != (scaler is not None):
        raise ValueError("checkpoint scaler mode differs from runtime")
    if scaler is not None:
        saved_state = saved["state_dict"]
        runtime_state = scaler.state_dict()
        if not isinstance(saved_state, Mapping) or set(saved_state) != set(runtime_state):
            raise ValueError("checkpoint scaler topology differs from runtime")


def _validate_persisted_steps(
    optimizer_state: Mapping[str, Any],
    scheduler_state: Mapping[str, Any],
    cursor: BatonTrainingCursor,
) -> None:
    slots = optimizer_state.get("state")
    groups = optimizer_state.get("param_groups")
    if not isinstance(slots, Mapping) or not isinstance(groups, list):
        raise ValueError("optimizer state topology is invalid")
    parameter_ids = [
        identifier
        for group in groups
        if isinstance(group, Mapping)
        for identifier in group.get("params", ())
    ]
    if cursor.global_step > 0 and set(parameter_ids) != set(slots):
        raise ValueError("optimizer step state is incomplete for the cursor")
    for slot in slots.values():
        if not isinstance(slot, Mapping):
            raise ValueError("optimizer state topology is invalid")
        step = slot.get("step")
        if isinstance(step, torch.Tensor):
            if step.numel() != 1:
                raise ValueError("optimizer step state is invalid")
            step = int(step.item())
        if type(step) is not int or step != cursor.global_step:
            raise ValueError("optimizer step differs from checkpoint cursor")
    if scheduler_state.get("last_epoch") != cursor.global_step:
        raise ValueError("scheduler step differs from checkpoint cursor")
    step_count = scheduler_state.get("_step_count")
    if step_count is not None and step_count != cursor.global_step + 1:
        raise ValueError("scheduler step differs from checkpoint cursor")


def load_baton_checkpoint(
    checkpoint_dir: Path,
    *,
    planner: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    expected_contract: BatonCheckpointMetadata,
    scaler: Any | None = None,
    distributed_rank: int | None = None,
    world_size: int | None = None,
) -> BatonResumeState:
    """Validate all provenance, hashes, and topology before mutating runtime state."""

    checkpoint = Path(checkpoint_dir)
    if not isinstance(planner, nn.Module):
        raise TypeError("planner must be a torch module")
    if not isinstance(expected_contract, BatonCheckpointMetadata):
        raise TypeError("expected_contract must be BatonCheckpointMetadata")
    missing = [
        name
        for name in (*_CHECKPOINT_FILES, "manifest.json")
        if not (checkpoint / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"incomplete Baton checkpoint {checkpoint}: missing {missing}"
        )

    # Metadata/provenance is intentionally the first parsed and validated state.
    raw_metadata = _load_json(checkpoint / "metadata.json", label="metadata")
    metadata = BatonCheckpointMetadata.from_dict(raw_metadata)
    _validate_expected_contract(metadata, expected_contract)

    manifest = _load_json(checkpoint / "manifest.json", label="manifest")
    hashes = manifest.get("files")
    if (
        manifest.get("format_version") != 1
        or not isinstance(hashes, Mapping)
        or set(hashes) != set(_CHECKPOINT_FILES)
    ):
        raise ValueError("checkpoint file hash manifest is invalid")
    for name in _CHECKPOINT_FILES:
        actual_hash = sha256_file(checkpoint / name)
        if hashes[name] != actual_hash:
            raise ValueError(f"checkpoint hash mismatch for {name}")

    cursor = BatonTrainingCursor.from_dict(
        _load_json(checkpoint / "cursor.json", label="cursor")
    )
    rng_payload = torch.load(
        checkpoint / "rank_rng.pt", weights_only=True, map_location="cpu"
    )
    if (
        not isinstance(rng_payload, Mapping)
        or set(rng_payload) != {"format_version", "world_size", "states"}
        or rng_payload["format_version"] != 1
        or type(rng_payload["world_size"]) is not int
        or rng_payload["world_size"] <= 0
        or not isinstance(rng_payload["states"], list)
        or len(rng_payload["states"]) != rng_payload["world_size"]
    ):
        raise ValueError("checkpoint RNG world-size/rank coverage is invalid")
    saved_world_size = int(rng_payload["world_size"])
    for rank, state in enumerate(rng_payload["states"]):
        _validate_rank_rng_state(state, expected_rank=rank)
    if sha256_file(checkpoint / "rank_rng.pt") != metadata.rng_state_hash:
        raise ValueError("checkpoint RNG hash differs from metadata")
    _validate_cursor_metadata(cursor, metadata, world_size=saved_world_size)

    if distributed_rank is None:
        distributed_rank = int(os.environ.get("RANK", "0"))
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", str(saved_world_size)))
    if (
        type(distributed_rank) is not int
        or type(world_size) is not int
        or world_size != saved_world_size
        or not 0 <= distributed_rank < world_size
    ):
        raise ValueError("checkpoint RNG world-size/rank coverage mismatch")

    optimizer_state = torch.load(
        checkpoint / "optimizer.pt", weights_only=True, map_location="cpu"
    )
    scheduler_state = torch.load(
        checkpoint / "scheduler.pt", weights_only=True, map_location="cpu"
    )
    scaler_state = torch.load(
        checkpoint / "scaler.pt", weights_only=True, map_location="cpu"
    )
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("optimizer state topology is invalid")
    if not isinstance(scheduler_state, Mapping):
        raise ValueError("scheduler state topology is invalid")
    optimizer_hash = sha256_json(
        _optimizer_topology(optimizer_state)
    )
    if optimizer_hash != metadata.optimizer_topology_hash:
        raise ValueError("optimizer topology hash differs from checkpoint metadata")
    if (
        sha256_json(_scheduler_topology(scheduler_state))
        != metadata.scheduler_topology_hash
    ):
        raise ValueError("scheduler topology hash differs from checkpoint metadata")
    _validate_persisted_steps(optimizer_state, scheduler_state, cursor)
    _validate_model_topology(checkpoint, planner)
    if optimizer is None:
        if optimizer_state.get("state") or optimizer_state.get("param_groups"):
            pass
    else:
        _validate_optimizer_runtime(
            optimizer_state,
            optimizer,
            expected_hash=metadata.optimizer_topology_hash,
        )
    if scheduler is not None:
        _validate_scheduler_runtime(
            scheduler_state,
            scheduler,
            expected_hash=metadata.scheduler_topology_hash,
        )
    _validate_scaler_runtime(scaler_state, scaler)

    # No model/optimizer/scheduler/scaler/RNG mutation occurs above this line.
    load_model(planner, str(checkpoint / "planner.safetensors"), strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
    if scaler is not None:
        scaler.load_state_dict(scaler_state["state_dict"])
    selected_rng = rng_payload["states"][distributed_rank]
    restore_rank_rng_state(selected_rng)
    return BatonResumeState(
        metadata=metadata,
        cursor=cursor,
        rank_rng_state=selected_rng,
    )
