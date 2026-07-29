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
import stat
import tempfile
from typing import Any

import numpy as np
from safetensors import safe_open
from safetensors.torch import (
    _TYPES,
    _remove_duplicate_names,
    load_model,
    save_model,
)
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
        "numpy_state": torch.from_numpy(numpy_state[1].copy()).to(torch.int64),
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
        or numpy_state.dtype not in (torch.uint32, torch.int64)
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


def _validate_cuda_rng_runtime(state: Mapping[str, Any]) -> None:
    """Prove saved CUDA streams are restorable without mutating global RNG."""

    cuda_states = state["torch_cuda"]
    if cuda_states and not torch.cuda.is_available():
        raise ValueError("checkpoint contains CUDA RNG streams but CUDA is unavailable")
    runtime_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(cuda_states) != runtime_count:
        raise ValueError("checkpoint CUDA RNG stream count differs from runtime")
    for index, saved_state in enumerate(cuda_states):
        try:
            temporary = torch.Generator(device=f"cuda:{index}")
            temporary.set_state(saved_state)
        except Exception as error:
            raise ValueError(
                f"checkpoint CUDA RNG state is invalid for runtime device {index}"
            ) from error


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
) -> list[dict[str, Any]]:
    groups = optimizer_state.get("param_groups")
    slots = optimizer_state.get("state")
    if not isinstance(groups, list) or not isinstance(slots, Mapping):
        raise ValueError("optimizer state topology is invalid")
    topology = []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise ValueError("optimizer state topology is invalid")
        parameter_shapes = []
        for position, identifier in enumerate(group["params"]):
            names = group.get("parameter_names")
            shapes = group.get("parameter_shapes")
            dtypes = group.get("parameter_dtypes")
            if (
                not isinstance(names, list)
                or not isinstance(shapes, list)
                or not isinstance(dtypes, list)
                or not len(names) == len(shapes) == len(dtypes) == len(group["params"])
            ):
                raise ValueError(
                    "optimizer parameter names/shapes/dtypes topology is invalid"
                )
            shape = shapes[position]
            slot = slots.get(identifier, {})
            if not isinstance(slot, Mapping):
                raise ValueError("optimizer slot topology is invalid")
            slot_shapes = {
                name: (
                    {"shape": list(value.shape), "dtype": str(value.dtype)}
                    if isinstance(value, torch.Tensor)
                    else {"type": type(value).__name__}
                )
                for name, value in sorted(slot.items())
            }
            parameter_shapes.append(
                {
                    "name": names[position],
                    "shape": shape,
                    "dtype": dtypes[position],
                    "slots": slot_shapes,
                }
            )
        topology.append(
            {
                "name": group.get("name"),
                "parameter_count": len(group["params"]),
                "initial_lr": float(group.get("initial_lr", group.get("lr", 0.0))),
                "parameters": parameter_shapes,
            }
        )
    names = [
        parameter["name"]
        for group in topology
        for parameter in group["parameters"]
    ]
    if (
        any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError("optimizer parameter names must be canonical and unique")
    return topology


def _optimizer_parameter_contract(
    planner: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> list[dict[str, list[Any]]]:
    canonical_names = {
        id(parameter): name for name, parameter in planner.named_parameters()
    }
    contracts: list[dict[str, list[Any]]] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        names: list[str] = []
        shapes: list[list[int]] = []
        dtypes: list[str] = []
        for parameter in group["params"]:
            if not isinstance(parameter, nn.Parameter):
                raise ValueError("optimizer runtime group contains a non-parameter")
            identifier = id(parameter)
            if identifier in seen:
                raise ValueError("optimizer parameters must be unique across groups")
            seen.add(identifier)
            parameter_name = canonical_names.get(identifier)
            if not isinstance(parameter_name, str) or not parameter_name:
                raise ValueError(
                    "optimizer parameter names must be canonical and unique"
                )
            names.append(parameter_name)
            shapes.append(list(parameter.shape))
            dtypes.append(str(parameter.dtype))
        contracts.append(
            {
                "parameter_names": names,
                "parameter_shapes": shapes,
                "parameter_dtypes": dtypes,
            }
        )
    return contracts


def _annotate_optimizer_state(
    optimizer_state: dict[str, Any],
    *,
    planner: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    groups = optimizer_state.get("param_groups")
    if not isinstance(groups, list) or len(groups) != len(optimizer.param_groups):
        raise ValueError("optimizer group topology differs from runtime")
    contracts = _optimizer_parameter_contract(planner, optimizer)
    for group, live_group, contract in zip(
        groups, optimizer.param_groups, contracts
    ):
        group.update(contract)
        live_group.update(contract)


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
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return {"type": type(value).__name__}

    return {str(key): describe(value) for key, value in sorted(state.items())}


def _validate_scheduler_state_values(state: Mapping[str, Any]) -> None:
    contract = state.get("baton_contract")
    if not isinstance(contract, Mapping) or set(contract) != {
        "schedule_type",
        "warmup_steps",
        "max_steps",
        "max_consecutive_skipped_updates",
        "base_lrs",
    }:
        raise ValueError("scheduler contract is missing or invalid")
    warmup_steps = contract["warmup_steps"]
    max_steps = contract["max_steps"]
    max_consecutive_skipped_updates = contract[
        "max_consecutive_skipped_updates"
    ]
    base_lrs = contract["base_lrs"]
    last_epoch = state.get("last_epoch")
    current_lrs = state.get("_last_lr")
    if (
        contract["schedule_type"] != "linear_warmup_cosine_v1"
        or type(warmup_steps) is not int
        or type(max_steps) is not int
        or type(max_consecutive_skipped_updates) is not int
        or max_consecutive_skipped_updates <= 0
        or warmup_steps < 0
        or max_steps <= warmup_steps
        or not isinstance(base_lrs, list)
        or not base_lrs
        or any(type(value) not in (int, float) or value <= 0 for value in base_lrs)
        or state.get("base_lrs") != base_lrs
        or type(last_epoch) is not int
        or last_epoch < 0
        or not isinstance(current_lrs, list)
        or len(current_lrs) != len(base_lrs)
    ):
        raise ValueError("scheduler contract or current LR state is invalid")
    if warmup_steps and last_epoch <= warmup_steps:
        multiplier = float(last_epoch) / float(warmup_steps)
    else:
        progress = min(
            1.0,
            max(
                0.0,
                float(last_epoch - warmup_steps)
                / float(max_steps - warmup_steps),
            ),
        )
        multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    for base_lr, current_lr in zip(base_lrs, current_lrs):
        if not math.isclose(
            float(current_lr),
            float(base_lr) * multiplier,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("scheduler current LR differs from its saved contract")


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


def planner_safetensors_topology(path: str | Path) -> dict[str, Any]:
    """Read ordered tensor shape/dtype and alias policy from a safe header."""

    artifact = Path(path)
    with safe_open(artifact, framework="pt", device="cpu") as handle:
        names = tuple(handle.keys())
        metadata = dict(handle.metadata() or {})
        tensors = [
            {
                "name": name,
                "shape": list(handle.get_slice(name).get_shape()),
                "dtype": handle.get_slice(name).get_dtype(),
            }
            for name in names
        ]
    aliases = {
        name: target
        for name, target in sorted(metadata.items())
        if name != "format" and name not in names and target in names
    }
    contract = {
        "format_version": 1,
        "tensors": tensors,
        "aliases": aliases,
    }
    validate_planner_topology_contract(contract)
    return contract


def planner_module_topology(planner: nn.Module) -> dict[str, Any]:
    """Derive the exact safe-model header topology without writing a checkpoint."""

    if not isinstance(planner, nn.Module):
        raise TypeError("planner must be a torch module")
    state = planner.state_dict()
    removals = _remove_duplicate_names(state)
    aliases = {
        removed: kept
        for kept, removed_names in removals.items()
        for removed in removed_names
    }
    removed = set(aliases)
    dtype_names = {dtype: name for name, dtype in _TYPES.items()}
    tensors = []
    for name in sorted(set(state).difference(removed)):
        value = state[name]
        dtype = dtype_names.get(value.dtype)
        if dtype is None:
            raise ValueError(
                f"planner tensor {name} has unsupported safetensors dtype {value.dtype}"
            )
        tensors.append(
            {"name": name, "shape": list(value.shape), "dtype": dtype}
        )
    contract = {
        "format_version": 1,
        "tensors": tensors,
        "aliases": dict(sorted(aliases.items())),
    }
    validate_planner_topology_contract(contract)
    return contract


def trusted_planner_topology_payload(
    topology: Mapping[str, Any],
) -> dict[str, Any]:
    """Wrap a canonical topology in its independently verifiable root envelope."""

    validate_planner_topology_contract(topology)
    canonical = json.loads(
        json.dumps(topology, sort_keys=True, allow_nan=False)
    )
    return {
        "format_version": 1,
        "topology": canonical,
        "sha256": sha256_json(canonical),
    }


def load_trusted_planner_topology(
    source: str | Path | Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Load and validate a root topology envelope or an injected raw contract."""

    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        path = Path(source).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"trusted planner topology JSON is invalid: {path}") from error
        if not isinstance(payload, Mapping):
            raise ValueError("trusted planner topology must contain a JSON object")
        payload = dict(payload)
    if set(payload) == {"format_version", "topology", "sha256"}:
        if payload["format_version"] != 1 or not isinstance(
            payload["topology"], Mapping
        ):
            raise ValueError("trusted planner topology envelope is invalid")
        topology = dict(payload["topology"])
        validate_planner_topology_contract(topology)
        actual_hash = sha256_json(topology)
        if payload["sha256"] != actual_hash:
            raise ValueError("trusted planner topology envelope hash is invalid")
        return topology, actual_hash
    validate_planner_topology_contract(payload)
    return payload, sha256_json(payload)


def publish_trusted_planner_topology(
    path: str | Path,
    topology: Mapping[str, Any],
) -> str:
    """Atomically publish a fsynced topology contract outside checkpoints."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = trusted_planner_topology_payload(topology)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.incomplete-",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o644)
        _json_write(temporary, payload)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            anchored, anchored_hash = load_trusted_planner_topology(destination)
            if (
                anchored != payload["topology"]
                or anchored_hash != payload["sha256"]
            ):
                raise ValueError(
                    "existing trusted planner topology differs from publisher"
                )
        published_mode = stat.S_IMODE(destination.stat().st_mode)
        if published_mode != 0o644:
            raise PermissionError(
                "trusted planner topology mode must be 0644, "
                f"got {published_mode:04o}"
            )
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
            _fsync_directory(destination.parent)
    return str(payload["sha256"])


def validate_planner_topology_contract(payload: Any) -> None:
    """Reject incomplete, unordered, duplicate, or malformed topology records."""

    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"format_version", "tensors", "aliases"}
        or payload["format_version"] != 1
        or not isinstance(payload["tensors"], list)
        or not payload["tensors"]
        or not isinstance(payload["aliases"], Mapping)
    ):
        raise ValueError("planner topology contract is invalid")
    names: list[str] = []
    for tensor in payload["tensors"]:
        if (
            not isinstance(tensor, Mapping)
            or set(tensor) != {"name", "shape", "dtype"}
            or not isinstance(tensor["name"], str)
            or not tensor["name"]
            or not isinstance(tensor["shape"], list)
            or any(type(value) is not int or value < 0 for value in tensor["shape"])
            or not isinstance(tensor["dtype"], str)
            or not tensor["dtype"]
        ):
            raise ValueError("planner topology tensor entry is invalid")
        names.append(tensor["name"])
    if len(names) != len(set(names)):
        raise ValueError("planner topology tensor names must be unique")
    if names != sorted(names):
        raise ValueError("planner topology tensor names must be ordered")
    for alias, target in payload["aliases"].items():
        if (
            not isinstance(alias, str)
            or not alias
            or alias in names
            or not isinstance(target, str)
            or target not in names
        ):
            raise ValueError("planner topology alias policy is invalid")


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
    trusted_topology = planner_module_topology(planner)
    trusted_hash = sha256_json(trusted_topology)
    if trusted_hash != metadata.planner_topology_hash:
        raise ValueError(
            "planner topology hash differs from trusted checkpoint metadata"
        )
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
        written_topology = planner_safetensors_topology(
            staging / "planner.safetensors"
        )
        if written_topology != trusted_topology:
            raise ValueError(
                "written planner safetensors topology differs from trusted metadata"
            )
        optimizer_state = optimizer.state_dict()
        _annotate_optimizer_state(
            optimizer_state,
            planner=planner,
            optimizer=optimizer,
        )
        scheduler_state = scheduler.state_dict()
        _validate_scheduler_state_values(scheduler_state)
        _validate_optimizer_scheduler_lrs(optimizer_state, scheduler_state)
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
            {"format_version": 2, "files": hashes},
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
    planner: nn.Module,
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
    runtime_contracts = _optimizer_parameter_contract(planner, optimizer)
    for saved_group, runtime_group, live_group, runtime_contract in zip(
        saved_groups,
        runtime_groups,
        optimizer.param_groups,
        runtime_contracts,
    ):
        if (
            saved_group.get("name") != runtime_group.get("name")
            or len(saved_group.get("params", ())) != len(runtime_group.get("params", ()))
            or len(saved_group.get("params", ())) != len(live_group.get("params", ()))
            or float(saved_group.get("initial_lr", saved_group.get("lr", 0.0)))
            != float(runtime_group.get("initial_lr", runtime_group.get("lr", 0.0)))
        ):
            raise ValueError("optimizer group topology differs from runtime")
        for field in (
            "parameter_names",
            "parameter_shapes",
            "parameter_dtypes",
        ):
            if saved_group.get(field) != runtime_contract[field]:
                raise ValueError(
                    "optimizer parameter names, order, shape, or dtype "
                    "differs from runtime"
                )
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
    _validate_scheduler_state_values(saved)
    if sha256_json(_scheduler_topology(saved)) != expected_hash:
        raise ValueError("scheduler topology hash differs from checkpoint metadata")
    runtime = scheduler.state_dict()
    _validate_scheduler_state_values(runtime)
    if saved.get("baton_contract") != runtime.get("baton_contract"):
        raise ValueError("scheduler contract differs from runtime")
    if set(saved) != set(runtime):
        raise ValueError("scheduler state topology differs from runtime")


def _validate_optimizer_scheduler_lrs(
    optimizer_state: Mapping[str, Any],
    scheduler_state: Mapping[str, Any],
) -> None:
    groups = optimizer_state.get("param_groups")
    contract = scheduler_state.get("baton_contract")
    current_lrs = scheduler_state.get("_last_lr")
    if (
        not isinstance(groups, list)
        or not isinstance(contract, Mapping)
        or not isinstance(contract.get("base_lrs"), list)
        or not isinstance(current_lrs, list)
        or not len(groups) == len(contract["base_lrs"]) == len(current_lrs)
    ):
        raise ValueError("optimizer and scheduler LR group topology differs")
    names = [group.get("name") for group in groups if isinstance(group, Mapping)]
    if (
        len(names) != len(groups)
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError("optimizer LR group names must be ordered and unique")
    for group, base_lr, current_lr in zip(
        groups, contract["base_lrs"], current_lrs
    ):
        saved_base_lr = group.get("initial_lr", group.get("lr"))
        saved_current_lr = group.get("lr")
        if (
            type(saved_base_lr) not in (int, float)
            or type(saved_current_lr) not in (int, float)
            or not math.isclose(
                float(saved_base_lr),
                float(base_lr),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError(
                "optimizer initial LR differs from scheduler base LR contract"
            )
        if not math.isclose(
            float(saved_current_lr),
            float(current_lr),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "optimizer current LR differs from scheduler current LR contract"
            )


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
    slot_ids = set(slots)
    parameter_id_set = set(parameter_ids)
    if cursor.global_step > 0 and not slot_ids:
        raise ValueError("optimizer step state is empty for a nonzero cursor")
    if not slot_ids.issubset(parameter_id_set):
        raise ValueError("optimizer step state contains parameters outside its groups")
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
    expected_sampler_seed: int,
    expected_microbatches_per_epoch: int,
    expected_planner_topology: str | Path | Mapping[str, Any],
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
        manifest.get("format_version") != 2
        or not isinstance(hashes, Mapping)
        or set(hashes) != set(_CHECKPOINT_FILES)
    ):
        raise ValueError("checkpoint file hash manifest is invalid")
    for name in _CHECKPOINT_FILES:
        actual_hash = sha256_file(checkpoint / name)
        if hashes[name] != actual_hash:
            raise ValueError(f"checkpoint hash mismatch for {name}")

    trusted_topology, trusted_hash = load_trusted_planner_topology(
        expected_planner_topology
    )
    if trusted_hash != metadata.planner_topology_hash:
        raise ValueError(
            "trusted planner topology hash differs from checkpoint metadata"
        )
    actual_topology = planner_safetensors_topology(
        checkpoint / "planner.safetensors"
    )
    if sha256_json(actual_topology) != metadata.planner_topology_hash:
        raise ValueError(
            "planner safetensors topology hash differs from checkpoint metadata"
        )
    if actual_topology != trusted_topology:
        raise ValueError(
            "planner safetensors topology differs from trusted planner topology"
        )

    cursor = BatonTrainingCursor.from_dict(
        _load_json(checkpoint / "cursor.json", label="cursor")
    )
    if cursor.sampler_seed != expected_sampler_seed:
        raise ValueError(
            "checkpoint sampler seed differs from runtime data contract"
        )
    if cursor.microbatches_per_epoch != expected_microbatches_per_epoch:
        raise ValueError(
            "checkpoint microbatches per epoch differs from runtime data contract"
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
    selected_rng = rng_payload["states"][distributed_rank]
    _validate_cuda_rng_runtime(selected_rng)

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
    _validate_persisted_steps(optimizer_state, scheduler_state, cursor)
    _validate_scheduler_state_values(scheduler_state)
    _validate_optimizer_scheduler_lrs(optimizer_state, scheduler_state)
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
    _validate_model_topology(checkpoint, planner)
    if optimizer is None:
        if optimizer_state.get("state") or optimizer_state.get("param_groups"):
            pass
    else:
        _validate_optimizer_runtime(
            optimizer_state,
            optimizer,
            planner=planner,
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
    restore_rank_rng_state(selected_rng)
    return BatonResumeState(
        metadata=metadata,
        cursor=cursor,
        rank_rng_state=selected_rng,
    )
