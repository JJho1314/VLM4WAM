"""Pure worker lifecycle scheduling for Stage-1 training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
from typing import Any

import torch


def validate_recycle_statuses(
    statuses: Sequence[Mapping[str, Any]],
    *,
    world_size: int,
    completed_epoch: int,
) -> float:
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if len(statuses) != world_size:
        raise RuntimeError(
            "worker recycling did not publish exactly one status per rank"
        )
    ranks: list[int] = []
    elapsed_values: list[float] = []
    for status in statuses:
        rank = status.get("rank")
        elapsed = status.get("elapsed")
        if (
            type(rank) is not int
            or status.get("epoch") != completed_epoch
            or status.get("recycled") is not True
            or status.get("error") is not None
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0.0
        ):
            raise RuntimeError(f"worker recycling failed closed: {statuses!r}")
        ranks.append(rank)
        elapsed_values.append(float(elapsed))
    if sorted(ranks) != list(range(world_size)):
        raise RuntimeError(
            "worker recycling did not publish one unique status per rank"
        )
    return max(elapsed_values)


def append_worker_lifecycle_event(
    path: Path,
    *,
    completed_epoch: int,
    next_epoch: int,
    restart_count: int,
    interval_epochs: int,
    elapsed_seconds: float,
) -> None:
    record = {
        "schema_version": 1,
        "event": "dataloader_workers_restarted",
        "completed_epoch": completed_epoch,
        "next_epoch": next_epoch,
        "restart_count": restart_count,
        "interval_epochs": interval_epochs,
        "elapsed_seconds": elapsed_seconds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def should_restart_workers(
    *,
    completed_epoch: int,
    interval_epochs: int | None,
) -> bool:
    if type(completed_epoch) is not int or completed_epoch < 0:
        raise ValueError("completed_epoch must be a non-negative integer")
    if interval_epochs is None:
        return False
    if type(interval_epochs) is not int or interval_epochs <= 0:
        raise ValueError("interval_epochs must be None or a positive integer")
    return (completed_epoch + 1) % interval_epochs == 0


def _underlying_torch_dataloader(batches: Any) -> torch.utils.data.DataLoader:
    current = batches
    seen: set[int] = set()
    while True:
        identity = id(current)
        if identity in seen:
            raise RuntimeError("DataLoader wrapper chain contains a cycle")
        seen.add(identity)
        nested = getattr(current, "base_dataloader", None)
        if nested is not None:
            current = nested
            continue
        if isinstance(current, torch.utils.data.DataLoader):
            return current
        raise RuntimeError(
            "cannot recycle workers for unsupported loader type "
            f"{type(current).__module__}.{type(current).__qualname__}"
        )


def recycle_persistent_dataloader_workers(batches: Any) -> bool:
    loader = _underlying_torch_dataloader(batches)
    if loader.num_workers == 0 or not loader.persistent_workers:
        return False
    missing = object()
    iterator = getattr(loader, "_iterator", missing)
    if iterator is missing:
        raise RuntimeError("persistent DataLoader does not expose iterator state")
    if iterator is None:
        return False
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if not callable(shutdown):
        raise RuntimeError(
            "active persistent DataLoader iterator does not expose worker shutdown"
        )
    shutdown()
    loader._iterator = None
    return True
