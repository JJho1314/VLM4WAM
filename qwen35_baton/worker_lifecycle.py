"""Pure worker lifecycle scheduling for Stage-1 training."""

from __future__ import annotations

from typing import Any

import torch


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
