"""Pure worker lifecycle scheduling for Stage-1 training."""

from __future__ import annotations


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
