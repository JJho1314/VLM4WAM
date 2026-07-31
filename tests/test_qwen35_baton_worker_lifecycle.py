from __future__ import annotations

import pytest

from qwen35_baton.worker_lifecycle import should_restart_workers


@pytest.mark.parametrize(
    ("completed_epoch", "interval", "expected"),
    [
        (0, None, False),
        (98, 100, False),
        (99, 100, True),
        (100, 100, False),
        (199, 100, True),
    ],
)
def test_worker_restart_schedule_uses_absolute_completed_epochs(
    completed_epoch: int,
    interval: int | None,
    expected: bool,
) -> None:
    assert (
        should_restart_workers(
            completed_epoch=completed_epoch,
            interval_epochs=interval,
        )
        is expected
    )
