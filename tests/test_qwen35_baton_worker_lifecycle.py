from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re

import pytest
import torch
from accelerate import Accelerator

from qwen35_baton.cli.train_semantic_planner import (
    Stage1TrainingConfig,
    build_stage1_dataloader,
)
from qwen35_baton.worker_lifecycle import (
    append_worker_lifecycle_event,
    recycle_persistent_dataloader_workers,
    should_restart_workers,
    validate_recycle_statuses,
)


def _config(output_dir: Path) -> Stage1TrainingConfig:
    return Stage1TrainingConfig(
        output_dir=str(output_dir),
        qwen_model_path="unused",
        qwen_processor_path="unused",
        qwen_tokenizer_path="unused",
        siglip2_model_path="unused",
        siglip2_config_hash="0" * 64,
        siglip2_artifact_hash="0" * 64,
        hdf5_manifest_path="unused",
        hdf5_manifest_hash="0" * 64,
        dataset_statistics_path="unused",
        max_steps=1,
        warmup_steps=0,
        save_every=1,
        initial_save_step=None,
        log_every=1,
        mixed_precision="no",
        tiny_test=True,
    )


def _worker_pids(loader: torch.utils.data.DataLoader) -> tuple[int, ...]:
    iterator = loader._iterator
    assert iterator is not None
    return tuple(worker.pid for worker in iterator._workers)


def test_recycle_status_validation_returns_the_slowest_rank_duration() -> None:
    statuses = [
        {"rank": 0, "epoch": 99, "recycled": True, "error": None, "elapsed": 1.25},
        {"rank": 1, "epoch": 99, "recycled": True, "error": None, "elapsed": 1.75},
    ]
    assert validate_recycle_statuses(
        statuses,
        world_size=2,
        completed_epoch=99,
    ) == 1.75


@pytest.mark.parametrize(
    "statuses",
    [
        [{"rank": 0, "epoch": 99, "recycled": False, "error": None, "elapsed": 1.0}],
        [{"rank": 0, "epoch": 98, "recycled": True, "error": None, "elapsed": 1.0}],
        [
            {
                "rank": 0,
                "epoch": 99,
                "recycled": False,
                "error": "shutdown failed",
                "elapsed": 1.0,
            }
        ],
    ],
)
def test_recycle_status_validation_fails_closed(
    statuses: list[dict[str, object]],
) -> None:
    with pytest.raises(RuntimeError, match="worker recycl"):
        validate_recycle_statuses(
            statuses,
            world_size=1,
            completed_epoch=99,
        )


def test_worker_lifecycle_event_is_separate_from_training_metrics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "worker_lifecycle.jsonl"
    append_worker_lifecycle_event(
        path,
        completed_epoch=99,
        next_epoch=100,
        restart_count=1,
        interval_epochs=100,
        elapsed_seconds=1.75,
    )
    assert json.loads(path.read_text()) == {
        "completed_epoch": 99,
        "elapsed_seconds": 1.75,
        "event": "dataloader_workers_restarted",
        "interval_epochs": 100,
        "next_epoch": 100,
        "restart_count": 1,
        "schema_version": 1,
    }


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


def test_recycling_preserves_order_and_replaces_real_worker_pids(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path),
        per_device_batch=1,
        num_workers=2,
        persistent_workers=True,
    )
    loader = build_stage1_dataloader(
        torch.utils.data.TensorDataset(torch.arange(8)),
        collate_fn=None,
        config=config,
    )
    try:
        first_values = [batch[0].item() for batch in loader]
        first_workers = tuple(loader._iterator._workers)
        first_pids = _worker_pids(loader)

        second_values = [batch[0].item() for batch in loader]
        assert _worker_pids(loader) == first_pids
        assert second_values == first_values

        assert recycle_persistent_dataloader_workers(loader) is True
        assert loader._iterator is None
        assert all(not worker.is_alive() for worker in first_workers)

        third_values = [batch[0].item() for batch in loader]
        third_pids = _worker_pids(loader)
        assert set(first_pids).isdisjoint(third_pids)
        assert third_values == first_values
    finally:
        recycle_persistent_dataloader_workers(loader)


def test_recycling_without_an_active_iterator_is_a_noop(tmp_path: Path) -> None:
    loader = build_stage1_dataloader(
        torch.utils.data.TensorDataset(torch.arange(4)),
        collate_fn=None,
        config=replace(
            _config(tmp_path),
            per_device_batch=1,
            num_workers=1,
            persistent_workers=True,
        ),
    )
    assert recycle_persistent_dataloader_workers(loader) is False


def test_recycling_rejects_missing_private_iterator_state(tmp_path: Path) -> None:
    loader = build_stage1_dataloader(
        torch.utils.data.TensorDataset(torch.arange(4)),
        collate_fn=None,
        config=replace(
            _config(tmp_path),
            per_device_batch=1,
            num_workers=1,
            persistent_workers=True,
        ),
    )
    del loader._iterator

    loader_type = f"{type(loader).__module__}.{type(loader).__qualname__}"
    with pytest.raises(
        RuntimeError,
        match=rf"{re.escape(loader_type)} does not expose iterator state",
    ):
        recycle_persistent_dataloader_workers(loader)


def test_recycling_rejects_missing_private_worker_shutdown(tmp_path: Path) -> None:
    class IteratorWithoutShutdown:
        pass

    loader = build_stage1_dataloader(
        torch.utils.data.TensorDataset(torch.arange(4)),
        collate_fn=None,
        config=replace(
            _config(tmp_path),
            per_device_batch=1,
            num_workers=1,
            persistent_workers=True,
        ),
    )
    loader._iterator = IteratorWithoutShutdown()

    loader_type = f"{type(loader).__module__}.{type(loader).__qualname__}"
    iterator_type = (
        f"{type(loader._iterator).__module__}.{type(loader._iterator).__qualname__}"
    )
    with pytest.raises(
        RuntimeError,
        match=(
            rf"{re.escape(loader_type)} iterator {re.escape(iterator_type)} "
            "does not expose worker shutdown"
        ),
    ):
        recycle_persistent_dataloader_workers(loader)


def test_recycling_rejects_an_unsupported_wrapper() -> None:
    class UnsupportedLoader:
        pass

    with pytest.raises(RuntimeError, match="UnsupportedLoader"):
        recycle_persistent_dataloader_workers(UnsupportedLoader())


def test_accelerate_prepared_loader_recycles_the_underlying_workers(
    tmp_path: Path,
) -> None:
    loader = build_stage1_dataloader(
        torch.utils.data.TensorDataset(torch.arange(8)),
        collate_fn=None,
        config=replace(
            _config(tmp_path),
            per_device_batch=1,
            num_workers=1,
            persistent_workers=True,
        ),
    )
    prepared = Accelerator(cpu=True).prepare_data_loader(loader)
    wrapped_loader = prepared.base_dataloader
    try:
        next(iter(prepared))
        old_pids = _worker_pids(wrapped_loader)
        assert recycle_persistent_dataloader_workers(prepared) is True
        assert wrapped_loader._iterator is None
        next(iter(prepared))
        assert set(old_pids).isdisjoint(_worker_pids(wrapped_loader))
    finally:
        recycle_persistent_dataloader_workers(prepared)
