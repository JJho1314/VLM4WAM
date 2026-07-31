from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from accelerate import Accelerator

from qwen35_baton.cli.train_semantic_planner import (
    Stage1TrainingConfig,
    build_stage1_dataloader,
)
from qwen35_baton.worker_lifecycle import (
    recycle_persistent_dataloader_workers,
    should_restart_workers,
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
