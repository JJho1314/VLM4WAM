from __future__ import annotations

from dataclasses import replace
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import random
import shutil
import stat
import tempfile
import threading
from typing import Any

import numpy as np
import pytest
from safetensors.torch import save_model
import torch
import torch.nn as nn

import qwen35_baton.checkpoint as checkpoint_module
from qwen35_baton.checkpoint import (
    BatonTrainingCursor,
    capture_rank_rng_state,
    load_baton_checkpoint as _load_baton_checkpoint,
    load_trusted_planner_topology,
    migrate_legacy_head_checkpoint_v4,
    planner_module_topology,
    planner_safetensors_topology,
    publish_trusted_planner_topology,
    save_baton_checkpoint,
    trusted_planner_topology_payload,
)
from qwen35_baton.cli.train_semantic_planner import BatonCosineWarmupScheduler
from qwen35_baton.config import BatonCheckpointMetadata
from qwen35_baton.hashing import sha256_file, sha256_json


def test_rank_rng_state_uses_pickle_safe_lossless_numpy_encoding() -> None:
    state = capture_rank_rng_state(distributed_rank=0)

    assert state["numpy_state"].dtype == torch.int64
    restored = pickle.loads(pickle.dumps(state))
    assert torch.equal(restored["numpy_state"], state["numpy_state"])
    assert torch.equal(restored["torch_cpu"], state["torch_cpu"])


def test_persisted_steps_allow_adamw_lazy_state_for_unused_parameters() -> None:
    from qwen35_baton.checkpoint import _validate_persisted_steps

    cursor = BatonTrainingCursor(
        global_step=2,
        epoch=0,
        consumed_microbatches=2,
        microbatches_per_epoch=4,
        sampler_seed=7,
    )
    _validate_persisted_steps(
        {
            "param_groups": [{"params": [0, 1]}],
            "state": {0: {"step": torch.tensor(2.0)}},
        },
        {"last_epoch": 2, "_step_count": 3},
        cursor,
    )


def test_persisted_steps_reject_empty_or_foreign_adamw_state() -> None:
    from qwen35_baton.checkpoint import _validate_persisted_steps

    cursor = BatonTrainingCursor(
        global_step=2,
        epoch=0,
        consumed_microbatches=2,
        microbatches_per_epoch=4,
        sampler_seed=7,
    )
    scheduler = {"last_epoch": 2, "_step_count": 3}

    with pytest.raises(ValueError, match="empty"):
        _validate_persisted_steps(
            {"param_groups": [{"params": [0, 1]}], "state": {}},
            scheduler,
            cursor,
        )
    with pytest.raises(ValueError, match="outside"):
        _validate_persisted_steps(
            {
                "param_groups": [{"params": [0, 1]}],
                "state": {2: {"step": torch.tensor(2.0)}},
            },
            scheduler,
            cursor,
        )


class _Scaler:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def state_dict(self) -> dict[str, float]:
        return {"scale": self.scale}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.scale = float(state["scale"])


def _race_topology(width: int) -> dict[str, Any]:
    return {
        "format_version": 1,
        "tensors": [
            {"name": "weight", "shape": [width], "dtype": "F32"},
        ],
        "aliases": {},
    }


def _publish_topology_worker(
    barrier: Any,
    destination: str,
    topology: dict[str, Any],
    results: Any,
) -> None:
    try:
        barrier.wait(timeout=10)
        published_hash = publish_trusted_planner_topology(
            destination,
            topology,
        )
        results.put(("ok", published_hash))
    except Exception as error:
        results.put(("error", type(error).__name__, str(error)))


def _run_concurrent_publications(
    destination: Path,
    topologies: tuple[dict[str, Any], dict[str, Any]],
) -> list[tuple[Any, ...]]:
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(len(topologies))
    results = context.Queue()
    processes = [
        context.Process(
            target=_publish_topology_worker,
            args=(barrier, str(destination), topology, results),
        )
        for topology in topologies
    ]
    for process in processes:
        process.start()
    publications = [results.get(timeout=20) for _ in processes]
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    return publications


def load_baton_checkpoint(*args: Any, **kwargs: Any):
    kwargs.setdefault("expected_sampler_seed", 41)
    kwargs.setdefault("expected_microbatches_per_epoch", 11)
    topology = planner_safetensors_topology(Path(args[0]) / "planner.safetensors")
    kwargs.setdefault("expected_planner_topology", topology)
    kwargs["expected_contract"] = replace(
        kwargs["expected_contract"],
        planner_topology_hash=sha256_json(topology),
    )
    return _load_baton_checkpoint(*args, **kwargs)


def _metadata_for_planner(
    planner: nn.Module,
    metadata: BatonCheckpointMetadata | None = None,
) -> BatonCheckpointMetadata:
    return replace(
        BatonCheckpointMetadata.example() if metadata is None else metadata,
        planner_topology_hash=sha256_json(planner_module_topology(planner)),
    )


def _runtime(seed: int = 7):
    torch.manual_seed(seed)
    planner = nn.Sequential(nn.Linear(2, 3), nn.Dropout(0.2), nn.Linear(3, 1))
    optimizer = torch.optim.AdamW(
        [{"name": "planner", "params": list(planner.parameters()), "lr": 5e-5}]
    )
    scheduler = BatonCosineWarmupScheduler(optimizer, warmup_steps=0, max_steps=10)
    value = planner(torch.tensor([[1.0, -2.0]])).square().mean()
    value.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return planner, optimizer, scheduler


def _cursor(*, global_step: int = 1) -> BatonTrainingCursor:
    return BatonTrainingCursor(
        global_step=global_step,
        epoch=3,
        consumed_microbatches=7,
        microbatches_per_epoch=11,
        sampler_seed=41,
    )


def _save_valid_checkpoint(
    checkpoint: Path,
    *,
    ranks: int = 1,
    scaler: _Scaler | None = None,
    metadata: BatonCheckpointMetadata | None = None,
):
    planner, optimizer, scheduler = _runtime()
    rank_states = {
        rank: capture_rank_rng_state(distributed_rank=rank) for rank in range(ranks)
    }
    save_baton_checkpoint(
        checkpoint,
        planner=planner,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        metadata=_metadata_for_planner(planner, metadata),
        cursor=_cursor(),
        rank_rng_state=rank_states,
    )
    return planner, optimizer, scheduler


def _rewrite_checkpoint_metadata_as_v3(
    checkpoint: Path,
) -> dict[str, Any]:
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    metadata["format_version"] = 3
    metadata["future_indices"] = [0, 3, 5, 8]
    del metadata["temporal_policy"]
    (checkpoint / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    manifest["files"]["metadata.json"] = sha256_file(checkpoint / "metadata.json")
    (checkpoint / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _assert_state_equal(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> None:
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def _assert_nested_equal(actual: Any, expected: Any) -> None:
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            _assert_nested_equal(actual_value, expected_value)
    else:
        assert actual == expected


def test_training_cursor_requires_the_next_microbatch_inside_an_epoch() -> None:
    with pytest.raises(ValueError, match="next microbatch"):
        BatonTrainingCursor(
            global_step=1,
            epoch=0,
            consumed_microbatches=9,
            microbatches_per_epoch=9,
            sampler_seed=0,
        )


def test_zero_step_checkpoint_supports_validated_model_only_loading(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    planner = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(
        [{"name": "planner", "params": list(planner.parameters()), "lr": 5e-5}]
    )
    scheduler = BatonCosineWarmupScheduler(optimizer, warmup_steps=0, max_steps=10)
    checkpoint = tmp_path / "step_000000"
    cursor = BatonTrainingCursor(
        global_step=0,
        epoch=0,
        consumed_microbatches=0,
        microbatches_per_epoch=11,
        sampler_seed=41,
    )
    save_baton_checkpoint(
        checkpoint,
        planner=planner,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=_metadata_for_planner(planner),
        cursor=cursor,
        rank_rng_state={0: capture_rank_rng_state(distributed_rank=0)},
    )
    runtime = nn.Linear(2, 1)

    state = load_baton_checkpoint(
        checkpoint,
        planner=runtime,
        optimizer=None,
        scheduler=None,
        expected_contract=BatonCheckpointMetadata.example(),
    )

    assert state.cursor == cursor
    _assert_state_equal(runtime.state_dict(), planner.state_dict())


def test_checkpoint_is_published_as_one_complete_atomic_directory(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "step_000001"

    _save_valid_checkpoint(destination, ranks=2, scaler=_Scaler(1024.0))

    assert {path.name for path in destination.iterdir()} == {
        "planner.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
        "rank_rng.pt",
        "cursor.json",
        "metadata.json",
        "manifest.json",
    }
    assert not list(tmp_path.glob(".step_000001.incomplete-*"))
    manifest = json.loads((destination / "manifest.json").read_text())
    assert set(manifest["files"]) == {
        path.name for path in destination.iterdir() if path.name != "manifest.json"
    }


def test_checkpoint_save_requires_every_contiguous_rank_publication(
    tmp_path: Path,
) -> None:
    planner, optimizer, scheduler = _runtime()
    states = {
        0: capture_rank_rng_state(distributed_rank=0),
        2: capture_rank_rng_state(distributed_rank=2),
    }

    with pytest.raises(ValueError, match="contiguous.*rank"):
        save_baton_checkpoint(
            tmp_path / "step_000001",
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata=_metadata_for_planner(planner),
            cursor=_cursor(),
            rank_rng_state=states,
        )

    assert not (tmp_path / "step_000001").exists()


def test_checkpoint_save_rejects_unnamed_optimizer_lr_groups(
    tmp_path: Path,
) -> None:
    planner = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(planner.parameters(), lr=5e-5)
    scheduler = BatonCosineWarmupScheduler(optimizer, warmup_steps=0, max_steps=10)
    destination = tmp_path / "step_000000"

    with pytest.raises(ValueError, match="LR group names"):
        save_baton_checkpoint(
            destination,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata=_metadata_for_planner(planner),
            cursor=BatonTrainingCursor(
                global_step=0,
                epoch=0,
                consumed_microbatches=0,
                microbatches_per_epoch=11,
                sampler_seed=41,
            ),
            rank_rng_state={0: capture_rank_rng_state(distributed_rank=0)},
        )

    assert not destination.exists()


def test_checkpoint_uses_the_first_registered_name_for_an_aliased_parameter(
    tmp_path: Path,
) -> None:
    class _AliasedPlanner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.lm_head = nn.Linear(2, 1, bias=False)
            self.frozen_base_embedding = self.backbone.lm_head

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.backbone.lm_head(value)

    planner = _AliasedPlanner()
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "planner",
                "params": [planner.backbone.lm_head.weight],
                "lr": 5e-5,
            }
        ]
    )
    scheduler = BatonCosineWarmupScheduler(optimizer, warmup_steps=0, max_steps=10)
    planner(torch.tensor([[1.0, -2.0]])).square().mean().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    destination = tmp_path / "step_000001"

    save_baton_checkpoint(
        destination,
        planner=planner,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=_metadata_for_planner(planner),
        cursor=_cursor(),
        rank_rng_state={0: capture_rank_rng_state(distributed_rank=0)},
    )

    optimizer_state = torch.load(
        destination / "optimizer.pt",
        weights_only=True,
        map_location="cpu",
    )
    assert optimizer_state["param_groups"][0]["parameter_names"] == [
        "backbone.lm_head.weight"
    ]


def test_continuous_loader_rejects_ta_tok_metadata_before_model_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(checkpoint)
    payload = json.loads((checkpoint / "metadata.json").read_text())
    payload["architecture_kind"] = "qwen35_planx_grounded"
    (checkpoint / "metadata.json").write_text(json.dumps(payload))
    planner, optimizer, scheduler = _runtime(seed=99)
    before = _clone_state(planner)

    with pytest.raises(ValueError, match="qwen35_baton_continuous"):
        load_baton_checkpoint(
            checkpoint,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_contract=BatonCheckpointMetadata.example(),
        )

    _assert_state_equal(planner.state_dict(), before)


def test_loader_rejects_file_hash_tampering_before_any_state_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(checkpoint)
    with (checkpoint / "optimizer.pt").open("ab") as stream:
        stream.write(b"tampered")
    planner, optimizer, scheduler = _runtime(seed=99)
    before = _clone_state(planner)
    before_optimizer = optimizer.state_dict()
    before_scheduler = scheduler.state_dict()

    with pytest.raises(ValueError, match="hash mismatch.*optimizer.pt"):
        load_baton_checkpoint(
            checkpoint,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_contract=BatonCheckpointMetadata.example(),
        )

    _assert_state_equal(planner.state_dict(), before)
    assert optimizer.state_dict() == before_optimizer
    assert scheduler.state_dict() == before_scheduler


def test_loader_rejects_model_topology_before_any_state_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(checkpoint)
    planner = nn.Linear(4, 1)
    optimizer = torch.optim.AdamW(planner.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    before = _clone_state(planner)

    with pytest.raises(ValueError, match="planner state topology"):
        load_baton_checkpoint(
            checkpoint,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_contract=BatonCheckpointMetadata.example(),
        )

    _assert_state_equal(planner.state_dict(), before)


def test_checkpoint_round_trip_restores_model_optimizer_scheduler_scaler_and_rank_rng(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step_000001"
    source_scaler = _Scaler(2048.0)
    source, source_optimizer, source_scheduler = _save_valid_checkpoint(
        checkpoint, ranks=2, scaler=source_scaler
    )
    expected_model = _clone_state(source)
    expected_optimizer = source_optimizer.state_dict()
    expected_scheduler = source_scheduler.state_dict()
    planner, optimizer, scheduler = _runtime(seed=99)
    scaler = _Scaler(1.0)

    state = load_baton_checkpoint(
        checkpoint,
        planner=planner,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        expected_contract=BatonCheckpointMetadata.example(),
        distributed_rank=1,
        world_size=2,
    )

    _assert_state_equal(planner.state_dict(), expected_model)
    _assert_nested_equal(optimizer.state_dict(), expected_optimizer)
    _assert_nested_equal(scheduler.state_dict(), expected_scheduler)
    assert scaler.state_dict() == {"scale": 2048.0}
    assert state.cursor == _cursor()
    assert state.rank_rng_state["distributed_rank"] == 1
    assert state.metadata.global_step == 1
    assert dict(state.metadata.distributed_cursor) == {
        "epoch": 3,
        "consumed_microbatches": 7,
        "microbatches_per_epoch": 11,
        "sampler_seed": 41,
        "world_size": 2,
    }


def test_rank_rng_state_restores_python_numpy_cpu_and_all_cuda_streams(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step_000001"
    random.seed(123)
    np.random.seed(124)
    torch.manual_seed(125)
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)
    random.seed(123)
    np.random.seed(124)
    torch.manual_seed(125)
    saved = capture_rank_rng_state(distributed_rank=0)
    planner, optimizer, scheduler = _runtime()
    save_baton_checkpoint(
        checkpoint,
        planner=planner,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=_metadata_for_planner(planner),
        cursor=_cursor(),
        rank_rng_state={0: saved},
    )
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    runtime, runtime_optimizer, runtime_scheduler = _runtime(seed=7)

    state = load_baton_checkpoint(
        checkpoint,
        planner=runtime,
        optimizer=runtime_optimizer,
        scheduler=runtime_scheduler,
        expected_contract=BatonCheckpointMetadata.example(),
    )

    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    torch.testing.assert_close(torch.rand(3), expected_torch, rtol=0, atol=0)
    assert len(state.rank_rng_state["torch_cuda"]) == (
        torch.cuda.device_count() if torch.cuda.is_available() else 0
    )


def test_loader_rejects_runtime_rank_or_world_size_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(checkpoint, ranks=2)
    planner, optimizer, scheduler = _runtime()

    with pytest.raises(ValueError, match="world-size/rank"):
        load_baton_checkpoint(
            checkpoint,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_contract=BatonCheckpointMetadata.example(),
            distributed_rank=0,
            world_size=1,
        )


def test_loader_rejects_runtime_sampler_contract_before_tensor_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(checkpoint)
    planner, optimizer, scheduler = _runtime(seed=99)
    before = _clone_state(planner)
    trusted_topology = planner_module_topology(planner)

    with pytest.raises(ValueError, match="sampler seed"):
        _load_baton_checkpoint(
            checkpoint,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_contract=_metadata_for_planner(planner),
            expected_sampler_seed=99,
            expected_microbatches_per_epoch=11,
            expected_planner_topology=trusted_topology,
        )

    _assert_state_equal(planner.state_dict(), before)


def test_loader_rejects_cuda_rng_runtime_before_model_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "step_000001"
    planner, optimizer, scheduler = _runtime()
    rank_state = capture_rank_rng_state(distributed_rank=0)
    rank_state["torch_cuda"] = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
    save_baton_checkpoint(
        checkpoint,
        planner=planner,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=_metadata_for_planner(planner),
        cursor=_cursor(),
        rank_rng_state={0: rank_state},
    )
    runtime, runtime_optimizer, runtime_scheduler = _runtime(seed=99)
    before = _clone_state(runtime)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA.*unavailable"):
        load_baton_checkpoint(
            checkpoint,
            planner=runtime,
            optimizer=runtime_optimizer,
            scheduler=runtime_scheduler,
            expected_contract=BatonCheckpointMetadata.example(),
        )

    _assert_state_equal(runtime.state_dict(), before)


def test_loader_rejects_scheduler_recipe_mismatch_before_model_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(checkpoint)
    planner, optimizer, _ = _runtime(seed=99)
    scheduler = BatonCosineWarmupScheduler(
        optimizer,
        warmup_steps=0,
        max_steps=10,
        max_consecutive_skipped_updates=7,
    )
    before = _clone_state(planner)

    with pytest.raises(ValueError, match="scheduler contract"):
        load_baton_checkpoint(
            checkpoint,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_contract=BatonCheckpointMetadata.example(),
        )

    _assert_state_equal(planner.state_dict(), before)


def test_loader_rejects_scheduler_current_lr_inconsistent_with_recipe(
    tmp_path: Path,
) -> None:
    from qwen35_baton.checkpoint import _scheduler_topology
    from qwen35_baton.hashing import sha256_file, sha256_json

    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(checkpoint)
    scheduler_state = torch.load(
        checkpoint / "scheduler.pt", weights_only=True, map_location="cpu"
    )
    scheduler_state["_last_lr"][0] *= 0.5
    torch.save(scheduler_state, checkpoint / "scheduler.pt")
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    metadata["scheduler_topology_hash"] = sha256_json(
        _scheduler_topology(scheduler_state)
    )
    (checkpoint / "metadata.json").write_text(json.dumps(metadata))
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    manifest["files"]["scheduler.pt"] = sha256_file(checkpoint / "scheduler.pt")
    manifest["files"]["metadata.json"] = sha256_file(checkpoint / "metadata.json")
    (checkpoint / "manifest.json").write_text(json.dumps(manifest))
    planner, optimizer, scheduler = _runtime(seed=99)
    before = _clone_state(planner)

    with pytest.raises(ValueError, match="scheduler current LR"):
        load_baton_checkpoint(
            checkpoint,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_contract=BatonCheckpointMetadata.example(),
        )

    _assert_state_equal(planner.state_dict(), before)


def test_loader_rejects_optimizer_current_lr_inconsistent_with_scheduler(
    tmp_path: Path,
) -> None:
    from qwen35_baton.hashing import sha256_file

    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(checkpoint)
    optimizer_state = torch.load(
        checkpoint / "optimizer.pt", weights_only=True, map_location="cpu"
    )
    optimizer_state["param_groups"][0]["lr"] = 0.123
    torch.save(optimizer_state, checkpoint / "optimizer.pt")
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    manifest["files"]["optimizer.pt"] = sha256_file(checkpoint / "optimizer.pt")
    (checkpoint / "manifest.json").write_text(json.dumps(manifest))
    planner, optimizer, scheduler = _runtime(seed=99)
    before_planner = _clone_state(planner)
    before_optimizer = optimizer.state_dict()
    before_scheduler = scheduler.state_dict()

    with pytest.raises(ValueError, match="optimizer current LR.*scheduler"):
        load_baton_checkpoint(
            checkpoint,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_contract=BatonCheckpointMetadata.example(),
        )

    _assert_state_equal(planner.state_dict(), before_planner)
    _assert_nested_equal(optimizer.state_dict(), before_optimizer)
    _assert_nested_equal(scheduler.state_dict(), before_scheduler)


def test_loader_rejects_same_shaped_optimizer_parameter_reordering(
    tmp_path: Path,
) -> None:
    class _SameShapePlanner(nn.Module):
        def __init__(self, seed: int) -> None:
            super().__init__()
            generator = torch.Generator().manual_seed(seed)
            self.first = nn.Parameter(torch.randn(2, generator=generator))
            self.second = nn.Parameter(torch.randn(2, generator=generator))

        def forward(self) -> torch.Tensor:
            return (self.first + self.second).square().mean()

    checkpoint = tmp_path / "step_000001"
    source = _SameShapePlanner(1)
    source_optimizer = torch.optim.AdamW(
        [{"name": "planner", "params": [source.first, source.second], "lr": 5e-5}]
    )
    source_scheduler = BatonCosineWarmupScheduler(
        source_optimizer, warmup_steps=0, max_steps=10
    )
    source().backward()
    source_optimizer.step()
    source_scheduler.step()
    source_optimizer.zero_grad(set_to_none=True)
    save_baton_checkpoint(
        checkpoint,
        planner=source,
        optimizer=source_optimizer,
        scheduler=source_scheduler,
        metadata=_metadata_for_planner(source),
        cursor=_cursor(),
        rank_rng_state={0: capture_rank_rng_state(distributed_rank=0)},
    )
    runtime = _SameShapePlanner(2)
    runtime_optimizer = torch.optim.AdamW(
        [{"name": "planner", "params": [runtime.second, runtime.first], "lr": 5e-5}]
    )
    runtime_scheduler = BatonCosineWarmupScheduler(
        runtime_optimizer, warmup_steps=0, max_steps=10
    )
    before = _clone_state(runtime)

    with pytest.raises(ValueError, match="parameter names|parameter order"):
        load_baton_checkpoint(
            checkpoint,
            planner=runtime,
            optimizer=runtime_optimizer,
            scheduler=runtime_scheduler,
            expected_contract=BatonCheckpointMetadata.example(),
        )

    _assert_state_equal(runtime.state_dict(), before)


def test_loader_rejects_cursor_metadata_inconsistency_before_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(checkpoint)
    payload = json.loads((checkpoint / "cursor.json").read_text())
    payload["global_step"] = 2
    (checkpoint / "cursor.json").write_text(json.dumps(payload))
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    from qwen35_baton.hashing import sha256_file

    manifest["files"]["cursor.json"] = sha256_file(checkpoint / "cursor.json")
    (checkpoint / "manifest.json").write_text(json.dumps(manifest))
    planner, optimizer, scheduler = _runtime(seed=99)
    before = _clone_state(planner)

    with pytest.raises(ValueError, match="cursor.*metadata"):
        load_baton_checkpoint(
            checkpoint,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_contract=BatonCheckpointMetadata.example(),
        )

    _assert_state_equal(planner.state_dict(), before)


@pytest.mark.parametrize(
    ("filename", "mutate", "message"),
    (
        (
            "optimizer.pt",
            lambda state: next(iter(state["state"].values()))["step"].fill_(2),
            "optimizer step.*cursor",
        ),
        (
            "scheduler.pt",
            lambda state: state.__setitem__("last_epoch", 2),
            "scheduler step.*cursor",
        ),
    ),
)
def test_loader_rejects_persisted_step_that_differs_from_cursor_before_mutation(
    tmp_path: Path,
    filename: str,
    mutate: Any,
    message: str,
) -> None:
    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(checkpoint)
    state = torch.load(checkpoint / filename, weights_only=True, map_location="cpu")
    mutate(state)
    torch.save(state, checkpoint / filename)
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    from qwen35_baton.hashing import sha256_file

    manifest["files"][filename] = sha256_file(checkpoint / filename)
    (checkpoint / "manifest.json").write_text(json.dumps(manifest))
    planner, optimizer, scheduler = _runtime(seed=99)
    before = _clone_state(planner)

    with pytest.raises(ValueError, match=message):
        load_baton_checkpoint(
            checkpoint,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_contract=BatonCheckpointMetadata.example(),
        )

    _assert_state_equal(planner.state_dict(), before)


def test_checkpoint_save_rejects_untrusted_all_zero_planner_topology(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step_000001"
    planner, optimizer, scheduler = _runtime()

    with pytest.raises(ValueError, match="planner topology hash"):
        save_baton_checkpoint(
            checkpoint,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata=replace(
                BatonCheckpointMetadata.example(),
                planner_topology_hash="0" * 64,
            ),
            cursor=_cursor(),
            rank_rng_state={0: capture_rank_rng_state(distributed_rank=0)},
        )

    assert not checkpoint.exists()


def test_distinct_concurrent_topology_publishers_create_exactly_one_anchor(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "planner_topology.json"
    first = _race_topology(1)
    second = _race_topology(2)

    results = _run_concurrent_publications(destination, (first, second))

    successes = [result for result in results if result[0] == "ok"]
    failures = [result for result in results if result[0] == "error"]
    assert len(successes) == 1
    assert len(failures) == 1
    anchored, anchored_hash = load_trusted_planner_topology(destination)
    assert anchored in (first, second)
    assert anchored_hash == successes[0][1]
    assert not list(tmp_path.glob(".planner_topology.json.incomplete-*"))


def test_identical_concurrent_topology_publishers_are_idempotent(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "planner_topology.json"
    topology = _race_topology(3)

    results = _run_concurrent_publications(
        destination,
        (topology, topology),
    )

    assert [result[0] for result in results] == ["ok", "ok"]
    anchored, anchored_hash = load_trusted_planner_topology(destination)
    assert anchored == topology
    assert {result[1] for result in results} == {anchored_hash}
    assert not list(tmp_path.glob(".planner_topology.json.incomplete-*"))


def test_corrupt_preexisting_topology_anchor_is_never_overwritten(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "planner_topology.json"
    corrupt = b'{"format_version": 1, "topology": '
    destination.write_bytes(corrupt)

    with pytest.raises(ValueError, match="trusted planner topology JSON"):
        publish_trusted_planner_topology(destination, _race_topology(4))

    assert destination.read_bytes() == corrupt
    assert not list(tmp_path.glob(".planner_topology.json.incomplete-*"))


@pytest.mark.skipif(os.name != "posix", reason="hard-link publication is POSIX-only")
def test_published_topology_anchor_has_canonical_shared_read_mode(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "planner_topology.json"

    publish_trusted_planner_topology(destination, _race_topology(5))

    assert stat.S_IMODE(destination.stat().st_mode) == 0o644


def test_topology_chmod_failure_leaves_no_target_or_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "planner_topology.json"

    def fail_chmod(path: Path, mode: int) -> None:
        del path, mode
        raise PermissionError("chmod denied")

    monkeypatch.setattr(checkpoint_module.os, "chmod", fail_chmod)

    with pytest.raises(PermissionError, match="chmod denied"):
        publish_trusted_planner_topology(destination, _race_topology(6))

    assert not destination.exists()
    assert not list(tmp_path.glob(".planner_topology.json.incomplete-*"))


@pytest.mark.skipif(os.name != "posix", reason="hard-link publication is POSIX-only")
def test_identical_preexisting_anchor_with_noncanonical_mode_fails_closed(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "planner_topology.json"
    topology = _race_topology(9)
    destination.write_text(json.dumps(trusted_planner_topology_payload(topology)))
    destination.chmod(0o600)

    with pytest.raises(PermissionError, match="mode.*0644"):
        publish_trusted_planner_topology(destination, topology)

    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".planner_topology.json.incomplete-*"))


def test_topology_link_failure_leaves_no_target_or_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "planner_topology.json"

    def fail_link(source: Path, target: Path) -> None:
        del source, target
        raise PermissionError("hard link denied")

    monkeypatch.setattr(checkpoint_module.os, "link", fail_link)

    with pytest.raises(PermissionError, match="hard link denied"):
        publish_trusted_planner_topology(destination, _race_topology(7))

    assert not destination.exists()
    assert not list(tmp_path.glob(".planner_topology.json.incomplete-*"))


def test_topology_directory_fsync_failure_leaves_only_complete_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "planner_topology.json"

    def fail_directory_fsync(path: Path) -> None:
        del path
        raise OSError("directory fsync denied")

    monkeypatch.setattr(
        checkpoint_module,
        "_fsync_directory",
        fail_directory_fsync,
    )

    with pytest.raises(OSError, match="directory fsync denied"):
        publish_trusted_planner_topology(destination, _race_topology(8))

    anchored, _ = load_trusted_planner_topology(destination)
    assert anchored == _race_topology(8)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o644
    assert not list(tmp_path.glob(".planner_topology.json.incomplete-*"))


def test_loader_rejects_refreshed_all_zero_metadata_topology_before_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(checkpoint)
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    metadata["planner_topology_hash"] = "0" * 64
    (checkpoint / "metadata.json").write_text(json.dumps(metadata))
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    from qwen35_baton.hashing import sha256_file

    manifest["files"]["metadata.json"] = sha256_file(checkpoint / "metadata.json")
    (checkpoint / "manifest.json").write_text(json.dumps(manifest))
    planner, optimizer, scheduler = _runtime(seed=99)
    before = _clone_state(planner)

    with pytest.raises(ValueError, match="planner_topology_hash"):
        load_baton_checkpoint(
            checkpoint,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_contract=BatonCheckpointMetadata.example(),
        )

    _assert_state_equal(planner.state_dict(), before)


def test_loader_rejects_refreshed_wrong_planner_dtype_before_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step_000001"
    source, _, _ = _save_valid_checkpoint(checkpoint)
    trusted_topology = planner_safetensors_topology(checkpoint / "planner.safetensors")
    expected_contract = replace(
        BatonCheckpointMetadata.example(),
        planner_topology_hash=sha256_json(trusted_topology),
    )
    source.double()
    save_model(source, checkpoint / "planner.safetensors")
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    from qwen35_baton.hashing import sha256_file

    manifest["files"]["planner.safetensors"] = sha256_file(
        checkpoint / "planner.safetensors"
    )
    (checkpoint / "manifest.json").write_text(json.dumps(manifest))
    planner, optimizer, scheduler = _runtime(seed=99)
    before = _clone_state(planner)

    with pytest.raises(ValueError, match="planner.*topology"):
        _load_baton_checkpoint(
            checkpoint,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_contract=expected_contract,
            expected_sampler_seed=41,
            expected_microbatches_per_epoch=11,
            expected_planner_topology=trusted_topology,
        )

    _assert_state_equal(planner.state_dict(), before)


def test_loader_rejects_wrong_external_topology_before_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(checkpoint)
    trusted_topology = planner_safetensors_topology(checkpoint / "planner.safetensors")
    expected_contract = replace(
        BatonCheckpointMetadata.example(),
        planner_topology_hash=sha256_json(trusted_topology),
    )
    trusted_topology["tensors"][0]["dtype"] = "F64"
    planner, optimizer, scheduler = _runtime(seed=99)
    before = _clone_state(planner)

    with pytest.raises(ValueError, match="trusted planner topology"):
        _load_baton_checkpoint(
            checkpoint,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_contract=expected_contract,
            expected_sampler_seed=41,
            expected_microbatches_per_epoch=11,
            expected_planner_topology=trusted_topology,
        )

    _assert_state_equal(planner.state_dict(), before)


def test_legacy_libero_v3_checkpoint_resumes_against_v4_contract(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(checkpoint)
    _rewrite_checkpoint_metadata_as_v3(checkpoint)
    planner, optimizer, scheduler = _runtime(seed=99)

    resumed = load_baton_checkpoint(
        checkpoint,
        planner=planner,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_contract=BatonCheckpointMetadata.example(),
    )

    assert resumed.metadata.format_version == 4
    assert resumed.metadata.camera_names == ("main", "wrist")


def test_legacy_head_v3_checkpoint_fails_before_runtime_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(
        checkpoint,
        metadata=BatonCheckpointMetadata.example(camera_names=("head",)),
    )
    _rewrite_checkpoint_metadata_as_v3(checkpoint)
    planner, optimizer, scheduler = _runtime(seed=99)
    before = _clone_state(planner)

    with pytest.raises(ValueError, match="legacy head.*migration required"):
        load_baton_checkpoint(
            checkpoint,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_contract=BatonCheckpointMetadata.example(camera_names=("head",)),
        )

    _assert_state_equal(planner.state_dict(), before)


def test_head_v3_migration_is_atomic_idempotent_and_preserves_all_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(
        checkpoint,
        metadata=BatonCheckpointMetadata.example(camera_names=("head",)),
    )
    _rewrite_checkpoint_metadata_as_v3(checkpoint)
    state_files = (
        "planner.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
        "rank_rng.pt",
        "cursor.json",
    )
    state_bytes = {name: (checkpoint / name).read_bytes() for name in state_files}
    old_manifest = json.loads((checkpoint / "manifest.json").read_text())

    result = migrate_legacy_head_checkpoint_v4(checkpoint)

    assert result.migrated is True
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert metadata["format_version"] == 4
    assert metadata["camera_names"] == ["head"]
    assert "future_indices" not in metadata
    assert metadata["temporal_policy"]["kind"] == ("normalized_remaining_horizon")
    assert manifest["files"]["metadata.json"] == sha256_file(
        checkpoint / "metadata.json"
    )
    for name in state_files:
        assert (checkpoint / name).read_bytes() == state_bytes[name]
        assert manifest["files"][name] == old_manifest["files"][name]
    assert not tuple(checkpoint.parent.glob(f".{checkpoint.name}.metadata-v4-*"))

    metadata_bytes = (checkpoint / "metadata.json").read_bytes()
    manifest_bytes = (checkpoint / "manifest.json").read_bytes()
    second = migrate_legacy_head_checkpoint_v4(checkpoint)
    assert second.migrated is False
    assert (checkpoint / "metadata.json").read_bytes() == metadata_bytes
    assert (checkpoint / "manifest.json").read_bytes() == manifest_bytes

    planner, optimizer, scheduler = _runtime(seed=99)
    resumed = load_baton_checkpoint(
        checkpoint,
        planner=planner,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_contract=BatonCheckpointMetadata.example(camera_names=("head",)),
    )
    assert resumed.metadata.camera_names == ("head",)


def test_head_v3_migration_failure_leaves_original_envelope_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(
        checkpoint,
        metadata=BatonCheckpointMetadata.example(camera_names=("head",)),
    )
    _rewrite_checkpoint_metadata_as_v3(checkpoint)
    before = {
        name: (checkpoint / name).read_bytes()
        for name in ("metadata.json", "manifest.json")
    }

    original_replace = checkpoint_module._atomic_replace_from
    calls = 0

    def fail_second_replace(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected atomic replace failure")
        original_replace(source, destination)

    monkeypatch.setattr(
        checkpoint_module,
        "_atomic_replace_from",
        fail_second_replace,
    )
    with pytest.raises(OSError, match="replace failure"):
        migrate_legacy_head_checkpoint_v4(checkpoint)

    for name, expected in before.items():
        assert (checkpoint / name).read_bytes() == expected
    assert not tuple(checkpoint.parent.glob(f".{checkpoint.name}.metadata-v4-*"))


@pytest.mark.parametrize("completed_replaces", (0, 1, 2))
def test_head_v3_migration_recovers_every_published_transaction_boundary(
    tmp_path: Path,
    completed_replaces: int,
) -> None:
    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(
        checkpoint,
        metadata=BatonCheckpointMetadata.example(camera_names=("head",)),
    )
    raw = _rewrite_checkpoint_metadata_as_v3(checkpoint)
    old_metadata_hash = sha256_file(checkpoint / "metadata.json")
    old_manifest_hash = sha256_file(checkpoint / "manifest.json")
    migrated = BatonCheckpointMetadata._from_legacy_v3(raw, allow_head=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{checkpoint.name}.metadata-v4-prepared-",
            dir=checkpoint.parent,
        )
    )
    shutil.copy2(checkpoint / "metadata.json", staging / "old-metadata.json")
    shutil.copy2(checkpoint / "manifest.json", staging / "old-manifest.json")
    checkpoint_module._json_write(staging / "metadata.json", migrated.to_dict())
    new_metadata_hash = sha256_file(staging / "metadata.json")
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    manifest["files"]["metadata.json"] = new_metadata_hash
    checkpoint_module._json_write(staging / "manifest.json", manifest)
    new_manifest_hash = sha256_file(staging / "manifest.json")
    checkpoint_module._json_write(
        staging / "transaction.json",
        {
            "format_version": 1,
            "checkpoint": checkpoint.name,
            "old_metadata_sha256": old_metadata_hash,
            "old_manifest_sha256": old_manifest_hash,
            "new_metadata_sha256": new_metadata_hash,
            "new_manifest_sha256": new_manifest_hash,
        },
    )
    (staging / ".replace-metadata.json-crashed").write_text(
        "incomplete disposable copy",
        encoding="utf-8",
    )
    replacements = (
        (staging / "metadata.json", checkpoint / "metadata.json"),
        (staging / "manifest.json", checkpoint / "manifest.json"),
    )
    for source, destination in replacements[:completed_replaces]:
        checkpoint_module._atomic_replace_from(source, destination)

    result = migrate_legacy_head_checkpoint_v4(checkpoint)

    assert result.migrated is (completed_replaces < 2)
    assert BatonCheckpointMetadata.from_dict(
        json.loads((checkpoint / "metadata.json").read_text())
    ).camera_names == ("head",)
    assert not tuple(checkpoint.parent.glob(f".{checkpoint.name}.metadata-v4-*"))


@pytest.mark.parametrize("prepared_file_count", range(6))
def test_head_v3_migration_discards_incomplete_unpublished_building_directory(
    tmp_path: Path,
    prepared_file_count: int,
) -> None:
    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(
        checkpoint,
        metadata=BatonCheckpointMetadata.example(camera_names=("head",)),
    )
    _rewrite_checkpoint_metadata_as_v3(checkpoint)
    building = tmp_path / f".{checkpoint.name}.metadata-v4-building-crashed"
    building.mkdir()
    staged_names = (
        "old-metadata.json",
        "old-manifest.json",
        "metadata.json",
        "manifest.json",
        "transaction.json",
    )
    for name in staged_names[:prepared_file_count]:
        if name == "transaction.json":
            (building / name).write_text("{}\n", encoding="utf-8")
        else:
            source_name = name.removeprefix("old-")
            shutil.copy2(checkpoint / source_name, building / name)

    result = migrate_legacy_head_checkpoint_v4(checkpoint)

    assert result.migrated is True
    assert not building.exists()
    assert not tuple(checkpoint.parent.glob(f".{checkpoint.name}.metadata-v4-*"))


def test_head_v4_migration_discards_partial_post_commit_cleanup_residue(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(
        checkpoint,
        metadata=BatonCheckpointMetadata.example(camera_names=("head",)),
    )
    _rewrite_checkpoint_metadata_as_v3(checkpoint)
    migrate_legacy_head_checkpoint_v4(checkpoint)
    metadata_bytes = (checkpoint / "metadata.json").read_bytes()
    manifest_bytes = (checkpoint / "manifest.json").read_bytes()
    cleanup = tmp_path / f".{checkpoint.name}.metadata-v4-cleanup-crashed"
    cleanup.mkdir()
    (cleanup / "partial").write_text("crash residue", encoding="utf-8")

    result = migrate_legacy_head_checkpoint_v4(checkpoint)

    assert result.migrated is False
    assert (checkpoint / "metadata.json").read_bytes() == metadata_bytes
    assert (checkpoint / "manifest.json").read_bytes() == manifest_bytes
    assert not cleanup.exists()
    assert not tuple(checkpoint.parent.glob(f".{checkpoint.name}.metadata-v4-*"))


def test_concurrent_head_v3_migrations_are_serialized_without_rollback_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(
        checkpoint,
        metadata=BatonCheckpointMetadata.example(camera_names=("head",)),
    )
    _rewrite_checkpoint_metadata_as_v3(checkpoint)
    first_replacing = threading.Event()
    release_first = threading.Event()
    original_replace = checkpoint_module._atomic_replace_from

    def pause_first_migrator(source: Path, destination: Path) -> None:
        if (
            threading.current_thread().name == "first-migrator"
            and destination.name == "metadata.json"
            and not first_replacing.is_set()
        ):
            first_replacing.set()
            assert release_first.wait(timeout=5)
        original_replace(source, destination)

    monkeypatch.setattr(
        checkpoint_module,
        "_atomic_replace_from",
        pause_first_migrator,
    )
    results: list[Any] = []
    failures: list[BaseException] = []

    def migrate() -> None:
        try:
            results.append(migrate_legacy_head_checkpoint_v4(checkpoint))
        except BaseException as error:
            failures.append(error)

    first = threading.Thread(target=migrate, name="first-migrator")
    second = threading.Thread(target=migrate, name="second-migrator")
    first.start()
    assert first_replacing.wait(timeout=5)
    second.start()
    second.join(timeout=0.2)
    assert second.is_alive()
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert failures == []
    assert sorted(result.migrated for result in results) == [False, True]
    metadata = BatonCheckpointMetadata.from_dict(
        json.loads((checkpoint / "metadata.json").read_text())
    )
    assert metadata.camera_names == ("head",)
    assert not tuple(checkpoint.parent.glob(f".{checkpoint.name}.metadata-v4-*"))


def test_head_v3_migration_cli_reports_changed_then_idempotent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from qwen35_baton.cli.migrate_checkpoint_v4 import main

    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(
        checkpoint,
        metadata=BatonCheckpointMetadata.example(camera_names=("head",)),
    )
    _rewrite_checkpoint_metadata_as_v3(checkpoint)

    assert main([str(checkpoint)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first[0]["migrated"] is True
    assert main([str(checkpoint)]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second[0]["migrated"] is False


def test_loader_rejects_format_v1_checkpoint_before_mutation(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "step_000001"
    _save_valid_checkpoint(checkpoint)
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    metadata["format_version"] = 1
    (checkpoint / "metadata.json").write_text(json.dumps(metadata))
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    from qwen35_baton.hashing import sha256_file

    manifest["files"]["metadata.json"] = sha256_file(checkpoint / "metadata.json")
    (checkpoint / "manifest.json").write_text(json.dumps(manifest))
    planner, optimizer, scheduler = _runtime(seed=99)
    before = _clone_state(planner)

    with pytest.raises(ValueError, match="versions 1 and 2.*version 4"):
        load_baton_checkpoint(
            checkpoint,
            planner=planner,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_contract=BatonCheckpointMetadata.example(),
        )

    _assert_state_equal(planner.state_dict(), before)
