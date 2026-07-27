from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from qwen35_baton.checkpoint import (
    BatonTrainingCursor,
    capture_rank_rng_state,
    load_baton_checkpoint,
    save_baton_checkpoint,
)
from qwen35_baton.config import BatonCheckpointMetadata


class _Scaler:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def state_dict(self) -> dict[str, float]:
        return {"scale": self.scale}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.scale = float(state["scale"])


def _runtime(seed: int = 7):
    torch.manual_seed(seed)
    planner = nn.Sequential(nn.Linear(2, 3), nn.Dropout(0.2), nn.Linear(3, 1))
    optimizer = torch.optim.AdamW(
        [{"name": "planner", "params": list(planner.parameters()), "lr": 5e-5}]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1.0 - min(step, 10) / 10
    )
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
):
    planner, optimizer, scheduler = _runtime()
    rank_states = {
        rank: capture_rank_rng_state(distributed_rank=rank)
        for rank in range(ranks)
    }
    save_baton_checkpoint(
        checkpoint,
        planner=planner,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        metadata=BatonCheckpointMetadata.example(),
        cursor=_cursor(),
        rank_rng_state=rank_states,
    )
    return planner, optimizer, scheduler


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
    optimizer = torch.optim.AdamW(planner.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
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
        metadata=BatonCheckpointMetadata.example(),
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
            metadata=BatonCheckpointMetadata.example(),
            cursor=_cursor(),
            rank_rng_state=states,
        )

    assert not (tmp_path / "step_000001").exists()


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
        metadata=BatonCheckpointMetadata.example(),
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
