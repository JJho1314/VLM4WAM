from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from qwen35_baton.cli.preflight import preflight_stage1
from qwen35_baton.cli.train_semantic_planner import (
    Stage1TrainingArtifacts,
    Stage1TrainingConfig,
    build_cosine_warmup_scheduler,
    build_stage1_optimizer_groups,
    checkpoint_steps,
    load_local_artifacts,
    run_training,
    validate_global_batch,
)
from qwen35_baton.config import BatonCheckpointMetadata
from qwen35_baton.ownership import configure_stage1_trainable_modules


REPO_ROOT = Path(__file__).resolve().parents[1]


class _TinyLanguage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(1, 1) for _ in range(12))
        self.norm = nn.LayerNorm(1)


class _TinyBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _TinyLanguage()
        self.visual = nn.Linear(1, 1)


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyBase()
        self.lm_head = nn.Linear(1, 1)


class _TinyPlanner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _TinyBackbone()
        self.query_tower = nn.Linear(1, 1)
        self.sem_mlp = nn.Linear(1, 1)
        self.plan_token_adapter = nn.Linear(1, 1)

    def forward(self, batch: "_TinyBatch") -> Any:
        value = batch.x
        value = self.backbone.model.visual(value)
        for layer in self.backbone.model.language_model.layers[-8:]:
            value = torch.tanh(layer(value))
        value = self.query_tower(value)
        value = self.sem_mlp(value)
        value = self.plan_token_adapter(value)
        prediction = value.reshape(-1, 1, 1, 1, 1).expand(-1, 2, 4, 1, 1)
        return SimpleNamespace(positive=prediction, negative=-prediction)


class _TinyTeacher:
    def __init__(
        self,
        *,
        nonfinite: bool = False,
        output_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = nn.Linear(1, 1)
        with torch.no_grad():
            self.model.weight.fill_(1)
            self.model.bias.zero_()
        self.model.requires_grad_(False)
        self.nonfinite = nonfinite
        self.output_dtype = output_dtype

    def encode_future(self, images: torch.Tensor) -> torch.Tensor:
        result = self.model(images)
        if self.nonfinite:
            result = result.clone()
            result.flatten()[0] = float("inf")
        return result.to(self.output_dtype)

    def encode_current(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images).to(self.output_dtype)


@dataclass(frozen=True)
class _TinyBatch:
    x: torch.Tensor
    current_images: torch.Tensor
    future_images: torch.Tensor


def _tiny_batches(count: int = 25) -> tuple[_TinyBatch, ...]:
    return tuple(
        _TinyBatch(
            x=torch.tensor([[0.1 + index / 100]], dtype=torch.float32),
            current_images=torch.zeros(1, 2, 1, 1),
            future_images=torch.full(
                (1, 2, 4, 1, 1), 0.2 + index / 100, dtype=torch.float32
            ),
        )
        for index in range(count)
    )


def _config(
    output_dir: Path,
    *,
    max_steps: int = 1,
    save_every: int = 1,
    resume_from: Path | None = None,
) -> Stage1TrainingConfig:
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
        per_device_batch=32,
        gradient_accumulation_steps=4,
        max_steps=max_steps,
        warmup_steps=0 if max_steps == 1 else 1,
        save_every=save_every,
        log_every=1,
        seed=17,
        mixed_precision="no",
        tiny_test=True,
        resume_from=None if resume_from is None else str(resume_from),
    )


def _artifacts(
    config: Stage1TrainingConfig,
    *,
    nonfinite: bool = False,
    teacher_dtype: torch.dtype = torch.float32,
):
    torch.manual_seed(2026)
    planner = _TinyPlanner()
    ownership = configure_stage1_trainable_modules(planner)
    groups = build_stage1_optimizer_groups(planner, ownership, config)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    scheduler = build_cosine_warmup_scheduler(optimizer, config)
    return Stage1TrainingArtifacts(
        planner=planner,
        teacher=_TinyTeacher(nonfinite=nonfinite, output_dtype=teacher_dtype),
        train_batches=_tiny_batches(),
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=BatonCheckpointMetadata.example(),
        ownership=ownership,
    )


def _skipping_accelerator(skip_updates: list[bool]) -> type:
    class _PreparedOptimizer:
        def __init__(self, optimizer: torch.optim.Optimizer, accelerator: Any) -> None:
            self.optimizer = optimizer
            self.accelerator = accelerator
            self.step_was_skipped = False

        def step(self) -> None:
            if self.accelerator.sync_gradients:
                if not self.accelerator.skip_updates:
                    raise AssertionError(
                        "training exceeded the configured synchronized update pattern"
                    )
                self.step_was_skipped = self.accelerator.skip_updates.pop(0)
                if not self.step_was_skipped:
                    self.optimizer.step()

        def zero_grad(self, *, set_to_none: bool) -> None:
            if self.accelerator.sync_gradients:
                self.optimizer.zero_grad(set_to_none=set_to_none)

    class _FakeAccelerator:
        def __init__(self, **_: Any) -> None:
            self.device = torch.device("cpu")
            self.num_processes = 1
            self.process_index = 0
            self.is_main_process = True
            self.scaler = None
            self.sync_gradients = False
            self.microbatch = 0
            self.skip_updates = list(skip_updates)
            self.prepared_optimizer: _PreparedOptimizer | None = None

        def prepare(self, planner: nn.Module, optimizer: torch.optim.Optimizer):
            self.prepared_optimizer = _PreparedOptimizer(optimizer, self)
            return planner, self.prepared_optimizer

        def accumulate(self, _planner: nn.Module):
            self.microbatch += 1
            self.sync_gradients = self.microbatch % 4 == 0
            return nullcontext()

        def backward(self, loss: torch.Tensor) -> None:
            loss.backward()

        def clip_grad_norm_(self, parameters: Any, max_norm: float) -> None:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm)

        @property
        def optimizer_step_was_skipped(self) -> bool:
            assert self.prepared_optimizer is not None
            return self.prepared_optimizer.step_was_skipped

        def reduce(self, tensor: torch.Tensor, reduction: str) -> torch.Tensor:
            assert reduction == "mean"
            return tensor

        def gather(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor

        def unwrap_model(self, planner: nn.Module) -> nn.Module:
            return planner

        def wait_for_everyone(self) -> None:
            pass

    return _FakeAccelerator


def _clone_parameters(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right):
            _assert_nested_equal(left_value, right_value)
    else:
        assert left == right


def _durable_metrics_record(
    *, step: int, metrics: dict[str, float]
) -> dict[str, Any]:
    unsigned = {
        "schema_version": 1,
        "step": step,
        "metrics": {name: float(value) for name, value in metrics.items()},
    }
    canonical = json.dumps(
        unsigned,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **unsigned,
        "checksum": hashlib.sha256(canonical).hexdigest(),
    }


def test_stage1_optimizer_groups_are_exact_and_exhaustive(tmp_path: Path) -> None:
    config = _config(tmp_path)
    planner = _TinyPlanner()
    ownership = configure_stage1_trainable_modules(planner)

    groups = build_stage1_optimizer_groups(planner, ownership, config)

    assert {group["name"]: group["lr"] for group in groups} == {
        "planner": 5e-5,
        "qwen_top8": 1e-6,
        "qwen_vision": 5e-7,
    }
    grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
    trainable_ids = [
        id(parameter) for parameter in planner.parameters() if parameter.requires_grad
    ]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == set(trainable_ids)
    canonical_names = {
        id(parameter): name for name, parameter in planner.named_parameters()
    }
    for group in groups:
        assert group["parameter_names"] == [
            canonical_names[id(parameter)] for parameter in group["params"]
        ]
        assert group["parameter_shapes"] == [
            list(parameter.shape) for parameter in group["params"]
        ]
        assert group["parameter_dtypes"] == [
            str(parameter.dtype) for parameter in group["params"]
        ]


def test_one_tiny_stage1_step_updates_only_owned_parameters(tmp_path: Path) -> None:
    config = _config(tmp_path)
    artifacts = _artifacts(config)
    before = _clone_parameters(artifacts.planner)
    owned_ids = {
        id(parameter)
        for modules in (
            artifacts.ownership.planner_modules,
            artifacts.ownership.qwen_top_layers,
            artifacts.ownership.qwen_vision_modules,
        )
        for module in modules
        for parameter in module.parameters()
    }

    result = run_training(config, artifacts=artifacts)

    assert result.global_step == 1
    for name, parameter in artifacts.planner.named_parameters():
        if id(parameter) in owned_ids:
            assert not torch.equal(parameter.detach(), before[name]), name
        else:
            torch.testing.assert_close(parameter.detach(), before[name], rtol=0, atol=0)
    assert all(
        parameter.grad is None for parameter in artifacts.teacher.model.parameters()
    )
    assert {
        "loss/total",
        "loss/mse",
        "loss/cosine",
        "loss/delta",
        "loss/instruction_counterfactual",
        "counterfactual_ranking_accuracy",
        "throughput",
        "data_time",
        "qwen_time",
        "teacher_time",
        "query_tower_time",
        "backward_time",
        "mse/main/frame_0",
        "mse/wrist/frame_0",
        "cosine/main/frame_0",
        "cosine/wrist/frame_0",
    }.issubset(result.last_metrics)
    assert result.last_metrics["qwen_time"] > 0
    assert result.last_metrics["query_tower_time"] > 0


@pytest.mark.parametrize(
    ("per_device", "world_size", "accumulation"),
    ((1, 8, 16), (2, 8, 8), (4, 8, 4)),
)
def test_global_batch_validation_accepts_exactly_128(
    per_device: int, world_size: int, accumulation: int
) -> None:
    assert (
        validate_global_batch(
            per_device_batch=per_device,
            world_size=world_size,
            gradient_accumulation_steps=accumulation,
        )
        == 128
    )


def test_global_batch_validation_rejects_any_other_effective_batch() -> None:
    with pytest.raises(ValueError, match="exactly 128"):
        validate_global_batch(
            per_device_batch=1,
            world_size=8,
            gradient_accumulation_steps=15,
        )


def test_checkpoint_cadence_is_every_5000_through_30000() -> None:
    assert checkpoint_steps(max_steps=30_000, save_every=5_000) == (
        5_000,
        10_000,
        15_000,
        20_000,
        25_000,
        30_000,
    )


def test_cosine_scheduler_uses_linear_warmup_then_reaches_zero(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, max_steps=10)
    config = Stage1TrainingConfig(**{**config.to_dict(), "warmup_steps": 2})
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([{"params": [parameter], "lr": 5e-5}])
    scheduler = build_cosine_warmup_scheduler(optimizer, config)

    multipliers = []
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        multipliers.append(scheduler.get_last_lr()[0] / 5e-5)

    assert multipliers[0] == pytest.approx(0.5)
    assert multipliers[1] == pytest.approx(1.0)
    assert multipliers[5] == pytest.approx(0.5)
    assert multipliers[-1] == pytest.approx(0.0)


def test_scheduler_state_carries_the_exact_schedule_contract(tmp_path: Path) -> None:
    config = _config(tmp_path, max_steps=10)
    config = Stage1TrainingConfig(**{**config.to_dict(), "warmup_steps": 2})
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([{"params": [parameter], "lr": 5e-5}])

    scheduler = build_cosine_warmup_scheduler(optimizer, config)

    assert scheduler.state_dict()["baton_contract"] == {
        "schedule_type": "linear_warmup_cosine_v1",
        "warmup_steps": 2,
        "max_steps": 10,
        "max_consecutive_skipped_updates": 8,
        "base_lrs": [5e-5],
    }


def test_nonfinite_loss_fails_before_optimizer_step(tmp_path: Path) -> None:
    config = _config(tmp_path)
    artifacts = _artifacts(config, nonfinite=True)
    before = _clone_parameters(artifacts.planner)

    with pytest.raises(FloatingPointError, match="nonfinite"):
        run_training(config, artifacts=artifacts)

    for name, parameter in artifacts.planner.named_parameters():
        torch.testing.assert_close(parameter.detach(), before[name], rtol=0, atol=0)


def test_training_promotes_bf16_teacher_targets_for_fp32_planner_loss(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    artifacts = _artifacts(config, teacher_dtype=torch.bfloat16)

    result = run_training(config, artifacts=artifacts)

    assert result.global_step == 1
    assert torch.isfinite(torch.tensor(result.last_metrics["loss/total"]))


def test_training_prepares_the_dataloader_for_distributed_rank_sharding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from accelerate import Accelerator

    config = _config(tmp_path)
    artifacts = _artifacts(config)
    artifacts.train_batches = torch.utils.data.DataLoader(
        _tiny_batches(),
        batch_size=None,
        collate_fn=lambda sample: sample,
    )
    original_prepare_data_loader = Accelerator.prepare_data_loader
    prepared_dataloader = []

    def recording_prepare_data_loader(
        self: Accelerator,
        data_loader: torch.utils.data.DataLoader,
        *args: Any,
        **kwargs: Any,
    ):
        prepared_dataloader.append(data_loader)
        return original_prepare_data_loader(self, data_loader, *args, **kwargs)

    monkeypatch.setattr(
        Accelerator, "prepare_data_loader", recording_prepare_data_loader
    )

    run_training(config, artifacts=artifacts)

    assert prepared_dataloader == [artifacts.train_batches]


def test_training_keeps_scheduler_outside_accelerate_prepare(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from accelerate import Accelerator

    config = _config(tmp_path)
    artifacts = _artifacts(config)
    original_prepare = Accelerator.prepare
    prepared_schedulers = []

    def recording_prepare(self: Accelerator, *args: Any, **kwargs: Any):
        prepared_schedulers.extend(
            value
            for value in args
            if isinstance(value, torch.optim.lr_scheduler.LRScheduler)
        )
        return original_prepare(self, *args, **kwargs)

    monkeypatch.setattr(Accelerator, "prepare", recording_prepare)

    run_training(config, artifacts=artifacts)

    assert prepared_schedulers == []
    assert artifacts.scheduler.last_epoch == 1


def test_epoch_tail_does_not_force_a_partial_accumulation_update(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, max_steps=2, save_every=2)
    artifacts = _artifacts(config)
    artifacts.train_batches = torch.utils.data.DataLoader(
        _tiny_batches(5),
        batch_size=None,
        collate_fn=lambda sample: sample,
    )

    result = run_training(config, artifacts=artifacts)

    assert result.cursor == __import__(
        "qwen35_baton.checkpoint", fromlist=["BatonTrainingCursor"]
    ).BatonTrainingCursor(
        global_step=2,
        epoch=1,
        consumed_microbatches=3,
        microbatches_per_epoch=5,
        sampler_seed=config.seed,
    )
    assert artifacts.scheduler.last_epoch == 2


def test_skipped_optimizer_update_does_not_advance_step_scheduler_or_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import accelerate

    config = _config(tmp_path)
    artifacts = _artifacts(config)
    monkeypatch.setattr(
        accelerate,
        "Accelerator",
        _skipping_accelerator([True, False]),
    )

    result = run_training(config, artifacts=artifacts)

    assert result.global_step == 1
    assert result.cursor.consumed_microbatches == 8
    assert artifacts.scheduler.last_epoch == 1
    records = [
        json.loads(line)
        for line in (tmp_path / "training_metrics.jsonl").read_text().splitlines()
    ]
    assert [record["step"] for record in records] == [1]
    assert records[0]["schema_version"] == 1
    assert set(records[0]) == {
        "schema_version",
        "step",
        "metrics",
        "checksum",
    }
    assert result.checkpoint == tmp_path / "step_000001"


@pytest.mark.parametrize("invalid_limit", (True, 0, -1, 1.5))
def test_stage1_config_rejects_invalid_consecutive_skip_limit(
    tmp_path: Path, invalid_limit: Any
) -> None:
    baseline = _config(tmp_path)

    with pytest.raises(ValueError, match="max_consecutive_skipped_updates"):
        Stage1TrainingConfig(
            **{
                **baseline.to_dict(),
                "max_consecutive_skipped_updates": invalid_limit,
            }
        )


def test_always_skipped_optimizer_updates_fail_at_the_bounded_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import accelerate

    config = _config(tmp_path)
    artifacts = _artifacts(config)
    monkeypatch.setattr(
        accelerate,
        "Accelerator",
        _skipping_accelerator([True] * 8),
    )

    with pytest.raises(
        FloatingPointError,
        match=(
            "8 consecutive synchronized optimizer updates were skipped"
            ".*global_step=0.*epoch=1.*consumed_microbatches=7"
        ),
    ):
        run_training(config, artifacts=artifacts)

    assert artifacts.scheduler.last_epoch == 0
    assert not (tmp_path / "training_metrics.jsonl").exists()
    assert not list(tmp_path.glob("step_*"))


def test_successful_optimizer_update_resets_consecutive_skip_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import accelerate

    baseline = _config(tmp_path, max_steps=2, save_every=2)
    config = Stage1TrainingConfig(
        **{
            **baseline.to_dict(),
            "max_consecutive_skipped_updates": 2,
        }
    )
    artifacts = _artifacts(config)
    monkeypatch.setattr(
        accelerate,
        "Accelerator",
        _skipping_accelerator([True, False, True, False]),
    )

    result = run_training(config, artifacts=artifacts)

    assert result.global_step == 2
    assert result.cursor.consumed_microbatches == 16
    assert artifacts.scheduler.last_epoch == 2
    records = [
        json.loads(line)
        for line in (tmp_path / "training_metrics.jsonl").read_text().splitlines()
    ]
    assert [record["step"] for record in records] == [1, 2]


def test_rank_zero_jsonl_metrics_cover_the_full_accumulation_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from accelerate import Accelerator

    config = _config(tmp_path)
    artifacts = _artifacts(config)

    def unexpected_tracker_log(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("training must not call an unconfigured tracker")

    monkeypatch.setattr(Accelerator, "log", unexpected_tracker_log)

    result = run_training(config, artifacts=artifacts)

    records = [
        json.loads(line)
        for line in (tmp_path / "training_metrics.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["step"] == 1
    assert records[0]["metrics"]["microbatches"] == 4.0
    assert records[0]["metrics"]["loss/total"] == pytest.approx(
        result.last_metrics["loss/total"]
    )
    assert records[0]["metrics"]["throughput"] > 0
    assert records[0] == _durable_metrics_record(
        step=1,
        metrics=result.last_metrics,
    )


def test_throughput_uses_slowest_rank_elapsed_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from accelerate import Accelerator

    config = _config(tmp_path)
    artifacts = _artifacts(config)
    original_reduce = Accelerator.reduce
    original_gather = Accelerator.gather
    reductions: list[tuple[int, str]] = []
    gather_calls = 0

    def simulated_two_rank_reduce(
        self: Accelerator,
        tensor: torch.Tensor,
        reduction: str = "sum",
        scale: float = 1.0,
    ) -> torch.Tensor:
        reductions.append((tensor.numel(), reduction))
        return original_reduce(self, tensor, reduction=reduction, scale=scale)

    def simulated_two_rank_gather(
        self: Accelerator,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal gather_calls
        gather_calls += 1
        if tensor.numel() == 1:
            return tensor.new_tensor([float(tensor.item()), 4.0])
        return original_gather(self, tensor)

    monkeypatch.setattr(Accelerator, "reduce", simulated_two_rank_reduce)
    monkeypatch.setattr(Accelerator, "gather", simulated_two_rank_gather)

    result = run_training(config, artifacts=artifacts)

    assert result.last_metrics["throughput"] == pytest.approx(32.0)
    assert gather_calls == 1
    assert any(size > 1 and reduction == "mean" for size, reduction in reductions)


def test_resume_ignores_partial_record_before_complete_record_for_same_step(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, max_steps=4, save_every=2)
    first = run_training(config, artifacts=_artifacts(config), stop_at_step=2)
    metrics_path = tmp_path / "training_metrics.jsonl"
    complete_step_two = _durable_metrics_record(
        step=2,
        metrics=dict(first.last_metrics),
    )
    future_step = _durable_metrics_record(
        step=3,
        metrics=dict(first.last_metrics),
    )
    metrics_path.write_text(
        "\n".join(
            (
                json.dumps({"step": 2, "loss/total": 999.0}),
                json.dumps(complete_step_two),
                json.dumps(future_step),
                "malformed crash tail",
                "",
            )
        ),
        encoding="utf-8",
    )
    resumed_config = _config(
        tmp_path,
        max_steps=4,
        save_every=2,
        resume_from=first.checkpoint,
    )

    run_training(resumed_config, artifacts=_artifacts(resumed_config))

    reconciled = [
        json.loads(line) for line in metrics_path.read_text().splitlines()
    ]
    assert [record["step"] for record in reconciled] == [2, 3, 4]
    assert reconciled[0] == complete_step_two


def test_resume_rejects_metrics_record_with_corrupt_checksum(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, max_steps=4, save_every=2)
    first = run_training(config, artifacts=_artifacts(config), stop_at_step=2)
    metrics_path = tmp_path / "training_metrics.jsonl"
    corrupt = _durable_metrics_record(step=2, metrics=dict(first.last_metrics))
    corrupt["checksum"] = "0" * 64
    metrics_path.write_text(json.dumps(corrupt) + "\n", encoding="utf-8")
    resumed_config = _config(
        tmp_path,
        max_steps=4,
        save_every=2,
        resume_from=first.checkpoint,
    )

    run_training(resumed_config, artifacts=_artifacts(resumed_config))

    records = [
        json.loads(line) for line in metrics_path.read_text().splitlines()
    ]
    assert [record["step"] for record in records] == [3, 4]


def test_resume_deduplicates_identical_integrity_valid_metrics_records(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, max_steps=4, save_every=2)
    first = run_training(config, artifacts=_artifacts(config), stop_at_step=2)
    metrics_path = tmp_path / "training_metrics.jsonl"
    complete = _durable_metrics_record(step=2, metrics=dict(first.last_metrics))
    serialized = json.dumps(complete)
    metrics_path.write_text(f"{serialized}\n{serialized}\n", encoding="utf-8")
    resumed_config = _config(
        tmp_path,
        max_steps=4,
        save_every=2,
        resume_from=first.checkpoint,
    )

    run_training(resumed_config, artifacts=_artifacts(resumed_config))

    records = [
        json.loads(line) for line in metrics_path.read_text().splitlines()
    ]
    assert [record["step"] for record in records] == [2, 3, 4]
    assert records[0] == complete


def test_resume_fails_closed_on_conflicting_integrity_valid_metrics_records(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, max_steps=4, save_every=2)
    first = run_training(config, artifacts=_artifacts(config), stop_at_step=2)
    metrics_path = tmp_path / "training_metrics.jsonl"
    first_record = _durable_metrics_record(
        step=2,
        metrics=dict(first.last_metrics),
    )
    conflicting_metrics = dict(first.last_metrics)
    conflicting_metrics["loss/total"] += 1.0
    second_record = _durable_metrics_record(
        step=2,
        metrics=conflicting_metrics,
    )
    original = f"{json.dumps(first_record)}\n{json.dumps(second_record)}\n"
    metrics_path.write_text(original, encoding="utf-8")
    resumed_config = _config(
        tmp_path,
        max_steps=4,
        save_every=2,
        resume_from=first.checkpoint,
    )

    with pytest.raises(ValueError, match="conflicting.*step 2"):
        run_training(resumed_config, artifacts=_artifacts(resumed_config))

    assert metrics_path.read_text(encoding="utf-8") == original


def test_fresh_training_rejects_a_stale_metrics_file(tmp_path: Path) -> None:
    metrics_path = tmp_path / "training_metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text('{"step": 99, "loss/total": 0.0}\n')
    config = _config(tmp_path)

    with pytest.raises(FileExistsError, match="stale Stage-1 metrics"):
        run_training(config, artifacts=_artifacts(config))

    assert metrics_path.read_text() == '{"step": 99, "loss/total": 0.0}\n'


def test_fresh_training_atomically_publishes_trusted_planner_topology(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    result = run_training(config, artifacts=_artifacts(config))

    topology_path = tmp_path / "planner_topology.json"
    payload = json.loads(topology_path.read_text())
    metadata = json.loads((result.checkpoint / "metadata.json").read_text())
    assert set(payload) == {"format_version", "topology", "sha256"}
    assert payload["format_version"] == 1
    assert payload["sha256"] == metadata["planner_topology_hash"]
    assert not list(tmp_path.glob(".planner_topology.json.incomplete-*"))


def test_interrupted_non_epoch_boundary_resume_matches_uninterrupted_training(
    tmp_path: Path,
) -> None:
    full_config = _config(tmp_path / "full", max_steps=4, save_every=2)
    full_artifacts = _artifacts(full_config)
    full = run_training(full_config, artifacts=full_artifacts)
    resume_config = _config(tmp_path / "resume", max_steps=4, save_every=2)
    first_artifacts = _artifacts(resume_config)
    first = run_training(
        resume_config,
        artifacts=first_artifacts,
        stop_at_step=2,
    )
    assert first.cursor.epoch == 0
    assert first.cursor.consumed_microbatches == 8
    assert first.cursor.microbatches_per_epoch == 25
    assert resume_config.gradient_accumulation_steps > 1
    resumed_config = _config(
        tmp_path / "resume",
        max_steps=4,
        save_every=2,
        resume_from=first.checkpoint,
    )
    resumed_artifacts = _artifacts(resumed_config)
    resumed = run_training(resumed_config, artifacts=resumed_artifacts)

    _assert_nested_equal(
        full_artifacts.planner.state_dict(), resumed_artifacts.planner.state_dict()
    )
    _assert_nested_equal(
        full_artifacts.optimizer.state_dict(),
        resumed_artifacts.optimizer.state_dict(),
    )
    _assert_nested_equal(
        full_artifacts.scheduler.state_dict(),
        resumed_artifacts.scheduler.state_dict(),
    )
    assert full.cursor == resumed.cursor
    assert full.cursor.global_step == 4
    assert full.cursor.consumed_microbatches == 16


def _write_preflight_fixture(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    qwen = tmp_path / "qwen"
    processor = tmp_path / "processor"
    tokenizer = tmp_path / "tokenizer"
    siglip = tmp_path / "siglip"
    dataset = tmp_path / "dataset"
    for path in (qwen, processor, tokenizer, siglip, dataset):
        path.mkdir()
    (qwen / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "num_hidden_layers": 24,
                    "hidden_size": 2048,
                    "intermediate_size": 6144,
                },
                "vision_config": {
                    "depth": 24,
                    "hidden_size": 1024,
                    "out_hidden_size": 2048,
                },
            }
        )
    )
    (processor / "processor_config.json").write_text("{}")
    from qwen35_baton.sequence import ADDED_TOKENS

    (tokenizer / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": 100 + index, "content": token}
                    for index, token in enumerate(ADDED_TOKENS)
                ]
            }
        )
    )
    (siglip / "config.json").write_text(
        json.dumps(
            {
                "model_type": "siglip",
                "vision_config": {
                    "image_size": 256,
                    "hidden_size": 1024,
                },
            }
        )
    )
    manifest = dataset / "manifest.json"
    manifest.write_text('{"format_version":1}\n')
    from qwen35_baton.hashing import sha256_file

    payload = {
        "output_dir": str(tmp_path / "output"),
        "qwen_model_path": str(qwen),
        "qwen_processor_path": str(processor),
        "qwen_tokenizer_path": str(tokenizer),
        "siglip2_model_path": str(siglip),
        "siglip2_config_hash": sha256_file(siglip / "config.json"),
        "siglip2_artifact_hash": __import__(
            "qwen35_baton.hashing", fromlist=["sha256_artifact"]
        ).sha256_artifact(siglip),
        "hdf5_manifest_path": str(manifest),
        "hdf5_manifest_hash": sha256_file(manifest),
        "dataset_statistics_path": str(dataset / "stats.json"),
        "per_device_batch": 2,
        "gradient_accumulation_steps": 8,
        "max_steps": 30_000,
        "warmup_steps": 1_000,
        "save_every": 5_000,
    }
    (dataset / "stats.json").write_text("{}")
    config = tmp_path / "stage1.json"
    config.write_text(json.dumps(payload))
    return config, payload


def test_preflight_is_cpu_side_and_validates_local_artifact_contracts(
    tmp_path: Path,
) -> None:
    config, _ = _write_preflight_fixture(tmp_path)
    cuda_was_initialized = torch.cuda.is_initialized()

    report = preflight_stage1(config, world_size=8)

    assert torch.cuda.is_initialized() is cuda_was_initialized
    assert report["global_batch"] == 128
    assert report["qwen_backbone"] == "dense Qwen3.5-2B"
    assert report["siglip2_geometry"] == {
        "image_size": 256,
        "patch_size": 16,
        "hidden_size": 1024,
    }
    assert len(set(report["added_token_ids"])) == 7


def test_preflight_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    config, payload = _write_preflight_fixture(tmp_path)
    payload["hdf5_manifest_hash"] = "f" * 64
    config.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="manifest hash"):
        preflight_stage1(config, world_size=8)


def test_production_config_rejects_any_nonapproved_step_or_save_cadence(
    tmp_path: Path,
) -> None:
    config, payload = _write_preflight_fixture(tmp_path)
    payload["max_steps"] = 20_000
    config.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="30000.*5000"):
        Stage1TrainingConfig.from_json(config)


def test_preflight_rejects_dense_qwen_or_siglip_geometry_mismatch(
    tmp_path: Path,
) -> None:
    config, payload = _write_preflight_fixture(tmp_path)
    qwen_config = Path(payload["qwen_model_path"]) / "config.json"
    qwen_payload = json.loads(qwen_config.read_text())
    qwen_payload["model_type"] = "qwen3_5_moe"
    qwen_config.write_text(json.dumps(qwen_payload))
    with pytest.raises(ValueError, match="dense Qwen3.5-2B"):
        preflight_stage1(config, world_size=8)

    qwen_payload["model_type"] = "qwen3_5"
    qwen_config.write_text(json.dumps(qwen_payload))
    siglip_config = Path(payload["siglip2_model_path"]) / "config.json"
    siglip_payload = json.loads(siglip_config.read_text())
    siglip_payload["vision_config"]["patch_size"] = 14
    siglip_config.write_text(json.dumps(siglip_payload))
    with pytest.raises(ValueError, match="patch_size.*16"):
        preflight_stage1(config, world_size=8)


def test_preflight_rejects_siglip_lookalikes_by_exact_hash(
    tmp_path: Path,
) -> None:
    config, payload = _write_preflight_fixture(tmp_path)
    siglip = Path(payload["siglip2_model_path"])
    (siglip / "lookalike.bin").write_bytes(b"different artifact")

    with pytest.raises(ValueError, match="SigLIP2 artifact hash mismatch"):
        preflight_stage1(config, world_size=8)


def test_preflight_rejects_output_ancestor_of_model_or_dataset(
    tmp_path: Path,
) -> None:
    config, payload = _write_preflight_fixture(tmp_path)
    payload["output_dir"] = str(tmp_path)
    config.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="output.*ancestor"):
        preflight_stage1(config, world_size=8)


def test_local_model_loading_forces_offline_only_transformers_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path, _ = _write_preflight_fixture(tmp_path)
    config = Stage1TrainingConfig.from_json(config_path)
    calls: list[tuple[str, dict[str, Any]]] = []

    class _Loader:
        def __init__(self, label: str) -> None:
            self.label = label

        def from_pretrained(self, path: str, **kwargs: Any) -> Any:
            calls.append((self.label, kwargs))
            if self.label == "qwen-tokenizer":
                return SimpleNamespace(
                    convert_tokens_to_ids=lambda token: {
                        value: 100 + index
                        for index, value in enumerate(
                            __import__(
                                "qwen35_baton.sequence", fromlist=["ADDED_TOKENS"]
                            ).ADDED_TOKENS
                        )
                    }[token]
                )
            if self.label == "qwen-processor":
                return SimpleNamespace(tokenizer=None)
            if self.label == "qwen-model":
                return nn.Linear(1, 1)
            if self.label == "siglip-model":
                return SimpleNamespace(vision_model=nn.Linear(1, 1))
            return lambda **_: {}

    fake_transformers = SimpleNamespace(
        AutoTokenizer=_Loader("qwen-tokenizer"),
        AutoProcessor=_Loader("qwen-processor"),
        AutoModelForImageTextToText=_Loader("qwen-model"),
        AutoImageProcessor=_Loader("siglip-processor"),
        AutoModel=_Loader("siglip-model"),
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

    load_local_artifacts(config, models_only=True)

    assert calls
    assert all(kwargs.get("local_files_only") is True for _, kwargs in calls)


def test_stage1_recipe_requirements_and_launchers_are_fixed(tmp_path: Path) -> None:
    config = json.loads(
        (REPO_ROOT / "qwen35_baton/configs/libero_stage1.json").read_text()
    )
    assert config["max_steps"] == 30_000
    assert config["save_every"] == 5_000
    assert config["max_consecutive_skipped_updates"] == 8
    assert config["planner_lr"] == 5e-5
    assert config["qwen_top8_lr"] == 1e-6
    assert config["qwen_vision_lr"] == 5e-7
    requirements = [
        line.strip()
        for line in (
            REPO_ROOT / "ge_act/requirements-qwen35-baton.txt"
        ).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert requirements == ["-r requirements.txt"]

    log = tmp_path / "python.log"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "${BATON_TEST_LOG}"\n'
    )
    fake_python.chmod(0o755)
    environment = {
        **os.environ,
        "BATON_TEST_LOG": str(log),
        "PYTHON_BIN": str(fake_python),
        "CONFIG": str(tmp_path / "config.json"),
        "NUM_GPUS": "2",
        "PER_DEVICE_BATCH": "32",
        "GLOBAL_BATCH": "128",
    }

    subprocess.run(
        ["bash", str(REPO_ROOT / "qwen35_baton/scripts/train_semantic_planner.sh")],
        check=True,
        cwd=REPO_ROOT,
        env=environment,
    )

    calls = log.read_text().splitlines()
    assert calls[0].startswith("-m qwen35_baton.cli.preflight ")
    assert "--per-device-batch 32" in calls[0]
    assert "--gradient-accumulation-steps 2" in calls[0]
    assert calls[1].startswith("-m torch.distributed.run ")
    assert "--per-device-batch 32" in calls[1]
    assert "--gradient-accumulation-steps 2" in calls[1]
    assert environment["GLOBAL_BATCH"] == "128"


def test_production_artifacts_are_constructed_after_global_seeding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import qwen35_baton.cli.preflight as preflight_module
    import qwen35_baton.cli.train_semantic_planner as training_module

    draws: list[tuple[float, float, float]] = []

    def fake_loader(config: Stage1TrainingConfig) -> Stage1TrainingArtifacts:
        draws.append(
            (random.random(), float(__import__("numpy").random.random()), float(torch.rand(())))
        )
        return _artifacts(config)

    monkeypatch.setattr(preflight_module, "preflight_stage1", lambda *args, **kwargs: {})
    monkeypatch.setattr(training_module, "load_local_artifacts", fake_loader)
    first_config = _config(tmp_path / "first")
    second_config = Stage1TrainingConfig(
        **{**first_config.to_dict(), "output_dir": str(tmp_path / "second")}
    )
    random.seed(1)
    __import__("numpy").random.seed(2)
    torch.manual_seed(3)
    run_training(first_config)
    random.seed(101)
    __import__("numpy").random.seed(102)
    torch.manual_seed(103)
    run_training(second_config)

    assert draws[0] == draws[1]


def test_run_training_passes_preflight_a_module_stable_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import qwen35_baton.cli.preflight as preflight_module
    import qwen35_baton.cli.train_semantic_planner as training_module

    received: list[Any] = []
    config = _config(tmp_path / "run")
    monkeypatch.setattr(
        preflight_module,
        "preflight_stage1",
        lambda payload, **_: received.append(payload),
    )
    monkeypatch.setattr(
        training_module,
        "load_local_artifacts",
        lambda _: _artifacts(config),
    )

    run_training(config)

    assert len(received) == 1
    assert received[0] == config.to_dict()


def test_launcher_rejects_nondivisible_global_batch_before_preflight(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "qwen35_baton/scripts/train_semantic_planner.sh")],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "CONFIG": str(tmp_path / "config.json"),
            "NUM_GPUS": "3",
            "PER_DEVICE_BATCH": "7",
            "GLOBAL_BATCH": "128",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "divisible" in result.stderr
