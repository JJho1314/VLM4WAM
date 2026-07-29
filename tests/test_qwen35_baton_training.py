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

from qwen35_baton.cli.preflight import (
    preflight_stage1,
    require_qwen35_fast_path,
)
from qwen35_baton.cli.train_semantic_planner import (
    Stage1TrainingArtifacts,
    Stage1TrainingConfig,
    build_cosine_warmup_scheduler,
    build_stage1_optimizer,
    build_stage1_optimizer_groups,
    checkpoint_steps,
    load_local_artifacts,
    require_stage1_global_batch,
    resolve_deepspeed_runtime_config,
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
        return SimpleNamespace(positive=prediction)


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
        self.move_count = 0

    def to(self, device: torch.device | str) -> "_TinyTeacher":
        self.model.to(device)
        self.move_count += 1
        return self

    def encode_future(self, images: torch.Tensor) -> torch.Tensor:
        result = self.model(images)
        if self.nonfinite:
            result = result.clone()
            result.flatten()[0] = float("inf")
        return result.to(self.output_dtype)

    def encode_current(self, images: torch.Tensor) -> torch.Tensor:
        raise AssertionError("strict Baton Stage 1 must not encode current frames")


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
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        warmup_steps=0 if max_steps == 1 else 1,
        save_every=save_every,
        initial_save_step=None,
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
    optimizer = build_stage1_optimizer(planner, ownership, config)
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
            self.sync_gradients = True
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


def test_stage1_trains_the_entire_va_planner() -> None:
    planner = _TinyPlanner()

    ownership = configure_stage1_trainable_modules(planner)

    assert all(parameter.requires_grad for parameter in planner.parameters())
    assert ownership.trainable_modules == (planner,)


def test_stage1_optimizer_is_one_exhaustive_group(tmp_path: Path) -> None:
    config = _config(tmp_path)
    planner = _TinyPlanner()
    ownership = configure_stage1_trainable_modules(planner)

    groups = build_stage1_optimizer_groups(planner, ownership, config)

    assert [(group["name"], group["lr"]) for group in groups] == [
        ("va_planner", 1e-5)
    ]
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


def test_stage1_uses_baton_adamw_contract(tmp_path: Path) -> None:
    config = _config(tmp_path)
    planner = _TinyPlanner()
    ownership = configure_stage1_trainable_modules(planner)

    optimizer = build_stage1_optimizer(planner, ownership, config)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert {group["lr"] for group in optimizer.param_groups} == {1e-5}
    assert {group["weight_decay"] for group in optimizer.param_groups} == {
        config.weight_decay
    }


def test_one_tiny_stage1_step_updates_only_owned_parameters(tmp_path: Path) -> None:
    config = _config(tmp_path)
    artifacts = _artifacts(config)
    before = _clone_parameters(artifacts.planner)
    result = run_training(config, artifacts=artifacts)

    assert result.global_step == 1
    assert all(
        parameter.requires_grad for parameter in artifacts.planner.parameters()
    )
    assert any(
        not torch.equal(parameter.detach(), before[name])
        for name, parameter in artifacts.planner.named_parameters()
    )
    assert all(
        parameter.grad is None for parameter in artifacts.teacher.model.parameters()
    )
    assert artifacts.teacher.move_count == 1
    assert {
        "loss/total",
        "loss/mse",
        "throughput",
        "data_time",
        "qwen_time",
        "teacher_time",
        "query_tower_time",
        "backward_time",
        "mse/main/frame_0",
        "mse/wrist/frame_0",
    }.issubset(result.last_metrics)
    assert not any("cosine" in name for name in result.last_metrics)
    assert not any("counterfactual" in name for name in result.last_metrics)
    assert result.last_metrics["qwen_time"] > 0
    assert result.last_metrics["query_tower_time"] > 0


def test_global_batch_uses_microbatch_world_size_and_accumulation() -> None:
    assert (
        validate_global_batch(
            per_device_batch=4,
            world_size=8,
            gradient_accumulation_steps=4,
        )
        == 128
    )


def test_production_stage1_requires_global_batch_128() -> None:
    assert (
        require_stage1_global_batch(
            per_device_batch=4,
            world_size=8,
            gradient_accumulation_steps=4,
        )
        == 128
    )

    with pytest.raises(ValueError, match="global batch must be exactly 128"):
        require_stage1_global_batch(
            per_device_batch=4,
            world_size=8,
            gradient_accumulation_steps=2,
        )


def test_stage1_accepts_positive_gradient_accumulation(tmp_path: Path) -> None:
    config = _config(tmp_path)

    accumulated = Stage1TrainingConfig(
        **{**config.to_dict(), "gradient_accumulation_steps": 4}
    )

    assert accumulated.gradient_accumulation_steps == 4


def test_zero2_runtime_config_is_explicit() -> None:
    config = json.loads(
        (REPO_ROOT / "qwen35_baton/configs/deepspeed_zero2.json").read_text()
    )

    assert config["zero_optimization"]["stage"] == 2
    assert config["bf16"]["enabled"] is True
    assert config["train_micro_batch_size_per_gpu"] == "auto"
    assert config["gradient_accumulation_steps"] == "auto"
    assert config["train_batch_size"] == "auto"
    assert "offload_optimizer" not in config["zero_optimization"]
    assert "offload_param" not in config["zero_optimization"]


def test_deepspeed_runtime_config_resolves_micro_and_global_batch(
    tmp_path: Path,
) -> None:
    source = REPO_ROOT / "qwen35_baton/configs/deepspeed_zero2.json"
    config = _config(tmp_path)
    config = Stage1TrainingConfig(
        **{
            **config.to_dict(),
            "tiny_test": False,
            "mixed_precision": "bf16",
                "max_steps": 30_000,
                "warmup_steps": 1_000,
                "save_every": 5_000,
                "initial_save_step": 20,
                "per_device_batch": 2,
            "gradient_accumulation_steps": 4,
            "deepspeed_config_path": str(source),
        }
    )

    resolved = resolve_deepspeed_runtime_config(config, world_size=8)

    assert resolved["train_micro_batch_size_per_gpu"] == 2
    assert resolved["gradient_accumulation_steps"] == 4
    assert resolved["train_batch_size"] == 64
    assert resolved["zero_optimization"]["stage"] == 2


def test_production_stage1_uses_checkpoint_compatible_ddp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import accelerate
    import accelerate.utils
    import qwen35_baton.cli.train_semantic_planner as training_module

    baseline = _config(tmp_path)
    config = Stage1TrainingConfig(
        **{
            **baseline.to_dict(),
            "tiny_test": False,
            "mixed_precision": "bf16",
            "per_device_batch": 32,
            "gradient_accumulation_steps": 4,
            "max_steps": 30_000,
            "warmup_steps": 1_000,
            "save_every": 5_000,
            "initial_save_step": 20,
        }
    )
    artifacts = _artifacts(config)
    fake_accelerator = _skipping_accelerator([False])

    def forbidden_deepspeed_plugin(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "Stage-1 must keep a full AdamW state on every DDP rank"
        )

    monkeypatch.setattr(accelerate, "Accelerator", fake_accelerator)
    monkeypatch.setattr(
        accelerate.utils,
        "DeepSpeedPlugin",
        forbidden_deepspeed_plugin,
    )
    monkeypatch.setattr(
        training_module,
        "_configure_gradient_checkpointing",
        lambda *args, **kwargs: None,
    )

    result = run_training(config, artifacts=artifacts, stop_at_step=1)

    assert result.global_step == 1


def test_qwen35_fast_path_fails_closed_when_compiled_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    real_import = importlib.import_module

    def fail_fla(name: str, package: str | None = None) -> Any:
        if name == "fla":
            raise ModuleNotFoundError("No module named 'fla'")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fail_fla)

    with pytest.raises(RuntimeError, match="flash-linear-attention.*causal-conv1d"):
        require_qwen35_fast_path()


@pytest.mark.parametrize(
    ("enabled", "expected"),
    ((True, "enable"), (False, "disable")),
)
def test_gradient_checkpointing_uses_qwen_public_api(
    enabled: bool,
    expected: str,
) -> None:
    from qwen35_baton.cli.train_semantic_planner import (
        _configure_gradient_checkpointing,
    )

    calls: list[str] = []

    class _Backbone(nn.Module):
        def gradient_checkpointing_enable(self) -> None:
            calls.append("enable")

        def gradient_checkpointing_disable(self) -> None:
            calls.append("disable")

    planner = nn.Module()
    planner.backbone = _Backbone()

    _configure_gradient_checkpointing(planner, enabled=enabled)

    assert calls == [expected]


def test_checkpoint_cadence_probes_step_20_then_saves_every_5000() -> None:
    assert checkpoint_steps(
        max_steps=30_000,
        save_every=5_000,
        initial_save_step=20,
    ) == (
        20,
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


def test_each_microbatch_is_one_optimizer_update(
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
            epoch=0,
            consumed_microbatches=2,
        microbatches_per_epoch=5,
        sampler_seed=config.seed,
    )
    assert artifacts.scheduler.last_epoch == 2


def test_four_microbatches_make_one_optimizer_and_scheduler_update(
    tmp_path: Path,
) -> None:
    baseline = _config(tmp_path)
    config = Stage1TrainingConfig(
        **{
            **baseline.to_dict(),
            "gradient_accumulation_steps": 4,
        }
    )
    artifacts = _artifacts(config)
    before = _clone_parameters(artifacts.planner)

    result = run_training(config, artifacts=artifacts)

    assert result.global_step == 1
    assert result.cursor.consumed_microbatches == 4
    assert artifacts.scheduler.last_epoch == 1
    assert result.last_metrics["microbatches"] == 4.0
    assert any(
        not torch.equal(parameter.detach(), before[name])
        for name, parameter in artifacts.planner.named_parameters()
    )


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
    assert result.cursor.consumed_microbatches == 2
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
                ".*global_step=0.*epoch=0.*consumed_microbatches=8"
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
    assert result.cursor.consumed_microbatches == 4
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
    assert records[0]["metrics"]["microbatches"] == 1.0
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

    assert result.last_metrics["throughput"] == pytest.approx(8.0)
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
    assert first.cursor.consumed_microbatches == 2
    assert first.cursor.microbatches_per_epoch == 25
    assert resume_config.gradient_accumulation_steps == 1
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
    assert full.cursor.consumed_microbatches == 4


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
        "per_device_batch": 4,
        "gradient_accumulation_steps": 4,
        "max_steps": 30_000,
        "warmup_steps": 1_000,
        "save_every": 5_000,
    }
    (dataset / "stats.json").write_text("{}")
    config = tmp_path / "stage1.json"
    config.write_text(json.dumps(payload))
    return config, payload


def test_preflight_is_cpu_side_and_validates_local_artifact_contracts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "qwen35_baton.cli.preflight.require_qwen35_fast_path",
        lambda: {"fla": "test", "causal_conv1d": "test"},
    )
    config, _ = _write_preflight_fixture(tmp_path)
    cuda_was_initialized = torch.cuda.is_initialized()

    report = preflight_stage1(config, world_size=8)

    assert torch.cuda.is_initialized() is cuda_was_initialized
    assert report["global_batch"] == 128
    assert report["qwen35_fast_path"] == {
        "fla": "test",
        "causal_conv1d": "test",
    }
    assert report["qwen_backbone"] == "dense Qwen3.5-2B"
    assert report["distributed_strategy"] == "ddp"
    assert report["siglip2_geometry"] == {
        "image_size": 256,
        "patch_size": 16,
        "hidden_size": 1024,
    }
    assert len(set(report["added_token_ids"])) == 7


def test_preflight_rejects_non_128_production_global_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "qwen35_baton.cli.preflight.require_qwen35_fast_path",
        lambda: {"fla": "test", "causal_conv1d": "test"},
    )
    config, payload = _write_preflight_fixture(tmp_path)
    payload["gradient_accumulation_steps"] = 2
    config.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="global batch must be exactly 128"):
        preflight_stage1(config, world_size=8)


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
    assert config["per_device_batch"] == 4
    assert config["gradient_accumulation_steps"] == 4
    assert config["gradient_checkpointing"] is False
    assert config["learning_rate"] == 1e-5
    assert "planner_lr" not in config
    assert "qwen_top8_lr" not in config
    assert "qwen_vision_lr" not in config
    requirements = [
        line.strip()
        for line in (
            REPO_ROOT / "ge_act/requirements-qwen35-baton.txt"
        ).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert requirements == [
        "-r requirements.txt",
        "flash-linear-attention[cuda]",
        "causal-conv1d",
    ]

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
        "GRAD_ACCUM": "4",
        "DEEPSPEED_CONFIG": str(
            REPO_ROOT / "qwen35_baton/configs/deepspeed_zero2.json"
        ),
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
    assert "--gradient-accumulation-steps 4" in calls[0]
    assert "--deepspeed-config-path" not in calls[0]
    assert calls[1].startswith("-m torch.distributed.run ")
    assert "--per-device-batch 32" in calls[1]
    assert "--gradient-accumulation-steps 4" in calls[1]
    assert "--deepspeed-config-path" not in calls[1]


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


def test_launcher_rejects_nonpositive_gradient_accumulation(
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
            "GRAD_ACCUM": "0",
        },
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "GRAD_ACCUM must be positive" in result.stderr
