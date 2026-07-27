from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace
from typing import Any, Sequence
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml
from accelerate.data_loader import skip_first_batches
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Subset
from qwen35_baton.checkpoint import (
    BatonTrainingCursor,
    capture_rank_rng_state,
    planner_module_topology,
    planner_safetensors_topology,
    publish_trusted_planner_topology,
    save_baton_checkpoint,
)
from qwen35_baton.cli.train_semantic_planner import (
    BatonCosineWarmupScheduler,
)
from qwen35_baton.config import BatonCheckpointMetadata
from qwen35_baton.hashing import sha256_json


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GE_ACT_ROOT = REPOSITORY_ROOT / "ge_act"
for path in (REPOSITORY_ROOT, GE_ACT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.ltx_models.semantic_conditioning import (  # noqa: E402
    build_patch_center_positions,
)
from data.libero_fastwam_hdf5_dataset import (  # noqa: E402
    LiberoFastWAMHDF5Dataset,
)
from data.libero_fastwam_hdf5_schema import EpisodeRecord  # noqa: E402
from runner import ge_inferencer as ge_inferencer_module  # noqa: E402
from runner import ge_trainer as ge_trainer_module  # noqa: E402
from runner.ge_trainer import (  # noqa: E402
    BatonConditioningComponents,
    Trainer,
    TrainingCursor,
    build_baton_semantic_condition,
    build_optimizer_parameter_groups,
    forward_baton_ge_act,
    load_baton_training_checkpoint,
    prepare_baton_conditioning,
    save_baton_training_checkpoint,
    validate_baton_stage2_checkpoint_envelope,
)
from scripts.preflight_libero_fastwam_hdf5 import (  # noqa: E402
    collect_hdf5_preflight_errors,
)
from scripts.preflight_ltx_siglip2 import (  # noqa: E402
    collect_preflight_errors,
    materialize_baton_config,
)


STAGE2_CONFIG = (
    GE_ACT_ROOT
    / "configs/ltx_model/libero/action_model_libero_baton_stage2_hdf5.yaml"
)
STAGE3_CONFIG = (
    GE_ACT_ROOT
    / "configs/ltx_model/libero/action_model_libero_baton_stage3_hdf5.yaml"
)
STAGE2_LAUNCHER = GE_ACT_ROOT / "scripts/train_ltx_baton_stage2.sh"
STAGE3_LAUNCHER = GE_ACT_ROOT / "scripts/train_ltx_baton_stage3.sh"
STAGE2_SBATCH = GE_ACT_ROOT / "scripts/sbatch_train_ltx_baton_stage2_hpc3.sh"
STAGE3_SBATCH = GE_ACT_ROOT / "scripts/sbatch_train_ltx_baton_stage3_hpc3.sh"
FUTURE_INDICES = (0, 3, 5, 8)
BATON_SAMPLING_ALGORITHM = "libero_fastwam_hdf5_stateless_sha256"
BATON_SAMPLING_VERSION = 1


class _CompactWindowDataset(LiberoFastWAMHDF5Dataset):
    """Exercise the production sampler while keeping worker IPC payloads tiny."""

    def read_by_indexes(
        self,
        index: int,
        frame_indexes: Sequence[int],
        action_indexes: Sequence[int],
    ) -> dict[str, torch.Tensor | str]:
        record = self.records[self._resolve_index(index)]
        return {
            "video": torch.tensor(frame_indexes, dtype=torch.int64),
            "actions": torch.tensor(action_indexes, dtype=torch.int64),
            "state": torch.tensor([index], dtype=torch.int64),
            "caption": record.key,
        }


def _write_compact_window_fixture(
    root: Path,
    *,
    episodes: int = 12,
) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    shard = root / "shard_00000.h5"
    shard.touch()
    records = [
        {
            "key": f"domain:{index:06d}",
            "shard": shard.name,
            "group": f"episodes/domain:{index:06d}",
            "caption": f"caption {index}",
            "domain": "domain",
            "episode_index": index,
            "length": 80 + index,
        }
        for index in range(episodes)
    ]
    manifest = {
        "schema_version": 1,
        "camera_names": ["main", "wrist"],
        "image_size": [256, 256],
        "source_fps": 20,
        "n_previous": 4,
        "chunk": 9,
        "action_chunk": 36,
        "action_type": "absolute",
        "action_space": "eef",
        "compression": "none",
        "source_roots": [str(root / "source")],
        "datasets": {
            "rgb_main": {
                "shape_tail": [256, 256, 3],
                "dtype": "uint8",
            },
            "rgb_wrist": {
                "shape_tail": [256, 256, 3],
                "dtype": "uint8",
            },
            "action": {"width": 7, "dtype": "float32"},
            "state": {"width": 8, "dtype": "float32"},
        },
        "converter_fingerprint": "a" * 64,
        "episodes": records,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    statistics_path = root / "stats.json"
    statistics_path.write_text(
        json.dumps(
            {
                "domain_eef": {
                    "mean": [0.0] * 7,
                    "std": [1.0] * 7,
                },
                "domain_state_eef": {
                    "mean": [0.0] * 8,
                    "std": [1.0] * 8,
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, statistics_path


def _compact_window_dataset(
    root: Path,
    *,
    seed: int = 42,
) -> _CompactWindowDataset:
    manifest, statistics = _write_compact_window_fixture(root)
    return _CompactWindowDataset(
        manifest_path=manifest,
        stat_file=statistics,
        baton_sampling_algorithm=BATON_SAMPLING_ALGORITHM,
        baton_sampling_version=BATON_SAMPLING_VERSION,
        baton_sampling_seed=seed,
    )


def _compact_window_loader(
    dataset: torch.utils.data.Dataset,
    *,
    num_workers: int,
    sampler_seed: int = 42,
) -> DataLoader:
    return DataLoader(
        dataset,
        sampler=ge_trainer_module.EpochSeededRandomSampler(
            dataset,
            seed=sampler_seed,
        ),
        batch_size=2,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        generator=torch.Generator().manual_seed(sampler_seed),
    )


def _collect_compact_batches(loader: DataLoader) -> list[dict[str, Any]]:
    batches = []
    try:
        for batch in loader:
            batches.append(
                {
                    "video": batch["video"].clone(),
                    "actions": batch["actions"].clone(),
                    "state": batch["state"].clone(),
                    "caption": tuple(batch["caption"]),
                }
            )
    finally:
        iterator = getattr(loader, "_iterator", None)
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
    return batches


def _assert_compact_batches_equal(
    actual: Sequence[dict[str, Any]],
    expected: Sequence[dict[str, Any]],
) -> None:
    assert len(actual) == len(expected)
    for actual_batch, expected_batch in zip(actual, expected, strict=True):
        assert actual_batch["caption"] == expected_batch["caption"]
        for field in ("video", "actions", "state"):
            torch.testing.assert_close(
                actual_batch[field],
                expected_batch[field],
                rtol=0,
                atol=0,
            )


def test_baton_stateless_windows_resume_exactly_across_worker_assignments(
    tmp_path: Path,
) -> None:
    epoch = 7
    interrupted_batches = 3
    baseline_dataset = _compact_window_dataset(tmp_path / "baseline")
    baseline_loader = _compact_window_loader(
        baseline_dataset,
        num_workers=2,
    )
    ge_trainer_module.set_dataloader_epoch(
        baseline_loader,
        epoch=epoch,
        sampler_seed=42,
    )
    baseline = _collect_compact_batches(baseline_loader)

    interrupted_dataset = _compact_window_dataset(tmp_path / "interrupted")
    interrupted_loader = _compact_window_loader(
        interrupted_dataset,
        num_workers=1,
    )
    ge_trainer_module.set_dataloader_epoch(
        interrupted_loader,
        epoch=epoch,
        sampler_seed=42,
    )
    interrupted = []
    iterator = iter(interrupted_loader)
    try:
        for _ in range(interrupted_batches):
            batch = next(iterator)
            interrupted.append(
                {
                    "video": batch["video"].clone(),
                    "actions": batch["actions"].clone(),
                    "state": batch["state"].clone(),
                    "caption": tuple(batch["caption"]),
                }
            )
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
    _assert_compact_batches_equal(
        interrupted,
        baseline[:interrupted_batches],
    )

    resumed_dataset = _compact_window_dataset(tmp_path / "resumed")
    resumed_loader = _compact_window_loader(
        resumed_dataset,
        num_workers=3,
    )
    ge_trainer_module.set_dataloader_epoch(
        resumed_loader,
        epoch=epoch,
        sampler_seed=42,
    )
    remaining_loader = skip_first_batches(
        resumed_loader,
        num_batches=interrupted_batches,
    )
    remaining = _collect_compact_batches(remaining_loader)

    _assert_compact_batches_equal(
        remaining,
        baseline[interrupted_batches:],
    )


def test_baton_epoch_reaches_subset_workers_and_item_rng_is_local(
    tmp_path: Path,
) -> None:
    dataset = _compact_window_dataset(tmp_path / "dataset")
    subset = Subset(dataset, [0, 1, 2, 3])
    loader = _compact_window_loader(subset, num_workers=1)

    random.seed(123)
    np.random.seed(123)
    expected_python = random.Random(123).random()
    expected_numpy = np.random.RandomState(123).rand()
    ge_trainer_module.set_dataloader_epoch(
        loader,
        epoch=9,
        sampler_seed=42,
    )
    first = dataset[0]
    second = dataset[0]

    assert dataset.baton_sampling_epoch == 9
    torch.testing.assert_close(first["video"], second["video"], rtol=0, atol=0)
    torch.testing.assert_close(
        first["actions"],
        second["actions"],
        rtol=0,
        atol=0,
    )
    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy


def test_baton_dataset_without_stateless_sampling_contract_fails_closed(
    tmp_path: Path,
) -> None:
    manifest, statistics = _write_compact_window_fixture(tmp_path / "legacy")
    legacy_dataset = _CompactWindowDataset(
        manifest_path=manifest,
        stat_file=statistics,
    )

    with pytest.raises(ValueError, match="stateless.*sampling contract"):
        ge_trainer_module.validate_baton_dataset_sampling_contract(
            legacy_dataset,
            expected_seed=42,
        )


def _semantic_config(source: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "semantic_plan": {
            "enabled": True,
            "source": source,
            "tokens_per_frame": 256,
            "feature_dim": 1024,
            "keyframe_indices": [0, 3, 5, 8],
            "dropout": 0.15,
        },
        "data": {
            "train": {
                "manifest_path": "/dataset/manifest.json",
                "n_previous": 4,
                "chunk": 9,
            }
        },
    }
    semantic = config["semantic_plan"]
    if source == "qwen35_baton_teacher":
        semantic.update(
            {
                "siglip2_model_path": "/models/siglip2",
                "siglip2_config_hash": "1" * 64,
                "siglip2_artifact_hash": "2" * 64,
                "teacher_preprocessing_hash": "2" * 64,
                "frame_microbatch_size": 8,
                "validation_mode": "teacher",
                "validation_modes": ["teacher", "semantic_disabled"],
            }
        )
    elif source == "qwen35_baton_prediction":
        semantic.update(
            {
                "planner_checkpoint": "/checkpoints/step_030000",
                "expected_planner_topology": "/checkpoints/planner_topology.json",
                "qwen_model_path": "/models/qwen",
                "qwen_tokenizer_path": "/models/qwen",
                "qwen_processor_path": "/models/qwen",
                "siglip2_model_path": "/models/siglip2",
                "validation_mode": "prediction",
                "validation_modes": ["prediction", "semantic_disabled"],
            }
        )
    return config


class _FakeTeacher:
    calls = 0
    kwargs: dict[str, Any] | None = None

    def __init__(self, model_name_or_path: str, **kwargs: Any) -> None:
        type(self).calls += 1
        type(self).kwargs = {
            "model_name_or_path": model_name_or_path,
            **kwargs,
        }
        self.model = nn.Linear(1, 1)
        self.model.requires_grad_(False)
        self.model.eval()


class _FakePlanner(nn.Module):
    calls = 0
    kwargs: dict[str, Any] | None = None

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()), requires_grad=False)
        self.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint: str, **kwargs: Any) -> "_FakePlanner":
        cls.calls += 1
        cls.kwargs = {"checkpoint": checkpoint, **kwargs}
        return cls()


@pytest.fixture(autouse=True)
def _reset_component_fakes() -> None:
    _FakeTeacher.calls = 0
    _FakeTeacher.kwargs = None
    _FakePlanner.calls = 0
    _FakePlanner.kwargs = None


def test_stage2_constructs_exactly_one_online_teacher_and_never_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ge_trainer_module, "FrozenSiglip2Teacher", _FakeTeacher)
    monkeypatch.setattr(
        ge_trainer_module,
        "FrozenDualCameraBatonPlanner",
        _FakePlanner,
    )
    monkeypatch.setattr(
        ge_trainer_module,
        "validate_baton_siglip2_provenance",
        lambda _semantic: None,
    )

    components = prepare_baton_conditioning(
        _semantic_config("qwen35_baton_teacher"),
        dataset=SimpleNamespace(),
        device="cpu",
        dtype=torch.float32,
    )

    assert components.source == "qwen35_baton_teacher"
    assert components.teacher is not None
    assert components.planner is None
    assert _FakeTeacher.calls == 1
    assert _FakePlanner.calls == 0
    assert all(not parameter.requires_grad for parameter in components.teacher.model.parameters())
    assert not components.teacher.model.training


def test_stage3_constructs_exactly_one_frozen_planner_and_never_teacher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ge_trainer_module, "FrozenSiglip2Teacher", _FakeTeacher)
    monkeypatch.setattr(
        ge_trainer_module,
        "FrozenDualCameraBatonPlanner",
        _FakePlanner,
    )

    components = prepare_baton_conditioning(
        _semantic_config("qwen35_baton_prediction"),
        dataset=SimpleNamespace(),
        device="cpu",
        dtype=torch.float32,
    )

    assert components.source == "qwen35_baton_prediction"
    assert components.teacher is None
    assert components.planner is not None
    assert _FakeTeacher.calls == 0
    assert _FakePlanner.calls == 1
    assert all(not parameter.requires_grad for parameter in components.planner.parameters())
    assert not components.planner.training
    assert _FakePlanner.kwargs == {
        "checkpoint": "/checkpoints/step_030000",
        "qwen_model_path": "/models/qwen",
        "qwen_tokenizer_path": "/models/qwen",
        "qwen_processor_path": "/models/qwen",
        "siglip2_model_path": "/models/siglip2",
        "expected_planner_topology": "/checkpoints/planner_topology.json",
        "device": torch.device("cpu"),
        "torch_dtype": torch.float32,
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda config: config["semantic_plan"].pop("source"), "source"),
        (
            lambda config: config["semantic_plan"].update(
                {"source": "gt_siglip2"}
            ),
            "qwen35_baton",
        ),
        (
            lambda config: config["semantic_plan"].update(
                {"planner_checkpoint": "/ambiguous"}
            ),
            "planner",
        ),
        (
            lambda config: config["data"]["train"].update(
                {"hindsight_cache": "/forbidden"}
            ),
            "hindsight",
        ),
        (
            lambda config: config.update({"planner_aux_weight": 0.25}),
            "auxiliary",
        ),
        (
            lambda config: config["semantic_plan"].update(
                {"relevance": True}
            ),
            "relevance",
        ),
        (
            lambda config: config["semantic_plan"].update(
                {"semantic_plan_mask": True}
            ),
            "mask",
        ),
    ],
)
def test_baton_source_configs_fail_closed_before_any_component_construction(
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    message: str,
) -> None:
    config = _semantic_config("qwen35_baton_teacher")
    mutate(config)
    monkeypatch.setattr(ge_trainer_module, "FrozenSiglip2Teacher", _FakeTeacher)
    monkeypatch.setattr(
        ge_trainer_module,
        "FrozenDualCameraBatonPlanner",
        _FakePlanner,
    )

    with pytest.raises((TypeError, ValueError), match=message):
        prepare_baton_conditioning(
            config,
            dataset=SimpleNamespace(),
            device="cpu",
            dtype=torch.float32,
        )

    assert _FakeTeacher.calls == 0
    assert _FakePlanner.calls == 0


class _RecordingTeacher:
    def __init__(self) -> None:
        self.model = nn.Linear(1, 1)
        self.model.requires_grad_(False)
        self.model.eval()
        self.calls: list[tuple[torch.Tensor, bool]] = []

    def encode_future(self, images: torch.Tensor) -> torch.Tensor:
        self.calls.append((images.detach().clone(), torch.is_grad_enabled()))
        values = images[:, :, :, 0, 0, 0].float().unsqueeze(-1).unsqueeze(-1)
        return values.expand(-1, -1, -1, 256, 1024).detach()


@dataclass(frozen=True)
class _Plan:
    tokens: torch.Tensor
    future_indices: tuple[int, ...]
    positions_xy: torch.Tensor
    relevance: None = None


class _RecordingPlanner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()), requires_grad=False)
        self.calls: list[tuple[torch.Tensor, tuple[str, ...], bool]] = []

    def predict(
        self,
        current_images: torch.Tensor,
        instructions: Sequence[str],
    ) -> _Plan:
        self.calls.append(
            (
                current_images.detach().clone(),
                tuple(instructions),
                torch.is_grad_enabled(),
            )
        )
        batch_size = current_images.shape[0]
        return _Plan(
            tokens=torch.ones(batch_size, 2, 4, 256, 1024),
            future_indices=FUTURE_INDICES,
            positions_xy=build_patch_center_positions(
                batch_size,
                2,
                4,
            ),
        )


def _video(batch_size: int = 2) -> torch.Tensor:
    video = torch.empty(batch_size, 3, 2, 13, 2, 2)
    for batch in range(batch_size):
        for view in range(2):
            for frame in range(13):
                video[batch, :, view, frame].fill_(
                    -1 + (batch * 40 + view * 14 + frame) / 64
                )
    return video


def test_stage2_batch_selects_both_cameras_at_exact_future_offsets_under_no_grad() -> None:
    teacher = _RecordingTeacher()
    components = BatonConditioningComponents(
        source="qwen35_baton_teacher",
        teacher=teacher,
        planner=None,
    )
    video = _video()

    condition = build_baton_semantic_condition(
        components,
        _semantic_config("qwen35_baton_teacher")["semantic_plan"],
        video,
        ("pick", "place"),
        n_previous=4,
        num_future_frames=9,
        num_latent_frames=6,
        device="cpu",
        dtype=torch.float32,
    )

    assert len(teacher.calls) == 1
    selected, grad_enabled = teacher.calls[0]
    torch.testing.assert_close(
        selected,
        video[:, :, :, [4, 7, 9, 12]].permute(0, 2, 3, 1, 4, 5),
    )
    assert grad_enabled is False
    assert condition.tokens.shape == (2, 2, 4, 256, 1024)
    assert condition.positions.shape == (2, 2, 4, 256, 2)
    torch.testing.assert_close(
        condition.times,
        torch.tensor([[0.8, 0.875, 0.925, 1.0]] * 4),
    )
    assert condition.mask is None
    assert condition.relevance is None
    assert not condition.tokens.requires_grad


def test_stage3_batch_uses_last_current_observation_and_keeps_provider_positions() -> None:
    planner = _RecordingPlanner()
    components = BatonConditioningComponents(
        source="qwen35_baton_prediction",
        teacher=None,
        planner=planner,
    )
    video = _video()

    condition = build_baton_semantic_condition(
        components,
        _semantic_config("qwen35_baton_prediction")["semantic_plan"],
        video,
        ("pick", "place"),
        n_previous=4,
        num_future_frames=9,
        num_latent_frames=6,
        device="cpu",
        dtype=torch.bfloat16,
    )

    assert len(planner.calls) == 1
    images, instructions, grad_enabled = planner.calls[0]
    expected = (
        video[:, :, :, 3]
        .permute(0, 2, 1, 3, 4)
        .add(1)
        .mul(127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
    )
    assert torch.equal(images, expected)
    assert instructions == ("pick", "place")
    assert grad_enabled is False
    assert condition.tokens.shape == (2, 2, 4, 256, 1024)
    assert condition.tokens.dtype == torch.bfloat16
    assert condition.positions.dtype == torch.float32
    assert condition.mask is None
    assert condition.relevance is None
    assert not condition.tokens.requires_grad


def test_stage3_batch_rejects_wrong_provider_indices_or_grounding_metadata() -> None:
    class MalformedPlanner(_RecordingPlanner):
        def predict(self, current_images, instructions):
            plan = super().predict(current_images, instructions)
            return SimpleNamespace(
                tokens=plan.tokens,
                future_indices=(0, 2, 5, 8),
                positions_xy=plan.positions_xy,
                relevance=torch.ones(1),
            )

    components = BatonConditioningComponents(
        source="qwen35_baton_prediction",
        teacher=None,
        planner=MalformedPlanner(),
    )

    with pytest.raises(ValueError, match="future_indices|relevance"):
        build_baton_semantic_condition(
            components,
            _semantic_config("qwen35_baton_prediction")["semantic_plan"],
            _video(batch_size=1),
            ("pick",),
            n_previous=4,
            num_future_frames=9,
            num_latent_frames=6,
            device="cpu",
            dtype=torch.float32,
        )


class _FullGeActModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video_proj = nn.Linear(2, 2)
        self.action_proj = nn.Linear(2, 2)
        self.semantic_adapter = nn.Linear(2, 2)
        self.semantic_adapter_alias = self.semantic_adapter


class _RecordingGeActForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        output = kwargs["hidden_states"].clone()
        semantic_plan = kwargs.get("semantic_plan")
        semantic_gate = kwargs.get("semantic_condition_mask")
        if semantic_plan is not None and semantic_gate is not None:
            per_view = semantic_plan.float().mean(
                dim=(1, 2, 3, 4),
            ).repeat_interleave(2)
            output = output + per_view[:, None, None] * semantic_gate[:, None, None]
        return ({"video": output, "action": torch.zeros(1, 1, 1)},)


def _minimal_diffusion_kwargs() -> dict[str, Any]:
    return {
        "timesteps": torch.ones(2),
        "noisy_latents": torch.zeros(2, 1, 1),
        "prompt_embeds": torch.zeros(1, 1, 1),
        "prompt_attention_mask": torch.ones(1, 1),
        "num_frames": 6,
        "height": 1,
        "width": 1,
        "n_view": 2,
    }


@pytest.mark.parametrize(
    "source",
    ["qwen35_baton_teacher", "qwen35_baton_prediction"],
)
def test_trainer_model_forward_passes_full_baton_grid_positions_and_no_relevance(
    source: str,
) -> None:
    if source == "qwen35_baton_teacher":
        components = BatonConditioningComponents(
            source=source,
            teacher=_RecordingTeacher(),
            planner=None,
        )
    else:
        components = BatonConditioningComponents(
            source=source,
            teacher=None,
            planner=_RecordingPlanner(),
        )
    condition = build_baton_semantic_condition(
        components,
        _semantic_config(source)["semantic_plan"],
        _video(batch_size=1),
        ("pick",),
        n_previous=4,
        num_future_frames=9,
        num_latent_frames=6,
        device="cpu",
        dtype=torch.float32,
    )
    model = _RecordingGeActForward()

    forward_baton_ge_act(
        model,
        condition,
        semantic_condition_mask=torch.ones(2),
        diffusion_kwargs=_minimal_diffusion_kwargs(),
    )

    call = model.calls[0]
    assert call["semantic_plan"].shape == (1, 2, 4, 256, 1024)
    assert call["semantic_plan_positions"].shape == (1, 2, 4, 256, 2)
    assert call["semantic_plan_mask"] is None
    assert call["semantic_plan_relevance"] is None


def test_semantic_disabled_baton_forward_equals_omitted_condition_exactly() -> None:
    components = BatonConditioningComponents(
        source="qwen35_baton_prediction",
        teacher=None,
        planner=_RecordingPlanner(),
    )
    condition = build_baton_semantic_condition(
        components,
        _semantic_config("qwen35_baton_prediction")["semantic_plan"],
        _video(batch_size=1),
        ("pick",),
        n_previous=4,
        num_future_frames=9,
        num_latent_frames=6,
        device="cpu",
        dtype=torch.float32,
    )
    model = _RecordingGeActForward()
    kwargs = _minimal_diffusion_kwargs()

    disabled = forward_baton_ge_act(
        model,
        condition,
        semantic_condition_mask=torch.zeros(2),
        diffusion_kwargs=kwargs,
    )["latents"]["video"]
    omitted = ge_trainer_module.forward_pass(
        model=model,
        **kwargs,
    )["latents"]["video"]

    torch.testing.assert_close(disabled, omitted, rtol=0, atol=0)


@pytest.mark.parametrize(
    "source",
    ["qwen35_baton_teacher", "qwen35_baton_prediction"],
)
def test_baton_optimizer_has_three_id_exact_nonempty_disjoint_exhaustive_groups(
    source: str,
) -> None:
    model = _FullGeActModel()

    groups = build_optimizer_parameter_groups(
        model,
        train_mode="all",
        base_lr=2e-5,
        action_lr=1e-4,
        semantic_lr=5e-5,
        baton_source=source,
    )

    assert {group["name"]: group["lr"] for group in groups} == {
        "ltx_video": 2e-5,
        "action_expert": 1e-4,
        "semantic_adapter": 5e-5,
    }
    grouped_ids = [
        id(parameter)
        for group in groups
        for parameter in group["params"]
    ]
    trainable_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    assert grouped_ids
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == trainable_ids
    assert all(group["params"] for group in groups)


def test_frozen_baton_components_never_enter_optimizer_groups() -> None:
    model = _FullGeActModel()
    teacher = _RecordingTeacher()
    planner = _RecordingPlanner()
    groups = build_optimizer_parameter_groups(
        model,
        train_mode="all",
        base_lr=2e-5,
        action_lr=1e-4,
        semantic_lr=5e-5,
        baton_source="qwen35_baton_prediction",
    )

    grouped = {id(parameter) for group in groups for parameter in group["params"]}
    frozen = {
        id(parameter)
        for module in (teacher.model, planner)
        for parameter in module.parameters()
    }
    assert grouped.isdisjoint(frozen)


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("path", "source", "steps", "mode"),
    [
        (
            STAGE2_CONFIG,
            "qwen35_baton_teacher",
            20_000,
            "teacher",
        ),
        (
            STAGE3_CONFIG,
            "qwen35_baton_prediction",
            30_000,
            "prediction",
        ),
    ],
)
def test_baton_recipe_matches_approved_training_contract(
    path: Path,
    source: str,
    steps: int,
    mode: str,
) -> None:
    config = _load_config(path)

    assert config["train_data_class"] == "LiberoFastWAMHDF5Dataset"
    assert config["val_data_class"] == "LiberoFastWAMHDF5Dataset"
    assert config["return_video"] is True
    assert config["return_action"] is True
    assert config["train_mode"] == "all"
    assert config["train_steps"] == steps
    assert config["steps_to_save"] == 5_000
    assert config["lr"] == 2e-5
    assert config["action_lr"] == 1e-4
    assert config["semantic_lr"] == 5e-5
    assert config["batch_size"] * config["gradient_accumulation_steps"] * 8 == 128
    semantic = config["semantic_plan"]
    assert semantic["source"] == source
    assert semantic["keyframe_indices"] == [0, 3, 5, 8]
    assert semantic["tokens_per_frame"] == 256
    assert semantic["feature_dim"] == 1024
    assert semantic["dropout"] == 0.15
    assert semantic["validation_mode"] == mode
    assert semantic["validation_modes"] == [mode, "semantic_disabled"]
    model = config["diffusion_model"]["config"]
    assert model["action_expert"] is True
    assert model["semantic_plan_context"] is True
    assert model["semantic_plan_in_dim"] == 1024
    assert model["semantic_plan_num_keyframes"] == 4
    assert model["semantic_plan_num_views"] == 2
    assert model["semantic_plan_cross_attention_blocks"] == list(range(28))
    assert config["data"]["train"]["manifest_path"] == config["data"]["val"]["manifest_path"]
    for split in ("train", "val"):
        assert config["data"][split]["baton_sampling_algorithm"] == (
            BATON_SAMPLING_ALGORITHM
        )
        assert config["data"][split]["baton_sampling_version"] == (
            BATON_SAMPLING_VERSION
        )
        assert config["data"][split]["baton_sampling_seed"] == config["seed"]


@pytest.mark.parametrize("path", [STAGE2_CONFIG, STAGE3_CONFIG])
def test_baton_recipe_passes_semantic_static_preflight(path: Path) -> None:
    template = _load_config(path)
    source = template["semantic_plan"]["source"]
    config = materialize_baton_config(
        template,
        _materialization_environment(source),
    )

    assert collect_preflight_errors(
        config,
        world_size=8,
        check_paths=False,
    ) == []


@pytest.mark.parametrize("path", [STAGE2_CONFIG, STAGE3_CONFIG])
def test_baton_recipe_passes_hdf5_static_preflight(path: Path) -> None:
    config = _load_config(path)

    assert collect_hdf5_preflight_errors(
        config,
        world_size=8,
        check_paths=False,
    ) == []


@pytest.mark.parametrize(
    ("path", "mutation", "message"),
    [
        (
            STAGE2_CONFIG,
            lambda config: config.update({"train_steps": 30_000}),
            "train_steps",
        ),
        (
            STAGE3_CONFIG,
            lambda config: config.update({"return_action": False}),
            "return_action",
        ),
        (
            STAGE3_CONFIG,
            lambda config: config["data"]["train"].update(
                {"valid_cam": ["observation.images.image"]}
            ),
            "valid_cam",
        ),
        (
            STAGE2_CONFIG,
            lambda config: config["data"]["train"].update(
                {"baton_sampling_seed": 7}
            ),
            "baton_sampling_seed",
        ),
    ],
)
def test_baton_hdf5_preflight_rejects_schedule_objective_or_camera_drift(
    path: Path,
    mutation,
    message: str,
) -> None:
    config = _load_config(path)
    mutation(config)

    errors = collect_hdf5_preflight_errors(
        config,
        world_size=8,
        check_paths=False,
    )

    assert any(message in error for error in errors), errors


def test_stage2_semantic_preflight_rejects_zero_provenance_placeholders() -> None:
    config = _load_config(STAGE2_CONFIG)

    errors = collect_preflight_errors(
        config,
        world_size=8,
        check_paths=False,
    )

    assert any("placeholder" in error for error in errors), errors


@pytest.mark.parametrize("launcher", [STAGE2_LAUNCHER, STAGE3_LAUNCHER])
def test_launcher_derives_accumulation_and_runs_both_preflights_before_distributed(
    launcher: Path,
) -> None:
    source = launcher.read_text(encoding="utf-8")

    assert "GLOBAL_BATCH" in source
    assert "PER_DEVICE_BATCH" in source
    assert "GRADIENT_ACCUMULATION_STEPS" in source
    assert "GLOBAL_BATCH %" in source
    materialize = source.index("--materialize-output")
    hdf5 = source.index("preflight_libero_fastwam_hdf5.py")
    semantic = source.rindex("preflight_ltx_siglip2.py")
    distributed = source.index("-m torch.distributed.run")
    assert materialize < hdf5 < semantic < distributed


@pytest.mark.parametrize(
    ("wrapper", "delegate"),
    [
        (STAGE2_SBATCH, "train_ltx_baton_stage2.sh"),
        (STAGE3_SBATCH, "train_ltx_baton_stage3.sh"),
    ],
)
def test_hpc3_wrapper_only_delegates_training_arguments(
    wrapper: Path,
    delegate: str,
) -> None:
    source = wrapper.read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:8" in source
    assert delegate in source
    assert "torch.distributed.run" not in source
    assert "preflight_ltx_siglip2.py" not in source


def test_semantic_disabled_validation_keeps_loaded_condition_and_zeros_only_mask() -> None:
    components = BatonConditioningComponents(
        source="qwen35_baton_prediction",
        teacher=None,
        planner=_RecordingPlanner(),
    )

    tokens = torch.randn(1, 2, 4, 256, 1024)
    actual = ge_trainer_module.apply_baton_validation_mode(
        components,
        tokens=tokens,
        mode="semantic_disabled",
        batch_size=1,
        n_view=2,
        device="cpu",
        dtype=torch.float32,
    )

    assert actual.tokens is tokens
    assert torch.equal(actual.condition_mask, torch.zeros(2))
    assert actual.source == "qwen35_baton_prediction"
    assert actual.mode == "semantic_disabled"
    assert components.planner is not None


@pytest.mark.parametrize(
    ("source", "mode", "metric", "expected"),
    [
        (
            "qwen35_baton_teacher",
            "teacher",
            "video",
            "validation/qwen35_baton_teacher/teacher/video",
        ),
        (
            "qwen35_baton_prediction",
            "prediction",
            "action",
            "validation/qwen35_baton_prediction/prediction/action",
        ),
    ],
)
def test_validation_metric_names_expose_source_mode_and_objective(
    source: str,
    mode: str,
    metric: str,
    expected: str,
) -> None:
    assert ge_trainer_module.baton_validation_metric_name(
        source,
        mode,
        metric,
    ) == expected


def test_generic_inferencer_accepts_predicted_baton_source() -> None:
    config = _semantic_config("qwen35_baton_prediction")

    assert (
        ge_inferencer_module.validate_baton_inference_source(config)
        == "qwen35_baton_prediction"
    )


def test_deployment_inference_rejects_teacher_only_baton_source() -> None:
    config = _semantic_config("qwen35_baton_teacher")

    with pytest.raises(ValueError, match="qwen35_baton_prediction"):
        ge_inferencer_module.validate_baton_inference_source(config)


def test_libero_and_libero_plus_keep_official_accounting_while_using_baton_prediction() -> None:
    libero_source = (
        GE_ACT_ROOT / "experiments/eval_libero.py"
    ).read_text(encoding="utf-8")
    plus_source = (
        GE_ACT_ROOT / "experiments/eval_libero_plus.py"
    ).read_text(encoding="utf-8")

    assert "validate_baton_inference_source" in libero_source
    assert "semantic_plan_positions" in libero_source
    assert "from experiments.eval_libero import InferenceLibero" in plus_source
    assert 'grand["overall"]' in plus_source


def _materialization_environment(source: str) -> dict[str, str]:
    common = {
        "BATON_LTX_PRETRAINED_PATH": "/resolved/ltx",
        "BATON_HDF5_MANIFEST_PATH": "/resolved/data/manifest.json",
        "BATON_STAT_FILE": "/resolved/data/stat.json",
        "BATON_OUTPUT_DIR": "/resolved/output",
    }
    if source == "qwen35_baton_teacher":
        return {
            **common,
            "BATON_GE_BASE_CHECKPOINT": "/resolved/ge_base.safetensors",
            "BATON_SIGLIP2_MODEL_PATH": "/resolved/siglip2",
            "BATON_SIGLIP2_CONFIG_HASH": "1" * 64,
            "BATON_SIGLIP2_ARTIFACT_HASH": "2" * 64,
            "BATON_TEACHER_PREPROCESSING_HASH": "2" * 64,
        }
    return {
        **common,
        "BATON_STAGE2_INIT_CHECKPOINT": "/resolved/stage2/step_020000",
        "BATON_STAGE2_INIT_TOPOLOGY_HASH": "3" * 64,
        "BATON_PLANNER_CHECKPOINT": "/resolved/stage1/step_030000",
        "BATON_PLANNER_TOPOLOGY": "/resolved/stage1/planner_topology.json",
        "BATON_QWEN_MODEL_PATH": "/resolved/qwen",
        "BATON_QWEN_TOKENIZER_PATH": "/resolved/qwen",
        "BATON_QWEN_PROCESSOR_PATH": "/resolved/qwen",
        "BATON_SIGLIP2_MODEL_PATH": "/resolved/siglip2",
    }


@pytest.mark.parametrize(
    ("path", "source"),
    [
        (STAGE2_CONFIG, "qwen35_baton_teacher"),
        (STAGE3_CONFIG, "qwen35_baton_prediction"),
    ],
)
def test_materializer_requires_explicit_artifacts_and_returns_private_resolved_config(
    path: Path,
    source: str,
) -> None:
    template = _load_config(path)
    original = deepcopy(template)

    with pytest.raises(ValueError, match="BATON_"):
        materialize_baton_config(template, {})

    resolved = materialize_baton_config(
        template,
        _materialization_environment(source),
    )

    assert template == original
    assert resolved["pretrained_model_name_or_path"] == "/resolved/ltx"
    assert resolved["data"]["train"]["manifest_path"] == (
        "/resolved/data/manifest.json"
    )
    assert resolved["data"]["val"]["manifest_path"] == (
        "/resolved/data/manifest.json"
    )
    if source == "qwen35_baton_teacher":
        assert resolved["diffusion_model"]["model_path"] == (
            "/resolved/ge_base.safetensors"
        )
        assert resolved["semantic_plan"]["siglip2_config_hash"] == "1" * 64
        assert resolved["semantic_plan"]["siglip2_artifact_hash"] == "2" * 64
    else:
        assert resolved["stage2_init_checkpoint"] == (
            "/resolved/stage2/step_020000"
        )
        assert resolved["stage2_init_topology_hash"] == "3" * 64
        assert resolved["diffusion_model"]["model_path"] == (
            "/resolved/stage2/step_020000/diffusion_model"
        )
        assert resolved["semantic_plan"]["planner_checkpoint"] == (
            "/resolved/stage1/step_030000"
        )


@pytest.mark.parametrize(
    "launcher",
    [STAGE2_LAUNCHER, STAGE3_LAUNCHER],
)
def test_launcher_missing_materialization_inputs_fails_before_torchrun(
    launcher: Path,
    tmp_path: Path,
) -> None:
    environment = {
        "PATH": os.environ["PATH"],
        "PYTHON_BIN": str(
            Path("/data/LFT-W02_data/.conda/envs/ge-act/bin/python")
        ),
        "TMPDIR": str(tmp_path),
        "NUM_PROCESSES": "8",
        "NNODES": "1",
    }

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "required deployment variable is missing" in completed.stdout
    assert "torch.distributed.run" not in completed.stdout


def test_stage3_launcher_propagates_concrete_stage1_and_stage2_paths_to_torchrun(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "calls.jsonl"
    capture_path = tmp_path / "resolved.yaml"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        """#!/data/LFT-W02_data/.conda/envs/ge-act/bin/python
import json
import os
from pathlib import Path
import sys
import yaml
from ge_act.scripts.preflight_ltx_siglip2 import materialize_baton_config

args = sys.argv[1:]
with Path(os.environ["CALL_LOG"]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(args) + "\\n")
if "--materialize-output" in args:
    template = yaml.safe_load(
        Path(args[args.index("--config") + 1]).read_text(encoding="utf-8")
    )
    resolved = materialize_baton_config(template, dict(os.environ))
    Path(args[args.index("--materialize-output") + 1]).write_text(
        yaml.safe_dump(resolved, sort_keys=False),
        encoding="utf-8",
    )
elif args[:2] == ["-m", "torch.distributed.run"]:
    config_path = Path(args[args.index("--config_file") + 1])
    Path(os.environ["CAPTURE_CONFIG"]).write_text(
        config_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = {
        **os.environ,
        **_materialization_environment("qwen35_baton_prediction"),
        "PYTHON_BIN": str(fake_python),
        "TMPDIR": str(tmp_path),
        "NUM_PROCESSES": "8",
        "NNODES": "1",
        "CALL_LOG": str(log_path),
        "CAPTURE_CONFIG": str(capture_path),
    }

    completed = subprocess.run(
        ["bash", str(STAGE3_LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    calls = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert "--materialize-output" in calls[0]
    assert calls[1][0].endswith("preflight_libero_fastwam_hdf5.py")
    assert calls[2][0].endswith("preflight_ltx_siglip2.py")
    assert calls[3][:2] == ["-m", "torch.distributed.run"]
    resolved = yaml.safe_load(capture_path.read_text(encoding="utf-8"))
    assert resolved["stage2_init_checkpoint"] == (
        "/resolved/stage2/step_020000"
    )
    assert resolved["diffusion_model"]["model_path"] == (
        "/resolved/stage2/step_020000/diffusion_model"
    )
    assert resolved["semantic_plan"]["planner_checkpoint"] == (
        "/resolved/stage1/step_030000"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _teacher_training_provenance(
    **updates: Any,
) -> dict[str, str | int]:
    provenance: dict[str, str | int] = {
        "hdf5_manifest_hash": "4" * 64,
        "siglip2_config_hash": "1" * 64,
        "siglip2_artifact_hash": "2" * 64,
        "teacher_preprocessing_hash": "2" * 64,
        "window_sampling_algorithm": BATON_SAMPLING_ALGORITHM,
        "window_sampling_version": BATON_SAMPLING_VERSION,
        "window_sampling_seed": 42,
        "window_sampling_topology_hash": (
            "e98a7afcd00fda75192014b1104b89016e4d78ca5eee27b6d60a9e65626feb5a"
        ),
    }
    provenance.update(updates)
    return provenance


def _prediction_training_provenance(
    **updates: Any,
) -> dict[str, str | int]:
    provenance = {
        **_teacher_training_provenance(),
        "planner_manifest_hash": "5" * 64,
        "planner_topology_hash": "6" * 64,
        "qwen_config_hash": "7" * 64,
        "tokenizer_hash": "8" * 64,
        "processor_hash": "9" * 64,
        "input_template_hash": "a" * 64,
    }
    provenance.update(updates)
    return provenance


class _Stage1TinyPlanner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))


def _write_stage1_checkpoint_envelope(
    root: Path,
    *,
    hdf5_manifest_hash: str = "4" * 64,
    siglip2_config_hash: str = "1" * 64,
    siglip2_artifact_hash: str = "2" * 64,
    teacher_preprocessing_hash: str = "2" * 64,
) -> tuple[Path, Path, BatonCheckpointMetadata]:
    checkpoint = root / "stage1"
    planner = _Stage1TinyPlanner()
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "planner",
                "params": list(planner.parameters()),
                "lr": 5e-5,
            }
        ]
    )
    scheduler = BatonCosineWarmupScheduler(
        optimizer,
        warmup_steps=0,
        max_steps=10,
    )
    metadata = replace(
        BatonCheckpointMetadata.example(),
        hdf5_manifest_hash=hdf5_manifest_hash,
        siglip2_config_hash=siglip2_config_hash,
        siglip2_artifact_hash=siglip2_artifact_hash,
        teacher_preprocessing_hash=teacher_preprocessing_hash,
        planner_topology_hash=sha256_json(
            planner_module_topology(planner)
        ),
    )
    save_baton_checkpoint(
        checkpoint,
        planner=planner,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
        cursor=BatonTrainingCursor(
            global_step=0,
            epoch=0,
            consumed_microbatches=0,
            microbatches_per_epoch=1,
            sampler_seed=0,
        ),
        rank_rng_state={0: capture_rank_rng_state(distributed_rank=0)},
    )
    trusted_topology = root / "stage1-trusted-topology.json"
    publish_trusted_planner_topology(
        trusted_topology,
        planner_safetensors_topology(
            checkpoint / "planner.safetensors"
        ),
    )
    persisted_metadata = BatonCheckpointMetadata.from_dict(
        json.loads(
            (checkpoint / "metadata.json").read_text(encoding="utf-8")
        )
    )
    return checkpoint, trusted_topology, persisted_metadata


def _write_stage2_checkpoint_envelope(
    root: Path,
    *,
    source: str = "qwen35_baton_teacher",
    topology_hash: str = "3" * 64,
    global_step: int = 20_000,
    provenance_updates: dict[str, Any] | None = None,
) -> Path:
    checkpoint = root / "step_020000"
    diffusion = checkpoint / "diffusion_model"
    diffusion.mkdir(parents=True)
    model_path = diffusion / "diffusion_pytorch_model.safetensors"
    save_file({"video.weight": torch.ones(1)}, str(model_path))
    accelerator_path = checkpoint / "accelerator_state.pt"
    accelerator_path.write_bytes(b"complete-accelerator-state")
    snapshot_topology_hash = hashlib.sha256(
        json.dumps(
            [
                {
                    "name": "video.weight",
                    "shape": [1],
                    "dtype": "torch.float32",
                }
            ],
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    metadata = {
        "format_version": 1,
        "checkpoint_kind": "ge_act_baton",
        "model_children": ["diffusion_model"],
        "source": source,
        "topology_hash": topology_hash,
        "snapshot_topology_hash": snapshot_topology_hash,
        "cursor": {
            "global_step": global_step,
            "epoch": 12,
            "consumed_microbatches": 0,
            "microbatches_per_epoch": 16,
            "sampler_seed": 42,
        },
        "accelerator_files": {
            "accelerator_state.pt": _sha256(accelerator_path),
        },
        "diffusion_subdir": "diffusion_model",
        "diffusion_files": {
            "diffusion_pytorch_model.safetensors": _sha256(model_path),
        },
        "training_provenance": _teacher_training_provenance(
            **(provenance_updates or {})
        ),
    }
    (checkpoint / "baton_state.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return checkpoint


def test_stage3_init_accepts_only_complete_final_teacher_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = _write_stage2_checkpoint_envelope(tmp_path)
    metadata = json.loads(
        (checkpoint / "baton_state.json").read_text(encoding="utf-8")
    )

    model_dir = validate_baton_stage2_checkpoint_envelope(
        checkpoint,
        expected_topology_hash=metadata["snapshot_topology_hash"],
        expected_hdf5_manifest_hash="4" * 64,
    )

    assert model_dir == checkpoint / "diffusion_model"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda metadata: metadata.update(
                {"source": "qwen35_baton_prediction"}
            ),
            "provenance|teacher",
        ),
        (
            lambda metadata: metadata.update(
                {"snapshot_topology_hash": "5" * 64}
            ),
            "topology",
        ),
        (
            lambda metadata: metadata["cursor"].update({"global_step": 19_999}),
            "20000",
        ),
        (
            lambda metadata: metadata["diffusion_files"].update(
                {"diffusion_pytorch_model.safetensors": "6" * 64}
            ),
            "artifact",
        ),
    ],
)
def test_stage3_init_rejects_wrong_source_topology_step_or_artifact(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    checkpoint = _write_stage2_checkpoint_envelope(tmp_path)
    metadata_path = checkpoint / "baton_state.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_topology_hash = metadata["snapshot_topology_hash"]
    mutation(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_baton_stage2_checkpoint_envelope(
            checkpoint,
            expected_topology_hash=expected_topology_hash,
            expected_hdf5_manifest_hash="4" * 64,
        )


def test_stage3_artifact_chain_accepts_one_validated_semantic_target_space(
    tmp_path: Path,
) -> None:
    stage1, trusted_topology, stage1_metadata = (
        _write_stage1_checkpoint_envelope(tmp_path / "planner")
    )
    stage2 = _write_stage2_checkpoint_envelope(tmp_path / "teacher")
    stage2_metadata = json.loads(
        (stage2 / "baton_state.json").read_text(encoding="utf-8")
    )

    chain = ge_trainer_module.validate_baton_stage3_artifact_chain(
        stage2_checkpoint=stage2,
        expected_stage2_topology_hash=stage2_metadata[
            "snapshot_topology_hash"
        ],
        planner_checkpoint=stage1,
        expected_planner_topology=trusted_topology,
        expected_hdf5_manifest_hash="4" * 64,
    )

    assert chain.diffusion_model_dir == stage2 / "diffusion_model"
    assert chain.stage1_metadata == stage1_metadata
    assert chain.stage2_training_provenance == (
        _teacher_training_provenance()
    )
    assert chain.diffusion_files == {
        "diffusion_pytorch_model.safetensors": _sha256(
            stage2
            / "diffusion_model"
            / "diffusion_pytorch_model.safetensors"
        )
    }


@pytest.mark.parametrize(
    "field",
    (
        "siglip2_config_hash",
        "siglip2_artifact_hash",
        "teacher_preprocessing_hash",
        "hdf5_manifest_hash",
    ),
)
def test_stage3_artifact_chain_rejects_stage1_stage2_target_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    stage1, trusted_topology, _ = _write_stage1_checkpoint_envelope(
        tmp_path / "planner"
    )
    stage2 = _write_stage2_checkpoint_envelope(
        tmp_path / "teacher",
        provenance_updates={field: "b" * 64},
    )
    stage2_metadata = json.loads(
        (stage2 / "baton_state.json").read_text(encoding="utf-8")
    )

    with pytest.raises(ValueError, match=f"{field}|dataset provenance"):
        ge_trainer_module.validate_baton_stage3_artifact_chain(
            stage2_checkpoint=stage2,
            expected_stage2_topology_hash=stage2_metadata[
                "snapshot_topology_hash"
            ],
            planner_checkpoint=stage1,
            expected_planner_topology=trusted_topology,
            expected_hdf5_manifest_hash="4" * 64,
        )


class _Stage2EnvelopeDiffusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video = nn.Module()
        self.video.register_parameter(
            "weight",
            nn.Parameter(torch.zeros(1)),
        )


def test_stage3_strict_loader_rejects_same_topology_byte_replacement(
    tmp_path: Path,
) -> None:
    stage1, trusted_topology, _ = _write_stage1_checkpoint_envelope(
        tmp_path / "planner"
    )
    stage2 = _write_stage2_checkpoint_envelope(tmp_path / "teacher")
    stage2_metadata = json.loads(
        (stage2 / "baton_state.json").read_text(encoding="utf-8")
    )
    chain = ge_trainer_module.validate_baton_stage3_artifact_chain(
        stage2_checkpoint=stage2,
        expected_stage2_topology_hash=stage2_metadata[
            "snapshot_topology_hash"
        ],
        planner_checkpoint=stage1,
        expected_planner_topology=trusted_topology,
        expected_hdf5_manifest_hash="4" * 64,
    )
    save_file(
        {"video.weight": torch.full((1,), 9.0)},
        str(
            chain.diffusion_model_dir
            / "diffusion_pytorch_model.safetensors"
        ),
    )
    runtime = _Stage2EnvelopeDiffusion()
    before = runtime.video.weight.detach().clone()

    with pytest.raises(ValueError, match="artifact|hash|manifest"):
        ge_trainer_module.strict_load_baton_stage3_diffusion_model(
            runtime,
            chain.diffusion_model_dir,
            expected_snapshot_topology_hash=stage2_metadata[
                "snapshot_topology_hash"
            ],
            expected_diffusion_files=chain.diffusion_files,
        )

    torch.testing.assert_close(runtime.video.weight, before, rtol=0, atol=0)


class _TinyDiffusion(nn.Linear):
    def save_pretrained(
        self,
        output_dir: str | Path,
        *,
        safe_serialization: bool,
    ) -> None:
        assert safe_serialization is True
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        save_file(
            {name: value.detach().cpu() for name, value in self.state_dict().items()},
            str(output / "diffusion_pytorch_model.safetensors"),
        )


def test_stage3_strict_loader_requires_exact_runtime_topology_before_loading(
    tmp_path: Path,
) -> None:
    source = _TinyDiffusion(1, 1, bias=False)
    snapshot = tmp_path / "snapshot"
    source.save_pretrained(snapshot, safe_serialization=True)
    expected_topology = (
        ge_trainer_module._diffusion_snapshot_topology_hash(snapshot)
    )
    runtime = _TinyDiffusion(2, 1, bias=False)
    before = runtime.weight.detach().clone()

    with pytest.raises(ValueError, match="runtime.*topology"):
        ge_trainer_module.strict_load_baton_stage3_diffusion_model(
            runtime,
            snapshot,
            expected_snapshot_topology_hash=expected_topology,
            expected_diffusion_files={
                "diffusion_pytorch_model.safetensors": _sha256(
                    snapshot / "diffusion_pytorch_model.safetensors"
                ),
            },
        )

    torch.testing.assert_close(runtime.weight, before, rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_stage3_strict_loader_rejects_incomplete_tensor_set_without_partial_init(
    tmp_path: Path,
    mutation: str,
) -> None:
    runtime = _TinyDiffusion(1, 1, bias=False)
    runtime.weight.data.zero_()
    before = runtime.weight.detach().clone()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    tensors = (
        {"unexpected": torch.ones(1)}
        if mutation == "missing"
        else {
            "weight": torch.full((1, 1), 3.0),
            "unexpected": torch.ones(1),
        }
    )
    save_file(
        tensors,
        str(snapshot / "diffusion_pytorch_model.safetensors"),
    )

    with pytest.raises(ValueError, match="snapshot.*topology|strict"):
        ge_trainer_module.strict_load_baton_stage3_diffusion_model(
            runtime,
            snapshot,
            expected_snapshot_topology_hash=(
                ge_trainer_module._module_topology_hash(runtime)
            ),
            expected_diffusion_files={
                "diffusion_pytorch_model.safetensors": _sha256(
                    snapshot / "diffusion_pytorch_model.safetensors"
                ),
            },
        )

    torch.testing.assert_close(runtime.weight, before, rtol=0, atol=0)


def test_stage3_strict_loader_loads_complete_exact_snapshot(
    tmp_path: Path,
) -> None:
    source = _TinyDiffusion(1, 1, bias=False)
    source.weight.data.fill_(3.0)
    snapshot = tmp_path / "snapshot"
    source.save_pretrained(snapshot, safe_serialization=True)
    runtime = _TinyDiffusion(1, 1, bias=False)
    runtime.weight.data.zero_()

    loaded = ge_trainer_module.strict_load_baton_stage3_diffusion_model(
        runtime,
        snapshot,
        expected_snapshot_topology_hash=(
            ge_trainer_module._diffusion_snapshot_topology_hash(snapshot)
        ),
        expected_diffusion_files={
            "diffusion_pytorch_model.safetensors": _sha256(
                snapshot / "diffusion_pytorch_model.safetensors"
            ),
        },
    )

    assert loaded is runtime
    torch.testing.assert_close(
        runtime.weight,
        torch.full((1, 1), 3.0),
        rtol=0,
        atol=0,
    )


class _StateAccelerator:
    is_main_process = True

    def __init__(self, model, optimizer, scheduler) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loaded: Path | None = None

    def wait_for_everyone(self) -> None:
        return None

    def prepare(self, *components):
        self.prepared_components = components
        return components

    def unwrap_model(self, model):
        return model

    def save_state(self, output_dir: str) -> None:
        output = Path(output_dir)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
            },
            output / "accelerator_state.pt",
        )

    def load_state(self, checkpoint_dir: str) -> None:
        self.loaded = Path(checkpoint_dir)
        payload = torch.load(
            self.loaded / "accelerator_state.pt",
            weights_only=False,
        )
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"])


def _tiny_training_state():
    model = _TinyDiffusion(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.8,
    )
    return model, optimizer, scheduler


def _tiny_step(model, optimizer, scheduler) -> None:
    scalar = random.random() + float(np.random.rand()) + torch.rand(()).item()
    loss = (model(torch.tensor([[scalar]])) - 0.25).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()


def test_baton_resume_matches_uninterrupted_optimizer_scheduler_and_rng(
    tmp_path: Path,
) -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    baseline_model, baseline_optimizer, baseline_scheduler = (
        _tiny_training_state()
    )
    for _ in range(4):
        _tiny_step(baseline_model, baseline_optimizer, baseline_scheduler)
    baseline_weight = baseline_model.weight.detach().clone()
    baseline_lr = baseline_scheduler.get_last_lr()
    baseline_next_rng = (
        random.random(),
        float(np.random.rand()),
        torch.rand(()),
    )

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    first_model, first_optimizer, first_scheduler = _tiny_training_state()
    for _ in range(2):
        _tiny_step(first_model, first_optimizer, first_scheduler)
    first_accelerator = _StateAccelerator(
        first_model,
        first_optimizer,
        first_scheduler,
    )
    cursor = TrainingCursor(
        global_step=2,
        epoch=0,
        consumed_microbatches=2,
        microbatches_per_epoch=4,
        sampler_seed=42,
    )
    checkpoint = save_baton_training_checkpoint(
        first_accelerator,
        tmp_path / "run",
        cursor=cursor,
        diffusion_model=first_model,
        source="qwen35_baton_teacher",
        training_provenance=_teacher_training_provenance(),
    )

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    resumed_model, resumed_optimizer, resumed_scheduler = (
        _tiny_training_state()
    )
    resumed_accelerator = _StateAccelerator(
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
    )
    restored = load_baton_training_checkpoint(
        resumed_accelerator,
        checkpoint,
        diffusion_model=resumed_model,
        expected_source="qwen35_baton_teacher",
        expected_microbatches_per_epoch=4,
        expected_sampler_seed=42,
        expected_training_provenance=_teacher_training_provenance(),
    )
    for _ in range(2):
        _tiny_step(resumed_model, resumed_optimizer, resumed_scheduler)
    resumed_next_rng = (
        random.random(),
        float(np.random.rand()),
        torch.rand(()),
    )

    assert restored == cursor
    torch.testing.assert_close(resumed_model.weight, baseline_weight, rtol=0, atol=0)
    assert resumed_scheduler.get_last_lr() == baseline_lr
    assert resumed_next_rng[:2] == baseline_next_rng[:2]
    torch.testing.assert_close(
        resumed_next_rng[2],
        baseline_next_rng[2],
        rtol=0,
        atol=0,
    )
    metadata = json.loads(
        (checkpoint / "baton_state.json").read_text(encoding="utf-8")
    )
    assert metadata["model_children"] == ["diffusion_model"]


@pytest.mark.parametrize(
    "mutation",
    ("mismatch", "missing", "extra", "alternate_digest"),
)
def test_baton_resume_rejects_any_runtime_provenance_difference_before_load(
    tmp_path: Path,
    mutation: str,
) -> None:
    model, optimizer, scheduler = _tiny_training_state()
    accelerator = _StateAccelerator(model, optimizer, scheduler)
    stored_provenance = _teacher_training_provenance()
    checkpoint = save_baton_training_checkpoint(
        accelerator,
        tmp_path / "run",
        cursor=TrainingCursor(
            global_step=2,
            epoch=0,
            consumed_microbatches=2,
            microbatches_per_epoch=4,
            sampler_seed=42,
        ),
        diffusion_model=model,
        source="qwen35_baton_teacher",
        training_provenance=stored_provenance,
    )
    expected_provenance = dict(stored_provenance)
    if mutation == "mismatch":
        expected_provenance["teacher_preprocessing_hash"] = "3" * 64
    elif mutation == "missing":
        del expected_provenance["teacher_preprocessing_hash"]
    elif mutation == "extra":
        expected_provenance["undeclared_hash"] = "8" * 64
    else:
        expected_provenance["hdf5_manifest_hash"] = "9" * 64
    resumed_model, resumed_optimizer, resumed_scheduler = (
        _tiny_training_state()
    )
    resumed_accelerator = _StateAccelerator(
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
    )

    with pytest.raises(ValueError, match="training provenance"):
        load_baton_training_checkpoint(
            resumed_accelerator,
            checkpoint,
            diffusion_model=resumed_model,
            expected_source="qwen35_baton_teacher",
            expected_microbatches_per_epoch=4,
            expected_sampler_seed=42,
            expected_training_provenance=expected_provenance,
        )

    assert resumed_accelerator.loaded is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("partial", "incomplete"),
        ("state", "Accelerator"),
        ("source", "source"),
        ("topology", "topology"),
        ("cursor", "cursor"),
    ],
)
def test_baton_resume_rejects_partial_source_topology_or_cursor_mismatch(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    model, optimizer, scheduler = _tiny_training_state()
    accelerator = _StateAccelerator(model, optimizer, scheduler)
    checkpoint = save_baton_training_checkpoint(
        accelerator,
        tmp_path / "run",
        cursor=TrainingCursor(
            global_step=2,
            epoch=0,
            consumed_microbatches=2,
            microbatches_per_epoch=4,
            sampler_seed=42,
        ),
        diffusion_model=model,
        source="qwen35_baton_teacher",
        training_provenance=_teacher_training_provenance(),
    )
    metadata_path = checkpoint / "baton_state.json"
    if mutation == "partial":
        metadata_path.unlink()
    elif mutation == "state":
        (checkpoint / "accelerator_state.pt").unlink()
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if mutation == "source":
            metadata["source"] = "qwen35_baton_prediction"
            metadata["training_provenance"] = (
                _prediction_training_provenance()
            )
        elif mutation == "topology":
            metadata["topology_hash"] = "7" * 64
        else:
            metadata["cursor"]["microbatches_per_epoch"] = 3
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    resumed_model, resumed_optimizer, resumed_scheduler = (
        _tiny_training_state()
    )
    resumed_accelerator = _StateAccelerator(
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
    )

    with pytest.raises(ValueError, match=message):
        load_baton_training_checkpoint(
            resumed_accelerator,
            checkpoint,
            diffusion_model=resumed_model,
            expected_source="qwen35_baton_teacher",
            expected_microbatches_per_epoch=4,
            expected_sampler_seed=42,
            expected_training_provenance=_teacher_training_provenance(),
        )


def test_trainer_prepares_baton_resume_without_registering_frozen_source(
    tmp_path: Path,
) -> None:
    first_model, first_optimizer, first_scheduler = _tiny_training_state()
    first_accelerator = _StateAccelerator(
        first_model,
        first_optimizer,
        first_scheduler,
    )
    cursor = TrainingCursor(
        global_step=2,
        epoch=0,
        consumed_microbatches=2,
        microbatches_per_epoch=4,
        sampler_seed=42,
    )
    checkpoint = save_baton_training_checkpoint(
        first_accelerator,
        tmp_path / "run",
        cursor=cursor,
        diffusion_model=first_model,
        source="qwen35_baton_teacher",
        training_provenance=_teacher_training_provenance(),
    )

    resumed_model, resumed_optimizer, resumed_scheduler = (
        _tiny_training_state()
    )
    accelerator = _StateAccelerator(
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
    )
    frozen_teacher = _RecordingTeacher()
    trainer = Trainer.__new__(Trainer)
    trainer.grounded_training_enabled = False
    trainer.baton_components = BatonConditioningComponents(
        source="qwen35_baton_teacher",
        teacher=frozen_teacher,
        planner=None,
    )
    trainer.diffusion_model = resumed_model
    trainer.optimizer = resumed_optimizer
    trainer.train_dataloader = [0, 1, 2, 3]
    trainer.lr_scheduler = resumed_scheduler
    trainer.state = SimpleNamespace(accelerator=accelerator)
    trainer.args = SimpleNamespace(resume_from_checkpoint=str(checkpoint))
    trainer.sampler_seed = 42
    trainer.baton_training_provenance = _teacher_training_provenance()
    trainer.resume_cursor = None
    trainer.current_cursor = None

    trainer.prepare_for_training()

    assert trainer.resume_cursor == cursor
    assert trainer.current_cursor == cursor
    assert accelerator.prepared_components == (
        resumed_model,
        resumed_optimizer,
        trainer.train_dataloader,
        resumed_scheduler,
    )
    prepared_ids = {
        id(parameter)
        for component in accelerator.prepared_components
        if isinstance(component, nn.Module)
        for parameter in component.parameters()
    }
    frozen_ids = {
        id(parameter) for parameter in frozen_teacher.model.parameters()
    }
    assert prepared_ids.isdisjoint(frozen_ids)


class _PairedValidationPipeline:
    calls: list[dict[str, Any]] = []

    def __init__(self, *args) -> None:
        pass

    def infer(self, **kwargs):
        generator = kwargs["generator"]
        trace = {
            "prompt_embeds": torch.tensor([[1.0, 2.0]]),
            "prompt_attention_mask": torch.tensor([[True, True]]),
            "initial_video_noise": torch.rand(2, generator=generator),
            "initial_actions": torch.rand(2, generator=generator),
            "timesteps": torch.tensor([1000, 500, 1]),
        }
        type(self).calls.append(
            {
                "image": kwargs["image"].clone(),
                "prompt": tuple(kwargs["prompt"]),
                "semantic_plan": kwargs["semantic_plan"],
                "semantic_plan_times": kwargs["semantic_plan_times"],
                "semantic_plan_positions": kwargs[
                    "semantic_plan_positions"
                ],
                "semantic_plan_relevance": kwargs[
                    "semantic_plan_relevance"
                ],
                "semantic_condition_mask": kwargs[
                    "semantic_condition_mask"
                ].clone(),
                "trace": trace,
            }
        )
        return [
            {
                "video": torch.zeros(2, 3, 9, 2, 2),
                "action": torch.zeros(1, 36, 7),
                "validation_trace": trace,
            }
        ]


def test_paired_validation_reuses_batch_condition_noise_timesteps_and_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _PairedValidationPipeline.calls = []
    monkeypatch.setattr(ge_trainer_module, "save_video", lambda *args, **kwargs: None)
    planner = _RecordingPlanner()
    trainer = Trainer.__new__(Trainer)
    trainer.pipeline_class = _PairedValidationPipeline
    trainer.scheduler = object()
    trainer.vae = object()
    trainer.text_encoder = object()
    trainer.tokenizer = object()
    trainer.diffusion_model = _TinyDiffusion(1, 1, bias=False)
    trainer.baton_components = BatonConditioningComponents(
        source="qwen35_baton_prediction",
        teacher=None,
        planner=planner,
    )
    trainer.semantic_encoder = None
    trainer.semantic_planner = None
    trainer.grounded_training_enabled = False
    trainer.TEMPORAL_DOWN_RATIO = 2
    trainer.video_frame_rate = 20
    trainer.writer = None
    trainer.state = SimpleNamespace(weight_dtype=torch.float32)
    trainer.args = SimpleNamespace(
        data={
            "train": {
                "n_previous": 4,
                "chunk": 9,
                "action_chunk": 36,
            }
        },
        semantic_plan=_semantic_config(
            "qwen35_baton_prediction"
        )["semantic_plan"],
        return_action=True,
        return_video=True,
        add_state=False,
        num_inference_step=3,
        pixel_wise_timestep=True,
        diffusion_model={"config": {"action_in_channels": 7}},
        basic_fps=30,
    )
    batch = {
        "video": _video(batch_size=1),
        "caption": ["pick"],
        "actions": torch.arange(36 * 7).reshape(1, 36, 7).float(),
    }
    trainer.val_dataloader = [batch]
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        unwrap_model=lambda model: model,
    )

    results = trainer.validate_baton_modes(
        accelerator,
        tmp_path,
        global_step=5_000,
        n_view=2,
        n_chunk=1,
        to_log=False,
    )

    assert len(results) == 2
    assert results[0]["batch"] is batch
    assert results[1]["batch"] is batch
    assert len(planner.calls) == 1
    assert len(_PairedValidationPipeline.calls) == 2
    enabled, disabled = _PairedValidationPipeline.calls
    for field in (
        "image",
        "semantic_plan",
        "semantic_plan_times",
        "semantic_plan_positions",
    ):
        torch.testing.assert_close(
            enabled[field],
            disabled[field],
            rtol=0,
            atol=0,
        )
    assert enabled["prompt"] == disabled["prompt"] == ("pick",)
    assert enabled["semantic_plan_relevance"] is None
    assert disabled["semantic_plan_relevance"] is None
    for field in (
        "prompt_embeds",
        "prompt_attention_mask",
        "initial_video_noise",
        "initial_actions",
        "timesteps",
    ):
        torch.testing.assert_close(
            enabled["trace"][field],
            disabled["trace"][field],
            rtol=0,
            atol=0,
        )
    assert torch.equal(
        enabled["semantic_condition_mask"],
        torch.ones(2),
    )
    assert torch.equal(
        disabled["semantic_condition_mask"],
        torch.zeros(2),
    )
