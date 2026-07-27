from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence
import sys

import pytest
import torch
import torch.nn as nn
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GE_ACT_ROOT = REPOSITORY_ROOT / "ge_act"
for path in (REPOSITORY_ROOT, GE_ACT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.ltx_models.semantic_conditioning import (  # noqa: E402
    build_patch_center_positions,
)
from runner import ge_inferencer as ge_inferencer_module  # noqa: E402
from runner import ge_trainer as ge_trainer_module  # noqa: E402
from runner.ge_trainer import (  # noqa: E402
    BatonConditioningComponents,
    build_baton_semantic_condition,
    build_optimizer_parameter_groups,
    prepare_baton_conditioning,
)
from scripts.preflight_ltx_siglip2 import collect_preflight_errors  # noqa: E402


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


@pytest.mark.parametrize("path", [STAGE2_CONFIG, STAGE3_CONFIG])
def test_baton_recipe_passes_semantic_static_preflight(path: Path) -> None:
    config = _load_config(path)

    assert collect_preflight_errors(
        config,
        world_size=8,
        check_paths=False,
    ) == []


@pytest.mark.parametrize("launcher", [STAGE2_LAUNCHER, STAGE3_LAUNCHER])
def test_launcher_derives_accumulation_and_runs_both_preflights_before_distributed(
    launcher: Path,
) -> None:
    source = launcher.read_text(encoding="utf-8")

    assert "GLOBAL_BATCH" in source
    assert "PER_DEVICE_BATCH" in source
    assert "GRADIENT_ACCUMULATION_STEPS" in source
    assert "GLOBAL_BATCH %" in source
    hdf5 = source.index("preflight_libero_fastwam_hdf5.py")
    semantic = source.index("preflight_ltx_siglip2.py")
    distributed = source.index("-m torch.distributed.run")
    assert hdf5 < semantic < distributed


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
