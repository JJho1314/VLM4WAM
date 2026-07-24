from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
GE_ACT_ROOT = REPO_ROOT / "ge_act"
if str(GE_ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(GE_ACT_ROOT))

from experiments.joint_libero_eval_contract import (
    SemanticConditionedPipelineProxy,
    build_joint_semantic_condition,
    normalize_joint_current_images,
    validate_joint_evaluation_checkpoint,
)

EVAL_LIBERO_PATH = GE_ACT_ROOT / "experiments" / "eval_libero.py"
JOINT_EVAL_PATH = GE_ACT_ROOT / "experiments" / "eval_libero_joint.py"
EVAL_CONFIG_PATH = (
    GE_ACT_ROOT
    / "configs"
    / "ltx_model"
    / "libero"
    / "action_model_libero_joint_step40000_eval.yaml"
)
EVAL_LAUNCHER_PATH = (
    GE_ACT_ROOT / "scripts" / "eval_joint_vlm_geact_a6000.sh"
)


def valid_k4_planner_metadata() -> dict[str, object]:
    return {
        "planner_input_layout": "separate_camera_images",
        "camera_names": ["main", "wrist"],
        "num_camera_views": 2,
        "camera_head_sharing": "shared_head_per_view_image_context",
        "semantic_output_layout": "batch_view_keyframe_token_feature",
        "semantic_teacher": "siglip2-large-patch16-256",
        "future_keyframe_offsets": [2, 4, 6, 8],
        "num_keyframes": 4,
        "sequence_length": 9,
        "grid_size": 16,
        "semantic_dim": 1024,
        "target_tokens": 1024,
        "target_tokens_per_keyframe": 256,
        "planner_token_count": 384,
        "video_target_type": "siglip2",
        "use_current_alignment": False,
        "step": 40_000,
        "plan_token_strings": [
            f"<|sem_plan_{index}|>" for index in range(384)
        ],
    }


def write_joint_export(tmp_path: Path, *, global_step: int = 40_000) -> Path:
    root = tmp_path / "step_40000"
    ltx = root / "ltx"
    planner = root / "planner"
    ltx.mkdir(parents=True)
    (planner / "qwen3vl_lora_or_model").mkdir(parents=True)
    (planner / "processor").mkdir()
    (root / "joint_meta.json").write_text(
        json.dumps(
            {
                "global_step": global_step,
                "num_camera_views": 2,
                "num_keyframes": 4,
                "tokens_per_keyframe": 256,
                "future_keyframe_offsets": [2, 4, 6, 8],
            }
        ),
        encoding="utf-8",
    )
    (ltx / "config.json").write_text(
        json.dumps(
            {
                "semantic_plan_context": True,
                "semantic_plan_in_dim": 1024,
                "semantic_plan_num_keyframes": 4,
                "semantic_plan_num_views": 2,
            }
        ),
        encoding="utf-8",
    )
    (ltx / "diffusion_pytorch_model.safetensors").touch()
    (planner / "planner_meta.json").write_text(
        json.dumps(valid_k4_planner_metadata()),
        encoding="utf-8",
    )
    for name in ("plan_head.pt", "depth_head.pt", "plan_token_embedding.pt"):
        (planner / name).touch()
    return root


class RecordingPlanner:
    def __init__(
        self,
        *,
        fill_value: float = 0.0,
        shape: tuple[int, ...] = (1, 2, 4, 256, 1024),
        times: torch.Tensor | None = None,
    ) -> None:
        self.fill_value = fill_value
        self.shape = shape
        self.times = times
        self.current_images: torch.Tensor | None = None
        self.instructions: list[str] | None = None

    def predict(
        self,
        current_images: torch.Tensor,
        instructions: list[str],
    ) -> SimpleNamespace:
        self.current_images = current_images.clone()
        self.instructions = list(instructions)
        times = self.times
        if times is None:
            times = torch.tensor([[0.25, 0.5, 0.75, 1.0]] * 2)
        return SimpleNamespace(
            semantic_tokens=torch.full(self.shape, self.fill_value),
            times=times,
        )


class RecordingPipeline:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def infer(self, **kwargs: object) -> dict[str, object]:
        self.kwargs = kwargs
        return kwargs


def test_joint_checkpoint_contract_accepts_exact_step40000(tmp_path: Path) -> None:
    root = write_joint_export(tmp_path)

    checkpoint = validate_joint_evaluation_checkpoint(root)

    assert checkpoint.root == root
    assert checkpoint.ltx_dir == root / "ltx"
    assert checkpoint.planner_dir == root / "planner"
    assert checkpoint.metadata["future_keyframe_offsets"] == [2, 4, 6, 8]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("global_step", 39_999),
        ("num_camera_views", 1),
        ("num_keyframes", 1),
        ("tokens_per_keyframe", 64),
        ("future_keyframe_offsets", [1, 3, 5, 7]),
    ],
)
def test_joint_checkpoint_contract_rejects_metadata_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root = write_joint_export(tmp_path)
    metadata_path = root / "joint_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        validate_joint_evaluation_checkpoint(root)


def test_joint_checkpoint_contract_requires_model_exports(tmp_path: Path) -> None:
    root = write_joint_export(tmp_path)
    (root / "planner" / "plan_head.pt").unlink()

    with pytest.raises(FileNotFoundError, match="plan_head.pt"):
        validate_joint_evaluation_checkpoint(root)


def test_joint_checkpoint_contract_rejects_ltx_semantic_drift(
    tmp_path: Path,
) -> None:
    root = write_joint_export(tmp_path)
    config_path = root / "ltx" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["semantic_plan_num_views"] = 1
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="semantic_plan_num_views"):
        validate_joint_evaluation_checkpoint(root)


def test_joint_checkpoint_contract_rejects_planner_step_drift(
    tmp_path: Path,
) -> None:
    root = write_joint_export(tmp_path)
    metadata_path = root / "planner" / "planner_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["step"] = 39_999
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="step"):
        validate_joint_evaluation_checkpoint(root)


def test_joint_image_normalization_converts_uint8_exactly_once() -> None:
    raw = torch.stack(
        (
            torch.zeros(3, 8, 8, dtype=torch.uint8),
            torch.full((3, 8, 8), 255, dtype=torch.uint8),
        )
    )

    normalized = normalize_joint_current_images(raw)

    assert normalized.dtype == torch.float32
    assert normalized[0].eq(-1).all()
    assert normalized[1].eq(1).all()


def test_joint_image_normalization_accepts_hwc_and_preserves_view_order() -> None:
    raw = torch.stack(
        (
            torch.zeros(8, 8, 3, dtype=torch.uint8),
            torch.full((8, 8, 3), 255, dtype=torch.uint8),
        )
    )

    normalized = normalize_joint_current_images(raw)

    assert normalized.shape == (2, 3, 8, 8)
    assert normalized[0].eq(-1).all()
    assert normalized[1].eq(1).all()


def test_joint_image_normalization_rejects_ambiguous_float_range() -> None:
    with pytest.raises(ValueError, match=r"\[-1,1\]"):
        normalize_joint_current_images(torch.full((2, 3, 8, 8), 255.0))


def test_semantic_condition_preserves_main_wrist_order() -> None:
    planner = RecordingPlanner()
    current = torch.stack(
        (
            torch.full((3, 8, 8), -0.5),
            torch.full((3, 8, 8), 0.5),
        )
    )

    plan, times, mask = build_joint_semantic_condition(
        planner,
        current,
        "pick the bowl",
        device="cpu",
        dtype=torch.bfloat16,
    )

    assert planner.current_images is not None
    assert planner.current_images.shape == (1, 2, 3, 8, 8)
    torch.testing.assert_close(planner.current_images[0], current)
    assert planner.instructions == ["pick the bowl"]
    assert plan.shape == (1, 2, 4, 256, 1024)
    assert plan.dtype == torch.bfloat16
    assert times.shape == (2, 4)
    assert times.dtype == torch.float32
    assert mask.tolist() == [1.0, 1.0]


@pytest.mark.parametrize(
    ("planner", "message"),
    [
        (RecordingPlanner(fill_value=float("nan")), "finite"),
        (RecordingPlanner(shape=(1, 1, 4, 256, 1024)), "shape"),
        (
            RecordingPlanner(times=torch.tensor([[0.0, 0.5, 0.75, 1.0]] * 2)),
            "values",
        ),
    ],
)
def test_semantic_condition_rejects_invalid_output(
    planner: RecordingPlanner,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        build_joint_semantic_condition(
            planner,
            torch.zeros(2, 3, 8, 8),
            "pick",
            device="cpu",
            dtype=torch.bfloat16,
        )


def test_pipeline_proxy_requires_and_forwards_semantics() -> None:
    base = RecordingPipeline()
    pending: list[dict[str, torch.Tensor]] = [
        {
            "semantic_plan": torch.zeros(1, 2, 4, 256, 1024),
            "semantic_plan_times": torch.zeros(2, 4),
            "semantic_condition_mask": torch.ones(2),
        }
    ]
    proxy = SemanticConditionedPipelineProxy(base, pending.pop)

    output = proxy.infer(image=torch.zeros(2, 3, 4, 8, 8))

    assert output == base.kwargs
    assert set(base.kwargs) >= {
        "semantic_plan",
        "semantic_plan_times",
        "semantic_condition_mask",
    }
    with pytest.raises(RuntimeError, match="conditioning"):
        proxy.infer(image=torch.zeros(2, 3, 4, 8, 8))


def test_pipeline_proxy_rejects_incomplete_or_duplicate_semantics() -> None:
    base = RecordingPipeline()
    incomplete = [{"semantic_plan": torch.zeros(1)}]
    proxy = SemanticConditionedPipelineProxy(base, incomplete.pop)
    with pytest.raises(RuntimeError, match="keys"):
        proxy.infer(image=torch.zeros(1))

    complete: list[dict[str, Any]] = [
        {
            "semantic_plan": torch.zeros(1),
            "semantic_plan_times": torch.zeros(1),
            "semantic_condition_mask": torch.ones(1),
        }
    ]
    proxy = SemanticConditionedPipelineProxy(base, complete.pop)
    with pytest.raises(RuntimeError, match="duplicate"):
        proxy.infer(
            image=torch.zeros(1),
            semantic_plan=torch.zeros(1),
        )


def test_joint_evaluator_loads_planner_and_uses_fail_closed_proxy() -> None:
    source = JOINT_EVAL_PATH.read_text(encoding="utf-8")

    assert "FrozenDualCameraVLMPlanner.from_checkpoint" in source
    assert "normalize_joint_current_images(" in source
    assert "build_joint_semantic_condition(" in source
    assert "SemanticConditionedPipelineProxy(" in source
    assert "return super().play(" in source
    assert "return {}" not in source


def test_joint_evaluator_has_no_semantic_disable_cli() -> None:
    source = JOINT_EVAL_PATH.read_text(encoding="utf-8")

    assert '"--joint_ckpt_dir"' in source
    assert '"--disable_semantic"' not in source
    assert '"--semantic_mode"' not in source


def test_original_evaluator_is_not_modified_for_joint_conditioning() -> None:
    source = EVAL_LIBERO_PATH.read_text(encoding="utf-8")

    assert "joint_libero_eval_contract" not in source
    assert "semantic_planner" not in source


def test_a6000_eval_config_matches_joint_training_geometry() -> None:
    config = yaml.safe_load(EVAL_CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["pretrained_model_name_or_path"] == (
        "/data/LFT-W02_data/junjie/weights/LTX-Video"
    )
    assert config["diffusion_model"]["model_path"].endswith(
        "joint_vlm_geact_action_k4_50k/step_40000/ltx"
    )
    assert config["return_action"] is True
    assert config["return_video"] is False
    assert config["add_state"] is True
    assert config["data"]["train"]["sample_size"] == [256, 256]
    assert config["data"]["train"]["n_previous"] == 4
    assert config["data"]["train"]["chunk"] == 9
    assert config["data"]["train"]["action_chunk"] == 36
    model = config["diffusion_model"]["config"]
    assert model["semantic_plan_context"] is True
    assert model["semantic_plan_num_views"] == 2
    assert model["semantic_plan_num_keyframes"] == 4
    assert model["semantic_plan_in_dim"] == 1024
    assert model["action_in_channels"] == 15
    assert model["action_out_channels"] == 15


def test_a6000_launcher_smoke_gates_full_evaluation() -> None:
    source = EVAL_LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=1" in source
    assert "SMOKE_MAX_TASKS=1" in source
    assert "--num_trails_per_task 1" in source
    assert "libero_spatial libero_object libero_goal libero_10" in source
    assert "--num_trails_per_task 50" in source
    assert "training_state" not in source


def test_a6000_launcher_uses_local_joint_paths_and_ge_act_environment() -> None:
    source = EVAL_LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "/data/LFT-W02_data/.conda/envs/ge-act/bin/python" in source
    assert "/data/LFT-W02_data/junjie/VLA_RL/docker_libero/LIBERO" in source
    assert 'PYTHONPATH="$ROOT:$ROOT/ge_act:$LIBERO_ROOT' in source
    assert "joint_vlm_geact_action_k4_50k/step_40000" in source
    assert "eval_libero_joint.py" in source
