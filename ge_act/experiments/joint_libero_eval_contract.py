"""Fail-closed contracts for joint VLM–GE-Act LIBERO evaluation."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from models.ltx_models.vlm_semantic_planner import (
    validate_dual_camera_planner_metadata,
)
from qwen3_vl_semantic_planner.libero_target_text import (
    LIBERO_TGT_PREPROCESSING,
    preprocess_libero_instructions,
)


EXPECTED_CAMERA_VIEWS = 2
EXPECTED_KEYFRAMES = 4
EXPECTED_TOKENS_PER_KEYFRAME = 256
EXPECTED_FEATURE_DIM = 1024
EXPECTED_KEYFRAME_OFFSETS = [2, 4, 6, 8]
EXPECTED_KEYFRAME_TIMES = [0.25, 0.5, 0.75, 1.0]


def prepare_joint_inference_prompt(prompt: str) -> str:
    """Apply the target-aware instruction contract used by joint inference."""

    return preprocess_libero_instructions(
        [prompt],
        preprocessing=LIBERO_TGT_PREPROCESSING,
    )[0]


@dataclass(frozen=True)
class JointEvaluationCheckpoint:
    """Validated paths and metadata for an inference-only joint export."""

    root: Path
    ltx_dir: Path
    planner_dir: Path
    metadata: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint metadata: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON metadata: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"metadata must be a JSON object: {path}")
    return value


def _require_equal(
    metadata: dict[str, Any],
    field: str,
    expected: Any,
    *,
    source: str,
) -> None:
    actual = metadata.get(field)
    if actual != expected:
        raise ValueError(
            f"incompatible {source} field {field}: "
            f"expected {expected!r}, got {actual!r}"
        )


def validate_joint_evaluation_checkpoint(
    checkpoint_dir: str | Path,
    *,
    expected_step: int = 40_000,
) -> JointEvaluationCheckpoint:
    """Validate the exact two-view K4 inference bundle before model loading."""

    root = Path(checkpoint_dir).expanduser().resolve()
    joint = _read_json(root / "joint_meta.json")
    for field, expected in {
        "global_step": int(expected_step),
        "num_camera_views": EXPECTED_CAMERA_VIEWS,
        "num_keyframes": EXPECTED_KEYFRAMES,
        "tokens_per_keyframe": EXPECTED_TOKENS_PER_KEYFRAME,
        "future_keyframe_offsets": EXPECTED_KEYFRAME_OFFSETS,
        "instruction_preprocessing": LIBERO_TGT_PREPROCESSING,
    }.items():
        _require_equal(joint, field, expected, source="joint metadata")

    ltx_dir = root / "ltx"
    planner_dir = root / "planner"
    ltx_config = _read_json(ltx_dir / "config.json")
    for field, expected in {
        "semantic_plan_context": True,
        "semantic_plan_in_dim": EXPECTED_FEATURE_DIM,
        "semantic_plan_num_keyframes": EXPECTED_KEYFRAMES,
        "semantic_plan_num_views": EXPECTED_CAMERA_VIEWS,
    }.items():
        _require_equal(ltx_config, field, expected, source="LTX config")
    if not any(ltx_dir.glob("*.safetensors")):
        raise FileNotFoundError(f"missing LTX safetensors export in {ltx_dir}")

    planner_meta = _read_json(planner_dir / "planner_meta.json")
    validate_dual_camera_planner_metadata(
        planner_meta,
        expected_instruction_preprocessing=LIBERO_TGT_PREPROCESSING,
    )
    _require_equal(
        planner_meta,
        "step",
        int(expected_step),
        source="planner metadata",
    )
    for name in ("plan_head.pt", "depth_head.pt", "plan_token_embedding.pt"):
        path = planner_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing planner export file: {path}")
    for name in ("qwen3vl_lora_or_model", "processor"):
        path = planner_dir / name
        if not path.is_dir():
            raise FileNotFoundError(f"missing planner export directory: {path}")

    return JointEvaluationCheckpoint(
        root=root,
        ltx_dir=ltx_dir,
        planner_dir=planner_dir,
        metadata=joint,
    )


def normalize_joint_current_images(images: torch.Tensor) -> torch.Tensor:
    """Return ordered main/wrist images as finite float32 BCHW in [-1, 1]."""

    value = torch.as_tensor(images)
    if value.ndim != 4:
        raise ValueError(
            f"two-view images must be rank 4, got {tuple(value.shape)}"
        )
    if value.shape[1] != 3 and value.shape[-1] == 3:
        value = value.permute(0, 3, 1, 2)
    if tuple(value.shape[:2]) != (EXPECTED_CAMERA_VIEWS, 3):
        raise ValueError(
            "images must be ordered [main,wrist] as [2,3,H,W], "
            f"got {tuple(value.shape)}"
        )
    if value.dtype == torch.uint8:
        value = value.float().div(127.5).sub(1.0)
    else:
        value = value.float()
        if not torch.isfinite(value).all():
            raise ValueError("current images must be finite")
        if value.min() < -1.0001 or value.max() > 1.0001:
            raise ValueError("float current images must already be in [-1,1]")
    return value.contiguous()


def build_joint_semantic_condition(
    planner: Any,
    current_images: torch.Tensor,
    instruction: str,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict and validate the exact K4 semantic tensors consumed by LTX."""

    if current_images.ndim != 4 or tuple(current_images.shape[:2]) != (2, 3):
        raise ValueError(
            "current images must be ordered [main,wrist] as [2,3,H,W], "
            f"got {tuple(current_images.shape)}"
        )
    if not torch.isfinite(current_images).all():
        raise ValueError("current images must be finite")
    if current_images.min() < -1.0001 or current_images.max() > 1.0001:
        raise ValueError("current images must be normalized to [-1,1]")

    result = planner.predict(
        current_images.unsqueeze(0),
        [str(instruction)],
    )
    plan = result.semantic_tokens
    times = result.times
    expected_plan_shape = (
        1,
        EXPECTED_CAMERA_VIEWS,
        EXPECTED_KEYFRAMES,
        EXPECTED_TOKENS_PER_KEYFRAME,
        EXPECTED_FEATURE_DIM,
    )
    if not torch.is_tensor(plan) or tuple(plan.shape) != expected_plan_shape:
        shape = tuple(plan.shape) if torch.is_tensor(plan) else None
        raise RuntimeError(
            f"semantic plan shape mismatch: expected {expected_plan_shape}, got {shape}"
        )
    expected_time_shape = (EXPECTED_CAMERA_VIEWS, EXPECTED_KEYFRAMES)
    if not torch.is_tensor(times) or tuple(times.shape) != expected_time_shape:
        shape = tuple(times.shape) if torch.is_tensor(times) else None
        raise RuntimeError(
            f"semantic time shape mismatch: expected {expected_time_shape}, got {shape}"
        )
    if not torch.isfinite(plan).all() or not torch.isfinite(times).all():
        raise RuntimeError("semantic plan and times must be finite")
    expected_times = torch.tensor(
        EXPECTED_KEYFRAME_TIMES,
        device=times.device,
        dtype=times.dtype,
    ).repeat(EXPECTED_CAMERA_VIEWS, 1)
    if not torch.allclose(times, expected_times, atol=1e-6, rtol=0):
        raise RuntimeError(
            "semantic time values mismatch: expected "
            f"{expected_times.tolist()}, got {times.tolist()}"
        )

    target_device = torch.device(device)
    return (
        plan.to(device=target_device, dtype=dtype),
        times.to(device=target_device, dtype=torch.float32),
        torch.ones(
            EXPECTED_CAMERA_VIEWS,
            device=target_device,
            dtype=dtype,
        ),
    )


class SemanticConditionedPipelineProxy:
    """Require one pending semantic condition for every pipeline inference."""

    _REQUIRED_KEYS = {
        "semantic_plan",
        "semantic_plan_times",
        "semantic_condition_mask",
    }

    def __init__(
        self,
        pipeline: Any,
        consume_conditioning: Callable[[], dict[str, torch.Tensor]],
    ) -> None:
        self._pipeline = pipeline
        self._consume_conditioning = consume_conditioning

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pipeline, name)

    def infer(self, *args: Any, **kwargs: Any) -> Any:
        try:
            conditioning = self._consume_conditioning()
        except IndexError as error:
            raise RuntimeError(
                "joint pipeline inference has no pending semantic conditioning"
            ) from error
        if not isinstance(conditioning, dict):
            raise RuntimeError("joint pipeline conditioning must be a dictionary")
        if set(conditioning) != self._REQUIRED_KEYS:
            raise RuntimeError(
                "joint pipeline conditioning keys mismatch: "
                f"expected {sorted(self._REQUIRED_KEYS)}, "
                f"got {sorted(conditioning)}"
            )
        overlap = self._REQUIRED_KEYS.intersection(kwargs)
        if overlap:
            raise RuntimeError(
                f"duplicate semantic conditioning keys: {sorted(overlap)}"
            )
        return self._pipeline.infer(*args, **kwargs, **conditioning)
