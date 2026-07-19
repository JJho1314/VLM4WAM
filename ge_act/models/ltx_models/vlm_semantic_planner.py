"""Frozen two-view Qwen planner used as GE-Act semantic conditioning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from PIL import Image
from torch import nn


_DUAL_CAMERA_COMMON_METADATA = {
    "planner_input_layout": "separate_camera_images",
    "camera_names": ["main", "wrist"],
    "num_camera_views": 2,
    "camera_head_sharing": "shared_head_per_view_image_context",
    "semantic_teacher": "siglip2-large-patch16-256",
}

_K1_METADATA = {
    "future_keyframe_offsets": [8],
    "num_keyframes": 1,
    "semantic_output_layout": "batch_view_token_feature",
    "grid_size": 16,
    "semantic_dim": 1024,
    "target_tokens": 256,
    "video_target_type": "siglip2",
}

_K4_METADATA = {
    "future_keyframe_offsets": [2, 4, 6, 8],
    "num_keyframes": 4,
    "semantic_output_layout": "batch_view_keyframe_token_feature",
    "sequence_length": 9,
    "grid_size": 16,
    "semantic_dim": 1024,
    "target_tokens": 4 * 256,
    "target_tokens_per_keyframe": 256,
    "planner_token_count": 384,
    "video_target_type": "siglip2",
    "use_current_alignment": False,
}

_CHECKPOINT_FILES = (
    "planner_meta.json",
    "plan_head.pt",
    "depth_head.pt",
    "plan_token_embedding.pt",
)

_CURRENT_ALIGNMENT_FILES = (
    "current_plan_head.pt",
    "current_depth_head.pt",
)


def _metadata_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return type(actual) is bool and actual == expected
    if isinstance(expected, int):
        return type(actual) is int and actual == expected
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _metadata_matches(item, expected_item)
                for item, expected_item in zip(actual, expected, strict=True)
            )
        )
    return actual == expected


def validate_dual_camera_planner_metadata(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Validate the exact independently trained main/wrist export geometry."""
    if not isinstance(metadata, dict):
        raise ValueError("dual-camera planner metadata must be a dictionary")
    for field, expected in _DUAL_CAMERA_COMMON_METADATA.items():
        actual = metadata.get(field)
        if not _metadata_matches(actual, expected):
            raise ValueError(
                f"incompatible dual-camera planner metadata field {field}: "
                f"expected {expected!r}, got {actual!r}"
            )
    num_keyframes = metadata.get("num_keyframes")
    if num_keyframes == 1:
        geometry = _K1_METADATA
        planner_token_count = 256
    elif num_keyframes == 4:
        geometry = _K4_METADATA
        planner_token_count = 384
    else:
        raise ValueError(
            "incompatible dual-camera planner metadata field num_keyframes: "
            f"expected 1 or 4, got {num_keyframes!r}"
        )
    for field, expected in geometry.items():
        actual = metadata.get(field)
        if not _metadata_matches(actual, expected):
            raise ValueError(
                f"incompatible dual-camera planner metadata field {field}: "
                f"expected {expected!r}, got {actual!r}"
            )
    input_frame = metadata.get("planner_input_frame")
    if input_frame not in (None, "separate_camera_images"):
        raise ValueError(
            "incompatible dual-camera planner metadata field planner_input_frame: "
            f"expected 'separate_camera_images', got {input_frame!r}"
        )
    expected_tokens = [
        f"<|sem_plan_{index}|>" for index in range(planner_token_count)
    ]
    if metadata.get("plan_token_strings") != expected_tokens:
        raise ValueError(
            "incompatible dual-camera planner metadata field plan_token_strings: "
            f"expected {planner_token_count} ordered planner tokens"
        )
    return metadata


def normalized_bvchw_to_pil_pairs(
    current_images: torch.Tensor,
) -> list[tuple[Image.Image, Image.Image]]:
    if current_images.ndim != 5 or current_images.shape[1:3] != (2, 3):
        raise ValueError(
            f"current_images must be [B,2,3,H,W], got {tuple(current_images.shape)}"
        )
    if not torch.isfinite(current_images).all():
        raise ValueError("current_images must be finite")
    if current_images.min() < -1.0001 or current_images.max() > 1.0001:
        raise ValueError("current_images must be normalized to [-1,1]")
    rgb = (
        (current_images.detach().float().cpu() + 1.0)
        .mul(127.5)
        .round()
        .clamp(0, 255)
        .byte()
        .permute(0, 1, 3, 4, 2)
        .contiguous()
    )
    return [
        (Image.fromarray(sample[0].numpy()), Image.fromarray(sample[1].numpy()))
        for sample in rgb
    ]


@dataclass(frozen=True)
class DualCameraSemanticPlan:
    semantic_tokens: torch.Tensor
    times: torch.Tensor


class FrozenDualCameraVLMPlanner:
    """Own a frozen planner and expose only its future SigLIP2 view grids."""

    def __init__(
        self,
        *,
        wrapper: nn.Module,
        processor: Any,
        input_builder: Callable[..., dict[str, Any]],
        input_mover: Callable[[dict[str, Any]], dict[str, Any]],
        plan_tokens: Sequence[str],
        device: torch.device | str,
        num_keyframes: int,
        future_keyframe_offsets: Sequence[int],
        sequence_length: int,
        target_tokens_per_keyframe: int,
    ) -> None:
        self.wrapper = wrapper
        self.processor = processor
        self.input_builder = input_builder
        self.input_mover = input_mover
        self.plan_tokens = list(plan_tokens)
        self.device = torch.device(device)
        self.num_keyframes = int(num_keyframes)
        self.future_keyframe_offsets = tuple(
            int(offset) for offset in future_keyframe_offsets
        )
        self.sequence_length = int(sequence_length)
        self.target_tokens_per_keyframe = int(target_tokens_per_keyframe)

    @classmethod
    def from_components(
        cls,
        *,
        wrapper: nn.Module,
        processor: Any,
        input_builder: Callable[..., dict[str, Any]],
        input_mover: Callable[[dict[str, Any]], dict[str, Any]],
        device: torch.device | str,
        plan_tokens: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "FrozenDualCameraVLMPlanner":
        if metadata is not None:
            validate_dual_camera_planner_metadata(metadata)
            num_keyframes = int(metadata["num_keyframes"])
            offsets = tuple(int(value) for value in metadata["future_keyframe_offsets"])
            sequence_length = int(metadata.get("sequence_length", 9))
            tokens_per_keyframe = int(
                metadata.get(
                    "target_tokens_per_keyframe",
                    int(metadata["target_tokens"]) // num_keyframes,
                )
            )
            expected_plan_tokens = int(
                metadata.get("planner_token_count", 256)
            )
        else:
            num_keyframes = int(getattr(wrapper, "num_keyframes", 1))
            offsets = (8,) if num_keyframes == 1 else (2, 4, 6, 8)
            sequence_length = 9
            target_len = int(getattr(wrapper, "target_len", num_keyframes * 256))
            if target_len % num_keyframes != 0:
                raise ValueError(
                    f"target_len {target_len} does not divide K={num_keyframes}"
                )
            tokens_per_keyframe = target_len // num_keyframes
            expected_plan_tokens = int(getattr(wrapper, "latent_len", 256))
        if plan_tokens is None:
            plan_tokens = [
                f"<|sem_plan_{index}|>" for index in range(expected_plan_tokens)
            ]
        if len(plan_tokens) != expected_plan_tokens:
            raise ValueError(
                "dual-camera planner requires "
                f"{expected_plan_tokens} plan tokens, got {len(plan_tokens)}"
            )
        wrapper.eval()
        for parameter in wrapper.parameters():
            parameter.requires_grad_(False)
        return cls(
            wrapper=wrapper,
            processor=processor,
            input_builder=input_builder,
            input_mover=input_mover,
            plan_tokens=plan_tokens,
            device=device,
            num_keyframes=num_keyframes,
            future_keyframe_offsets=offsets,
            sequence_length=sequence_length,
            target_tokens_per_keyframe=tokens_per_keyframe,
        )

    @staticmethod
    def _load_checkpoint_components(
        checkpoint_dir: Path,
        metadata: dict[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[nn.Module, Any, Callable[..., dict[str, Any]], Callable[..., Any]]:
        from qwen3_vl_semantic_planner.ge_act_dual_camera import (
            build_dual_camera_planner_inputs,
        )
        from qwen3_vl_semantic_planner.qwen3vl_wrapper import (
            load_qwen3vl_model_and_processor,
            move_qwen_inputs_to_device,
        )
        from qwen3_vl_semantic_planner.train_qwen3vl4b_lingbot_dino_planner import (
            PlannerWrapper,
        )

        dtype_names = {
            torch.bfloat16: "bf16",
            torch.float16: "fp16",
            torch.float32: "fp32",
        }
        if dtype not in dtype_names:
            raise ValueError(f"unsupported planner dtype: {dtype}")
        model, processor = load_qwen3vl_model_and_processor(
            checkpoint_dir / "qwen3vl_lora_or_model",
            processor_path=checkpoint_dir / "processor",
            device=device,
            dtype=dtype_names[dtype],
            attn_implementation="sdpa",
            local_files_only=True,
            eval_mode=True,
        )
        wrapper = PlannerWrapper.from_exported_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            metadata=metadata,
        ).to(device=device, dtype=dtype)
        model_dtype = next(wrapper.model.parameters()).dtype

        def move(inputs: dict[str, Any]) -> dict[str, Any]:
            return move_qwen_inputs_to_device(
                inputs,
                device,
                model_dtype=model_dtype,
            )

        return wrapper, processor, build_dual_camera_planner_inputs, move

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "FrozenDualCameraVLMPlanner":
        checkpoint_dir = Path(checkpoint_dir)
        metadata_path = checkpoint_dir / "planner_meta.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"dual-camera planner checkpoint is missing {metadata_path}"
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(f"invalid planner metadata: {metadata_path}") from error
        validate_dual_camera_planner_metadata(metadata)

        required_files = list(_CHECKPOINT_FILES)
        if bool(metadata.get("use_current_alignment", metadata["num_keyframes"] == 1)):
            required_files.extend(_CURRENT_ALIGNMENT_FILES)
        missing = [
            name
            for name in required_files
            if not (checkpoint_dir / name).is_file()
        ]
        for directory in ("qwen3vl_lora_or_model", "processor"):
            if not (checkpoint_dir / directory).is_dir():
                missing.append(directory + "/")
        if missing:
            raise FileNotFoundError(
                f"incomplete dual-camera planner checkpoint {checkpoint_dir}: "
                f"missing {missing}"
            )

        resolved_device = torch.device(device)
        wrapper, processor, input_builder, input_mover = (
            cls._load_checkpoint_components(
                checkpoint_dir,
                metadata,
                device=resolved_device,
                dtype=dtype,
            )
        )
        return cls.from_components(
            wrapper=wrapper,
            processor=processor,
            input_builder=input_builder,
            input_mover=input_mover,
            plan_tokens=metadata["plan_token_strings"],
            metadata=metadata,
            device=resolved_device,
        )

    def prepare_inputs(
        self,
        current_images: torch.Tensor,
        instructions: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        image_pairs = normalized_bvchw_to_pil_pairs(current_images)
        return self.input_mover(
            self.input_builder(
                self.processor,
                image_pairs,
                [str(value) for value in instructions],
                self.plan_tokens,
            )
        )

    @torch.no_grad()
    def predict(
        self,
        current_images: torch.Tensor,
        instructions: Sequence[str],
    ) -> DualCameraSemanticPlan:
        if current_images.ndim != 5 or current_images.shape[1:3] != (2, 3):
            raise ValueError(
                f"current_images must be [B,2,3,H,W], got {tuple(current_images.shape)}"
            )
        if current_images.shape[0] != len(instructions):
            raise ValueError(
                "current_images/instructions batch mismatch: "
                f"{current_images.shape[0]} != {len(instructions)}"
            )
        model_inputs = self.prepare_inputs(current_images, instructions)
        if bool(getattr(self.wrapper, "use_current_alignment", self.num_keyframes == 1)):
            plans = self.wrapper.predict_current_future_plans(**model_inputs)
            future = plans.get("future_dino")
        elif hasattr(self.wrapper, "predict_dino_depth_plan"):
            future, _depth = self.wrapper.predict_dino_depth_plan(**model_inputs)
        else:
            future = self.wrapper.predict_semantic_plan(**model_inputs)
        flat_tokens = self.num_keyframes * self.target_tokens_per_keyframe
        expected = (current_images.shape[0], 2, flat_tokens, 1024)
        if (
            not torch.is_tensor(future)
            or tuple(future.shape) != expected
            or not torch.isfinite(future).all()
        ):
            shape = tuple(future.shape) if torch.is_tensor(future) else None
            raise RuntimeError(
                f"future_siglip must be finite with shape {expected}, got {shape}"
            )
        return DualCameraSemanticPlan(
            semantic_tokens=future.detach().reshape(
                current_images.shape[0],
                2,
                self.num_keyframes,
                self.target_tokens_per_keyframe,
                1024,
            ),
            times=(
                torch.tensor(
                    self.future_keyframe_offsets,
                    device=future.device,
                    dtype=torch.float32,
                )
                / float(self.sequence_length - 1)
            )
            .reshape(1, self.num_keyframes)
            .expand(current_images.shape[0] * 2, -1)
            .clone(),
        )
