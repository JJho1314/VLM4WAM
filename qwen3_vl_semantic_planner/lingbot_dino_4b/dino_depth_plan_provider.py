"""Frozen online DINO+depth planner for the strict FastWAM contract."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image


EXPECTED_METADATA = {
    "sequence_length": 9,
    "num_keyframes": 1,
    "grid_size": 16,
    "semantic_dim": 1024,
    "target_tokens": 256,
    "keyframe_offsets": [8],
    "keyframe_scheme": "even_future",
    "has_depth_head": True,
    "depth_feature_dim": 1024,
    "depth_grid_size": 16,
    "shared_latent_per_keyframe": 32,
    "private_latent_per_keyframe": 32,
    "branch_latent_per_keyframe": 8,
    "total_unique_latent_per_keyframe": 16,
    "latent_len": 16,
    "use_current_alignment": True,
    "num_task_tokens": 8,
    "query_layout": "current_8_then_future_8__dino_depth_shared_within_time",
    "plan_head_type": "lingbot_dino",
    "plan_head_num_heads": 16,
    "plan_head_dropout": 0.0,
    "sem_mlp_hidden_size": 0,
    "planner_input_frame": "fastwam_current_multicamera_composite",
    "token_order": "keyframe_major_row_major",
}

REQUIRED_FINITE_NUMERIC_METADATA = (
    "mse_loss_weight",
    "cosine_loss_weight",
    "norm_loss_weight",
    "variance_loss_weight",
    "infonce_loss_weight",
    "infonce_temperature",
    "depth_loss_weight",
    "current_dino_loss_weight",
    "future_dino_loss_weight",
    "current_depth_loss_weight",
    "future_depth_loss_weight",
)


@dataclass(frozen=True)
class PlannerContract:
    sequence_length: int
    num_keyframes: int
    grid_size: int
    semantic_dim: int
    target_tokens: int
    shared_latent_per_keyframe: int
    private_latent_per_keyframe: int
    branch_latent_per_keyframe: int
    total_unique_latent_per_keyframe: int
    num_task_tokens: int
    independent_modality_task_tokens: bool
    keyframe_offsets: tuple[int, ...]
    normalized_keyframe_times: tuple[float, ...]
    plan_token_strings: tuple[str, ...]


@dataclass(frozen=True)
class DinoDepthPlan:
    dino_plan: torch.Tensor
    depth_plan: torch.Tensor
    semantic_plan_times: torch.Tensor


def _is_finite_number(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _matches_expected_metadata_value(actual, expected) -> bool:
    if isinstance(expected, bool):
        return type(actual) is bool and actual == expected
    if isinstance(expected, int):
        return type(actual) is int and actual == expected
    if isinstance(expected, float):
        return _is_finite_number(actual) and float(actual) == expected
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _matches_expected_metadata_value(item, expected_item)
                for item, expected_item in zip(actual, expected, strict=True)
            )
        )
    return actual == expected


def validate_planner_metadata(metadata: dict) -> PlannerContract:
    """Validate and freeze the exact planner geometry consumed by FastWAM."""
    independent = metadata.get("independent_modality_task_tokens", False)
    if type(independent) is not bool:
        raise ValueError(
            "incompatible planner metadata field "
            "independent_modality_task_tokens: expected a boolean, "
            f"got {independent!r}"
        )
    num_task_tokens = metadata.get("num_task_tokens", 8)
    if type(num_task_tokens) is not int or num_task_tokens <= 0:
        raise ValueError(
            "incompatible planner metadata field num_task_tokens: "
            f"expected a positive integer, got {num_task_tokens!r}"
        )
    expected_metadata = dict(EXPECTED_METADATA)
    expected_metadata.update(
        {
            "num_task_tokens": num_task_tokens,
            "branch_latent_per_keyframe": num_task_tokens,
            "total_unique_latent_per_keyframe": 2 * num_task_tokens,
            "latent_len": 2 * num_task_tokens,
            "query_layout": (
                f"current_{num_task_tokens}_then_future_{num_task_tokens}__"
                "dino_depth_shared_within_time"
            ),
        }
    )
    if independent:
        expected_metadata.update(
            {
                "total_unique_latent_per_keyframe": 4 * num_task_tokens,
                "latent_len": 4 * num_task_tokens,
                "query_layout": (
                    f"current_dino_{num_task_tokens}_then_"
                    f"current_depth_{num_task_tokens}_then_"
                    f"future_dino_{num_task_tokens}_then_"
                    f"future_depth_{num_task_tokens}"
                ),
            }
        )
    for field, expected in expected_metadata.items():
        actual = metadata.get(field)
        if not _matches_expected_metadata_value(actual, expected):
            raise ValueError(
                f"incompatible planner metadata field {field}: "
                f"expected {expected!r}, got {actual!r}"
            )

    for field in REQUIRED_FINITE_NUMERIC_METADATA:
        actual = metadata.get(field)
        if not _is_finite_number(actual):
            raise ValueError(
                f"incompatible planner metadata field {field}: "
                f"expected a finite number, got {actual!r}"
            )

    plan_token_ids = metadata.get("plan_token_ids")
    if (
        not isinstance(plan_token_ids, list)
        or len(plan_token_ids) != expected_metadata["latent_len"]
        or any(type(token_id) is not int or token_id < 0 for token_id in plan_token_ids)
        or len(set(plan_token_ids)) != len(plan_token_ids)
    ):
        raise ValueError(
            "incompatible planner metadata field plan_token_ids: expected "
            f"{expected_metadata['latent_len']} unique non-negative integers, "
            f"got {plan_token_ids!r}"
        )

    expected_times = tuple(
        offset / (EXPECTED_METADATA["sequence_length"] - 1)
        for offset in EXPECTED_METADATA["keyframe_offsets"]
    )
    raw_times = metadata.get("normalized_keyframe_times", ())
    if not isinstance(raw_times, list) or any(
        not _is_finite_number(value) for value in raw_times
    ):
        raise ValueError(
            "incompatible planner metadata field normalized_keyframe_times: "
            f"expected {expected_times!r}, got {raw_times!r}"
        )
    actual_times = tuple(float(value) for value in raw_times)
    if len(actual_times) != len(expected_times) or any(
        not math.isfinite(actual) or abs(actual - expected) > 1e-7
        for actual, expected in zip(actual_times, expected_times, strict=True)
    ):
        raise ValueError(
            "incompatible planner metadata field normalized_keyframe_times: "
            f"expected {expected_times!r}, got {actual_times!r}"
        )

    expected_plan_tokens = tuple(
        f"<|sem_plan_{index}|>"
        for index in range(expected_metadata["latent_len"])
    )
    raw_plan_tokens = metadata.get("plan_token_strings", ())
    try:
        actual_plan_tokens = tuple(raw_plan_tokens)
    except TypeError as error:
        raise ValueError(
            "incompatible planner metadata field plan_token_strings: "
            f"expected {expected_plan_tokens!r}, got {raw_plan_tokens!r}"
        ) from error
    if actual_plan_tokens != expected_plan_tokens:
        raise ValueError(
            "incompatible planner metadata field plan_token_strings: "
            f"expected {expected_plan_tokens!r}, "
            f"got {actual_plan_tokens!r}"
        )

    return PlannerContract(
        sequence_length=9,
        num_keyframes=1,
        grid_size=16,
        semantic_dim=1024,
        target_tokens=256,
        shared_latent_per_keyframe=32,
        private_latent_per_keyframe=32,
        branch_latent_per_keyframe=expected_metadata[
            "branch_latent_per_keyframe"
        ],
        total_unique_latent_per_keyframe=expected_metadata[
            "total_unique_latent_per_keyframe"
        ],
        num_task_tokens=num_task_tokens,
        independent_modality_task_tokens=independent,
        keyframe_offsets=(8,),
        normalized_keyframe_times=expected_times,
        plan_token_strings=expected_plan_tokens,
    )


def validate_checkpoint_files(checkpoint_dir: str | Path) -> Path:
    """Require the complete exported planner checkpoint before model loading."""
    checkpoint_dir = Path(checkpoint_dir)
    required_files = (
        "plan_head.pt",
        "depth_head.pt",
        "current_plan_head.pt",
        "current_depth_head.pt",
        "plan_token_embedding.pt",
        "planner_meta.json",
    )
    missing = [name for name in required_files if not (checkpoint_dir / name).is_file()]
    required_dirs = ("qwen3vl_lora_or_model", "processor")
    missing.extend(
        name for name in required_dirs if not (checkpoint_dir / name).is_dir()
    )
    if missing:
        raise FileNotFoundError(
            f"incomplete planner checkpoint {checkpoint_dir}: missing {missing}"
        )
    return checkpoint_dir


def image_tensor_batch_to_pil(images: torch.Tensor) -> list[Image.Image]:
    """Convert normalized FastWAM images from BCHW tensors to RGB PIL images."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(
            f"planner images must have shape [B, 3, H, W], got {tuple(images.shape)}"
        )
    if not torch.isfinite(images).all():
        raise ValueError("planner images contain non-finite values")
    images = images.detach().to(device="cpu", dtype=torch.float32)
    if images.min().item() < -1.0001 or images.max().item() > 1.0001:
        raise ValueError("planner images must be normalized to [-1, 1]")
    images = ((images + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return [Image.fromarray(image.permute(1, 2, 0).numpy()) for image in images]


class FrozenDinoDepthPlanProvider:
    """Plain, frozen inference provider for online DINO and depth plans."""

    def __init__(
        self,
        *,
        processor,
        wrapper,
        contract: PlannerContract,
        device: torch.device,
        input_builder,
        input_mover,
    ):
        self.processor = processor
        self.wrapper = wrapper.eval()
        self.contract = contract
        self.device = torch.device(device)
        self.input_builder = input_builder
        self.input_mover = input_mover
        for parameter in self.wrapper.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_components(
        cls,
        *,
        processor,
        wrapper,
        contract: PlannerContract,
        device: str | torch.device,
        input_builder,
        input_mover=None,
    ) -> "FrozenDinoDepthPlanProvider":
        return cls(
            processor=processor,
            wrapper=wrapper,
            contract=contract,
            device=torch.device(device),
            input_builder=input_builder,
            input_mover=(
                input_mover
                if input_mover is not None
                else lambda inputs: inputs.to(device)
            ),
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> "FrozenDinoDepthPlanProvider":
        checkpoint_dir = validate_checkpoint_files(checkpoint_dir)
        metadata = json.loads(
            (checkpoint_dir / "planner_meta.json").read_text(encoding="utf-8")
        )
        contract = validate_planner_metadata(metadata)

        from transformers import (
            AutoProcessor,
            Qwen3VLForConditionalGeneration,
        )

        trainer_dir = str(Path(__file__).resolve().parent.parent)
        if trainer_dir not in sys.path:
            sys.path.insert(0, trainer_dir)
        from train_semantic_planner import (
            PlannerWrapper,
            build_planner_inputs,
            move_qwen_inputs_to_device,
        )

        target_device = torch.device(device)
        processor = AutoProcessor.from_pretrained(
            checkpoint_dir / "processor",
            local_files_only=True,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint_dir / "qwen3vl_lora_or_model",
            torch_dtype=dtype,
            local_files_only=True,
        ).to(target_device)
        embedding_vocabulary_size = int(model.get_input_embeddings().weight.shape[0])
        largest_plan_token_id = max(metadata["plan_token_ids"])
        if largest_plan_token_id >= embedding_vocabulary_size:
            raise ValueError(
                "incompatible planner metadata field plan_token_ids: "
                f"token id {largest_plan_token_id} is outside model embedding "
                f"vocabulary size {embedding_vocabulary_size}"
            )
        wrapper = PlannerWrapper.from_exported_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            metadata=metadata,
        ).to(target_device)
        return cls(
            processor=processor,
            wrapper=wrapper,
            contract=contract,
            device=target_device,
            input_builder=lambda images, instructions, plan_sequence: (
                build_planner_inputs(
                    processor,
                    images,
                    instructions,
                    plan_sequence,
                )
            ),
            input_mover=lambda inputs: move_qwen_inputs_to_device(
                inputs,
                target_device,
                model_dtype=dtype,
            ),
        )

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        instructions: Sequence[str],
    ) -> DinoDepthPlan:
        if images.shape[0] != len(instructions):
            raise ValueError(
                "image/instruction batch mismatch: "
                f"{images.shape[0]} != {len(instructions)}"
            )
        pil_images = image_tensor_batch_to_pil(images)
        model_inputs = self.input_mover(
            self.input_builder(
                pil_images,
                list(instructions),
                list(self.contract.plan_token_strings),
            )
        )
        dino_plan, depth_plan = self.wrapper.predict_dino_depth_plan(**model_inputs)
        expected_shape = (
            images.shape[0],
            self.contract.target_tokens,
            self.contract.semantic_dim,
        )
        for name, tensor in (
            ("dino_plan", dino_plan),
            ("depth_plan", depth_plan),
        ):
            if tuple(tensor.shape) != expected_shape:
                raise RuntimeError(
                    f"{name} must have shape {expected_shape}, "
                    f"got {tuple(tensor.shape)}"
                )
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"{name} contains non-finite values")
        times = (
            torch.tensor(
                self.contract.normalized_keyframe_times,
                device=dino_plan.device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .expand(images.shape[0], -1)
        )
        return DinoDepthPlan(
            dino_plan=dino_plan.detach(),
            depth_plan=depth_plan.detach(),
            semantic_plan_times=times.detach(),
        )
