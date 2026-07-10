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
    "num_keyframes": 4,
    "grid_size": 16,
    "semantic_dim": 1024,
    "target_tokens": 1024,
    "keyframe_offsets": [2, 4, 6, 8],
    "keyframe_scheme": "even_future",
    "has_depth_head": True,
    "depth_feature_dim": 1024,
    "depth_grid_size": 16,
    "shared_latent_per_keyframe": 32,
    "private_latent_per_keyframe": 32,
    "branch_latent_per_keyframe": 64,
    "total_unique_latent_per_keyframe": 96,
    "latent_len": 384,
    "query_layout": ("keyframe_major__shared_dino_private_depth_private"),
    "plan_head_type": "lingbot_dino",
    "planner_input_frame": "fastwam_current_multicamera_composite",
    "token_order": "keyframe_major_row_major",
}


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
    keyframe_offsets: tuple[int, ...]
    normalized_keyframe_times: tuple[float, ...]
    plan_token_strings: tuple[str, ...]


@dataclass(frozen=True)
class DinoDepthPlan:
    dino_plan: torch.Tensor
    depth_plan: torch.Tensor
    semantic_plan_times: torch.Tensor


def validate_planner_metadata(metadata: dict) -> PlannerContract:
    """Validate and freeze the exact planner geometry consumed by FastWAM."""
    for field, expected in EXPECTED_METADATA.items():
        actual = metadata.get(field)
        if actual != expected:
            raise ValueError(
                f"incompatible planner metadata field {field}: "
                f"expected {expected!r}, got {actual!r}"
            )

    expected_times = tuple(
        offset / (EXPECTED_METADATA["sequence_length"] - 1)
        for offset in EXPECTED_METADATA["keyframe_offsets"]
    )
    raw_times = metadata.get("normalized_keyframe_times", ())
    try:
        actual_times = tuple(float(value) for value in raw_times)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "incompatible planner metadata field normalized_keyframe_times: "
            f"expected {expected_times!r}, got {raw_times!r}"
        ) from error
    if len(actual_times) != len(expected_times) or any(
        not math.isfinite(actual) or abs(actual - expected) > 1e-7
        for actual, expected in zip(actual_times, expected_times, strict=True)
    ):
        raise ValueError(
            "incompatible planner metadata field normalized_keyframe_times: "
            f"expected {expected_times!r}, got {actual_times!r}"
        )

    expected_plan_tokens = tuple(f"<|sem_plan_{index}|>" for index in range(4 * 96))
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
        num_keyframes=4,
        grid_size=16,
        semantic_dim=1024,
        target_tokens=1024,
        shared_latent_per_keyframe=32,
        private_latent_per_keyframe=32,
        branch_latent_per_keyframe=64,
        total_unique_latent_per_keyframe=96,
        keyframe_offsets=(2, 4, 6, 8),
        normalized_keyframe_times=expected_times,
        plan_token_strings=expected_plan_tokens,
    )


def validate_checkpoint_files(checkpoint_dir: str | Path) -> Path:
    """Require the complete exported planner checkpoint before model loading."""
    checkpoint_dir = Path(checkpoint_dir)
    required_files = (
        "plan_head.pt",
        "depth_head.pt",
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
        from train_qwen3vl4b_lingbot_dino_planner import (
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
