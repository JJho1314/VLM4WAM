from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


CAMERA_NAMES = ("main", "wrist")

PLANNER_USER_TEMPLATE = (
    "You are a robot video semantic planner. Given the main and wrist camera frames and "
    "instruction, predict future spatial semantic plan tokens for the manipulation video.\n"
    "Instruction: {instruction}"
)


def normalized_hwc_camera_frames_to_pil(
    frames: torch.Tensor,
) -> tuple[Image.Image, Image.Image]:
    if tuple(frames.shape[:1]) != (2,) or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"camera frames must be [2,H,W,3], got {tuple(frames.shape)}")
    if not torch.isfinite(frames).all() or frames.min() < -1.0001 or frames.max() > 1.0001:
        raise ValueError("camera frames must be finite and normalized to [-1,1]")
    rgb = ((frames.float().cpu() + 1.0) * 127.5).round().clamp(0, 255).byte()
    return tuple(Image.fromarray(frame.numpy()) for frame in rgb)


class GEActDualCameraPlannerDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        *,
        n_previous: int = 4,
        future_offset: int = 8,
        future_offsets: Sequence[int] | None = None,
    ):
        """Select one legacy future frame or K ordered future keyframes per camera."""
        self.dataset = dataset
        self.n_previous = int(n_previous)
        offsets = (
            (int(future_offset),)
            if future_offsets is None
            else tuple(int(offset) for offset in future_offsets)
        )
        if (
            not offsets
            or any(offset <= 0 for offset in offsets)
            or any(left >= right for left, right in zip(offsets, offsets[1:]))
        ):
            raise ValueError(
                "future_offsets must be strictly increasing positive integers, "
                f"got {offsets}"
            )
        self.future_offsets = offsets
        self.future_offset = offsets[-1]

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, _epoch: int) -> None:
        return None

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        video = sample["video"]
        future_indices = [self.n_previous + off for off in self.future_offsets]
        if video.ndim != 5 or video.shape[0] != 3 or video.shape[1] != 2:
            raise ValueError(
                f"GE-Act planner video must be [3,2,T,H,W], got {tuple(video.shape)}"
            )
        if future_indices[-1] >= video.shape[2]:
            raise ValueError(f"future index {future_indices[-1]} exceeds T={video.shape[2]}")
        current = video[:, :, self.n_previous - 1].permute(1, 2, 3, 0).contiguous()
        # Preserve the legacy [V,H,W,3] result for K=1. K>1 is view-major
        # [V,K,H,W,3], matching planner outputs [B,V,K*tokens,D].
        futures = [video[:, :, i].permute(1, 2, 3, 0).contiguous() for i in future_indices]
        future = futures[0] if len(futures) == 1 else torch.stack(futures, dim=1)
        return {
            "stem": f"geact_{index:09d}",
            "images": normalized_hwc_camera_frames_to_pil(current),
            "current_camera_images": current,
            "future_camera_images": future,
            "prompt": str(sample["caption"]),
        }


def build_dual_camera_planner_inputs(
    processor: Any,
    image_pairs: list[tuple[Any, Any]],
    instructions: list[Any],
    plan_sequence: str | list[str],
) -> Any:
    if len(image_pairs) != len(instructions):
        raise ValueError(
            f"images/instructions batch mismatch: {len(image_pairs)} != {len(instructions)}"
        )
    plan_text = plan_sequence if isinstance(plan_sequence, str) else "".join(plan_sequence)
    conversations = []
    flat_images = []
    for image_pair, instruction in zip(image_pairs, instructions, strict=True):
        if len(image_pair) != len(CAMERA_NAMES):
            raise ValueError(
                f"each sample must contain {len(CAMERA_NAMES)} camera images, "
                f"got {len(image_pair)}"
            )
        flat_images.extend(image_pair)
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Main camera:"},
                        {"type": "image"},
                        {"type": "text", "text": "Wrist camera:"},
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": PLANNER_USER_TEMPLATE.format(
                                instruction=str(instruction)
                            ),
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": plan_text,
                },
            ]
        )
    texts = [
        processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        for conversation in conversations
    ]
    return processor(
        text=texts,
        images=flat_images,
        padding=True,
        return_tensors="pt",
    )


@dataclass
class DualCameraPlannerCollator:
    processor: Any
    plan_sequence: list[str]

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        inputs = build_dual_camera_planner_inputs(
            self.processor,
            [item["images"] for item in batch],
            [item["prompt"] for item in batch],
            self.plan_sequence,
        )
        inputs["current_camera_images"] = torch.stack(
            [item["current_camera_images"] for item in batch], dim=0
        )
        inputs["future_camera_images"] = torch.stack(
            [item["future_camera_images"] for item in batch], dim=0
        )
        inputs["stems"] = [item["stem"] for item in batch]
        return inputs
