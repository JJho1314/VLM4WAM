from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from qwen3_vl_semantic_planner.ge_act_dual_camera import (
    DualCameraPlannerCollator,
    GEActDualCameraPlannerDataset,
    build_dual_camera_planner_inputs,
)


class FakeDataset(Dataset):
    def __init__(self, sample: dict[str, Any]):
        self.sample = sample

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index != 0:
            raise IndexError(index)
        return self.sample


class RecordingProcessor:
    def __init__(self) -> None:
        self.images: list[Image.Image] = []
        self.texts: list[str] = []
        self.rendered_conversations: list[str] = []

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        assert not add_generation_prompt
        rendered: list[str] = []
        for message in conversation:
            content = message["content"]
            if isinstance(content, str):
                rendered.append(content)
                continue
            for part in content:
                if part["type"] == "image":
                    rendered.append("<|image_pad|>")
                else:
                    rendered.append(part["text"])
        rendered_conversation = "\n".join(rendered)
        self.rendered_conversations.append(rendered_conversation)
        return rendered_conversation

    def __call__(
        self,
        *,
        text: list[str],
        images: list[Image.Image],
        padding: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert padding
        assert return_tensors == "pt"
        self.texts = text
        self.images = images
        return {"input_ids": torch.arange(len(text)).unsqueeze(1)}


def test_ge_act_adapter_selects_current_and_future_endpoint_without_concat() -> None:
    video = torch.zeros(3, 2, 13, 2, 2)
    video[:, 0, 3].fill_(-0.5)
    video[:, 1, 3].fill_(0.0)
    video[:, 0, 12].fill_(0.5)
    video[:, 1, 12].fill_(1.0)
    wrapped = GEActDualCameraPlannerDataset(
        FakeDataset({"video": video, "caption": "pick the cup"}),
        n_previous=4,
        future_offset=8,
    )

    item = wrapped[0]

    assert item["current_camera_images"].shape == (2, 2, 2, 3)
    assert item["future_camera_images"].shape == (2, 2, 2, 3)
    torch.testing.assert_close(
        item["future_camera_images"][0],
        torch.full((2, 2, 3), 0.5),
    )
    torch.testing.assert_close(
        item["future_camera_images"][1],
        torch.full((2, 2, 3), 1.0),
    )
    assert item["images"][0].getpixel((0, 0))[0] == 64
    assert item["images"][1].getpixel((0, 0))[0] == 128
    assert item["prompt"] == "pick the cup"


def test_dual_camera_input_builder_flattens_images_main_then_wrist() -> None:
    processor = RecordingProcessor()
    main = Image.new("RGB", (2, 2), "red")
    wrist = Image.new("RGB", (2, 2), "blue")

    build_dual_camera_planner_inputs(
        processor,
        [(main, wrist)],
        ["pick"],
        ["<|sem_plan_0|>"],
    )

    assert processor.images == [main, wrist]
    assert processor.texts[0].count("<|image_pad|>") == 2
    assert "Main camera" in processor.rendered_conversations[0]
    assert "Wrist camera" in processor.rendered_conversations[0]


def test_dual_camera_collator_stacks_targets_and_keeps_sample_major_image_order() -> None:
    processor = RecordingProcessor()
    main_0 = Image.new("RGB", (2, 2), "red")
    wrist_0 = Image.new("RGB", (2, 2), "blue")
    main_1 = Image.new("RGB", (2, 2), "green")
    wrist_1 = Image.new("RGB", (2, 2), "yellow")
    batch = [
        {
            "images": (main_0, wrist_0),
            "current_camera_images": torch.full((2, 2, 2, 3), 1.0),
            "future_camera_images": torch.full((2, 2, 2, 3), 2.0),
            "prompt": "pick",
            "stem": "geact_000000000",
        },
        {
            "images": (main_1, wrist_1),
            "current_camera_images": torch.full((2, 2, 2, 3), 3.0),
            "future_camera_images": torch.full((2, 2, 2, 3), 4.0),
            "prompt": "place",
            "stem": "geact_000000001",
        },
    ]

    result = DualCameraPlannerCollator(
        processor=processor,
        plan_sequence=["<|sem_plan_0|>"],
    )(batch)

    assert processor.images == [main_0, wrist_0, main_1, wrist_1]
    assert result["current_camera_images"].shape == (2, 2, 2, 2, 3)
    assert result["future_camera_images"].shape == (2, 2, 2, 2, 3)
    assert result["current_camera_images"][:, 0, 0, 0, 0].tolist() == [1.0, 3.0]
    assert result["future_camera_images"][:, 1, 0, 0, 0].tolist() == [2.0, 4.0]
    assert result["stems"] == ["geact_000000000", "geact_000000001"]
