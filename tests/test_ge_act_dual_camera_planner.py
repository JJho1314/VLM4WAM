from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from PIL import Image
from torch.utils.data import Dataset

from qwen3_vl_semantic_planner.ge_act_dual_camera import (
    DualCameraPlannerCollator,
    GEActDualCameraPlannerDataset,
    build_dual_camera_planner_inputs,
)

PLANNER_ROOT = Path(__file__).resolve().parents[1] / "qwen3_vl_semantic_planner"
if str(PLANNER_ROOT) not in sys.path:
    sys.path.insert(0, str(PLANNER_ROOT))

from qwen3_vl_semantic_planner.train_qwen3vl4b_lingbot_dino_planner import (
    PlannerWrapper,
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


class ViewAwareHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(
        self,
        image_hidden: torch.Tensor,
        task_hidden: torch.Tensor,
    ) -> torch.Tensor:
        batch = image_hidden.shape[0]
        view_value = image_hidden.mean(dim=(1, 2)).reshape(batch, 1, 1)
        return self.scale * view_value.expand(batch, 256, 1024)


def make_fake_alignment_wrapper(*, num_camera_views: int) -> PlannerWrapper:
    wrapper = PlannerWrapper.__new__(PlannerWrapper)
    nn.Module.__init__(wrapper)
    wrapper.use_current_alignment = True
    wrapper.independent_modality_task_tokens = True
    wrapper.num_task_tokens = 64
    wrapper.latent_len = 4 * wrapper.num_task_tokens
    wrapper.num_camera_views = num_camera_views
    wrapper.da3_align_strategy = "last_layer"
    wrapper.current_plan_head = ViewAwareHead()
    wrapper.plan_head = ViewAwareHead()
    wrapper.current_depth_head = ViewAwareHead()
    wrapper.depth_head = ViewAwareHead()

    def forward_hiddens(**inputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        batch = inputs["input_ids"].shape[0]
        if num_camera_views == 1:
            image_hidden = torch.zeros(batch, 2, 4)
        else:
            image_hidden = torch.stack(
                [torch.zeros(batch, 2, 4), torch.full((batch, 2, 4), 10.0)],
                dim=1,
            )
        task_hidden = torch.zeros(batch, wrapper.latent_len, 4)
        return image_hidden, task_hidden

    wrapper._forward_hiddens = forward_hiddens
    return wrapper


def make_loss_only_wrapper_with_unit_branch_weights() -> PlannerWrapper:
    wrapper = PlannerWrapper.__new__(PlannerWrapper)
    nn.Module.__init__(wrapper)
    wrapper.da3_align_strategy = "last_layer"
    wrapper.current_dino_loss_weight = 1.0
    wrapper.future_dino_loss_weight = 1.0
    wrapper.current_depth_loss_weight = 1.0
    wrapper.future_depth_loss_weight = 1.0
    return wrapper


def make_four_branch_plans(
    *,
    main_value: float,
    wrist_value: float,
) -> dict[str, torch.Tensor]:
    values = torch.tensor([main_value, wrist_value]).reshape(1, 2, 1, 1)
    plans = values.expand(1, 2, 256, 4).clone()
    return {
        "current_dino": plans.clone(),
        "future_dino": plans.clone(),
        "current_depth": plans.clone(),
        "future_depth": plans.clone(),
    }


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


def test_collect_image_hidden_keeps_two_contiguous_camera_spans() -> None:
    wrapper = PlannerWrapper.__new__(PlannerWrapper)
    wrapper.image_token_id = 99
    hidden = torch.arange(1 * 10 * 3).reshape(1, 10, 3).float()
    input_ids = torch.tensor([[1, 99, 99, 2, 3, 99, 99, 4, 5, 6]])

    actual = wrapper.collect_image_hidden_by_view(hidden, input_ids, num_views=2)

    assert actual.shape == (1, 2, 2, 3)
    torch.testing.assert_close(actual[0, 0], hidden[0, 1:3])
    torch.testing.assert_close(actual[0, 1], hidden[0, 5:7])


def test_dual_camera_wrapper_reuses_four_query_groups_for_both_views() -> None:
    wrapper = make_fake_alignment_wrapper(num_camera_views=2)

    plans = wrapper.predict_current_future_plans(
        input_ids=torch.ones(2, 1, dtype=torch.long)
    )

    assert wrapper.latent_len == 4 * 64
    assert set(plans) == {
        "current_dino",
        "future_dino",
        "current_depth",
        "future_depth",
    }
    assert all(value.shape == (2, 2, 256, 1024) for value in plans.values())
    assert not torch.equal(plans["future_dino"][:, 0], plans["future_dino"][:, 1])


def test_single_camera_wrapper_keeps_legacy_output_shape() -> None:
    wrapper = make_fake_alignment_wrapper(num_camera_views=1)

    plans = wrapper.predict_current_future_plans(
        input_ids=torch.ones(2, 1, dtype=torch.long)
    )

    assert all(value.shape == (2, 256, 1024) for value in plans.values())


def test_dual_camera_loss_detects_swapped_teacher_views() -> None:
    wrapper = make_loss_only_wrapper_with_unit_branch_weights()
    plans = make_four_branch_plans(main_value=0.0, wrist_value=10.0)
    aligned = make_four_branch_plans(main_value=0.0, wrist_value=10.0)
    swapped = {name: value.flip(1) for name, value in aligned.items()}

    aligned_loss = wrapper.compute_current_future_losses(plans, aligned)["loss"]
    swapped_loss = wrapper.compute_current_future_losses(plans, swapped)["loss"]

    assert aligned_loss == 0
    assert swapped_loss > 0


def test_dual_camera_forward_validates_target_tokens_before_view_dimension() -> None:
    wrapper = make_loss_only_wrapper_with_unit_branch_weights()
    wrapper.plan_head_type = "lingbot_dino"
    wrapper.use_current_alignment = True
    wrapper.target_len = 256
    plans = make_four_branch_plans(main_value=0.0, wrist_value=10.0)
    wrapper.predict_current_future_plans = lambda **_inputs: plans

    result = wrapper(
        semantic_plan_labels=plans["future_dino"],
        depth_plan_labels=plans["future_depth"],
        current_dino_labels=plans["current_dino"],
        current_depth_labels=plans["current_depth"],
    )

    assert result["loss"] == 0


def test_wrapper_rejects_unsupported_camera_view_count() -> None:
    with pytest.raises(ValueError, match="num_camera_views must be 1 or 2"):
        PlannerWrapper(
            model=nn.Module(),
            hidden_size=4,
            semantic_dim=2,
            plan_token_ids=[1],
            target_len=1,
            num_keyframes=1,
            grid_size=1,
            num_camera_views=3,
        )
