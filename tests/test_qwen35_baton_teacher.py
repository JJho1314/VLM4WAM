from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn as nn

from qwen35_baton.teacher import FrozenSiglip2Teacher


ARTIFACT = Path(__file__).resolve().parents[1] / "third_party/siglip2-large-patch16-256"


class FakeProcessor:
    def __call__(self, *, images: list[torch.Tensor], return_tensors: str) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        return {"pixel_values": torch.stack(images).float().div(255).mul(2).sub(1)}


class FakeVisionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.requested_hidden_states = False
        self.received_pixel_values: list[torch.Tensor] = []

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        output_hidden_states: bool,
    ) -> SimpleNamespace:
        self.requested_hidden_states = output_hidden_states
        self.received_pixel_values.append(pixel_values.detach().cpu().clone())
        patch_values = pixel_values.mean(dim=(1, 2, 3), keepdim=False)
        patches = patch_values[:, None, None].expand(-1, 256, 1024) * self.scale
        leading_token = torch.zeros_like(patches[:, :1])
        penultimate = patches + 2
        final = patches + 7
        return SimpleNamespace(
            hidden_states=(patches, torch.cat((leading_token, penultimate), dim=1), final)
        )


def make_teacher(*, frame_microbatch_size: int = 3) -> tuple[FrozenSiglip2Teacher, FakeVisionModel]:
    vision_model = FakeVisionModel()
    return (
        FrozenSiglip2Teacher.from_components(
            processor=FakeProcessor(),
            vision_model=vision_model,
            frame_microbatch_size=frame_microbatch_size,
        ),
        vision_model,
    )


def test_teacher_extracts_penultimate_patch_grid() -> None:
    teacher, vision_model = make_teacher()

    features = teacher.encode_future(torch.zeros((2, 2, 4, 3, 256, 256), dtype=torch.uint8))

    assert features.shape == (2, 2, 4, 256, 1024)
    assert vision_model.requested_hidden_states is True
    assert vision_model.training is False
    assert len(vision_model.received_pixel_values) == 6
    torch.testing.assert_close(features, torch.ones_like(features))


def test_teacher_targets_are_detached() -> None:
    teacher, _ = make_teacher()
    future_images = torch.randint(256, (1, 2, 4, 3, 256, 256), dtype=torch.uint8)
    current_images = torch.randint(256, (1, 2, 3, 256, 256), dtype=torch.uint8)

    future = teacher.encode_future(future_images)
    current = teacher.encode_current(current_images)

    assert future.requires_grad is False
    assert current.requires_grad is False
    assert all(parameter.requires_grad is False for parameter in teacher.model.parameters())


def test_teacher_to_keeps_input_device_in_sync_with_vision_model() -> None:
    teacher, _ = make_teacher()

    returned = teacher.to(torch.device("meta"))

    assert returned is teacher
    assert teacher.device == torch.device("meta")
    assert next(teacher.model.parameters()).device == torch.device("meta")


def test_teacher_loads_only_the_siglip2_vision_tower(monkeypatch) -> None:
    import transformers

    class FullModel(nn.Module):
        def __init__(self, vision_model: nn.Module) -> None:
            super().__init__()
            self.vision_model = vision_model
            self.text_model = nn.Linear(2, 2)

    vision_model = FakeVisionModel()
    full_model = FullModel(vision_model)
    monkeypatch.setattr(
        transformers.AutoImageProcessor,
        "from_pretrained",
        lambda *args, **kwargs: FakeProcessor(),
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: full_model,
    )

    teacher = FrozenSiglip2Teacher("unused", dtype=torch.float32)

    assert teacher.model is vision_model
    assert not hasattr(teacher, "full_model")
    assert all(parameter.requires_grad is False for parameter in teacher.model.parameters())


def test_teacher_accepts_normalized_minus_one_to_one_rgb() -> None:
    teacher, _ = make_teacher()
    uint8_images = torch.tensor([0, 255], dtype=torch.uint8).repeat(1, 2, 3, 256, 128)
    normalized_images = uint8_images.float().div(127.5).sub(1)

    uint8_features = teacher.encode_current(uint8_images)
    normalized_features = teacher.encode_current(normalized_images)

    torch.testing.assert_close(normalized_features, uint8_features)


def test_teacher_uses_persisted_siglip2_processor_pixel_values() -> None:
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(ARTIFACT, use_fast=False)
    vision_model = FakeVisionModel()
    teacher = FrozenSiglip2Teacher.from_components(
        processor=processor,
        vision_model=vision_model,
        frame_microbatch_size=2,
    )
    images = torch.stack(
        (
            torch.zeros((3, 256, 256), dtype=torch.uint8),
            torch.full((3, 256, 256), 255, dtype=torch.uint8),
        )
    )

    teacher._encode_frames(images)
    expected = processor(images=list(images), return_tensors="pt")["pixel_values"]

    assert len(vision_model.received_pixel_values) == 1
    torch.testing.assert_close(vision_model.received_pixel_values[0], expected)
