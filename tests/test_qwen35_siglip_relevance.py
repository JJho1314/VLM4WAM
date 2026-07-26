from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from qwen35_planx.instruction import InstructionFields


class _FakeSiglipProcessor:
    def __call__(
        self,
        *,
        images: torch.Tensor,
        text: list[str],
        padding: str,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert images.shape[-2:] == (384, 384)
        assert padding == "max_length"
        assert return_tensors == "pt"
        tokens = torch.tensor(
            [[(sum(map(ord, phrase)) % 13) + 1] for phrase in text],
            dtype=torch.long,
        )
        return {
            "pixel_values": images,
            "input_ids": tokens,
            "attention_mask": torch.ones_like(tokens),
        }


class _FakeSiglip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.spatial_gradients: list[torch.Tensor] = []

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> SimpleNamespace:
        del attention_mask
        frames = pixel_values.shape[0]
        pattern = torch.linspace(
            0.2,
            1.2,
            729,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        ).view(1, 729, 1)
        image_signal = pixel_values.mean(dim=(1, 2, 3)).view(frames, 1, 1)
        spatial = pattern + image_signal
        spatial.register_hook(lambda gradient: self.spatial_gradients.append(gradient))

        phrase_signal = input_ids.float().mean().clamp_min(1.0)
        logits = spatial.squeeze(-1).mean(-1, keepdim=True) * phrase_signal
        embedding = torch.zeros(1, 1152, device=spatial.device)
        embedding[0, 0] = phrase_signal
        embedding[0, 1] = 1.0
        return SimpleNamespace(
            logits_per_image=logits,
            text_embeds=embedding,
            vision_model_output=SimpleNamespace(last_hidden_state=spatial),
            pooling_attention=torch.softmax(pattern.squeeze(-1), dim=-1).expand(
                frames, -1
            ),
        )


@pytest.fixture
def fake_siglip() -> SimpleNamespace:
    return SimpleNamespace(model=_FakeSiglip(), processor=_FakeSiglipProcessor())


def test_phrase_teacher_returns_normalized_dense_maps(fake_siglip) -> None:
    from qwen35_planx.siglip_relevance import SiglipRelevanceTeacher

    teacher = SiglipRelevanceTeacher.from_components(
        model=fake_siglip.model,
        processor=fake_siglip.processor,
    )
    output = teacher.encode(
        torch.zeros(2, 3, 256, 256),
        phrases=("pick up", "black bowl", "on the plate"),
    )

    assert output.phrase_embeddings.shape == (3, 1152)
    assert output.maps.shape == (2, 3, 27, 27)
    assert output.confidence.shape == (2, 3)
    torch.testing.assert_close(
        output.maps.flatten(-2).sum(-1),
        torch.ones(2, 3),
    )
    torch.testing.assert_close(
        output.phrase_embeddings.norm(dim=-1),
        torch.ones(3),
    )
    assert torch.all(output.confidence > 0)
    assert all(not parameter.requires_grad for parameter in teacher.model.parameters())


def test_relevance_backpropagates_only_to_captured_spatial_activations(
    fake_siglip,
) -> None:
    from qwen35_planx.siglip_relevance import SiglipRelevanceTeacher

    teacher = SiglipRelevanceTeacher.from_components(
        model=fake_siglip.model,
        processor=fake_siglip.processor,
    )
    teacher.encode(
        torch.zeros(1, 3, 384, 384),
        phrases=("pick up", "black bowl", "on the plate"),
    )

    assert len(fake_siglip.model.spatial_gradients) == 3
    assert all(torch.count_nonzero(gradient) for gradient in fake_siglip.model.spatial_gradients)
    assert all(parameter.grad is None for parameter in teacher.model.parameters())
    assert teacher.captured_spatial_activations is None


def test_empty_phrase_has_zero_embedding_map_and_confidence(fake_siglip) -> None:
    from qwen35_planx.siglip_relevance import SiglipRelevanceTeacher

    output = SiglipRelevanceTeacher.from_components(
        model=fake_siglip.model,
        processor=fake_siglip.processor,
    ).encode(
        torch.zeros(2, 3, 384, 384),
        phrases=("pick up", "", "on the plate"),
    )

    assert torch.count_nonzero(output.phrase_embeddings[1]) == 0
    assert torch.count_nonzero(output.maps[:, 1]) == 0
    assert torch.count_nonzero(output.confidence[:, 1]) == 0


def test_instruction_fields_are_encoded_in_canonical_role_order(fake_siglip) -> None:
    from qwen35_planx.siglip_relevance import SiglipRelevanceTeacher

    fields = InstructionFields(
        original="put the bowl on the plate",
        action="put",
        source="the bowl",
        target="on the plate",
        confidences=(1.0, 1.0, 1.0),
    )
    teacher = SiglipRelevanceTeacher.from_components(
        model=fake_siglip.model,
        processor=fake_siglip.processor,
    )

    direct = teacher.encode(
        torch.zeros(1, 3, 384, 384),
        phrases=("the bowl", "on the plate", "put"),
    )
    structured = teacher.encode_fields(
        torch.zeros(1, 3, 384, 384),
        fields,
    )

    torch.testing.assert_close(structured.phrase_embeddings, direct.phrase_embeddings)
    torch.testing.assert_close(structured.maps, direct.maps)


def test_nonfinite_teacher_output_is_sanitized_and_disables_confidence(
    fake_siglip,
) -> None:
    from qwen35_planx.siglip_relevance import SiglipRelevanceTeacher

    original_forward = fake_siglip.model.forward

    def nonfinite_forward(**inputs):
        output = original_forward(**inputs)
        output.text_embeds[:] = torch.nan
        return output

    fake_siglip.model.forward = nonfinite_forward
    output = SiglipRelevanceTeacher.from_components(
        model=fake_siglip.model,
        processor=fake_siglip.processor,
    ).encode(
        torch.zeros(1, 3, 384, 384),
        phrases=("pick up",),
    )

    assert torch.isfinite(output.phrase_embeddings).all()
    assert output.confidence.item() == 0.0
