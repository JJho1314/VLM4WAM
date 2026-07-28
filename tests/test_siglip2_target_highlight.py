from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from qwen3_vl_semantic_planner.dinov3_da3_2b.siglip2_target_highlight import (
    normalize_relevance,
    SiglipPairGradCAM,
    token_gradcam,
)


def test_normalize_relevance_uses_fixed_quantiles_and_handles_zero_range():
    cam = torch.arange(100, dtype=torch.float32).reshape(1, 10, 10)
    normalized = normalize_relevance(cam, low_q=0.05, high_q=0.95)
    assert normalized.shape == (1, 10, 10)
    assert normalized.min().item() == 0.0
    assert normalized.max().item() == 1.0
    torch.testing.assert_close(
        normalize_relevance(torch.ones(1, 4, 4)),
        torch.zeros(1, 4, 4),
    )


def test_token_gradcam_keeps_positive_activation_gradient_product():
    tokens = torch.zeros(1, 4, 2)
    gradients = torch.zeros_like(tokens)
    tokens[0, 3] = torch.tensor([2.0, 1.0])
    gradients[0, 3] = torch.tensor([1.0, 2.0])
    cam = token_gradcam(
        tokens,
        gradients,
        grid_size=2,
        output_size=2,
    )
    assert cam.shape == (1, 2, 2)
    assert cam[0, 1, 1] == 1.0
    assert torch.count_nonzero(cam) == 1


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, *, text, images, padding, return_tensors):
        self.calls.append((text, images, padding, return_tensors))
        return {
            "pixel_values": torch.ones(len(images), 3, 1, 1),
            "input_ids": torch.arange(len(text)).view(-1, 1),
            "attention_mask": torch.ones(len(text), 1, dtype=torch.long),
        }


class _FakePairwiseSiglip(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unused_parameter = torch.nn.Parameter(torch.ones(1))
        self.vision_model = SimpleNamespace(
            config=SimpleNamespace(image_size=256, patch_size=16, hidden_size=1024)
        )
        self.text_model = SimpleNamespace()
        pattern = torch.zeros(1, 256, 1024)
        for row in range(4):
            for column in range(4):
                pattern[0, row * 16 + column, 0] = 1.0
                pattern[0, (15 - row) * 16 + 15 - column, 1] = 1.0
        self.register_buffer("token_pattern", pattern)
        self.forward_kwargs = None

    def forward(
        self,
        *,
        pixel_values,
        input_ids,
        attention_mask,
        output_hidden_states,
        return_dict,
    ):
        del attention_mask
        self.forward_kwargs = {
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
        tokens = pixel_values[:, :1, 0, 0].view(-1, 1, 1) * self.token_pattern
        visual_features = tokens.sum(dim=1)[:, :2]
        text_features = torch.eye(2, device=tokens.device)[input_ids[:, 0]]
        logits_per_image = visual_features @ text_features.T
        decoy_final_layer = tokens * 0
        return SimpleNamespace(
            logits_per_image=logits_per_image,
            vision_model_output=SimpleNamespace(
                hidden_states=(torch.zeros_like(tokens), tokens, decoy_final_layer)
            ),
        )


def test_siglip_pair_gradcam_backpropagates_each_pairwise_diagonal_score():
    processor = _FakeProcessor()
    model = _FakePairwiseSiglip()
    highlighter = SiglipPairGradCAM(
        Path("unused-fake-checkpoint"),
        torch.device("cpu"),
        model=model,
        processor=processor,
    )
    images = [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]

    maps = highlighter(images, ["first", "second"])

    assert maps.shape == (2, 256, 256)
    assert maps.dtype == np.float32
    assert maps[0, 0, 0] == 1.0
    assert maps[0, -1, -1] == 0.0
    assert maps[1, 0, 0] == 0.0
    assert maps[1, -1, -1] == 1.0
    assert model.forward_kwargs == {"output_hidden_states": True, "return_dict": True}
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert processor.calls[0][2:] == ("max_length", "pt")

    with pytest.raises(ValueError, match="same length"):
        highlighter(images, ["only one phrase"])
