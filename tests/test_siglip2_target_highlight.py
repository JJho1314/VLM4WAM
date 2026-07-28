import hashlib
import json
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
from qwen3_vl_semantic_planner.dinov3_da3_2b.generate_siglip_target_highlight_comparison import (
    active_target,
    combine_target_highlight,
    generate_comparison,
    permute_palette,
)


def test_active_target_switches_at_frame_128():
    assert active_target(112) == "the white textured mug"
    assert active_target(127) == "the white textured mug"
    assert active_target(128) == "the yellow and white mug"
    assert active_target(160) == "the yellow and white mug"


def test_palette_candidates_are_exact_channel_permutations():
    pixel = np.array([[[10, 20, 30]]], dtype=np.uint8)
    np.testing.assert_array_equal(
        permute_palette(pixel, (0, 1, 2)),
        np.array([[[10, 20, 30]]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        permute_palette(pixel, (1, 2, 0)),
        np.array([[[20, 30, 10]]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        permute_palette(pixel, (2, 0, 1)),
        np.array([[[30, 10, 20]]], dtype=np.uint8),
    )


def test_combined_highlight_preserves_shape_and_emphasizes_target():
    feature = np.full((8, 8, 3), [40, 120, 200], dtype=np.uint8)
    relevance = np.zeros((8, 8), dtype=np.float32)
    relevance[2:6, 2:6] = 1.0
    combined = combine_target_highlight(feature, relevance)
    assert combined.shape == (8, 8, 3)
    assert combined.dtype == np.uint8
    assert combined[3, 3].sum() > combined[0, 0].sum()
    assert combined[2, 2, 0] > combined[0, 0, 0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeHighlighter:
    def __init__(self) -> None:
        self.phrases = None

    def __call__(self, images, phrases):
        self.phrases = list(phrases)
        maps = np.zeros((len(images), 256, 256), dtype=np.float32)
        maps[:, 64:192, 64:192] = 1.0
        return maps


def test_generate_comparison_uses_all_cameras_and_preserves_export_sources(tmp_path):
    export_root = tmp_path / "export"
    source_paths = []
    for frame_index, value in ((112, 40), (160, 160)):
        for camera_index, camera in enumerate(("main", "wrist")):
            source_dir = export_root / camera / f"frame_{frame_index:06d}"
            source_dir.mkdir(parents=True)
            rgb = source_dir / "rgb.png"
            probe = source_dir / "siglip_probe.png"
            Image.fromarray(np.full((512, 512, 3), value, dtype=np.uint8)).save(rgb)
            Image.fromarray(
                np.full((256, 256, 3), value + camera_index + 1, dtype=np.uint8)
            ).save(probe)
            source_paths.extend((rgb, probe))

    hashes_before = {path: _sha256(path) for path in source_paths}
    paths_before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    output_dir = tmp_path / "output"
    highlighter = _FakeHighlighter()

    png_path, json_path = generate_comparison(
        export_root,
        tmp_path / "siglip2-model",
        output_dir,
        torch.device("cpu"),
        highlighter=highlighter,
    )

    assert highlighter.phrases == [
        "the white textured mug",
        "the white textured mug",
        "the yellow and white mug",
        "the yellow and white mug",
    ]
    with Image.open(png_path) as output:
        assert output.size == (1096, 1104)
        assert output.mode == "RGB"
    metadata = json.loads(json_path.read_text())
    assert metadata["model_path"] == str(tmp_path / "siglip2-model")
    assert metadata["phase_boundary"] == 128
    assert metadata["frames"] == [112, 160]
    assert metadata["cameras"] == ["main", "wrist"]
    assert metadata["phrases"] == highlighter.phrases
    assert metadata["quantiles"] == {"low": 0.05, "high": 0.95}
    assert metadata["palettes"] == {
        "A_current": [0, 1, 2],
        "B_warm_balanced": [1, 2, 0],
        "C_cool_balanced": [2, 0, 1],
    }
    assert metadata["panel_order"] == [
        {"frame": 112, "camera": "main", "phrase": "the white textured mug"},
        {"frame": 112, "camera": "wrist", "phrase": "the white textured mug"},
        {"frame": 160, "camera": "main", "phrase": "the yellow and white mug"},
        {"frame": 160, "camera": "wrist", "phrase": "the yellow and white mug"},
    ]
    assert {path: _sha256(path) for path in source_paths} == hashes_before
    paths_after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert set(paths_after) - set(paths_before) == {
        Path("output"),
        Path("output/siglip_target_highlight_palettes.png"),
        Path("output/siglip_target_highlight_palettes.json"),
    }


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
