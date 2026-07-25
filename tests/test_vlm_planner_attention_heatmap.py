from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch
from PIL import Image

from qwen3_vl_semantic_planner.lingbot_dino_4b.lingbot_resampler import (
    PerceiverAttention,
)


MODULE = (
    "qwen3_vl_semantic_planner.dinov3_da3_2b."
    "visualize_vlm_planner_attention_dual_camera_k4"
)


def _visualizer():
    return importlib.import_module(MODULE)


def test_reconstructed_attention_reproduces_perceiver_output() -> None:
    visualizer = _visualizer()
    torch.manual_seed(7)
    module = PerceiverAttention(dim=16, dim_head=4, heads=2).eval()
    x = torch.randn(2, 9, 16)
    latents = torch.randn(2, 5, 16)

    weights, values = visualizer.reconstruct_perceiver_attention(
        module,
        x,
        latents,
    )
    manual = weights @ values
    manual = manual.permute(0, 2, 1, 3).reshape(2, 5, -1)
    manual = module.to_out(manual)

    torch.testing.assert_close(manual, module(x, latents))


def test_reduce_image_attention_excludes_non_image_columns() -> None:
    visualizer = _visualizer()
    weights = torch.zeros(1, 2, 3, 7)
    weights[..., :4] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    weights[..., 4:] = 1000.0

    reduced = visualizer.reduce_image_attention(
        weights,
        image_token_count=4,
        reduction="mean",
    )

    torch.testing.assert_close(
        reduced,
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
    )


def test_capture_registers_temporarily_and_records_cpu_maps() -> None:
    visualizer = _visualizer()
    module = PerceiverAttention(dim=16, dim_head=4, heads=2).eval()
    x = torch.randn(1, 9, 16)
    latents = torch.randn(1, 5, 16)

    capture = visualizer.PlannerAttentionCapture(
        module,
        image_token_count=6,
        reduction="max",
    )
    assert len(module._forward_hooks) == 0
    with capture:
        module(x, latents)
        assert len(module._forward_hooks) == 1
    assert len(module._forward_hooks) == 0
    assert len(capture.maps) == 1
    assert capture.maps[0].shape == (1, 6)
    assert capture.maps[0].device.type == "cpu"


def test_merged_image_grid_matches_qwen_spatial_merge() -> None:
    visualizer = _visualizer()

    assert visualizer.merged_image_grid(
        torch.tensor([1, 18, 18]),
        spatial_merge_size=2,
        expected_tokens=81,
    ) == (9, 9)


def test_merged_image_grid_rejects_token_mismatch() -> None:
    visualizer = _visualizer()

    with pytest.raises(ValueError, match="81"):
        visualizer.merged_image_grid(
            torch.tensor([1, 18, 18]),
            spatial_merge_size=2,
            expected_tokens=80,
        )


def test_joint_normalization_is_finite_and_shared_across_k4() -> None:
    visualizer = _visualizer()
    maps = torch.stack(
        [torch.arange(81).reshape(9, 9).float() + 10 * index for index in range(4)]
    )

    normalized = visualizer.normalize_attention_stack(
        maps,
        lower_quantile=0.02,
        upper_quantile=0.98,
    )

    assert normalized.shape == (4, 9, 9)
    assert torch.isfinite(normalized).all()
    assert normalized.min() >= 0
    assert normalized.max() <= 1
    assert normalized[0].mean() < normalized[-1].mean()


def test_attention_products_and_composite_are_rgb(tmp_path) -> None:
    visualizer = _visualizer()
    rgb = np.full((40, 48, 3), 128, dtype=np.uint8)
    heatmap, overlay = visualizer.attention_products(
        rgb,
        torch.linspace(0, 1, 9).reshape(3, 3),
        alpha=0.55,
    )

    assert heatmap.shape == rgb.shape
    assert overlay.shape == rgb.shape
    assert heatmap.dtype == np.uint8
    assert overlay.dtype == np.uint8

    output_path = tmp_path / "composite.png"
    observations = {"main": rgb, "wrist": rgb}
    overlays = {
        "main": [overlay.copy() for _ in range(4)],
        "wrist": [overlay.copy() for _ in range(4)],
    }
    visualizer.render_composite(
        output_path,
        instruction="pick up the bowl",
        observations=observations,
        overlays=overlays,
        offsets=[2, 4, 6, 8],
    )

    assert output_path.is_file()
    with Image.open(output_path) as image:
        assert image.mode == "RGB"
        assert image.width > image.height


def test_validate_checkpoint_contract_accepts_dual_camera_k4() -> None:
    visualizer = _visualizer()

    visualizer.validate_checkpoint_contract(
        {
            "plan_head_type": "lingbot_dino",
            "num_camera_views": 2,
            "camera_names": ["main", "wrist"],
            "num_keyframes": 4,
            "future_keyframe_offsets": [2, 4, 6, 8],
            "target_tokens_per_keyframe": 256,
            "branch_latent_per_keyframe": 64,
        }
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("plan_head_type", "mlp"),
        ("num_camera_views", 1),
        ("camera_names", ["main", "main"]),
        ("num_keyframes", 2),
        ("future_keyframe_offsets", [2, 4]),
    ],
)
def test_validate_checkpoint_contract_rejects_wrong_geometry(
    field: str,
    value: object,
) -> None:
    visualizer = _visualizer()
    metadata = {
        "plan_head_type": "lingbot_dino",
        "num_camera_views": 2,
        "camera_names": ["main", "wrist"],
        "num_keyframes": 4,
        "future_keyframe_offsets": [2, 4, 6, 8],
        "target_tokens_per_keyframe": 256,
        "branch_latent_per_keyframe": 64,
    }
    metadata[field] = value

    with pytest.raises(ValueError, match=field):
        visualizer.validate_checkpoint_contract(metadata)


def test_group_attention_captures_is_view_major_keyframe_major() -> None:
    visualizer = _visualizer()
    captures = [torch.full((1, 9), float(index)) for index in range(8)]

    grouped = visualizer.group_attention_captures(
        captures,
        num_views=2,
        num_keyframes=4,
    )

    assert grouped.shape == (2, 4, 9)
    assert grouped[0, 3, 0] == 3
    assert grouped[1, 0, 0] == 4


def test_group_attention_captures_rejects_missing_call() -> None:
    visualizer = _visualizer()

    with pytest.raises(ValueError, match="8"):
        visualizer.group_attention_captures(
            [torch.zeros(1, 9) for _ in range(7)],
            num_views=2,
            num_keyframes=4,
        )
