from __future__ import annotations

import importlib

import torch

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
