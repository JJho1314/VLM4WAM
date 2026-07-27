from __future__ import annotations

import pytest
import torch

from qwen35_baton.query_tower import (
    RotaryPosition3D,
    SpatiotemporalQueryTower,
    build_block_causal_allowed_mask,
    build_spatiotemporal_positions,
)


def _tiny_tower() -> SpatiotemporalQueryTower:
    torch.manual_seed(7)
    return SpatiotemporalQueryTower._from_test_config(
        qwen_dim=32,
        query_dim=16,
        num_frames=4,
        tokens_per_frame=4,
        num_heads=4,
        ffn_dim=32,
        dropout=0.0,
    )


def test_block_causal_mask_allows_same_and_past_frames_only() -> None:
    allowed = build_block_causal_allowed_mask(num_frames=4, tokens_per_frame=3)

    assert allowed.dtype == torch.bool
    assert allowed.shape == (12, 12)
    assert allowed[4, :6].all()
    assert not allowed[4, 6:].any()
    assert allowed[0, :3].all()
    assert not allowed[0, 3:].any()


def test_spatiotemporal_positions_are_frame_then_row_major_yx() -> None:
    positions = build_spatiotemporal_positions(num_frames=2, tokens_per_frame=4)

    torch.testing.assert_close(
        positions,
        torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=torch.float32,
        ),
    )


def test_3d_rope_rotates_independent_temporal_y_and_x_bands() -> None:
    rope = RotaryPosition3D(head_dim=6, rotary_dimensions=(2, 2, 2))
    values = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).repeat(1, 1, 4, 1)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    sine = torch.sin(torch.tensor(1.0))
    cosine = torch.cos(torch.tensor(1.0))

    rotated = rope(values, positions)

    torch.testing.assert_close(
        rotated[0, 0],
        torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [cosine, sine, 1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, cosine, sine, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, cosine, sine],
            ]
        ),
    )


def test_query_tower_keeps_one_query_per_target_patch() -> None:
    tower = _tiny_tower()

    output = tower(
        torch.randn(4, 4, 4, 32),
        camera_ids=torch.tensor([0, 1, 0, 1]),
    )

    assert output.hidden_states.shape == (4, 4, 4, 16)
    assert output.cross_attention_maps is None


def test_rows_are_isolated_even_when_camera_id_matches() -> None:
    tower = _tiny_tower().eval()
    qwen_states = torch.randn(4, 4, 4, 32)
    camera_ids = torch.tensor([0, 1, 0, 1])
    changed = qwen_states.clone()
    changed[2].mul_(1000)

    actual = tower(qwen_states, camera_ids)
    mutated = tower(changed, camera_ids)

    torch.testing.assert_close(actual.hidden_states[0], mutated.hidden_states[0])
    assert not torch.allclose(actual.hidden_states[2], mutated.hidden_states[2])


def test_frame_one_gradients_reach_only_same_and_past_context_frames() -> None:
    tower = _tiny_tower().eval()
    context = torch.randn(1, 4, 4, 32, requires_grad=True)

    output = tower(context, camera_ids=torch.tensor([0]))
    output.hidden_states[:, 1].square().sum().backward()

    frame_gradient = context.grad.abs().sum(dim=(0, 2, 3))
    assert frame_gradient[0] > 0
    assert frame_gradient[1] > 0
    torch.testing.assert_close(frame_gradient[2:], torch.zeros(2))


def test_attention_traces_are_detached_head_means_and_only_opt_in() -> None:
    tower = _tiny_tower().eval()
    context = torch.randn(2, 4, 4, 32, requires_grad=True)

    traced = tower(
        context,
        camera_ids=torch.tensor([0, 1]),
        return_attention_maps=True,
    )

    assert traced.cross_attention_maps is not None
    assert len(traced.cross_attention_maps) == 4
    for layer_map in traced.cross_attention_maps:
        assert layer_map.shape == (2, 16, 16)
        assert not layer_map.requires_grad
        torch.testing.assert_close(
            layer_map.sum(dim=-1),
            torch.ones(2, 16),
            atol=1e-6,
            rtol=1e-6,
        )
        assert not layer_map[:, :4, 4:].any()

    untraced = tower(context, camera_ids=torch.tensor([0, 1]))
    assert untraced.cross_attention_maps is None


def test_reduced_test_towers_cannot_serialize_as_checkpoints() -> None:
    tower = _tiny_tower()

    with pytest.raises(RuntimeError, match="test-config.*checkpoint"):
        tower.state_dict()
