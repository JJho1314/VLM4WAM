from __future__ import annotations

import torch
import torch.nn as nn

from qwen35_baton.query_tower import BatonVisualAlignmentTower


def _tiny_tower() -> BatonVisualAlignmentTower:
    torch.manual_seed(7)
    return BatonVisualAlignmentTower._from_test_config(
        qwen_dim=16,
        num_frames=2,
        tokens_per_frame=4,
        num_heads=4,
    )


def test_visual_alignment_tower_is_baton_query_cross_attention_only() -> None:
    tower = _tiny_tower()

    assert tower.learned_queries.shape == (2, 4, 16)
    assert isinstance(tower.cross_attention, nn.MultiheadAttention)
    assert not hasattr(tower, "blocks")
    assert not hasattr(tower, "allowed_mask")
    assert not hasattr(tower, "positions")
    assert not hasattr(tower, "camera_embeddings")
    assert not hasattr(tower, "context_projection")


def test_visual_alignment_tower_keeps_one_query_per_target_patch() -> None:
    tower = _tiny_tower()

    output = tower(torch.randn(3, 2, 4, 16))

    assert output.hidden_states.shape == (3, 2, 4, 16)
    assert output.cross_attention_maps is None


def test_visual_alignment_rows_are_independent() -> None:
    tower = _tiny_tower().eval()
    plan_states = torch.randn(3, 2, 4, 16)
    changed = plan_states.clone()
    changed[1].mul_(1000)

    baseline = tower(plan_states).hidden_states
    mutated = tower(changed).hidden_states

    torch.testing.assert_close(baseline[0], mutated[0])
    assert not torch.allclose(baseline[1], mutated[1])
    torch.testing.assert_close(baseline[2], mutated[2])


def test_each_query_can_cross_attend_all_plan_states() -> None:
    tower = _tiny_tower().eval()
    plan_states = torch.randn(1, 2, 4, 16, requires_grad=True)

    output = tower(plan_states).hidden_states
    output[:, 0, 0].square().sum().backward()

    context_gradient = plan_states.grad.abs().sum(dim=(0, 2, 3))
    assert torch.all(context_gradient > 0)


def test_attention_trace_is_one_detached_cross_attention_map() -> None:
    tower = _tiny_tower().eval()
    plan_states = torch.randn(2, 2, 4, 16)

    output = tower(plan_states, return_attention_maps=True)

    assert output.cross_attention_maps is not None
    assert len(output.cross_attention_maps) == 1
    attention = output.cross_attention_maps[0]
    assert attention.shape == (2, 8, 8)
    assert attention.requires_grad is False
    torch.testing.assert_close(
        attention.sum(dim=-1),
        torch.ones(2, 8),
        atol=1e-6,
        rtol=1e-6,
    )


def test_learned_queries_use_baton_normal_initialization() -> None:
    tower = BatonVisualAlignmentTower._from_test_config(
        qwen_dim=64,
        num_frames=4,
        tokens_per_frame=16,
        num_heads=8,
    )

    assert abs(float(tower.learned_queries.mean())) < 0.005
    assert 0.015 < float(tower.learned_queries.std()) < 0.025
