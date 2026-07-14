import importlib.util
from pathlib import Path

import pytest
import torch


def _load_module():
    module_path = Path(__file__).with_name("semantic_plan_conditioning.py")
    spec = importlib.util.spec_from_file_location("semantic_plan_conditioning_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_infers_keyframe_grid_shape_from_flat_tokens():
    module = _load_module()

    assert module.infer_semantic_plan_shape(
        num_tokens=16 * 9 * 9,
        num_keyframes=16,
        spatial_grid_size=9,
    ) == (16, 9, 9)


def test_builds_time_major_shared_3d_coordinates():
    module = _load_module()

    coords = module.build_semantic_plan_coords(
        batch_size=2,
        num_tokens=2 * 3 * 3,
        num_keyframes=2,
        spatial_grid_size=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert coords.shape == (2, 18, 3)
    assert torch.allclose(coords[0, 0], torch.tensor([-1.0, -1.0, -1.0]))
    assert torch.allclose(coords[0, 8], torch.tensor([-1.0, 1.0, 1.0]))
    assert torch.allclose(coords[0, 9], torch.tensor([1.0, -1.0, -1.0]))
    assert torch.allclose(coords[0, -1], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(coords[0], coords[1])


def test_selects_uniform_keyframes_from_dense_plan():
    module = _load_module()
    source_n = 16
    target_n = 8
    grid = 2
    spatial_tokens = grid * grid
    semantic_plan = torch.arange(source_n, dtype=torch.float32).view(1, source_n, 1, 1)
    semantic_plan = semantic_plan.expand(1, source_n, spatial_tokens, 1).reshape(1, source_n * spatial_tokens, 1)

    selected = module.select_semantic_plan_keyframes(
        semantic_plan,
        num_keyframes=target_n,
        source_num_keyframes=source_n,
        spatial_grid_size=grid,
    )

    selected_times = selected.reshape(1, target_n, spatial_tokens, 1)[0, :, 0, 0]
    assert selected.shape == (1, target_n * spatial_tokens, 1)
    assert torch.equal(selected_times, torch.tensor([0, 2, 4, 6, 9, 11, 13, 15], dtype=torch.float32))


def test_selects_native_grid_keyframes_using_source_count():
    module = _load_module()
    source_n = 4
    target_n = 2
    spatial_tokens = 9  # native (non-configured) grid: inferred from source keyframe count
    semantic_plan = torch.arange(source_n, dtype=torch.float32).view(1, source_n, 1, 1)
    semantic_plan = semantic_plan.expand(1, source_n, spatial_tokens, 1).reshape(1, source_n * spatial_tokens, 1)

    selected = module.select_semantic_plan_keyframes(
        semantic_plan,
        num_keyframes=target_n,
        source_num_keyframes=source_n,
        spatial_grid_size=0,
    )

    selected_times = selected.reshape(1, target_n, spatial_tokens, 1)[0, :, 0, 0]
    assert selected.shape == (1, target_n * spatial_tokens, 1)
    assert torch.equal(selected_times, torch.tensor([0.0, 3.0]))


def test_selection_carries_keyframe_times_with_same_indices():
    module = _load_module()
    source_n = 16
    target_n = 8
    grid = 2
    spatial_tokens = grid * grid
    semantic_plan = torch.randn(1, source_n * spatial_tokens, 3)
    times = torch.linspace(0.1, 1.0, source_n).unsqueeze(0)

    selected, selected_times = module.select_semantic_plan_keyframes_and_times(
        semantic_plan,
        times,
        num_keyframes=target_n,
        source_num_keyframes=source_n,
        spatial_grid_size=grid,
    )

    expected_indices = torch.tensor([0, 2, 4, 6, 9, 11, 13, 15])
    assert selected.shape == (1, target_n * spatial_tokens, 3)
    assert torch.allclose(selected_times[0], times[0, expected_indices])


def test_rope_positions_use_explicit_keyframe_times():
    module = _load_module()
    times = torch.tensor([[0.25, 1.0]])

    positions = module.build_semantic_plan_rope_positions(
        batch_size=1,
        num_tokens=2 * 2 * 2,
        num_keyframes=2,
        spatial_grid_size=2,
        latent_t=13,
        latent_h=5,
        latent_w=5,
        keyframe_times_N=times,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    t_positions = positions[0, :, 0].reshape(2, 4)
    assert torch.allclose(t_positions[0], torch.full((4,), 0.25 * 12))
    assert torch.allclose(t_positions[1], torch.full((4,), 12.0))


def test_rope_positions_fall_back_to_uniform_on_sentinel_times():
    module = _load_module()
    sentinel = torch.full((1, 2), -1.0)

    positions = module.build_semantic_plan_rope_positions(
        batch_size=1,
        num_tokens=2 * 2 * 2,
        num_keyframes=2,
        spatial_grid_size=2,
        latent_t=13,
        latent_h=5,
        latent_w=5,
        keyframe_times_N=sentinel,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    t_positions = positions[0, :, 0].reshape(2, 4)
    assert torch.allclose(t_positions[0], torch.zeros(4))
    assert torch.allclose(t_positions[1], torch.full((4,), 12.0))


def test_coords_use_explicit_keyframe_times():
    module = _load_module()
    times = torch.tensor([[0.5, 1.0]])

    coords = module.build_semantic_plan_coords(
        batch_size=2,
        num_tokens=2 * 2 * 2,
        num_keyframes=2,
        spatial_grid_size=2,
        keyframe_times_B_N=times,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert coords.shape == (2, 8, 3)
    assert torch.allclose(coords[0, :4, 0], torch.zeros(4))  # 0.5 -> 0.0 in [-1,1]
    assert torch.allclose(coords[0, 4:, 0], torch.ones(4))
    assert torch.allclose(coords[0], coords[1])


def test_semantic_plan_times_from_frame_indices():
    module = _load_module()

    times = module.semantic_plan_times_from_frame_indices([11, 51, 101], list(range(1, 102)))
    assert torch.allclose(times, torch.tensor([0.1, 0.5, 1.0]))

    assert module.semantic_plan_times_from_frame_indices([], [0, 1]) is None
    assert module.semantic_plan_times_from_frame_indices([1, 2], None) is None
    assert module.semantic_plan_times_from_frame_indices([1, 2], [5]) is None


def test_rejects_ambiguous_non_square_spatial_tokens():
    module = _load_module()

    with pytest.raises(ValueError, match="square spatial grid"):
        module.infer_semantic_plan_shape(
            num_tokens=20,
            num_keyframes=2,
            spatial_grid_size=0,
        )


def test_context_adapter_injects_position_into_equal_semantic_features():
    module = _load_module()
    adapter = module.SemanticPlanContextAdapter(
        in_dim=4,
        hidden_dim=16,
        out_dim=8,
        num_keyframes=2,
        spatial_grid_size=2,
        coord_hidden_dim=16,
    )

    semantic_plan = torch.ones(1, 2 * 2 * 2, 4)
    tokens, valid, times = adapter(semantic_plan)

    assert valid.all()
    assert times is None
    assert tokens.shape == (1, 8, 8)
    assert not torch.allclose(tokens[:, 0], tokens[:, -1])


def test_context_adapter_selects_and_returns_keyframe_times():
    module = _load_module()
    adapter = module.SemanticPlanContextAdapter(
        in_dim=4,
        hidden_dim=16,
        out_dim=8,
        num_keyframes=2,
        source_num_keyframes=4,
        spatial_grid_size=2,
        coord_hidden_dim=16,
    )

    semantic_plan = torch.ones(1, 4 * 2 * 2, 4)
    input_times = torch.tensor([[0.1, 0.4, 0.7, 1.0]])
    tokens, valid, times = adapter(semantic_plan, keyframe_times_B_N=input_times)

    assert valid.all()
    assert tokens.shape == (1, 8, 8)
    assert torch.allclose(times[0], torch.tensor([0.1, 1.0]))


def test_minimal_v4_keeps_semantic_branch_without_extra_residual_write():
    source = Path(__file__).with_name("minimal_v4_dit.py").read_text()

    assert "self.semantic_plan_cross_attn = semantic_attn_cls" in source
    assert "append_semantic_plan_context" not in source
    assert "x_B_T_H_W_D = x_B_T_H_W_D + semantic_result_B_T_H_W_D" not in source


def test_minimal_v4_uses_baton_style_independent_semantic_cross_attention_gate():
    source = Path(__file__).with_name("minimal_v4_dit.py").read_text()

    assert "self.adaln_modulation_semantic_plan_cross_attn" in source
    assert "gate_semantic_plan_cross_attn_B_T_1_1_D * semantic_result_B_T_H_W_D" in source
    assert "cross_result_B_T_H_W_D = cross_result_B_T_H_W_D + semantic_result_B_T_H_W_D" not in source
