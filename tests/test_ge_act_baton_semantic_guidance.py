from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pytest
import torch
import torch.nn as nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GE_ACT_ROOT = REPOSITORY_ROOT / "ge_act"
for path in (REPOSITORY_ROOT, GE_ACT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.ltx_models.baton_semantic_planner import (  # noqa: E402
    FrozenDualCameraBatonPlanner,
    build_baton_semantic_context,
    build_patch_center_positions,
)
from models.ltx_models.transformer_ltx_multiview import (  # noqa: E402
    LTXVideoTransformerBlock,
    LTXVideoTransformer3DModel,
)
from qwen35_baton.provider import BatonSemanticPlan  # noqa: E402


FUTURE_INDICES = (0, 3, 5, 8)


@dataclass(frozen=True)
class _Plan:
    tokens: torch.Tensor
    future_indices: tuple[int, ...]
    positions_xy: torch.Tensor
    relevance: torch.Tensor | None = None


def _model(*, semantic: bool = True) -> LTXVideoTransformer3DModel:
    return LTXVideoTransformer3DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=6,
        cross_attention_dim=12,
        caption_channels=8,
        num_layers=1,
        semantic_plan_context=semantic,
        semantic_plan_in_dim=1024,
        semantic_plan_coordinate_dim=4,
        semantic_plan_num_keyframes=4,
        semantic_plan_num_views=2,
        semantic_plan_cross_attention_blocks=(0,) if semantic else None,
        semantic_plan_adaln_rank=4,
    )


def _plan(
    *,
    batch_size: int = 1,
    tokens: torch.Tensor | None = None,
    positions_xy: torch.Tensor | None = None,
    future_indices: tuple[int, ...] = FUTURE_INDICES,
    relevance: torch.Tensor | None = None,
) -> _Plan:
    if tokens is None:
        tokens = torch.randn(batch_size, 2, 4, 256, 1024)
    if positions_xy is None:
        positions_xy = build_patch_center_positions(
            batch_size=batch_size,
            num_views=2,
            num_keyframes=4,
            grid_size=16,
            device=tokens.device,
        )
    return _Plan(
        tokens=tokens,
        future_indices=future_indices,
        positions_xy=positions_xy,
        relevance=relevance,
    )


def _context(model: LTXVideoTransformer3DModel, plan: Any):
    return build_baton_semantic_context(
        model,
        plan,
        n_previous=4,
        num_future_frames=9,
        latent_shape=(6, 32, 32),
    )


def _identity_rope(batch_size: int, length: int, dim: int):
    return (
        torch.ones(batch_size, length, dim),
        torch.zeros(batch_size, length, dim),
    )


def test_baton_patch_centers_are_exact_sixteenth_grid_centers() -> None:
    xy = build_patch_center_positions(
        batch_size=1,
        num_views=2,
        num_keyframes=4,
        grid_size=16,
        device=torch.device("cpu"),
    )

    assert xy.shape == (1, 2, 4, 256, 2)
    assert xy.dtype == torch.float32
    assert xy.device == torch.device("cpu")
    torch.testing.assert_close(
        xy[0, 0, 0, 0],
        torch.tensor([1 / 32, 1 / 32]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        xy[0, 0, 0, 1],
        torch.tensor([3 / 32, 1 / 32]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        xy[0, 0, 0, 16],
        torch.tensor([1 / 32, 3 / 32]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        xy[0, 0, 0, -1],
        torch.tensor([31 / 32, 31 / 32]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(xy[0, 0], xy[0, 1], rtol=0, atol=0)


def test_baton_semantic_context_keeps_every_token_and_never_adds_grounding() -> None:
    model = _model()
    observed_projection_shapes: list[tuple[int, ...]] = []
    handle = model.semantic_adapter.feature_projection.register_forward_pre_hook(
        lambda _module, inputs: observed_projection_shapes.append(
            tuple(inputs[0].shape)
        )
    )
    try:
        context = _context(model, _plan())
    finally:
        handle.remove()

    assert model.inner_dim == 12
    assert observed_projection_shapes == [(2, 4, 256, 1024)]
    assert context.hidden_states.shape == (2, 1024, model.inner_dim)
    assert context.positions.shape == (2, 1024, 3)
    assert context.key_mask is None
    assert context.relevance is None


def test_online_teacher_tensor_uses_the_same_full_grid_contract() -> None:
    model = _model()
    teacher_tokens = torch.randn(2, 2, 4, 256, 1024)

    context = _context(model, teacher_tokens)

    assert context.hidden_states.shape == (4, 1024, model.inner_dim)
    assert context.positions.shape == (4, 1024, 3)
    assert context.key_mask is None
    assert context.relevance is None


def test_baton_context_uses_exact_latent_times_and_patch_center_coordinates() -> None:
    context = _context(_model(), _plan())
    positions = context.positions[0]

    expected_times = torch.tensor([4.0, 35 / 8, 37 / 8, 5.0])
    torch.testing.assert_close(
        positions[:, 0].reshape(4, 256)[:, 0],
        expected_times,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        positions[0],
        torch.tensor([4.0, 31 / 32, 31 / 32]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        positions[1],
        torch.tensor([4.0, 31 / 32, 93 / 32]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        positions[16],
        torch.tensor([4.0, 93 / 32, 31 / 32]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        positions[255],
        torch.tensor([4.0, 961 / 32, 961 / 32]),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("plan_update", "message"),
    [
        ({"tokens": torch.randn(1, 1, 4, 256, 1024)}, r"\[B,2,4,256,1024\]"),
        ({"tokens": torch.randn(1, 2, 3, 256, 1024)}, r"\[B,2,4,256,1024\]"),
        ({"tokens": torch.randn(1, 2, 4, 255, 1024)}, r"\[B,2,4,256,1024\]"),
        ({"tokens": torch.randn(1, 2, 4, 256, 16)}, r"\[B,2,4,256,1024\]"),
        (
            {"tokens": torch.zeros(1, 2, 4, 256, 1024, dtype=torch.long)},
            "floating",
        ),
        (
            {
                "tokens": torch.full(
                    (1, 2, 4, 256, 1024),
                    float("nan"),
                )
            },
            "finite",
        ),
        ({"future_indices": (0, 2, 5, 8)}, "future_indices"),
        (
            {"positions_xy": torch.zeros(1, 2, 4, 255, 2)},
            "positions_xy",
        ),
        (
            {
                "positions_xy": torch.zeros(
                    1,
                    2,
                    4,
                    256,
                    2,
                    dtype=torch.float64,
                )
            },
            "float32",
        ),
        (
            {"positions_xy": torch.zeros(1, 2, 4, 256, 2)},
            "exact.*patch centers",
        ),
        (
            {"relevance": torch.ones(1, 2, 4, 256)},
            "relevance",
        ),
    ],
)
def test_baton_context_rejects_malformed_or_lossy_plans(
    plan_update: dict[str, Any],
    message: str,
) -> None:
    tokens = plan_update.get("tokens")
    batch_size = 1 if tokens is None else int(tokens.shape[0])
    plan = _plan(
        batch_size=batch_size,
        tokens=tokens,
        positions_xy=plan_update.get("positions_xy"),
        future_indices=plan_update.get("future_indices", FUTURE_INDICES),
        relevance=plan_update.get("relevance"),
    )

    with pytest.raises((TypeError, ValueError), match=message):
        _context(_model(), plan)


def test_baton_context_rejects_token_adapter_dtype_mismatch() -> None:
    tokens = torch.randn(1, 2, 4, 256, 1024, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="dtype.*semantic_adapter"):
        _context(_model(), _plan(tokens=tokens))


def test_baton_context_rejects_token_adapter_device_mismatch() -> None:
    tokens = torch.empty(1, 2, 4, 256, 1024, device="meta")

    with pytest.raises(ValueError, match="device.*semantic_adapter"):
        _context(_model(), _plan(tokens=tokens))


def test_baton_context_rejects_position_device_mismatch() -> None:
    positions = torch.empty(1, 2, 4, 256, 2, device="meta")

    with pytest.raises(ValueError, match="positions_xy device"):
        _context(_model(), _plan(positions_xy=positions))


def test_baton_context_rejects_nonfinite_positions() -> None:
    positions = build_patch_center_positions(1, 2, 4)
    positions[0, 0, 0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="positions_xy.*finite"):
        _context(_model(), _plan(positions_xy=positions))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_previous": 0}, "n_previous"),
        ({"num_future_frames": 8}, "future_indices"),
        ({"latent_shape": (4, 32, 32)}, "latent temporal"),
        ({"latent_shape": (6, 32)}, "latent_shape"),
        ({"latent_shape": (6, 0, 32)}, "positive"),
    ],
)
def test_baton_context_rejects_invalid_temporal_or_latent_geometry(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    arguments = {
        "n_previous": 4,
        "num_future_frames": 9,
        "latent_shape": (6, 32, 32),
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        build_baton_semantic_context(_model(), _plan(), **arguments)


def test_baton_context_rejects_models_without_semantic_attention_allocation() -> None:
    baseline = _model(semantic=False)

    assert not hasattr(baseline, "semantic_adapter")
    assert not hasattr(baseline.transformer_blocks[0], "semantic_attn")
    with pytest.raises(ValueError, match="semantic_plan_context"):
        _context(baseline, _plan())


def test_semantic_block_order_is_self_then_text_then_baton_then_ffn() -> None:
    block = _model().transformer_blocks[0]
    order: list[str] = []
    handles = [
        module.register_forward_pre_hook(
            lambda _module, _inputs, name=name: order.append(name)
        )
        for name, module in (
            ("self", block.attn1),
            ("text", block.attn2),
            ("baton", block.semantic_attn),
            ("ffn", block.ff),
        )
    ]
    try:
        block(
            hidden_states=torch.randn(2, 3, 12),
            encoder_hidden_states=torch.randn(1, 5, 12),
            temb=torch.randn(2, 3, 6 * 12),
            image_rotary_emb=_identity_rope(2, 3, 12),
            n_view=2,
            embedded_timestep=torch.randn(2, 3, 12),
            semantic_hidden_states=torch.randn(2, 7, 12),
        )
    finally:
        for handle in handles:
            handle.remove()

    assert order == ["self", "text", "baton", "ffn"]


def test_baton_attention_never_reads_the_other_camera_context() -> None:
    torch.manual_seed(17)
    block = _model().transformer_blocks[0]
    with torch.no_grad():
        block.semantic_modulation[1].weight.fill_(0.1)
        block.semantic_modulation[-1].weight.fill_(0.1)
    hidden = torch.randn(2, 3, 12)
    text = torch.randn(1, 5, 12)
    temb = torch.randn(2, 3, 6 * 12)
    embedded_timestep = torch.randn(2, 3, 12)
    context = torch.randn(2, 7, 12)
    changed_context = context.clone()
    changed_context[1].mul_(1000)
    rope = _identity_rope(2, 3, 12)

    expected = block(
        hidden,
        text,
        temb,
        image_rotary_emb=rope,
        n_view=2,
        embedded_timestep=embedded_timestep,
        semantic_hidden_states=context,
    )
    changed = block(
        hidden,
        text,
        temb,
        image_rotary_emb=rope,
        n_view=2,
        embedded_timestep=embedded_timestep,
        semantic_hidden_states=changed_context,
    )

    torch.testing.assert_close(expected[0], changed[0], rtol=0, atol=0)
    assert not torch.allclose(expected[1], changed[1])


def test_enabled_baton_path_backpropagates_without_cross_camera_token_leak() -> None:
    torch.manual_seed(31)
    model = _model()
    block = model.transformer_blocks[0]
    with torch.no_grad():
        block.semantic_modulation[1].weight.fill_(0.05)
        block.semantic_modulation[-1].weight.fill_(0.05)
    semantic_plan = torch.randn(
        1,
        2,
        4,
        256,
        1024,
        requires_grad=True,
    )
    output = model(
        hidden_states=torch.randn(2, 2, 4),
        encoder_hidden_states=torch.randn(1, 3, 8),
        timestep=torch.ones(2, 2),
        encoder_attention_mask=torch.ones(1, 3),
        n_view=2,
        rope_interpolation_scale=(1.6, 32.0, 32.0),
        num_frames=2,
        height=1,
        width=1,
        semantic_plan=semantic_plan,
        semantic_plan_times=torch.tensor(
            [[0.8, 0.875, 0.925, 1.0]] * 2
        ),
        semantic_plan_positions=build_patch_center_positions(
            batch_size=1,
            num_views=2,
            num_keyframes=4,
            grid_size=16,
        ),
        return_dict=False,
    )[0]["video"]

    output[0].square().mean().backward()

    adapter_grad = model.semantic_adapter.feature_projection[0].weight.grad
    gate_grad = block.semantic_modulation[-1].weight.grad
    assert adapter_grad is not None
    assert gate_grad is not None
    assert torch.count_nonzero(adapter_grad) > 0
    assert torch.count_nonzero(gate_grad) > 0
    assert semantic_plan.grad is not None
    assert torch.count_nonzero(semantic_plan.grad[:, 0]) > 0
    assert torch.count_nonzero(semantic_plan.grad[:, 1]) == 0


def test_zero_condition_cloned_block_matches_baseline_bitwise() -> None:
    torch.manual_seed(7)
    baseline = LTXVideoTransformerBlock(
        dim=12,
        num_attention_heads=2,
        attention_head_dim=6,
        cross_attention_dim=12,
    )
    baton = LTXVideoTransformerBlock(
        dim=12,
        num_attention_heads=2,
        attention_head_dim=6,
        cross_attention_dim=12,
        semantic_cross_attention=True,
        semantic_adaln_rank=4,
    )
    baton.load_state_dict(baseline.state_dict(), strict=False)
    hidden = torch.randn(2, 3, 12)
    text = torch.randn(1, 5, 12)
    temb = torch.randn(2, 3, 6 * 12)
    rope = _identity_rope(2, 3, 12)

    expected = baseline(
        hidden,
        text,
        temb,
        n_view=2,
        image_rotary_emb=rope,
    )
    actual = baton(
        hidden,
        text,
        temb,
        n_view=2,
        image_rotary_emb=rope,
        embedded_timestep=torch.randn(2, 3, 12),
        semantic_hidden_states=torch.randn(2, 1024, 12),
        semantic_condition_mask=torch.zeros(2),
    )

    assert torch.count_nonzero(baton.semantic_modulation[-1].weight) == 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_omitted_semantic_inputs_preserve_complete_model_output_bitwise() -> None:
    torch.manual_seed(19)
    baseline = _model(semantic=False)
    baton = _model()
    baton.load_state_dict(baseline.state_dict(), strict=False)
    inputs = {
        "hidden_states": torch.randn(2, 2, 4),
        "encoder_hidden_states": torch.randn(1, 3, 8),
        "timestep": torch.ones(2, 2),
        "encoder_attention_mask": torch.ones(1, 3),
        "n_view": 2,
        "rope_interpolation_scale": (1.6, 32.0, 32.0),
        "num_frames": 2,
        "height": 1,
        "width": 1,
        "return_dict": False,
    }

    expected = baseline(**inputs)[0]["video"]
    actual = baton(**inputs)[0]["video"]

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _FakeFrozenProvider(nn.Module):
    def __init__(self, plan: BatonSemanticPlan) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.plan = plan
        self.calls: list[tuple[torch.Tensor, Sequence[str]]] = []

    def predict(
        self,
        current_images: torch.Tensor,
        instructions: Sequence[str],
        **_: Any,
    ) -> BatonSemanticPlan:
        self.calls.append((current_images, instructions))
        return self.plan


def test_frozen_dual_camera_wrapper_delegates_the_provider_contract_unchanged() -> None:
    tokens = torch.randn(1, 2, 4, 256, 1024)
    plan = BatonSemanticPlan(
        tokens=tokens,
        future_indices=FUTURE_INDICES,
        positions_xy=build_patch_center_positions(
            batch_size=1,
            num_views=2,
            num_keyframes=4,
            grid_size=16,
        ),
        cross_attention_maps=None,
        instruction_sensitivity=None,
    )
    provider = _FakeFrozenProvider(plan)
    wrapper = FrozenDualCameraBatonPlanner(provider)
    images = torch.zeros(1, 2, 3, 8, 8, dtype=torch.uint8)
    instructions = ("pick",)

    actual = wrapper.predict(images, instructions)
    wrapper.train()

    assert actual is plan
    assert provider.calls == [(images, instructions)]
    assert not actual.tokens.requires_grad
    assert all(not module.training for module in wrapper.modules())
    assert all(not parameter.requires_grad for parameter in wrapper.parameters())
