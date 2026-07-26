from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


GE_ACT_ROOT = Path(__file__).resolve().parents[1] / "ge_act"
if str(GE_ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(GE_ACT_ROOT))

from models.ltx_models.ltx_attention_processor import Attention  # noqa: E402
from models.ltx_models.semantic_conditioning import SemanticContextAdapter  # noqa: E402
from models.ltx_models import transformer_ltx_multiview as transformer_module  # noqa: E402
from models.ltx_models.transformer_ltx_multiview import (  # noqa: E402
    LTXVideoSemanticAttentionProcessor2_0,
    LTXVideoTransformer3DModel,
    LTXVideoTransformerBlock,
)


def _deterministic_semantic_attention() -> Attention:
    attention = Attention(
        query_dim=2,
        cross_attention_dim=2,
        heads=1,
        kv_heads=1,
        dim_head=2,
        bias=False,
        out_bias=False,
        qk_norm=None,
        processor=LTXVideoSemanticAttentionProcessor2_0(),
    )
    with torch.no_grad():
        attention.to_q.weight.zero_()
        attention.to_k.weight.zero_()
        attention.to_v.weight.copy_(torch.eye(2))
        attention.to_out[0].weight.copy_(torch.eye(2))
    return attention


def test_adapter_maps_compressed_xy_positions_and_flattens_camera_samples() -> None:
    adapter = SemanticContextAdapter(
        input_dim=4,
        hidden_dim=6,
        coordinate_dim=3,
        num_views=2,
    )
    tokens = torch.randn(1, 2, 2, 96, 4, requires_grad=True)
    positions = torch.zeros(1, 2, 2, 96, 2, requires_grad=True)
    with torch.no_grad():
        positions[0, 0, 0, 0] = torch.tensor([0.25, 0.75])
        positions[0, 1, 1, 95] = torch.tensor([1.0, 0.0])
    mask = torch.ones(1, 2, 2, 96, dtype=torch.bool)
    mask[0, 0, 1, 3] = False
    relevance = torch.linspace(0.01, 1.0, 1 * 2 * 2 * 96).reshape(1, 2, 2, 96)
    times = torch.tensor([[[0.25, 0.75], [0.5, 1.0]]])

    context = adapter(
        tokens,
        semantic_plan_times=times,
        latent_height=9,
        latent_width=5,
        latent_num_frames=5,
        semantic_positions_xy=positions,
        semantic_token_mask=mask,
        semantic_relevance=relevance,
    )

    assert context.hidden_states.shape == (2, 2 * 96, 6)
    assert context.positions.shape == (2, 2 * 96, 3)
    assert context.key_mask.shape == (2, 2 * 96)
    assert context.relevance.shape == (2, 2 * 96)
    torch.testing.assert_close(context.positions[0, 0], torch.tensor([1.0, 6.0, 1.0]))
    torch.testing.assert_close(context.positions[1, -1], torch.tensor([4.0, 0.0, 4.0]))
    assert not context.key_mask[0, 96 + 3]
    torch.testing.assert_close(context.relevance, relevance.reshape(2, 2 * 96))

    context.hidden_states.sum().backward()
    assert tokens.grad is not None
    assert positions.grad is not None
    assert torch.count_nonzero(positions.grad) > 0


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        (
            "semantic_positions_xy",
            torch.zeros(1, 2, 2, 95, 2),
            "semantic_positions_xy must have shape",
        ),
        (
            "semantic_token_mask",
            torch.ones(1, 2, 2, 96),
            "boolean",
        ),
        (
            "semantic_relevance",
            torch.full((1, 2, 2, 96), -0.1),
            "non-negative",
        ),
    ],
)
def test_adapter_rejects_invalid_grounding_fields(
    field: str,
    value: torch.Tensor,
    error: str,
) -> None:
    kwargs = {
        "semantic_positions_xy": torch.zeros(1, 2, 2, 96, 2),
        "semantic_token_mask": torch.ones(1, 2, 2, 96, dtype=torch.bool),
        "semantic_relevance": torch.ones(1, 2, 2, 96),
    }
    kwargs[field] = value

    with pytest.raises((TypeError, ValueError), match=error):
        SemanticContextAdapter(
            input_dim=4,
            hidden_dim=6,
            coordinate_dim=3,
            num_views=2,
        )(
            torch.randn(1, 2, 2, 96, 4),
            semantic_plan_times=torch.ones(2, 2),
            latent_height=4,
            latent_width=4,
            **kwargs,
        )


def test_zero_bias_gate_uses_baseline_sdpa_and_matches_bitwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = _deterministic_semantic_attention()
    queries = torch.randn(2, 3, 2)
    context = torch.randn(2, 4, 2)
    attention_masks: list[torch.Tensor | None] = []
    scaled_dot_product_attention = transformer_module.F.scaled_dot_product_attention

    def record_attention_mask(*args, **kwargs):
        attention_masks.append(kwargs.get("attn_mask"))
        return scaled_dot_product_attention(*args, **kwargs)

    monkeypatch.setattr(
        transformer_module.F,
        "scaled_dot_product_attention",
        record_attention_mask,
    )

    expected = attention(queries, encoder_hidden_states=context)
    attention_masks.clear()
    actual = attention(
        queries,
        encoder_hidden_states=context,
        relevance=torch.tensor([[1.0, 0.5, 0.01, 0.0]]).repeat(2, 1),
    )

    assert attention.processor.raw_semantic_bias_gate.item() == 0.0
    assert len(attention_masks) == 2
    assert attention_masks[0] is None
    assert attention_masks[1] is not None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_zero_bias_gate_matches_cuda_mixed_precision_bitwise(
    dtype: torch.dtype,
) -> None:
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")
    attention = _deterministic_semantic_attention().to(device="cuda", dtype=dtype)
    queries = torch.randn(2, 7, 2, device="cuda", dtype=dtype)
    context = torch.randn(2, 11, 2, device="cuda", dtype=dtype)
    relevance = torch.linspace(
        0.01,
        1.0,
        2 * 11,
        device="cuda",
        dtype=dtype,
    ).reshape(2, 11)

    expected = attention(queries, encoder_hidden_states=context)
    actual = attention(
        queries,
        encoder_hidden_states=context,
        relevance=relevance,
    )

    assert torch.equal(actual, expected)
    actual.float().square().mean().backward()
    gate_grad = attention.processor.raw_semantic_bias_gate.grad
    assert gate_grad is not None
    assert torch.count_nonzero(gate_grad) > 0


def test_bias_is_bounded_and_prefers_the_more_relevant_key() -> None:
    attention = _deterministic_semantic_attention()
    attention.processor.raw_semantic_bias_gate.data.fill_(10.0)

    output = attention(
        torch.zeros(1, 1, 2),
        encoder_hidden_states=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        relevance=torch.tensor([[1.0, 0.01]]),
    )

    assert 1.99 < attention.processor.semantic_bias_gate.item() <= 2.0
    assert output[0, 0, 0] > output[0, 0, 1]


@pytest.mark.parametrize("raw_gate", [0.0, 1.0])
def test_padding_mask_blocks_keys_and_all_masked_context_is_finite(
    raw_gate: float,
) -> None:
    attention = _deterministic_semantic_attention()
    attention.processor.raw_semantic_bias_gate.data.fill_(raw_gate)
    context = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    output = attention(
        torch.zeros(1, 1, 2),
        encoder_hidden_states=context,
        attention_mask=torch.tensor([[False, True]]),
        relevance=torch.ones(1, 2),
    )
    all_masked = attention(
        torch.zeros(1, 1, 2),
        encoder_hidden_states=context,
        attention_mask=torch.zeros(1, 2, dtype=torch.bool),
        relevance=torch.ones(1, 2),
    )

    torch.testing.assert_close(output, torch.tensor([[[0.0, 1.0]]]))
    assert torch.isfinite(all_masked).all()
    torch.testing.assert_close(all_masked, torch.zeros_like(all_masked))


def test_relevance_bias_keeps_camera_batches_isolated_and_differentiable() -> None:
    attention = _deterministic_semantic_attention()
    attention.processor.raw_semantic_bias_gate.data.fill_(0.5)
    queries = torch.zeros(2, 1, 2)
    context = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    relevance = torch.tensor(
        [[1.0, 0.01], [1.0, 0.01]],
        requires_grad=True,
    )

    baseline = attention(
        queries,
        encoder_hidden_states=context,
        relevance=relevance,
    )
    changed_relevance = relevance.detach().clone()
    changed_relevance[1] = torch.tensor([0.01, 1.0])
    changed = attention(
        queries,
        encoder_hidden_states=context,
        relevance=changed_relevance,
    )

    torch.testing.assert_close(baseline[0], changed[0], rtol=0, atol=0)
    assert not torch.allclose(baseline[1], changed[1])
    baseline.sum().backward()
    assert relevance.grad is not None
    assert torch.count_nonzero(relevance.grad) > 0
    assert attention.processor.raw_semantic_bias_gate.grad is not None
    assert torch.count_nonzero(attention.processor.raw_semantic_bias_gate.grad) > 0


def test_old_semantic_attention_state_dict_loads_with_zero_bias_gate() -> None:
    block = LTXVideoTransformerBlock(
        dim=12,
        num_attention_heads=2,
        attention_head_dim=6,
        cross_attention_dim=12,
        semantic_cross_attention=True,
        semantic_adaln_rank=4,
    )
    old_state_dict = {
        key: value
        for key, value in block.state_dict().items()
        if not key.endswith("raw_semantic_bias_gate")
    }
    restored = LTXVideoTransformerBlock(
        dim=12,
        num_attention_heads=2,
        attention_head_dim=6,
        cross_attention_dim=12,
        semantic_cross_attention=True,
        semantic_adaln_rank=4,
    )

    restored.load_state_dict(old_state_dict, strict=True)

    assert restored.semantic_attn.processor.raw_semantic_bias_gate.item() == 0.0


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_untouched_zero_residual_learns_bias_gate_on_first_backward(
    gradient_checkpointing: bool,
) -> None:
    torch.manual_seed(29)
    model = LTXVideoTransformer3DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=6,
        cross_attention_dim=12,
        caption_channels=8,
        num_layers=1,
        semantic_plan_context=True,
        semantic_plan_in_dim=8,
        semantic_plan_coordinate_dim=4,
        semantic_plan_num_keyframes=2,
        semantic_plan_num_views=2,
        semantic_plan_cross_attention_blocks=(0,),
        semantic_plan_adaln_rank=4,
    )
    if gradient_checkpointing:
        model.enable_gradient_checkpointing()
    block = model.transformer_blocks[0]
    assert torch.count_nonzero(block.semantic_modulation[-1].weight) == 0
    assert block.semantic_attn.processor.raw_semantic_bias_gate.item() == 0.0

    common_inputs = {
        "hidden_states": torch.randn(2, 2, 4),
        "encoder_hidden_states": torch.randn(1, 3, 8),
        "timestep": torch.ones(2, 2),
        "encoder_attention_mask": torch.ones(1, 3),
        "n_view": 2,
        "rope_interpolation_scale": (1.6, 32.0, 32.0),
        "num_frames": 2,
        "height": 1,
        "width": 1,
        "semantic_plan": torch.randn(1, 2, 2, 3, 8),
        "semantic_plan_times": torch.tensor([[0.5, 1.0]]).repeat(2, 1),
        "semantic_plan_positions": torch.rand(1, 2, 2, 3, 2),
        "return_dict": False,
    }
    with torch.no_grad():
        expected = model(**common_inputs)[0]["video"]

    semantic_relevance = torch.tensor(
        [[[[1.0, 0.4, 0.2], [0.8, 0.3, 0.1]]] * 2],
        requires_grad=True,
    )
    actual = model(
        **common_inputs,
        semantic_plan_relevance=semantic_relevance,
    )[0]["video"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()

    gate_grad = block.semantic_attn.processor.raw_semantic_bias_gate.grad
    assert gate_grad is not None
    assert torch.count_nonzero(gate_grad) > 0
    for projection in (
        block.semantic_attn.to_q,
        block.semantic_attn.to_k,
        block.semantic_attn.to_v,
        block.semantic_attn.to_out[0],
    ):
        projection_grad = projection.weight.grad
        assert projection_grad is None or torch.count_nonzero(projection_grad) == 0


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_transformer_threads_grounding_to_every_semantic_block(
    gradient_checkpointing: bool,
) -> None:
    torch.manual_seed(23)
    model = LTXVideoTransformer3DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=6,
        cross_attention_dim=12,
        caption_channels=8,
        num_layers=2,
        semantic_plan_context=True,
        semantic_plan_in_dim=8,
        semantic_plan_coordinate_dim=4,
        semantic_plan_num_keyframes=2,
        semantic_plan_num_views=2,
        semantic_plan_cross_attention_blocks=(0, 1),
        semantic_plan_adaln_rank=4,
    )
    if gradient_checkpointing:
        model.enable_gradient_checkpointing()
    with torch.no_grad():
        for block in model.transformer_blocks:
            block.semantic_modulation[1].weight.fill_(0.1)
            block.semantic_modulation[-1].weight.fill_(0.1)
            block.semantic_attn.processor.raw_semantic_bias_gate.fill_(0.5)

    semantic_plan = torch.randn(1, 2, 2, 3, 8, requires_grad=True)
    semantic_positions = torch.rand(1, 2, 2, 3, 2, requires_grad=True)
    semantic_relevance = torch.tensor(
        [[[[1.0, 0.4, 0.2], [0.8, 0.3, 0.1]]] * 2],
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
        semantic_plan_times=torch.tensor([[0.5, 1.0]]).repeat(2, 1),
        semantic_plan_positions=semantic_positions,
        semantic_plan_mask=torch.ones(1, 2, 2, 3, dtype=torch.bool),
        semantic_plan_relevance=semantic_relevance,
        return_dict=False,
    )[0]["video"]
    output.square().mean().backward()

    assert semantic_plan.grad is not None
    assert semantic_positions.grad is not None
    assert semantic_relevance.grad is not None
    assert torch.count_nonzero(semantic_relevance.grad) > 0
    for block in model.transformer_blocks:
        gate_grad = block.semantic_attn.processor.raw_semantic_bias_gate.grad
        assert gate_grad is not None
        assert torch.count_nonzero(gate_grad) > 0
