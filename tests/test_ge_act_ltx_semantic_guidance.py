from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


GE_ACT_ROOT = Path(__file__).resolve().parents[1] / "ge_act"
if str(GE_ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(GE_ACT_ROOT))

from models.ltx_models.ltx_attention_processor import Attention
from models.ltx_models.semantic_conditioning import (
    SemanticContextAdapter,
    build_semantic_plan_times,
    select_future_keyframes,
)
from models.ltx_models.transformer_ltx_multiview import (
    LTXVideoRotaryPosEmbed,
    LTXVideoSemanticAttentionProcessor2_0,
    LTXVideoTransformerBlock,
    LTXVideoTransformer3DModel,
)


def test_selects_four_canonical_future_frames() -> None:
    video = torch.arange(1 * 2 * 9 * 3 * 1 * 1).reshape(1, 2, 9, 3, 1, 1)
    selected = select_future_keyframes(video)

    assert selected.shape == (1, 2, 4, 3, 1, 1)
    assert torch.equal(selected, video[:, :, [0, 3, 5, 8]])


def test_semantic_times_align_to_six_ltx_latent_frames() -> None:
    times = build_semantic_plan_times(
        batch_size=2,
        n_view=2,
        n_previous=4,
        num_future_frames=9,
        num_latent_frames=6,
    )

    expected = torch.tensor([0.8, 0.875, 0.925, 1.0]).repeat(4, 1)
    assert times.shape == (4, 4)
    torch.testing.assert_close(times, expected)


def test_semantic_adapter_keeps_16_by_16_tokens_per_frame() -> None:
    adapter = SemanticContextAdapter(
        input_dim=8,
        hidden_dim=12,
        coordinate_dim=4,
        num_views=2,
    )
    tokens = torch.randn(1, 2, 4, 16 * 16, 8)
    times = torch.tensor([[0.8, 0.875, 0.925, 1.0]]).repeat(2, 1)

    context = adapter(
        tokens,
        semantic_plan_times=times,
        latent_height=32,
        latent_width=32,
    )
    adapted, positions = context

    assert adapted.shape == (2, 4 * 16 * 16, 12)
    assert positions.shape == (2, 4 * 16 * 16, 3)
    assert context.key_mask is None
    assert context.relevance is None
    torch.testing.assert_close(positions[0, : 16 * 16, 0], torch.full((16 * 16,), 4.0))
    torch.testing.assert_close(positions[0, -16 * 16 :, 0], torch.full((16 * 16,), 5.0))
    assert positions[..., 1].min() == 0
    assert positions[..., 1].max() == 31
    assert positions[..., 2].min() == 0
    assert positions[..., 2].max() == 31


def test_explicit_position_rope_matches_regular_grid_rope() -> None:
    rope = LTXVideoRotaryPosEmbed(dim=12)
    hidden = torch.zeros(1, 8, 12)
    scale = (0.2, 8.0, 8.0)
    regular = rope(hidden, scale, num_frames=2, height=2, width=2)
    f, h, w = torch.meshgrid(
        torch.arange(2, dtype=torch.float32),
        torch.arange(2, dtype=torch.float32),
        torch.arange(2, dtype=torch.float32),
        indexing="ij",
    )
    positions = torch.stack((f, h, w), dim=-1).reshape(1, 8, 3)
    explicit = rope.forward_positions(hidden, positions, scale)

    torch.testing.assert_close(explicit[0], regular[0])
    torch.testing.assert_close(explicit[1], regular[1])


def test_semantic_attention_never_reads_another_camera_context() -> None:
    torch.manual_seed(0)
    attention = Attention(
        query_dim=12,
        cross_attention_dim=12,
        heads=2,
        kv_heads=2,
        dim_head=6,
        qk_norm=None,
        processor=LTXVideoSemanticAttentionProcessor2_0(),
    )
    queries = torch.randn(2, 3, 12)
    context = torch.randn(2, 5, 12)
    changed_context = context.clone()
    changed_context[1].mul_(1000)

    output = attention(queries, encoder_hidden_states=context)
    changed_output = attention(queries, encoder_hidden_states=changed_context)

    torch.testing.assert_close(output[0], changed_output[0])
    assert not torch.allclose(output[1], changed_output[1])


def test_zero_initialized_semantic_gate_preserves_base_block_output() -> None:
    torch.manual_seed(7)
    base = LTXVideoTransformerBlock(
        dim=12,
        num_attention_heads=2,
        attention_head_dim=6,
        cross_attention_dim=12,
    )
    semantic = LTXVideoTransformerBlock(
        dim=12,
        num_attention_heads=2,
        attention_head_dim=6,
        cross_attention_dim=12,
        semantic_cross_attention=True,
        semantic_adaln_rank=4,
    )
    semantic.load_state_dict(base.state_dict(), strict=False)
    base.eval()
    semantic.eval()

    hidden = torch.randn(2, 3, 12)
    text = torch.randn(1, 5, 12)
    temb = torch.randn(2, 3, 6 * 12)
    embedded_timestep = torch.randn(2, 3, 12)
    semantic_context = torch.randn(2, 7, 12)
    identity_rope = (torch.ones(2, 3, 12), torch.zeros(2, 3, 12))

    expected = base(hidden, text, temb, n_view=2, image_rotary_emb=identity_rope)
    actual = semantic(
        hidden,
        text,
        temb,
        n_view=2,
        image_rotary_emb=identity_rope,
        embedded_timestep=embedded_timestep,
        semantic_hidden_states=semantic_context,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_online_encoder_discards_the_unused_siglip_text_tower(monkeypatch) -> None:
    import transformers
    from models.ltx_models.semantic_conditioning import OnlineSiglip2SemanticEncoder

    class FakeVisionTower(nn.Module):
        pass

    class FakeFullModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision_model = FakeVisionTower()
            self.text_model = nn.Linear(2, 2)

    full_model = FakeFullModel()
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: full_model,
    )

    encoder = OnlineSiglip2SemanticEncoder("unused", device="cpu", dtype=torch.float32)

    assert encoder.model is full_model.vision_model
    assert not hasattr(encoder, "full_model")


def test_zero_gate_preserves_the_complete_ltx_model_output() -> None:
    torch.manual_seed(19)
    common = dict(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=6,
        cross_attention_dim=12,
        caption_channels=8,
        num_layers=1,
    )
    base = LTXVideoTransformer3DModel(**common)
    semantic = LTXVideoTransformer3DModel(
        **common,
        semantic_plan_context=True,
        semantic_plan_in_dim=8,
        semantic_plan_coordinate_dim=4,
        semantic_plan_num_keyframes=2,
        semantic_plan_num_views=2,
        semantic_plan_adaln_rank=4,
    )
    semantic.load_state_dict(base.state_dict(), strict=False)
    base.eval()
    semantic.eval()
    model_inputs = dict(
        hidden_states=torch.randn(2, 2, 4),
        encoder_hidden_states=torch.randn(1, 3, 8),
        timestep=torch.ones(2, 2),
        encoder_attention_mask=torch.ones(1, 3),
        n_view=2,
        rope_interpolation_scale=(1.6, 32.0, 32.0),
        num_frames=2,
        height=1,
        width=1,
        return_dict=False,
    )

    expected = base(**model_inputs)[0]["video"]
    actual = semantic(
        **model_inputs,
        semantic_plan=torch.randn(1, 2, 2, 4, 8),
        semantic_plan_times=torch.tensor([[0.5, 1.0]]).repeat(2, 1),
    )[0]["video"]

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
