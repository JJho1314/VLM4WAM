from __future__ import annotations

import io

import pytest
import torch

from qwen3_vl_semantic_planner.dinov3_da3_2b.wsa_depth_probe import (
    WSAMultiLayerDPTProbe,
)


def make_probe() -> WSAMultiLayerDPTProbe:
    return WSAMultiLayerDPTProbe(
        in_dim=32,
        feat=16,
        grid=4,
        output_size=32,
        teacher_layers=(11, 15, 19, 23),
    )


def test_wsa_probe_shape_is_finite_and_config_is_complete() -> None:
    probe = make_probe().eval()

    output = probe(torch.randn(2, 4, 16, 32))

    assert output.shape == (2, 1, 32, 32)
    assert torch.isfinite(output).all()
    assert probe.config() == {
        "in_dim": 32,
        "feat": 16,
        "grid": 4,
        "output_size": 32,
        "teacher_layers": [11, 15, 19, 23],
        "out_ch": 1,
        "normalization": "per_token_layer_norm_no_affine",
    }


@pytest.mark.parametrize(
    ("tokens", "message"),
    [
        (torch.randn(2, 3, 16, 32), "4 layers"),
        (torch.randn(2, 4, 15, 32), "16 tokens"),
        (torch.randn(2, 4, 16, 31), "feature width 32"),
        (torch.randn(2, 4, 16), r"\[B,L,N,D\]"),
    ],
)
def test_wsa_probe_rejects_incompatible_geometry(
    tokens: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_probe()(tokens)


def test_wsa_probe_is_invariant_to_per_token_scale_and_offset() -> None:
    torch.manual_seed(7)
    probe = make_probe().eval()
    tokens = torch.randn(2, 4, 16, 32)
    positive_scale = torch.rand(2, 4, 16, 1) + 0.5
    channel_independent_offset = torch.randn(2, 4, 16, 1)

    baseline = probe(tokens)
    transformed = probe(
        tokens * positive_scale + channel_independent_offset
    )

    torch.testing.assert_close(
        transformed,
        baseline,
        atol=3e-5,
        rtol=3e-5,
    )


def test_wsa_probe_state_dict_round_trip_preserves_output() -> None:
    torch.manual_seed(11)
    probe = make_probe().eval()
    tokens = torch.randn(1, 4, 16, 32)
    expected = probe(tokens)
    payload = io.BytesIO()
    torch.save(
        {"config": probe.config(), "state_dict": probe.state_dict()},
        payload,
    )
    payload.seek(0)
    saved = torch.load(payload, weights_only=False)
    restored = WSAMultiLayerDPTProbe.from_config(saved["config"]).eval()
    restored.load_state_dict(saved["state_dict"], strict=True)

    torch.testing.assert_close(restored(tokens), expected)
