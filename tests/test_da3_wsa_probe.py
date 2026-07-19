from __future__ import annotations

import io

import pytest
import torch

from qwen3_vl_semantic_planner.dinov3_da3_2b.train_feature_probes import (
    PROBE_CHOICES,
    probe_checkpoint_stem,
)
from qwen3_vl_semantic_planner.dinov3_da3_2b.visualize_qwen3vl2b_siglip2_da3_dual_camera_k4 import (
    depth_features_for_probe,
    probe_kind_for_metadata,
)
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


def test_probe_training_cli_exposes_wsa_without_renaming_legacy_artifacts() -> None:
    assert "da3_wsa" in PROBE_CHOICES
    assert probe_checkpoint_stem("da3_wsa") == "da3_depth_wsa"
    assert probe_checkpoint_stem("da3_v2") == "da3_depth_v2"
    assert probe_checkpoint_stem("dino_up") == "dino_upsample"


def test_probe_kind_routes_wsa_and_legacy_metadata() -> None:
    assert (
        probe_kind_for_metadata({"da3_align_strategy": "wsa_multilayer"})
        == "wsa"
    )
    assert probe_kind_for_metadata({"da3_align_strategy": "last_layer"}) == "last_layer"
    assert probe_kind_for_metadata({}) == "last_layer"
    with pytest.raises(ValueError, match="unsupported DA3 alignment strategy"):
        probe_kind_for_metadata({"da3_align_strategy": "unknown"})


def test_depth_features_for_wsa_probe_preserve_all_layers_and_token_order() -> None:
    target = torch.arange(1 * 2 * 4 * 1024 * 8).reshape(1, 2, 4, 1024, 8)
    prediction = target.transpose(2, 3).contiguous() + 3
    token_slice = slice(256, 512)

    target_out, prediction_out = depth_features_for_probe(
        target,
        prediction,
        camera_index=1,
        token_slice=token_slice,
        probe_kind="wsa",
    )

    assert target_out.shape == prediction_out.shape == (1, 4, 256, 8)
    torch.testing.assert_close(target_out, target[:, 1, :, token_slice])
    torch.testing.assert_close(
        prediction_out,
        prediction[:, 1, token_slice].transpose(1, 2),
    )


def test_depth_features_for_legacy_probe_keep_single_layer_geometry() -> None:
    target = torch.randn(1, 2, 1024, 8)
    prediction = torch.randn(1, 2, 1024, 8)

    target_out, prediction_out = depth_features_for_probe(
        target,
        prediction,
        camera_index=0,
        token_slice=slice(0, 256),
        probe_kind="last_layer",
    )

    assert target_out.shape == prediction_out.shape == (1, 256, 8)
    torch.testing.assert_close(target_out, target[:, 0, :256])
    torch.testing.assert_close(prediction_out, prediction[:, 0, :256])
