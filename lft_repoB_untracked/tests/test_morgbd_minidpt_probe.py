from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "qwen3_vl_semantic_planner"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from morgbd_minidpt_probe import (  # noqa: E402
    MiniDPTDepthProbe,
    dense_log_depth_target,
    multiscale_gradient_loss,
    silog_loss,
)


def test_minidpt_outputs_dense_log_depth() -> None:
    probe = MiniDPTDepthProbe(
        in_dim=32,
        feat=32,
        grid=4,
        output_size=28,
    )
    output = probe(torch.randn(2, 16, 32))
    assert tuple(output.shape) == (2, 1, 28, 28)
    assert bool(torch.isfinite(output).all())
    assert probe.config() == {
        "in_dim": 32,
        "feat": 32,
        "grid": 4,
        "out_ch": 1,
        "output_size": 28,
    }


def test_minidpt_rejects_wrong_token_geometry() -> None:
    probe = MiniDPTDepthProbe(in_dim=32, feat=32, grid=4, output_size=28)
    with pytest.raises(ValueError, match="tokens must be"):
        probe(torch.randn(2, 15, 32))


def test_dense_target_and_losses_are_scale_invariant() -> None:
    depth = torch.linspace(1, 4, 64).reshape(1, 8, 8)
    target = dense_log_depth_target(depth, output_size=8)
    shifted = target + 2.0
    assert tuple(target.shape) == (1, 1, 8, 8)
    assert float(silog_loss(shifted, target)) < 1.0
    assert float(multiscale_gradient_loss(shifted, target)) == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_dense_target_sanitizes_invalid_depth() -> None:
    depth = torch.tensor([[[0.0, float("nan")], [float("inf"), 2.0]]])
    target = dense_log_depth_target(depth, output_size=4)
    assert tuple(target.shape) == (1, 1, 4, 4)
    assert bool(torch.isfinite(target).all())
