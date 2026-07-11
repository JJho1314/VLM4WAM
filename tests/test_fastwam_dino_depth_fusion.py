from __future__ import annotations

import importlib
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
FASTWAM_SRC = ROOT / "third_party/FastWAM/src"
FUSION_PATH = (
    FASTWAM_SRC
    / "fastwam/models/cosmos/semantic_plan_fusion.py"
)


def load_fusion_module():
    spec = importlib.util.spec_from_file_location(
        "fastwam_semantic_plan_fusion_under_test",
        FUSION_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_video_expert_module(monkeypatch):
    monkeypatch.syspath_prepend(str(FASTWAM_SRC))
    return importlib.import_module("fastwam.models.cosmos.video_expert")


def test_fusion_defaults_preserve_the_dense_same_position_contract():
    module = load_fusion_module()
    fusion = module.DinoDepthPlanFusion()
    dino = torch.zeros(1, 1024, 1024)
    depth = torch.zeros_like(dino)

    with torch.no_grad():
        output = fusion(dino, depth)

    assert fusion.feature_dim == 1024
    assert fusion.max_tokens == 1024
    assert output.shape == dino.shape
    assert math.isclose(
        torch.sigmoid(fusion.depth_gate_logit).item(),
        0.1,
        rel_tol=0.0,
        abs_tol=1e-6,
    )


def test_fusion_implements_the_normalize_project_gate_normalize_equation():
    module = load_fusion_module()
    fusion = module.DinoDepthPlanFusion(
        feature_dim=3,
        max_tokens=2,
        initial_depth_gate=0.25,
    )
    dino = torch.tensor([[[1.0, 2.0, 4.0], [3.0, -1.0, 2.0]]])
    depth = torch.tensor([[[2.0, 0.0, -1.0], [4.0, 1.0, 3.0]]])

    expected_dino = fusion.dino_proj(fusion.dino_norm(dino))
    expected_depth = fusion.depth_proj(fusion.depth_norm(depth))
    expected = fusion.out_norm(
        expected_dino
        + torch.sigmoid(fusion.depth_gate_logit).to(expected_dino.dtype)
        * expected_depth
    )

    assert torch.allclose(fusion(dino, depth), expected)


def test_fusion_trains_both_projections_and_depth_gate():
    module = load_fusion_module()
    fusion = module.DinoDepthPlanFusion(4, 3, 0.1)

    output = fusion(
        torch.randn(2, 3, 4),
        torch.randn(2, 3, 4),
    )
    output.square().mean().backward()

    assert fusion.dino_proj.weight.grad is not None
    assert fusion.depth_proj.weight.grad is not None
    assert fusion.depth_gate_logit.grad is not None


@pytest.mark.parametrize("initial_gate", [0.0, 1.0, -0.1, 1.1, float("nan")])
def test_fusion_rejects_invalid_initial_gate(initial_gate):
    module = load_fusion_module()

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        module.DinoDepthPlanFusion(4, 3, initial_gate)


@pytest.mark.parametrize(
    ("dino_shape", "depth_shape", "message"),
    [
        ((1, 3, 4), (2, 3, 4), "same shape"),
        ((3, 4), (3, 4), r"\[B, 3, 4\]"),
        ((1, 2, 4), (1, 2, 4), r"\[B, 3, 4\]"),
        ((1, 3, 5), (1, 3, 5), r"\[B, 3, 4\]"),
    ],
)
def test_fusion_rejects_shape_contract_violations(
    dino_shape,
    depth_shape,
    message,
):
    module = load_fusion_module()
    fusion = module.DinoDepthPlanFusion(4, 3, 0.1)

    with pytest.raises(ValueError, match=message):
        fusion(torch.zeros(dino_shape), torch.zeros(depth_shape))


@pytest.mark.parametrize("branch", ["DINO", "depth"])
def test_fusion_rejects_non_finite_values(branch):
    module = load_fusion_module()
    fusion = module.DinoDepthPlanFusion(4, 3, 0.1)
    dino = torch.zeros(1, 3, 4)
    depth = torch.zeros_like(dino)
    target = dino if branch == "DINO" else depth
    target[0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match=f"{branch} plan contains non-finite"):
        fusion(dino, depth)


def test_video_expert_owns_fusion_state_and_routes_provider_dtype(monkeypatch):
    fusion_module = load_fusion_module()
    video_module = load_video_expert_module(monkeypatch)
    fusion = fusion_module.DinoDepthPlanFusion(4, 3, 0.1).to(torch.float64)
    expert = video_module.CosmosVideoExpert(nn.Identity(), fusion)

    output = expert.fuse_semantic_plan(
        torch.randn(1, 3, 4, dtype=torch.float32).detach(),
        torch.randn(1, 3, 4, dtype=torch.float32).detach(),
    )
    output.square().mean().backward()

    state_keys = set(expert.state_dict())
    assert "semantic_plan_fusion.depth_gate_logit" in state_keys
    assert "semantic_plan_fusion.dino_proj.weight" in state_keys
    assert "semantic_plan_fusion.depth_proj.weight" in state_keys
    assert output.dtype == torch.float64
    assert output.device == fusion.depth_gate_logit.device
    assert output.requires_grad
    assert fusion.dino_proj.weight.grad is not None


def test_video_expert_disabled_fusion_fails_clearly(monkeypatch):
    video_module = load_video_expert_module(monkeypatch)
    expert = video_module.CosmosVideoExpert(nn.Identity())

    with pytest.raises(RuntimeError, match="requires semantic_plan_fusion"):
        expert.fuse_semantic_plan(
            torch.zeros(1, 3, 4),
            torch.zeros(1, 3, 4),
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_video_expert_factory_optionally_builds_owned_fusion(
    monkeypatch,
    enabled,
):
    video_module = load_video_expert_module(monkeypatch)
    monkeypatch.setattr(
        video_module,
        "build_cosmos_2b_net",
        lambda **_kwargs: nn.Linear(2, 2),
    )

    expert = video_module.CosmosVideoExpert.from_pretrained(
        device="cpu",
        torch_dtype=torch.float64,
        semantic_plan_fusion_enabled=enabled,
        semantic_plan_feature_dim=4,
        semantic_plan_fusion_max_tokens=3,
        semantic_plan_initial_depth_gate=0.2,
    )

    if enabled:
        assert expert.semantic_plan_fusion is not None
        assert expert.semantic_plan_fusion.feature_dim == 4
        assert expert.semantic_plan_fusion.max_tokens == 3
        assert expert.semantic_plan_fusion.depth_gate_logit.dtype == torch.float64
        assert torch.sigmoid(expert.semantic_plan_fusion.depth_gate_logit).item() == pytest.approx(0.2)
    else:
        assert expert.semantic_plan_fusion is None
