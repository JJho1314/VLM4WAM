from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


GE_ACT_ROOT = Path(__file__).resolve().parents[1] / "ge_act"
if str(GE_ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(GE_ACT_ROOT))

from models.ltx_models.joint_vlm_geact import (  # noqa: E402
    JointVLMGEActModel,
    build_joint_optimizer_parameter_groups,
)


NUM_KEYFRAMES = 4
TOKENS_PER_KEYFRAME = 2
SEMANTIC_DIM = 1024
DEPTH_LAYERS = 4
DEPTH_DIM = 2048


def _finite_nonzero(value: torch.Tensor | None) -> bool:
    return bool(
        value is not None
        and torch.isfinite(value).all()
        and torch.count_nonzero(value) > 0
    )


class TinyQwenModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1, 1, bias=False)
        self.visual = nn.Linear(1, 1, bias=False)
        self.lm_head = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.proj.weight, 0.5)


class TinyPlanner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = TinyQwenModel()
        self.plan_head = nn.Linear(1, 1, bias=False)
        self.depth_head = nn.Linear(1, 1, bias=False)
        self.plan_embedding_injector = nn.Embedding(2, 1)
        self.shared_query_bank = nn.Parameter(torch.ones(1, 1))
        self.private_query_bank = nn.Parameter(torch.ones(1, 1))
        self.forward_calls = 0

    def predict_dino_depth_plan_with_losses(
        self,
        *,
        semantic_plan_labels: torch.Tensor,
        depth_plan_labels: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        self.forward_calls += 1
        batch_size = input_ids.shape[0]
        qwen_hidden = self.model.proj(input_ids.float()).mean(dim=1)
        plan_value = self.plan_head(qwen_hidden)
        depth_value = self.depth_head(qwen_hidden)
        semantic = plan_value.reshape(batch_size, 1, 1, 1).expand(
            batch_size,
            2,
            NUM_KEYFRAMES * TOKENS_PER_KEYFRAME,
            SEMANTIC_DIM,
        )
        depth = depth_value.reshape(batch_size, 1, 1, 1, 1).expand(
            batch_size,
            2,
            NUM_KEYFRAMES * TOKENS_PER_KEYFRAME,
            DEPTH_LAYERS,
            DEPTH_DIM,
        )
        semantic_loss = (semantic - semantic_plan_labels).square().mean()
        depth_target = depth_plan_labels.transpose(2, 3)
        depth_loss = (depth - depth_target).square().mean()
        return (
            semantic,
            depth,
            {
                "loss": semantic_loss + depth_loss,
                "semantic_mse": semantic_loss.detach(),
                "depth_lnmse": depth_loss.detach(),
            },
        )


class TinyLTX(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_proj = nn.Linear(2, 2, bias=False)
        self.semantic_attn = nn.Linear(SEMANTIC_DIM, 2, bias=False)
        nn.init.constant_(self.base_proj.weight, 0.25)
        nn.init.constant_(self.semantic_attn.weight, 0.01)
        self.forward_calls = 0

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        semantic_plan: torch.Tensor,
        **_kwargs,
    ) -> tuple[dict[str, torch.Tensor]]:
        self.forward_calls += 1
        batch_size, num_views = semantic_plan.shape[:2]
        semantic_context = semantic_plan.mean(dim=(2, 3)).reshape(
            batch_size * num_views, SEMANTIC_DIM
        )
        semantic_update = self.semantic_attn(semantic_context).unsqueeze(1)
        return ({"video": self.base_proj(hidden_states) + semantic_update},)


def _planner_labels(batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    semantic = torch.zeros(
        batch_size,
        2,
        NUM_KEYFRAMES * TOKENS_PER_KEYFRAME,
        SEMANTIC_DIM,
    )
    depth = torch.zeros(
        batch_size,
        2,
        DEPTH_LAYERS,
        NUM_KEYFRAMES * TOKENS_PER_KEYFRAME,
        DEPTH_DIM,
    )
    return semantic, depth


def _ltx_inputs(batch_size: int = 1) -> dict[str, object]:
    batch_views = batch_size * 2
    return {
        "prompt_embeds": torch.zeros(batch_size, 1, 2),
        "prompt_attention_mask": torch.ones(batch_size, 1),
        "noisy_latents": torch.ones(batch_views, 1, 2),
        "timesteps": torch.ones(batch_views, 1),
        "num_frames": 1,
        "height": 1,
        "width": 1,
        "n_view": 2,
        "frame_rate": 5,
        "temporal_compression_ratio": 8,
        "spatial_compression_ratio": 32,
        "semantic_plan_times": torch.tensor([[0.25, 0.5, 0.75, 1.0]]).repeat(
            batch_views, 1
        ),
    }


def _make_tiny_joint_model() -> JointVLMGEActModel:
    return JointVLMGEActModel(
        TinyPlanner(),
        TinyLTX(),
        num_keyframes=NUM_KEYFRAMES,
        tokens_per_keyframe=TOKENS_PER_KEYFRAME,
    )


def _run_joint(joint: JointVLMGEActModel):
    semantic_labels, depth_labels = _planner_labels()
    return joint(
        planner_inputs={"input_ids": torch.ones(1, 1, dtype=torch.long)},
        semantic_labels=semantic_labels,
        depth_labels=depth_labels,
        ltx_inputs=_ltx_inputs(),
    )


def test_video_loss_reaches_qwen_and_ltx_semantic_parameters() -> None:
    joint = _make_tiny_joint_model()

    output = _run_joint(joint)
    output.ltx_predictions["video"].square().mean().backward()

    assert joint.planner.forward_calls == 1
    assert joint.ltx.forward_calls == 1
    assert output.semantic_plan.shape == (
        1,
        2,
        NUM_KEYFRAMES,
        TOKENS_PER_KEYFRAME,
        SEMANTIC_DIM,
    )
    assert output.depth_plan.shape == (
        1,
        2,
        NUM_KEYFRAMES * TOKENS_PER_KEYFRAME,
        DEPTH_LAYERS,
        DEPTH_DIM,
    )
    assert set(output.ltx_predictions) == {"video"}
    assert output.planner_losses["loss"].requires_grad
    assert _finite_nonzero(joint.planner.model.proj.weight.grad)
    assert _finite_nonzero(joint.ltx.semantic_attn.weight.grad)


def test_joint_casts_semantic_plan_to_ltx_dtype_without_detaching() -> None:
    joint = _make_tiny_joint_model()
    joint.ltx.double()
    semantic_labels, depth_labels = _planner_labels()
    ltx_inputs = _ltx_inputs()
    for name in ("prompt_embeds", "noisy_latents", "timesteps"):
        ltx_inputs[name] = ltx_inputs[name].double()

    output = joint(
        planner_inputs={"input_ids": torch.ones(1, 1, dtype=torch.long)},
        semantic_labels=semantic_labels,
        depth_labels=depth_labels,
        ltx_inputs=ltx_inputs,
    )
    output.ltx_predictions["video"].square().mean().backward()

    assert output.semantic_plan.dtype == torch.float64
    assert _finite_nonzero(joint.planner.model.proj.weight.grad)


@pytest.mark.parametrize(
    ("bad_shape", "message"),
    [
        ((1, 1, 8, SEMANTIC_DIM), "semantic prediction must have shape"),
        ((1, 2, 7, SEMANTIC_DIM), "semantic prediction must have shape"),
        ((1, 2, 8, 16), "semantic prediction must have shape"),
    ],
)
def test_joint_rejects_bad_semantic_shape_before_ltx(
    bad_shape: tuple[int, ...], message: str
) -> None:
    joint = _make_tiny_joint_model()
    original = joint.planner.predict_dino_depth_plan_with_losses

    def bad_prediction(**kwargs):
        _semantic, depth, losses = original(**kwargs)
        return torch.zeros(bad_shape), depth, losses

    joint.planner.predict_dino_depth_plan_with_losses = bad_prediction

    with pytest.raises(ValueError, match=message):
        _run_joint(joint)

    assert joint.ltx.forward_calls == 0


def test_joint_rejects_bad_depth_shape_before_ltx() -> None:
    joint = _make_tiny_joint_model()
    original = joint.planner.predict_dino_depth_plan_with_losses

    def bad_prediction(**kwargs):
        semantic, _depth, losses = original(**kwargs)
        return semantic, torch.zeros(1, 2, 8, 3, DEPTH_DIM), losses

    joint.planner.predict_dino_depth_plan_with_losses = bad_prediction

    with pytest.raises(ValueError, match="depth prediction must have shape"):
        _run_joint(joint)

    assert joint.ltx.forward_calls == 0


def test_joint_rejects_bad_semantic_times_before_ltx() -> None:
    joint = _make_tiny_joint_model()
    semantic_labels, depth_labels = _planner_labels()
    ltx_inputs = _ltx_inputs()
    ltx_inputs["semantic_plan_times"] = torch.ones(1, NUM_KEYFRAMES)

    with pytest.raises(ValueError, match="semantic_plan_times must have shape"):
        joint(
            planner_inputs={"input_ids": torch.ones(1, 1, dtype=torch.long)},
            semantic_labels=semantic_labels,
            depth_labels=depth_labels,
            ltx_inputs=ltx_inputs,
        )

    assert joint.ltx.forward_calls == 0


def test_joint_rejects_inconsistent_ltx_latent_geometry_before_ltx() -> None:
    joint = _make_tiny_joint_model()
    semantic_labels, depth_labels = _planner_labels()
    ltx_inputs = _ltx_inputs()
    ltx_inputs["noisy_latents"] = torch.ones(2, 2, 2)

    with pytest.raises(ValueError, match="noisy_latents token dimension"):
        joint(
            planner_inputs={"input_ids": torch.ones(1, 1, dtype=torch.long)},
            semantic_labels=semantic_labels,
            depth_labels=depth_labels,
            ltx_inputs=ltx_inputs,
        )

    assert joint.planner.forward_calls == 0
    assert joint.ltx.forward_calls == 0


def test_joint_optimizer_groups_are_disjoint_complete_and_ordered() -> None:
    joint = _make_tiny_joint_model()

    groups = build_joint_optimizer_parameter_groups(
        joint,
        ltx_lr=2e-5,
        semantic_lr=1e-4,
        qwen_lr=1e-6,
        planner_head_lr=3e-5,
    )

    assert [group["name"] for group in groups] == [
        "base_ltx",
        "semantic_ltx",
        "qwen",
        "planner_heads",
    ]
    assert [group["lr"] for group in groups] == [2e-5, 1e-4, 1e-6, 3e-5]
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert set(parameter_ids) == {
        id(parameter) for parameter in joint.parameters() if parameter.requires_grad
    }

    ids_by_group = {
        group["name"]: {id(parameter) for parameter in group["params"]}
        for group in groups
    }
    assert id(joint.ltx.base_proj.weight) in ids_by_group["base_ltx"]
    assert id(joint.ltx.semantic_attn.weight) in ids_by_group["semantic_ltx"]
    assert id(joint.planner.model.visual.weight) in ids_by_group["qwen"]
    assert id(joint.planner.model.lm_head.weight) in ids_by_group["qwen"]
    assert id(joint.planner.plan_head.weight) in ids_by_group["planner_heads"]
    assert id(joint.planner.depth_head.weight) in ids_by_group["planner_heads"]
    assert (
        id(joint.planner.plan_embedding_injector.weight)
        in ids_by_group["planner_heads"]
    )
    assert id(joint.planner.shared_query_bank) in ids_by_group["planner_heads"]
    assert id(joint.planner.private_query_bank) in ids_by_group["planner_heads"]


def test_joint_optimizer_groups_reject_cross_group_parameter_aliases() -> None:
    joint = _make_tiny_joint_model()
    joint.planner.plan_head.weight = joint.planner.model.proj.weight

    with pytest.raises(ValueError, match="duplicate trainable parameter"):
        build_joint_optimizer_parameter_groups(
            joint,
            ltx_lr=2e-5,
            semantic_lr=1e-4,
            qwen_lr=1e-6,
            planner_head_lr=3e-5,
        )


def test_joint_optimizer_groups_reject_unclassified_parameters() -> None:
    joint = _make_tiny_joint_model()
    joint.unclassified = nn.Parameter(torch.ones(1))

    with pytest.raises(ValueError, match="missing trainable parameter"):
        build_joint_optimizer_parameter_groups(
            joint,
            ltx_lr=2e-5,
            semantic_lr=1e-4,
            qwen_lr=1e-6,
            planner_head_lr=3e-5,
        )
