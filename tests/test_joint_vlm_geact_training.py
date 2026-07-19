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
    _require_tensor_shape,
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
    def __init__(self, tokens_per_keyframe: int = TOKENS_PER_KEYFRAME) -> None:
        super().__init__()
        self.tokens_per_keyframe = int(tokens_per_keyframe)
        self.model = TinyQwenModel()
        self.plan_head = nn.Linear(1, 1, bias=False)
        self.depth_head = nn.Linear(1, 1, bias=False)
        self.plan_embedding_injector = nn.Embedding(2, 1)
        self.shared_query_bank = nn.Parameter(torch.ones(1, 1))
        self.private_query_bank = nn.Parameter(torch.ones(1, 1))
        self.forward_calls = 0
        nn.init.constant_(self.plan_head.weight, 0.25)
        nn.init.constant_(self.depth_head.weight, 0.125)

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
        semantic_base = plan_value.new_zeros(SEMANTIC_DIM)
        semantic_base[1] = 1.0
        semantic_direction = plan_value.new_zeros(SEMANTIC_DIM)
        semantic_direction[0] = 1.0
        semantic_features = semantic_base.unsqueeze(0)
        semantic_features = (
            semantic_features + plan_value * semantic_direction.unsqueeze(0)
        )
        semantic = semantic_features.reshape(batch_size, 1, 1, SEMANTIC_DIM).expand(
            batch_size,
            2,
            NUM_KEYFRAMES * self.tokens_per_keyframe,
            SEMANTIC_DIM,
        )
        depth = depth_value.reshape(batch_size, 1, 1, 1, 1).expand(
            batch_size,
            2,
            NUM_KEYFRAMES * self.tokens_per_keyframe,
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


class TinyGateOpenLTX(nn.Module):
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


def _planner_labels(
    batch_size: int = 1,
    tokens_per_keyframe: int = TOKENS_PER_KEYFRAME,
) -> tuple[torch.Tensor, torch.Tensor]:
    semantic = torch.zeros(
        batch_size,
        2,
        NUM_KEYFRAMES * tokens_per_keyframe,
        SEMANTIC_DIM,
    )
    depth = torch.zeros(
        batch_size,
        2,
        DEPTH_LAYERS,
        NUM_KEYFRAMES * tokens_per_keyframe,
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


def _real_ltx_inputs(batch_size: int = 1) -> dict[str, object]:
    batch_views = batch_size * 2
    return {
        "prompt_embeds": torch.randn(batch_size, 3, 8),
        "prompt_attention_mask": torch.ones(batch_size, 3),
        "noisy_latents": torch.randn(batch_views, 2, 4),
        "timesteps": torch.ones(batch_views, 2),
        "num_frames": 2,
        "height": 1,
        "width": 1,
        "n_view": 2,
        "frame_rate": 5,
        "temporal_compression_ratio": 8,
        "spatial_compression_ratio": 32,
        "semantic_plan_times": torch.tensor([[0.25, 0.5, 0.75, 1.0]]).repeat(
            batch_views, 1
        ),
        "semantic_condition_mask": torch.ones(batch_views),
    }


@pytest.fixture(scope="module")
def real_ltx_class():
    try:
        from models.ltx_models.transformer_ltx_multiview import (  # noqa: PLC0415
            LTXVideoTransformer3DModel,
        )
    except (ImportError, RuntimeError) as error:
        if "HybridCache" in str(error):
            pytest.skip(f"local diffusers/transformers mismatch: {error}")
        raise
    return LTXVideoTransformer3DModel


def _make_real_zero_gate_joint(real_ltx_class) -> JointVLMGEActModel:
    torch.manual_seed(17)
    ltx = real_ltx_class(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=6,
        cross_attention_dim=12,
        caption_channels=8,
        num_layers=1,
        semantic_plan_context=True,
        semantic_plan_in_dim=SEMANTIC_DIM,
        semantic_plan_coordinate_dim=4,
        semantic_plan_num_keyframes=NUM_KEYFRAMES,
        semantic_plan_num_views=2,
        semantic_plan_adaln_rank=4,
    )
    return _make_tiny_joint_model(ltx=ltx, tokens_per_keyframe=4)


def _make_tiny_joint_model(
    *,
    ltx: nn.Module | None = None,
    tokens_per_keyframe: int = TOKENS_PER_KEYFRAME,
) -> JointVLMGEActModel:
    return JointVLMGEActModel(
        TinyPlanner(tokens_per_keyframe=tokens_per_keyframe),
        TinyGateOpenLTX() if ltx is None else ltx,
        num_keyframes=NUM_KEYFRAMES,
        tokens_per_keyframe=tokens_per_keyframe,
    )


def _run_joint(
    joint: JointVLMGEActModel,
    *,
    ltx_inputs: dict[str, object] | None = None,
):
    semantic_labels, depth_labels = _planner_labels(
        tokens_per_keyframe=joint.tokens_per_keyframe
    )
    return joint(
        planner_inputs={"input_ids": torch.ones(1, 1, dtype=torch.long)},
        semantic_labels=semantic_labels,
        depth_labels=depth_labels,
        ltx_inputs=_ltx_inputs() if ltx_inputs is None else ltx_inputs,
    )


def test_gate_open_tiny_contract_reaches_qwen_and_semantic_parameters() -> None:
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


def test_real_ltx_first_video_backward_only_opens_zero_gate(real_ltx_class) -> None:
    joint = _make_real_zero_gate_joint(real_ltx_class)
    block = joint.ltx.transformer_blocks[0]
    gate = block.semantic_modulation[-1].weight
    assert torch.count_nonzero(gate) == 0

    output = _run_joint(joint, ltx_inputs=_real_ltx_inputs())
    output.ltx_predictions["video"].square().mean().backward()

    assert _finite_nonzero(gate.grad)
    assert not _finite_nonzero(block.semantic_attn.to_k.weight.grad)
    assert not _finite_nonzero(joint.planner.model.proj.weight.grad)


def test_real_ltx_first_combined_backward_updates_qwen_via_alignment(
    real_ltx_class,
) -> None:
    joint = _make_real_zero_gate_joint(real_ltx_class)

    output = _run_joint(joint, ltx_inputs=_real_ltx_inputs())
    video_loss = output.ltx_predictions["video"].square().mean()
    qwen_weight = joint.planner.model.proj.weight
    video_qwen_grad = torch.autograd.grad(
        video_loss,
        qwen_weight,
        retain_graph=True,
        allow_unused=True,
    )[0]
    alignment_qwen_grad = torch.autograd.grad(
        output.planner_losses["loss"],
        qwen_weight,
        retain_graph=True,
    )[0]
    assert not _finite_nonzero(video_qwen_grad)
    assert _finite_nonzero(alignment_qwen_grad)

    combined_loss = video_loss + 0.1 * output.planner_losses["loss"]
    combined_loss.backward()

    assert _finite_nonzero(qwen_weight.grad)
    gate_grad = joint.ltx.transformer_blocks[0].semantic_modulation[-1].weight.grad
    assert _finite_nonzero(gate_grad)


def test_real_ltx_second_video_backward_reaches_attention_and_qwen(
    real_ltx_class,
) -> None:
    joint = _make_real_zero_gate_joint(real_ltx_class)
    block = joint.ltx.transformer_blocks[0]
    gate = block.semantic_modulation[-1].weight

    first_output = _run_joint(joint, ltx_inputs=_real_ltx_inputs())
    first_output.ltx_predictions["video"].square().mean().backward()
    assert _finite_nonzero(gate.grad)
    with torch.no_grad():
        gate.add_(gate.grad, alpha=-1e-2)
    joint.zero_grad(set_to_none=True)

    second_output = _run_joint(joint, ltx_inputs=_real_ltx_inputs())
    second_output.ltx_predictions["video"].square().mean().backward()

    assert _finite_nonzero(block.semantic_attn.to_k.weight.grad)
    assert _finite_nonzero(joint.planner.model.proj.weight.grad)


def test_joint_documents_zero_gate_warmup_contract() -> None:
    doc = JointVLMGEActModel.__doc__ or ""

    assert "zero-initialized semantic gates" in doc
    assert "planner alignment loss" in doc


def test_shape_validation_does_not_scan_tensor_values() -> None:
    value = torch.tensor([float("nan")])

    assert _require_tensor_shape(value, (1,), name="value") is value


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
