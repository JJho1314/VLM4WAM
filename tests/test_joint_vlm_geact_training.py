from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn
from yaml import Loader, load


GE_ACT_ROOT = Path(__file__).resolve().parents[1] / "ge_act"
if str(GE_ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(GE_ACT_ROOT))
GE_TRAINER_PATH = GE_ACT_ROOT / "runner" / "ge_trainer.py"
GE_MAIN_PATH = GE_ACT_ROOT / "main.py"

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


def _load_ge_trainer_symbols(*names: str) -> SimpleNamespace:
    """Load selected light contracts without importing the broken local diffusers."""

    tree = ast.parse(GE_TRAINER_PATH.read_text(encoding="utf-8"))
    requested = set(names)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in requested
    ]
    found = {node.name for node in nodes}
    assert found == requested, f"missing trainer contracts: {sorted(requested - found)}"
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Accelerator": object,
        "DistributedType": SimpleNamespace(DEEPSPEED="deepspeed"),
        "JointVLMGEActModel": JointVLMGEActModel,
        "Path": Path,
        "SummaryWriter": lambda *_args, **_kwargs: None,
        "argparse": argparse,
        "build_joint_optimizer_parameter_groups": (
            build_joint_optimizer_parameter_groups
        ),
        "datetime": datetime,
        "deepcopy": deepcopy,
        "dist": SimpleNamespace(broadcast=lambda *_args, **_kwargs: None),
        "json": json,
        "load": load,
        "Loader": Loader,
        "logger": SimpleNamespace(info=lambda *_args, **_kwargs: None),
        "math": math,
        "os": os,
        "torch": torch,
        "init_logging": lambda *_args, **_kwargs: None,
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), GE_TRAINER_PATH, "exec"),
        namespace,
    )
    return SimpleNamespace(**namespace)


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


def test_joint_teacher_frames_match_planner_training_offsets() -> None:
    contracts = _load_ge_trainer_symbols("select_joint_planner_frames")
    video = torch.arange(1 * 3 * 2 * 13 * 2 * 2).reshape(1, 3, 2, 13, 2, 2)

    current, future = contracts.select_joint_planner_frames(
        video,
        n_previous=4,
        offsets=(2, 4, 6, 8),
    )

    torch.testing.assert_close(
        current,
        video[:, :, :, 3].permute(0, 2, 3, 4, 1),
    )
    for keyframe, source_index in enumerate((6, 8, 10, 12)):
        torch.testing.assert_close(
            future[:, :, keyframe],
            video[:, :, :, source_index].permute(0, 2, 3, 4, 1),
        )
    leaked_indices = (5, 7, 9, 11)
    assert all(
        not torch.equal(
            future[:, :, keyframe], video[:, :, :, leaked].permute(0, 2, 3, 4, 1)
        )
        for keyframe, leaked in enumerate(leaked_indices)
    )


def test_joint_teacher_frames_reject_boundary_instead_of_leaking_wrong_frame() -> None:
    contracts = _load_ge_trainer_symbols("select_joint_planner_frames")
    video = torch.zeros(1, 3, 2, 12, 2, 2)

    with pytest.raises(ValueError, match=r"source index 12.*T=12"):
        contracts.select_joint_planner_frames(
            video,
            n_previous=4,
            offsets=(2, 4, 6, 8),
        )


def test_joint_teacher_targets_are_encoded_under_no_grad() -> None:
    contracts = _load_ge_trainer_symbols("encode_joint_planner_targets")
    grad_states: list[bool] = []

    def target_encoder(current, future, *, appearance_encoder, depth_encoder):
        grad_states.append(torch.is_grad_enabled())
        assert appearance_encoder == "siglip"
        assert depth_encoder == "da3"
        return {
            "semantic_plan_labels": future.new_zeros(1, 2, 1024, 1024),
            "depth_plan_labels": future.new_zeros(1, 2, 4, 1024, 2048),
        }

    current = torch.zeros(1, 2, 2, 2, 3, requires_grad=True)
    future = torch.zeros(1, 2, 4, 2, 2, 3, requires_grad=True)
    targets = contracts.encode_joint_planner_targets(
        current,
        future,
        semantic_teacher="siglip",
        depth_teacher="da3",
        target_encoder=target_encoder,
    )

    assert grad_states == [False]
    assert all(not value.requires_grad for value in targets.values())


def test_joint_loss_uses_configured_planner_weight() -> None:
    contracts = _load_ge_trainer_symbols("combine_joint_training_loss")
    video_loss = torch.tensor(2.0, requires_grad=True)
    planner_loss = torch.tensor(3.0, requires_grad=True)

    total = contracts.combine_joint_training_loss(
        video_loss,
        {"loss": planner_loss},
        planner_loss_weight=0.1,
    )

    torch.testing.assert_close(total, torch.tensor(2.3))
    total.backward()
    torch.testing.assert_close(video_loss.grad, torch.tensor(1.0))
    torch.testing.assert_close(planner_loss.grad, torch.tensor(0.1))


def test_joint_teacher_parameters_are_frozen_and_excluded() -> None:
    contracts = _load_ge_trainer_symbols(
        "State",
        "Trainer",
        "_configure_qwen_gradient_checkpointing",
        "_joint_training_enabled",
        "compute_effective_video_fps",
        "freeze_conditioning_modules",
    )
    trainer = contracts.Trainer.__new__(contracts.Trainer)
    planner = TinyPlanner()
    planner.requires_grad_(False)
    planner.eval()
    planner.model.gradient_checkpointing_enable = lambda **_kwargs: setattr(
        planner.model,
        "is_gradient_checkpointing",
        True,
    )
    planner.model.enable_input_require_grads = lambda: setattr(
        planner.model,
        "input_grads_enabled",
        True,
    )
    ltx = TinyGateOpenLTX()
    ltx.enable_gradient_checkpointing = lambda: setattr(
        ltx,
        "gradient_checkpointing_enabled",
        True,
    )
    trainer.semantic_planner = SimpleNamespace(wrapper=planner)
    trainer.semantic_teacher = nn.Linear(2, 2)
    trainer.depth_teacher = nn.Linear(2, 2)
    trainer.semantic_encoder = None
    trainer.text_encoder = nn.Linear(2, 2)
    trainer.vae = nn.Linear(2, 2)
    trainer.diffusion_model = ltx
    trainer.joint_model = JointVLMGEActModel(
        planner,
        ltx,
        num_keyframes=4,
        tokens_per_keyframe=TOKENS_PER_KEYFRAME,
    )
    trainer.args = SimpleNamespace(
        joint_training={
            "enabled": True,
            "qwen_gradient_checkpointing": True,
            "qwen_lr": 1e-6,
            "planner_head_lr": 3e-5,
        },
        gradient_checkpointing=True,
        allow_tf32=False,
        train_epochs=1,
        train_steps=1,
        mixed_precision="bf16",
        lr=2e-5,
        semantic_lr=1e-4,
        scale_lr=False,
        gradient_accumulation_steps=1,
        batch_size=1,
        optimizer="adamw",
        beta1=0.9,
        beta2=0.95,
        beta3=None,
        epsilon=1e-8,
        weight_decay=1e-5,
        optimizer_8bit=False,
        optimizer_torchao=False,
        lr_scheduler="constant",
        lr_warmup_steps=1000,
        lr_num_cycles=1,
        lr_power=1.0,
        max_grad_norm=1.0,
    )
    trainer.state = SimpleNamespace(
        weight_dtype=torch.bfloat16,
        accelerator=SimpleNamespace(num_processes=1),
    )
    trainer.train_dataloader = [object()]

    captured: dict[str, object] = {}

    class FakeOptimizer:
        def __init__(self, groups):
            self.param_groups = groups

    method_globals = contracts.Trainer.prepare_optimizer.__globals__
    method_globals["get_optimizer"] = lambda *, params_to_optimize, **_kwargs: (
        captured.setdefault("groups", params_to_optimize)
        or FakeOptimizer(params_to_optimize)
    )
    method_globals["get_scheduler"] = lambda **_kwargs: object()
    method_globals["cast_training_params"] = lambda *_args, **_kwargs: None

    trainer.prepare_trainable_parameters()
    trainer.prepare_optimizer()

    assert all(not p.requires_grad for p in trainer.semantic_teacher.parameters())
    assert all(not p.requires_grad for p in trainer.depth_teacher.parameters())
    assert not trainer.semantic_teacher.training
    assert not trainer.depth_teacher.training
    assert planner.training
    assert all(parameter.requires_grad for parameter in planner.parameters())
    assert planner.model.is_gradient_checkpointing
    assert planner.model.input_grads_enabled
    assert ltx.gradient_checkpointing_enabled
    groups = captured["groups"]
    assert [group["name"] for group in groups] == [
        "base_ltx",
        "semantic_ltx",
        "qwen",
        "planner_heads",
    ]
    assert [group["lr"] for group in groups] == [2e-5, 1e-4, 1e-6, 3e-5]
    optimized_ids = {id(parameter) for group in groups for parameter in group["params"]}
    teacher_ids = {
        id(parameter)
        for teacher in (trainer.semantic_teacher, trainer.depth_teacher)
        for parameter in teacher.parameters()
    }
    assert optimized_ids.isdisjoint(teacher_ids)


def test_joint_prepare_passes_exactly_one_composite_model() -> None:
    contracts = _load_ge_trainer_symbols("State", "Trainer")
    trainer = contracts.Trainer.__new__(contracts.Trainer)
    trainer.joint_model = _make_tiny_joint_model()
    trainer.diffusion_model = trainer.joint_model.ltx
    trainer.optimizer = object()
    trainer.train_dataloader = object()
    trainer.lr_scheduler = object()
    trainer.args = SimpleNamespace(joint_training={"enabled": True})
    calls: list[tuple[object, ...]] = []

    def prepare(*values):
        calls.append(values)
        return values

    trainer.state = SimpleNamespace(accelerator=SimpleNamespace(prepare=prepare))
    contracts.Trainer.prepare_for_training.__globals__["_joint_training_enabled"] = (
        lambda args: bool(args.joint_training["enabled"])
    )

    trainer.prepare_for_training()

    assert calls == [
        (
            trainer.joint_model,
            trainer.optimizer,
            trainer.train_dataloader,
            trainer.lr_scheduler,
        )
    ]


def test_trainer_overrides_are_visible_during_distributed_initialization(
    tmp_path: Path,
) -> None:
    contracts = _load_ge_trainer_symbols(
        "State",
        "Trainer",
        "_synchronize_save_folder",
        "compute_effective_video_fps",
    )
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(
        "\n".join(
            (
                "lr: 2.0e-5",
                "semantic_lr: 1.0e-4",
                "epsilon: 1.0e-8",
                "weight_decay: 1.0e-5",
                "batch_size: 8",
                "gradient_accumulation_steps: 16",
                "use_deepspeed: true",
                "load_weights: true",
                f"output_dir: {tmp_path / 'out'}",
                "model_name: tiny",
                "data:",
                "  train:",
                "    source_fps: 20",
                "    chunk: 9",
                "    action_chunk: 36",
            )
        ),
        encoding="utf-8",
    )

    class TinyTrainer(contracts.Trainer):
        def _init_distributed(self):
            self.observed_during_init = (
                self.args.batch_size,
                self.args.gradient_accumulation_steps,
                self.args.use_deepspeed,
            )
            self.state.accelerator = SimpleNamespace(
                is_main_process=True,
                device=torch.device("cpu"),
                process_index=0,
            )

        def _init_logging(self):
            return None

        def _init_directories_and_repositories(self):
            return None

    trainer = TinyTrainer(
        config_path,
        config_overrides={
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "use_deepspeed": False,
        },
        to_log=False,
    )

    assert trainer.observed_during_init == (1, 1, False)


def test_single_process_constructor_does_not_touch_default_process_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contracts = _load_ge_trainer_symbols(
        "State",
        "Trainer",
        "_synchronize_save_folder",
        "compute_effective_video_fps",
    )
    config_path = tmp_path / "single-process.yaml"
    config_path.write_text(
        "\n".join(
            (
                "lr: 2.0e-5",
                "semantic_lr: 1.0e-4",
                "epsilon: 1.0e-8",
                "weight_decay: 1.0e-5",
                "batch_size: 1",
                "gradient_accumulation_steps: 1",
                "use_deepspeed: false",
                "load_weights: true",
                f"output_dir: {tmp_path / 'out'}",
                "model_name: tiny",
                "data:",
                "  train:",
                "    source_fps: 20",
                "    chunk: 9",
                "    action_chunk: 36",
            )
        ),
        encoding="utf-8",
    )

    def unexpected_collective(*_args, **_kwargs):
        raise AssertionError("single-process constructor called dist.broadcast")

    monkeypatch.setattr(torch.distributed, "broadcast", unexpected_collective)
    contracts.Trainer.__init__.__globals__["dist"] = torch.distributed
    contracts._synchronize_save_folder.__globals__["dist"] = torch.distributed

    class SingleProcessTrainer(contracts.Trainer):
        def _init_distributed(self):
            self.state.accelerator = SimpleNamespace(
                is_main_process=True,
                num_processes=1,
                device=torch.device("cpu"),
                process_index=0,
            )

        def _init_logging(self):
            return None

        def _init_directories_and_repositories(self):
            return None

    trainer = SingleProcessTrainer(config_path, to_log=False)

    assert Path(trainer.save_folder).is_dir()


def test_save_folder_sync_skips_uninitialized_dist_and_broadcasts_when_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contracts = _load_ge_trainer_symbols("_synchronize_save_folder")
    accelerator = SimpleNamespace(
        is_main_process=True,
        num_processes=2,
        device=torch.device("cpu"),
    )

    def unexpected_collective(*_args, **_kwargs):
        raise AssertionError("uninitialized process group used a collective")

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "broadcast", unexpected_collective)
    contracts._synchronize_save_folder.__globals__["dist"] = torch.distributed
    assert contracts._synchronize_save_folder(accelerator, "/tmp/run") == "/tmp/run"

    class FakeInitializedDist:
        def __init__(self) -> None:
            self.payloads: list[torch.Tensor] = []

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_initialized() -> bool:
            return True

        def broadcast(self, value: torch.Tensor, *, src: int) -> None:
            assert src == 0
            self.payloads.append(value.detach().clone())

    fake_dist = FakeInitializedDist()
    contracts._synchronize_save_folder.__globals__["dist"] = fake_dist

    assert contracts._synchronize_save_folder(accelerator, "/tmp/run") == "/tmp/run"
    assert len(fake_dist.payloads) == 2
    assert int(fake_dist.payloads[0].item()) == len("/tmp/run".encode())
    assert bytes(fake_dist.payloads[1].tolist()).decode() == "/tmp/run"


def test_zero2_gradient_norm_reduces_squared_sum_before_sqrt() -> None:
    contracts = _load_ge_trainer_symbols("_global_gradient_norm")
    first = nn.Parameter(torch.zeros(1))
    second = nn.Parameter(torch.zeros(1))
    first.grad = torch.tensor([3.0])
    second.grad = torch.tensor([4.0])
    reductions: list[tuple[torch.Tensor, str]] = []

    class FakeAccelerator:
        device = torch.device("cpu")

        def reduce(self, value: torch.Tensor, *, reduction: str) -> torch.Tensor:
            reductions.append((value.detach().clone(), reduction))
            return value + 144.0

    norm = contracts._global_gradient_norm(
        (first, second),
        accelerator=FakeAccelerator(),
    )

    torch.testing.assert_close(norm, torch.tensor(13.0))
    assert len(reductions) == 1
    torch.testing.assert_close(reductions[0][0], torch.tensor(25.0))
    assert reductions[0][1] == "sum"


def test_joint_validation_disables_dropout_and_restores_training_on_error() -> None:
    contracts = _load_ge_trainer_symbols("_run_joint_validation")

    class Composite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.planner = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))

    composite = Composite().train()

    class ValidationError(RuntimeError):
        pass

    def validation_call():
        assert not composite.training
        assert not composite.planner.training
        raise ValidationError("stop after observing eval mode")

    with pytest.raises(ValidationError, match="observing eval mode"):
        contracts._run_joint_validation(composite, validation_call)

    assert composite.training
    assert composite.planner.training


def test_positive_lm_plan_weight_rejects_bidirectional_attention() -> None:
    contracts = _load_ge_trainer_symbols("_configure_joint_lm_plan_objective")
    wrapper = SimpleNamespace(
        lm_plan_loss_weight=0.0,
        bidirectional_plan_attn=True,
    )

    with pytest.raises(
        ValueError,
        match="causal plan-token CE.*bidirectional_plan_attn.*label leakage",
    ):
        contracts._configure_joint_lm_plan_objective(
            wrapper,
            {"lm_plan_loss_weight": 1e-3},
        )

    assert wrapper.lm_plan_loss_weight == 1e-3


def test_zero_lm_plan_weight_allows_bidirectional_legacy_path() -> None:
    contracts = _load_ge_trainer_symbols("_configure_joint_lm_plan_objective")
    wrapper = SimpleNamespace(
        lm_plan_loss_weight=1.0,
        bidirectional_plan_attn=True,
    )

    contracts._configure_joint_lm_plan_objective(
        wrapper,
        {"lm_plan_loss_weight": 0.0},
    )

    assert wrapper.lm_plan_loss_weight == 0.0


def test_main_smoke_flags_are_forwarded_as_constructor_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location("ge_act_main_task4", GE_MAIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, config_file, *, config_overrides=None):
            captured["config_file"] = config_file
            captured["config_overrides"] = config_overrides
            self.args = SimpleNamespace(train_steps=99)

        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: None

    monkeypatch.setattr(module, "import_custom_class", lambda *_args: FakeRunner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ge_act/main.py",
            "--config_file",
            str(tmp_path / "config.yaml"),
            "--max_train_steps",
            "1",
            "--batch_size_override",
            "1",
            "--gradient_accumulation_steps_override",
            "1",
            "--disable_deepspeed",
        ],
    )

    module.main()

    assert captured["config_overrides"] == {
        "train_steps": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "use_deepspeed": False,
    }


class _SaveableModule(nn.Module):
    def __init__(self, marker: str) -> None:
        super().__init__()
        self.marker = marker
        self.weight = nn.Parameter(torch.ones(1))

    def save_pretrained(self, path, **_kwargs) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{self.marker}.txt").write_text(self.marker, encoding="utf-8")


def test_joint_checkpoint_exports_both_models_metadata_and_training_state(
    tmp_path: Path,
) -> None:
    contracts = _load_ge_trainer_symbols(
        "_export_joint_planner",
        "save_joint_checkpoint",
    )
    source_planner = tmp_path / "source_planner"
    source_planner.mkdir()
    (source_planner / "planner_meta.json").write_text(
        json.dumps(
            {
                "future_keyframe_offsets": [2, 4, 6, 8],
                "num_keyframes": 4,
                "target_tokens_per_keyframe": 256,
            }
        ),
        encoding="utf-8",
    )

    planner = TinyPlanner()
    planner.lm_plan_loss_weight = 7e-4
    planner.model = _SaveableModule("qwen")
    provider = SimpleNamespace(
        wrapper=planner,
        processor=_SaveableModule("processor"),
    )
    composite = SimpleNamespace(
        ltx=_SaveableModule("ltx"),
        planner=planner,
    )
    saved_states: list[Path] = []
    accelerator = SimpleNamespace(
        is_main_process=True,
        num_processes=8,
        unwrap_model=lambda model: model,
        wait_for_everyone=lambda: None,
        save_state=lambda path: saved_states.append(Path(path)),
    )
    args = SimpleNamespace(
        semantic_plan={"planner_checkpoint": str(source_planner)},
        diffusion_model={"model_path": "/checkpoints/ltx_step_50000"},
        joint_training={
            "planner_loss_weight": 0.1,
            "lm_plan_loss_weight": 1e-3,
            "qwen_lr": 1e-6,
            "planner_head_lr": 3e-5,
        },
        lr=2e-5,
        semantic_lr=1e-4,
        batch_size=1,
        gradient_accumulation_steps=16,
    )
    optimizer = SimpleNamespace(
        param_groups=[
            {"name": "base_ltx", "lr": 4e-5},
            {"name": "semantic_ltx", "lr": 2e-4},
            {"name": "qwen", "lr": 2e-6},
            {"name": "planner_heads", "lr": 6e-5},
        ]
    )
    step_dir = tmp_path / "step_20000"

    contracts.save_joint_checkpoint(
        accelerator=accelerator,
        joint_model=composite,
        planner_provider=provider,
        optimizer=optimizer,
        step_dir=step_dir,
        args=args,
        global_step=20000,
    )

    assert (step_dir / "ltx" / "ltx.txt").is_file()
    assert (step_dir / "planner" / "qwen3vl_lora_or_model" / "qwen.txt").is_file()
    assert (step_dir / "planner" / "processor" / "processor.txt").is_file()
    for filename in (
        "plan_head.pt",
        "depth_head.pt",
        "plan_token_embedding.pt",
        "planner_meta.json",
    ):
        assert (step_dir / "planner" / filename).is_file()
    joint_meta = json.loads((step_dir / "joint_meta.json").read_text(encoding="utf-8"))
    assert joint_meta["global_step"] == 20000
    assert joint_meta["source_planner_checkpoint"] == str(source_planner)
    assert joint_meta["lm_plan_loss_weight"] == 7e-4
    assert joint_meta["optimizer_group_lrs"] == {
        "base_ltx": 4e-5,
        "semantic_ltx": 2e-4,
        "qwen": 2e-6,
        "planner_heads": 6e-5,
    }
    assert joint_meta["future_keyframe_offsets"] == [2, 4, 6, 8]
    assert saved_states == [step_dir / "training_state"]


def test_joint_checkpoint_calls_save_state_on_non_main_rank(tmp_path: Path) -> None:
    contracts = _load_ge_trainer_symbols(
        "_export_joint_planner",
        "save_joint_checkpoint",
    )
    saved_states: list[Path] = []
    accelerator = SimpleNamespace(
        is_main_process=False,
        num_processes=8,
        wait_for_everyone=lambda: None,
        save_state=lambda path: saved_states.append(Path(path)),
    )

    contracts.save_joint_checkpoint(
        accelerator=accelerator,
        joint_model=object(),
        planner_provider=object(),
        optimizer=object(),
        step_dir=tmp_path / "step_20000",
        args=SimpleNamespace(),
        global_step=20000,
    )

    assert saved_states == [tmp_path / "step_20000" / "training_state"]


def test_joint_train_source_has_single_composite_and_required_logs() -> None:
    source = GE_TRAINER_PATH.read_text(encoding="utf-8")
    required_log_keys = (
        '"loss_video"',
        '"planner_loss"',
        '"planner_semantic_mse"',
        '"planner_depth_wsa_loss"',
        '"vlm_grad_norm"',
        '"ltx_grad_norm"',
        '"lr/base_ltx"',
        '"lr/semantic_ltx"',
        '"lr/qwen"',
        '"lr/planner_heads"',
        '"peak_memory_allocated"',
    )

    assert "accelerator.accumulate(self.joint_model)" in source
    assert "accelerator.clip_grad_norm_(" in source
    assert "self.joint_model.parameters()" in source
    assert (
        "self.joint_model, self.optimizer, self.train_dataloader, self.lr_scheduler"
        in source
    )
    assert "_run_joint_validation(" in source
    assert "_configure_joint_lm_plan_objective(" in source
    for key in required_log_keys:
        assert key in source
