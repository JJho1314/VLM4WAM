from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file


GE_ACT_ROOT = Path(__file__).resolve().parents[1] / "ge_act"
if str(GE_ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(GE_ACT_ROOT))

from runner.ge_trainer import (
    build_optimizer_parameter_groups,
    compute_effective_video_fps,
    compute_ltx_latent_frames,
    freeze_conditioning_modules,
    sample_semantic_condition_mask,
    should_save_checkpoint,
)
from utils.model_utils import forward_pass, resolve_checkpoint_files
from models.ltx_models.transformer_ltx_multiview import LTXVideoTransformer3DModel
from runner import ge_trainer as ge_trainer_module


class TinySemanticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj_in = nn.Linear(2, 2)
        self.semantic_adapter = nn.Linear(2, 2)
        self.semantic_attn = nn.Linear(2, 2)
        self.action_proj = nn.Linear(2, 2)


def test_deepspeed_config_preserves_requested_gradient_accumulation() -> None:
    config = ge_trainer_module.build_deepspeed_batch_config(
        {"zero_optimization": {"stage": 2}},
        per_device_batch_size=2,
        world_size=8,
        gradient_accumulation_steps=8,
    )

    assert config["train_batch_size"] == 128
    assert config["gradient_accumulation_steps"] == 8


def test_optimizer_uses_separate_base_and_semantic_learning_rates() -> None:
    model = TinySemanticModel()
    groups = build_optimizer_parameter_groups(
        model,
        train_mode="video_only",
        base_lr=2e-5,
        semantic_lr=1e-4,
    )

    assert [group["name"] for group in groups] == ["base_ltx", "semantic"]
    assert [group["lr"] for group in groups] == [2e-5, 1e-4]
    semantic_ids = {id(param) for name, param in model.named_parameters() if "semantic_" in name}
    assert {id(param) for param in groups[1]["params"]} == semantic_ids
    assert not model.action_proj.weight.requires_grad


def test_conditioning_encoders_are_explicitly_frozen() -> None:
    text_encoder = nn.Linear(2, 2)
    vae = nn.Linear(2, 2)
    siglip = type("Encoder", (), {"model": nn.Linear(2, 2)})()

    freeze_conditioning_modules(text_encoder, vae, siglip)

    assert all(not parameter.requires_grad for parameter in text_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in vae.parameters())
    assert all(not parameter.requires_grad for parameter in siglip.model.parameters())
    assert not text_encoder.training
    assert not vae.training
    assert not siglip.model.training


def test_fastwam_effective_fps_and_latent_length() -> None:
    data_config = {"source_fps": 20, "action_chunk": 36, "chunk": 9, "n_previous": 4}

    assert compute_effective_video_fps(data_config) == 5.0
    assert compute_ltx_latent_frames(9, temporal_compression_ratio=8, n_previous=4) == 6


def test_semantic_dropout_is_shared_between_views() -> None:
    generator = torch.Generator().manual_seed(11)
    mask = sample_semantic_condition_mask(
        batch_size=8,
        n_view=2,
        dropout_probability=0.5,
        generator=generator,
    )

    assert mask.shape == (16,)
    assert torch.equal(mask.reshape(8, 2)[:, 0], mask.reshape(8, 2)[:, 1])
    assert set(mask.tolist()) == {0.0, 1.0}


def test_forward_pass_uses_real_frame_rate_and_forwards_semantics() -> None:
    captured = {}

    class CaptureModel:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return ({"video": kwargs["hidden_states"]},)

    semantic_plan = torch.randn(1, 2, 4, 256, 8)
    semantic_times = torch.tensor([[0.8, 0.875, 0.925, 1.0]]).repeat(2, 1)
    forward_pass(
        model=CaptureModel(),
        prompt_embeds=torch.randn(1, 2, 4),
        prompt_attention_mask=torch.ones(1, 2),
        noisy_latents=torch.randn(2, 6, 4),
        timesteps=torch.ones(2, 6),
        num_frames=6,
        height=1,
        width=1,
        n_view=2,
        frame_rate=5,
        temporal_compression_ratio=8,
        semantic_plan=semantic_plan,
        semantic_plan_times=semantic_times,
    )

    assert captured["rope_interpolation_scale"][0] == 1.6
    assert captured["semantic_plan"] is semantic_plan
    assert captured["semantic_plan_times"] is semantic_times


def test_checkpoint_directory_accepts_single_safetensors_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "diffusion_pytorch_model.safetensors"
    save_file({"weight": torch.ones(1)}, checkpoint)

    assert resolve_checkpoint_files(tmp_path) == [checkpoint]


def test_explicit_save_steps_override_periodic_checkpointing() -> None:
    args = type("Args", (), {"save_steps": [20_000, 25_000, 30_000], "steps_to_save": 5_000})()

    assert not should_save_checkpoint(5_000, args)
    assert should_save_checkpoint(20_000, args)
    assert should_save_checkpoint(25_000, args)
    assert should_save_checkpoint(30_000, args)


@torch.no_grad()
def _tiny_semantic_inputs():
    return {
        "hidden_states": torch.randn(2, 2, 4),
        "encoder_hidden_states": torch.randn(1, 3, 8),
        "timestep": torch.ones(2, 2),
        "encoder_attention_mask": torch.ones(1, 3),
        "n_view": 2,
        "rope_interpolation_scale": (1.6, 32.0, 32.0),
        "num_frames": 2,
        "height": 1,
        "width": 1,
        "semantic_plan": torch.randn(1, 2, 2, 4, 8),
        "semantic_plan_times": torch.tensor([[0.5, 1.0]]).repeat(2, 1),
        "return_dict": False,
    }


def test_semantic_model_backward_reaches_base_and_zero_gate_with_checkpointing() -> None:
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
        semantic_plan_adaln_rank=4,
    )
    model.enable_gradient_checkpointing()
    output = model(**_tiny_semantic_inputs())[0]["video"]
    output.square().mean().backward()

    assert model.proj_in.weight.grad is not None
    gate_grad = model.transformer_blocks[0].semantic_modulation[-1].weight.grad
    assert gate_grad is not None
    assert torch.count_nonzero(gate_grad) > 0
