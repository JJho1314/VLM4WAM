"""Composite Qwen planner and GE-Act LTX module for joint training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from utils.model_utils import forward_pass


_NUM_CAMERA_VIEWS = 2
_SEMANTIC_DIM = 1024
_DEPTH_LAYERS = 4
_DEPTH_DIM = 2048


@dataclass
class JointVLMGEActOutput:
    ltx_predictions: dict[str, torch.Tensor]
    semantic_plan: torch.Tensor
    depth_plan: torch.Tensor
    planner_losses: dict[str, torch.Tensor]


def _require_tensor_shape(
    value: Any,
    expected_shape: tuple[int, ...],
    *,
    name: str,
) -> torch.Tensor:
    if not torch.is_tensor(value) or tuple(value.shape) != expected_shape:
        actual_shape = tuple(value.shape) if torch.is_tensor(value) else None
        raise ValueError(f"{name} must have shape {expected_shape}, got {actual_shape}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


class JointVLMGEActModel(nn.Module):
    """Run one planner pass and inject its differentiable K4 plan into LTX."""

    def __init__(
        self,
        planner: nn.Module,
        ltx: nn.Module,
        *,
        num_keyframes: int = 4,
        tokens_per_keyframe: int = 256,
    ) -> None:
        super().__init__()
        if not isinstance(planner, nn.Module):
            raise TypeError("planner must be an nn.Module")
        if not isinstance(ltx, nn.Module):
            raise TypeError("ltx must be an nn.Module")
        self.num_keyframes = int(num_keyframes)
        self.tokens_per_keyframe = int(tokens_per_keyframe)
        if self.num_keyframes != 4:
            raise ValueError(
                f"joint VLM/GE-Act training requires K4, got K={self.num_keyframes}"
            )
        if self.tokens_per_keyframe <= 0:
            raise ValueError("tokens_per_keyframe must be positive")
        self.planner = planner
        self.ltx = ltx

    def _validate_labels(
        self,
        semantic_labels: torch.Tensor,
        depth_labels: torch.Tensor,
    ) -> int:
        if not torch.is_tensor(semantic_labels) or semantic_labels.ndim != 4:
            shape = (
                tuple(semantic_labels.shape)
                if torch.is_tensor(semantic_labels)
                else None
            )
            raise ValueError(f"semantic labels must be [B,2,K*P,1024], got {shape}")
        batch_size = int(semantic_labels.shape[0])
        flat_tokens = self.num_keyframes * self.tokens_per_keyframe
        _require_tensor_shape(
            semantic_labels,
            (batch_size, _NUM_CAMERA_VIEWS, flat_tokens, _SEMANTIC_DIM),
            name="semantic labels",
        )
        _require_tensor_shape(
            depth_labels,
            (
                batch_size,
                _NUM_CAMERA_VIEWS,
                _DEPTH_LAYERS,
                flat_tokens,
                _DEPTH_DIM,
            ),
            name="depth labels",
        )
        return batch_size

    def _validate_ltx_inputs(
        self,
        ltx_inputs: Mapping[str, Any],
        *,
        batch_size: int,
    ) -> dict[str, Any]:
        required = {
            "prompt_embeds",
            "prompt_attention_mask",
            "noisy_latents",
            "timesteps",
            "num_frames",
            "height",
            "width",
            "n_view",
            "semantic_plan_times",
        }
        missing = sorted(required.difference(ltx_inputs))
        if missing:
            raise ValueError(f"ltx_inputs is missing required keys: {missing}")
        reserved = sorted({"model", "semantic_plan"}.intersection(ltx_inputs))
        if reserved:
            raise ValueError(f"ltx_inputs contains reserved keys: {reserved}")

        inputs = dict(ltx_inputs)
        if inputs["n_view"] != _NUM_CAMERA_VIEWS:
            raise ValueError(
                f"ltx_inputs n_view must be {_NUM_CAMERA_VIEWS}, "
                f"got {inputs['n_view']!r}"
            )
        batch_views = batch_size * _NUM_CAMERA_VIEWS
        _require_tensor_shape(
            inputs["semantic_plan_times"],
            (batch_views, self.num_keyframes),
            name="semantic_plan_times",
        )
        semantic_times = inputs["semantic_plan_times"]
        if bool((semantic_times < 0).any() or (semantic_times > 1).any()):
            raise ValueError("semantic_plan_times must be normalized to [0,1]")

        for name, expected_batch in (
            ("prompt_embeds", batch_size),
            ("prompt_attention_mask", batch_size),
            ("noisy_latents", batch_views),
            ("timesteps", batch_views),
        ):
            value = inputs[name]
            if (
                not torch.is_tensor(value)
                or value.ndim == 0
                or value.shape[0] != expected_batch
            ):
                shape = tuple(value.shape) if torch.is_tensor(value) else None
                raise ValueError(
                    f"{name} must have leading dimension {expected_batch}, got {shape}"
                )
        if not torch.is_floating_point(inputs["noisy_latents"]):
            raise ValueError("noisy_latents must be floating point")
        condition_mask = inputs.get("semantic_condition_mask")
        if condition_mask is not None:
            _require_tensor_shape(
                condition_mask,
                (batch_views,),
                name="semantic_condition_mask",
            )
        for name in ("num_frames", "height", "width"):
            value = inputs[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        prompt_embeds = inputs["prompt_embeds"]
        prompt_attention_mask = inputs["prompt_attention_mask"]
        if prompt_embeds.ndim != 3:
            raise ValueError(
                f"prompt_embeds must be [B,L,D], got {tuple(prompt_embeds.shape)}"
            )
        if tuple(prompt_attention_mask.shape) != tuple(prompt_embeds.shape[:2]):
            raise ValueError(
                "prompt_attention_mask must match prompt_embeds [B,L], got "
                f"{tuple(prompt_attention_mask.shape)} and {tuple(prompt_embeds.shape)}"
            )
        noisy_latents = inputs["noisy_latents"]
        if noisy_latents.ndim != 3:
            raise ValueError(
                f"noisy_latents must be [B*V,F*H*W,C], got {tuple(noisy_latents.shape)}"
            )
        expected_latent_tokens = (
            inputs["num_frames"] * inputs["height"] * inputs["width"]
        )
        if noisy_latents.shape[1] != expected_latent_tokens:
            raise ValueError(
                "noisy_latents token dimension must equal num_frames*height*width: "
                f"{noisy_latents.shape[1]} != {expected_latent_tokens}"
            )
        timesteps = inputs["timesteps"]
        valid_timestep_shapes = {
            (batch_views, inputs["num_frames"]),
            (batch_views, expected_latent_tokens),
        }
        if tuple(timesteps.shape) not in valid_timestep_shapes:
            raise ValueError(
                "timesteps must be [B*V,F] or [B*V,F*H*W], got "
                f"{tuple(timesteps.shape)}"
            )
        for name in ("prompt_embeds", "semantic_plan_times", "timesteps"):
            if inputs[name].device != noisy_latents.device:
                raise ValueError(f"{name} and noisy_latents must be on the same device")
        return inputs

    def forward(
        self,
        *,
        planner_inputs: Mapping[str, Any],
        semantic_labels: torch.Tensor,
        depth_labels: torch.Tensor,
        ltx_inputs: Mapping[str, Any],
    ) -> JointVLMGEActOutput:
        if not isinstance(planner_inputs, Mapping):
            raise TypeError("planner_inputs must be a mapping")
        if not isinstance(ltx_inputs, Mapping):
            raise TypeError("ltx_inputs must be a mapping")
        if (
            "semantic_plan_labels" in planner_inputs
            or "depth_plan_labels" in planner_inputs
        ):
            raise ValueError(
                "planner_inputs must not override planner supervision labels"
            )

        batch_size = self._validate_labels(semantic_labels, depth_labels)
        validated_ltx_inputs = self._validate_ltx_inputs(
            ltx_inputs,
            batch_size=batch_size,
        )

        planner_result = self.planner.predict_dino_depth_plan_with_losses(
            semantic_plan_labels=semantic_labels,
            depth_plan_labels=depth_labels,
            **dict(planner_inputs),
        )
        if not isinstance(planner_result, tuple) or len(planner_result) != 3:
            raise TypeError(
                "planner must return (semantic_plan, depth_plan, planner_losses)"
            )
        semantic_plan, depth_plan, planner_losses = planner_result
        flat_tokens = self.num_keyframes * self.tokens_per_keyframe
        _require_tensor_shape(
            semantic_plan,
            (
                batch_size,
                _NUM_CAMERA_VIEWS,
                flat_tokens,
                _SEMANTIC_DIM,
            ),
            name="semantic prediction",
        )
        _require_tensor_shape(
            depth_plan,
            (
                batch_size,
                _NUM_CAMERA_VIEWS,
                flat_tokens,
                _DEPTH_LAYERS,
                _DEPTH_DIM,
            ),
            name="depth prediction",
        )
        if not isinstance(planner_losses, Mapping):
            raise TypeError("planner_losses must be a mapping")
        if "loss" not in planner_losses or not torch.is_tensor(planner_losses["loss"]):
            raise ValueError("planner_losses must contain a tensor named 'loss'")

        # Reshape without copy or detach so video loss reaches the planner/Qwen graph.
        semantic_plan = semantic_plan.reshape(
            batch_size,
            _NUM_CAMERA_VIEWS,
            self.num_keyframes,
            self.tokens_per_keyframe,
            _SEMANTIC_DIM,
        ).to(
            device=validated_ltx_inputs["noisy_latents"].device,
            dtype=validated_ltx_inputs["noisy_latents"].dtype,
        )
        forwarded = forward_pass(
            model=self.ltx,
            semantic_plan=semantic_plan,
            **validated_ltx_inputs,
        )
        if not isinstance(forwarded, Mapping) or "latents" not in forwarded:
            raise TypeError("forward_pass must return a mapping containing 'latents'")
        ltx_predictions = forwarded["latents"]
        if not isinstance(ltx_predictions, Mapping):
            raise TypeError(
                "LTX forward latents must be a prediction mapping, not a tensor"
            )
        if "video" not in ltx_predictions or not torch.is_tensor(
            ltx_predictions["video"]
        ):
            raise ValueError("LTX predictions must contain a tensor named 'video'")

        return JointVLMGEActOutput(
            ltx_predictions=dict(ltx_predictions),
            semantic_plan=semantic_plan,
            depth_plan=depth_plan,
            planner_losses=dict(planner_losses),
        )


def _named_trainable_parameters(
    module: nn.Module,
) -> list[tuple[str, nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in module.named_parameters(remove_duplicate=False)
        if parameter.requires_grad
    ]


def build_joint_optimizer_parameter_groups(
    model: JointVLMGEActModel,
    ltx_lr: float,
    semantic_lr: float,
    qwen_lr: float,
    planner_head_lr: float,
) -> list[dict[str, Any]]:
    """Classify every trainable joint parameter into one explicit LR group."""

    if not isinstance(model, JointVLMGEActModel):
        raise TypeError("model must be a JointVLMGEActModel")
    planner_model = getattr(model.planner, "model", None)
    if not isinstance(planner_model, nn.Module):
        raise ValueError("joint planner must expose its Qwen module as planner.model")

    parameters_by_group: dict[str, list[nn.Parameter]] = {
        "base_ltx": [],
        "semantic_ltx": [],
        "qwen": [],
        "planner_heads": [],
    }
    ids_by_group: dict[str, set[int]] = {name: set() for name in parameters_by_group}

    def add(group_name: str, parameter: nn.Parameter) -> None:
        parameter_id = id(parameter)
        if parameter_id not in ids_by_group[group_name]:
            ids_by_group[group_name].add(parameter_id)
            parameters_by_group[group_name].append(parameter)

    for name, parameter in _named_trainable_parameters(model.ltx):
        add("semantic_ltx" if "semantic_" in name else "base_ltx", parameter)

    for name, parameter in _named_trainable_parameters(model.planner):
        add("qwen" if name.startswith("model.") else "planner_heads", parameter)

    group_order = ("base_ltx", "semantic_ltx", "qwen", "planner_heads")
    owner_by_id: dict[int, str] = {}
    for group_name in group_order:
        for parameter in parameters_by_group[group_name]:
            parameter_id = id(parameter)
            previous_owner = owner_by_id.get(parameter_id)
            if previous_owner is not None:
                raise ValueError(
                    "duplicate trainable parameter classified into optimizer groups "
                    f"{previous_owner!r} and {group_name!r}"
                )
            owner_by_id[parameter_id] = group_name

    expected_by_id = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    missing_ids = set(expected_by_id).difference(owner_by_id)
    if missing_ids:
        missing_names = sorted(
            expected_by_id[parameter_id] for parameter_id in missing_ids
        )
        raise ValueError(f"missing trainable parameter classification: {missing_names}")
    extra_ids = set(owner_by_id).difference(expected_by_id)
    if extra_ids:
        raise ValueError("optimizer groups contain parameters outside the joint model")

    learning_rates = {
        "base_ltx": ltx_lr,
        "semantic_ltx": semantic_lr,
        "qwen": qwen_lr,
        "planner_heads": planner_head_lr,
    }
    return [
        {
            "name": group_name,
            "params": parameters_by_group[group_name],
            "lr": learning_rates[group_name],
        }
        for group_name in group_order
    ]
