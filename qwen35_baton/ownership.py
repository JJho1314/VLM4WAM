"""Explicit, exhaustive Stage-1 parameter ownership for the Baton planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn


_LANGUAGE_LAYER_PATHS = (
    "model.language_model.layers",
    "model.language_model.model.layers",
    "language_model.layers",
    "language_model.model.layers",
)
_VISION_MODULE_PATHS = (
    "model.visual",
    "model.vision_model",
    "model.vision_tower",
    "visual",
    "vision_model",
    "vision_tower",
)


@dataclass(frozen=True)
class Stage1Ownership:
    """Nonoverlapping modules that own every trainable Stage-1 parameter."""

    planner_modules: tuple[nn.Module, ...]
    qwen_top_layers: tuple[nn.Module, ...]
    qwen_vision_modules: tuple[nn.Module, ...]


def _resolve_module_path(root: nn.Module, path: str) -> Any:
    value: Any = root
    for part in path.split("."):
        if not hasattr(value, part):
            return None
        value = getattr(value, part)
    return value


def _resolve_language_layers(backbone: nn.Module) -> tuple[nn.Module, ...]:
    matches: list[tuple[str, tuple[nn.Module, ...]]] = []
    for path in _LANGUAGE_LAYER_PATHS:
        value = _resolve_module_path(backbone, path)
        if isinstance(value, nn.ModuleList) and len(value) > 0:
            layers = tuple(value)
            if all(isinstance(layer, nn.Module) for layer in layers):
                matches.append((path, layers))
    if len(matches) != 1:
        paths = ", ".join(path for path, _ in matches) or "none"
        raise ValueError(
            "Qwen language layers must resolve through exactly one supported "
            f"explicit path; resolved: {paths}"
        )
    return matches[0][1]


def _resolve_vision_module(backbone: nn.Module) -> nn.Module:
    matches: list[tuple[str, nn.Module]] = []
    for path in _VISION_MODULE_PATHS:
        value = _resolve_module_path(backbone, path)
        if isinstance(value, nn.Module):
            matches.append((path, value))
    if len(matches) != 1:
        paths = ", ".join(path for path, _ in matches) or "none"
        raise ValueError(
            "Qwen vision module must resolve through exactly one supported "
            f"explicit path; resolved: {paths}"
        )
    return matches[0][1]


def _parameter_ids(modules: tuple[nn.Module, ...]) -> set[int]:
    return {
        id(parameter)
        for module in modules
        for parameter in module.parameters()
    }


def _require_nonoverlapping(ownership: Stage1Ownership) -> None:
    owners = tuple(
        (f"{group_name}[{index}]", _parameter_ids((module,)))
        for group_name, modules in (
            ("planner_modules", ownership.planner_modules),
            ("qwen_top_layers", ownership.qwen_top_layers),
            ("qwen_vision_modules", ownership.qwen_vision_modules),
        )
        for index, module in enumerate(modules)
    )
    for index, (left_name, left_ids) in enumerate(owners):
        for right_name, right_ids in owners[index + 1 :]:
            if left_ids & right_ids:
                raise ValueError(
                    f"Stage-1 parameter ownership overlap: "
                    f"{left_name} and {right_name}"
                )


def configure_stage1_trainable_modules(planner: nn.Module) -> Stage1Ownership:
    """Freeze the planner, then enable exactly the three Stage-1 owner groups."""

    if not isinstance(planner, nn.Module):
        raise TypeError("planner must be a torch module")
    planner.requires_grad_(False)

    backbone = getattr(planner, "backbone", None)
    query_tower = getattr(planner, "query_tower", None)
    sem_mlp = getattr(planner, "sem_mlp", None)
    plan_token_adapter = getattr(planner, "plan_token_adapter", None)
    if not isinstance(backbone, nn.Module):
        raise ValueError("planner must expose its Qwen backbone")
    if not all(
        isinstance(module, nn.Module)
        for module in (query_tower, sem_mlp, plan_token_adapter)
    ):
        raise ValueError(
            "planner must expose query_tower, sem_mlp, and plan_token_adapter"
        )

    language_layers = _resolve_language_layers(backbone)
    if len(language_layers) < 8:
        raise ValueError(
            f"Qwen exposes only {len(language_layers)} language layers; "
            "Stage 1 requires the top eight"
        )
    vision_module = _resolve_vision_module(backbone)
    ownership = Stage1Ownership(
        planner_modules=(query_tower, sem_mlp, plan_token_adapter),
        qwen_top_layers=tuple(language_layers[-8:]),
        qwen_vision_modules=(vision_module,),
    )
    _require_nonoverlapping(ownership)

    for modules in (
        ownership.planner_modules,
        ownership.qwen_top_layers,
        ownership.qwen_vision_modules,
    ):
        for module in modules:
            module.requires_grad_(True)

    owned_ids = set().union(
        _parameter_ids(ownership.planner_modules),
        _parameter_ids(ownership.qwen_top_layers),
        _parameter_ids(ownership.qwen_vision_modules),
    )
    trainable_ids = {
        id(parameter)
        for parameter in planner.parameters()
        if parameter.requires_grad
    }
    if owned_ids != trainable_ids:
        missing = len(trainable_ids.difference(owned_ids))
        frozen_owned = len(owned_ids.difference(trainable_ids))
        raise RuntimeError(
            "Stage-1 ownership is not exhaustive: "
            f"{missing} unowned trainable, {frozen_owned} owned frozen"
        )
    return ownership
