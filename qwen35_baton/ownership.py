"""Explicit full-model ownership for Baton Stage-1 training."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass(frozen=True)
class Stage1Ownership:
    """The complete VA-Planner is the single Stage-1 owner."""

    trainable_modules: tuple[nn.Module, ...]


def configure_stage1_trainable_modules(planner: nn.Module) -> Stage1Ownership:
    """Enable gradients for the full visual-language alignment planner."""

    if not isinstance(planner, nn.Module):
        raise TypeError("planner must be a torch module")
    planner.requires_grad_(True)
    parameters = tuple(planner.parameters())
    if not parameters:
        raise ValueError("planner must contain trainable parameters")
    if not all(parameter.requires_grad for parameter in parameters):
        raise RuntimeError("full Baton VA-Planner ownership is not exhaustive")
    return Stage1Ownership(trainable_modules=(planner,))
