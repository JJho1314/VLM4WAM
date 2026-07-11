"""Compatibility helpers for tensor-formatted Hugging Face dataset columns."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def stack_hf_column(
    column: torch.Tensor | Iterable[torch.Tensor],
) -> torch.Tensor:
    """Stack a Hugging Face column after materializing its tensor elements."""
    if isinstance(column, torch.Tensor):
        return column
    return torch.stack(tuple(column))
