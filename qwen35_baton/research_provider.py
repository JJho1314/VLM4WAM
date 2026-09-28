"""Build and serve research planner variants (E4) outside the strict v3 contract."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from qwen35_baton.provider import FrozenBatonPlanner


def build_research_planner(
    *,
    qwen_path: str | Path,
    added_token_ids: tuple[int, ...],
    current_mode: str,
    spatial: bool,
    torch_dtype: torch.dtype = torch.float32,
) -> tuple[Any, nn.Module]:
    """Construct the processor and an unweighted planner for one variant."""

    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    from qwen35_baton.model import BatonQwen35Planner
    from qwen35_baton.sequence import ADDED_TOKENS

    if current_mode not in ("none", "context", "residual"):
        raise ValueError(f"unknown current_mode {current_mode!r}")
    tokenizer = AutoTokenizer.from_pretrained(str(qwen_path), local_files_only=True)
    processor = AutoProcessor.from_pretrained(str(qwen_path), local_files_only=True)
    processor.tokenizer = tokenizer
    actual = tuple(int(tokenizer.convert_tokens_to_ids(token)) for token in ADDED_TOKENS)
    if actual != added_token_ids:
        raise ValueError("tokenizer added-token IDs differ from the planner checkpoint")
    qwen = AutoModelForImageTextToText.from_pretrained(
        str(qwen_path),
        local_files_only=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    planner = BatonQwen35Planner(
        qwen,
        added_token_ids=added_token_ids,
        current_context=current_mode != "none",
        residual=current_mode == "residual",
        spatial=spatial,
    )
    return processor, planner


def init_checkpoint_token_ids(init_checkpoint: str | Path) -> tuple[int, ...]:
    metadata = json.loads(
        (Path(init_checkpoint) / "metadata.json").read_text(encoding="utf-8")
    )
    return tuple(int(value) for value in metadata["added_token_ids"])


class ResearchBatonPlanner(FrozenBatonPlanner):
    """Frozen research variant; computes nothing itself, callers pass current grids."""

    @classmethod
    def from_research_checkpoint(
        cls,
        step_dir: str | Path,
        *,
        qwen_path: str | Path,
        device: torch.device | str,
    ) -> "ResearchBatonPlanner":
        from safetensors.torch import load_model

        step_dir = Path(step_dir)
        variant = json.loads((step_dir.parent / "variant.json").read_text(encoding="utf-8"))
        added_token_ids = init_checkpoint_token_ids(variant["init_checkpoint"])
        processor, planner = build_research_planner(
            qwen_path=qwen_path,
            added_token_ids=added_token_ids,
            current_mode=variant.get("current_mode", "none"),
            spatial=float(variant.get("spatial_weight", 0.0)) > 0,
            torch_dtype=torch.bfloat16,
        )
        missing, unexpected = load_model(
            planner, str(step_dir / "planner.safetensors"), strict=False
        )
        if unexpected or [name for name in missing if name != "frozen_base_embedding.weight"]:
            raise ValueError(
                f"research checkpoint topology mismatch: missing={missing[:5]} "
                f"unexpected={unexpected[:5]}"
            )
        planner.to(device=torch.device(device), dtype=torch.bfloat16)
        provider = cls(planner=planner, processor=processor, added_token_ids=added_token_ids)
        provider.variant = variant
        return provider

    @property
    def uses_current_context(self) -> bool:
        return bool(getattr(self.planner, "current_context", False))

    @torch.no_grad()
    def predict_tokens(
        self,
        current_images: torch.Tensor,
        instructions: Sequence[str],
        *,
        current_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``[B,2,4,256,1024]``; ``current_features`` is ``[B,2,256,1024]``."""

        batch_size, positive = self._validate_inputs(current_images, instructions)
        if self.uses_current_context != (current_features is not None):
            raise ValueError("current_features must match the variant's current mode")
        qwen_inputs, plan_positions = self._build_rows(current_images, positive)
        kwargs = {}
        if current_features is not None:
            kwargs["current_features"] = current_features.reshape(
                batch_size * 2, 256, 1024
            ).to(self._device())
        autocast = self._autocast_context() if self._device().type == "cuda" else nullcontext()
        with autocast:
            output = self.planner.forward_rows(qwen_inputs, plan_positions, **kwargs)
        return output.flat.reshape(self.geometry.output_shape(batch_size)).float()
