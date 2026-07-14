#!/usr/bin/env python3
"""Small Qwen3-VL helper layer for semantic planner experiments.

This module intentionally depends only on Transformers Qwen3-VL classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def torch_dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def configure_qwen3vl_processor(
    processor: Any,
    target_short_side: int | None = None,
    video_nframes: int | None = None,
) -> Any:
    processor.tokenizer.padding_side = "left"

    if target_short_side is not None:
        target_long = target_short_side * 4
        min_pixels = target_short_side * target_short_side
        max_pixels = target_short_side * target_long
        size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        if hasattr(processor, "image_processor"):
            processor.image_processor.size = size
            processor.image_processor.min_pixels = min_pixels
            processor.image_processor.max_pixels = max_pixels
        if hasattr(processor, "video_processor"):
            processor.video_processor.size = size
            processor.video_processor.min_pixels = min_pixels
            processor.video_processor.max_pixels = max_pixels

    if video_nframes is not None and hasattr(processor, "video_processor"):
        processor.video_processor.num_frames = video_nframes
        processor.video_processor.fps = None

    return processor


def load_qwen3vl_model_and_processor(
    model_path: str | Path,
    *,
    device: torch.device | str | None = None,
    dtype: str = "bf16",
    attn_implementation: str = "sdpa",
    local_files_only: bool = True,
    target_short_side: int | None = None,
    video_nframes: int | None = None,
    device_map: str | None = None,
    eval_mode: bool = True,
) -> tuple[Any, Any]:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_path = str(model_path)
    torch_dtype = torch_dtype_from_name(dtype)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_files_only)
    processor = configure_qwen3vl_processor(
        processor,
        target_short_side=target_short_side,
        video_nframes=video_nframes,
    )

    model_kwargs: dict[str, Any] = {
        "dtype": torch_dtype,
        "local_files_only": local_files_only,
        "attn_implementation": attn_implementation,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    if device_map is None and device is not None:
        model = model.to(device)

    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
        model.config.hidden_size = model.config.text_config.hidden_size

    if eval_mode:
        model.eval()
    else:
        model.train()
    return model, processor


def move_qwen_inputs_to_device(inputs: dict[str, Any], device: torch.device | str, model_dtype: torch.dtype | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if model_dtype is not None and "pixel_values" in key and value.dtype != model_dtype:
                value = value.to(model_dtype)
        out[key] = value
    return out
