"""Render local frozen-planner cross-attention for dual-camera RGB."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from qwen35_baton.provider import FrozenBatonPlanner
from qwen35_baton.visualization import render_attention_panels


def _string_list(path: Path, *, label: str) -> tuple[str, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} JSON is invalid: {path}") from error
    if (
        not isinstance(payload, list)
        or not payload
        or any(type(value) is not str or not value.strip() for value in payload)
    ):
        raise ValueError(f"{label} must be a nonempty JSON list of nonempty strings")
    return tuple(payload)


def _current_images(path: Path) -> torch.Tensor:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"current_images"}:
                raise ValueError(
                    "input NPZ must contain only a current_images array"
                )
            values = np.array(archive["current_images"], copy=True)
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and "input NPZ" in str(error):
            raise
        raise ValueError(f"input NPZ is invalid: {path}") from error
    if (
        values.dtype != np.uint8
        or values.ndim != 5
        or tuple(values.shape[1:3]) != (2, 3)
        or values.shape[0] <= 0
        or values.shape[-2] <= 0
        or values.shape[-1] <= 0
    ):
        raise ValueError("current_images must be uint8 [B,2,3,H,W]")
    return torch.from_numpy(values)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-model-path", type=Path, required=True)
    parser.add_argument("--qwen-tokenizer-path", type=Path, required=True)
    parser.add_argument("--qwen-processor-path", type=Path, required=True)
    parser.add_argument("--siglip2-model-path", type=Path, required=True)
    parser.add_argument(
        "--expected-planner-topology",
        type=Path,
        help=(
            "trusted topology override; defaults to planner_topology.json beside "
            "the checkpoint directory and is required after independent relocation"
        ),
    )
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--instructions-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # The implementation already passes local_files_only=True. These flags
    # additionally fail closed if a dependency attempts an implicit Hub call.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    images = _current_images(args.input_npz)
    instructions = _string_list(args.instructions_json, label="instructions")
    if len(instructions) != images.shape[0]:
        raise ValueError("image and instruction batch sizes must match")
    provider = FrozenBatonPlanner.from_checkpoint(
        args.checkpoint,
        qwen_model_path=args.qwen_model_path,
        qwen_tokenizer_path=args.qwen_tokenizer_path,
        qwen_processor_path=args.qwen_processor_path,
        siglip2_model_path=args.siglip2_model_path,
        expected_planner_topology=args.expected_planner_topology,
        device=args.device,
        torch_dtype=(
            torch.bfloat16 if args.dtype == "bf16" else torch.float32
        ),
    )
    plan = provider.predict(
        images,
        instructions,
        return_attention=True,
    )
    sample = {
        "current_images": images,
        "instructions": instructions,
    }
    outputs: list[str] = []
    for sample_index in range(images.shape[0]):
        outputs.extend(
            str(path)
            for path in render_attention_panels(
                sample,
                plan,
                output_dir=args.output_dir,
                sample_index=sample_index,
            )
        )
        outputs.append(
            str(args.output_dir / f"sample_{sample_index:03d}_attention.npz")
        )
    print(json.dumps({"outputs": outputs}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
