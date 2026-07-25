#!/usr/bin/env python3
"""Visualize dual-camera K4 VLM planner query-to-image attention.

The semantic planner's Perceiver implementation does not return attention
weights.  This module reconstructs the exact trained attention operation from
the forward-hook inputs without changing model code or checkpoint contents.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


AttentionReduction = Literal["mean", "max"]

HERE = Path(__file__).resolve().parent
PLANNER_DIR = HERE.parent
REPO_ROOT = PLANNER_DIR.parent
for _import_path in (
    REPO_ROOT,
    PLANNER_DIR,
    HERE,
    PLANNER_DIR / "lingbot_dino_4b",
):
    _import_string = str(_import_path)
    if _import_string not in sys.path:
        sys.path.insert(0, _import_string)

EXPECTED_CHECKPOINT_CONTRACT: dict[str, Any] = {
    "plan_head_type": "lingbot_dino",
    "num_camera_views": 2,
    "camera_names": ["main", "wrist"],
    "num_keyframes": 4,
    "future_keyframe_offsets": [2, 4, 6, 8],
    "target_tokens_per_keyframe": 256,
    "branch_latent_per_keyframe": 64,
}


def _reshape_attention_heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
    """Convert ``[B, N, heads * dim_head]`` to ``[B, heads, N, dim_head]``."""

    batch, length, width = tensor.shape
    if width % heads:
        raise ValueError(f"attention width {width} is not divisible by {heads} heads")
    return tensor.view(batch, length, heads, -1).transpose(1, 2).contiguous()


def reconstruct_perceiver_attention(
    module: nn.Module,
    x: torch.Tensor,
    latents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct the exact softmax weights and values used by the resampler."""

    x_norm = module.norm1(x)
    latent_norm = module.norm2(latents)
    query = _reshape_attention_heads(module.to_q(latent_norm), module.heads)
    key, value = module.to_kv(torch.cat((x_norm, latent_norm), dim=-2)).chunk(
        2,
        dim=-1,
    )
    key = _reshape_attention_heads(key, module.heads)
    value = _reshape_attention_heads(value, module.heads)
    scale = 1.0 / math.sqrt(math.sqrt(module.dim_head))
    weights = torch.softmax(
        ((query * scale) @ (key * scale).transpose(-2, -1)).float(),
        dim=-1,
    ).to(query.dtype)
    return weights, value


def reduce_image_attention(
    weights: torch.Tensor,
    *,
    image_token_count: int,
    reduction: AttentionReduction = "mean",
) -> torch.Tensor:
    """Reduce heads and output queries while retaining image keys only.

    The key sequence also includes the semantic latent tokens fed to the head
    and the output query tokens appended inside ``PerceiverAttention``.  Qwen
    image tokens are the leading ``image_token_count`` columns.
    """

    if weights.ndim != 4:
        raise ValueError(
            "attention weights must have shape [batch, heads, queries, keys], "
            f"got {tuple(weights.shape)}"
        )
    if image_token_count <= 0 or image_token_count > weights.shape[-1]:
        raise ValueError(
            f"invalid image_token_count={image_token_count} for "
            f"{weights.shape[-1]} attention keys"
        )
    image_weights = weights[..., :image_token_count]
    if reduction == "mean":
        reduced = image_weights.mean(dim=(1, 2))
    elif reduction == "max":
        reduced = image_weights.amax(dim=(1, 2))
    else:
        raise ValueError(f"unsupported query reduction: {reduction!r}")
    if not torch.isfinite(reduced).all():
        raise ValueError("planner attention contains non-finite values")
    return reduced


class PlannerAttentionCapture:
    """Temporarily capture reduced attention from one Perceiver layer."""

    def __init__(
        self,
        module: nn.Module,
        *,
        image_token_count: int,
        reduction: AttentionReduction = "mean",
    ) -> None:
        self.module = module
        self.image_token_count = int(image_token_count)
        self.reduction = reduction
        self.maps: list[torch.Tensor] = []
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def _forward_hook(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ) -> None:
        if len(inputs) != 2:
            raise ValueError(
                "expected PerceiverAttention inputs (x, latents), "
                f"received {len(inputs)} tensors"
            )
        with torch.no_grad():
            weights, _ = reconstruct_perceiver_attention(module, inputs[0], inputs[1])
            reduced = reduce_image_attention(
                weights,
                image_token_count=self.image_token_count,
                reduction=self.reduction,
            )
        self.maps.append(reduced.detach().float().cpu())

    def __enter__(self) -> "PlannerAttentionCapture":
        if self._handle is not None:
            raise RuntimeError("attention capture context is already active")
        self.maps.clear()
        self._handle = self.module.register_forward_hook(self._forward_hook)
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def merged_image_grid(
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
    expected_tokens: int,
) -> tuple[int, int]:
    """Restore the post-merge 2D Qwen image-token grid."""

    if image_grid_thw.numel() != 3:
        raise ValueError(
            "image_grid_thw must contain temporal, height, width; "
            f"got shape {tuple(image_grid_thw.shape)}"
        )
    temporal, height, width = (int(value) for value in image_grid_thw.tolist())
    merge = int(spatial_merge_size)
    if temporal != 1 or merge <= 0 or height % merge or width % merge:
        raise ValueError(
            f"unsupported Qwen image grid {(temporal, height, width)} "
            f"with spatial_merge_size={merge}"
        )
    merged = (height // merge, width // merge)
    token_count = merged[0] * merged[1]
    if token_count != int(expected_tokens):
        raise ValueError(
            f"Qwen merged grid has {token_count} tokens, expected "
            f"{int(expected_tokens)}"
        )
    return merged


def normalize_attention_stack(
    maps: torch.Tensor,
    *,
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.98,
) -> torch.Tensor:
    """Jointly normalize a camera's K attention maps to a shared [0, 1] scale."""

    maps = torch.as_tensor(maps).detach().float()
    if maps.ndim != 3:
        raise ValueError(f"attention stack must have shape [K, H, W], got {maps.shape}")
    if not torch.isfinite(maps).all():
        raise ValueError("attention maps contain non-finite values")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError(
            "normalization quantiles must satisfy "
            f"0 <= lower < upper <= 1, got {lower_quantile}, {upper_quantile}"
        )
    lower = torch.quantile(maps, lower_quantile)
    upper = torch.quantile(maps, upper_quantile)
    scale = upper - lower
    if float(scale) <= torch.finfo(maps.dtype).eps:
        return torch.zeros_like(maps)
    return ((maps - lower) / scale).clamp_(0.0, 1.0)


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"RGB image must have shape [H, W, 3], got {array.shape}")
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating):
        upper = float(np.nanmax(array))
        if upper <= 1.0:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def attention_products(
    rgb: np.ndarray,
    normalized_map: torch.Tensor,
    *,
    alpha: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an unblended Turbo heatmap and an RGB overlay."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"overlay alpha must be within [0, 1], got {alpha}")
    rgb_uint8 = _validate_rgb(rgb)
    attention = torch.as_tensor(normalized_map).detach().float()
    if attention.ndim != 2:
        raise ValueError(f"normalized attention must be 2D, got {attention.shape}")
    if not torch.isfinite(attention).all():
        raise ValueError("normalized attention contains non-finite values")
    resized = F.interpolate(
        attention[None, None],
        size=rgb_uint8.shape[:2],
        mode="bilinear",
        align_corners=False,
    )[0, 0].clamp(0.0, 1.0)

    import matplotlib

    matplotlib.use("Agg", force=True)
    colorized = matplotlib.colormaps["turbo"](resized.cpu().numpy())[..., :3]
    heatmap = np.round(colorized * 255.0).astype(np.uint8)
    overlay = np.round(
        alpha * heatmap.astype(np.float32)
        + (1.0 - alpha) * rgb_uint8.astype(np.float32)
    )
    return heatmap, np.clip(overlay, 0, 255).astype(np.uint8)


def render_composite(
    output_path: str | Path,
    *,
    instruction: str,
    observations: Mapping[str, np.ndarray],
    overlays: Mapping[str, Sequence[np.ndarray]],
    offsets: Sequence[int],
) -> None:
    """Render a paper-ready 2x(1+K) dual-camera attention comparison."""

    if tuple(observations) != ("main", "wrist"):
        raise ValueError("observations must contain ordered cameras: main, wrist")
    if not offsets:
        raise ValueError("at least one future offset is required")
    for camera in observations:
        if camera not in overlays or len(overlays[camera]) != len(offsets):
            raise ValueError(
                f"{camera} overlays must contain exactly {len(offsets)} maps"
            )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    columns = 1 + len(offsets)
    figure, axes = plt.subplots(
        2,
        columns,
        figsize=(2.6 * columns, 5.5),
        squeeze=False,
    )
    headers = ["Observation", *[f"Planner t+{offset}" for offset in offsets]]
    row_names = {"main": "Main Camera", "wrist": "Wrist Camera"}

    for row, camera in enumerate(("main", "wrist")):
        panels = [_validate_rgb(observations[camera]), *overlays[camera]]
        for column, (axis, panel) in enumerate(zip(axes[row], panels)):
            axis.imshow(_validate_rgb(panel))
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(headers[column], fontsize=10, weight="semibold", pad=7)
            for spine in axis.spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(
            row_names[camera],
            fontsize=11,
            weight="bold",
            rotation=90,
            labelpad=10,
        )

    compact_instruction = " ".join(str(instruction).split())
    figure.suptitle(
        f'VLM Planner Query-to-Image Attention\nInstruction: "{compact_instruction}"',
        fontsize=13,
        weight="bold",
        y=0.985,
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.055,
        top=0.87,
        hspace=0.32,
        wspace=0.08,
    )
    figure.canvas.draw()
    for row, facecolor in enumerate(("#eef3f8", "#fffbe6")):
        boxes = [axis.get_position() for axis in axes[row]]
        left = min(box.x0 for box in boxes) - 0.018
        bottom = min(box.y0 for box in boxes) - 0.035
        right = max(box.x1 for box in boxes) + 0.012
        top = max(box.y1 for box in boxes) + 0.04
        container = FancyBboxPatch(
            (left, bottom),
            right - left,
            top - bottom,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            transform=figure.transFigure,
            linewidth=1.0,
            edgecolor="#333333",
            facecolor=facecolor,
            zorder=-1,
            clip_on=False,
        )
        figure.add_artist(container)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    with Image.open(destination) as rendered:
        rendered.convert("RGB").save(destination)


def validate_checkpoint_contract(metadata: Mapping[str, Any]) -> None:
    """Require the exact dual-camera K4 planner geometry used by this figure."""

    for field, expected in EXPECTED_CHECKPOINT_CONTRACT.items():
        actual = metadata.get(field)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"incompatible planner metadata field {field}: "
                f"expected {expected!r}, got {actual!r}"
            )


def group_attention_captures(
    captures: Sequence[torch.Tensor],
    *,
    num_views: int,
    num_keyframes: int,
) -> torch.Tensor:
    """Group view-major, keyframe-major hook calls as ``[V, K, N_image]``."""

    expected_calls = int(num_views) * int(num_keyframes)
    if len(captures) != expected_calls:
        raise ValueError(
            f"expected {expected_calls} planner attention calls "
            f"({num_views} views x {num_keyframes} keyframes), got {len(captures)}"
        )
    rows = []
    for index, capture in enumerate(captures):
        capture = torch.as_tensor(capture).detach().float().cpu()
        if capture.ndim == 2 and capture.shape[0] == 1:
            capture = capture[0]
        if capture.ndim != 1:
            raise ValueError(
                f"capture {index} must contain one sample [N_image], "
                f"got {tuple(capture.shape)}"
            )
        rows.append(capture)
    if len({tuple(row.shape) for row in rows}) != 1:
        raise ValueError("planner attention captures have inconsistent token counts")
    return torch.stack(rows).reshape(int(num_views), int(num_keyframes), -1)


def _required_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing {description}: {path}")
    return path


def _camera_rgb(frame: torch.Tensor) -> np.ndarray:
    frame = torch.as_tensor(frame).detach().float().cpu()
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"camera frame must be [H,W,3], got {tuple(frame.shape)}")
    return (
        ((frame + 1.0) * 127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .numpy()
    )


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_validate_rgb(rgb), mode="RGB").save(path)


def _write_manifest_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sample_attention_products(
    *,
    output_dir: Path,
    sample_number: int,
    sample_index: int,
    stem: str,
    instruction: str,
    current: torch.Tensor,
    grouped_attention: torch.Tensor,
    grid_shapes: Sequence[tuple[int, int]],
    image_grid_thw: torch.Tensor,
    camera_names: Sequence[str],
    offsets: Sequence[int],
    overlay_alpha: float,
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, Any]:
    sample_dir = output_dir / f"sample_{sample_number:02d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    observations = {
        camera: _camera_rgb(current[0, camera_index])
        for camera_index, camera in enumerate(camera_names)
    }
    overlays: dict[str, list[np.ndarray]] = {}
    normalized_cameras = []
    product_records = []

    for camera_index, camera in enumerate(camera_names):
        height, width = grid_shapes[camera_index]
        raw_maps = grouped_attention[camera_index].reshape(
            len(offsets),
            height,
            width,
        )
        normalized = normalize_attention_stack(
            raw_maps,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
        normalized_cameras.append(normalized)
        overlays[camera] = []
        for keyframe_index, offset in enumerate(offsets):
            heatmap, overlay = attention_products(
                observations[camera],
                normalized[keyframe_index],
                alpha=overlay_alpha,
            )
            prefix = f"{camera}_k{keyframe_index}_off{offset}"
            heatmap_name = f"{prefix}_heatmap.png"
            overlay_name = f"{prefix}_overlay.png"
            _save_rgb(sample_dir / heatmap_name, heatmap)
            _save_rgb(sample_dir / overlay_name, overlay)
            overlays[camera].append(overlay)
            product_records.append(
                {
                    "camera": camera,
                    "keyframe_index": keyframe_index,
                    "future_offset": int(offset),
                    "raw_attention_min": float(raw_maps[keyframe_index].min()),
                    "raw_attention_max": float(raw_maps[keyframe_index].max()),
                    "raw_attention_mean": float(raw_maps[keyframe_index].mean()),
                    "heatmap": heatmap_name,
                    "overlay": overlay_name,
                }
            )

    composite_name = f"sample_{sample_number:02d}_planner_attention.png"
    render_composite(
        sample_dir / composite_name,
        instruction=instruction,
        observations=observations,
        overlays=overlays,
        offsets=offsets,
    )
    arrays_name = f"sample_{sample_number:02d}_planner_attention.npz"
    np.savez_compressed(
        sample_dir / arrays_name,
        attention_raw=grouped_attention.numpy(),
        attention_normalized=torch.stack(normalized_cameras).numpy(),
        image_grid_thw=image_grid_thw.cpu().numpy(),
        camera_names=np.asarray(camera_names),
        future_offsets=np.asarray(offsets, dtype=np.int64),
    )
    return {
        "sample_number": int(sample_number),
        "sample_index": int(sample_index),
        "stem": str(stem),
        "instruction": str(instruction),
        "sample_dir": sample_dir.name,
        "composite": composite_name,
        "arrays": arrays_name,
        "products": product_records,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render real query-to-image attention from a dual-camera K4 "
            "VLM semantic planner."
        )
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--ge-act-data-config",
        type=Path,
        default=REPO_ROOT
        / "ge_act/configs/ltx_model/libero/planner_data_libero_fastwam_ola.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--query-reduction",
        choices=("mean", "max"),
        default="mean",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    parser.add_argument("--lower-quantile", type=float, default=0.02)
    parser.add_argument("--upper-quantile", type=float, default=0.98)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be within [0,1]")
    checkpoint_dir = _required_path(args.checkpoint_dir, "planner checkpoint")
    _required_path(args.ge_act_data_config, "GE-Act data config")
    metadata_path = _required_path(
        checkpoint_dir / "planner_meta.json",
        "planner metadata",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_checkpoint_contract(metadata)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import train_semantic_planner as trainer
    from ge_act_dual_camera import DualCameraPlannerCollator
    from qwen3vl_wrapper import (
        configure_qwen3vl_processor,
        move_qwen_inputs_to_device,
    )
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    processor = configure_qwen3vl_processor(
        AutoProcessor.from_pretrained(
            str(checkpoint_dir / "processor"),
            local_files_only=True,
        )
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(checkpoint_dir / "qwen3vl_lora_or_model"),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(device).eval()
    if hasattr(model.config, "text_config"):
        model.config.hidden_size = model.config.text_config.hidden_size
    model.config.use_cache = False
    wrapper = trainer.PlannerWrapper.from_exported_checkpoint(
        model=model,
        checkpoint_dir=checkpoint_dir,
        metadata=metadata,
    ).to(device).eval()

    camera_names = [str(name) for name in metadata["camera_names"]]
    offsets = [int(offset) for offset in metadata["future_keyframe_offsets"]]
    dataset = trainer.load_ge_act_dual_camera_planner_dataset(
        args.ge_act_data_config,
        future_offsets=offsets,
    )
    collator = DualCameraPlannerCollator(
        processor=processor,
        plan_sequence=list(metadata["plan_token_strings"]),
    )
    indices = random.Random(args.seed).sample(
        range(len(dataset)),
        min(args.num_samples, len(dataset)),
    )
    merge_size = int(model.config.vision_config.spatial_merge_size)
    attention_module = wrapper.plan_head.resampler.layers[0][0]
    records = []

    with torch.inference_mode():
        for sample_number, sample_index in enumerate(indices):
            item = dataset[sample_index]
            batch = collator([item])
            stems = batch.pop("stems")
            current = batch.pop("current_camera_images")
            batch.pop("future_camera_images")
            image_grid_thw = torch.as_tensor(batch["image_grid_thw"]).detach().cpu()
            if tuple(image_grid_thw.shape) != (len(camera_names), 3):
                raise ValueError(
                    "one-sample dual-camera Qwen input must have image_grid_thw "
                    f"[{len(camera_names)},3], got {tuple(image_grid_thw.shape)}"
                )
            expected_image_tokens = [
                (int(row[1]) // merge_size) * (int(row[2]) // merge_size)
                for row in image_grid_thw
            ]
            if len(set(expected_image_tokens)) != 1:
                raise ValueError(
                    "main and wrist cameras must produce equal Qwen image-token "
                    f"counts, got {expected_image_tokens}"
                )
            grid_shapes = [
                merged_image_grid(
                    row,
                    spatial_merge_size=merge_size,
                    expected_tokens=expected_image_tokens[index],
                )
                for index, row in enumerate(image_grid_thw)
            ]
            model_inputs = move_qwen_inputs_to_device(
                batch,
                device,
                model_dtype=next(model.parameters()).dtype,
            )
            with PlannerAttentionCapture(
                attention_module,
                image_token_count=expected_image_tokens[0],
                reduction=args.query_reduction,
            ) as capture:
                prediction = wrapper.predict_semantic_plan(**model_inputs)
            expected_prediction_shape = (
                1,
                len(camera_names),
                len(offsets) * int(metadata["target_tokens_per_keyframe"]),
                int(metadata["semantic_dim"]),
            )
            if tuple(prediction.shape) != expected_prediction_shape:
                raise ValueError(
                    "planner prediction geometry differs from metadata: "
                    f"expected {expected_prediction_shape}, got "
                    f"{tuple(prediction.shape)}"
                )
            grouped = group_attention_captures(
                capture.maps,
                num_views=len(camera_names),
                num_keyframes=len(offsets),
            )
            if grouped.shape[-1] != expected_image_tokens[0]:
                raise ValueError(
                    f"captured {grouped.shape[-1]} image keys, expected "
                    f"{expected_image_tokens[0]}"
                )
            record = _sample_attention_products(
                output_dir=args.output_dir,
                sample_number=sample_number,
                sample_index=sample_index,
                stem=stems[0],
                instruction=str(item["prompt"]),
                current=current,
                grouped_attention=grouped,
                grid_shapes=grid_shapes,
                image_grid_thw=image_grid_thw,
                camera_names=camera_names,
                offsets=offsets,
                overlay_alpha=args.overlay_alpha,
                lower_quantile=args.lower_quantile,
                upper_quantile=args.upper_quantile,
            )
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    manifest = {
        "checkpoint": str(checkpoint_dir),
        "ge_act_data_config": str(args.ge_act_data_config),
        "seed": args.seed,
        "query_reduction": args.query_reduction,
        "normalization_quantiles": [
            args.lower_quantile,
            args.upper_quantile,
        ],
        "overlay_alpha": args.overlay_alpha,
        "attention_source": "plan_head.resampler.layers.0.0",
        "camera_names": camera_names,
        "future_keyframe_offsets": offsets,
        "samples": records,
    }
    manifest_path = args.output_dir / "manifest.json"
    _write_manifest_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "done",
                "sample_count": len(records),
                "output_dir": str(args.output_dir),
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
