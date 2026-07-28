"""Cross-attention heatmaps for frozen strict-Baton predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from qwen35_baton.config import BatonGeometry
from qwen35_baton.provider import BatonSemanticPlan


HEATMAP_LABEL = "VLM planner cross-attention focus"
AGGREGATION_LABEL = (
    "display: per-keyframe min-max; Baton cross-attention: mean head, "
    "normalized query entropy over all plan states"
)
_HEATMAP_ALPHA = 0.45


def summarize_query_cross_attention(
    attention_maps: tuple[torch.Tensor, ...],
    *,
    num_frames: int = 4,
    tokens_per_frame: int = 256,
) -> torch.Tensor:
    """Summarize query-to-plan attention as unnormalized frame-to-frame mass.

    Each tuple entry is one Query Tower layer. Inputs may be either
    ``[B,C,Q,K]`` (the production tower has already averaged heads for tracing)
    or ``[B,C,H,Q,K]``. Layers and any explicit heads are averaged, keys are
    summed inside each Qwen plan block, then query patches are averaged inside
    each future frame. The result is ``[B,C,F_query,F_key]``.
    """

    if (
        not isinstance(attention_maps, tuple)
        or not attention_maps
        or type(num_frames) is not int
        or num_frames <= 0
        or type(tokens_per_frame) is not int
        or tokens_per_frame <= 0
    ):
        raise ValueError(
            "attention_maps and positive frame/token geometry are required"
        )
    first = attention_maps[0]
    if not isinstance(first, torch.Tensor) or first.ndim not in (4, 5):
        raise ValueError(
            "attention maps must be [B,C,Q,K] or [B,C,H,Q,K]"
        )
    expected_tokens = num_frames * tokens_per_frame
    expected_tail = (expected_tokens, expected_tokens)
    for attention in attention_maps:
        if (
            not isinstance(attention, torch.Tensor)
            or attention.ndim != first.ndim
            or attention.shape != first.shape
            or tuple(attention.shape[-2:]) != expected_tail
            or not attention.dtype.is_floating_point
            or attention.device != first.device
            or not bool(torch.isfinite(attention).all())
        ):
            raise ValueError(
                "all attention maps must share finite floating-point geometry"
            )
    stacked = torch.stack(attention_maps, dim=0).float()
    averaged = stacked.mean(dim=0)
    if averaged.ndim == 5:
        averaged = averaged.mean(dim=2)
    batch_size, cameras = averaged.shape[:2]
    blocked = averaged.reshape(
        batch_size,
        cameras,
        num_frames,
        tokens_per_frame,
        num_frames,
        tokens_per_frame,
    )
    return blocked.sum(dim=-1).mean(dim=3).detach()


def query_cross_attention_focus(
    attention_maps: tuple[torch.Tensor, ...],
    *,
    num_frames: int = 4,
    tokens_per_frame: int = 256,
) -> torch.Tensor:
    """Return per-query attention concentration as ``[B,C,F,patch]``."""

    if not attention_maps:
        raise ValueError("cross-attention maps are required")
    first = attention_maps[0]
    expected_tokens = num_frames * tokens_per_frame
    if (
        not isinstance(first, torch.Tensor)
        or first.ndim not in (4, 5)
        or tuple(first.shape[-2:]) != (expected_tokens, expected_tokens)
    ):
        raise ValueError("attention maps have incompatible Baton geometry")
    maps = []
    for attention in attention_maps:
        if (
            not isinstance(attention, torch.Tensor)
            or attention.shape != first.shape
            or not attention.dtype.is_floating_point
            or not bool(torch.isfinite(attention).all())
        ):
            raise ValueError("all attention maps must share finite geometry")
        maps.append(attention.float())
    averaged = torch.stack(maps).mean(dim=0)
    if averaged.ndim == 5:
        averaged = averaged.mean(dim=2)
    probabilities = averaged.clamp_min(0)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
        1e-12
    )
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    maximum_entropy = float(np.log(expected_tokens))
    focus = 1.0 - entropy / maximum_entropy
    return focus.clamp(0, 1).reshape(
        first.shape[0],
        first.shape[1],
        num_frames,
        tokens_per_frame,
    ).detach()


def _sample_payload(
    sample: Mapping[str, Any],
    plan: BatonSemanticPlan,
    *,
    sample_index: int,
) -> tuple[torch.Tensor, str]:
    if not isinstance(sample, Mapping):
        raise TypeError("sample must be a mapping")
    images = sample.get("current_images")
    if (
        not isinstance(images, torch.Tensor)
        or images.ndim != 5
        or tuple(images.shape[1:3]) != (2, 3)
    ):
        raise ValueError("sample current_images must be [B,2,3,H,W]")
    if images.dtype != torch.uint8:
        raise TypeError("sample current_images must contain uint8 RGB")
    if images.shape[0] != plan.tokens.shape[0]:
        raise ValueError("sample and plan batch sizes must match")
    if type(sample_index) is not int or not 0 <= sample_index < images.shape[0]:
        raise IndexError("sample_index is outside the batch")
    instructions = sample.get("instructions")
    if instructions is None:
        instruction = sample.get("instruction")
        instructions = (instruction,) if isinstance(instruction, str) else None
    if (
        not isinstance(instructions, Sequence)
        or isinstance(instructions, (str, bytes))
        or len(instructions) != images.shape[0]
        or any(type(value) is not str or not value for value in instructions)
    ):
        raise ValueError("sample instructions must contain one nonempty string per sample")
    return images[sample_index].detach().cpu(), instructions[sample_index]


def _display_heatmap(values: torch.Tensor, *, height: int, width: int) -> np.ndarray:
    if values.shape != (256,) or not bool(torch.isfinite(values).all()):
        raise ValueError("each attention frame must contain 256 finite values")
    values = values.float()
    minimum = values.min()
    maximum = values.max()
    span = maximum - minimum
    normalized = (
        torch.zeros_like(values)
        if float(span.item()) == 0.0
        else (values - minimum) / span
    )
    resized = F.interpolate(
        normalized.reshape(1, 1, 16, 16),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].clamp(0, 1).cpu().numpy()


def _fixed_colormap(values: np.ndarray) -> np.ndarray:
    """Dependency-light fixed blue-purple-yellow colormap."""

    values = np.asarray(values, dtype=np.float32)
    red = np.clip(1.8 * values - 0.25, 0.0, 1.0)
    green = np.clip(1.8 * values - 0.9, 0.0, 1.0)
    blue = np.clip(1.15 - 1.25 * values, 0.0, 1.0)
    return np.stack((red, green, blue), axis=-1)


def _overlay(rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    color = _fixed_colormap(heatmap) * 255.0
    blended = (1.0 - _HEATMAP_ALPHA) * rgb.astype(np.float32) + (
        _HEATMAP_ALPHA * color
    )
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _save_attention_archive(
    plan: BatonSemanticPlan,
    *,
    sample_index: int,
    path: Path,
) -> None:
    if plan.cross_attention_maps is None:
        raise ValueError("attention visualization requires cross-attention maps")
    focus = query_cross_attention_focus(plan.cross_attention_maps)
    query_attention = (
        summarize_query_cross_attention(plan.cross_attention_maps)[sample_index]
        .cpu()
        .numpy()
    )
    _atomic_npz(
        path,
        query_attention_focus=focus[sample_index].float().cpu().numpy(),
        query_tower_frame_attention=query_attention,
        future_indices=np.asarray(plan.future_indices, dtype=np.int64),
    )


def render_attention_panels(
    sample: Mapping[str, Any],
    plan: BatonSemanticPlan,
    *,
    output_dir: str | Path,
    sample_index: int = 0,
    filename_prefix: str = "sample",
) -> list[Path]:
    """Write one truthful five-panel PNG per camera plus one raw-data NPZ."""

    if not isinstance(plan, BatonSemanticPlan):
        raise TypeError("plan must be BatonSemanticPlan")
    if plan.cross_attention_maps is None:
        raise ValueError("attention visualization requires cross-attention maps")
    if (
        not isinstance(filename_prefix, str)
        or not filename_prefix
        or any(character in filename_prefix for character in ("/", "\\"))
    ):
        raise ValueError("filename_prefix must be a safe nonempty filename component")
    images, instruction = _sample_payload(
        sample,
        plan,
        sample_index=sample_index,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    # Pillow is optional for non-visual inference and is imported only here.
    try:
        from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required only for rendering Baton attention panels"
        ) from error

    geometry = BatonGeometry()
    focus = query_cross_attention_focus(plan.cross_attention_maps)[
        sample_index
    ].cpu()
    paths: list[Path] = []
    gap = 8
    header = 82
    for camera_index, camera_name in enumerate(geometry.camera_names):
        chw = images[camera_index]
        height, width = int(chw.shape[-2]), int(chw.shape[-1])
        rgb = chw.permute(1, 2, 0).contiguous().numpy()
        panels = [rgb]
        for frame_index in range(len(geometry.future_indices)):
            heatmap = _display_heatmap(
                focus[camera_index, frame_index],
                height=height,
                width=width,
            )
            panels.append(_overlay(rgb, heatmap))

        canvas_width = len(panels) * width + (len(panels) - 1) * gap
        canvas = Image.new("RGB", (canvas_width, height + header), "white")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        draw.text(
            (4, 4),
            f"{camera_name} | {HEATMAP_LABEL}",
            fill="black",
            font=font,
        )
        draw.text((4, 20), f"Instruction: {instruction}", fill="black", font=font)
        draw.text((4, 36), AGGREGATION_LABEL, fill="black", font=font)
        labels = ("current RGB",) + tuple(
            f"future +{offset}" for offset in geometry.future_indices
        )
        for panel_index, (panel, label) in enumerate(
            zip(panels, labels, strict=True)
        ):
            x_offset = panel_index * (width + gap)
            canvas.paste(Image.fromarray(panel), (x_offset, header))
            draw.text((x_offset + 2, header - 14), label, fill="black", font=font)

        info = PngImagePlugin.PngInfo()
        info.add_text("camera", camera_name)
        info.add_text("instruction", instruction)
        info.add_text("heatmap_label", HEATMAP_LABEL)
        info.add_text("aggregation", AGGREGATION_LABEL)
        output_path = (
            destination / f"{filename_prefix}_{sample_index:03d}_{camera_name}.png"
        )
        canvas.save(output_path, format="PNG", pnginfo=info, optimize=False)
        paths.append(output_path)

    _save_attention_archive(
        plan,
        sample_index=sample_index,
        path=destination / f"{filename_prefix}_{sample_index:03d}_attention.npz",
    )
    return paths
