"""Utilities for visualizing target-conditioned attention in Cosmos DiT.

The helpers in this file intentionally operate on plain tensors instead of a
training or inference object.  This keeps the visualization reusable from
validation, offline evaluation, or single-sample debugging scripts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


@dataclass
class TargetAttentionMetric:
    block: int
    mask_area: float
    attn_mass_inside_mask: float
    attn_inside_mean: float
    attn_outside_mean: float
    inside_outside_ratio: float
    entropy: float
    peak_frame: int
    peak_y: int
    peak_x: int
    peak_inside_mask: float


def _as_t_h_w(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected [T,H,W] or [1,T,H,W] tensor, got {tuple(tensor.shape)}")
    return tensor


def _to_uint8_frame(frame_C_H_W: torch.Tensor) -> np.ndarray:
    frame = frame_C_H_W.detach().float().cpu()
    if frame.max() <= 2.0:
        if frame.min() < 0:
            frame = (frame.clamp(-1, 1) + 1.0) * 127.5
        else:
            frame = frame.clamp(0, 1) * 255.0
    return frame.clamp(0, 255).byte().permute(1, 2, 0).numpy()


def normalize_heatmap(tensor: torch.Tensor, q_low: float = 0.01, q_high: float = 0.99) -> torch.Tensor:
    x = tensor.detach().float().cpu()
    flat = x.flatten()
    if flat.numel() == 0:
        return x
    if flat.numel() > 1_000_000:
        step = int(np.ceil(flat.numel() / 1_000_000))
        flat = flat[::step]
    lo = torch.quantile(flat, q_low)
    hi = torch.quantile(flat, q_high)
    return ((x - lo) / (hi - lo + 1e-6)).clamp(0, 1)


def resize_volume(volume_T_H_W: torch.Tensor, size: tuple[int, int, int], mode: str = "trilinear") -> torch.Tensor:
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(volume_T_H_W[None, None].float(), size=size, mode=mode, **kwargs)[0, 0]


def _frame_indices(num_frames: int, max_frames: int = 7) -> list[int]:
    if num_frames <= max_frames:
        return list(range(num_frames))
    return sorted({round(i * (num_frames - 1) / (max_frames - 1)) for i in range(max_frames)})


def _heat_color(heat_H_W: torch.Tensor) -> np.ndarray:
    heat = heat_H_W.detach().float().cpu().clamp(0, 1).numpy()
    rgb = np.zeros((*heat.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(1.8 * heat, 0, 1)
    rgb[..., 1] = np.clip(1.8 * (1.0 - np.abs(heat - 0.55) / 0.55), 0, 1)
    rgb[..., 2] = np.clip(1.4 * (1.0 - heat), 0, 1) * (heat > 0.05)
    return (rgb * 255).astype(np.uint8)


def _overlay_heat(frame_rgb: np.ndarray, heat_H_W: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    heat = heat_H_W.detach().float().cpu().clamp(0, 1)
    heat_rgb = _heat_color(heat)
    mask = heat.numpy()[..., None]
    out = frame_rgb.astype(np.float32) * (1.0 - alpha * mask) + heat_rgb.astype(np.float32) * (alpha * mask)
    return out.clip(0, 255).astype(np.uint8)


def _overlay_mask(frame_rgb: np.ndarray, mask_H_W: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    mask = mask_H_W.detach().float().cpu().clamp(0, 1).numpy()[..., None]
    red = np.zeros_like(frame_rgb, dtype=np.float32)
    red[..., 0] = 255
    out = frame_rgb.astype(np.float32) * (1.0 - alpha * mask) + red * (alpha * mask)
    return out.clip(0, 255).astype(np.uint8)


def _load_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return None


def _save_grid(path: Path, rows: list[tuple[str, list[np.ndarray]]], col_labels: list[str]) -> None:
    if not rows or not rows[0][1]:
        return
    tile_h, tile_w = rows[0][1][0].shape[:2]
    label_w = 330
    header_h = 42
    font = _load_font(24)
    small_font = _load_font(20)
    canvas = Image.new("RGB", (label_w + tile_w * len(col_labels), header_h + tile_h * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, header_h), fill=(250, 250, 250))
    for col, label in enumerate(col_labels):
        draw.text((label_w + col * tile_w + 8, 10), label, fill=(0, 0, 0), font=small_font)
    for row_idx, (row_label, images) in enumerate(rows):
        y = header_h + row_idx * tile_h
        draw.rectangle((0, y, label_w, y + tile_h), fill=(246, 246, 246))
        draw.text((12, y + 12), row_label[:34], fill=(0, 0, 0), font=font)
        for col_idx, image in enumerate(images):
            canvas.paste(Image.fromarray(image), (label_w + col_idx * tile_w, y))
    canvas.save(path, quality=95)


def compute_attention_metric(attn_T_H_W: torch.Tensor, mask_T_H_W: torch.Tensor, block: int) -> TargetAttentionMetric:
    attn = _as_t_h_w(attn_T_H_W).clamp(min=0)
    mask = _as_t_h_w(mask_T_H_W).clamp(0, 1)
    if mask.shape != attn.shape:
        mask = resize_volume(mask, tuple(attn.shape), mode="nearest")

    attn_sum = attn.sum() + 1e-6
    mask_sum = mask.sum() + 1e-6
    inv = 1.0 - mask
    inv_sum = inv.sum() + 1e-6
    inside_mean = (attn * mask).sum() / mask_sum
    outside_mean = (attn * inv).sum() / inv_sum

    prob = (attn.flatten() / attn_sum).clamp(min=1e-12)
    entropy = -(prob * prob.log()).sum() / torch.log(torch.tensor(float(prob.numel())))
    peak_index = int(torch.argmax(attn).item())
    peak_t = peak_index // (attn.shape[1] * attn.shape[2])
    peak_rem = peak_index % (attn.shape[1] * attn.shape[2])
    peak_y = peak_rem // attn.shape[2]
    peak_x = peak_rem % attn.shape[2]

    return TargetAttentionMetric(
        block=int(block),
        mask_area=float(mask.mean().item()),
        attn_mass_inside_mask=float(((attn * mask).sum() / attn_sum).item()),
        attn_inside_mean=float(inside_mean.item()),
        attn_outside_mean=float(outside_mean.item()),
        inside_outside_ratio=float((inside_mean / (outside_mean + 1e-6)).item()),
        entropy=float(entropy.item()),
        peak_frame=int(peak_t),
        peak_y=int(peak_y),
        peak_x=int(peak_x),
        peak_inside_mask=float(mask[peak_t, peak_y, peak_x].item()),
    )


def _mean_selected_attention(
    attn_maps: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    selected_blocks: set[int] | None,
) -> torch.Tensor:
    selected = [
        _as_t_h_w(attn)
        for block, attn in zip(block_ids, attn_maps)
        if not selected_blocks or int(block) in selected_blocks
    ]
    if not selected:
        selected = [_as_t_h_w(attn) for attn in attn_maps]
    return torch.stack(selected).mean(dim=0)


def render_attention_overlay_grid(
    path: Path,
    raw_video_C_T_H_W: torch.Tensor,
    mask_T_H_W: torch.Tensor,
    attn_maps: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    selected_blocks: Iterable[int] | None = None,
    max_frames: int = 7,
) -> None:
    raw = raw_video_C_T_H_W.detach().cpu()
    if raw.ndim != 4:
        raise ValueError(f"Expected raw video [C,T,H,W], got {tuple(raw.shape)}")
    mask = _as_t_h_w(mask_T_H_W)
    selected = set(int(x) for x in selected_blocks) if selected_blocks is not None else set()
    T_raw, H_raw, W_raw = raw.shape[1], raw.shape[2], raw.shape[3]
    frames = _frame_indices(T_raw, max_frames=max_frames)
    frame_labels = [f"f{idx}" for idx in frames]
    raw_frames = [_to_uint8_frame(raw[:, idx]) for idx in frames]
    mask_up = resize_volume(mask, (T_raw, H_raw, W_raw), mode="nearest")

    rows: list[tuple[str, list[np.ndarray]]] = [
        ("RGB", raw_frames),
        ("target mask", [_overlay_mask(raw_frames[i], mask_up[idx]) for i, idx in enumerate(frames)]),
    ]
    for block, attn in zip(block_ids, attn_maps):
        attn_up = normalize_heatmap(resize_volume(_as_t_h_w(attn), (T_raw, H_raw, W_raw)))
        mark = "*" if selected and int(block) in selected else " "
        rows.append((f"{mark} block {int(block)}", [_overlay_heat(raw_frames[i], attn_up[idx]) for i, idx in enumerate(frames)]))

    mean_attn = normalize_heatmap(
        resize_volume(_mean_selected_attention(attn_maps, block_ids, selected), (T_raw, H_raw, W_raw))
    )
    rows.append(("selected mean", [_overlay_heat(raw_frames[i], mean_attn[idx]) for i, idx in enumerate(frames)]))
    _save_grid(path, rows, frame_labels)


def render_temporal_metric_strip(
    path: Path,
    metrics_by_frame: dict[int, list[float]],
    title: str,
    vmin: float = 0.0,
    vmax: float | None = None,
) -> None:
    if not metrics_by_frame:
        return
    blocks = list(metrics_by_frame)
    T = max(len(values) for values in metrics_by_frame.values())
    if vmax is None:
        vmax = max(max(values) for values in metrics_by_frame.values() if values) if metrics_by_frame else 1.0
    vmax = max(vmax, vmin + 1e-6)
    cell_w, cell_h = 18, 28
    label_w, header_h = 160, 54
    font = _load_font(18)
    small_font = _load_font(14)
    canvas = Image.new("RGB", (label_w + T * cell_w, header_h + len(blocks) * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill=(0, 0, 0), font=font)
    for t in range(0, T, max(1, T // 8)):
        draw.text((label_w + t * cell_w, 32), str(t), fill=(60, 60, 60), font=small_font)
    for row, block in enumerate(blocks):
        y = header_h + row * cell_h
        draw.text((10, y + 6), f"block {block}", fill=(0, 0, 0), font=small_font)
        values = metrics_by_frame[block]
        for t, value in enumerate(values):
            ratio = float(np.clip((value - vmin) / (vmax - vmin), 0, 1))
            color = (int(255 * ratio), int(190 * ratio), int(255 * (1 - ratio)))
            x = label_w + t * cell_w
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), fill=color)
    canvas.save(path, quality=95)


def framewise_mass_inside(attn_T_H_W: torch.Tensor, mask_T_H_W: torch.Tensor) -> list[float]:
    attn = _as_t_h_w(attn_T_H_W).clamp(min=0)
    mask = _as_t_h_w(mask_T_H_W).clamp(0, 1)
    if mask.shape != attn.shape:
        mask = resize_volume(mask, tuple(attn.shape), mode="nearest")
    denom = attn.flatten(1).sum(dim=1).clamp(min=1e-6)
    inside = (attn * mask).flatten(1).sum(dim=1)
    return (inside / denom).tolist()


def save_attention_debug_pack(
    output_dir: Path | str,
    prefix: str,
    raw_video_C_T_H_W: torch.Tensor,
    raw_mask_T_H_W: torch.Tensor,
    attn_maps: Sequence[torch.Tensor],
    block_ids: Sequence[int],
    selected_blocks: Iterable[int] | None = None,
    metric_mask_T_H_W: torch.Tensor | None = None,
    save_tensors: bool = True,
) -> dict:
    """Save a reusable attention debug pack.

    Outputs:
      - ``*_attention_trace.pt``: raw attention maps and masks for later analysis.
      - ``*_attention_overlay_grid.jpg``: RGB/mask/per-block attention overlays.
      - ``*_attention_temporal_mass.jpg``: per-frame mass inside mask strips.
      - ``*_attention_metrics.json``: scalar diagnostics.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected = sorted(int(x) for x in selected_blocks) if selected_blocks is not None else []
    metric_mask = _as_t_h_w(metric_mask_T_H_W) if metric_mask_T_H_W is not None else _as_t_h_w(raw_mask_T_H_W)
    attn_cpu = [_as_t_h_w(attn) for attn in attn_maps]
    block_cpu = [int(block) for block in block_ids[: len(attn_cpu)]]
    metrics = [compute_attention_metric(attn, metric_mask, block) for block, attn in zip(block_cpu, attn_cpu)]
    frame_metrics = {
        block: framewise_mass_inside(attn, metric_mask)
        for block, attn in zip(block_cpu, attn_cpu)
    }

    overlay_path = output / f"{prefix}_attention_overlay_grid.jpg"
    temporal_path = output / f"{prefix}_attention_temporal_mass.jpg"
    metrics_path = output / f"{prefix}_attention_metrics.json"
    trace_path = output / f"{prefix}_attention_trace.pt"

    render_attention_overlay_grid(
        overlay_path,
        raw_video_C_T_H_W,
        raw_mask_T_H_W,
        attn_cpu,
        block_cpu,
        selected_blocks=selected,
    )
    render_temporal_metric_strip(
        temporal_path,
        frame_metrics,
        title="Attention mass inside target mask",
        vmin=0.0,
        vmax=max([max(values) for values in frame_metrics.values() if values] + [1.0]),
    )

    summary = {
        "prefix": prefix,
        "block_ids": block_cpu,
        "selected_blocks": selected,
        "attention_shape": [list(attn.shape) for attn in attn_cpu],
        "raw_video_shape": list(raw_video_C_T_H_W.shape),
        "raw_mask_shape": list(_as_t_h_w(raw_mask_T_H_W).shape),
        "metric_mask_shape": list(metric_mask.shape),
        "overlay_figure": str(overlay_path),
        "temporal_figure": str(temporal_path),
        "trace_file": str(trace_path) if save_tensors else None,
        "metrics": [asdict(metric) for metric in metrics],
        "framewise_mass_inside": {str(k): v for k, v in frame_metrics.items()},
    }
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    if save_tensors:
        torch.save(
            {
                "block_ids": block_cpu,
                "selected_blocks": selected,
                "attn_maps": torch.stack(attn_cpu) if attn_cpu else torch.empty(0),
                "raw_mask": _as_t_h_w(raw_mask_T_H_W),
                "metric_mask": metric_mask,
                "metrics": summary["metrics"],
            },
            trace_path,
        )
    return summary


__all__ = [
    "TargetAttentionMetric",
    "compute_attention_metric",
    "framewise_mass_inside",
    "normalize_heatmap",
    "render_attention_overlay_grid",
    "render_temporal_metric_strip",
    "resize_volume",
    "save_attention_debug_pack",
]
