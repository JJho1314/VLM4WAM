#!/usr/bin/env python3
"""Visualize the mask-free target matching head logits.

This is a companion to ``visualize_tavid_cross_attention.py``.  It runs the
training forward pass on a validation split, reads
``model.net.target_matching_logits_B_T_H_W``, and saves probability heatmaps
against the training-time GT mask.  The GT mask is used only for evaluation and
visualization; it is not fed as an inference-time condition.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "packages" / "cosmos-oss"))

from cosmos_oss.init import init_environment
from cosmos_predict2._src.imaginaire.utils import distributed, misc
from scripts.visualize_tavid_cross_attention import (
    add_dummy_text_embeddings,
    add_online_text_embeddings,
    frame_indices,
    forward_with_optional_vae_offload,
    heat_color,
    load_model_and_dataloader,
    normalize_heatmap,
    overlay_heat,
    overlay_mask,
    save_grid,
    save_horizontal_tiles,
    to_uint8_frame,
    upsample_volume,
)


@dataclass
class MatchingMetric:
    prob_inside: float
    prob_outside: float
    inside_outside_ratio: float
    prob_mass_inside: float
    mask_mass: float
    pred_mean: float
    pred_max: float
    pred_min: float


@dataclass
class SampleRecord:
    sample_index: int
    caption: str
    tgt_token_index: int
    initial_matching_figure: str
    grid_matching_figure: str
    metric: MatchingMetric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=160)
    parser.add_argument("--num-conditional-frames", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model-only-load", action="store_true")
    parser.add_argument("--skip-init-environment", action="store_true")
    parser.add_argument(
        "--token-source",
        choices=("config", "text", "feature", "text_feature"),
        default="feature",
        help="Passed through to the shared model loader; matching itself does not use this filter.",
    )
    parser.add_argument("--dummy-text-embeddings", action="store_true")
    parser.add_argument("--dummy-text-tokens", type=int, default=512)
    parser.add_argument("--offload-denoiser-during-vae", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _metric(prob_T_H_W: torch.Tensor, mask_T_H_W: torch.Tensor) -> MatchingMetric:
    prob = prob_T_H_W.detach().float().cpu().clamp(0, 1)
    mask = mask_T_H_W.detach().float().cpu().clamp(0, 1)
    if mask.shape != prob.shape:
        mask = upsample_volume(mask, tuple(prob.shape), mode="nearest")
    eps = 1e-6
    inv = 1.0 - mask
    prob_sum = prob.sum() + eps
    inside = (prob * mask).sum() / (mask.sum() + eps)
    outside = (prob * inv).sum() / (inv.sum() + eps)
    return MatchingMetric(
        prob_inside=float(inside.item()),
        prob_outside=float(outside.item()),
        inside_outside_ratio=float((inside / (outside + eps)).item()),
        prob_mass_inside=float(((prob * mask).sum() / prob_sum).item()),
        mask_mass=float(mask.mean().item()),
        pred_mean=float(prob.mean().item()),
        pred_max=float(prob.max().item()),
        pred_min=float(prob.min().item()),
    )


def _save_panel_grid(path: Path, tiles: list[tuple[str, np.ndarray]], columns: int = 2, max_tile_width: int = 560) -> None:
    if not tiles:
        return
    tile_h, tile_w = tiles[0][1].shape[:2]
    scale = min(1.0, max_tile_width / float(tile_w))
    out_w = max(1, int(round(tile_w * scale)))
    out_h = max(1, int(round(tile_h * scale)))
    header_h = 30
    rows = int(np.ceil(len(tiles) / columns))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font = None
    canvas = Image.new("RGB", (columns * out_w, rows * (header_h + out_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(tiles):
        col = idx % columns
        row = idx // columns
        x = col * out_w
        y = row * (header_h + out_h)
        draw.rectangle((x, y, x + out_w, y + header_h), fill=(0, 0, 0))
        draw.text((x + 8, y + 6), label, fill=(255, 255, 255), font=font)
        pil = Image.fromarray(image).resize((out_w, out_h), Image.BILINEAR)
        canvas.paste(pil, (x, y + header_h))
    canvas.save(path, quality=95)


def _make_initial(path: Path, raw: torch.Tensor, mask: torch.Tensor, prob: torch.Tensor) -> None:
    T_raw = raw.shape[1]
    H_raw, W_raw = raw.shape[-2:]
    frame_idx = 0
    frame_rgb = to_uint8_frame(raw[:, frame_idx])
    mask_up = upsample_volume(mask, (T_raw, H_raw, W_raw), mode="nearest")
    prob_up = upsample_volume(prob, (T_raw, H_raw, W_raw), mode="trilinear").clamp(0, 1)
    _save_panel_grid(
        path,
        [
            ("initial frame", frame_rgb),
            ("target mask", overlay_mask(frame_rgb, mask_up[frame_idx])),
            ("matching prob (absolute)", heat_color(prob_up[frame_idx])),
            ("prob overlay (absolute)", overlay_heat(frame_rgb, prob_up[frame_idx])),
        ],
    )


def _make_grid(path: Path, raw: torch.Tensor, mask: torch.Tensor, prob: torch.Tensor) -> None:
    T_raw = raw.shape[1]
    H_raw, W_raw = raw.shape[-2:]
    frames = frame_indices(T_raw)
    frame_labels = [f"f{idx}" for idx in frames]
    raw_frames = [to_uint8_frame(raw[:, idx]) for idx in frames]
    mask_up = upsample_volume(mask, (T_raw, H_raw, W_raw), mode="nearest")
    prob_up = upsample_volume(prob, (T_raw, H_raw, W_raw), mode="trilinear").clamp(0, 1)
    prob_norm = normalize_heatmap(prob_up)
    rows = [
        ("RGB", raw_frames),
        ("target mask", [overlay_mask(raw_frames[i], mask_up[idx]) for i, idx in enumerate(frames)]),
        ("matching prob", [overlay_heat(raw_frames[i], prob_up[idx]) for i, idx in enumerate(frames)]),
        ("norm matching prob", [overlay_heat(raw_frames[i], prob_norm[idx]) for i, idx in enumerate(frames)]),
    ]
    save_grid(path, rows, frame_labels)


def collect(args: argparse.Namespace) -> tuple[list[SampleRecord], int]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, dataloader, loaded_iter = load_model_and_dataloader(args, args.checkpoint, [8, 12, 16, 20])
    output_dir = Path(args.output_dir)
    records: list[SampleRecord] = []
    saved = 0
    with torch.no_grad():
        for batch_idx, cpu_batch in enumerate(dataloader):
            if batch_idx >= args.max_batches or saved >= args.num_samples:
                break
            if "target_mask" not in cpu_batch or float(cpu_batch["target_mask"].sum()) <= 0:
                continue

            raw_cpu = cpu_batch[model.input_data_key][0].detach().cpu()
            mask_cpu = cpu_batch["target_mask"][0, 0].detach().cpu()
            caption = cpu_batch.get(model.input_caption_key, [""])[0]

            data_batch = misc.to(cpu_batch, device="cuda")
            data_batch["num_conditional_frames"] = torch.full(
                (data_batch[model.input_data_key].shape[0],),
                args.num_conditional_frames,
                dtype=torch.long,
                device="cuda",
            )
            add_online_text_embeddings(model, data_batch)
            add_dummy_text_embeddings(args, model, data_batch)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_batch, loss = forward_with_optional_vae_offload(args, model, data_batch)

            logits = getattr(model.net, "target_matching_logits_B_T_H_W", None)
            latent_mask = getattr(model.net, "tavid_target_mask_B_T_H_W", None)
            if logits is None or latent_mask is None:
                continue
            prob = torch.sigmoid(logits[0]).detach().float().cpu()
            latent_mask_cpu = latent_mask[0].detach().float().cpu()

            tgt_indices = data_batch.get("tgt_token_indices", torch.full((1,), -1, device="cuda"))
            if tgt_indices.ndim > 1:
                tgt_indices = tgt_indices[:, 0]
            tgt_index = int(tgt_indices[0].item()) if tgt_indices.numel() else -1

            if distributed.is_rank0():
                metric = _metric(prob, latent_mask_cpu)
                initial_path = output_dir / f"sample_{saved:03d}_initial_matching_logits.jpg"
                grid_path = output_dir / f"sample_{saved:03d}_matching_logits_grid.jpg"
                _make_initial(initial_path, raw_cpu, mask_cpu, prob)
                _make_grid(grid_path, raw_cpu, mask_cpu, prob)
                record = SampleRecord(
                    sample_index=saved,
                    caption=caption,
                    tgt_token_index=tgt_index,
                    initial_matching_figure=str(initial_path),
                    grid_matching_figure=str(grid_path),
                    metric=metric,
                )
                records.append(record)
                print(
                    json.dumps(
                        {
                            "sample_index": saved,
                            "caption": caption,
                            "tgt_token_index": tgt_index,
                            "metric": asdict(metric),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            saved += 1
    return records, loaded_iter


def main() -> None:
    args = parse_args()
    if not args.skip_init_environment:
        init_environment()
    output_dir = Path(args.output_dir)
    if distributed.is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)

    records, loaded_iter = collect(args)
    if not distributed.is_rank0():
        return
    summary = {
        "checkpoint": args.checkpoint,
        "loaded_iter": loaded_iter,
        "split": args.split,
        "num_samples": len(records),
        "samples": [
            {
                **asdict(record),
                "metric": asdict(record.metric),
            }
            for record in records
        ],
    }
    (output_dir / "target_matching_logits_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
