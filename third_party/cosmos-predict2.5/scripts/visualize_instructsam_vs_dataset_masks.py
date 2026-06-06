#!/usr/bin/env python3
"""Compare InstructSAM-predicted masks against dataset masks for target phrases."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from cosmos_predict2._src.predict2.target_aware.instructsam_mask import InstructSAMTargetMaskGenerator
from visualize_target_embedding_diagnostics import (
    bbox_and_centroid,
    first_range_start,
    load_frame,
    load_mask_frame,
    overlay_mask,
)


def torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def extract_phrase(caption: str) -> str:
    if "[TGT]" not in caption:
        return "target object"
    tail = caption.split("[TGT]", 1)[1].strip()
    for sep in [",", ".", ";", ":", " and ", " then "]:
        if sep in tail:
            tail = tail.split(sep, 1)[0]
    words = tail.strip().split()
    stop = {"in", "inside", "on", "onto", "to", "from", "off", "over", "under", "with", "near", "next"}
    kept = []
    for word in words:
        clean = word.strip(" ,.;:!?").lower()
        if kept and clean in stop:
            break
        kept.append(word.strip(" ,.;:!?"))
    return " ".join(kept).strip() or "target object"


def label_image(image: Image.Image, label: str) -> Image.Image:
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, 28), fill=(0, 0, 0))
    draw.text((6, 8), label[:110], fill=(255, 255, 255))
    return out


def draw_bbox_centroid(image: Image.Image, mask: np.ndarray, label: str) -> Image.Image:
    if mask.shape != (image.height, image.width):
        mask = resize_mask(mask, (image.height, image.width))
    out = overlay_mask(image, mask)
    draw = ImageDraw.Draw(out)
    bbox, centroid = bbox_and_centroid(mask)
    if bbox:
        draw.rectangle(bbox, outline=(0, 255, 255), width=4)
    if centroid:
        cx, cy = centroid
        r = 10
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(0, 255, 0), width=4)
    draw.rectangle((0, 0, out.width, 28), fill=(0, 0, 0))
    draw.text((6, 8), label[:110], fill=(255, 255, 255))
    return out


def make_grid(rows: list[list[Image.Image]], tile_w: int = 320) -> Image.Image:
    if not rows:
        return Image.new("RGB", (1, 1), "white")
    thumbs = []
    for row in rows:
        trow = []
        for img in row:
            scale = tile_w / img.width
            tile_h = int(img.height * scale)
            trow.append(img.resize((tile_w, tile_h), Image.Resampling.BILINEAR))
        thumbs.append(trow)
    tile_h = max(img.height for row in thumbs for img in row)
    ncols = max(len(row) for row in thumbs)
    canvas = Image.new("RGB", (tile_w * ncols, tile_h * len(thumbs)), "white")
    for r, row in enumerate(thumbs):
        for c, img in enumerate(row):
            canvas.paste(img, (c * tile_w, r * tile_h))
    return canvas


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def resize_mask(mask: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    if mask.shape == (h, w):
        return mask.astype(bool)
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_img = mask_img.resize((w, h), Image.Resampling.NEAREST)
    return np.asarray(mask_img) > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--combine-mode", choices=["best", "union"], default="best")
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        device_map: str | dict[str, str] = {"": "cuda:0"}
    else:
        device_map = "cpu"

    generator = InstructSAMTargetMaskGenerator(
        args.model_path,
        source_root=args.source_root,
        device_map=device_map,
        torch_dtype=torch_dtype_from_name(args.torch_dtype),
    )

    rows = list(csv.DictReader(open(args.dataset_dir / "metadata.csv")))[: args.limit]
    panels = []
    records = []
    for row in rows:
        stem = row["name"]
        try:
            caption = row.get("target_caption") or (args.dataset_dir / "metas" / f"{stem}.txt").read_text().strip()
            phrase = row.get("rewrite_object_phrase") or extract_phrase(caption)
            phrase = phrase.strip()
            if phrase.lower().startswith(("the ", "a ", "an ")):
                phrase_query = " ".join(phrase.split()[1:])
            else:
                phrase_query = phrase
            query = f"Please segment '{phrase_query}' in the image."
            frame_idx = first_range_start(args.dataset_dir, stem)
            frame = load_frame(args.dataset_dir / "videos" / f"{stem}.mp4", frame_idx)
            dataset_mask = load_mask_frame(args.dataset_dir / "masks" / f"{stem}.npz", frame_idx)
            result = generator.predict(
                frame,
                query,
                combine_mode=args.combine_mode,
                mask_threshold=args.mask_threshold,
                output_size=(frame.height, frame.width),
                feature_mode="mask_query",
            )
            pred_raw = result.mask_B_C_T_H_W[0, 0, 0].detach().cpu().numpy() > 0
            pred = resize_mask(pred_raw, (frame.height, frame.width))
            pred_for_metric = resize_mask(pred_raw, dataset_mask.shape)
            pred_path = args.output_dir / f"{stem}_instructsam_mask.png"
            overlay_path = args.output_dir / f"{stem}_compare.jpg"
            Image.fromarray((pred.astype(np.uint8) * 255)).save(pred_path)
            compare = [
                label_image(frame, f"{stem} | {phrase_query}"),
                draw_bbox_centroid(frame, dataset_mask, f"dataset mask pixels={int(dataset_mask.sum())}"),
                draw_bbox_centroid(frame, pred, f"InstructSAM score={result.score} pixels={int(pred.sum())}"),
            ]
            make_grid([compare], tile_w=360).save(overlay_path, quality=95)
            panels.append(compare)
            records.append(
                {
                    "stem": stem,
                    "phrase": phrase_query,
                    "query": query,
                    "status": "ok",
                    "score": result.score,
                    "dataset_pixels": int(dataset_mask.sum()),
                    "instructsam_pixels": int(pred.sum()),
                    "raw_instructsam_shape": list(pred_raw.shape),
                    "resized_instructsam_shape": list(pred.shape),
                    "dataset_occupancy": float(dataset_mask.mean()),
                    "instructsam_occupancy": float(pred.mean()),
                    "iou_with_dataset_mask": iou(dataset_mask, pred_for_metric),
                    "compare": str(overlay_path),
                    "mask_png": str(pred_path),
                    "instructsam_text": result.text,
                }
            )
        except Exception as exc:
            records.append(
                {
                    "stem": stem,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        print(json.dumps(records[-1], ensure_ascii=False), flush=True)

    make_grid(panels, tile_w=320).save(args.output_dir / "instructsam_vs_dataset_mask_contact.jpg", quality=95)
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "num_samples": len(records),
        "contact_sheet": str(args.output_dir / "instructsam_vs_dataset_mask_contact.jpg"),
        "records": records,
    }
    (args.output_dir / "instructsam_vs_dataset_mask_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
