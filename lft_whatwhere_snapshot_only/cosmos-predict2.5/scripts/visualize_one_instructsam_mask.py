#!/usr/bin/env python3
"""Visualize one InstructSAM target mask for an image or first video frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra_path in (REPO_ROOT, REPO_ROOT / "packages" / "cosmos-oss"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from cosmos_predict2._src.predict2.target_aware.instructsam_mask import (
    InstructSAMTargetMaskGenerator,
    read_first_frame_image,
)


def torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(name)


def bbox_and_centroid(mask: np.ndarray) -> tuple[tuple[int, int, int, int] | None, tuple[int, int] | None]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), (int(xs.mean()), int(ys.mean()))


def overlay_mask(image: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGBA")
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(base.size, Image.Resampling.NEAREST)
    color = Image.new("RGBA", base.size, (255, 80, 20, 0))
    alpha_img = Image.fromarray((np.asarray(mask_img).astype(np.float32) * alpha).astype(np.uint8), mode="L")
    color.putalpha(alpha_img)
    return Image.alpha_composite(base, color).convert("RGB")


def draw_labeled_overlay(image: Image.Image, mask: np.ndarray, label: str) -> Image.Image:
    out = overlay_mask(image, mask)
    draw = ImageDraw.Draw(out)
    bbox, centroid = bbox_and_centroid(mask)
    if bbox is not None:
        draw.rectangle(bbox, outline=(0, 255, 255), width=4)
    if centroid is not None:
        cx, cy = centroid
        radius = 9
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(0, 255, 0), width=4)
    draw.rectangle((0, 0, out.width, 42), fill=(0, 0, 0))
    draw.text((8, 12), label[:150], fill=(255, 255, 255))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--combine-mode", choices=["best", "union"], default="best")
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--feature-mode", choices=["mask_query", "raw_seg", "decoder_dense"], default="decoder_dense")
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device_map: str | dict[str, str] = {"": "cuda:0"} if torch.cuda.is_available() else "cpu"
    generator = InstructSAMTargetMaskGenerator(
        args.model_path,
        source_root=args.source_root,
        device_map=device_map,
        torch_dtype=torch_dtype_from_name(args.torch_dtype),
    )
    image = read_first_frame_image(args.input_path)
    result = generator.predict(
        image,
        args.query,
        combine_mode=args.combine_mode,
        mask_threshold=args.mask_threshold,
        output_size=(image.height, image.width),
        feature_mode=args.feature_mode,
    )
    mask = result.mask_B_C_T_H_W[0, 0, 0].detach().cpu().numpy() > 0
    bbox, centroid = bbox_and_centroid(mask)
    feature_shape = None
    if result.feature_B_L_D is not None:
        feature_shape = list(result.feature_B_L_D.shape)

    first_frame_path = args.output_dir / "first_frame.jpg"
    mask_path = args.output_dir / "instructsam_mask.png"
    overlay_path = args.output_dir / "instructsam_mask_overlay.jpg"
    summary_path = args.output_dir / "instructsam_mask_summary.json"
    image.save(first_frame_path, quality=95)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)
    draw_labeled_overlay(image, mask, f"score={result.score} query={args.query}").save(overlay_path, quality=95)

    summary = {
        "input_path": str(args.input_path),
        "query": args.query,
        "model_path": str(args.model_path),
        "source_root": str(args.source_root),
        "combine_mode": args.combine_mode,
        "mask_threshold": args.mask_threshold,
        "feature_mode": args.feature_mode,
        "instructsam_text": result.text,
        "score": result.score,
        "mask_pixels": int(mask.sum()),
        "mask_occupancy": float(mask.mean()),
        "mask_bbox_xyxy": bbox,
        "mask_centroid_xy": centroid,
        "feature_shape": feature_shape,
        "first_frame": str(first_frame_path),
        "mask_png": str(mask_path),
        "overlay": str(overlay_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
