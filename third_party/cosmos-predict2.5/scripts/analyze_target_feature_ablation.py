#!/usr/bin/env python3
"""Create visual and numeric diagnostics for target-feature ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["keep", "zero", "drop", "wrong_black_mug", "precise_prompt"])
    parser.add_argument("--mask-npz", type=Path, default=None)
    parser.add_argument("--output-prefix", default="feature_ablation")
    parser.add_argument("--frames", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--tile-width", type=int, default=384)
    return parser.parse_args()


def read_video(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return np.stack(frames, axis=0)


def load_mask(path: Path | None, height: int, width: int) -> np.ndarray | None:
    if path is None:
        return None
    data = np.load(path)
    key = "masks" if "masks" in data.files else data.files[0]
    mask = np.asarray(data[key])
    mask = np.squeeze(mask)
    while mask.ndim > 2:
        mask = mask[0]
    mask = (mask > 0).astype(np.uint8)
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def get_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def label_tile(image: np.ndarray, label: str, font: ImageFont.ImageFont, label_h: int = 38) -> Image.Image:
    pil = Image.fromarray(image).convert("RGB")
    out = Image.new("RGB", (pil.width, pil.height + label_h), (255, 255, 255))
    out.paste(pil, (0, label_h))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, label_h), fill=(0, 0, 0))
    draw.text((8, 7), label, fill=(255, 255, 255), font=font)
    return out


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame.copy()
    color = np.zeros_like(out)
    color[..., 0] = 255
    color[..., 1] = 40
    out[mask] = (0.45 * out[mask] + 0.55 * color[mask]).astype(np.uint8)
    ys, xs = np.where(mask)
    if len(xs) > 0:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 3)
        cv2.circle(out, (int(xs.mean()), int(ys.mean())), 6, (0, 255, 0), 3)
    return out


def resize_tile(tile: Image.Image, width: int) -> Image.Image:
    height = int(round(tile.height * width / tile.width))
    return tile.resize((width, height), Image.Resampling.BILINEAR)


def save_grid(rows: list[list[Image.Image]], path: Path, tile_width: int) -> None:
    resized_rows: list[list[Image.Image]] = []
    for row in rows:
        resized_rows.append([resize_tile(tile, tile_width) for tile in row])
    row_heights = [max(tile.height for tile in row) for row in resized_rows]
    col_count = max(len(row) for row in resized_rows)
    canvas = Image.new("RGB", (tile_width * col_count, sum(row_heights)), (245, 245, 245))
    y = 0
    for row, row_h in zip(resized_rows, row_heights):
        x = 0
        for tile in row:
            canvas.paste(tile, (x, y))
            x += tile_width
        y += row_h
    canvas.save(path, quality=95)


def metric_dict(video: np.ndarray, ref: np.ndarray, mask: np.ndarray | None) -> dict[str, float]:
    n = min(len(video), len(ref))
    video = video[:n].astype(np.float32)
    ref = ref[:n].astype(np.float32)
    diff = np.abs(video - ref)
    out = {
        "mean_abs_rgb": float(diff.mean()),
        "max_abs_rgb": float(diff.max()),
        "num_frames_compared": int(n),
    }
    if mask is not None and mask.any():
        mask3 = mask[None, :, :, None]
        inside = diff[mask3.repeat(n, axis=0).repeat(3, axis=3)]
        outside = diff[(~mask)[None, :, :, None].repeat(n, axis=0).repeat(3, axis=3)]
        out["target_mask_mean_abs_rgb"] = float(inside.mean())
        out["background_mean_abs_rgb"] = float(outside.mean())
        out["target_to_background_diff_ratio"] = float((inside.mean() + 1e-6) / (outside.mean() + 1e-6))
        out["target_mask_area_fraction"] = float(mask.mean())
    return out


def main() -> None:
    args = parse_args()
    videos: dict[str, np.ndarray] = {}
    for variant in args.variants:
        path = args.run_root / variant / "sample_000_generated.mp4"
        if not path.exists():
            raise FileNotFoundError(path)
        videos[variant] = read_video(path)

    ref = videos[args.variants[0]]
    height, width = ref.shape[1:3]
    mask = load_mask(args.mask_npz, height, width)

    frame_indices = sorted(set(int(round(frac * (len(ref) - 1))) for frac in args.frames))
    font = get_font(20)
    rows: list[list[Image.Image]] = []
    for variant in args.variants:
        video = videos[variant]
        row: list[Image.Image] = []
        for idx in frame_indices:
            idx = min(idx, len(video) - 1)
            frame = video[idx]
            if mask is not None and idx == frame_indices[0]:
                frame = overlay_mask(frame, mask)
                label = f"{variant} | f{idx} | target mask"
            else:
                label = f"{variant} | f{idx}"
            row.append(label_tile(frame, label, font))
        rows.append(row)

    grid_path = args.run_root / f"{args.output_prefix}_contact_sheet.jpg"
    save_grid(rows, grid_path, args.tile_width)

    diff_rows: list[list[Image.Image]] = []
    for variant in args.variants[1:]:
        video = videos[variant]
        row = []
        for idx in frame_indices:
            idx = min(idx, len(video) - 1, len(ref) - 1)
            diff = np.abs(video[idx].astype(np.int16) - ref[idx].astype(np.int16)).astype(np.uint8)
            diff = np.clip(diff * 4, 0, 255).astype(np.uint8)
            row.append(label_tile(diff, f"{variant} - keep | f{idx} | x4", font))
        diff_rows.append(row)
    diff_grid_path = args.run_root / f"{args.output_prefix}_diff_vs_keep.jpg"
    save_grid(diff_rows, diff_grid_path, args.tile_width)

    metrics = {
        "run_root": str(args.run_root),
        "reference": args.variants[0],
        "mask_npz": str(args.mask_npz) if args.mask_npz else None,
        "frame_indices": frame_indices,
        "variants": {},
    }
    for variant in args.variants:
        metrics["variants"][variant] = metric_dict(videos[variant], ref, mask)

    metrics_path = args.run_root / f"{args.output_prefix}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps({"contact_sheet": str(grid_path), "diff_sheet": str(diff_grid_path), "metrics": str(metrics_path)}, indent=2))


if __name__ == "__main__":
    main()
