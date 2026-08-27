#!/usr/bin/env python3
"""Visualize whether InstructSAM dense target features expose target location.

This script does not run Cosmos.  It probes the saved InstructSAM
``target_dense_feature`` tensors directly as 32x32 spatial maps and compares
their saliency against a GT target mask used only for analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrot-dense", type=Path, required=True)
    parser.add_argument("--banana-dense", type=Path, required=True)
    parser.add_argument("--carrot-raw", type=Path, default=None)
    parser.add_argument("--banana-raw", type=Path, default=None)
    parser.add_argument("--mask-npz", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stem", default="target_dense_feature")
    return parser.parse_args()


def load_feature(path: Path) -> tuple[torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload if isinstance(payload, dict) else {}
    feat = payload["target_feature"] if isinstance(payload, dict) else payload
    feat = torch.as_tensor(feat, dtype=torch.float32)
    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = feat[0]
    if feat.ndim != 2:
        raise ValueError(f"Expected [L,D] feature at {path}, got {tuple(feat.shape)}")
    return torch.nan_to_num(feat), meta


def spatialize(feat: torch.Tensor) -> torch.Tensor:
    n, dim = feat.shape
    side = int(round(n**0.5))
    if side * side != n:
        raise ValueError(f"Feature token count must be square, got {n}")
    return feat.reshape(side, side, dim)


def load_mask_32(path: Path, side: int) -> np.ndarray:
    data = np.load(path)
    key = "masks" if "masks" in data.files else data.files[0]
    mask = np.asarray(data[key])
    mask = np.squeeze(mask)
    while mask.ndim > 2:
        mask = mask[0]
    mask = (mask > 0).astype(np.uint8)
    return cv2.resize(mask, (side, side), interpolation=cv2.INTER_NEAREST).astype(bool)


def read_first_frame(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def zscore_map(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return (arr - float(arr.mean())) / (float(arr.std()) + 1e-6)


def heat_tile(
    base_rgb: np.ndarray,
    heat_32: np.ndarray,
    title: str,
    subtitle: str,
    mask_32: np.ndarray | None,
    width: int = 420,
) -> Image.Image:
    h, w = base_rgb.shape[:2]
    heat = normalize01(heat_32)
    heat_up = cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC)
    color = cv2.applyColorMap((heat_up * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    overlay = (0.56 * base_rgb.astype(np.float32) + 0.44 * color.astype(np.float32)).clip(0, 255).astype(np.uint8)
    if mask_32 is not None:
        mask_up = cv2.resize(mask_32.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        ys, xs = np.where(mask_up)
        if len(xs):
            cv2.rectangle(overlay, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), (0, 255, 255), 4)
    tile = Image.fromarray(overlay)
    tile_h = int(round(tile.height * width / tile.width))
    tile = tile.resize((width, tile_h), Image.Resampling.BILINEAR)
    out = Image.new("RGB", (width, tile_h + 58), (255, 255, 255))
    out.paste(tile, (0, 58))
    draw = ImageDraw.Draw(out)
    font = get_font(18)
    small = get_font(13)
    draw.rectangle((0, 0, width, 58), fill=(0, 0, 0))
    draw.text((8, 7), title[:48], fill=(255, 255, 255), font=font)
    draw.text((8, 33), subtitle[:72], fill=(210, 210, 210), font=small)
    return out


def mask_tile(base_rgb: np.ndarray, mask_32: np.ndarray, title: str, width: int = 420) -> Image.Image:
    h, w = base_rgb.shape[:2]
    mask_up = cv2.resize(mask_32.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay = base_rgb.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    overlay[mask_up] = (0.45 * overlay[mask_up] + 0.55 * red[mask_up]).astype(np.uint8)
    ys, xs = np.where(mask_up)
    if len(xs):
        cv2.rectangle(overlay, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), (0, 255, 255), 4)
    tile = Image.fromarray(overlay)
    tile_h = int(round(tile.height * width / tile.width))
    tile = tile.resize((width, tile_h), Image.Resampling.BILINEAR)
    out = Image.new("RGB", (width, tile_h + 58), (255, 255, 255))
    out.paste(tile, (0, 58))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, width, 58), fill=(0, 0, 0))
    draw.text((8, 12), title, fill=(255, 255, 255), font=get_font(18))
    draw.text((8, 36), f"mask area @32x32 = {mask_32.mean():.4f}", fill=(210, 210, 210), font=get_font(13))
    return out


def get_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def stats_for_map(arr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float32)
    inside = arr[mask]
    outside = arr[~mask]
    out = {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "inside_mean": float(inside.mean()) if inside.size else float("nan"),
        "outside_mean": float(outside.mean()) if outside.size else float("nan"),
        "inside_minus_outside": float(inside.mean() - outside.mean()) if inside.size and outside.size else float("nan"),
        "inside_outside_ratio": float((inside.mean() + 1e-6) / (outside.mean() + 1e-6))
        if inside.size and outside.size
        else float("nan"),
    }
    return out


def save_grid(tiles: list[Image.Image], path: Path, cols: int = 3) -> None:
    rows = [tiles[i : i + cols] for i in range(0, len(tiles), cols)]
    col_w = max(tile.width for tile in tiles)
    row_h = [max(tile.height for tile in row) for row in rows]
    canvas = Image.new("RGB", (col_w * cols, sum(row_h)), (245, 245, 245))
    y = 0
    for row, h in zip(rows, row_h):
        x = 0
        for tile in row:
            canvas.paste(tile, (x, y))
            x += col_w
        y += h
    canvas.save(path, quality=95)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    carrot_dense, carrot_meta = load_feature(args.carrot_dense)
    banana_dense, banana_meta = load_feature(args.banana_dense)
    if carrot_dense.shape != banana_dense.shape:
        raise ValueError(f"Dense feature shape mismatch: {carrot_dense.shape} vs {banana_dense.shape}")

    side = int(round(carrot_dense.shape[0] ** 0.5))
    carrot_grid = spatialize(carrot_dense)
    banana_grid = spatialize(banana_dense)
    mask_32 = load_mask_32(args.mask_npz, side)
    frame = read_first_frame(args.video)

    carrot_norm = carrot_grid.norm(dim=-1).numpy()
    banana_norm = banana_grid.norm(dim=-1).numpy()
    dense_l2 = (carrot_grid - banana_grid).norm(dim=-1).numpy()
    dense_cos_dist = (
        1.0
        - F.cosine_similarity(carrot_grid.reshape(-1, carrot_grid.shape[-1]), banana_grid.reshape(-1, banana_grid.shape[-1]), dim=-1)
        .reshape(side, side)
        .numpy()
    )
    carrot_mean = carrot_dense.mean(dim=0, keepdim=True)
    banana_mean = banana_dense.mean(dim=0, keepdim=True)
    carrot_cos_mean = F.cosine_similarity(carrot_dense, carrot_mean.expand_as(carrot_dense), dim=-1).reshape(side, side).numpy()
    banana_cos_mean = F.cosine_similarity(banana_dense, banana_mean.expand_as(banana_dense), dim=-1).reshape(side, side).numpy()

    centered = carrot_dense - carrot_dense.mean(dim=0, keepdim=True)
    _, _, vh = torch.pca_lowrank(centered, q=3, center=False)
    pc1 = (centered @ vh[:, 0]).reshape(side, side).numpy()
    if np.isfinite(pc1[mask_32]).any() and np.isfinite(pc1[~mask_32]).any():
        if pc1[mask_32].mean() < pc1[~mask_32].mean():
            pc1 = -pc1
    pc1_z = zscore_map(pc1)

    maps = {
        "carrot_dense_norm": carrot_norm,
        "banana_dense_norm": banana_norm,
        "carrot_pc1_oriented_z": pc1_z,
        "carrot_cos_to_dense_mean": carrot_cos_mean,
        "banana_cos_to_dense_mean": banana_cos_mean,
        "carrot_vs_banana_l2": dense_l2,
        "carrot_vs_banana_cos_distance": dense_cos_dist,
    }

    raw_summary = {}
    if args.carrot_raw and args.banana_raw and args.carrot_raw.exists() and args.banana_raw.exists():
        carrot_raw, carrot_raw_meta = load_feature(args.carrot_raw)
        banana_raw, banana_raw_meta = load_feature(args.banana_raw)
        n = min(carrot_raw.numel(), banana_raw.numel())
        raw_summary = {
            "carrot_raw_shape": list(carrot_raw.shape),
            "banana_raw_shape": list(banana_raw.shape),
            "carrot_raw_query": carrot_raw_meta.get("query"),
            "banana_raw_query": banana_raw_meta.get("query"),
            "raw_global_cosine": float(F.cosine_similarity(carrot_raw.reshape(-1)[:n], banana_raw.reshape(-1)[:n], dim=0)),
            "raw_relative_l2": float(
                torch.linalg.vector_norm(carrot_raw.reshape(-1)[:n] - banana_raw.reshape(-1)[:n])
                / torch.linalg.vector_norm(carrot_raw.reshape(-1)[:n]).clamp_min(1e-6)
            ),
        }

    dense_summary = {
        "carrot_dense_shape": list(carrot_dense.shape),
        "banana_dense_shape": list(banana_dense.shape),
        "carrot_dense_query": carrot_meta.get("query"),
        "banana_dense_query": banana_meta.get("query"),
        "dense_global_cosine": float(F.cosine_similarity(carrot_dense.reshape(-1), banana_dense.reshape(-1), dim=0)),
        "dense_relative_l2": float(
            torch.linalg.vector_norm(carrot_dense.reshape(-1) - banana_dense.reshape(-1))
            / torch.linalg.vector_norm(carrot_dense.reshape(-1)).clamp_min(1e-6)
        ),
        "mask_area_32": float(mask_32.mean()),
    }

    metrics = {
        "dense_summary": dense_summary,
        "raw_summary": raw_summary,
        "maps": {name: stats_for_map(arr, mask_32) for name, arr in maps.items()},
    }

    (args.out_dir / f"{args.stem}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    for name, arr in maps.items():
        np.save(args.out_dir / f"{args.stem}_{name}.npy", arr)

    tiles = [mask_tile(frame, mask_32, "GT target mask (analysis only)")]
    for name, arr in maps.items():
        st = metrics["maps"][name]
        subtitle = f"in-out={st['inside_minus_outside']:.4f}, ratio={st['inside_outside_ratio']:.3f}"
        tiles.append(heat_tile(frame, arr, name, subtitle, mask_32))
    figure_path = args.out_dir / f"{args.stem}_spatial_maps.jpg"
    save_grid(tiles, figure_path, cols=3)

    print(json.dumps({"figure": str(figure_path), "metrics": str(args.out_dir / f"{args.stem}_metrics.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
