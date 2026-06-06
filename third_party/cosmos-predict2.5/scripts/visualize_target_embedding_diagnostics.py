#!/usr/bin/env python3
"""Visual diagnostics for target feature and lightweight spatial embeddings.

This is an analysis tool. It does not change model inputs. It visualizes:
  1. mask-derived bbox/centroid spatial anchors;
  2. cosine similarity among InstructSAM target feature tokens;
  3. nearest-neighbor retrieval using the current target_feature embedding.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def load_frame(video_path: Path, frame_idx: int) -> Image.Image:
    from decord import VideoReader, cpu

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    frame_idx = max(0, min(int(frame_idx), len(vr) - 1))
    arr = vr.get_batch([frame_idx]).asnumpy()[0]
    return Image.fromarray(arr).convert("RGB")


def load_mask_frame(mask_path: Path, frame_idx: int) -> np.ndarray:
    npz = np.load(mask_path, allow_pickle=True)
    if "masks_packed" in npz.files and "shape" in npz.files:
        shape = tuple(int(x) for x in npz["shape"].tolist())
        frame_idx = max(0, min(int(frame_idx), shape[0] - 1))
        flat_pixels = int(np.prod(shape[1:]))
        row = np.unpackbits(npz["masks_packed"][frame_idx])[:flat_pixels]
        return row.reshape(shape[1:]).astype(bool)
    if "masks" in npz.files:
        arr = np.asarray(npz["masks"])
    else:
        arr = np.asarray(npz[npz.files[0]])
    while arr.ndim > 3:
        arr = arr.max(axis=0)
    if arr.ndim == 3:
        frame_idx = max(0, min(int(frame_idx), arr.shape[0] - 1))
        arr = arr[frame_idx]
    return arr.astype(bool)


def first_range_start(dataset_dir: Path, stem: str) -> int:
    ranges_path = dataset_dir / "frame_ranges.json"
    if not ranges_path.exists():
        return 0
    ranges = json.load(open(ranges_path)).get(stem)
    if not ranges:
        return 0
    return int(ranges[0][0])


def bbox_and_centroid(mask: np.ndarray) -> tuple[tuple[int, int, int, int] | None, tuple[float, float] | None]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), (float(xs.mean()), float(ys.mean()))


def overlay_mask(image: Image.Image, mask: np.ndarray, color=(255, 0, 0), alpha=0.42) -> Image.Image:
    img = image.convert("RGB")
    if mask.shape != (img.height, img.width):
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255).resize((img.width, img.height), Image.Resampling.NEAREST)
        mask = np.asarray(mask_img) > 0
    arr = np.asarray(img).astype(np.float32)
    c = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    arr[mask] = arr[mask] * (1.0 - alpha) + c * alpha
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def centroid_heatmap(size: tuple[int, int], centroid: tuple[float, float] | None, sigma: float = 45.0) -> Image.Image:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    if centroid is None:
        heat = np.zeros((h, w), dtype=np.float32)
    else:
        cx, cy = centroid
        heat = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(255 * heat, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(210 * (1 - np.abs(heat - 0.45) / 0.45), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)


def draw_anchor(image: Image.Image, mask: np.ndarray, label: str) -> Image.Image:
    out = overlay_mask(image, mask)
    draw = ImageDraw.Draw(out)
    bbox, centroid = bbox_and_centroid(mask)
    if bbox is not None:
        draw.rectangle(bbox, outline=(0, 255, 255), width=4)
    if centroid is not None:
        cx, cy = centroid
        r = 10
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(0, 255, 0), width=4)
        draw.line((cx - 16, cy, cx + 16, cy), fill=(0, 255, 0), width=3)
        draw.line((cx, cy - 16, cx, cy + 16), fill=(0, 255, 0), width=3)
    draw.rectangle((0, 0, out.width, 26), fill=(0, 0, 0))
    draw.text((6, 7), label[:95], fill=(255, 255, 255))
    return out


def make_grid(images: list[Image.Image], labels: list[str], ncols: int, tile_w: int = 320) -> Image.Image:
    thumbs = []
    for img, label in zip(images, labels):
        scale = tile_w / img.width
        tile_h = max(1, int(img.height * scale))
        tile = img.resize((tile_w, tile_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (tile_w, tile_h + 24), "white")
        canvas.paste(tile, (0, 24))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, tile_w, 24), fill=(0, 0, 0))
        draw.text((5, 6), label[:48], fill=(255, 255, 255))
        thumbs.append(canvas)
    if not thumbs:
        return Image.new("RGB", (1, 1), "white")
    ncols = max(1, int(ncols))
    nrows = math.ceil(len(thumbs) / ncols)
    tile_h = max(t.height for t in thumbs)
    canvas = Image.new("RGB", (tile_w * ncols, tile_h * nrows), "white")
    for idx, tile in enumerate(thumbs):
        x = (idx % ncols) * tile_w
        y = (idx // ncols) * tile_h
        canvas.paste(tile, (x, y))
    return canvas


def heat_color(x: np.ndarray) -> Image.Image:
    x = x.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    rgb = np.zeros((*x.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(255 * x, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(255 * (1 - np.abs(x - 0.5) / 0.5), 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(255 * (1 - x), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb)


def load_feature(path: Path) -> tuple[torch.Tensor, dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    feat = data["target_feature"].float()
    if feat.ndim == 3:
        feat = feat[0]
    return feat.mean(dim=0), data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--neighbors", type=int, default=4)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(dataset_dir / "metadata.csv")))
    rows = sorted(rows, key=lambda r: r["name"])

    records = []
    features = []
    for row in rows:
        stem = row["name"]
        feature_path = dataset_dir / "target_features" / f"{stem}.pt"
        if not feature_path.exists():
            continue
        feat, payload = load_feature(feature_path)
        feat = feat / (feat.norm() + 1e-6)
        features.append(feat)
        records.append(
            {
                "stem": stem,
                "phrase": payload.get("target_phrase") or row.get("rewrite_object_phrase", ""),
                "caption": payload.get("caption") or row.get("target_caption", ""),
                "score": payload.get("score"),
                "row": row,
            }
        )

    if not records:
        raise RuntimeError(f"No target_features found under {dataset_dir}")
    feats = torch.stack(features)
    sim = (feats @ feats.T).numpy()

    heat = heat_color(sim)
    heat = heat.resize((max(540, len(records) * 10), max(540, len(records) * 10)), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(heat)
    draw.text((8, 8), "target_feature cosine similarity; diagonal/self should be high, off-diagonal clusters reveal semantic but not spatial identity", fill=(255, 255, 255))
    heat.save(out_dir / "target_feature_similarity_heatmap.jpg", quality=95)

    diag_records = []
    panels = []
    labels = []
    for record in records[: args.limit]:
        stem = record["stem"]
        frame_idx = first_range_start(dataset_dir, stem)
        frame = load_frame(dataset_dir / "videos" / f"{stem}.mp4", frame_idx)
        mask = load_mask_frame(dataset_dir / "masks" / f"{stem}.npz", frame_idx)
        bbox, centroid = bbox_and_centroid(mask)
        anchor = draw_anchor(frame, mask, f"{stem} | {record['phrase']} | score={record['score']}")
        heat_img = centroid_heatmap(frame.size, centroid)
        panels.extend([frame, anchor, heat_img])
        labels.extend([f"{stem} RGB", "mask+bbox+centroid", "centroid emb prior"])
        diag_records.append(
            {
                "stem": stem,
                "phrase": record["phrase"],
                "score": record["score"],
                "frame_idx": frame_idx,
                "mask_pixels": int(mask.sum()),
                "mask_occupancy": float(mask.mean()),
                "bbox_xyxy": bbox,
                "centroid_xy": centroid,
            }
        )
    make_grid(panels, labels, ncols=3, tile_w=320).save(out_dir / "spatial_anchor_diagnostics.jpg", quality=95)

    nn_rows = []
    nn_panels = []
    nn_labels = []
    k = min(args.neighbors, len(records) - 1)
    for qi, record in enumerate(records[: min(args.limit, len(records))]):
        order = np.argsort(-sim[qi])
        order = [idx for idx in order if idx != qi][:k]
        nn_rows.append(
            {
                "query": record["stem"],
                "query_phrase": record["phrase"],
                "neighbors": [
                    {
                        "stem": records[idx]["stem"],
                        "phrase": records[idx]["phrase"],
                        "cosine": float(sim[qi, idx]),
                    }
                    for idx in order
                ],
            }
        )
        for idx in [qi, *order]:
            r = records[idx]
            frame_idx = first_range_start(dataset_dir, r["stem"])
            frame = load_frame(dataset_dir / "videos" / f"{r['stem']}.mp4", frame_idx)
            mask = load_mask_frame(dataset_dir / "masks" / f"{r['stem']}.npz", frame_idx)
            nn_panels.append(draw_anchor(frame, mask, f"{r['stem']} | {r['phrase']}"))
            prefix = "Q" if idx == qi else f"NN {float(sim[qi, idx]):.2f}"
            nn_labels.append(f"{prefix}: {r['phrase']}")
    make_grid(nn_panels, nn_labels, ncols=k + 1, tile_w=260).save(out_dir / "target_feature_nearest_neighbors.jpg", quality=95)

    summary = {
        "dataset_dir": str(dataset_dir),
        "num_records": len(records),
        "figures": {
            "spatial_anchor_diagnostics": str(out_dir / "spatial_anchor_diagnostics.jpg"),
            "target_feature_similarity_heatmap": str(out_dir / "target_feature_similarity_heatmap.jpg"),
            "target_feature_nearest_neighbors": str(out_dir / "target_feature_nearest_neighbors.jpg"),
        },
        "spatial_records": diag_records,
        "nearest_neighbors": nn_rows,
    }
    (out_dir / "embedding_diagnostics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary["figures"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
