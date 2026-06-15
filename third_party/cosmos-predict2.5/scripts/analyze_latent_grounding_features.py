#!/usr/bin/env python3
"""Explainability probes for mask-free InstructSAM latent grounding features.

This script is an offline diagnostic. It may compare feature-derived saliency to
dataset masks, but those masks are never fed into Cosmos. The purpose is to
verify whether the dense hidden feature already contains target-location
information and whether a latent-space injection is plausible.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def _load_feature(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    meta: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                meta[key] = value
        for key in ("target_feature", "feature", "features", "decoder_dense", "hidden_states"):
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                feature = value
                break
        else:
            tensors = [(key, value) for key, value in payload.items() if isinstance(value, torch.Tensor)]
            if not tensors:
                raise ValueError(f"No tensor feature found in {path}")
            key, feature = tensors[0]
            meta["selected_tensor_key"] = key
    elif isinstance(payload, torch.Tensor):
        feature = payload
    else:
        raise ValueError(f"Unsupported feature payload type {type(payload)} in {path}")

    if feature.ndim == 3 and feature.shape[0] == 1:
        feature = feature[0]
    if feature.ndim != 2:
        raise ValueError(f"Expected feature [L,D], got {tuple(feature.shape)} in {path}")
    return feature.float(), meta


def _load_mask(path: Path, mode: str) -> np.ndarray | None:
    if not path.exists():
        return None
    payload = np.load(path)
    if "masks_packed" in payload and "shape" in payload:
        shape = tuple(int(v) for v in payload["shape"].tolist())
        packed = payload["masks_packed"]
        if mode == "max":
            flat = np.unpackbits(packed, axis=-1)[:, : shape[1] * shape[2]]
            masks = flat.reshape(shape).astype(np.float32)
        else:
            if mode == "first_nonempty":
                nonempty = np.flatnonzero(packed.any(axis=1))
                frame_idx = int(nonempty[0]) if len(nonempty) else 0
            else:
                frame_idx = 0
            flat = np.unpackbits(packed[frame_idx], axis=-1)[: shape[1] * shape[2]]
            return flat.reshape(shape[1], shape[2]).astype(np.float32)
    else:
        key = "mask" if "mask" in payload.files else payload.files[0]
        masks = payload[key].astype(np.float32)
        if masks.ndim == 2:
            masks = masks[None]
    if masks.ndim != 3:
        raise ValueError(f"Expected mask [T,H,W], got {masks.shape} in {path}")
    masks = (masks > 0).astype(np.float32)
    if mode == "max":
        return masks.max(axis=0)
    if mode == "first":
        return masks[0]
    if mode != "first_nonempty":
        raise ValueError(f"Unknown mask mode: {mode}")
    sums = masks.reshape(masks.shape[0], -1).sum(axis=1)
    idx = int(np.argmax(sums > 0)) if bool((sums > 0).any()) else 0
    return masks[idx]


def _read_first_frame(path: Path) -> Image.Image | None:
    if not path.exists():
        return None
    try:
        import imageio.v2 as imageio

        reader = imageio.get_reader(str(path))
        frame = reader.get_data(0)
        reader.close()
        return Image.fromarray(frame).convert("RGB")
    except Exception:
        pass
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame).convert("RGB")
    except Exception:
        pass
    return None


def _normalize_map(values: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(values.float())
    lo = torch.quantile(values.flatten(), 0.01)
    hi = torch.quantile(values.flatten(), 0.99)
    return ((values - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)


def _feature_maps(feature: torch.Tensor) -> dict[str, torch.Tensor]:
    num_tokens, dim = feature.shape
    side = int(round(math.sqrt(num_tokens)))
    if side * side != num_tokens:
        raise ValueError(f"Feature token count {num_tokens} is not a square; cannot make dense map")

    feat = torch.nan_to_num(feature.float()).reshape(side, side, dim)
    feat_flat = feat.reshape(-1, dim)
    feat_norm = F.normalize(feat_flat, dim=-1)
    centered = feat_norm - feat_norm.mean(dim=0, keepdim=True)

    saliency_norm = feature.float().norm(dim=-1).reshape(side, side)
    saliency_centered = centered.norm(dim=-1).reshape(side, side)

    try:
        covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
        _, eigvecs = torch.linalg.eigh(covariance)
        pc = centered @ eigvecs[:, -1]
        saliency_pca_abs = pc.abs().reshape(side, side)
    except Exception:
        saliency_pca_abs = saliency_centered

    return {
        "feature_norm": _normalize_map(saliency_norm),
        "centered_energy": _normalize_map(saliency_centered),
        "pca1_abs": _normalize_map(saliency_pca_abs),
    }


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    tensor = torch.from_numpy(mask).float()[None, None]
    resized = F.interpolate(tensor, size=size, mode="area")[0, 0]
    return resized.clamp(0, 1)


def _auc_score(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    scores = scores.flatten().float()
    labels = (labels.flatten().float() > 0.5)
    pos = int(labels.sum().item())
    neg = int((~labels).sum().item())
    if pos == 0 or neg == 0:
        return None
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    pos_rank_sum = ranks[labels].sum()
    auc = (pos_rank_sum - pos * (pos + 1) / 2.0) / float(pos * neg)
    return float(auc.item())


def _map_metrics(scores: torch.Tensor, mask: torch.Tensor) -> dict[str, float | None]:
    mask_bin = mask > 0.5
    if not bool(mask_bin.any()):
        return {
            "auc": None,
            "inside_mean": None,
            "outside_mean": None,
            "inside_outside_ratio": None,
            "pointing_game": None,
            "top10_inside_fraction": None,
            "mask_area_fraction": 0.0,
        }
    inside = scores[mask_bin]
    outside = scores[~mask_bin]
    topk = max(1, int(round(scores.numel() * 0.1)))
    top_idx = torch.topk(scores.flatten(), k=topk).indices
    top_mask = mask_bin.flatten()[top_idx]
    max_idx = int(torch.argmax(scores.flatten()).item())
    inside_mean = float(inside.mean().item())
    outside_mean = float(outside.mean().item()) if outside.numel() else 0.0
    return {
        "auc": _auc_score(scores, mask_bin.float()),
        "inside_mean": inside_mean,
        "outside_mean": outside_mean,
        "inside_outside_ratio": inside_mean / max(outside_mean, 1e-6),
        "pointing_game": float(mask_bin.flatten()[max_idx].item()),
        "top10_inside_fraction": float(top_mask.float().mean().item()),
        "mask_area_fraction": float(mask_bin.float().mean().item()),
    }


def _oracle_linear_probe_map(feature: torch.Tensor, mask_grid: torch.Tensor) -> torch.Tensor | None:
    num_tokens, dim = feature.shape
    side = int(round(math.sqrt(num_tokens)))
    mask = mask_grid.flatten() > 0.5
    if side * side != num_tokens or not bool(mask.any()) or not bool((~mask).any()):
        return None
    feat = F.normalize(torch.nan_to_num(feature.float()), dim=-1)
    pos = feat[mask].mean(dim=0)
    neg = feat[~mask].mean(dim=0)
    direction = F.normalize(pos - neg, dim=0)
    scores = (feat @ direction).reshape(side, side)
    return _normalize_map(scores)


def _heatmap_image(values: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    values_np = (values.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    red = values_np
    green = np.clip(values_np.astype(np.int16) * 2 - 96, 0, 255).astype(np.uint8)
    blue = (255 - values_np).astype(np.uint8)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR)


def _mask_image(mask: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    arr = (mask.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(rgb).resize(size, Image.Resampling.NEAREST)


def _overlay(base: Image.Image, heat: Image.Image, alpha: float = 0.45) -> Image.Image:
    return Image.blend(base.convert("RGB"), heat.convert("RGB"), alpha)


def _label(img: Image.Image, text: str) -> Image.Image:
    pad = 28
    out = Image.new("RGB", (img.width, img.height + pad), "white")
    out.paste(img, (0, pad))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = None
    draw.text((6, 5), text, fill=(0, 0, 0), font=font)
    return out


def _make_sheet(
    frame: Image.Image | None,
    maps: dict[str, torch.Tensor],
    mask_grid: torch.Tensor | None,
    out_path: Path,
) -> None:
    size = (320, 180)
    base = frame.resize(size, Image.Resampling.BILINEAR) if frame is not None else Image.new("RGB", size, "white")
    panels = [_label(base, "first frame")]
    if mask_grid is not None:
        panels.append(_label(_mask_image(mask_grid, size), "eval mask probe"))
    for name, values in maps.items():
        heat = _heatmap_image(values, size)
        panels.append(_label(heat, name))
        panels.append(_label(_overlay(base, heat), f"{name} overlay"))
    cols = 3
    rows = int(math.ceil(len(panels) / cols))
    sheet = Image.new("RGB", (cols * size[0], rows * (size[1] + 28)), "white")
    for idx, panel in enumerate(panels):
        x = (idx % cols) * size[0]
        y = (idx // cols) * (size[1] + 28)
        sheet.paste(panel, (x, y))
    sheet.save(out_path)


def analyze_one(stem: str, args: argparse.Namespace) -> dict[str, Any]:
    feature_path = args.dataset_dir / args.feature_dir_name / f"{stem}.pt"
    mask_path = args.dataset_dir / args.mask_dir_name / f"{stem}.npz"
    video_path = args.dataset_dir / args.video_dir_name / f"{stem}.mp4"
    feature, meta = _load_feature(feature_path)
    maps = _feature_maps(feature)
    side = next(iter(maps.values())).shape[-1]

    mask = _load_mask(mask_path, args.mask_mode)
    mask_grid = _resize_mask(mask, (side, side)) if mask is not None else None
    if mask_grid is not None:
        oracle = _oracle_linear_probe_map(feature, mask_grid)
        if oracle is not None:
            maps["oracle_linear_probe"] = oracle

    sample_metrics: dict[str, Any] = {
        "stem": stem,
        "feature_shape": list(feature.shape),
        "feature_side": side,
        "meta": meta,
        "maps": {},
    }
    for name, values in maps.items():
        sample_metrics["maps"][name] = (
            _map_metrics(values, mask_grid) if mask_grid is not None else {"auc": None}
        )

    if not args.metrics_only:
        frame = None if args.no_video else _read_first_frame(video_path)
        _make_sheet(frame, maps, mask_grid, args.output_dir / f"{stem}_latent_grounding_probe.jpg")
    return sample_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--feature-dir-name", default="target_features_instructsam_decoder_dense_stage2_lora")
    parser.add_argument("--mask-dir-name", default="masks")
    parser.add_argument("--video-dir-name", default="videos")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--mask-mode", choices=["first", "first_nonempty", "max"], default="first_nonempty")
    parser.add_argument("--no-video", action="store_true", help="Skip first-frame decoding and render heatmaps only.")
    parser.add_argument("--metrics-only", action="store_true", help="Skip image sheets and only write JSON metrics.")
    args = parser.parse_args()

    feature_dir = args.dataset_dir / args.feature_dir_name
    if not feature_dir.exists():
        raise FileNotFoundError(feature_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stems = [p.stem for p in sorted(feature_dir.glob("*.pt"))[: args.max_samples]]
    if not stems:
        raise RuntimeError(f"No .pt feature files found in {feature_dir}")

    results = []
    for stem in stems:
        try:
            results.append(analyze_one(stem, args))
            print(f"[ok] {stem}")
        except Exception as exc:
            print(f"[skip] {stem}: {exc}")

    aggregate: dict[str, dict[str, float]] = {}
    for result in results:
        for name, metrics in result["maps"].items():
            bucket = aggregate.setdefault(name, {})
            for key, value in metrics.items():
                if value is None:
                    continue
                bucket.setdefault(key, []).append(float(value))
    aggregate_mean = {
        name: {key: float(np.mean(values)) for key, values in metrics.items()}
        for name, metrics in aggregate.items()
    }
    summary = {"num_samples": len(results), "aggregate_mean": aggregate_mean, "samples": results}
    (args.output_dir / "latent_grounding_feature_probe_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(json.dumps({"num_samples": len(results), "aggregate_mean": aggregate_mean}, indent=2))


if __name__ == "__main__":
    main()
