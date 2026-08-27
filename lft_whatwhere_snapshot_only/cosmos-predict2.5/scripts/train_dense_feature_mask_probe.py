#!/usr/bin/env python3
"""Train a lightweight probe from InstructSAM dense features to target masks."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--feature-dir-name", default="target_features_instructsam_decoder_dense_stage2_lora")
    parser.add_argument("--mask-dir-name", default="masks")
    parser.add_argument("--video-dir-name", default="videos")
    parser.add_argument("--stems-file", default="selected_stems.txt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-train", type=int, default=800)
    parser.add_argument("--max-val", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-samples", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=0, help="0 means linear probe.")
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--mask-mode", choices=["first", "first_nonempty", "max"], default="first_nonempty")
    parser.add_argument("--normalize-features", action="store_true")
    parser.add_argument("--num-visuals", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_feature(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    feat = payload["target_feature"] if isinstance(payload, dict) else payload
    feat = torch.as_tensor(feat, dtype=torch.float32)
    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = feat[0]
    if feat.ndim != 2:
        raise ValueError(f"Expected [L,D] feature at {path}, got {tuple(feat.shape)}")
    return torch.nan_to_num(feat).contiguous()


def load_mask(path: Path, mode: str, side: int) -> torch.Tensor:
    data = np.load(path)
    if "masks_packed" in data.files and "shape" in data.files:
        shape = tuple(int(x) for x in np.asarray(data["shape"]).tolist())
        if len(shape) != 3:
            raise ValueError(f"Expected packed mask shape [T,H,W] at {path}, got {shape}")
        time, height, width = shape
        packed = np.asarray(data["masks_packed"])
        flat = np.unpackbits(packed, axis=-1, count=height * width)
        mask = flat.reshape(time, height, width)
        return choose_and_resize_mask(mask, mode, side)
    key = "masks" if "masks" in data.files else data.files[0]
    mask = np.asarray(data[key])
    mask = np.squeeze(mask)
    return choose_and_resize_mask(mask, mode, side)


def choose_and_resize_mask(mask: np.ndarray, mode: str, side: int) -> torch.Tensor:
    if mask.ndim == 2:
        chosen = mask
    elif mask.ndim == 3:
        if mode == "first":
            chosen = mask[0]
        elif mode == "max":
            chosen = mask.max(axis=0)
        else:
            areas = mask.reshape(mask.shape[0], -1).sum(axis=1)
            idx = int(np.argmax(areas > 0)) if np.any(areas > 0) else 0
            chosen = mask[idx]
    else:
        while mask.ndim > 3:
            mask = mask[0]
        return choose_and_resize_mask(mask, mode, side)
    return load_mask_array(chosen, mode, side)


def load_mask_array(mask: np.ndarray, _mode: str, side: int) -> torch.Tensor:
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    mask = cv2.resize(mask, (side, side), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(mask.reshape(-1).astype(np.float32))


def read_stems(dataset_dir: Path, stems_file: str, feature_dir: Path, mask_dir: Path, limit: int) -> list[str]:
    path = dataset_dir / stems_file
    if path.exists():
        stems = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    else:
        stems = [p.stem for p in sorted(feature_dir.glob("*.pt"))]
    valid = []
    for stem in stems:
        if (feature_dir / f"{stem}.pt").exists() and (mask_dir / f"{stem}.npz").exists():
            valid.append(stem)
        if len(valid) >= limit:
            break
    return valid


def build_model(dim: int, hidden_dim: int) -> nn.Module:
    if hidden_dim <= 0:
        return nn.Linear(dim, 1)
    return nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))


def load_batch(
    stems: list[str],
    feature_dir: Path,
    mask_dir: Path,
    mask_mode: str,
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    feats = []
    masks = []
    for stem in stems:
        feat = load_feature(feature_dir / f"{stem}.pt")
        side = int(round(math.sqrt(feat.shape[0])))
        if side * side != feat.shape[0]:
            raise ValueError(f"{stem}: feature token count is not square: {feat.shape[0]}")
        if normalize:
            feat = F.normalize(feat, dim=-1)
        mask = load_mask(mask_dir / f"{stem}.npz", mask_mode, side)
        if not bool(mask.any()) or bool((mask < 0.5).sum() == 0):
            continue
        feats.append(feat)
        masks.append(mask)
    if not feats:
        raise RuntimeError("Empty batch after filtering masks")
    return torch.cat(feats, dim=0), torch.cat(masks, dim=0)


def auc_score(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    labels = labels.float()
    pos = int(labels.sum().item())
    neg = int(labels.numel() - pos)
    if pos == 0 or neg == 0:
        return None
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=torch.float32)
    pos_rank_sum = ranks[labels > 0.5].sum()
    auc = (pos_rank_sum - pos * (pos + 1) / 2.0) / float(pos * neg)
    return float(auc.item())


def eval_one(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float | None]:
    scores = scores.detach().float().cpu()
    labels = labels.detach().float().cpu()
    mask = labels > 0.5
    if not bool(mask.any()) or not bool((~mask).any()):
        return {}
    inside = scores[mask]
    outside = scores[~mask]
    top10 = max(1, int(round(scores.numel() * 0.10)))
    top_area = max(1, int(mask.sum().item()))
    top10_inside = labels[torch.topk(scores, k=top10).indices].float().mean()
    top_area_inside = labels[torch.topk(scores, k=top_area).indices].float().mean()
    pred = torch.zeros_like(labels, dtype=torch.bool)
    pred[torch.topk(scores, k=top_area).indices] = True
    inter = (pred & mask).sum().float()
    union = (pred | mask).sum().float().clamp_min(1)
    return {
        "auc": auc_score(scores, labels),
        "inside_mean": float(inside.mean().item()),
        "outside_mean": float(outside.mean().item()),
        "inside_outside_ratio": float((inside.mean() + 1e-6) / (outside.mean() + 1e-6)),
        "inside_minus_outside": float((inside.mean() - outside.mean()).item()),
        "pointing_game": float(labels[int(torch.argmax(scores).item())].item()),
        "top10_inside_fraction": float(top10_inside.item()),
        "top_area_precision": float(top_area_inside.item()),
        "top_area_iou": float((inter / union).item()),
        "mask_area_fraction": float(labels.mean().item()),
    }


def aggregate(items: list[dict[str, float | None]]) -> dict[str, float]:
    out: dict[str, float] = {}
    keys = sorted({key for item in items for key in item})
    for key in keys:
        vals = [
            float(item[key])
            for item in items
            if item.get(key) is not None and isinstance(item.get(key), (int, float))
        ]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def read_first_frame(path: Path) -> Image.Image | None:
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def heat_image(values: torch.Tensor, side: int, size: tuple[int, int]) -> Image.Image:
    arr = values.reshape(side, side).detach().float().cpu().numpy()
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        arr = np.zeros_like(arr)
    else:
        arr = (arr - lo) / (hi - lo)
    color = cv2.applyColorMap((arr * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return Image.fromarray(color).resize(size, Image.Resampling.BILINEAR)


def mask_image(labels: torch.Tensor, side: int, size: tuple[int, int]) -> Image.Image:
    arr = labels.reshape(side, side).detach().cpu().numpy()
    rgb = np.stack([arr * 255, arr * 255, arr * 255], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb).resize(size, Image.Resampling.NEAREST)


def label(img: Image.Image, text: str) -> Image.Image:
    pad = 26
    out = Image.new("RGB", (img.width, img.height + pad), "white")
    out.paste(img.convert("RGB"), (0, pad))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = None
    draw.text((5, 5), text[:50], fill=(0, 0, 0), font=font)
    return out


def save_visuals(
    model: nn.Module,
    stems: list[str],
    feature_dir: Path,
    mask_dir: Path,
    video_dir: Path,
    out_dir: Path,
    mask_mode: str,
    normalize: bool,
    device: torch.device,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    size = (320, 180)
    for stem in stems:
        feat = load_feature(feature_dir / f"{stem}.pt")
        side = int(round(math.sqrt(feat.shape[0])))
        labels = load_mask(mask_dir / f"{stem}.npz", mask_mode, side)
        x = F.normalize(feat, dim=-1) if normalize else feat
        with torch.no_grad():
            scores = torch.sigmoid(model(x.to(device)).squeeze(-1)).cpu()
        frame = read_first_frame(video_dir / f"{stem}.mp4") or Image.new("RGB", size, "white")
        frame = frame.resize(size, Image.Resampling.BILINEAR)
        heat = heat_image(scores, side, size)
        mask = mask_image(labels, side, size)
        overlay = Image.blend(frame, heat, 0.45)
        panels = [label(frame, "first frame"), label(mask, "GT mask @32x32"), label(heat, "probe prob"), label(overlay, "overlay")]
        canvas = Image.new("RGB", (size[0] * 2, (size[1] + 26) * 2), "white")
        for idx, panel in enumerate(panels):
            canvas.paste(panel, ((idx % 2) * size[0], (idx // 2) * (size[1] + 26)))
        canvas.save(out_dir / f"{stem}_probe.jpg", quality=95)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    feature_dir = args.dataset_dir / args.feature_dir_name
    mask_dir = args.dataset_dir / args.mask_dir_name
    video_dir = args.dataset_dir / args.video_dir_name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stems = read_stems(args.dataset_dir, args.stems_file, feature_dir, mask_dir, args.max_train + args.max_val)
    random.shuffle(stems)
    train_stems = stems[: args.max_train]
    val_stems = stems[args.max_train : args.max_train + args.max_val]
    if not train_stems or not val_stems:
        raise RuntimeError(f"Not enough stems: train={len(train_stems)} val={len(val_stems)}")

    first_feat = load_feature(feature_dir / f"{train_stems[0]}.pt")
    dim = first_feat.shape[-1]
    model = build_model(dim, args.hidden_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pos_count = 0.0
    total_count = 0.0
    for start in range(0, len(train_stems), args.batch_samples):
        _, y = load_batch(train_stems[start : start + args.batch_samples], feature_dir, mask_dir, args.mask_mode, False)
        pos_count += float(y.sum().item())
        total_count += float(y.numel())
    neg_count = max(1.0, total_count - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], device=args.device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_stems)
        model.train()
        losses = []
        for start in range(0, len(train_stems), args.batch_samples):
            batch = train_stems[start : start + args.batch_samples]
            x_cpu, y_cpu = load_batch(batch, feature_dir, mask_dir, args.mask_mode, args.normalize_features)
            x = x_cpu.to(args.device)
            y = y_cpu.to(args.device)
            logits = model(x).squeeze(-1)
            loss = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
        print(json.dumps(history[-1]), flush=True)

    model.eval()
    val_metrics = []
    with torch.no_grad():
        for stem in val_stems:
            feat = load_feature(feature_dir / f"{stem}.pt")
            side = int(round(math.sqrt(feat.shape[0])))
            labels = load_mask(mask_dir / f"{stem}.npz", args.mask_mode, side)
            if not bool(labels.any()) or not bool((labels < 0.5).any()):
                continue
            x = F.normalize(feat, dim=-1) if args.normalize_features else feat
            scores = torch.sigmoid(model(x.to(args.device)).squeeze(-1)).cpu()
            item = eval_one(scores, labels)
            item["stem"] = stem
            val_metrics.append(item)

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "feature_dir_name": args.feature_dir_name,
        "mask_dir_name": args.mask_dir_name,
        "num_train": len(train_stems),
        "num_val": len(val_metrics),
        "hidden_dim": args.hidden_dim,
        "normalize_features": args.normalize_features,
        "pos_fraction_train": pos_count / max(total_count, 1.0),
        "pos_weight": float(pos_weight.item()),
        "history": history,
        "val_mean": aggregate(val_metrics),
        "val_samples": val_metrics,
    }
    summary_path = args.output_dir / "dense_feature_mask_probe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    torch.save({"model": model.state_dict(), "args": vars(args), "summary": summary}, args.output_dir / "dense_feature_mask_probe.pt")
    save_visuals(
        model,
        val_stems[: args.num_visuals],
        feature_dir,
        mask_dir,
        video_dir,
        args.output_dir / "visuals",
        args.mask_mode,
        args.normalize_features,
        torch.device(args.device),
    )
    print(json.dumps({"summary": str(summary_path), "val_mean": summary["val_mean"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
