#!/usr/bin/env python3
"""Precompute oracle target features from GT masks and InstructSAM/SAM3 image features."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

try:
    import huggingface_hub

    if not hasattr(huggingface_hub, "is_offline_mode"):
        huggingface_hub.is_offline_mode = lambda: os.environ.get("HF_HUB_OFFLINE", "0") == "1"
except Exception:
    pass

from transformers import AutoProcessor


def _rank_info() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    return rank, local_rank, world_size


def _default_source_root() -> Path:
    if os.environ.get("INSTRUCTSAM_SOURCE_ROOT"):
        return Path(os.environ["INSTRUCTSAM_SOURCE_ROOT"])
    return Path(__file__).resolve().parents[2] / "InstructSAM"


def _default_model_path() -> Path:
    if os.environ.get("INSTRUCTSAM_MODEL_PATH"):
        return Path(os.environ["INSTRUCTSAM_MODEL_PATH"])
    return _default_source_root() / "work_dirs" / "InstructSAM-2B"


def torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def load_excluded_stems(dataset_dir: Path, exclude_file: str) -> set[str]:
    if exclude_file.lower() == "none":
        return set()
    path = dataset_dir / "exclude_no_tgt_stems.txt" if exclude_file == "auto" else Path(exclude_file)
    if not path.exists():
        return set()
    return set(path.read_text().split())


def iter_videos(dataset_dir: Path, exclude_file: str) -> list[Path]:
    videos_dir = dataset_dir / "videos"
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Missing videos directory: {videos_dir}")
    excluded = load_excluded_stems(dataset_dir, exclude_file)
    videos = sorted(path for path in videos_dir.glob("*.mp4") if path.stem not in excluded)
    if not videos:
        raise RuntimeError(f"No active mp4 videos found in {videos_dir}")
    return videos


def load_caption(dataset_dir: Path, stem: str) -> str | None:
    text_path = dataset_dir / "metas" / f"{stem}.txt"
    if text_path.exists():
        return text_path.read_text().strip()
    json_path = dataset_dir / "captions" / f"{stem}.json"
    if json_path.exists():
        data = json.loads(json_path.read_text())
        first_model_value = next(iter(data.values()))
        if isinstance(first_model_value, dict):
            return str(next(iter(first_model_value.values()))).strip()
        return str(first_model_value).strip()
    return None


def load_mask_array(mask_path: Path) -> np.ndarray:
    with np.load(mask_path, allow_pickle=True) as npz:
        if "masks" in npz.files:
            arr = npz["masks"]
        elif "masks_packed" in npz.files and "shape" in npz.files:
            shape = tuple(int(dim) for dim in npz["shape"].tolist())
            flat_pixels = int(np.prod(shape[1:]))
            arr = np.unpackbits(npz["masks_packed"], axis=1)[:, :flat_pixels].reshape(shape)
        else:
            arr = npz[npz.files[0]]
    arr = np.asarray(arr)
    if arr.ndim == 5:
        arr = arr.max(axis=0)
    if arr.ndim == 4:
        if arr.shape[1] == 1:
            arr = arr[:, 0]
        elif arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr.max(axis=0)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported mask shape {arr.shape} in {mask_path}")
    return arr.astype(np.float32)


def choose_mask_frame(mask_video: np.ndarray, policy: str) -> int:
    if policy == "first":
        return 0
    if policy == "middle":
        return int(mask_video.shape[0] // 2)
    if policy == "first_nonempty":
        areas = mask_video.reshape(mask_video.shape[0], -1).sum(axis=1)
        nonempty = np.flatnonzero(areas > 0)
        return int(nonempty[0]) if len(nonempty) else 0
    raise ValueError(f"Unsupported mask-frame-policy: {policy}")


def read_video_frame(video_path: Path, frame_idx: int) -> Image.Image:
    from decord import VideoReader, cpu

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    frame_idx = max(0, min(int(frame_idx), len(vr) - 1))
    frame = vr.get_batch([frame_idx]).asnumpy()[0]
    return Image.fromarray(frame).convert("RGB")


def prepare_sam_pixel_values(pixel_values: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(pixel_values, torch.Tensor):
        tensor = pixel_values
    elif isinstance(pixel_values, np.ndarray):
        tensor = torch.from_numpy(pixel_values)
    elif isinstance(pixel_values, (list, tuple)):
        tensors = [
            item if isinstance(item, torch.Tensor) else torch.as_tensor(item)
            for item in pixel_values
        ]
        if not tensors:
            raise ValueError("SAM processor returned empty pixel_values")
        if len(tensors) == 1:
            tensor = tensors[0]
        elif all(item.ndim == tensors[0].ndim for item in tensors):
            tensor = torch.cat(tensors, dim=0) if tensors[0].ndim == 4 else torch.stack(tensors, dim=0)
        else:
            raise ValueError(f"Unsupported mixed SAM pixel_values shapes: {[tuple(item.shape) for item in tensors]}")
    else:
        raise TypeError(f"Unsupported SAM pixel_values type: {type(pixel_values)}")

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Unsupported SAM pixel_values shape: {tuple(tensor.shape)}")
    return tensor.to(device=device, dtype=dtype)


def extract_vision_tensor(vision_outputs: Any) -> torch.Tensor:
    if isinstance(vision_outputs, dict):
        candidates = [vision_outputs.get("last_hidden_state"), vision_outputs.get("pooler_output")]
    else:
        candidates = [
            getattr(vision_outputs, "last_hidden_state", None),
            getattr(vision_outputs, "pooler_output", None),
        ]
        if isinstance(vision_outputs, (tuple, list)):
            candidates.extend(vision_outputs)
    for candidate in candidates:
        if isinstance(candidate, torch.Tensor):
            return candidate.detach().float().cpu()
        if isinstance(candidate, (tuple, list)):
            for item in candidate:
                if isinstance(item, torch.Tensor):
                    return item.detach().float().cpu()
    raise ValueError(f"Could not find tensor in vision outputs of type {type(vision_outputs)}")


def vision_tensor_to_grid(features: torch.Tensor) -> tuple[torch.Tensor, int | None, int | None]:
    """Return features as [N,C] plus optional spatial H,W."""
    if features.ndim >= 1 and features.shape[0] == 1:
        features = features[0]
    if features.ndim == 4:
        if features.shape[0] == 1:
            features = features[0]
        if features.shape[0] >= features.shape[-1]:
            # [C,H,W]
            c, h, w = features.shape
            return features.permute(1, 2, 0).reshape(h * w, c).contiguous(), h, w
        # [H,W,C]
        h, w, c = features.shape
        return features.reshape(h * w, c).contiguous(), h, w
    if features.ndim == 3:
        if features.shape[0] == 1:
            features = features[0]
        if features.ndim == 3:
            h, w, c = features.shape
            return features.reshape(h * w, c).contiguous(), h, w
    if features.ndim == 2:
        n, _ = features.shape
        side = int(round(math.sqrt(n)))
        if side * side == n:
            return features.contiguous(), side, side
        return features.contiguous(), None, None
    raise ValueError(f"Unsupported vision feature shape {tuple(features.shape)}")


def deterministic_project(tokens: torch.Tensor, out_dim: int, seed: int) -> torch.Tensor:
    tokens = torch.nan_to_num(tokens.float())
    tokens = F.layer_norm(tokens, (tokens.shape[-1],))
    if tokens.shape[-1] == out_dim:
        return tokens.contiguous()
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + tokens.shape[-1] * 1009 + out_dim * 9173)
    weight = torch.randn(tokens.shape[-1], out_dim, generator=generator, dtype=torch.float32)
    weight = weight / math.sqrt(float(tokens.shape[-1]))
    projected = tokens @ weight
    return F.layer_norm(projected, (out_dim,)).contiguous()


def build_gt_mask_tokens(
    grid_features_N_C: torch.Tensor,
    grid_h: int | None,
    grid_w: int | None,
    mask_2d: np.ndarray,
    *,
    out_dim: int,
    max_tokens: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if grid_h is None or grid_w is None:
        pooled = grid_features_N_C.mean(dim=0, keepdim=True)
        tokens = pooled.repeat(max(1, int(max_tokens)), 1)
        return deterministic_project(tokens, out_dim, seed), {"mask_grid_shape": None, "mask_grid_area": None}

    mask = torch.from_numpy(mask_2d.astype(np.float32))[None, None]
    mask = F.interpolate(mask, size=(grid_h, grid_w), mode="nearest")[0, 0].reshape(-1)
    mask = (mask > 0.5).float()
    area = float(mask.sum().item())
    token_count = max(1, int(max_tokens))
    if area <= 0:
        pooled = torch.zeros(1, grid_features_N_C.shape[-1], dtype=torch.float32)
        tokens = pooled.repeat(token_count, 1)
        return deterministic_project(tokens, out_dim, seed), {"mask_grid_shape": [grid_h, grid_w], "mask_grid_area": 0.0}

    fg_indices = torch.nonzero(mask > 0, as_tuple=False).flatten()
    if fg_indices.numel() > token_count:
        sample_positions = torch.linspace(0, fg_indices.numel() - 1, token_count).round().long()
        fg_indices = fg_indices[sample_positions]
    selected = grid_features_N_C[fg_indices]
    y = (fg_indices // grid_w).float()
    x = (fg_indices % grid_w).float()
    y_norm = (y / max(grid_h - 1, 1)) * 2.0 - 1.0
    x_norm = (x / max(grid_w - 1, 1)) * 2.0 - 1.0
    cy = y_norm.mean()
    cx = x_norm.mean()
    coord_features = torch.stack([y_norm, x_norm, y_norm - cy, x_norm - cx], dim=1)
    selected = torch.cat([selected, coord_features], dim=1)
    if selected.shape[0] < token_count:
        pad = torch.zeros(token_count - selected.shape[0], selected.shape[1], dtype=selected.dtype)
        selected = torch.cat([selected, pad], dim=0)
    return deterministic_project(selected, out_dim, seed), {
        "mask_grid_shape": [grid_h, grid_w],
        "mask_grid_area": area,
        "num_fg_patch_tokens": int(min(area, token_count)),
        "max_tokens": token_count,
    }


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", action="append", required=True)
    parser.add_argument("--model-path", type=Path, default=_default_model_path())
    parser.add_argument("--source-root", type=Path, default=_default_source_root())
    parser.add_argument("--mask-dir-name", default="masks")
    parser.add_argument("--output-dir-name", default="target_features_gt_mask_spatial64")
    parser.add_argument("--exclude-video-stems-file", default="auto")
    parser.add_argument("--mask-frame-policy", choices=["first", "first_nonempty", "middle"], default="first")
    parser.add_argument("--expected-feature-dim", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--projection-seed", type=int, default=20260606)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rank, local_rank, world_size = _rank_info()
    all_items: list[tuple[Path, Path, Path, Path]] = []
    for dataset_dir_str in args.dataset_dir:
        dataset_dir = Path(dataset_dir_str)
        output_dir = dataset_dir / args.output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for video_path in iter_videos(dataset_dir, args.exclude_video_stems_file):
            mask_path = dataset_dir / args.mask_dir_name / f"{video_path.stem}.npz"
            all_items.append((dataset_dir, output_dir, video_path, mask_path))
    if args.limit > 0:
        all_items = all_items[: args.limit]
    shard_items = [item for idx, item in enumerate(all_items) if idx % world_size == rank]
    print(
        f"rank={rank} local_rank={local_rank} world_size={world_size} total_items={len(all_items)} "
        f"shard_items={len(shard_items)} output_dir_name={args.output_dir_name}",
        flush=True,
    )
    if args.dry_run:
        for dataset_dir, output_dir, video_path, mask_path in shard_items[:10]:
            print(f"DRYRUN {video_path} + {mask_path} -> {output_dir / (video_path.stem + '.pt')}")
        return 0

    if not args.source_root.exists():
        raise FileNotFoundError(f"InstructSAM source root does not exist: {args.source_root}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"InstructSAM model path does not exist: {args.model_path}")

    if str(args.source_root) not in sys.path:
        sys.path.insert(0, str(args.source_root))
    from instructsam.models import load_pretrained_model

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        device_map: str | dict[str, str] = {"": f"cuda:{local_rank}"}
    else:
        device = torch.device("cpu")
        device_map = "cpu"

    tokenizer, model, _processor = load_pretrained_model(
        str(args.model_path),
        None,
        device_map=device_map,
        attn_implementation=args.attn_implementation,
    )
    del tokenizer, _processor
    dtype = torch_dtype_from_name(args.torch_dtype)
    model.to(dtype=dtype)
    model.eval()
    seg_processor = AutoProcessor.from_pretrained(model.config.mask_decoder_model)

    processed = 0
    skipped = 0
    errors = 0
    start = time.time()
    for dataset_dir, output_dir, video_path, mask_path in shard_items:
        output_path = output_dir / f"{video_path.stem}.pt"
        summary_path = output_dir / f"precompute_gt_rank{rank:03d}.jsonl"
        if output_path.exists() and args.skip_existing and not args.overwrite:
            skipped += 1
            continue
        try:
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing GT mask: {mask_path}")
            mask_video = load_mask_array(mask_path)
            frame_idx = choose_mask_frame(mask_video, args.mask_frame_policy)
            frame_idx = min(frame_idx, mask_video.shape[0] - 1)
            image = read_video_frame(video_path, frame_idx)
            image = ImageOps.exif_transpose(image).convert("RGB")
            sam_inputs = seg_processor(image)
            pixel_values = prepare_sam_pixel_values(sam_inputs["pixel_values"], device=device, dtype=dtype)
            with torch.inference_mode():
                vision_outputs = model.model.grounding_model.encoder(pixel_values)
            vision_tensor = extract_vision_tensor(vision_outputs)
            grid_features, grid_h, grid_w = vision_tensor_to_grid(vision_tensor)
            target_feature, meta = build_gt_mask_tokens(
                grid_features,
                grid_h,
                grid_w,
                mask_video[frame_idx],
                out_dim=args.expected_feature_dim,
                max_tokens=args.max_tokens,
                seed=args.projection_seed,
            )
            payload = {
                "target_feature": target_feature,
                "feature_mode": "gt_mask_sam3_spatial_tokens",
                "source": "gt_mask",
                "video": str(video_path),
                "mask_path": str(mask_path),
                "mask_frame_idx": frame_idx,
                "caption": load_caption(dataset_dir, video_path.stem),
                "vision_feature_shape": list(vision_tensor.shape),
                "projection_seed": args.projection_seed,
                **meta,
            }
            tmp_path = output_path.with_suffix(output_path.suffix + f".rank{rank}.tmp")
            torch.save(payload, tmp_path)
            os.replace(tmp_path, output_path)
            write_jsonl(
                summary_path,
                {
                    "status": "ok",
                    "stem": video_path.stem,
                    "feature_shape": list(target_feature.shape),
                    "mask_frame_idx": frame_idx,
                    **meta,
                },
            )
            processed += 1
        except Exception as exc:
            errors += 1
            write_jsonl(
                summary_path,
                {
                    "status": "error",
                    "stem": video_path.stem,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"[rank {rank}] ERROR {video_path}: {exc}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            return 1
        if processed and processed % args.log_every == 0:
            elapsed = time.time() - start
            print(
                f"[rank {rank}] processed={processed} skipped={skipped} errors={errors} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    print(f"rank={rank} done processed={processed} skipped={skipped} errors={errors}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
