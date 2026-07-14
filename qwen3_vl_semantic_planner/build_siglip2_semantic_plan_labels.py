#!/usr/bin/env python3
"""Precompute Baton-style continuous semantic blueprints with SigLIP2.

Future RGB keyframes are encoded with a frozen SigLIP2 vision tower.  The
penultimate spatial patch features are saved as continuous semantic plan labels
for Stage-1 planner training.  By default the patch features are pooled to a
fixed grid; pass ``--grid-size 0`` to keep the native SigLIP2 spatial patch
grid without pooling.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-ranges", type=Path, default=None)
    parser.add_argument("--num-keyframes", type=int, default=6)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=9,
        help="Output spatial grid side. Use <=0 to keep native SigLIP2 patch tokens.",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--expand-ranges", action="store_true")
    parser.add_argument("--window-length", type=int, default=0)
    parser.add_argument("--window-stride", type=int, default=0)
    parser.add_argument("--min-window-frames", type=int, default=2)
    parser.add_argument("--windows-per-stem", type=int, default=0)
    parser.add_argument("--window-seed", type=int, default=20260629)
    parser.add_argument(
        "--window-manifest",
        type=Path,
        default=None,
        help="Optional external window manifest with exact video_frame_indices.",
    )
    parser.add_argument("--cosmos-stride-windows", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--frame-strides", default="")
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--manifest-name", default="manifest.jsonl")
    parser.add_argument("--progress-name", default="progress.jsonl")
    parser.add_argument("--summary-name", default="summary.json")
    return parser.parse_args()


@dataclass(frozen=True)
class WindowItem:
    sample_id: str
    stem: str
    start: int
    end: int
    range_index: int
    window_index: int
    frame_stride: int = 0
    sequence_length: int = 0
    video_frame_indices: tuple[int, ...] | None = None


def torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def load_frame_ranges(path: Path) -> dict[str, list[tuple[int, int]]]:
    data = json.loads(path.read_text())
    items: dict[str, list[tuple[int, int]]] = {}
    if isinstance(data, dict):
        iterable = data.items()
    elif isinstance(data, list):
        iterable = ((str(x.get("stem") or x.get("video_id") or x.get("id")), x) for x in data)
    else:
        raise TypeError(f"Unsupported frame_ranges type: {type(data)!r}")

    for stem, ranges in iterable:
        if isinstance(ranges, dict):
            start = int(ranges.get("start", ranges.get("frame_start", 0)))
            end = int(ranges.get("end", ranges.get("frame_end", start + 49)))
            parsed_ranges = [(start, end)]
        elif isinstance(ranges, list) and ranges and isinstance(ranges[0], (list, tuple)):
            parsed_ranges = [(int(x[0]), int(x[1])) for x in ranges if len(x) >= 2]
        elif isinstance(ranges, list) and len(ranges) >= 2:
            start, end = int(ranges[0]), int(ranges[1])
            parsed_ranges = [(start, end)]
        else:
            continue
        valid = [(max(0, start), max(0, end)) for start, end in parsed_ranges if end > start]
        if valid:
            items[stem] = valid
    return items


def parse_int_list(text: str) -> list[int]:
    out = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            value = int(item)
            if value <= 0:
                raise ValueError(f"Expected positive stride, got {value}")
            out.append(value)
    return out


def make_sample_id(
    stem: str,
    range_index: int,
    window_index: int,
    start: int,
    end: int,
    expanded: bool,
    frame_stride: int = 0,
) -> str:
    if not expanded and range_index == 0 and window_index == 0:
        return stem
    stride_part = f"__fs{frame_stride:02d}" if frame_stride > 0 else ""
    return f"{stem}__r{range_index:02d}__w{window_index:04d}{stride_part}__s{start:06d}_e{end:06d}"


def expand_frame_ranges(
    ranges_by_stem: dict[str, list[tuple[int, int]]],
    *,
    expand_ranges: bool,
    window_length: int,
    window_stride: int,
    min_window_frames: int,
    cosmos_stride_windows: bool = False,
    sequence_length: int = 0,
    frame_strides: list[int] | None = None,
) -> list[WindowItem]:
    items: list[WindowItem] = []
    frame_strides = frame_strides or []
    for stem, ranges in ranges_by_stem.items():
        source_ranges = ranges if expand_ranges else ranges[:1]
        for range_index, (range_start, range_end) in enumerate(source_ranges):
            range_start = max(0, int(range_start))
            range_end = max(range_start, int(range_end))
            if range_end - range_start < min_window_frames:
                continue
            if cosmos_stride_windows:
                if sequence_length <= 1:
                    raise ValueError("--sequence-length must be > 1 with --cosmos-stride-windows")
                if not frame_strides:
                    raise ValueError("--frame-strides must be set with --cosmos-stride-windows")
                window_index = 0
                for frame_stride in frame_strides:
                    span = frame_stride * (sequence_length - 1) + 1
                    if range_end - range_start < span:
                        continue
                    step = window_stride if window_stride > 0 else span
                    last_start = range_end - span
                    starts = list(range(range_start, last_start + 1, step))
                    if not starts or starts[-1] != last_start:
                        starts.append(last_start)
                    for start in starts:
                        end = start + span
                        items.append(
                            WindowItem(
                                sample_id=make_sample_id(
                                    stem,
                                    range_index,
                                    window_index,
                                    start,
                                    end,
                                    expanded=True,
                                    frame_stride=frame_stride,
                                ),
                                stem=stem,
                                start=start,
                                end=end,
                                range_index=range_index,
                                window_index=window_index,
                                frame_stride=frame_stride,
                                sequence_length=sequence_length,
                            )
                        )
                        window_index += 1
            elif expand_ranges and window_length > 0 and range_end - range_start > window_length:
                stride = window_stride if window_stride > 0 else window_length
                last_start = range_end - window_length
                starts = list(range(range_start, last_start + 1, stride))
                if not starts or starts[-1] != last_start:
                    starts.append(last_start)
                for window_index, start in enumerate(starts):
                    end = start + window_length
                    items.append(
                        WindowItem(
                            sample_id=make_sample_id(stem, range_index, window_index, start, end, expanded=True),
                            stem=stem,
                            start=start,
                            end=end,
                            range_index=range_index,
                            window_index=window_index,
                        )
                    )
            else:
                items.append(
                    WindowItem(
                        sample_id=make_sample_id(
                            stem,
                            range_index,
                            0,
                            range_start,
                            range_end,
                            expanded=expand_ranges,
                        ),
                        stem=stem,
                        start=range_start,
                        end=range_end,
                        range_index=range_index,
                        window_index=0,
                    )
                )
    return items


def sample_windows_per_stem(items: list[WindowItem], windows_per_stem: int, seed: int) -> list[WindowItem]:
    if windows_per_stem <= 0:
        return items
    grouped: dict[str, list[WindowItem]] = {}
    for item in items:
        grouped.setdefault(item.stem, []).append(item)
    sampled: list[WindowItem] = []
    rng = random.Random(seed)
    for stem in sorted(grouped):
        candidates = grouped[stem]
        if len(candidates) <= windows_per_stem:
            chosen = candidates
        else:
            chosen = rng.sample(candidates, windows_per_stem)
            chosen = sorted(chosen, key=lambda x: (x.start, x.end, x.range_index, x.window_index))
        sampled.extend(chosen)
    return sampled


def infer_frame_stride(frame_indices: list[int]) -> int:
    if len(frame_indices) < 2:
        return 1
    diffs = [b - a for a, b in zip(frame_indices[:-1], frame_indices[1:]) if b > a]
    if not diffs:
        return 1
    return max(1, int(round(float(sum(diffs)) / len(diffs))))


def load_window_manifest(path: Path) -> list[WindowItem]:
    items: list[WindowItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            frame_indices = record.get("video_frame_indices")
            if not frame_indices:
                raise ValueError(f"{path}:{line_no} missing video_frame_indices")
            frame_indices = [int(x) for x in frame_indices]
            if "sample_id" not in record:
                raise ValueError(f"{path}:{line_no} missing sample_id")
            stem = str(record.get("stem") or str(record["sample_id"]).split("__", 1)[0])
            start = int(frame_indices[0])
            end = int(frame_indices[-1]) + 1
            items.append(
                WindowItem(
                    sample_id=str(record["sample_id"]),
                    stem=stem,
                    start=start,
                    end=end,
                    range_index=int(record.get("range_index", 0)),
                    window_index=int(record.get("window_index", len(items))),
                    frame_stride=int(record.get("frame_stride") or infer_frame_stride(frame_indices)),
                    sequence_length=int(record.get("sequence_length") or len(frame_indices)),
                    video_frame_indices=tuple(frame_indices),
                )
            )
    if not items:
        raise ValueError(f"No window records found in {path}")
    return items


def read_meta(dataset_root: Path, stem: str) -> str:
    meta = dataset_root / "metas" / f"{stem}.txt"
    if not meta.exists():
        return "A robot manipulates the target object."
    text = meta.read_text(errors="ignore").strip()
    return " ".join(text.split())


def get_video_len(video_path: Path) -> int:
    import decord

    return len(decord.VideoReader(str(video_path), ctx=decord.cpu(0)))


def load_frames(video_path: Path, frame_indices: list[int]) -> list[Image.Image]:
    import decord

    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    n = len(vr)
    idx = [min(max(int(i), 0), n - 1) for i in frame_indices]
    batch = vr.get_batch(idx).asnumpy()
    return [Image.fromarray(arr).convert("RGB") for arr in batch]


def sample_indices(start: int, end: int, video_len: int, num_future: int) -> tuple[int, list[int]]:
    first = min(max(start, 0), max(video_len - 1, 0))
    last = min(max(end - 1, first + 1), max(video_len - 1, 0))
    if num_future <= 1:
        return first, [last]
    raw = torch.linspace(first + 1, last, steps=num_future)
    future = [min(max(int(round(x.item())), 0), max(video_len - 1, 0)) for x in raw]
    return first, future


def sample_cosmos_indices(
    start: int,
    frame_stride: int,
    sequence_length: int,
    video_len: int,
    num_future: int,
) -> tuple[int, list[int], list[int]]:
    first = min(max(start, 0), max(video_len - 1, 0))
    clip = [min(max(first + frame_stride * i, 0), max(video_len - 1, 0)) for i in range(sequence_length)]
    if num_future <= 1:
        return first, [clip[-1]], clip
    raw = torch.linspace(1, max(sequence_length - 1, 1), steps=num_future)
    future = [clip[min(max(int(round(x.item())), 0), len(clip) - 1)] for x in raw]
    return first, future, clip


def sample_future_from_explicit_clip(
    clip: list[int],
    video_len: int,
    num_future: int,
) -> tuple[int, list[int], list[int]]:
    if not clip:
        raise ValueError("clip must contain at least one frame index")
    max_frame = max(video_len - 1, 0)
    safe_clip = [min(max(int(frame), 0), max_frame) for frame in clip]
    first = safe_clip[0]
    if num_future <= 1:
        return first, [safe_clip[-1]], safe_clip
    raw = torch.linspace(1, max(len(safe_clip) - 1, 1), steps=num_future)
    future = [safe_clip[min(max(int(round(x.item())), 0), len(safe_clip) - 1)] for x in raw]
    return first, future, safe_clip


def infer_square_grid(num_tokens: int) -> tuple[int, bool]:
    side = int(math.sqrt(num_tokens))
    if side * side == num_tokens:
        return side, False
    side = int(math.sqrt(num_tokens - 1))
    if side * side == num_tokens - 1:
        return side, True
    raise RuntimeError(f"Cannot infer square SigLIP token grid from {num_tokens} tokens.")


def semantic_feature_type(grid_size: int) -> str:
    if int(grid_size) <= 0:
        return "siglip2_penultimate_spatial_native"
    return "siglip2_penultimate_spatial_pooled"


@torch.no_grad()
def encode_images(
    model: torch.nn.Module,
    processor: object,
    images: list[Image.Image],
    grid_size: int,
    device: torch.device,
    store_dtype: torch.dtype,
) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=next(model.parameters()).dtype)
    if not hasattr(model, "vision_model"):
        raise RuntimeError("Expected an AutoModel with a vision_model attribute for SigLIP2.")
    vision = model.vision_model
    hidden = vision.embeddings(pixel_values, interpolate_pos_encoding=False)
    layers = list(vision.encoder.layers)
    if len(layers) < 2:
        raise RuntimeError(f"Expected at least two SigLIP2 vision layers, got {len(layers)}.")
    penultimate = None
    for idx, layer in enumerate(layers):
        hidden = layer(hidden, None)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        if idx == len(layers) - 2:
            penultimate = hidden
    if penultimate is None:
        raise RuntimeError("Failed to capture SigLIP2 penultimate spatial features.")
    tokens = penultimate.float()

    labels: list[torch.Tensor] = []
    for item in tokens:
        side, has_cls = infer_square_grid(item.shape[0])
        patches = item[1:] if has_cls else item
        if grid_size <= 0:
            labels.append(patches.contiguous())
            continue
        grid = patches.reshape(side, side, -1).permute(2, 0, 1).unsqueeze(0)
        pooled = F.adaptive_avg_pool2d(grid, (grid_size, grid_size))
        labels.append(pooled.squeeze(0).permute(1, 2, 0).reshape(grid_size * grid_size, -1))
    return torch.stack(labels, dim=0).to(dtype=store_dtype).cpu()


def load_siglip2(model_path: Path, device: torch.device, dtype_name: str) -> tuple[torch.nn.Module, object]:
    from transformers import AutoModel, AutoProcessor

    dtype = torch_dtype_from_name(dtype_name)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, local_files_only=True)
    model.to(device)
    model.eval()
    return model, processor


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / args.manifest_name
    progress_path = args.output_dir / args.progress_name
    if args.overwrite:
        for path in (manifest_path, progress_path):
            if path.exists():
                path.unlink()

    frame_strides = parse_int_list(args.frame_strides)
    ranges_by_stem: dict[str, list[tuple[int, int]]] = {}
    if args.window_manifest is not None:
        items = load_window_manifest(args.window_manifest)
        frame_ranges = args.frame_ranges or (args.dataset_root / "frame_ranges.json")
    else:
        frame_ranges = args.frame_ranges or (args.dataset_root / "frame_ranges.json")
        ranges_by_stem = load_frame_ranges(frame_ranges)
        items = expand_frame_ranges(
            ranges_by_stem,
            expand_ranges=args.expand_ranges,
            window_length=args.window_length,
            window_stride=args.window_stride,
            min_window_frames=args.min_window_frames,
            cosmos_stride_windows=args.cosmos_stride_windows,
            sequence_length=args.sequence_length,
            frame_strides=frame_strides,
        )
        items = sample_windows_per_stem(items, args.windows_per_stem, args.window_seed)
    total_window_items = len(items)
    if args.count_only:
        print(len(items))
        return
    if args.start_index:
        items = items[args.start_index :]
    if args.max_samples > 0:
        items = items[: args.max_samples]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    store_dtype = torch_dtype_from_name(args.dtype)
    model, processor = load_siglip2(args.model_path, device, args.dtype)

    written = 0
    with manifest_path.open("a", encoding="utf-8") as manifest, progress_path.open("a", encoding="utf-8") as progress:
        for _, item in enumerate(tqdm(items, desc="siglip2 semantic labels")):
            out_path = args.output_dir / f"{item.sample_id}.pt"
            if out_path.exists() and not args.overwrite:
                continue
            video_path = args.dataset_root / "videos" / f"{item.stem}.mp4"
            if not video_path.exists():
                continue
            try:
                video_len = get_video_len(video_path)
                if item.video_frame_indices:
                    first_idx, future_indices, video_frame_indices = sample_future_from_explicit_clip(
                        list(item.video_frame_indices),
                        video_len,
                        args.num_keyframes,
                    )
                elif item.frame_stride > 0 and item.sequence_length > 0:
                    first_idx, future_indices, video_frame_indices = sample_cosmos_indices(
                        item.start,
                        item.frame_stride,
                        item.sequence_length,
                        video_len,
                        args.num_keyframes,
                    )
                else:
                    first_idx, future_indices = sample_indices(item.start, item.end, video_len, args.num_keyframes)
                    video_frame_indices = []
                frames = load_frames(video_path, future_indices)
                semantic_plan = encode_images(
                    model=model,
                    processor=processor,
                    images=frames,
                    grid_size=args.grid_size,
                    device=device,
                    store_dtype=store_dtype,
                )
                spatial_tokens = int(semantic_plan.shape[1])
                spatial_grid_size = int(math.sqrt(spatial_tokens))
                if spatial_grid_size * spatial_grid_size != spatial_tokens:
                    raise RuntimeError(
                        f"Expected square spatial semantic tokens, got {spatial_tokens} tokens "
                        f"for sample {item.sample_id}"
                    )
                feature_type = semantic_feature_type(args.grid_size)
                payload = {
                    "sample_id": item.sample_id,
                    "stem": item.stem,
                    "video_path": str(video_path),
                    "prompt": read_meta(args.dataset_root, item.stem),
                    "first_frame_index": first_idx,
                    "range_start": item.start,
                    "range_end": item.end,
                    "range_index": item.range_index,
                    "window_index": item.window_index,
                    "frame_stride": item.frame_stride,
                    "sequence_length": item.sequence_length,
                    "video_frame_indices": video_frame_indices,
                    "future_frame_indices": future_indices,
                    "semantic_plan": semantic_plan,
                    "semantic_plan_shape": tuple(semantic_plan.shape),
                    "semantic_dim": int(semantic_plan.shape[-1]),
                    "grid_size": spatial_grid_size,
                    "requested_grid_size": args.grid_size,
                    "spatial_tokens_per_keyframe": spatial_tokens,
                    "num_keyframes": args.num_keyframes,
                    "model_path": str(args.model_path),
                    "feature_type": feature_type,
                }
                torch.save(payload, out_path)
                rec = {
                    "sample_id": item.sample_id,
                    "stem": item.stem,
                    "path": str(out_path),
                    "range_start": item.start,
                    "range_end": item.end,
                    "range_index": item.range_index,
                    "window_index": item.window_index,
                    "frame_stride": item.frame_stride,
                    "sequence_length": item.sequence_length,
                    "video_frame_indices": video_frame_indices,
                    "shape": list(semantic_plan.shape),
                    "grid_size": spatial_grid_size,
                    "requested_grid_size": args.grid_size,
                    "spatial_tokens_per_keyframe": spatial_tokens,
                    "feature_type": feature_type,
                    "first_frame_index": first_idx,
                    "future_frame_indices": future_indices,
                }
                manifest.write(json.dumps(rec) + "\n")
                manifest.flush()
                written += 1
                if written % max(args.log_every, 1) == 0:
                    progress.write(json.dumps({"written": written, "last": item.sample_id}) + "\n")
                    progress.flush()
            except Exception as exc:
                progress.write(json.dumps({"sample_id": item.sample_id, "stem": item.stem, "error": repr(exc)}) + "\n")
                progress.flush()

    summary = {
        "dataset_root": str(args.dataset_root),
        "model_path": str(args.model_path),
        "output_dir": str(args.output_dir),
        "num_keyframes": args.num_keyframes,
        "requested_grid_size": args.grid_size,
        "expand_ranges": bool(args.expand_ranges),
        "window_length": args.window_length,
        "window_stride": args.window_stride,
        "min_window_frames": args.min_window_frames,
        "windows_per_stem": args.windows_per_stem,
        "window_seed": args.window_seed,
        "window_manifest": str(args.window_manifest) if args.window_manifest is not None else None,
        "cosmos_stride_windows": bool(args.cosmos_stride_windows),
        "sequence_length": args.sequence_length,
        "frame_strides": frame_strides,
        "num_frame_range_stems": len(ranges_by_stem),
        "num_window_items_total": total_window_items,
        "num_window_items_this_shard": len(items),
        "feature_type": semantic_feature_type(args.grid_size),
        "written_this_run": written,
    }
    (args.output_dir / args.summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
