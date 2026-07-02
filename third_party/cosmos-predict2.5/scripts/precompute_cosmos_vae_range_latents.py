#!/usr/bin/env python3
"""Precompute stride-specific range/chunk VAE latents for Cosmos training.

This follows the storage pattern used by Ctrl-World, but adapts it for the Wan
video VAE's temporal compression.  Instead of saving one latent file per
sliding training window, it saves longer stride-specific cache chunks and writes
a second manifest describing aligned training windows as latent crops.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import types
from pathlib import Path
from typing import Any, Iterable

import torch


COSMOS_ROOT = Path(__file__).resolve().parents[1]
COSMOS_CUDA_ROOT = COSMOS_ROOT / "packages" / "cosmos-cuda"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, COSMOS_CUDA_ROOT, COSMOS_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def install_megatron_parallel_state_stub() -> Any:
    try:
        from megatron.core import parallel_state

        return parallel_state
    except Exception:
        pass

    megatron_module = sys.modules.get("megatron") or types.ModuleType("megatron")
    core_module = sys.modules.get("megatron.core") or types.ModuleType("megatron.core")
    parallel_state_module = types.ModuleType("megatron.core.parallel_state")

    def _zero(*args: Any, **kwargs: Any) -> int:
        return 0

    def _one(*args: Any, **kwargs: Any) -> int:
        return 1

    def _false(*args: Any, **kwargs: Any) -> bool:
        return False

    def _true(*args: Any, **kwargs: Any) -> bool:
        return True

    def _none(*args: Any, **kwargs: Any) -> None:
        return None

    for name in (
        "destroy_model_parallel",
        "initialize_model_parallel",
        "set_virtual_pipeline_model_parallel_rank",
    ):
        setattr(parallel_state_module, name, _none)
    for name in (
        "get_data_parallel_rank",
        "get_data_parallel_src_rank",
        "get_context_parallel_rank",
        "get_tensor_model_parallel_rank",
        "get_tensor_model_parallel_src_rank",
        "get_pipeline_model_parallel_rank",
        "get_pipeline_model_parallel_first_rank",
        "get_pipeline_model_parallel_last_rank",
        "get_pipeline_model_parallel_next_rank",
        "get_pipeline_model_parallel_prev_rank",
        "get_expert_model_parallel_rank",
    ):
        setattr(parallel_state_module, name, _zero)
    for name in (
        "get_data_parallel_world_size",
        "get_context_parallel_world_size",
        "get_tensor_model_parallel_world_size",
        "get_pipeline_model_parallel_world_size",
        "get_expert_model_parallel_world_size",
    ):
        setattr(parallel_state_module, name, _one)
    for name in (
        "get_data_parallel_group",
        "get_context_parallel_group",
        "get_tensor_model_parallel_group",
        "get_pipeline_model_parallel_group",
        "get_expert_model_parallel_group",
        "get_virtual_pipeline_model_parallel_rank",
    ):
        setattr(parallel_state_module, name, _none)
    for name in (
        "is_initialized",
        "is_pipeline_stage_before_split",
        "is_pipeline_stage_after_split",
    ):
        setattr(parallel_state_module, name, _false)
    for name in ("is_pipeline_first_stage", "is_pipeline_last_stage"):
        setattr(parallel_state_module, name, _true)

    core_module.parallel_state = parallel_state_module
    megatron_module.core = core_module
    sys.modules["megatron"] = megatron_module
    sys.modules["megatron.core"] = core_module
    sys.modules["megatron.core.parallel_state"] = parallel_state_module
    return parallel_state_module


def normalize_uint8_video_for_vae(video: torch.Tensor, device: str | torch.device | None = None) -> torch.Tensor:
    normalized = video.to(device=device, dtype=torch.float32) if device is not None else video.to(dtype=torch.float32)
    return normalized / 127.5 - 1.0


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"Expected a positive integer, got {value}")
        values.append(value)
    return values


def parse_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half", "torch.float16"}:
        return torch.float16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_frame_ranges(path: Path) -> dict[str, list[tuple[int, int]]]:
    payload = json.loads(path.read_text())
    ranges_by_stem: dict[str, list[tuple[int, int]]] = {}
    if isinstance(payload, dict):
        iterable = payload.items()
    elif isinstance(payload, list):
        iterable = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            stem = item.get("stem") or item.get("video_id") or item.get("id")
            if stem is not None:
                iterable.append((str(stem), item))
    else:
        raise TypeError(f"Unsupported frame_ranges type: {type(payload)!r}")

    for raw_stem, raw_ranges in iterable:
        stem = Path(str(raw_stem)).stem
        parsed: list[tuple[int, int]] = []
        if isinstance(raw_ranges, dict):
            if "ranges" in raw_ranges:
                raw_ranges = raw_ranges["ranges"]
            elif "frame_ranges" in raw_ranges:
                raw_ranges = raw_ranges["frame_ranges"]
            else:
                start = int(raw_ranges.get("start", raw_ranges.get("frame_start", raw_ranges.get("onset", 0))))
                end = int(raw_ranges.get("end", raw_ranges.get("frame_end", start + 1)))
                parsed.append((start, end))
        if isinstance(raw_ranges, list):
            if raw_ranges and isinstance(raw_ranges[0], (list, tuple)):
                parsed.extend((int(item[0]), int(item[1])) for item in raw_ranges if len(item) >= 2)
            elif raw_ranges and isinstance(raw_ranges[0], dict):
                for item in raw_ranges:
                    start = int(item.get("start", item.get("frame_start", 0)))
                    end = int(item.get("end", item.get("frame_end", start + 1)))
                    parsed.append((start, end))
            elif len(raw_ranges) >= 2:
                parsed.append((int(raw_ranges[0]), int(raw_ranges[1])))
        valid = [(max(0, start), max(0, end)) for start, end in parsed if end > start]
        if valid:
            ranges_by_stem[stem] = valid
    return ranges_by_stem


def latent_num_frames_for_sequence_length(sequence_length: int) -> int:
    if int(sequence_length) <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    return 1 + (int(sequence_length) - 1) // 4


def sequence_length_for_range(range_start: int, range_end: int, frame_stride: int) -> int:
    if range_end <= range_start:
        return 0
    return len(range(int(range_start), int(range_end), int(frame_stride)))


def cache_path_for_record(output_dir: Path, cache_id: str) -> Path:
    return Path(output_dir) / "range_latents" / f"{cache_id}.pt"


def iter_cache_spans(
    sequence_length: int,
    *,
    min_sequence_length: int,
    cache_num_frames: int,
    cache_step_frames: int,
) -> list[tuple[int, int]]:
    if sequence_length < min_sequence_length:
        return []
    if cache_num_frames <= 0 or sequence_length <= cache_num_frames:
        return [(0, sequence_length)]
    if cache_num_frames < min_sequence_length:
        raise ValueError("cache_num_frames must be >= min_sequence_length")

    step = int(cache_step_frames)
    if step <= 0:
        step = cache_num_frames - min_sequence_length + 1
    if step <= 0:
        raise ValueError("cache_step_frames must be positive")

    max_start = sequence_length - cache_num_frames
    starts = list(range(0, max_start + 1, step))
    if not starts or starts[-1] != max_start:
        starts.append(max_start)
    starts = sorted(dict.fromkeys(starts))
    return [(start, min(start + cache_num_frames, sequence_length)) for start in starts]


def make_cache_id(
    stem: str,
    *,
    range_index: int,
    frame_stride: int,
    chunk_index: int,
    sequence_start_index: int,
    sequence_end_index: int,
) -> str:
    return (
        f"{stem}__r{range_index:02d}__fs{frame_stride:02d}__c{chunk_index:04d}"
        f"__q{sequence_start_index:06d}_{sequence_end_index:06d}"
    )


def make_cache_records_from_frame_ranges(
    ranges_by_stem: dict[str, list[tuple[int, int]]],
    *,
    output_dir: Path,
    frame_strides: list[int],
    min_sequence_length: int,
    cache_num_frames: int,
    cache_step_frames: int,
    caches_per_range: int = 0,
    cache_seed: int = 0,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for stem in sorted(ranges_by_stem):
        for range_index, (range_start, range_end) in enumerate(ranges_by_stem[stem]):
            range_start = max(0, int(range_start))
            range_end = max(range_start, int(range_end))
            for frame_stride in frame_strides:
                full_sequence_length = sequence_length_for_range(range_start, range_end, frame_stride)
                spans = iter_cache_spans(
                    full_sequence_length,
                    min_sequence_length=min_sequence_length,
                    cache_num_frames=cache_num_frames,
                    cache_step_frames=cache_step_frames,
                )
                if caches_per_range > 0 and len(spans) > caches_per_range:
                    rng = random.Random(f"{int(cache_seed)}:{stem}:{range_index}:{frame_stride}")
                    spans = sorted(rng.sample(spans, caches_per_range))
                for chunk_index, (sequence_start_index, sequence_end_index) in enumerate(spans):
                    sequence_length = sequence_end_index - sequence_start_index
                    first_pixel = range_start + frame_stride * sequence_start_index
                    last_pixel = range_start + frame_stride * (sequence_end_index - 1)
                    cache_id = make_cache_id(
                        stem,
                        range_index=range_index,
                        frame_stride=frame_stride,
                        chunk_index=chunk_index,
                        sequence_start_index=sequence_start_index,
                        sequence_end_index=sequence_end_index,
                    )
                    latent_path = cache_path_for_record(output_dir, cache_id)
                    records.append(
                        {
                            "cache_id": cache_id,
                            "stem": stem,
                            "range_index": int(range_index),
                            "range_start": int(range_start),
                            "range_end": int(range_end),
                            "frame_stride": int(frame_stride),
                            "chunk_index": int(chunk_index),
                            "sequence_start_index": int(sequence_start_index),
                            "sequence_end_index": int(sequence_end_index),
                            "sequence_length": int(sequence_length),
                            "full_sequence_length": int(full_sequence_length),
                            "first_pixel_frame": int(first_pixel),
                            "last_pixel_frame": int(last_pixel),
                            "latent_num_frames": latent_num_frames_for_sequence_length(sequence_length),
                            "latent_path": latent_path.relative_to(output_dir).as_posix(),
                        }
                    )
    return records


def make_window_sample_id(
    stem: str,
    *,
    range_index: int,
    frame_stride: int,
    chunk_index: int,
    window_index: int,
    start: int,
    end: int,
) -> str:
    return (
        f"{stem}__r{range_index:02d}__fs{frame_stride:02d}__c{chunk_index:04d}"
        f"__w{window_index:04d}__s{start:06d}_e{end:06d}"
    )


def make_window_records_from_cache_records(
    cache_records: list[dict[str, Any]],
    *,
    num_frames: int,
    windows_per_cache: int,
    seed: int,
    temporal_alignment: int,
) -> list[dict[str, Any]]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if temporal_alignment <= 0:
        raise ValueError("temporal_alignment must be positive")
    latent_num_frames = latent_num_frames_for_sequence_length(num_frames)
    window_records: list[dict[str, Any]] = []
    for cache_record in cache_records:
        sequence_length = int(cache_record["sequence_length"])
        if sequence_length < num_frames:
            continue
        rel_starts = list(range(0, sequence_length - num_frames + 1, temporal_alignment))
        if windows_per_cache > 0 and len(rel_starts) > windows_per_cache:
            rng = random.Random(f"{int(seed)}:{cache_record['cache_id']}")
            rel_starts = sorted(rng.sample(rel_starts, windows_per_cache))
        for window_index, rel_start in enumerate(rel_starts):
            sequence_start_index = int(cache_record["sequence_start_index"]) + rel_start
            frame_stride = int(cache_record["frame_stride"])
            pixel_start = int(cache_record["range_start"]) + frame_stride * sequence_start_index
            video_frame_indices = [pixel_start + frame_stride * i for i in range(num_frames)]
            pixel_end = video_frame_indices[-1] + 1
            chunk_index = int(cache_record.get("chunk_index", 0))
            sample_id = make_window_sample_id(
                str(cache_record["stem"]),
                range_index=int(cache_record.get("range_index", 0)),
                frame_stride=frame_stride,
                chunk_index=chunk_index,
                window_index=window_index,
                start=pixel_start,
                end=pixel_end,
            )
            window_records.append(
                {
                    "sample_id": sample_id,
                    "stem": str(cache_record["stem"]),
                    "cache_id": str(cache_record["cache_id"]),
                    "latent_path": str(cache_record["latent_path"]),
                    "range_index": int(cache_record.get("range_index", 0)),
                    "range_start": int(cache_record["range_start"]),
                    "range_end": int(cache_record["range_end"]),
                    "frame_stride": frame_stride,
                    "sequence_length": int(num_frames),
                    "sequence_start_index": int(sequence_start_index),
                    "sequence_end_index": int(sequence_start_index + num_frames),
                    "cache_sequence_start_index": int(cache_record["sequence_start_index"]),
                    "cache_sequence_end_index": int(cache_record["sequence_end_index"]),
                    "chunk_index": chunk_index,
                    "window_index": int(window_index),
                    "latent_offset": int(rel_start // 4),
                    "latent_num_frames": int(latent_num_frames),
                    "video_frame_indices": video_frame_indices,
                }
            )
    return window_records


def crop_latent_window(latent: torch.Tensor, window_record: dict[str, Any]) -> torch.Tensor:
    offset = int(window_record["latent_offset"])
    length = int(window_record["latent_num_frames"])
    if offset < 0 or length <= 0:
        raise ValueError(f"Invalid latent crop offset={offset}, length={length}")
    if latent.shape[2] < offset + length:
        raise ValueError(
            f"Latent has {latent.shape[2]} frames; cannot crop offset={offset}, length={length}"
        )
    return latent[:, :, offset : offset + length]


def frame_ids_for_cache_record(record: dict[str, Any]) -> list[int]:
    range_start = int(record["range_start"])
    stride = int(record["frame_stride"])
    return [
        range_start + stride * seq_idx
        for seq_idx in range(int(record["sequence_start_index"]), int(record["sequence_end_index"]))
    ]


def load_video_tensor_for_cache(
    *,
    dataset_root: Path,
    record: dict[str, Any],
    video_size: tuple[int, int],
) -> tuple[torch.Tensor, float]:
    from decord import VideoReader, cpu
    from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import ResizePreprocess, ToTensorVideo

    video_path = dataset_root / "videos" / f"{record['stem']}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found for cache {record['cache_id']}: {video_path}")
    frame_ids = frame_ids_for_cache_record(record)
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
    total_frames = len(vr)
    safe_frame_ids = [min(max(int(frame_id), 0), total_frames - 1) for frame_id in frame_ids]
    frames = vr.get_batch(safe_frame_ids).asnumpy()
    try:
        fps = float(vr.get_avg_fps())
    except Exception:
        fps = 16.0
    vr.seek(0)
    del vr

    frames_tensor = torch.from_numpy(frames.astype("uint8")).permute(0, 3, 1, 2)
    frames_tensor = ToTensorVideo()(frames_tensor)
    frames_tensor = ResizePreprocess((int(video_size[0]), int(video_size[1])))(frames_tensor)
    frames_tensor = torch.clamp(frames_tensor * 255.0, 0, 255).to(torch.uint8)
    video = frames_tensor.permute(1, 0, 2, 3).contiguous()
    return video, fps / float(record["frame_stride"])


def build_tokenizer(*, vae_pth: Path, temporal_window: int) -> Any:
    install_megatron_parallel_state_stub()
    from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface

    return Wan2pt1VAEInterface(vae_pth=str(vae_pth), temporal_window=int(temporal_window))


def encode_cache_record(
    *,
    tokenizer: Any,
    dataset_root: Path,
    output_dir: Path,
    record: dict[str, Any],
    video_size: tuple[int, int],
    device: str,
    output_dtype: torch.dtype,
    overwrite: bool,
) -> dict[str, Any]:
    latent_path = output_dir / str(record["latent_path"])
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    if latent_path.exists() and not overwrite:
        payload = torch.load(latent_path, map_location="cpu", weights_only=False)
        latent = payload["latent"] if isinstance(payload, dict) and "latent" in payload else payload
        out = dict(record)
        out["latent_shape"] = list(latent.shape)
        out["latent_dtype"] = str(latent.dtype)
        return out

    video, fps = load_video_tensor_for_cache(dataset_root=dataset_root, record=record, video_size=video_size)
    vae_input = normalize_uint8_video_for_vae(video.unsqueeze(0), device=device)
    with torch.no_grad():
        latent = tokenizer.encode(vae_input).detach().to("cpu", dtype=output_dtype)
    out = dict(record)
    out["fps"] = float(fps)
    out["video_size"] = [int(video_size[0]), int(video_size[1])]
    out["latent_shape"] = list(latent.shape)
    out["latent_dtype"] = str(latent.dtype)
    torch.save({"latent": latent, "meta": out}, latent_path)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-ranges", type=Path, default=None)
    parser.add_argument("--frame-strides", default="1,2,3")
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--cache-num-frames", type=int, default=93)
    parser.add_argument("--cache-step-frames", type=int, default=0)
    parser.add_argument("--caches-per-range", type=int, default=0)
    parser.add_argument("--cache-seed", type=int, default=20260701)
    parser.add_argument("--windows-per-cache", type=int, default=4)
    parser.add_argument("--window-seed", type=int, default=20260701)
    parser.add_argument("--temporal-alignment", type=int, default=4)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--start-cache-index", type=int, default=0)
    parser.add_argument("--max-cache-records", type=int, default=0)
    parser.add_argument(
        "--vae-pth",
        type=Path,
        default=Path("/data/user/jhe724/workspace/weights/Cosmos-Predict2.5-2B/tokenizer.pth"),
    )
    parser.add_argument("--temporal-window", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dtype", default="bf16")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cache-manifest-name", default="cache_manifest.jsonl")
    parser.add_argument("--window-manifest-name", default="window_manifest.jsonl")
    parser.add_argument("--summary-name", default="summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_ranges_path = args.frame_ranges or args.dataset_root / "frame_ranges.json"
    frame_ranges = load_frame_ranges(frame_ranges_path)
    frame_strides = parse_int_list(args.frame_strides)
    cache_records_all = make_cache_records_from_frame_ranges(
        frame_ranges,
        output_dir=args.output_dir,
        frame_strides=frame_strides,
        min_sequence_length=args.num_frames,
        cache_num_frames=args.cache_num_frames,
        cache_step_frames=args.cache_step_frames,
        caches_per_range=args.caches_per_range,
        cache_seed=args.cache_seed,
    )
    start = max(0, int(args.start_cache_index))
    cache_records = cache_records_all[start:]
    if args.max_cache_records > 0:
        cache_records = cache_records[: int(args.max_cache_records)]

    encoded_records: list[dict[str, Any]] = []
    if args.manifest_only:
        encoded_records = cache_records
    else:
        tokenizer = build_tokenizer(vae_pth=args.vae_pth, temporal_window=args.temporal_window)
        output_dtype = parse_dtype(args.output_dtype)
        for idx, record in enumerate(cache_records, start=1):
            encoded = encode_cache_record(
                tokenizer=tokenizer,
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                record=record,
                video_size=(args.height, args.width),
                device=args.device,
                output_dtype=output_dtype,
                overwrite=args.overwrite,
            )
            encoded_records.append(encoded)
            if idx == 1 or idx % 10 == 0 or idx == len(cache_records):
                print(
                    f"[{idx}/{len(cache_records)}] {encoded['cache_id']} "
                    f"latent_shape={encoded.get('latent_shape')}",
                    flush=True,
                )

    window_records = make_window_records_from_cache_records(
        cache_records,
        num_frames=args.num_frames,
        windows_per_cache=args.windows_per_cache,
        seed=args.window_seed,
        temporal_alignment=args.temporal_alignment,
    )
    cache_manifest_path = args.output_dir / args.cache_manifest_name
    window_manifest_path = args.output_dir / args.window_manifest_name
    write_jsonl(cache_manifest_path, encoded_records)
    write_jsonl(window_manifest_path, window_records)
    summary = {
        "dataset_root": str(args.dataset_root),
        "frame_ranges": str(frame_ranges_path),
        "output_dir": str(args.output_dir),
        "num_frame_range_stems": len(frame_ranges),
        "frame_strides": frame_strides,
        "num_frames": int(args.num_frames),
        "cache_num_frames": int(args.cache_num_frames),
        "cache_step_frames": int(args.cache_step_frames),
        "caches_per_range": int(args.caches_per_range),
        "cache_seed": int(args.cache_seed),
        "windows_per_cache": int(args.windows_per_cache),
        "window_seed": int(args.window_seed),
        "temporal_alignment": int(args.temporal_alignment),
        "num_cache_records_total": len(cache_records_all),
        "start_cache_index": int(args.start_cache_index),
        "max_cache_records": int(args.max_cache_records),
        "num_cache_records_this_run": len(cache_records),
        "num_window_records_this_run": len(window_records),
        "manifest_only": bool(args.manifest_only),
        "cache_manifest": cache_manifest_path.name,
        "window_manifest": window_manifest_path.name,
    }
    (args.output_dir / args.summary_name).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
