#!/usr/bin/env python3
"""Pilot precompute script for Cosmos Wan VAE latents.

This script intentionally keeps the cache format simple: each sample stores the
standardized tokenizer latent returned by ``Wan2pt1VAEInterface.encode`` plus a
JSONL manifest with frame indices and metadata.  It is meant as a small pilot
before wiring latent-cache loading into the training dataloader.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import types
from pathlib import Path
from typing import Any, Iterable

import torch


COSMOS_ROOT = Path(__file__).resolve().parents[1]
COSMOS_CUDA_ROOT = COSMOS_ROOT / "packages" / "cosmos-cuda"
for path in (COSMOS_CUDA_ROOT, COSMOS_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def install_megatron_parallel_state_stub() -> Any:
    """Install a minimal Megatron parallel_state module for single-GPU VAE use."""

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
    """Match Cosmos RF training normalization: uint8 video to float32 in [-1, 1]."""

    normalized = video.to(device=device, dtype=torch.float32) if device is not None else video.to(dtype=torch.float32)
    return normalized / 127.5 - 1.0


def latent_path_for_sample(output_dir: Path, sample_id: str) -> Path:
    return Path(output_dir) / "latents" / f"{sample_id}.pt"


def select_records(records: list[dict[str, Any]], start_index: int, max_samples: int) -> list[dict[str, Any]]:
    start = max(0, int(start_index))
    if max_samples and max_samples > 0:
        return records[start : start + int(max_samples)]
    return records[start:]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def make_manifest_record(
    *,
    output_dir: Path,
    latent_path: Path,
    latent: torch.Tensor,
    source_record: dict[str, Any],
    fps: float,
    video_size: tuple[int, int],
    tokenizer_name: str,
) -> dict[str, Any]:
    return {
        "sample_id": str(source_record["sample_id"]),
        "stem": str(source_record.get("stem", str(source_record["sample_id"]).split("__", 1)[0])),
        "latent_path": latent_path.relative_to(output_dir).as_posix(),
        "latent_shape": list(latent.shape),
        "latent_dtype": str(latent.dtype),
        "video_size": [int(video_size[0]), int(video_size[1])],
        "fps": float(fps),
        "tokenizer": tokenizer_name,
        "video_frame_indices": [int(x) for x in source_record.get("video_frame_indices", [])],
        "frame_stride": int(source_record.get("frame_stride", 1) or 1),
    }


def parse_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half", "torch.float16"}:
        return torch.float16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "sample_id" not in record:
                raise ValueError(f"{path}:{line_no} missing sample_id")
            if "stem" not in record:
                record["stem"] = str(record["sample_id"]).split("__", 1)[0]
            records.append(record)
    return records


def load_manifest_records(manifest: str | None, base_dir: Path) -> list[dict[str, Any]]:
    if not manifest or str(manifest).lower() == "none":
        return []
    paths: list[str] = []
    for item in str(manifest).split(","):
        item = item.strip()
        if not item:
            continue
        pattern = item if Path(item).is_absolute() else str(base_dir / item)
        if any(char in pattern for char in "*?[]"):
            paths.extend(sorted(glob.glob(pattern)))
        else:
            paths.append(pattern)
    if not paths:
        raise FileNotFoundError(f"No manifest files matched: {manifest}")

    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(read_jsonl(Path(path)))
    if not records:
        raise ValueError(f"No records found in manifest(s): {paths}")
    return records


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


def make_records_from_frame_ranges(
    ranges_by_stem: dict[str, list[tuple[int, int]]],
    *,
    num_frames: int,
    frame_stride: int,
    max_windows_per_range: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    needed_span = (int(num_frames) - 1) * int(frame_stride) + 1
    if needed_span <= 0:
        raise ValueError("num_frames and frame_stride must define a positive span")
    for stem in sorted(ranges_by_stem):
        for range_idx, (range_start, range_end) in enumerate(ranges_by_stem[stem]):
            if range_end - range_start < needed_span:
                continue
            max_start = range_end - needed_span
            if max_windows_per_range and max_windows_per_range > 0:
                starts = [range_start]
                if max_windows_per_range > 1 and max_start > range_start:
                    starts.append(max_start)
                if max_windows_per_range > 2 and max_start > range_start + 1:
                    middle = (range_start + max_start) // 2
                    starts.insert(1, middle)
                starts = sorted(dict.fromkeys(starts))[:max_windows_per_range]
            else:
                starts = list(range(range_start, max_start + 1))
            for window_idx, start in enumerate(starts):
                frame_ids = [start + i * int(frame_stride) for i in range(int(num_frames))]
                end = frame_ids[-1] + 1
                sample_id = (
                    f"{stem}__r{range_idx:02d}__w{window_idx:04d}"
                    f"__fs{int(frame_stride):02d}__s{start:06d}_e{end:06d}"
                )
                records.append(
                    {
                        "sample_id": sample_id,
                        "stem": stem,
                        "range_start": int(start),
                        "range_end": int(end),
                        "frame_stride": int(frame_stride),
                        "video_frame_indices": frame_ids,
                    }
                )
    return records


def frame_ids_from_record(record: dict[str, Any], num_frames: int) -> list[int]:
    frame_ids = record.get("video_frame_indices")
    if frame_ids:
        frame_ids = [int(frame_id) for frame_id in frame_ids]
    else:
        start = int(record.get("range_start", record.get("start", 0)))
        frame_stride = int(record.get("frame_stride", 1) or 1)
        frame_ids = [start + frame_stride * i for i in range(int(num_frames))]
    if len(frame_ids) != int(num_frames):
        raise ValueError(
            f"Manifest sample {record.get('sample_id')} has {len(frame_ids)} frames; expected {num_frames}"
        )
    return frame_ids


def frame_stride_from_record(record: dict[str, Any], frame_ids: list[int]) -> float:
    if record.get("frame_stride") is not None:
        return max(float(record.get("frame_stride") or 1.0), 1.0)
    if len(frame_ids) < 2:
        return 1.0
    diffs = torch.diff(torch.tensor(frame_ids, dtype=torch.float32))
    positive = diffs[diffs > 0]
    if positive.numel() == 0:
        return 1.0
    return max(float(torch.median(positive).item()), 1.0)


def load_video_tensor(
    *,
    dataset_root: Path,
    record: dict[str, Any],
    num_frames: int,
    video_size: tuple[int, int],
) -> tuple[torch.Tensor, float]:
    from decord import VideoReader, cpu
    from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import ResizePreprocess, ToTensorVideo

    stem = str(record.get("stem", str(record["sample_id"]).split("__", 1)[0]))
    video_path = dataset_root / "videos" / f"{stem}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found for {record.get('sample_id')}: {video_path}")
    frame_ids = frame_ids_from_record(record, num_frames)

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
    total_frames = len(vr)
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    safe_frame_ids = [min(max(int(frame_id), 0), total_frames - 1) for frame_id in frame_ids]
    frames = vr.get_batch(safe_frame_ids).asnumpy()
    try:
        fps = float(vr.get_avg_fps())
    except Exception:
        fps = 16.0
    vr.seek(0)
    del vr

    frame_stride = frame_stride_from_record(record, frame_ids)
    frames_tensor = torch.from_numpy(frames.astype("uint8")).permute(0, 3, 1, 2)
    frames_tensor = ToTensorVideo()(frames_tensor)
    frames_tensor = ResizePreprocess((int(video_size[0]), int(video_size[1])))(frames_tensor)
    frames_tensor = torch.clamp(frames_tensor * 255.0, 0, 255).to(torch.uint8)
    video = frames_tensor.permute(1, 0, 2, 3).contiguous()
    return video, fps / frame_stride


def build_tokenizer(*, vae_pth: Path, temporal_window: int) -> Any:
    install_megatron_parallel_state_stub()
    from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface

    return Wan2pt1VAEInterface(vae_pth=str(vae_pth), temporal_window=int(temporal_window))


def encode_record(
    *,
    tokenizer: Any,
    dataset_root: Path,
    output_dir: Path,
    record: dict[str, Any],
    num_frames: int,
    video_size: tuple[int, int],
    device: str,
    output_dtype: torch.dtype,
    overwrite: bool,
) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    latent_path = latent_path_for_sample(output_dir, sample_id)
    latent_path.parent.mkdir(parents=True, exist_ok=True)

    if latent_path.exists() and not overwrite:
        payload = torch.load(latent_path, map_location="cpu")
        latent = payload["latent"] if isinstance(payload, dict) and "latent" in payload else payload
        return make_manifest_record(
            output_dir=output_dir,
            latent_path=latent_path,
            latent=latent,
            source_record=record,
            fps=float(payload.get("meta", {}).get("fps", 0.0)) if isinstance(payload, dict) else 0.0,
            video_size=video_size,
            tokenizer_name="wan2pt1_tokenizer",
        )

    video, fps = load_video_tensor(
        dataset_root=dataset_root,
        record=record,
        num_frames=num_frames,
        video_size=video_size,
    )
    vae_input = normalize_uint8_video_for_vae(video.unsqueeze(0), device=device)
    with torch.no_grad():
        latent = tokenizer.encode(vae_input).detach().to("cpu", dtype=output_dtype)
    manifest_record = make_manifest_record(
        output_dir=output_dir,
        latent_path=latent_path,
        latent=latent,
        source_record=record,
        fps=fps,
        video_size=video_size,
        tokenizer_name="wan2pt1_tokenizer",
    )
    torch.save({"latent": latent, "meta": manifest_record}, latent_path)
    return manifest_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--semantic-plan-manifest", default="")
    parser.add_argument("--semantic-plan-dir", type=Path, default=None)
    parser.add_argument("--frame-ranges", type=Path, default=None)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-windows-per-range", type=int, default=1)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument(
        "--vae-pth",
        type=Path,
        default=Path("/data/user/jhe724/workspace/weights/Cosmos-Predict2.5-2B/tokenizer.pth"),
    )
    parser.add_argument("--temporal-window", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dtype", default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_base = args.semantic_plan_dir or args.dataset_root
    records = load_manifest_records(args.semantic_plan_manifest, manifest_base)
    source_kind = "semantic_plan_manifest"
    if not records:
        frame_ranges_path = args.frame_ranges or args.dataset_root / "frame_ranges.json"
        ranges_by_stem = load_frame_ranges(frame_ranges_path)
        records = make_records_from_frame_ranges(
            ranges_by_stem,
            num_frames=args.num_frames,
            frame_stride=args.frame_stride,
            max_windows_per_range=args.max_windows_per_range,
        )
        source_kind = "frame_ranges"
    if not records:
        raise ValueError("No eligible VAE latent records were found")

    selected_records = select_records(records, args.start_index, args.max_samples)
    if not selected_records:
        raise ValueError(
            f"No records selected from {len(records)} records with start_index={args.start_index}, "
            f"max_samples={args.max_samples}"
        )

    output_dtype = parse_dtype(args.output_dtype)
    tokenizer = build_tokenizer(vae_pth=args.vae_pth, temporal_window=args.temporal_window)
    manifest_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(selected_records, start=1):
        manifest_record = encode_record(
            tokenizer=tokenizer,
            dataset_root=args.dataset_root,
            output_dir=output_dir,
            record=record,
            num_frames=args.num_frames,
            video_size=(args.height, args.width),
            device=args.device,
            output_dtype=output_dtype,
            overwrite=args.overwrite,
        )
        manifest_rows.append(manifest_record)
        if idx == 1 or idx % 10 == 0 or idx == len(selected_records):
            print(
                f"[{idx}/{len(selected_records)}] {manifest_record['sample_id']} "
                f"latent_shape={manifest_record['latent_shape']}",
                flush=True,
            )

    manifest_path = output_dir / "manifest.jsonl"
    write_jsonl(manifest_path, manifest_rows)
    summary = {
        "dataset_root": str(args.dataset_root),
        "source_kind": source_kind,
        "num_total_records": len(records),
        "num_written_records": len(manifest_rows),
        "start_index": int(args.start_index),
        "max_samples": int(args.max_samples),
        "num_frames": int(args.num_frames),
        "frame_stride": int(args.frame_stride),
        "video_size": [int(args.height), int(args.width)],
        "vae_pth": str(args.vae_pth),
        "tokenizer": "wan2pt1_tokenizer",
        "output_dtype": str(output_dtype),
        "manifest": manifest_path.name,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
