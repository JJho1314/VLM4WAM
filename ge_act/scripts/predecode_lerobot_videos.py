#!/usr/bin/env python3
"""Predecode LeRobot camera MP4s into strict episode-level RGB NumPy caches."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import av
import numpy as np
import yaml


def validate_rgb_array(frames: np.ndarray, *, source: str = "array") -> None:
    if (
        frames.ndim != 4
        or frames.shape[-1] != 3
        or frames.dtype != np.uint8
        or len(frames) == 0
    ):
        raise ValueError(
            f"invalid RGB cache {source}: expected nonempty [T,H,W,3] uint8, "
            f"got shape={frames.shape}, dtype={frames.dtype}"
        )


def cache_path_for_video(
    video_path: Path, data_root: Path, cache_root: Path
) -> Path:
    video_path = Path(video_path)
    data_root = Path(data_root)
    try:
        relative_path = video_path.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(
            f"video is outside data root: video={video_path}, root={data_root}"
        ) from exc
    return (Path(cache_root) / relative_path).with_suffix(".npy")


def write_rgb_cache_atomic(cache_path: Path, frames: np.ndarray) -> None:
    validate_rgb_array(frames, source=str(cache_path))
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(
        cache_path.suffix + f".{os.getpid()}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            np.save(handle, frames, allow_pickle=False)
        os.replace(temporary, cache_path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_rgb_cache(cache_path: Path) -> tuple[bool, str]:
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        return False, f"missing RGB cache: {cache_path}"
    try:
        frames = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        validate_rgb_array(frames, source=str(cache_path))
    except (OSError, ValueError) as exc:
        return False, str(exc)
    return True, ""


def decode_rgb_video(video_path: Path) -> np.ndarray:
    decoded_frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            decoded_frames.append(frame.to_ndarray(format="rgb24"))
    if not decoded_frames:
        raise ValueError(f"video has no frames: {video_path}")
    frames = np.stack(decoded_frames).astype(np.uint8, copy=False)
    validate_rgb_array(frames, source=str(video_path))
    return frames


def _training_data_config(config: dict[str, Any]) -> dict[str, Any]:
    try:
        return config["data"]["train"]
    except (KeyError, TypeError) as exc:
        raise ValueError("config is missing data.train") from exc


def _unique_source_specs(
    data_config: dict[str, Any],
) -> Iterable[tuple[Path, str, str]]:
    roots = list(data_config.get("data_roots", []))
    domains = list(data_config.get("domains", []))
    cameras = list(data_config.get("valid_cam", []))
    if len(roots) == 1 and len(domains) > 1:
        roots *= len(domains)
    if not roots or len(roots) != len(domains) or not cameras:
        raise ValueError(
            "data.train must define aligned data_roots/domains and valid_cam"
        )
    return sorted({(Path(root), domain, camera) for root, domain in zip(roots, domains) for camera in cameras})


def discover_video_pairs(
    config: dict[str, Any],
) -> list[tuple[Path, Path]]:
    data_config = _training_data_config(config)
    cache_root_value = data_config.get("predecoded_video_root")
    if not cache_root_value:
        raise ValueError("data.train.predecoded_video_root is required")
    cache_root = Path(cache_root_value)
    pairs: dict[Path, Path] = {}
    missing_specs = []
    for data_root, domain, camera in _unique_source_specs(data_config):
        camera_root = data_root / domain / "videos"
        videos = sorted(camera_root.glob(f"chunk-*/{camera}/episode_*.mp4"))
        if not videos:
            missing_specs.append(f"{data_root}:{domain}:{camera}")
            continue
        for video_path in videos:
            pairs[video_path] = cache_path_for_video(
                video_path, data_root, cache_root
            )
    if missing_specs:
        raise FileNotFoundError(
            "no source MP4 files for: " + ", ".join(missing_specs)
        )
    return sorted(pairs.items(), key=lambda pair: str(pair[0]))


def _process_video_pair(pair: tuple[Path, Path]) -> dict[str, Any]:
    video_path, cache_path = pair
    valid, _ = verify_rgb_cache(cache_path)
    if valid:
        frames = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        return {
            "status": "skipped",
            "source": str(video_path),
            "cache": str(cache_path),
            "frames": len(frames),
            "bytes": cache_path.stat().st_size,
        }
    try:
        frames = decode_rgb_video(video_path)
        write_rgb_cache_atomic(cache_path, frames)
        return {
            "status": "written",
            "source": str(video_path),
            "cache": str(cache_path),
            "frames": len(frames),
            "bytes": cache_path.stat().st_size,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "source": str(video_path),
            "cache": str(cache_path),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _cache_root(config: dict[str, Any]) -> Path:
    value = _training_data_config(config).get("predecoded_video_root")
    if not value:
        raise ValueError("data.train.predecoded_video_root is required")
    return Path(value)


def _write_manifest(cache_root: Path, manifest: dict[str, Any]) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / "manifest.json"
    temporary = cache_root / f"manifest.json.{os.getpid()}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def predecode_all(
    pairs: list[tuple[Path, Path]], *, workers: int
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total": len(pairs),
        "written": 0,
        "skipped": 0,
        "failed": 0,
        "frames": 0,
        "bytes": 0,
        "errors": [],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_video_pair, pair) for pair in pairs]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            status = result["status"]
            summary[status] += 1
            summary["frames"] += result.get("frames", 0)
            summary["bytes"] += result.get("bytes", 0)
            if status == "failed":
                summary["errors"].append(result)
            if completed % 100 == 0 or completed == len(futures):
                print(
                    f"predecode {completed}/{len(futures)} "
                    f"written={summary['written']} skipped={summary['skipped']} "
                    f"failed={summary['failed']}",
                    flush=True,
                )
    return summary


def verify_all(pairs: list[tuple[Path, Path]]) -> list[str]:
    errors = []
    for _, cache_path in pairs:
        valid, message = verify_rgb_cache(cache_path)
        if not valid:
            errors.append(message)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--workers", type=int, default=min(32, os.cpu_count() or 1)
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not _training_data_config(config).get("require_predecoded", False):
        print("predecode requires data.train.require_predecoded=true")
        return 1
    try:
        pairs = discover_video_pairs(config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"predecode discovery failed: {exc}")
        return 1
    print(f"discovered {len(pairs)} source camera videos", flush=True)

    if args.verify_only:
        errors = verify_all(pairs)
        if errors:
            print(f"predecode verification failed: {len(errors)} invalid caches")
            for error in errors[:20]:
                print(f"  - {error}")
            return 1
        print(f"predecode verification passed: {len(pairs)} caches")
        return 0

    summary = predecode_all(pairs, workers=args.workers)
    summary["config"] = str(args.config.resolve())
    _write_manifest(_cache_root(config), summary)
    if summary["failed"]:
        print(f"predecode failed for {summary['failed']} videos")
        return 1
    print(
        f"predecode complete: written={summary['written']} "
        f"skipped={summary['skipped']} bytes={summary['bytes']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
