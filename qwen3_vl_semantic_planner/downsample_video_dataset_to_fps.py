#!/usr/bin/env python3
"""Create a lower-FPS copy of a Cosmos/DROID video dataset.

The script writes a new dataset root with re-encoded videos and a time-aligned
``frame_ranges.json``.  Derived feature folders such as SigLIP/Qwen semantic
plans are intentionally not copied because their frame indices become stale
after downsampling.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
DEFAULT_SIDECAR_DIRS = ("captions", "metas")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--video-subdir", default="videos")
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument(
        "--source-fps",
        type=float,
        default=0.0,
        help="Optional fixed source FPS for frame range mapping; if unset, ffprobe is used per video.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--target-height", type=int, default=0)
    parser.add_argument("--target-width", type=int, default=0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-copy-sidecars", action="store_true")
    return parser.parse_args()


def format_fps(value: float) -> str:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"target fps must be positive, got {value}")
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def build_ffmpeg_command(
    input_path: Path,
    output_path: Path,
    *,
    target_fps: float,
    crf: int,
    preset: str,
    overwrite: bool,
    target_height: int = 0,
    target_width: int = 0,
) -> list[str]:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    command.append("-y" if overwrite else "-n")
    filters = [f"fps={format_fps(target_fps)}"]
    if target_height > 0 or target_width > 0:
        if target_height <= 0 or target_width <= 0:
            raise ValueError("target_height and target_width must both be positive when resizing")
        filters.append(f"scale={target_width}:{target_height}:flags=lanczos")

    command.extend(
        [
            "-i",
            str(input_path),
            "-vf",
            ",".join(filters),
            "-vsync",
            "cfr",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    if preset:
        insert_at = command.index("-crf")
        command[insert_at:insert_at] = ["-preset", preset]
    return command


def parse_fraction(text: str) -> float:
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            return 0.0
        return float(numerator) / denominator_value
    return float(text)


def probe_video_fps(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"No video stream found in {path}")
    stream = streams[0]
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = parse_fraction(str(stream.get(key, "0/0")))
        if value > 0:
            return value
    raise ValueError(f"Could not read FPS from {path}")


def discover_videos(
    dataset_root: Path,
    video_subdir: str,
    *,
    max_videos: int = 0,
    recursive: bool = False,
) -> list[Path]:
    video_root = dataset_root / video_subdir
    if not video_root.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_root}")
    source_iter = video_root.rglob("*") if recursive else video_root.iterdir()
    iterator = (path for path in source_iter if path.suffix.lower() in VIDEO_SUFFIXES)
    if max_videos > 0:
        videos: list[Path] = []
        for path in iterator:
            videos.append(path)
            if len(videos) >= max_videos:
                break
        return videos
    return sorted(iterator)


def load_frame_ranges(path: Path) -> dict[str, list[tuple[int, int]]]:
    data = json.loads(path.read_text())
    ranges_by_stem: dict[str, list[tuple[int, int]]] = {}
    if isinstance(data, dict):
        iterable = data.items()
    elif isinstance(data, list):
        iterable = ((str(x.get("stem") or x.get("video_id") or x.get("id")), x) for x in data)
    else:
        raise TypeError(f"Unsupported frame_ranges type: {type(data)!r}")

    for stem, ranges in iterable:
        parsed: list[tuple[int, int]] = []
        if isinstance(ranges, dict):
            start = int(ranges.get("start", ranges.get("frame_start", 0)))
            end = int(ranges.get("end", ranges.get("frame_end", start + 1)))
            parsed.append((start, end))
        elif isinstance(ranges, list) and ranges and isinstance(ranges[0], (list, tuple)):
            parsed.extend((int(item[0]), int(item[1])) for item in ranges if len(item) >= 2)
        elif isinstance(ranges, list) and len(ranges) >= 2:
            parsed.append((int(ranges[0]), int(ranges[1])))
        valid = [(max(0, start), max(0, end)) for start, end in parsed if end > start]
        if valid:
            ranges_by_stem[str(stem)] = valid
    return ranges_by_stem


def map_exclusive_frame_index(frame_index: int, *, source_fps: float, target_fps: float) -> int:
    if source_fps <= 0:
        raise ValueError(f"source fps must be positive, got {source_fps}")
    return max(0, int(math.floor(float(frame_index) * target_fps / source_fps + 1e-6)))


def map_exclusive_frame_range(start: int, end: int, *, source_fps: float, target_fps: float) -> tuple[int, int]:
    mapped_start = map_exclusive_frame_index(start, source_fps=source_fps, target_fps=target_fps)
    mapped_end = map_exclusive_frame_index(end, source_fps=source_fps, target_fps=target_fps)
    if mapped_end <= mapped_start:
        mapped_end = mapped_start + 1
    return mapped_start, mapped_end


def rewrite_frame_ranges(
    src_root: Path,
    dst_root: Path,
    *,
    target_fps: float,
    source_fps_by_stem: dict[str, float],
    default_source_fps: float = 0.0,
    frame_ranges_name: str = "frame_ranges.json",
) -> dict[str, list[list[int]]] | None:
    src_path = src_root / frame_ranges_name
    if not src_path.exists():
        return None
    ranges_by_stem = load_frame_ranges(src_path)
    rewritten: dict[str, list[list[int]]] = {}
    audit_lines = [
        "\t".join(
            [
                "stem",
                "source_fps",
                "target_fps",
                "old_start",
                "old_end",
                "new_start",
                "new_end",
            ]
        )
    ]
    for stem, ranges in sorted(ranges_by_stem.items()):
        source_fps = source_fps_by_stem.get(stem)
        if source_fps is None and default_source_fps > 0:
            source_fps = default_source_fps
        if source_fps is None:
            raise KeyError(f"No source FPS available for frame_ranges stem: {stem}")
        mapped_ranges: list[list[int]] = []
        for start, end in ranges:
            mapped_start, mapped_end = map_exclusive_frame_range(
                start,
                end,
                source_fps=source_fps,
                target_fps=target_fps,
            )
            mapped_ranges.append([mapped_start, mapped_end])
            audit_lines.append(
                "\t".join(
                    [
                        stem,
                        f"{source_fps:.6f}",
                        f"{target_fps:.6f}",
                        str(start),
                        str(end),
                        str(mapped_start),
                        str(mapped_end),
                    ]
                )
            )
        rewritten[stem] = mapped_ranges

    dst_root.mkdir(parents=True, exist_ok=True)
    (dst_root / frame_ranges_name).write_text(json.dumps(rewritten, indent=2, sort_keys=True) + "\n")
    (dst_root / f"frame_ranges_{format_fps(target_fps)}hz_audit.tsv").write_text("\n".join(audit_lines) + "\n")
    return rewritten


def copy_dataset_sidecars(
    src_root: Path,
    dst_root: Path,
    *,
    sidecar_dirs: tuple[str, ...] = DEFAULT_SIDECAR_DIRS,
) -> list[str]:
    copied: list[str] = []
    dst_root.mkdir(parents=True, exist_ok=True)
    for name in sorted(sidecar_dirs):
        src = src_root / name
        if not src.is_dir():
            continue
        dst = dst_root / name
        shutil.copytree(src, dst, dirs_exist_ok=True)
        copied.append(name)
    return copied


def downsample_one_video(
    video_path: Path,
    *,
    src_video_root: Path,
    dst_video_root: Path,
    target_fps: float,
    fixed_source_fps: float,
    crf: int,
    preset: str,
    overwrite: bool,
    dry_run: bool,
    target_height: int = 0,
    target_width: int = 0,
) -> dict[str, Any]:
    relative_path = video_path.relative_to(src_video_root)
    output_path = dst_video_root / relative_path
    source_fps = fixed_source_fps if fixed_source_fps > 0 else probe_video_fps(video_path)
    command = build_ffmpeg_command(
        video_path,
        output_path,
        target_fps=target_fps,
        target_height=target_height,
        target_width=target_width,
        crf=crf,
        preset=preset,
        overwrite=overwrite,
    )
    if dry_run:
        status = "dry_run"
    elif output_path.exists() and not overwrite:
        status = "exists"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)
        status = "written"
    return {
        "stem": video_path.stem,
        "source": str(video_path),
        "output": str(output_path),
        "source_fps": source_fps,
        "target_fps": target_fps,
        "target_height": target_height,
        "target_width": target_width,
        "status": status,
        "command": command,
    }


def default_output_root(dataset_root: Path, target_fps: float, *, target_height: int = 0, target_width: int = 0) -> Path:
    suffix = f"_{format_fps(target_fps)}hz"
    if target_height > 0 or target_width > 0:
        if target_height <= 0 or target_width <= 0:
            raise ValueError("target_height and target_width must both be positive when resizing")
        suffix += f"_{target_height}x{target_width}"
    return dataset_root.with_name(f"{dataset_root.name}{suffix}")


def downsample_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve()
    output_root = (
        args.output_root
        or default_output_root(
            dataset_root,
            args.target_fps,
            target_height=args.target_height,
            target_width=args.target_width,
        )
    ).resolve()
    videos = discover_videos(
        dataset_root,
        args.video_subdir,
        max_videos=args.max_videos,
        recursive=args.recursive,
    )
    if not videos:
        raise FileNotFoundError(f"No video files found under {dataset_root / args.video_subdir}")

    if output_root == dataset_root:
        raise ValueError("output root must be different from dataset root")
    output_root.mkdir(parents=True, exist_ok=True)
    dst_video_root = output_root / args.video_subdir
    src_video_root = dataset_root / args.video_subdir

    copied_sidecars: list[str] = []
    if not args.no_copy_sidecars and not args.dry_run:
        copied_sidecars = copy_dataset_sidecars(dataset_root, output_root)

    results: list[dict[str, Any]] = []
    max_workers = max(1, int(args.num_workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {
            executor.submit(
                downsample_one_video,
                video_path,
                src_video_root=src_video_root,
                dst_video_root=dst_video_root,
                target_fps=args.target_fps,
                fixed_source_fps=args.source_fps,
                target_height=args.target_height,
                target_width=args.target_width,
                crf=args.crf,
                preset=args.preset,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            ): video_path
            for video_path in videos
        }
        for future in concurrent.futures.as_completed(future_to_video):
            result = future.result()
            results.append(result)
            print(
                f"{result['status']}: {Path(result['source']).name} "
                f"{result['source_fps']:.3f}fps -> {result['target_fps']:.3f}fps",
                flush=True,
            )
    results.sort(key=lambda item: item["source"])

    source_fps_by_stem = {item["stem"]: float(item["source_fps"]) for item in results}
    frame_ranges = None
    if not args.dry_run:
        frame_ranges = rewrite_frame_ranges(
            dataset_root,
            output_root,
            target_fps=args.target_fps,
            source_fps_by_stem=source_fps_by_stem,
            default_source_fps=args.source_fps,
        )
        (output_root / "source_fps_by_stem.json").write_text(
            json.dumps(source_fps_by_stem, indent=2, sort_keys=True) + "\n"
        )

    summary = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "video_subdir": args.video_subdir,
        "target_fps": args.target_fps,
        "target_height": args.target_height,
        "target_width": args.target_width,
        "recursive": bool(args.recursive),
        "num_videos": len(results),
        "copied_sidecars": copied_sidecars,
        "frame_ranges_rewritten": frame_ranges is not None,
        "dry_run": bool(args.dry_run),
        "results": results,
    }
    if not args.dry_run:
        (output_root / f"downsample_{format_fps(args.target_fps)}hz_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    return summary


def main() -> None:
    args = parse_args()
    summary = downsample_dataset(args)
    print(
        json.dumps(
            {
                "output_root": summary["output_root"],
                "num_videos": summary["num_videos"],
                "target_fps": summary["target_fps"],
                "dry_run": summary["dry_run"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
