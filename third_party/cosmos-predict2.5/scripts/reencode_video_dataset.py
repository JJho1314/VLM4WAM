#!/usr/bin/env python3
"""Re-encode a Cosmos VideoDataset videos/ directory and symlink sidecar files."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def find_ffmpeg() -> str:
    candidates = [
        os.environ.get("FFMPEG"),
        shutil.which("ffmpeg"),
        "/data/apps/ffmpeg/7.0.2/ffmpeg",
        "/data/apps/ffmpeg/6.0.1/ffmpeg",
        "/data/apps/ffmpeg/5.1.1/ffmpeg",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise FileNotFoundError("ffmpeg not found; set FFMPEG=/path/to/ffmpeg")


def relink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def reencode_one(
    ffmpeg: str,
    src_video: Path,
    dst_video: Path,
    timeout: int,
    fps: int,
    height: int | None,
    width: int | None,
) -> tuple[str, str, float]:
    if dst_video.exists() and dst_video.stat().st_size > 0:
        return src_video.name, "skip", 0.0

    tmp_video = dst_video.with_suffix(".tmp.mp4")
    if tmp_video.exists():
        tmp_video.unlink()

    filters = [f"fps={fps}"]
    if height is not None and width is not None:
        filters.append(f"scale={width}:{height}:flags=bicubic")
    filters.append("format=yuv420p")

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src_video),
        "-vf",
        ",".join(filters),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-an",
        "-movflags",
        "+faststart",
        str(tmp_video),
    ]
    start = time.time()
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
        tmp_video.replace(dst_video)
        return src_video.name, "ok", time.time() - start
    except subprocess.CalledProcessError as exc:
        if tmp_video.exists():
            tmp_video.unlink()
        err = exc.stderr.decode(errors="replace")[:300].replace("\n", " ")
        return src_video.name, f"error:{err}", time.time() - start
    except subprocess.TimeoutExpired:
        if tmp_video.exists():
            tmp_video.unlink()
        return src_video.name, "timeout", time.time() - start


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--status-csv", required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    args = parser.parse_args()
    if (args.height is None) != (args.width is None):
        parser.error("--height and --width must be set together")

    src = Path(args.src)
    out = Path(args.out)
    src_videos = src / "videos"
    src_metas = src / "metas"
    out_videos = out / "videos"
    out_metas = out / "metas"
    out_videos.mkdir(parents=True, exist_ok=True)
    out_metas.mkdir(parents=True, exist_ok=True)

    ffmpeg = find_ffmpeg()
    print(f"using ffmpeg: {ffmpeg}", flush=True)

    videos = sorted(src_videos.glob("*.mp4"))
    for meta in sorted(src_metas.glob("*.txt")):
        relink(meta, out_metas / meta.name)
    metadata = src / "metadata.csv"
    if metadata.exists():
        relink(metadata, out / "metadata.csv")
    frame_ranges = src / "frame_ranges.json"
    if frame_ranges.exists():
        relink(frame_ranges, out / "frame_ranges.json")
    for mask_dirname in ("masks", "target_masks"):
        src_masks = src / mask_dirname
        if src_masks.exists():
            out_masks = out / mask_dirname
            out_masks.mkdir(parents=True, exist_ok=True)
            for mask in sorted(src_masks.glob("*.npz")):
                relink(mask, out_masks / mask.name)

    status_path = Path(args.status_csv)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    done_names: set[str] = set()
    if status_path.exists():
        with status_path.open() as f:
            for row in csv.DictReader(f):
                if row.get("status") in {"ok", "skip"}:
                    done_names.add(row["name"])

    to_process = [video for video in videos if video.name not in done_names]
    print(f"videos={len(videos)} already_done={len(done_names)} to_process={len(to_process)}", flush=True)

    write_header = not status_path.exists()
    with status_path.open("a", newline="") as status_file:
        writer = csv.DictWriter(status_file, fieldnames=["name", "status", "seconds"])
        if write_header:
            writer.writeheader()

        ok = skip = bad = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    reencode_one,
                    ffmpeg,
                    video,
                    out_videos / video.name,
                    args.timeout_sec,
                    args.fps,
                    args.height,
                    args.width,
                )
                for video in to_process
            ]
            for i, future in enumerate(as_completed(futures), 1):
                name, status, seconds = future.result()
                writer.writerow({"name": name, "status": status, "seconds": f"{seconds:.3f}"})
                status_file.flush()
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skip += 1
                else:
                    bad += 1
                    print(f"BAD {name} {status}", flush=True)
                if i % 100 == 0:
                    print(f"progress {i}/{len(to_process)} ok={ok} skip={skip} bad={bad}", flush=True)

    print(f"DONE out={out} videos={len(videos)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
