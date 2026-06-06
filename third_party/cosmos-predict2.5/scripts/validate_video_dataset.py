"""Validate mp4 files in a Cosmos VideoDataset directory.

The validator treats missing video streams, ffprobe failures, and very small
files as invalid. With --delete-bad, invalid files are removed so a later
conversion pass can regenerate them instead of skipping them as existing files.
"""
from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def check_video(args: tuple[Path, str, int]) -> tuple[Path, bool, str]:
    video, ffprobe, min_bytes = args
    try:
        size = video.stat().st_size
    except OSError as exc:
        return video, False, f"stat: {exc}"
    if size < min_bytes:
        return video, False, f"small:{size}"

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        str(video),
    ]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=20)
    except Exception as exc:
        return video, False, f"ffprobe: {exc!r}"
    if result.returncode != 0:
        return video, False, f"ffprobe: {result.stderr.strip()[:200]}"
    if not result.stdout.strip():
        return video, False, "no-video-stream"
    return video, True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--min-bytes", type=int, default=10_000)
    parser.add_argument("--bad-list", default=None)
    parser.add_argument("--delete-bad", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    videos = sorted((dataset / "videos").glob("*.mp4"))
    bad: list[tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(check_video, (video, args.ffprobe, args.min_bytes)) for video in videos]
        for future in as_completed(futures):
            video, ok, reason = future.result()
            if not ok:
                bad.append((video, reason))

    bad.sort(key=lambda item: str(item[0]))
    if args.bad_list:
        Path(args.bad_list).write_text(
            "".join(f"{path}\t{reason.replace(chr(10), ' | ')}\n" for path, reason in bad)
        )

    deleted = 0
    if args.delete_bad:
        for path, _ in bad:
            try:
                path.unlink()
                deleted += 1
            except FileNotFoundError:
                pass

    print(
        f"dataset={dataset} total={len(videos)} ok={len(videos) - len(bad)} "
        f"bad={len(bad)} deleted={deleted}"
    )
    for path, reason in bad[:20]:
        print(f"bad\t{path}\t{reason}")


if __name__ == "__main__":
    main()
