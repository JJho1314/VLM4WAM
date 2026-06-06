#!/usr/bin/env python3
"""Create a filtered Cosmos VideoDataset from a decode-scan JSONL file."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def relink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scan-jsonl", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    scan_jsonl = Path(args.scan_jsonl)

    src_videos = src / "videos"
    src_metas = src / "metas"
    out_videos = out / "videos"
    out_metas = out / "metas"

    video_paths = sorted(src_videos.glob("*.mp4"))
    scan_rows = []
    bad_names: set[str] = set()
    with scan_jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            scan_rows.append(row)
            if row.get("status") != "ok":
                bad_names.add(Path(row["path"]).name)

    if args.require_complete and len(scan_rows) != len(video_paths):
        raise RuntimeError(f"scan is incomplete: {len(scan_rows)} rows for {len(video_paths)} videos")

    if out.exists() and not out.is_symlink():
        shutil.rmtree(out)
    out_videos.mkdir(parents=True, exist_ok=True)
    out_metas.mkdir(parents=True, exist_ok=True)

    kept = skipped = missing_meta = 0
    for video in video_paths:
        if video.name in bad_names:
            skipped += 1
            continue
        stem = video.stem
        meta = src_metas / f"{stem}.txt"
        if not meta.exists():
            missing_meta += 1
            skipped += 1
            continue
        relink(video, out_videos / video.name)
        relink(meta, out_metas / meta.name)
        kept += 1

    metadata = src / "metadata.csv"
    if metadata.exists():
        relink(metadata, out / "metadata.csv")

    bad_list = out / "excluded_decode_failures.txt"
    bad_list.write_text("\n".join(sorted(bad_names)) + ("\n" if bad_names else ""))

    print(
        f"src={src}\nout={out}\nscan_rows={len(scan_rows)} total_videos={len(video_paths)} "
        f"kept={kept} skipped={skipped} bad={len(bad_names)} missing_meta={missing_meta}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
