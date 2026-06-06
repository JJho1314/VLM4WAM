"""Create a flat Cosmos VideoDataset containing only samples with target masks.

This is useful when a larger VideoDataset has videos for every episode but
target masks only for a subset. The output links videos, metas, and masks from
one or more source datasets into:

    <out>/videos
    <out>/metas
    <out>/masks

Optionally pass a LeRobot v2.1 episodes.jsonl file to propagate keep_ranges into
<out>/frame_ranges.json, so the dataloader samples motion-heavy windows.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path


EPISODE_RE = re.compile(r"episode_(\d+)")


def link_or_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.link(src, dst)
    except OSError:
        dst.symlink_to(src)


def load_v21_ranges(path: str | None, min_frames: int) -> dict[int, list[list[int]]]:
    if not path:
        return {}
    ranges_by_episode: dict[int, list[list[int]]] = {}
    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line)
            length = int(rec["length"])
            valid = []
            for start, end in rec.get("keep_ranges") or [[0, length - 1]]:
                start = max(0, int(start))
                end = min(length - 1, int(end))
                if end - start + 1 >= min_frames:
                    valid.append([start, end])
            if valid:
                ranges_by_episode[int(rec["episode_index"])] = valid
    return ranges_by_episode


def episode_index_from_name(name: str) -> int | None:
    match = EPISODE_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", help="Source Cosmos VideoDataset dirs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--v21-episodes-jsonl", default=None)
    parser.add_argument("--min-frames", type=int, default=33)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out).resolve()
    videos_out = out / "videos"
    metas_out = out / "metas"
    masks_out = out / "masks"
    videos_out.mkdir(parents=True, exist_ok=True)
    metas_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    ranges_by_episode = load_v21_ranges(args.v21_episodes_jsonl, args.min_frames)
    frame_ranges = {}
    rows = []
    counts = {"kept": 0, "missing_mask": 0, "missing_meta": 0, "missing_range": 0, "duplicate": 0}

    for src_arg in args.sources:
        src = Path(src_arg).resolve()
        for video in sorted((src / "videos").glob("*.mp4")):
            name = video.stem
            if (videos_out / video.name).exists() or (videos_out / video.name).is_symlink():
                counts["duplicate"] += 1
                continue
            mask = None
            for mask_dir in ("masks", "target_masks"):
                candidate = src / mask_dir / f"{name}.npz"
                if candidate.exists() or candidate.is_symlink():
                    mask = candidate
                    break
            if mask is None:
                counts["missing_mask"] += 1
                continue
            meta = src / "metas" / f"{name}.txt"
            if not meta.exists():
                counts["missing_meta"] += 1
                continue

            episode_index = episode_index_from_name(name)
            if ranges_by_episode:
                if episode_index is None or episode_index not in ranges_by_episode:
                    counts["missing_range"] += 1
                    continue
                frame_ranges[name] = ranges_by_episode[episode_index]

            link_or_symlink(video, videos_out / video.name)
            link_or_symlink(mask, masks_out / f"{name}.npz")
            link_or_symlink(meta, metas_out / f"{name}.txt")
            rows.append(
                {
                    "name": name,
                    "source_dataset": str(src),
                    "source_video": str(video),
                    "source_mask": str(mask),
                    "episode_index": episode_index if episode_index is not None else "",
                }
            )
            counts["kept"] += 1
            if args.max_samples > 0 and counts["kept"] >= args.max_samples:
                break
        if args.max_samples > 0 and counts["kept"] >= args.max_samples:
            break

    with open(out / "metadata.csv", "w", newline="") as f:
        fieldnames = ["name", "episode_index", "source_dataset", "source_video", "source_mask"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if frame_ranges:
        (out / "frame_ranges.json").write_text(json.dumps(frame_ranges, indent=2, sort_keys=True) + "\n")
    (out / "filter_summary.json").write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), **counts}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
