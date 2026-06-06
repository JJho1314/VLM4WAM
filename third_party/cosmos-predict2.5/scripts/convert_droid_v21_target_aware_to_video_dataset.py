"""Convert DROID v2.1 videos with target masks into Cosmos VideoDataset format.

Input layout:
    <src>/
      meta/{info,episodes}.jsonl
      videos/chunk-XXX/observation.images.left_external/episode_NNNNNN.mp4
      masks/chunk-XXX/observation.images.left_external/episode_NNNNNN.npz

Output layout:
    <out>/
      videos/episode_NNNNNN_left_external.mp4
      metas/episode_NNNNNN_left_external.txt
      masks/episode_NNNNNN_left_external.npz
      frame_ranges.json

The output files are hardlinks when possible and symlinks otherwise, so this is
cheap on the same filesystem. Only camera streams with an existing mask are
kept, which is required for target-aware training.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path


PROMPT_TEMPLATE = "A Franka robotic arm with a parallel-jaw gripper {task}."
DEFAULT_VIDEO_KEYS = ("observation.images.left_external", "observation.images.right_external")


def normalize_task(task: str) -> str:
    task = task.strip().rstrip(".").strip()
    if not task:
        return "performs a manipulation task"
    return task[0].lower() + task[1:]


def link_or_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.link(src, dst)
    except OSError:
        dst.symlink_to(src)


def camera_suffix(video_key: str) -> str:
    return video_key.removeprefix("observation.images.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="DROID v2.1 root")
    parser.add_argument("--out", required=True, help="Cosmos VideoDataset output root")
    parser.add_argument("--mask-dir", default="masks", help="Mask directory under --src")
    parser.add_argument("--video-keys", nargs="+", default=list(DEFAULT_VIDEO_KEYS))
    parser.add_argument("--min-frames", type=int, default=33)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle episodes before applying --max-samples.")
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--overwrite-metadata", action="store_true")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    videos_out = out / "videos"
    metas_out = out / "metas"
    masks_out = out / "masks"
    videos_out.mkdir(parents=True, exist_ok=True)
    metas_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    info = json.loads((src / "meta" / "info.json").read_text())
    chunks_size = int(info["chunks_size"])
    video_path_tpl = info["video_path"]

    episodes = {}
    for line in (src / "meta" / "episodes.jsonl").read_text().splitlines():
        rec = json.loads(line)
        tasks = rec.get("tasks") or []
        task = tasks[0] if tasks else rec.get("task", "")
        keep_ranges = rec.get("keep_ranges") or [[0, int(rec["length"]) - 1]]
        episodes[int(rec["episode_index"])] = {
            "length": int(rec["length"]),
            "task": task,
            "keep_ranges": keep_ranges,
            "uuid": rec.get("uuid", ""),
        }

    rows = []
    frame_ranges = {}
    counts = {
        "kept": 0,
        "short": 0,
        "missing_video": 0,
        "missing_mask": 0,
        "range_too_short": 0,
    }

    episode_indices = sorted(episodes)
    if args.shuffle:
        random.Random(args.seed).shuffle(episode_indices)

    for episode_index in episode_indices:
        rec = episodes[episode_index]
        if rec["length"] < args.min_frames:
            counts["short"] += len(args.video_keys)
            continue
        valid_ranges = []
        for start, end in rec["keep_ranges"]:
            start = max(0, int(start))
            end = min(rec["length"] - 1, int(end))
            if end - start + 1 >= args.min_frames:
                valid_ranges.append([start, end])
        if not valid_ranges:
            counts["range_too_short"] += len(args.video_keys)
            continue

        chunk = episode_index // chunks_size
        for video_key in args.video_keys:
            suffix = camera_suffix(video_key)
            name = f"episode_{episode_index:06d}_{suffix}"
            rel_video = video_path_tpl.format(
                episode_chunk=chunk,
                video_key=video_key,
                episode_index=episode_index,
            )
            src_video = src / rel_video
            src_mask = (
                src
                / args.mask_dir
                / f"chunk-{chunk:03d}"
                / video_key
                / f"episode_{episode_index:06d}.npz"
            )

            if not src_video.exists():
                counts["missing_video"] += 1
                continue
            if not src_mask.exists():
                counts["missing_mask"] += 1
                continue

            link_or_symlink(src_video, videos_out / f"{name}.mp4")
            link_or_symlink(src_mask, masks_out / f"{name}.npz")
            prompt = PROMPT_TEMPLATE.format(task=normalize_task(rec["task"]))
            meta_path = metas_out / f"{name}.txt"
            if args.overwrite_metadata or not meta_path.exists():
                meta_path.write_text(prompt + "\n")

            frame_ranges[name] = valid_ranges
            rows.append(
                {
                    "name": name,
                    "episode_index": episode_index,
                    "camera": suffix,
                    "length": rec["length"],
                    "task_orig": rec["task"],
                    "uuid": rec["uuid"],
                    "source_video": str(src_video),
                    "source_mask": str(src_mask),
                    "keep_ranges": json.dumps(valid_ranges),
                }
            )
            counts["kept"] += 1
            if args.max_samples > 0 and counts["kept"] >= args.max_samples:
                break
        if args.max_samples > 0 and counts["kept"] >= args.max_samples:
            break

    with open(out / "metadata.csv", "w", newline="") as f:
        fieldnames = [
            "name",
            "episode_index",
            "camera",
            "length",
            "task_orig",
            "uuid",
            "source_video",
            "source_mask",
            "keep_ranges",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out / "frame_ranges.json").write_text(json.dumps(frame_ranges, indent=2, sort_keys=True) + "\n")
    (out / "convert_summary.json").write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"out": str(out), **counts}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
