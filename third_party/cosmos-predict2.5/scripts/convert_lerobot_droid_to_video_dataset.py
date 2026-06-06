"""Convert LeRobot DROID v2.1 layout into the format Cosmos VideoDataset expects.

Cosmos VideoDataset wants:
    <out_dir>/
      videos/episode_NNNNNN.mp4
      metas/episode_NNNNNN.txt   # caption / prompt

LeRobot layout (input):
    <src>/
      meta/{episodes,tasks}.jsonl
      videos/chunk-XXX/observation.images.primary/episode_NNNNNN.mp4

For each episode we hardlink the primary-camera mp4 (no copy) and write a
prompt.txt derived from the task instruction, wrapped in an arm-subject
template that matches the v2 prompt convention used during inference.

Episodes shorter than --min-frames are dropped (Cosmos crashes if a clip has
fewer frames than the configured num_frames).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


PROMPT_TEMPLATE = "A Franka robotic arm with a parallel-jaw gripper {task}."


def normalize_task(task: str) -> str:
    task = task.strip().rstrip(".").strip()
    if not task:
        return "performs a manipulation task"
    return task[0].lower() + task[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/data/user/jhe724/workspace/data/lerobot_droid_anno")
    ap.add_argument("--out", required=True, help="Output dataset dir")
    ap.add_argument(
        "--max-episodes", type=int, default=None, help="Cap episodes (for sanity / staged training)"
    )
    ap.add_argument(
        "--min-frames",
        type=int,
        default=33,
        help="Drop episodes shorter than this (must be >= cosmos num_frames)",
    )
    ap.add_argument(
        "--video-key", default="observation.images.primary", help="Which lerobot camera stream to use"
    )
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "metas").mkdir(parents=True, exist_ok=True)

    info = json.loads((src / "meta" / "info.json").read_text())
    chunks_size = info["chunks_size"]
    video_path_tpl = info["video_path"]  # e.g. videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4

    # episode_index -> (length, task)
    episodes = {}
    for line in (src / "meta" / "episodes.jsonl").read_text().splitlines():
        rec = json.loads(line)
        episodes[rec["episode_index"]] = (rec["length"], rec["tasks"][0] if rec["tasks"] else "")

    csv_rows = []
    n_kept = 0
    n_short = 0
    n_missing = 0
    for ep_idx in sorted(episodes.keys()):
        if args.max_episodes is not None and n_kept >= args.max_episodes:
            break
        length, task = episodes[ep_idx]
        if length < args.min_frames:
            n_short += 1
            continue
        chunk = ep_idx // chunks_size
        rel_video = video_path_tpl.format(
            episode_chunk=chunk, video_key=args.video_key, episode_index=ep_idx
        )
        src_video = src / rel_video
        if not src_video.exists():
            n_missing += 1
            continue

        name = f"episode_{ep_idx:06d}"
        dst_video = out / "videos" / f"{name}.mp4"
        if not dst_video.exists():
            try:
                os.link(src_video, dst_video)
            except OSError:
                # cross-device or permission: fall back to symlink
                dst_video.symlink_to(src_video)

        prompt = PROMPT_TEMPLATE.format(task=normalize_task(task))
        (out / "metas" / f"{name}.txt").write_text(prompt + "\n")
        csv_rows.append((name, length, task))
        n_kept += 1

    # metadata.csv (optional but conventional)
    with open(out / "metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "length", "task_orig"])
        w.writerows(csv_rows)

    print(f"kept: {n_kept}  short(<{args.min_frames}): {n_short}  missing: {n_missing}")
    print(f"wrote videos/, metas/, metadata.csv to {out}")


if __name__ == "__main__":
    main()
