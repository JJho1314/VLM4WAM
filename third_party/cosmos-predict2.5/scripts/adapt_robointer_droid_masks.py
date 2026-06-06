"""Adapt RoboInter DROID SAM masks for Cosmos VideoDataset training videos.

RoboInter masks are named with the original DROID episode id and camera view,
for example ``31626_exterior_image_2_left.npz``. The Cosmos training videos are
named ``episode_031626.mp4`` and may have a different FPS, frame count, and
resolution. This script writes frame-aligned ``target_masks/episode_031626.npz``
files by resampling the mask timeline to the training video frame count.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


def build_mask_index(seg_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in seg_root.glob("*/sam_mask/*.npz"):
        index.setdefault(path.name, path.resolve())
    return index


def read_video_shape(video_path: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("cannot_open_video")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError("bad_video_metadata")
    return frame_count, height, width


def load_mask(mask_path: Path) -> np.ndarray:
    mask_npz = np.load(mask_path, allow_pickle=False)
    key = "masks" if "masks" in mask_npz.files else mask_npz.files[0]
    mask = mask_npz[key]
    if mask.ndim == 5:
        mask = mask.max(axis=0)
    if mask.ndim == 4:
        if mask.shape[1] == 1:
            mask = mask[:, 0]
        elif mask.shape[0] == 1:
            mask = mask[0]
        else:
            mask = mask.max(axis=0)
    if mask.ndim != 3:
        raise ValueError(f"bad_mask_shape:{mask.shape}")
    return mask.astype(np.uint8)


def adapt_mask(mask: np.ndarray, frame_count: int, height: int, width: int) -> np.ndarray:
    src_t = mask.shape[0]
    if src_t <= 0:
        raise ValueError("empty_mask")
    if frame_count == 1:
        frame_ids = np.array([0], dtype=np.int64)
    else:
        frame_ids = np.rint(np.linspace(0, src_t - 1, frame_count)).astype(np.int64)
    mask = mask[np.clip(frame_ids, 0, src_t - 1)]
    resized = np.empty((frame_count, height, width), dtype=np.uint8)
    for idx, frame in enumerate(mask):
        resized[idx] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def adapt_one(
    video_path: Path,
    mask_index: dict[str, Path],
    output_dir: Path,
    camera_view: str,
    overwrite: bool,
) -> tuple[str, str]:
    try:
        episode_id = int(video_path.stem.split("_", 1)[1])
    except Exception:
        return video_path.name, "bad_video_name"

    dst = output_dir / f"{video_path.stem}.npz"
    if dst.exists() and not overwrite:
        return video_path.name, "exists"

    mask_path = mask_index.get(f"{episode_id}_{camera_view}.npz")
    if mask_path is None:
        return video_path.name, "missing_mask"

    try:
        frame_count, height, width = read_video_shape(video_path)
        mask = load_mask(mask_path)
        adapted = adapt_mask(mask, frame_count, height, width)
        np.savez_compressed(
            dst,
            masks=adapted,
            source_mask=str(mask_path),
            source_camera_view=camera_view,
            source_num_frames=np.array(mask.shape[0], dtype=np.int32),
            video_num_frames=np.array(frame_count, dtype=np.int32),
            video_height=np.array(height, dtype=np.int32),
            video_width=np.array(width, dtype=np.int32),
        )
    except Exception as exc:
        return video_path.name, f"error:{type(exc).__name__}:{exc}"
    return video_path.name, "adapted"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--seg-root", required=True)
    parser.add_argument("--camera-view", required=True, help="Example: exterior_image_1_left")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--episode-ids", default="", help="Comma-separated original DROID episode ids to process first.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    video_dir = dataset_dir / "videos"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else dataset_dir / "target_masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.episode_ids:
        episode_ids = [int(item) for item in args.episode_ids.split(",") if item.strip()]
        videos = [video_dir / f"episode_{episode_id:06d}.mp4" for episode_id in episode_ids]
        videos = [video for video in videos if video.exists()]
    else:
        videos = sorted(video_dir.glob("episode_*.mp4"))
    if args.limit > 0 and not args.episode_ids:
        videos = videos[: args.limit]
    mask_index = build_mask_index(Path(args.seg_root).resolve())
    print(f"indexed {len(mask_index)} masks; processing {len(videos)} videos", flush=True)

    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(adapt_one, video, mask_index, output_dir, args.camera_view, args.overwrite)
            for video in videos
        ]
        for idx, fut in enumerate(as_completed(futures), start=1):
            _, status = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if idx % 1000 == 0:
                print(f"processed {idx}/{len(videos)} {counts}", flush=True)

    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "camera_view": args.camera_view,
        "counts": counts,
    }
    (output_dir / "adapt_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
