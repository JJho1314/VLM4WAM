"""Link RoboInter OXE-DROID SAM masks into Cosmos VideoDataset target_masks.

The converted Cosmos DROID datasets use names like ``episode_031626.mp4``.
RoboInter stores the corresponding SAM masks under names from the annotation
parquet ``episode_name`` field, e.g. ``27106_exterior_image_1_left.npz``.
This script creates ``target_masks/episode_031626.npz`` symlinks for every
video whose annotation and SAM mask are available.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def build_mask_index(seg_root: Path) -> dict[str, Path]:
    index = {}
    for path in seg_root.glob("*/sam_mask/*.npz"):
        index.setdefault(path.name, path.resolve())
    return index


def link_one(
    video_path: Path,
    annotation_root: Path,
    mask_index: dict[str, Path],
    target_dir: Path,
) -> tuple[str, str]:
    try:
        episode_id = int(video_path.stem.split("_", 1)[1])
    except Exception:
        return video_path.stem, "bad_video_name"

    annotation_path = annotation_root / f"chunk-{episode_id // 1000:03d}" / f"episode_{episode_id:06d}.parquet"
    if not annotation_path.exists():
        return video_path.stem, "missing_annotation"

    try:
        df = pd.read_parquet(annotation_path, columns=["episode_name"])
    except Exception as exc:
        return video_path.stem, f"bad_annotation:{type(exc).__name__}"

    episode_names = df["episode_name"].dropna().unique().tolist()
    if not episode_names:
        return video_path.stem, "missing_episode_name"

    mask_path = mask_index.get(f"{episode_names[0]}.npz")
    if mask_path is None:
        return video_path.stem, "missing_mask"

    dst = target_dir / f"{video_path.stem}.npz"
    if dst.exists() or dst.is_symlink():
        return video_path.stem, "exists"
    dst.symlink_to(mask_path)
    return video_path.stem, "linked"


def process_dataset(dataset_dir: Path, annotation_root: Path, seg_root: Path, workers: int) -> dict[str, int]:
    video_dir = dataset_dir / "videos"
    target_dir = dataset_dir / "target_masks"
    target_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(video_dir.glob("episode_*.mp4"))
    mask_index = build_mask_index(seg_root)
    print(f"{dataset_dir}: indexed {len(mask_index)} masks from {seg_root}", flush=True)

    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(link_one, video, annotation_root, mask_index, target_dir) for video in videos]
        for idx, fut in enumerate(as_completed(futures), start=1):
            _, status = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if idx % 5000 == 0:
                print(f"{dataset_dir}: processed {idx}/{len(videos)} {counts}", flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--seg-root", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("dataset_dirs", nargs="+")
    args = parser.parse_args()

    annotation_root = Path(args.annotation_root).resolve()
    seg_root = Path(args.seg_root).resolve()
    all_counts = {}
    for dataset in args.dataset_dirs:
        dataset_dir = Path(dataset).resolve()
        counts = process_dataset(dataset_dir, annotation_root, seg_root, args.workers)
        all_counts[str(dataset_dir)] = counts
        summary_path = dataset_dir / "target_masks_link_summary.json"
        summary_path.write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n")
        print(f"{dataset_dir}: {json.dumps(counts, sort_keys=True)}", flush=True)
    print(json.dumps(all_counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
