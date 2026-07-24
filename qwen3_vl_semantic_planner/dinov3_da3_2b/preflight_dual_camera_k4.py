#!/usr/bin/env python3
"""Fail-closed CPU/data preflight for the OLA dual-camera K4 planner run."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from qwen3_vl_semantic_planner.train_semantic_planner import (
    load_ge_act_dual_camera_planner_dataset,
)


OFFSETS = (2, 4, 6, 8)
CAMERAS = ("main", "wrist")
PLANNER_TOKEN_COUNT = 4 * (32 + 32 + 32)


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory is missing: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--siglip2-model-dir", type=Path, required=True)
    parser.add_argument("--da3-ckpt-dir", type=Path, required=True)
    parser.add_argument("--da3-code-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, label in (
        (args.model_path, "Qwen base model"),
        (args.siglip2_model_dir, "SigLIP2 teacher"),
        (args.da3_ckpt_dir, "DA3 checkpoint"),
        (args.da3_code_root, "DA3 source"),
    ):
        require_directory(path, label)
    da3_api = args.da3_code_root / "src" / "depth_anything_3" / "api.py"
    if not da3_api.is_file():
        raise FileNotFoundError(f"DA3 API is missing: {da3_api}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(args.output_dir).free
    required_bytes = int(args.minimum_free_gib) * 1024**3
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"output filesystem has {free_bytes / 1024**3:.1f} GiB free; "
            f"requires at least {args.minimum_free_gib} GiB"
        )

    dataset = load_ge_act_dual_camera_planner_dataset(
        args.config,
        future_offsets=OFFSETS,
    )
    if len(dataset) < 1:
        raise RuntimeError("GE-Act planner dataset is empty")
    source_sample = dataset.dataset[0]
    video = source_sample.get("video")
    if video is None or video.ndim != 5 or tuple(video.shape[:2]) != (3, 2):
        shape = tuple(video.shape) if video is not None else None
        raise RuntimeError(f"source video must be [3,2,T,H,W], got {shape}")
    sample = dataset[0]
    current = sample["current_camera_images"]
    future = sample["future_camera_images"]
    if current.ndim != 4 or tuple(current.shape[:1]) != (2,):
        raise RuntimeError(f"current frames must be [2,H,W,3], got {tuple(current.shape)}")
    if future.ndim != 5 or tuple(future.shape[:2]) != (2, 4):
        raise RuntimeError(
            f"future frames must be [2,4,H,W,3], got {tuple(future.shape)}"
        )
    if len(sample["images"]) != 2 or PLANNER_TOKEN_COUNT != 384:
        raise RuntimeError("dual-camera K4 planner geometry is inconsistent")

    print(
        json.dumps(
            {
                "status": "ok",
                "samples": len(dataset),
                "source_video_shape": list(video.shape),
                "camera_names": list(CAMERAS),
                "future_keyframe_offsets": list(OFFSETS),
                "planner_token_count": PLANNER_TOKEN_COUNT,
                "free_gib": round(free_bytes / 1024**3, 1),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
