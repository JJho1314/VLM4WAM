#!/usr/bin/env python3
"""Build a one-sample-per-scene target-aware eval subset.

The script leaves the source validation split untouched. It creates a new
dataset root with symlinks for videos/masks and local metadata/captions so later
feature precomputation and visualization outputs are isolated.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

from rewrite_droid_tgt_captions import rewrite_caption


def scene_key(uuid: str) -> str:
    parts = str(uuid).split("+")
    return "+".join(parts[:2]) if len(parts) >= 2 else str(uuid)


def choose_rows(rows: list[dict[str, str]], prefer_camera: str) -> list[dict[str, str]]:
    by_scene: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_scene.setdefault(scene_key(row.get("uuid", "")), []).append(row)

    chosen = []
    for scene in sorted(by_scene):
        candidates = by_scene[scene]
        candidates = sorted(
            candidates,
            key=lambda row: (
                0 if row.get("camera") == prefer_camera else 1,
                -int(row.get("length") or 0),
                row.get("name", ""),
            ),
        )
        picked = dict(candidates[0])
        picked["scene_key"] = scene
        chosen.append(picked)
    return chosen


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefer-camera", default="left_external")
    parser.add_argument("--copy-files", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    metadata_path = source_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for dirname in ("videos", "masks", "metas"):
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)

    with metadata_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    frame_ranges = read_json(source_dir / "frame_ranges.json")
    chosen = choose_rows(rows, prefer_camera=args.prefer_camera)
    out_ranges = {}
    out_rows = []
    missing = []
    for row in chosen:
        stem = row["name"]
        video_src = source_dir / "videos" / f"{stem}.mp4"
        mask_src = source_dir / "masks" / f"{stem}.npz"
        if not video_src.exists() or not mask_src.exists():
            missing.append(stem)
            continue

        link_or_copy(video_src, output_dir / "videos" / video_src.name, args.copy_files)
        link_or_copy(mask_src, output_dir / "masks" / mask_src.name, args.copy_files)

        task = row.get("task_orig", "").strip().rstrip(".")
        task = task[:1].lower() + task[1:] if task else "performs a manipulation task"
        caption = f"A Franka robotic arm with a parallel-jaw gripper {task}."
        rewritten, status, verb, phrase = rewrite_caption(caption)
        (output_dir / "metas" / f"{stem}.txt").write_text(rewritten.rstrip() + "\n")

        out_row = dict(row)
        out_row.update(
            {
                "scene_key": row["scene_key"],
                "target_caption": rewritten,
                "rewrite_status": status,
                "rewrite_verb": verb,
                "rewrite_object_phrase": phrase,
            }
        )
        out_rows.append(out_row)
        if stem in frame_ranges:
            out_ranges[stem] = frame_ranges[stem]

    if missing:
        raise RuntimeError(f"Missing source video/mask for {len(missing)} chosen rows: {missing[:10]}")

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with (output_dir / "metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    (output_dir / "frame_ranges.json").write_text(json.dumps(out_ranges, indent=2) + "\n")
    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "num_source_rows": len(rows),
        "num_scenes": len(chosen),
        "num_written": len(out_rows),
        "prefer_camera": args.prefer_camera,
        "copy_files": bool(args.copy_files),
        "scenes": [
            {
                "scene_key": row["scene_key"],
                "name": row["name"],
                "camera": row.get("camera", ""),
                "length": int(row.get("length") or 0),
                "task_orig": row.get("task_orig", ""),
                "target_caption": row.get("target_caption", ""),
                "rewrite_object_phrase": row.get("rewrite_object_phrase", ""),
            }
            for row in out_rows
        ],
    }
    (output_dir / "one_per_scene_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
