#!/usr/bin/env python3
"""Create train/val symlink splits for a flat Cosmos VideoDataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


EPISODE_RE = re.compile(r"episode_(\d+)")


def episode_key(name: str, row: dict[str, str] | None) -> str:
    if row and row.get("episode_index", "") != "":
        return row["episode_index"]
    match = EPISODE_RE.search(name)
    return match.group(1) if match else name


def safe_reset(path: Path, overwrite: bool) -> None:
    if path.exists() or path.is_symlink():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def symlink_file(src: Path, dst: Path) -> None:
    if not src.exists() and not src.is_symlink():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def read_metadata(src: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    metadata_path = src / "metadata.csv"
    if not metadata_path.exists():
        rows = [{"name": path.stem} for path in sorted((src / "videos").glob("*.mp4"))]
        return rows, {row["name"]: row for row in rows}
    with metadata_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows, {row["name"]: row for row in rows}


def write_metadata(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def materialize_split(src: Path, dst: Path, names: list[str], rows_by_name: dict[str, dict[str, str]], overwrite: bool) -> None:
    safe_reset(dst, overwrite)
    for dirname in ("videos", "masks", "target_masks", "metas"):
        src_dir = src / dirname
        if src_dir.exists():
            (dst / dirname).mkdir(parents=True, exist_ok=True)

    for name in names:
        symlink_file(src / "videos" / f"{name}.mp4", dst / "videos" / f"{name}.mp4")
        for mask_dirname in ("masks", "target_masks"):
            mask = src / mask_dirname / f"{name}.npz"
            if mask.exists() or mask.is_symlink():
                symlink_file(mask, dst / mask_dirname / f"{name}.npz")
        meta = src / "metas" / f"{name}.txt"
        if meta.exists() or meta.is_symlink():
            symlink_file(meta, dst / "metas" / f"{name}.txt")

    rows = [rows_by_name[name] for name in names if name in rows_by_name]
    if rows:
        write_metadata(dst / "metadata.csv", rows, list(rows[0].keys()))

    frame_ranges_path = src / "frame_ranges.json"
    if frame_ranges_path.exists():
        frame_ranges = json.loads(frame_ranges_path.read_text())
        filtered = {name: frame_ranges[name] for name in names if name in frame_ranges}
        (dst / "frame_ranges.json").write_text(json.dumps(filtered, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--val-out", required=True)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    train_out = Path(args.train_out).resolve()
    val_out = Path(args.val_out).resolve()

    rows, rows_by_name = read_metadata(src)
    names = sorted(path.stem for path in (src / "videos").glob("*.mp4"))
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        groups[episode_key(name, rows_by_name.get(name))].append(name)

    group_items = list(groups.items())
    random.Random(args.seed).shuffle(group_items)
    val_names_set: set[str] = set()
    for _, group_names in group_items:
        if len(val_names_set) >= args.val_samples:
            break
        val_names_set.update(group_names)

    val_names = [name for name in names if name in val_names_set]
    train_names = [name for name in names if name not in val_names_set]

    materialize_split(src, train_out, train_names, rows_by_name, args.overwrite)
    materialize_split(src, val_out, val_names, rows_by_name, args.overwrite)

    summary = {
        "src": str(src),
        "train_out": str(train_out),
        "val_out": str(val_out),
        "total_samples": len(names),
        "train_samples": len(train_names),
        "val_samples": len(val_names),
        "val_target_samples": args.val_samples,
        "seed": args.seed,
        "split_by": "episode_index",
    }
    (train_out / "split_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (val_out / "split_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
