#!/usr/bin/env python3
"""Build trajectory-safe TA-Tok and planner manifests for LIBERO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from qwen35_planx.config import CAMERA_KEYS, CAMERA_NAMES, PlanGeometry
from qwen35_planx.hashing import sha256_file, sha256_json
from qwen35_planx.libero_data import (
    discover_trajectories,
    iter_all_camera_frames,
    iter_planner_windows,
)


def _write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, object]],
) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
    return count


def build_manifests(
    *,
    dataset_roots: Sequence[Path | str],
    domains: Sequence[str],
    predecoded_root: Path | str,
    output_dir: Path | str,
    split_seed: int,
    window_stride: int = 10,
    max_windows_per_trajectory: int = 16,
) -> dict[str, object]:
    """Write sorted frame/window JSONL files and their compatibility hashes."""

    geometry = PlanGeometry()
    records = discover_trajectories(
        dataset_roots=dataset_roots,
        domains=domains,
        predecoded_root=predecoded_root,
        split_seed=split_seed,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_name: dict[str, list[dict[str, object]]] = {
        "trajectories.jsonl": [record.to_dict() for record in records],
        "ta_frames_train.jsonl": [],
        "ta_frames_val.jsonl": [],
        "planner_train.jsonl": [],
        "planner_val.jsonl": [],
    }
    for record in records:
        frame_name = f"ta_frames_{record.split}.jsonl"
        rows_by_name[frame_name].extend(
            frame.to_dict() for frame in iter_all_camera_frames(record)
        )
        planner_name = f"planner_{record.split}.jsonl"
        rows_by_name[planner_name].extend(
            window.to_dict()
            for window in iter_planner_windows(
                record,
                stride=window_stride,
                max_windows=max_windows_per_trajectory,
                geometry=geometry,
            )
        )

    file_metadata: dict[str, dict[str, object]] = {}
    for name in sorted(rows_by_name):
        path = output_dir / name
        count = _write_jsonl(path, rows_by_name[name])
        file_metadata[name] = {
            "records": count,
            "sha256": sha256_file(path),
        }

    contract = {
        "format_version": 1,
        "camera_names": list(CAMERA_NAMES),
        "camera_keys": list(CAMERA_KEYS),
        "num_keyframes": geometry.num_keyframes,
        "grid_size": geometry.grid_size,
        "visual_vocab_size": geometry.visual_vocab_size,
        "future_frame_offsets": list(geometry.future_frame_offsets),
        "ge_act_future_indices": list(geometry.ge_act_future_indices),
        "split_seed": split_seed,
        "window_stride": window_stride,
        "max_windows_per_trajectory": max_windows_per_trajectory,
        "domains": list(domains),
    }
    manifest: dict[str, object] = {
        **contract,
        "contract_hash": sha256_json(contract),
        "predecoded_root": str(Path(predecoded_root).resolve()),
        "dataset_roots": [
            str(Path(dataset_root).resolve()) for dataset_root in dataset_roots
        ],
        "files": file_metadata,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return manifest


def build_hdf5_manifests(
    *,
    hdf5_manifest: Path | str,
    output_dir: Path | str,
    split_seed: int,
    window_stride: int = 36,
    sample_n_frames: int = 500,
) -> dict[str, object]:
    """Write deterministic train/val windows for canonical GE-Act HDF5."""

    from qwen35_planx.hindsight_data import build_fixed_windows

    geometry = PlanGeometry()
    hdf5_manifest = Path(hdf5_manifest).resolve()
    windows = build_fixed_windows(
        hdf5_manifest,
        split_seed=split_seed,
        window_stride=window_stride,
        sample_n_frames=sample_n_frames,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_name: dict[str, list[dict[str, object]]] = {
        "hindsight_train.jsonl": [],
        "hindsight_val.jsonl": [],
    }
    for window in windows:
        rows_by_name[f"hindsight_{window.split}.jsonl"].append(window.to_dict())

    file_metadata: dict[str, dict[str, object]] = {}
    for name in sorted(rows_by_name):
        path = output_dir / name
        count = _write_jsonl(path, rows_by_name[name])
        file_metadata[name] = {
            "records": count,
            "sha256": sha256_file(path),
        }

    contract = {
        "format_version": 1,
        "camera_names": list(CAMERA_NAMES),
        "num_keyframes": geometry.num_keyframes,
        "ge_act_future_indices": list(geometry.ge_act_future_indices),
        "action_chunk": 36,
        "chunk": 9,
        "n_previous": 4,
        "video_temporal_stride": 4,
        "split_seed": split_seed,
        "window_stride": window_stride,
        "sample_n_frames": sample_n_frames,
    }
    contract_hash = sha256_json(contract)
    hdf5_manifest_hash = sha256_file(hdf5_manifest)
    window_manifest_hash = sha256_json([window.to_dict() for window in windows])
    manifest: dict[str, object] = {
        **contract,
        "contract_hash": contract_hash,
        "hdf5_manifest": str(hdf5_manifest),
        "hdf5_manifest_hash": hdf5_manifest_hash,
        "window_manifest_hash": window_manifest_hash,
        "files": file_metadata,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", action="append", type=Path)
    parser.add_argument("--domain", action="append")
    parser.add_argument("--predecoded-root", type=Path)
    parser.add_argument("--hdf5-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--window-stride", type=int)
    parser.add_argument("--sample-n-frames", type=int, default=500)
    parser.add_argument("--max-windows-per-trajectory", type=int, default=16)
    args = parser.parse_args()
    if args.hdf5_manifest is not None:
        if any(
            value is not None
            for value in (args.dataset_root, args.domain, args.predecoded_root)
        ):
            parser.error(
                "--hdf5-manifest cannot be combined with NPY dataset arguments"
            )
        return args
    if args.dataset_root is None or args.domain is None or args.predecoded_root is None:
        parser.error(
            "NPY mode requires --dataset-root, --domain, and --predecoded-root"
        )
    if len(args.dataset_root) != len(args.domain):
        parser.error("--dataset-root and --domain must appear the same number of times")
    return args


def main() -> int:
    args = _parse_args()
    if args.hdf5_manifest is not None:
        manifest = build_hdf5_manifests(
            hdf5_manifest=args.hdf5_manifest,
            output_dir=args.output_dir,
            split_seed=args.split_seed,
            window_stride=(36 if args.window_stride is None else args.window_stride),
            sample_n_frames=args.sample_n_frames,
        )
    else:
        manifest = build_manifests(
            dataset_roots=args.dataset_root,
            domains=args.domain,
            predecoded_root=args.predecoded_root,
            output_dir=args.output_dir,
            split_seed=args.split_seed,
            window_stride=(10 if args.window_stride is None else args.window_stride),
            max_windows_per_trajectory=args.max_windows_per_trajectory,
        )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "contract_hash": manifest["contract_hash"],
                "files": manifest["files"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
