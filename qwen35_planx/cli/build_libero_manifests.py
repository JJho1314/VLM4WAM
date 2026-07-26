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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", action="append", type=Path, required=True)
    parser.add_argument("--domain", action="append", required=True)
    parser.add_argument("--predecoded-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--window-stride", type=int, default=10)
    parser.add_argument("--max-windows-per-trajectory", type=int, default=16)
    args = parser.parse_args()
    if len(args.dataset_root) != len(args.domain):
        parser.error(
            "--dataset-root and --domain must appear the same number of times"
        )
    return args


def main() -> int:
    args = _parse_args()
    manifest = build_manifests(
        dataset_roots=args.dataset_root,
        domains=args.domain,
        predecoded_root=args.predecoded_root,
        output_dir=args.output_dir,
        split_seed=args.split_seed,
        window_stride=args.window_stride,
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
