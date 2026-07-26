from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


def _write_episode(
    dataset_root: Path,
    predecoded_root: Path,
    *,
    domain: str = "libero_goal",
    episode_index: int = 0,
    num_frames: int = 20,
    dtype: np.dtype = np.dtype(np.uint8),
    wrist_frames: int | None = None,
    omit_camera: str | None = None,
) -> None:
    domain_root = dataset_root / domain
    metadata_dir = domain_root / "meta"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    episode = {
        "episode_index": episode_index,
        "tasks": [f"instruction {episode_index}"],
        "length": num_frames,
    }
    with (metadata_dir / "episodes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(episode) + "\n")

    camera_specs = {
        "observation.images.image": num_frames,
        "observation.images.wrist_image": (
            num_frames if wrist_frames is None else wrist_frames
        ),
    }
    for camera_key, frame_count in camera_specs.items():
        if camera_key == omit_camera:
            continue
        path = (
            predecoded_root
            / domain
            / "videos"
            / "chunk-000"
            / camera_key
            / f"episode_{episode_index:06d}.npy"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            path,
            np.zeros((frame_count, 12, 16, 3), dtype=dtype),
            allow_pickle=False,
        )


def _discover_one(
    tmp_path: Path,
    **episode_kwargs: object,
):
    from qwen35_planx.libero_data import discover_trajectories

    dataset_root = tmp_path / "dataset"
    predecoded_root = tmp_path / "predecoded"
    _write_episode(dataset_root, predecoded_root, **episode_kwargs)
    return discover_trajectories(
        dataset_roots=(dataset_root,),
        domains=("libero_goal",),
        predecoded_root=predecoded_root,
        split_seed=0,
    )


def test_trajectory_split_never_splits_an_episode() -> None:
    from qwen35_planx.libero_data import trajectory_split

    assignments = {
        f"suite:{episode_index}": trajectory_split(
            f"suite:{episode_index}", seed=0
        )
        for episode_index in range(30)
    }

    assert set(assignments.values()) == {"train", "val"}
    assert len(assignments) == 30
    assert all(
        assignment == trajectory_split(trajectory_id, seed=0)
        for trajectory_id, assignment in assignments.items()
    )


def test_discovery_and_planner_windows_preserve_camera_order_and_offsets(
    tmp_path: Path,
) -> None:
    from qwen35_planx.libero_data import (
        iter_all_camera_frames,
        iter_planner_windows,
        load_predecoded_frames,
    )

    records = _discover_one(tmp_path, num_frames=20)
    assert len(records) == 1
    record = records[0]

    arrays = load_predecoded_frames(record)
    assert tuple(arrays) == ("main", "wrist")
    assert arrays["main"].shape == arrays["wrist"].shape == (20, 12, 16, 3)

    frames = list(iter_all_camera_frames(record))
    assert len(frames) == 40
    assert [(item.camera, item.frame_index) for item in frames[:3]] == [
        ("main", 0),
        ("wrist", 0),
        ("main", 1),
    ]

    windows = list(iter_planner_windows(record, stride=10, max_windows=16))
    assert len(windows) == 2
    assert windows[0].current_index == 0
    assert windows[0].future_indices == (1, 4, 6, 9)
    assert windows[0].camera_cache_paths == (
        record.camera_cache_paths["main"],
        record.camera_cache_paths["wrist"],
    )
    assert windows[1].current_index == 10
    assert windows[1].future_indices == (11, 14, 16, 19)


@pytest.mark.parametrize(
    ("episode_kwargs", "message"),
    [
        ({"dtype": np.float32}, "uint8"),
        (
            {"omit_camera": "observation.images.wrist_image"},
            "wrist",
        ),
        ({"wrist_frames": 19}, "frame count"),
    ],
)
def test_discovery_rejects_invalid_or_incomplete_camera_caches(
    tmp_path: Path,
    episode_kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((FileNotFoundError, ValueError), match=message):
        _discover_one(tmp_path, **episode_kwargs)


def test_window_record_rejects_indices_past_trajectory_end(
    tmp_path: Path,
) -> None:
    from qwen35_planx.libero_data import iter_planner_windows

    record = _discover_one(tmp_path, num_frames=20)[0]
    window = next(iter_planner_windows(record))

    with pytest.raises(ValueError, match="trajectory"):
        replace(window, future_indices=(1, 4, 6, record.num_frames))


def test_manifest_cli_writes_sorted_hash_bound_outputs(
    tmp_path: Path,
) -> None:
    from qwen35_planx.cli.build_libero_manifests import build_manifests

    dataset_root = tmp_path / "dataset"
    predecoded_root = tmp_path / "predecoded"
    _write_episode(
        dataset_root,
        predecoded_root,
        episode_index=2,
        num_frames=20,
    )
    _write_episode(
        dataset_root,
        predecoded_root,
        episode_index=1,
        num_frames=20,
    )
    output_dir = tmp_path / "manifests"

    manifest = build_manifests(
        dataset_roots=(dataset_root,),
        domains=("libero_goal",),
        predecoded_root=predecoded_root,
        output_dir=output_dir,
        split_seed=0,
        window_stride=10,
        max_windows_per_trajectory=16,
    )

    expected_files = {
        "trajectories.jsonl",
        "ta_frames_train.jsonl",
        "ta_frames_val.jsonl",
        "planner_train.jsonl",
        "planner_val.jsonl",
    }
    assert set(manifest["files"]) == expected_files
    for name in expected_files:
        assert (output_dir / name).is_file()
        assert len(manifest["files"][name]["sha256"]) == 64
    assert len(manifest["contract_hash"]) == 64

    trajectory_rows = [
        json.loads(line)
        for line in (output_dir / "trajectories.jsonl").read_text().splitlines()
    ]
    assert [row["episode_index"] for row in trajectory_rows] == [1, 2]

    persisted = json.loads((output_dir / "manifest.json").read_text())
    assert persisted == manifest
