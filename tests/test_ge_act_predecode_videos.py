from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
GE_ACT_ROOT = REPO_ROOT / "ge_act"
if str(GE_ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(GE_ACT_ROOT))

from scripts import predecode_lerobot_videos as predecode


def test_cache_path_mirrors_dataset_tree(tmp_path: Path) -> None:
    source = (
        tmp_path
        / "data/domain/videos/chunk-000/cam/episode_000001.mp4"
    )
    expected = (
        tmp_path
        / "cache/domain/videos/chunk-000/cam/episode_000001.npy"
    )

    actual = predecode.cache_path_for_video(
        source, tmp_path / "data", tmp_path / "cache"
    )

    assert actual == expected


def test_cache_path_rejects_video_outside_data_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside data root"):
        predecode.cache_path_for_video(
            tmp_path / "other/episode.mp4",
            tmp_path / "data",
            tmp_path / "cache",
        )


def test_atomic_cache_writer_round_trips_uint8(tmp_path: Path) -> None:
    frames = np.arange(3 * 4 * 5 * 3, dtype=np.uint8).reshape(3, 4, 5, 3)
    destination = tmp_path / "nested/episode.npy"

    predecode.write_rgb_cache_atomic(destination, frames)

    np.testing.assert_array_equal(np.load(destination), frames)
    assert not list(destination.parent.glob("*.tmp"))


def test_atomic_cache_writer_rejects_invalid_layout(tmp_path: Path) -> None:
    destination = tmp_path / "episode.npy"

    with pytest.raises(ValueError, match="RGB cache"):
        predecode.write_rgb_cache_atomic(
            destination, np.zeros((3, 4, 5), dtype=np.uint8)
        )

    assert not destination.exists()


def test_discovery_deduplicates_repeated_roots_and_mirrors_cameras(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    for camera in ("main", "wrist"):
        path = (
            data_root
            / f"domain/videos/chunk-000/{camera}/episode_000001.mp4"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    config = {
        "data": {
            "train": {
                "data_roots": [str(data_root), str(data_root)],
                "domains": ["domain", "domain"],
                "valid_cam": ["main", "wrist"],
                "predecoded_video_root": str(cache_root),
                "require_predecoded": True,
            }
        }
    }

    pairs = predecode.discover_video_pairs(config)

    assert len(pairs) == 2
    assert {destination for _, destination in pairs} == {
        cache_root / "domain/videos/chunk-000/main/episode_000001.npy",
        cache_root / "domain/videos/chunk-000/wrist/episode_000001.npy",
    }


def test_verify_cache_rejects_missing_and_malformed_arrays(tmp_path: Path) -> None:
    missing = tmp_path / "missing.npy"
    malformed = tmp_path / "malformed.npy"
    np.save(malformed, np.zeros((2, 3, 4), dtype=np.uint8))

    assert predecode.verify_rgb_cache(missing)[0] is False
    valid, message = predecode.verify_rgb_cache(malformed)

    assert valid is False
    assert "invalid RGB cache" in message
