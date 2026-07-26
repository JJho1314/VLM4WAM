from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from ge_act.data.libero_fastwam_hdf5_schema import load_manifest
from qwen35_planx.config import HindsightCacheMetadata


def _write_hdf5_manifest(
    root: Path,
    *,
    episode_lengths: tuple[int, ...] = (80,),
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    shard_path = root / "shard_00000.h5"
    episodes = []
    with h5py.File(shard_path, "w") as handle:
        for episode_index, length in enumerate(episode_lengths):
            key = f"libero_goal:{episode_index:06d}"
            group = handle.create_group(f"episodes/{key}")
            string_dtype = h5py.string_dtype(encoding="utf-8")
            group.create_dataset(
                "caption",
                data=f"pick up object {episode_index}",
                dtype=string_dtype,
            )
            group.create_dataset("domain", data="libero_goal", dtype=string_dtype)
            group.create_dataset("episode_index", data=episode_index, dtype=np.int64)
            group.create_dataset("length", data=length, dtype=np.int64)
            main = np.full((length, 256, 256, 3), episode_index, np.uint8)
            wrist = np.full((length, 256, 256, 3), 100 + episode_index, np.uint8)
            group.create_dataset("rgb_main", data=main)
            group.create_dataset("rgb_wrist", data=wrist)
            group.create_dataset(
                "action",
                data=np.arange(length * 7, dtype=np.float32).reshape(length, 7),
            )
            group.create_dataset(
                "state",
                data=np.arange(length * 8, dtype=np.float32).reshape(length, 8),
            )
            episodes.append(
                {
                    "key": key,
                    "shard": shard_path.name,
                    "group": f"episodes/{key}",
                    "caption": f"pick up object {episode_index}",
                    "domain": "libero_goal",
                    "episode_index": episode_index,
                    "length": length,
                }
            )

    manifest = {
        "schema_version": 1,
        "camera_names": ["main", "wrist"],
        "image_size": [256, 256],
        "source_fps": 20,
        "n_previous": 4,
        "chunk": 9,
        "action_chunk": 36,
        "action_type": "absolute",
        "action_space": "eef",
        "compression": "none",
        "source_roots": [str(root / "source")],
        "datasets": {
            "rgb_main": {"shape_tail": [256, 256, 3], "dtype": "uint8"},
            "rgb_wrist": {"shape_tail": [256, 256, 3], "dtype": "uint8"},
            "action": {"width": 7, "dtype": "float32"},
            "state": {"width": 8, "dtype": "float32"},
        },
        "converter_fingerprint": "a" * 64,
        "episodes": episodes,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


@pytest.fixture
def fake_hdf5_manifest(tmp_path: Path) -> Path:
    return _write_hdf5_manifest(tmp_path / "hdf5")


def test_fixed_window_matches_ge_act_keyframes(
    fake_hdf5_manifest: Path,
) -> None:
    from qwen35_planx.hindsight_data import build_fixed_windows

    windows = build_fixed_windows(
        fake_hdf5_manifest,
        split_seed=42,
        window_stride=36,
        sample_n_frames=500,
    )
    window = windows[0]
    assert len(window.frame_indices) == 13
    assert len(window.action_indices) == 40
    assert tuple(window.frame_indices) == (
        1,
        1,
        1,
        1,
        3,
        7,
        11,
        15,
        19,
        23,
        27,
        31,
        35,
    )
    assert tuple(window.action_indices) == (
        1,
        1,
        1,
        1,
        1,
        *range(1, 36),
    )
    assert window.current_index == window.frame_indices[3]
    assert window.future_indices == tuple(
        window.frame_indices[4 + index] for index in (0, 3, 5, 8)
    )
    assert window.camera_names == ("main", "wrist")


def test_windows_are_deterministic_sorted_and_trajectory_safe(
    fake_hdf5_manifest: Path,
) -> None:
    from qwen35_planx.hindsight_data import build_fixed_windows
    from qwen35_planx.libero_data import trajectory_split

    first = build_fixed_windows(
        fake_hdf5_manifest,
        split_seed=42,
        window_stride=36,
        sample_n_frames=500,
    )
    second = build_fixed_windows(
        fake_hdf5_manifest,
        split_seed=42,
        window_stride=36,
        sample_n_frames=500,
    )

    assert first == second
    assert [window.sample_id for window in first] == sorted(
        window.sample_id for window in first
    )
    assert {window.episode_key for window in first} == {"libero_goal:000000"}
    assert {window.split for window in first} == {
        trajectory_split("libero_goal:000000", seed=42)
    }
    assert all(max(window.frame_indices) < 80 for window in first)
    assert all(max(window.action_indices) < 80 for window in first)


def test_read_full_trajectory_validates_and_preserves_camera_order(
    fake_hdf5_manifest: Path,
) -> None:
    from qwen35_planx.hindsight_data import read_full_trajectory

    _, records = load_manifest(fake_hdf5_manifest)
    trajectory = read_full_trajectory(records[0])

    assert trajectory.rgb.shape == (2, 80, 256, 256, 3)
    assert trajectory.rgb.dtype == np.uint8
    assert trajectory.actions.shape == (80, 7)
    assert trajectory.actions.dtype == np.float32
    assert trajectory.states.shape == (80, 8)
    assert trajectory.states.dtype == np.float32
    assert np.all(trajectory.rgb[0] == 0)
    assert np.all(trajectory.rgb[1] == 100)


def _metadata(**overrides: str) -> HindsightCacheMetadata:
    values = {
        "format_version": 1,
        "hdf5_manifest_hash": "hdf5-hash",
        "window_manifest_hash": "split-hash",
        "instruction_parser_hash": "parser-hash",
        "ta_tok_hash": "ta-hash",
        "siglip2_hash": "siglip-hash",
        "dinov3_hash": "dino-hash",
        "preprocessing_hash": "preprocess-hash",
    }
    values.update(overrides)
    return HindsightCacheMetadata(**values)


def _cache_arrays(
    count: int,
    *,
    code_value: int = 7,
) -> dict[str, np.ndarray]:
    return {
        "codes": np.full((count, 2, 4, 729), code_value, dtype=np.int64),
        "relevance_q": np.ones((count, 2, 4, 3, 729), dtype=np.uint8),
        "relevance_scale": np.ones((count, 2, 4, 3), dtype=np.float16),
        "confidence": np.full((count, 2, 4, 3), 0.5, dtype=np.float16),
        "flow": np.zeros((count, 2, 3, 729, 3), dtype=np.float16),
        "phrase_embeddings": np.ones((count, 3, 1152), dtype=np.float16),
    }


def _two_windows(manifest_path: Path):
    from qwen35_planx.hindsight_data import build_fixed_windows

    return build_fixed_windows(
        manifest_path,
        split_seed=42,
        window_stride=36,
        sample_n_frames=500,
    )[:2]


def test_hindsight_shards_finalize_and_round_trip_as_read_only_memmaps(
    fake_hdf5_manifest: Path,
    tmp_path: Path,
) -> None:
    from qwen35_planx.hindsight_schema import (
        HindsightCache,
        HindsightShardWriter,
        finalize_hindsight_cache,
    )

    metadata = _metadata()
    windows = _two_windows(fake_hdf5_manifest)
    shard_b = tmp_path / "shards" / "b.npz"
    shard_a = tmp_path / "shards" / "a.npz"
    HindsightShardWriter(shard_b, metadata=metadata).write(
        [windows[1]], **_cache_arrays(1, code_value=11)
    )
    HindsightShardWriter(shard_a, metadata=metadata).write(
        [windows[0]], **_cache_arrays(1, code_value=7)
    )

    cache_dir = tmp_path / "cache"
    finalize_hindsight_cache(
        cache_dir,
        shard_paths=(shard_b, shard_a),
        metadata=metadata,
    )

    with HindsightCache.open(
        cache_dir,
        expected_metadata=metadata,
        expected_split_hash=metadata.window_manifest_hash,
    ) as cache:
        assert len(cache) == 2
        assert cache.codes.flags.writeable is False
        assert [cache[index].record.sample_id for index in range(2)] == [
            window.sample_id for window in windows
        ]
        sample = cache[1]
        assert sample.codes.shape == (2, 4, 729)
        assert sample.relevance.shape == (2, 4, 3, 729)
        assert sample.confidence.shape == (2, 4, 3)
        assert sample.flow.shape == (2, 3, 729, 3)
        assert sample.phrase_embeddings.shape == (3, 1152)
        assert torch.allclose(sample.relevance.sum(-1), torch.ones(2, 4, 3))
        assert torch.all(sample.codes == 11)


def test_hindsight_cache_rejects_partial_publication(tmp_path: Path) -> None:
    from qwen35_planx.hindsight_schema import HindsightCache

    cache_dir = tmp_path / "partial"
    cache_dir.mkdir()
    np.lib.format.open_memmap(
        cache_dir / "codes.npy",
        mode="w+",
        dtype=np.uint16,
        shape=(1, 2, 4, 729),
    )

    with pytest.raises(ValueError, match="manifest"):
        HindsightCache.open(cache_dir)


def test_finalized_cache_combines_multiple_trajectory_shards(
    tmp_path: Path,
) -> None:
    from qwen35_planx.hindsight_data import build_fixed_windows
    from qwen35_planx.hindsight_schema import (
        HindsightCache,
        HindsightShardWriter,
        finalize_hindsight_cache,
    )

    manifest_path = _write_hdf5_manifest(tmp_path / "hdf5", episode_lengths=(40, 40))
    windows = build_fixed_windows(
        manifest_path,
        split_seed=42,
        window_stride=36,
        sample_n_frames=500,
    )
    metadata = _metadata()
    shards = []
    for episode_key in sorted({window.episode_key for window in windows}):
        records = [window for window in windows if window.episode_key == episode_key]
        shard = tmp_path / "shards" / f"{episode_key[-6:]}.npz"
        HindsightShardWriter(shard, metadata=metadata).write(
            records, **_cache_arrays(len(records))
        )
        shards.append(shard)
    cache_dir = tmp_path / "cache"
    finalize_hindsight_cache(cache_dir, shard_paths=shards, metadata=metadata)

    with HindsightCache.open(cache_dir) as cache:
        assert len(cache) == 4
        assert {record.episode_key for record in cache.records} == {
            "libero_goal:000000",
            "libero_goal:000001",
        }


def test_finalize_rejects_duplicate_sample_ids_without_publishing(
    fake_hdf5_manifest: Path,
    tmp_path: Path,
) -> None:
    from qwen35_planx.hindsight_schema import (
        HindsightShardWriter,
        finalize_hindsight_cache,
    )

    metadata = _metadata()
    record = _two_windows(fake_hdf5_manifest)[0]
    shards = []
    for name in ("a.npz", "b.npz"):
        path = tmp_path / "shards" / name
        HindsightShardWriter(path, metadata=metadata).write(
            [record], **_cache_arrays(1)
        )
        shards.append(path)
    cache_dir = tmp_path / "cache"

    with pytest.raises(ValueError, match="duplicate sample_id"):
        finalize_hindsight_cache(
            cache_dir,
            shard_paths=shards,
            metadata=metadata,
        )
    assert not cache_dir.exists()


def test_finalize_rejects_partial_shard_without_publishing(
    tmp_path: Path,
) -> None:
    from qwen35_planx.hindsight_schema import finalize_hindsight_cache

    partial = tmp_path / "partial.npz"
    np.savez(partial, codes=np.zeros((1, 2, 4, 729), dtype=np.uint16))
    cache_dir = tmp_path / "cache"

    with pytest.raises(ValueError, match="shard fields"):
        finalize_hindsight_cache(
            cache_dir,
            shard_paths=[partial],
            metadata=_metadata(),
        )
    assert not cache_dir.exists()


def test_cache_open_rejects_wrong_split_and_teacher_hashes(
    fake_hdf5_manifest: Path,
    tmp_path: Path,
) -> None:
    from qwen35_planx.hindsight_schema import (
        HindsightCache,
        HindsightShardWriter,
        finalize_hindsight_cache,
    )

    metadata = _metadata()
    shard = tmp_path / "shard.npz"
    HindsightShardWriter(shard, metadata=metadata).write(
        [_two_windows(fake_hdf5_manifest)[0]], **_cache_arrays(1)
    )
    cache_dir = tmp_path / "cache"
    manifest = finalize_hindsight_cache(
        cache_dir, shard_paths=[shard], metadata=metadata
    )

    with pytest.raises(ValueError, match="split hash"):
        HindsightCache.open(cache_dir, expected_split_hash="wrong")
    with pytest.raises(ValueError, match="teacher hash"):
        HindsightCache.open(cache_dir, expected_teacher_hash="wrong")
    with HindsightCache.open(cache_dir, expected_teacher_hash=manifest["teacher_hash"]):
        pass


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("codes", "codes"),
        ("flow", "flow"),
        ("camera", "camera"),
    ],
)
def test_shard_writer_rejects_invalid_teacher_outputs_and_camera_reordering(
    fake_hdf5_manifest: Path,
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    from dataclasses import replace

    from qwen35_planx.hindsight_schema import HindsightShardWriter

    record = _two_windows(fake_hdf5_manifest)[0]
    arrays = _cache_arrays(1)
    if mutation == "codes":
        arrays["codes"][0, 0, 0, 0] = 65_536
    elif mutation == "flow":
        arrays["flow"][0, 0, 0, 0, 0] = np.nan

    with pytest.raises(ValueError, match=message):
        if mutation == "camera":
            record = replace(record, camera_names=("wrist", "main"))
        HindsightShardWriter(tmp_path / f"{mutation}.npz", metadata=_metadata()).write(
            [record], **arrays
        )
