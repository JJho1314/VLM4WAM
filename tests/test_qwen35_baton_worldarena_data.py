from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
import torch

from qwen35_baton.worldarena_data import (
    WorldArenaHDF5Dataset,
    WorldArenaMP4Dataset,
    WorldArenaRecord,
    audit_worldarena_hdf5_cache,
    canonical_source_frame_indices,
    future_frame_indices,
    load_worldarena_source_manifest,
    localize_source_path,
    predecode_worldarena,
)


def _write_video(path: Path, *, frame_count: int = 121) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        15.0,
        (48, 32),
    )
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            y, x = np.indices((32, 48), dtype=np.uint16)
            frame = np.stack(
                (
                    np.full_like(x, index % 251),
                    (x + index * 3) % 251,
                    (y + index * 7) % 251,
                ),
                axis=-1,
            ).astype(np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _write_source_episode(
    root: Path,
    *,
    frame_count: int = 121,
    episode_id: str = "pick_cup__episode0",
    instruction: str = "pick up the cup",
) -> tuple[WorldArenaRecord, ...]:
    dataset_root = (
        root
        if root.name == "worldarena2026-robotwin-data"
        else root / "worldarena2026-robotwin-data"
    )
    episode = dataset_root / "episodes" / episode_id
    video = episode / "video.mp4"
    actions = episode / "actions_16d.npy"
    intrinsic = root / "camera_params" / "head_intrinsic_params.json"
    extrinsic = root / "camera_params" / "head_extrinsic_params.json"
    _write_video(video, frame_count=frame_count)
    np.save(actions, np.arange(16, dtype=np.float32))
    intrinsic.parent.mkdir(parents=True, exist_ok=True)
    intrinsic.write_text('{"fx": 100.0}\n', encoding="utf-8")
    extrinsic.write_text('{"pose": [1, 0, 0, 0]}\n', encoding="utf-8")
    return (
        WorldArenaRecord(
            episode_id=episode_id,
            task_name=episode_id.split("__episode", 1)[0],
            instruction=instruction,
            video_path=video.resolve(),
            actions_16d_path=actions.resolve(),
            intrinsic_path=intrinsic.resolve(),
            extrinsic_path=extrinsic.resolve(),
            dataset_root=dataset_root.resolve(),
            source_video_relative_path=f"episodes/{episode_id}/video.mp4",
        ),
    )


def _write_cache(root: Path, *, frame_count: int = 121) -> Path:
    cache = root / "cache"
    shard = cache / "episodes" / "pick_cup__episode0.h5"
    shard.parent.mkdir(parents=True)
    rgb = np.empty((frame_count, 256, 256, 3), dtype=np.uint8)
    for index in range(frame_count):
        rgb[index].fill(index % 251)
    with h5py.File(shard, "w") as handle:
        handle.create_dataset(
            "rgb",
            data=rgb,
            dtype=np.uint8,
            chunks=(1, 256, 256, 3),
            compression="lzf",
        )
    source_root = root / "worldarena2026-robotwin-data"
    actions = source_root / "episodes" / "pick_cup__episode0" / "actions_16d.npy"
    actions.parent.mkdir(parents=True)
    np.save(actions, np.zeros(16, dtype=np.float32))
    manifest = cache / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "source_repository": "worldarena2026-robotwin-data",
                "records": [
                    {
                        "episode_id": "pick_cup__episode0",
                        "hdf5_path": "episodes/pick_cup__episode0.h5",
                        "source_dataset_root": str(source_root.resolve()),
                        "source_video_path": str(
                            source_root
                            / "episodes"
                            / "pick_cup__episode0"
                            / "video.mp4"
                        ),
                        "source_video_relative_path": (
                            "episodes/pick_cup__episode0/video.mp4"
                        ),
                        "source_video_sha256": "0" * 64,
                        "split": "train",
                        "task": "pick_cup",
                        "instruction": "pick up the cup",
                        "frame_count": frame_count,
                        "source_frame_count": frame_count,
                        "actions_16d_path": str(actions.resolve()),
                        "actions_16d_sha256": hashlib.sha256(
                            actions.read_bytes()
                        ).hexdigest(),
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_two_split_cache(root: Path) -> tuple[Path, dict[str, Path]]:
    manifest = _write_cache(root)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    train = payload["records"][0]
    validation = dict(train)
    validation_id = "pick_cup__episode1"
    validation["episode_id"] = validation_id
    validation["hdf5_path"] = f"episodes/{validation_id}.h5"
    validation["source_video_path"] = str(
        Path(validation["source_dataset_root"])
        / "episodes"
        / validation_id
        / "video.mp4"
    )
    validation["source_video_relative_path"] = f"episodes/{validation_id}/video.mp4"
    validation["split"] = "validation"
    validation.pop("actions_16d_path")
    validation.pop("actions_16d_sha256")
    validation_shard = manifest.parent / validation["hdf5_path"]
    with h5py.File(validation_shard, "w") as handle:
        handle.create_dataset(
            "rgb",
            shape=(121, 256, 256, 3),
            dtype=np.uint8,
            chunks=(1, 256, 256, 3),
            compression="lzf",
        )
    payload["records"].append(validation)
    manifest.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest, {
        "train": manifest.parent / train["hdf5_path"],
        "validation": validation_shard,
    }


def test_future_indices_are_strict_unique_and_cover_remaining_horizon() -> None:
    assert future_frame_indices(0) == (30, 60, 90, 120)
    assert future_frame_indices(116) == (117, 118, 119, 120)
    for current in range(117):
        future = future_frame_indices(current)
        assert current < future[0] < future[1] < future[2] < future[3] <= 120


def test_temporal_rounding_is_exact_half_even_for_future_and_source_indices() -> None:
    assert future_frame_indices(2) == (32, 61, 90, 120)
    canonical = canonical_source_frame_indices(61)
    assert canonical[:4] == (0, 0, 1, 2)
    assert canonical[-1] == 60
    for current, frame_count in ((-1, 121), (117, 121), (0, 4)):
        with pytest.raises(ValueError, match="strictly future"):
            future_frame_indices(current, frame_count)


def test_canonical_mapping_covers_endpoints_for_short_and_long_sources() -> None:
    short = canonical_source_frame_indices(76)
    long = canonical_source_frame_indices(787)
    assert len(short) == len(long) == 121
    assert tuple(short[index] for index in (0, 1, 2, 60, 119, 120)) == (
        0,
        1,
        1,
        38,
        74,
        75,
    )
    assert tuple(long[index] for index in (0, 1, 2, 60, 119, 120)) == (
        0,
        7,
        13,
        393,
        779,
        786,
    )
    assert tuple(sorted(short)) == short
    assert tuple(sorted(long)) == long
    with pytest.raises(ValueError, match="at least one"):
        canonical_source_frame_indices(0)


def test_source_manifest_rejects_official_episode_paths(tmp_path: Path) -> None:
    official_video = tmp_path / "official_episodes" / "task" / "episode0" / "video.mp4"
    _write_video(official_video)
    manifest = tmp_path / "metadata_train_a2v.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video": "official_episodes/task/episode0/video.mp4",
                "prompt": "pick up the cup",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="official"):
        load_worldarena_source_manifest(manifest, tmp_path)


def test_source_manifest_localizes_stale_training_data_prefix(tmp_path: Path) -> None:
    local = tmp_path / "episodes/task__episode0/actions_16d.npy"
    local.parent.mkdir(parents=True)
    local.touch()
    resolved = localize_source_path(
        "/mnt/afs/user/WorldArena/training_data/episodes/"
        "task__episode0/actions_16d.npy",
        dataset_root=tmp_path,
        required=True,
    )
    assert resolved == local.resolve()


def test_source_paths_fail_closed_on_escape_unknown_absolute_and_missing(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / "outside.mp4"
    outside.touch()
    with pytest.raises(ValueError, match="outside dataset root"):
        localize_source_path("../outside.mp4", dataset_root=tmp_path, required=True)
    with pytest.raises(ValueError, match="training_data"):
        localize_source_path(
            "/tmp/already-local/video.mp4", dataset_root=tmp_path, required=True
        )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        localize_source_path(
            "episodes/missing/video.mp4", dataset_root=tmp_path, required=True
        )


def test_source_manifest_requires_video_prompt_and_unique_episode_ids(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "worldarena2026-robotwin-data"
    episode = dataset_root / "episodes" / "pick_cup__episode0"
    _write_video(episode / "video.mp4")
    np.save(episode / "actions_16d.npy", np.zeros(1, dtype=np.float32))
    manifest = dataset_root / "metadata_train_a2v.jsonl"
    valid = {
        "video": "episodes/pick_cup__episode0/video.mp4",
        "prompt": "  pick up the cup  ",
        "action_path": "/mnt/afs/user/training_data/episodes/"
        "pick_cup__episode0/actions_16d.npy",
    }
    manifest.write_text(json.dumps(valid) + "\n", encoding="utf-8")
    records = load_worldarena_source_manifest(manifest, dataset_root)
    assert records == (
        WorldArenaRecord(
            episode_id="pick_cup__episode0",
            task_name="pick_cup",
            instruction="  pick up the cup  ",
            video_path=(episode / "video.mp4").resolve(),
            actions_16d_path=(episode / "actions_16d.npy").resolve(),
            intrinsic_path=None,
            extrinsic_path=None,
            dataset_root=dataset_root.resolve(),
            source_video_relative_path="episodes/pick_cup__episode0/video.mp4",
        ),
    )

    cache_manifest = predecode_worldarena(
        records, output_root=tmp_path / "padded-cache", seed=42
    )
    cached_payload = json.loads(cache_manifest.read_text())
    assert cached_payload["records"][0]["instruction"] == "  pick up the cup  "
    cached = WorldArenaHDF5Dataset(cache_manifest, seed=42, split="train")
    assert cached[0]["instruction"] == "  pick up the cup  "

    manifest.write_text(json.dumps({"video": valid["video"], "prompt": "  "}) + "\n")
    with pytest.raises(ValueError, match="prompt"):
        load_worldarena_source_manifest(manifest, dataset_root)
    manifest.write_text(json.dumps(valid) + "\n" + json.dumps(valid) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_worldarena_source_manifest(manifest, dataset_root)


def test_predecode_rejects_direct_record_without_canonical_training_provenance(
    tmp_path: Path,
) -> None:
    episode_id = "pick_cup__episode0"
    dataset_root = tmp_path / "worldarena2026-robotwin-data"
    episode = dataset_root / "official_validation" / episode_id
    video = episode / "video.mp4"
    actions = episode / "actions_16d.npy"
    _write_video(video)
    np.save(actions, np.zeros(16, dtype=np.float32))
    record = WorldArenaRecord(
        episode_id=episode_id,
        task_name="pick_cup",
        instruction="pick up the cup",
        video_path=video.resolve(),
        actions_16d_path=actions.resolve(),
        intrinsic_path=None,
        extrinsic_path=None,
        dataset_root=dataset_root.resolve(),
        source_video_relative_path=(f"official_validation/{episode_id}/video.mp4"),
    )
    with pytest.raises(ValueError, match="provenance|episodes"):
        predecode_worldarena((record,), output_root=tmp_path / "cache", seed=42)


def test_mp4_dataset_rejects_canonical_episode_path_symlinked_to_validation(
    tmp_path: Path,
) -> None:
    episode_id = "pick_cup__episode0"
    dataset_root = tmp_path / "worldarena2026-robotwin-data"
    official_episode = dataset_root / "official_validation" / episode_id
    video = official_episode / "video.mp4"
    _write_video(video)
    episodes = dataset_root / "episodes"
    episodes.mkdir()
    (episodes / episode_id).symlink_to(official_episode, target_is_directory=True)
    record = WorldArenaRecord(
        episode_id=episode_id,
        task_name="pick_cup",
        instruction="pick up the cup",
        video_path=video.resolve(),
        actions_16d_path=None,
        intrinsic_path=None,
        extrinsic_path=None,
        dataset_root=dataset_root.resolve(),
        source_video_relative_path=f"episodes/{episode_id}/video.mp4",
    )

    with pytest.raises(ValueError, match="symlink|canonical training provenance"):
        WorldArenaMP4Dataset((record,), seed=42, split="validation")


def test_hdf5_manifest_rejects_noncanonical_source_provenance(tmp_path: Path) -> None:
    manifest = _write_cache(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["records"][0]["source_video_path"] = str(
        tmp_path
        / "worldarena2026-robotwin-data"
        / "official_validation"
        / "pick_cup__episode0"
        / "video.mp4"
    )
    payload["records"][0]["source_video_relative_path"] = (
        "official_validation/pick_cup__episode0/video.mp4"
    )
    manifest.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="provenance|episodes"):
        WorldArenaHDF5Dataset(manifest, seed=42)

    source_root = tmp_path / "official-validation-data" / "worldarena2026-robotwin-data"
    payload["records"][0]["source_dataset_root"] = str(source_root)
    payload["records"][0]["source_video_path"] = str(
        source_root / "episodes" / "pick_cup__episode0" / "video.mp4"
    )
    payload["records"][0]["source_video_relative_path"] = (
        "episodes/pick_cup__episode0/video.mp4"
    )
    manifest.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="source_dataset_root"):
        WorldArenaHDF5Dataset(manifest, seed=42)


def test_predecode_rejects_episode_id_path_traversal(tmp_path: Path) -> None:
    record = _write_source_episode(tmp_path / "source")[0]
    invalid = (
        replace(record, episode_id="../escaped"),
        replace(record, episode_id="other__episode0"),
        replace(record, task_name="other"),
    )
    for malformed in invalid:
        with pytest.raises(ValueError, match="episode_id|task_name"):
            predecode_worldarena((malformed,), output_root=tmp_path / "cache", seed=42)
    assert not (tmp_path / "escaped.h5").exists()


def test_hdf5_manifest_rejects_wrong_identity_and_official_metadata(
    tmp_path: Path,
) -> None:
    manifest = _write_cache(tmp_path)
    original = json.loads(manifest.read_text())

    wrong_version = dict(original, version=999)
    manifest.write_text(json.dumps(wrong_version) + "\n")
    with pytest.raises(ValueError, match="version"):
        WorldArenaHDF5Dataset(manifest, seed=42)

    wrong_repository = dict(original, source_repository="official-validation-data")
    manifest.write_text(json.dumps(wrong_repository) + "\n")
    with pytest.raises(ValueError, match="source_repository"):
        WorldArenaHDF5Dataset(manifest, seed=42)

    official = json.loads(json.dumps(original))
    official["records"][0]["source_video_path"] = (
        "/dataset/official_episodes/episode0/video.mp4"
    )
    manifest.write_text(json.dumps(official) + "\n")
    with pytest.raises(ValueError, match="official"):
        WorldArenaHDF5Dataset(manifest, seed=42)


def test_hdf5_dataset_rejects_non_lzf_or_non_frame_chunked_rgb(
    tmp_path: Path,
) -> None:
    manifest = _write_cache(tmp_path)
    payload = json.loads(manifest.read_text())
    shard = manifest.parent / payload["records"][0]["hdf5_path"]
    with h5py.File(shard, "r") as handle:
        rgb = handle["rgb"][:]
    with h5py.File(shard, "w") as handle:
        handle.create_dataset("rgb", data=rgb, dtype=np.uint8)
    dataset = WorldArenaHDF5Dataset(manifest, seed=42)
    with pytest.raises(ValueError, match="LZF|chunk"):
        dataset[0]


def test_cache_audit_inspects_every_train_and_validation_shard(tmp_path: Path) -> None:
    manifest, _ = _write_two_split_cache(tmp_path)

    audit = audit_worldarena_hdf5_cache(manifest)

    assert audit.record_count == 2
    assert audit.train_count == 1
    assert audit.validation_count == 1


@pytest.mark.parametrize(
    ("split", "defect", "message"),
    [
        ("train", "dtype", "uint8"),
        ("validation", "shape", "shape"),
        ("train", "chunks", "chunked"),
        ("validation", "compression", "LZF"),
    ],
)
def test_cache_audit_rejects_malformed_shards_in_every_split(
    tmp_path: Path,
    split: str,
    defect: str,
    message: str,
) -> None:
    manifest, shards = _write_two_split_cache(tmp_path)
    shape = (120, 256, 256, 3) if defect == "shape" else (121, 256, 256, 3)
    dtype = np.float32 if defect == "dtype" else np.uint8
    chunks = (2, 256, 256, 3) if defect == "chunks" else (1, 256, 256, 3)
    compression = None if defect == "compression" else "lzf"
    with h5py.File(shards[split], "w") as handle:
        handle.create_dataset(
            "rgb",
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            compression=compression,
        )

    with pytest.raises(ValueError, match=rf"{split}.*{message}"):
        audit_worldarena_hdf5_cache(manifest)


def test_hdf5_dataset_returns_one_head_camera_and_metadata(tmp_path: Path) -> None:
    manifest = _write_cache(tmp_path, frame_count=121)
    dataset = WorldArenaHDF5Dataset(manifest, seed=42, split="train")
    sample = dataset[0]
    assert sample["current_images"].shape == (1, 3, 256, 256)
    assert sample["future_images"].shape == (1, 4, 3, 256, 256)
    assert sample["current_images"].dtype == torch.uint8
    assert sample["future_images"].dtype == torch.uint8
    assert sample["camera_names"] == ("head",)
    assert sample["instruction"] == "pick up the cup"
    assert sample["suite"] == "worldarena"
    assert sample["episode_key"] == "pick_cup__episode0"
    assert sample["source_indices"][0] < sample["source_indices"][1]
    assert "actions_16d_path" in sample["metadata"]


def test_hdf5_sampling_is_epoch_deterministic_and_validation_is_fixed(
    tmp_path: Path,
) -> None:
    manifest = _write_cache(tmp_path)
    training = WorldArenaHDF5Dataset(manifest, seed=42, split="train")
    first = training[0]["source_indices"]
    assert training[0]["source_indices"] == first
    training.set_epoch(1)
    second = training[0]["source_indices"]
    assert second == training[0]["source_indices"]
    assert second != first

    payload = json.loads(manifest.read_text())
    payload["records"][0]["split"] = "validation"
    manifest.write_text(json.dumps(payload) + "\n")
    validation = WorldArenaHDF5Dataset(manifest, seed=42, split="validation")
    fixed = validation[0]["source_indices"]
    validation.set_epoch(99)
    assert validation[0]["source_indices"] == fixed


def test_hdf5_all_window_sampling_enumerates_every_current_frame(
    tmp_path: Path,
) -> None:
    from qwen35_baton.worldarena_data import ALL_WINDOWS_SAMPLING_KIND

    manifest = _write_cache(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    first = payload["records"][0]
    second = dict(first)
    second_id = "stack_cube__episode0"
    second["episode_id"] = second_id
    second["task"] = "stack_cube"
    second["instruction"] = "stack the red cube"
    second["hdf5_path"] = f"episodes/{second_id}.h5"
    second["source_video_path"] = str(
        Path(second["source_dataset_root"]) / "episodes" / second_id / "video.mp4"
    )
    second["source_video_relative_path"] = f"episodes/{second_id}/video.mp4"
    second.pop("actions_16d_path")
    second.pop("actions_16d_sha256")
    second_shard = manifest.parent / second["hdf5_path"]
    with h5py.File(second_shard, "w") as handle:
        handle.create_dataset(
            "rgb",
            shape=(121, 256, 256, 3),
            dtype=np.uint8,
            chunks=(1, 256, 256, 3),
            compression="lzf",
        )
    payload["records"].append(second)
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    dataset = WorldArenaHDF5Dataset(
        manifest,
        seed=42,
        split="train",
        sampling_kind=ALL_WINDOWS_SAMPLING_KIND,
    )

    assert len(dataset) == 2 * 117
    assert dataset[0]["source_indices"][0] == 0
    assert dataset[116]["source_indices"][0] == 116
    assert dataset[117]["episode_key"] == second_id
    assert dataset[117]["source_indices"][0] == 0
    before = dataset[73]["source_indices"]
    dataset.set_epoch(99)
    assert dataset[73]["source_indices"] == before


def test_hdf5_all_window_sampling_keeps_validation_one_row_per_episode(
    tmp_path: Path,
) -> None:
    from qwen35_baton.worldarena_data import ALL_WINDOWS_SAMPLING_KIND

    manifest = _write_cache(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["records"][0]["split"] = "validation"
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    dataset = WorldArenaHDF5Dataset(
        manifest,
        seed=42,
        split="validation",
        sampling_kind=ALL_WINDOWS_SAMPLING_KIND,
    )

    assert len(dataset) == 1
    fixed = dataset[0]["source_indices"]
    dataset.set_epoch(99)
    assert dataset[0]["source_indices"] == fixed


def test_predecode_publishes_atomic_sorted_manifest_stats_and_hdf5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qwen35_baton import worldarena_data

    records = (
        *_write_source_episode(tmp_path / "source", episode_id="z_task__episode1"),
        *_write_source_episode(tmp_path / "source", episode_id="a_task__episode0"),
    )
    replaced: list[Path] = []
    real_replace = os.replace

    def recording_replace(
        source: str | os.PathLike[str], destination: str | os.PathLike[str]
    ):
        real_replace(source, destination)
        replaced.append(Path(destination))

    monkeypatch.setattr(worldarena_data.os, "replace", recording_replace)
    output = tmp_path / "cache"
    manifest_path = predecode_worldarena(
        records,
        output_root=output,
        seed=42,
        validation_fraction=0.1,
    )
    assert manifest_path == output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cached = manifest["records"]
    assert [record["episode_id"] for record in cached] == [
        "a_task__episode0",
        "z_task__episode1",
    ]
    assert [record["hdf5_path"] for record in cached] == [
        "episodes/a_task__episode0.h5",
        "episodes/z_task__episode1.h5",
    ]
    for source, record in zip(
        sorted(records, key=lambda item: item.episode_id), cached
    ):
        assert (
            record["source_video_sha256"]
            == hashlib.sha256(source.video_path.read_bytes()).hexdigest()
        )
        expected_split_value = (
            int.from_bytes(
                hashlib.sha256(f"42:{source.episode_id}".encode()).digest()[:8], "big"
            )
            / 2**64
        )
        assert record["split"] == (
            "validation" if expected_split_value < 0.1 else "train"
        )
        with h5py.File(output / record["hdf5_path"], "r") as handle:
            rgb = handle["rgb"]
            assert rgb.shape == (121, 256, 256, 3)
            assert rgb.dtype == np.dtype(np.uint8)
            assert rgb.chunks == (1, 256, 256, 3)
            assert rgb.compression == "lzf"

    stats_path = output / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["train_count"] + stats["validation_count"] == 2
    assert stats["task_counts"] == {"a_task": 1, "z_task": 1}
    assert stats["image_size"] == [256, 256]
    assert stats["frame_count"] == 121
    assert stats["source_frame_counts"] == {
        "a_task__episode0": 121,
        "z_task__episode1": 121,
    }
    assert stats["seed"] == 42
    assert stats["source_repository"] == "worldarena2026-robotwin-data"
    assert (
        stats["manifest_sha256"]
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert replaced[-1] == output
    assert [path.name for path in replaced[:-1]] == [
        "a_task__episode0.h5",
        "z_task__episode1.h5",
        "manifest.json",
        "stats.json",
    ]
    assert not list(output.rglob("*.tmp"))


def test_predecode_publication_failure_leaves_no_partial_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qwen35_baton import worldarena_data

    records = _write_source_episode(tmp_path / "source")
    output = tmp_path / "cache"
    real_replace = os.replace

    def fail_final_replace(
        source: str | os.PathLike[str], destination: str | os.PathLike[str]
    ) -> None:
        if Path(destination) == output:
            raise OSError("injected final publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(worldarena_data.os, "replace", fail_final_replace)
    with pytest.raises(OSError, match="injected"):
        predecode_worldarena(records, output_root=output, seed=42)
    assert not output.exists()
    assert not list(tmp_path.glob(".cache.*.tmp"))


def test_predecode_refuses_to_overwrite_nonempty_output_root(tmp_path: Path) -> None:
    records = _write_source_episode(tmp_path / "source")
    output = tmp_path / "cache"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("preserve me")
    with pytest.raises(ValueError, match="nonempty"):
        predecode_worldarena(records, output_root=output, seed=42)
    assert sentinel.read_text() == "preserve me"


@pytest.mark.parametrize("source_frame_count", [76, 787])
def test_predecode_uniformly_normalizes_noncanonical_source_frame_count(
    tmp_path: Path, source_frame_count: int
) -> None:
    output = tmp_path / "cache"
    records = _write_source_episode(tmp_path / "source", frame_count=source_frame_count)
    manifest_path = predecode_worldarena(records, output_root=output, seed=42)
    manifest = json.loads(manifest_path.read_text())
    record = manifest["records"][0]
    assert record["frame_count"] == 121
    assert record["source_frame_count"] == source_frame_count
    with h5py.File(output / record["hdf5_path"], "r") as handle:
        assert handle["rgb"].shape == (121, 256, 256, 3)


def test_hdf5_selected_rgb_matches_online_mp4_decode(tmp_path: Path) -> None:
    records = _write_source_episode(tmp_path / "source", frame_count=137)
    online = WorldArenaMP4Dataset(records, seed=42, split="validation")
    manifest = predecode_worldarena(
        records,
        output_root=tmp_path / "cache",
        seed=42,
        validation_fraction=1.0,
    )
    cached = WorldArenaHDF5Dataset(manifest, seed=42, split="validation")
    assert cached[0]["source_indices"] == online[0]["source_indices"]
    torch.testing.assert_close(cached[0]["current_images"], online[0]["current_images"])
    torch.testing.assert_close(cached[0]["future_images"], online[0]["future_images"])


def test_mp4_dataset_uses_actual_decodable_count_not_container_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qwen35_baton import worldarena_data

    records = _write_source_episode(tmp_path / "source", frame_count=76)
    real_capture = cv2.VideoCapture

    class _LyingCapture:
        def __init__(self, path: str) -> None:
            self._capture = real_capture(path)

        def __getattr__(self, name: str):
            return getattr(self._capture, name)

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 787.0
            return self._capture.get(prop)

    monkeypatch.setattr(worldarena_data.cv2, "VideoCapture", _LyingCapture)
    sample = WorldArenaMP4Dataset(records, seed=42, split="validation")[0]
    assert sample["metadata"]["source_frame_count"] == 76
    assert sample["future_images"].shape == (1, 4, 3, 256, 256)


def test_predecode_cli_uses_training_manifest_and_honors_limit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from qwen35_baton.cli.predecode_worldarena import main

    dataset_root = tmp_path / "worldarena2026-robotwin-data"
    _write_source_episode(dataset_root, episode_id="pick_cup__episode0")
    manifest = dataset_root / "metadata_train_a2v.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "video": "episodes/pick_cup__episode0/video.mp4",
                "prompt": "pick up the cup",
                "action_path": "/mnt/afs/user/training_data/episodes/"
                "pick_cup__episode0/actions_16d.npy",
                "intrinsic_path": "/mnt/afs/user/training_data/camera_params/"
                "head_intrinsic_params.json",
                "extrinsic_path": "/mnt/afs/user/training_data/camera_params/"
                "head_extrinsic_params.json",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "cache"
    assert (
        main(
            [
                "--dataset-root",
                str(dataset_root),
                "--output-root",
                str(output_root),
                "--seed",
                "42",
                "--validation-fraction",
                "0.1",
                "--limit",
                "1",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == str(output_root / "manifest.json")
    published = json.loads((output_root / "manifest.json").read_text())
    assert len(published["records"]) == 1
