import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from ge_act.data import libero_fastwam_hdf5_schema as schema


def make_manifest(tmp_path: Path, **overrides):
    shard_path = tmp_path / "shard_00000.h5"
    shard_path.touch()
    payload = {
        "schema_version": 1,
        "camera_names": ["main", "wrist"],
        "image_size": [256, 256],
        "source_fps": 20,
        "n_previous": 4,
        "chunk": 9,
        "action_chunk": 36,
        "action_type": "absolute",
        "action_space": "eef",
        "compression": "lzf",
        "episodes": [
            {
                "key": "libero_goal:000010",
                "shard": shard_path.name,
                "group": "episodes/000010",
                "domain": "libero_goal",
                "episode_index": 10,
                "length": 3,
            }
        ],
    }
    payload.update(overrides)
    return payload


def make_episode_group(file: h5py.File, record: schema.EpisodeRecord):
    group = file.create_group(record.group)
    group.attrs["key"] = record.key
    group.attrs["domain"] = record.domain
    group.attrs["episode_index"] = record.episode_index
    group.attrs["length"] = record.length
    group.create_dataset(
        "rgb_main", shape=(record.length, 256, 256, 3), dtype=np.uint8
    )
    group.create_dataset(
        "rgb_wrist", shape=(record.length, 256, 256, 3), dtype=np.uint8
    )
    group.create_dataset("action", shape=(record.length, 7), dtype=np.float32)
    group.create_dataset("state", shape=(record.length, 9), dtype=np.float32)
    return group


def test_manifest_accepts_fixed_libero_contract(tmp_path):
    payload = make_manifest(tmp_path, camera_names=["main", "wrist"])
    records = schema.validate_manifest(payload, tmp_path)
    assert records[0].key == "libero_goal:000010"
    assert records[0].shard_path == tmp_path / "shard_00000.h5"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("camera_names", ["wrist", "main"], "camera_names"),
        ("image_size", [512, 512], "image_size"),
        ("source_fps", 30, "source_fps"),
        ("n_previous", 3, "n_previous"),
        ("chunk", 8, "chunk"),
        ("action_chunk", 32, "action_chunk"),
    ],
)
def test_manifest_rejects_wrong_fixed_contract(tmp_path, field, value, message):
    payload = make_manifest(tmp_path)
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        schema.validate_manifest(payload, tmp_path)


def test_manifest_rejects_duplicate_episode_keys(tmp_path):
    payload = make_manifest(tmp_path)
    payload["episodes"].append(dict(payload["episodes"][0]))
    with pytest.raises(ValueError, match="duplicate episode key"):
        schema.validate_manifest(payload, tmp_path)


@pytest.mark.parametrize("version", [None, True, 2])
def test_manifest_rejects_unsupported_schema_version(tmp_path, version):
    payload = make_manifest(tmp_path, schema_version=version)
    with pytest.raises(ValueError, match="schema_version must be 1"):
        schema.validate_manifest(payload, tmp_path)


def test_manifest_rejects_missing_shard(tmp_path):
    payload = make_manifest(tmp_path)
    (tmp_path / "shard_00000.h5").unlink()
    with pytest.raises(FileNotFoundError, match="missing HDF5 shard"):
        schema.validate_manifest(payload, tmp_path)


def test_load_manifest_reads_json_and_validates_relative_to_parent(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(make_manifest(tmp_path)), encoding="utf-8")
    records = schema.load_manifest(manifest_path)
    assert records == [
        schema.EpisodeRecord(
            key="libero_goal:000010",
            shard_path=tmp_path / "shard_00000.h5",
            group="episodes/000010",
            domain="libero_goal",
            episode_index=10,
            length=3,
        )
    ]


def test_validate_episode_group_accepts_matching_hdf5_data(tmp_path):
    record = schema.validate_manifest(make_manifest(tmp_path), tmp_path)[0]
    with h5py.File(record.shard_path, "w") as file:
        group = make_episode_group(file, record)
        schema.validate_episode_group(group, record)


@pytest.mark.parametrize(
    ("dataset", "shape", "dtype", "message"),
    [
        ("rgb_main", (3, 128, 256, 3), np.uint8, "rgb_main"),
        ("rgb_wrist", (3, 256, 256, 3), np.float32, "rgb_wrist"),
        ("action", (2, 7), np.float32, "action"),
        ("state", (3, 9), np.float64, "state"),
    ],
)
def test_validate_episode_group_rejects_wrong_shape_or_dtype(
    tmp_path, dataset, shape, dtype, message
):
    record = schema.validate_manifest(make_manifest(tmp_path), tmp_path)[0]
    with h5py.File(record.shard_path, "w") as file:
        group = make_episode_group(file, record)
        del group[dataset]
        group.create_dataset(dataset, shape=shape, dtype=dtype)
        with pytest.raises(ValueError, match=message):
            schema.validate_episode_group(group, record)


@pytest.mark.parametrize("attribute", ["key", "domain", "episode_index", "length"])
def test_validate_episode_group_rejects_mismatched_scalar_metadata(
    tmp_path, attribute
):
    record = schema.validate_manifest(make_manifest(tmp_path), tmp_path)[0]
    with h5py.File(record.shard_path, "w") as file:
        group = make_episode_group(file, record)
        group.attrs[attribute] = "wrong"
        with pytest.raises(ValueError, match=attribute):
            schema.validate_episode_group(group, record)


def test_atomic_write_manifest_replaces_target_without_temp_files(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("old", encoding="utf-8")
    payload = make_manifest(tmp_path)
    schema.atomic_write_manifest(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert set(tmp_path.iterdir()) == {path, tmp_path / "shard_00000.h5"}


def test_atomic_write_manifest_removes_temp_file_when_replace_fails(
    tmp_path, monkeypatch
):
    path = tmp_path / "manifest.json"
    path.write_text("old", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(schema.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        schema.atomic_write_manifest(path, make_manifest(tmp_path))
    assert path.read_text(encoding="utf-8") == "old"
    assert set(tmp_path.iterdir()) == {path, tmp_path / "shard_00000.h5"}
