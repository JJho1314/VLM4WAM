import json
import os
import pickle
import random
from argparse import Namespace
from pathlib import Path

import av
import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

from ge_act.data import libero_fastwam_hdf5_schema as schema
from ge_act.data import libero_fastwam_hdf5_dataset as hdf5_dataset
from ge_act.data.libero_fastwam_hdf5_dataset import (
    LiberoFastWAMHDF5Dataset,
    read_rows_preserving_order,
)
from ge_act.scripts import convert_libero_fastwam_hdf5 as converter


pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame is deprecated:DeprecationWarning"
)


DATASET_DECLARATIONS = {
    "rgb_main": {"shape_tail": [256, 256, 3], "dtype": "uint8"},
    "rgb_wrist": {"shape_tail": [256, 256, 3], "dtype": "uint8"},
    "action": {"width": 7, "dtype": "float32"},
    "state": {"width": 8, "dtype": "float32"},
}
EXPECTED_MANIFEST_FIELDS = {
    "schema_version",
    "camera_names",
    "image_size",
    "source_fps",
    "n_previous",
    "chunk",
    "action_chunk",
    "action_type",
    "action_space",
    "compression",
    "source_roots",
    "datasets",
    "converter_fingerprint",
    "episodes",
}


def make_manifest(tmp_path: Path, **overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    shard_path = tmp_path / "shard_00000.h5"
    shard_path.touch()
    key = "libero_goal:000010"
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
        "source_roots": [str(tmp_path / "source")],
        "datasets": DATASET_DECLARATIONS,
        "converter_fingerprint": "a" * 64,
        "episodes": [
            {
                "key": key,
                "shard": shard_path.name,
                "group": f"episodes/{key}",
                "caption": "pick up the red mug",
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
    string_dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset("caption", data=record.caption, dtype=string_dtype)
    group.create_dataset("domain", data=record.domain, dtype=string_dtype)
    group.create_dataset("episode_index", data=record.episode_index, dtype=np.int64)
    group.create_dataset("length", data=record.length, dtype=np.int64)
    group.create_dataset("rgb_main", shape=(record.length, 256, 256, 3), dtype=np.uint8)
    group.create_dataset(
        "rgb_wrist", shape=(record.length, 256, 256, 3), dtype=np.uint8
    )
    group.create_dataset("action", shape=(record.length, 7), dtype=np.float32)
    group.create_dataset("state", shape=(record.length, 8), dtype=np.float32)
    return group


def test_manifest_accepts_canonical_libero_contract(tmp_path):
    payload = make_manifest(tmp_path)
    records = schema.validate_manifest(payload, tmp_path)
    assert records == [
        schema.EpisodeRecord(
            key="libero_goal:000010",
            shard_path=(tmp_path / "shard_00000.h5").resolve(),
            group="episodes/libero_goal:000010",
            caption="pick up the red mug",
            domain="libero_goal",
            episode_index=10,
            length=3,
        )
    ]


def test_manifest_fields_are_exact_canonical_top_level_contract():
    assert schema.MANIFEST_FIELDS == EXPECTED_MANIFEST_FIELDS


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("camera_names", ["wrist", "main"]),
        ("image_size", [512, 512]),
        ("source_fps", 30),
        ("source_fps", 20.0),
        ("n_previous", 3),
        ("chunk", 8),
        ("action_chunk", 32),
        ("action_type", "relative"),
        ("action_space", "joint"),
    ],
)
def test_manifest_rejects_wrong_top_level_contract_field(tmp_path, field, value):
    payload = make_manifest(tmp_path)
    payload[field] = value
    with pytest.raises(ValueError, match=field):
        schema.validate_manifest(payload, tmp_path)


@pytest.mark.parametrize("version", [None, True, 2, 1.0])
def test_manifest_rejects_unsupported_schema_version(tmp_path, version):
    payload = make_manifest(tmp_path, schema_version=version)
    with pytest.raises(ValueError, match="schema_version"):
        schema.validate_manifest(payload, tmp_path)


@pytest.mark.parametrize(
    "field",
    sorted(EXPECTED_MANIFEST_FIELDS),
)
def test_manifest_rejects_missing_required_section(tmp_path, field):
    payload = make_manifest(tmp_path)
    del payload[field]
    with pytest.raises(ValueError, match=field):
        schema.validate_manifest(payload, tmp_path)


def test_manifest_rejects_unexpected_top_level_field(tmp_path):
    payload = make_manifest(tmp_path)
    payload["unexpected"] = "value"
    with pytest.raises(ValueError, match="unexpected.*unexpected"):
        schema.validate_manifest(payload, tmp_path)


@pytest.mark.parametrize("payload", [None, [], "manifest"])
def test_manifest_rejects_wrong_top_level_type(tmp_path, payload):
    with pytest.raises(ValueError, match="manifest payload must be a dict"):
        schema.validate_manifest(payload, tmp_path)


@pytest.mark.parametrize("compression", [None, "gzip", 1])
def test_manifest_rejects_wrong_compression(tmp_path, compression):
    payload = make_manifest(tmp_path, compression=compression)
    with pytest.raises(ValueError, match="compression"):
        schema.validate_manifest(payload, tmp_path)


@pytest.mark.parametrize("source_roots", [[], "source", [1], [""]])
def test_manifest_rejects_bad_source_roots(tmp_path, source_roots):
    payload = make_manifest(tmp_path, source_roots=source_roots)
    with pytest.raises(ValueError, match="source_roots"):
        schema.validate_manifest(payload, tmp_path)


@pytest.mark.parametrize(
    "fingerprint",
    [None, "a" * 63, "A" * 64, "g" * 64, 1],
)
def test_manifest_rejects_bad_converter_fingerprint(tmp_path, fingerprint):
    payload = make_manifest(tmp_path, converter_fingerprint=fingerprint)
    with pytest.raises(ValueError, match="converter_fingerprint"):
        schema.validate_manifest(payload, tmp_path)


@pytest.mark.parametrize(
    "datasets",
    [
        None,
        {key: value for key, value in DATASET_DECLARATIONS.items() if key != "state"},
        dict(DATASET_DECLARATIONS, state={"width": 9, "dtype": "float32"}),
        dict(DATASET_DECLARATIONS, state={"width": 8.0, "dtype": "float32"}),
        dict(DATASET_DECLARATIONS, action={"width": 7, "dtype": "float64"}),
        dict(
            DATASET_DECLARATIONS,
            rgb_main={"shape_tail": [128, 256, 3], "dtype": "uint8"},
        ),
        dict(DATASET_DECLARATIONS, extra={"dtype": "uint8"}),
    ],
)
def test_manifest_rejects_bad_dataset_declarations(tmp_path, datasets):
    payload = make_manifest(tmp_path, datasets=datasets)
    with pytest.raises(ValueError, match="datasets"):
        schema.validate_manifest(payload, tmp_path)


def test_manifest_rejects_duplicate_episode_keys(tmp_path):
    payload = make_manifest(tmp_path)
    payload["episodes"].append(dict(payload["episodes"][0]))
    with pytest.raises(ValueError, match="duplicate episode key"):
        schema.validate_manifest(payload, tmp_path)


@pytest.mark.parametrize("record", [None, [], "episode"])
def test_manifest_rejects_wrong_episode_record_type(tmp_path, record):
    payload = make_manifest(tmp_path, episodes=[record])
    with pytest.raises(ValueError, match="episode 0 must be a dict"):
        schema.validate_manifest(payload, tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("key", 10),
        ("shard", Path("shard_00000.h5")),
        ("group", 10),
        ("caption", 10),
        ("domain", 10),
        ("episode_index", True),
        ("episode_index", 10.0),
        ("length", True),
        ("length", 3.0),
    ],
)
def test_manifest_rejects_wrong_episode_field_type(tmp_path, field, value):
    payload = make_manifest(tmp_path)
    payload["episodes"][0][field] = value
    with pytest.raises(ValueError, match=rf"episode 0 field {field}"):
        schema.validate_manifest(payload, tmp_path)


def test_manifest_rejects_missing_episode_field(tmp_path):
    payload = make_manifest(tmp_path)
    del payload["episodes"][0]["caption"]
    with pytest.raises(ValueError, match="episode 0.*caption"):
        schema.validate_manifest(payload, tmp_path)


def test_manifest_rejects_extra_episode_field(tmp_path):
    payload = make_manifest(tmp_path)
    payload["episodes"][0]["unexpected"] = "value"
    with pytest.raises(ValueError, match="episode 0.*unexpected"):
        schema.validate_manifest(payload, tmp_path)


def test_manifest_rejects_group_that_does_not_encode_episode_key(tmp_path):
    payload = make_manifest(tmp_path)
    payload["episodes"][0]["group"] = "episodes/000010"
    with pytest.raises(ValueError, match="group"):
        schema.validate_manifest(payload, tmp_path)


def test_manifest_rejects_missing_shard(tmp_path):
    payload = make_manifest(tmp_path)
    (tmp_path / "shard_00000.h5").unlink()
    with pytest.raises(FileNotFoundError, match="missing HDF5 shard"):
        schema.validate_manifest(payload, tmp_path)


def test_manifest_rejects_absolute_shard_path(tmp_path):
    outside = tmp_path.parent / "outside-absolute.h5"
    outside.touch()
    payload = make_manifest(tmp_path)
    payload["episodes"][0]["shard"] = str(outside.resolve())
    with pytest.raises(ValueError, match="shard.*relative"):
        schema.validate_manifest(payload, tmp_path)


def test_manifest_rejects_parent_shard_escape(tmp_path):
    root = tmp_path / "root"
    payload = make_manifest(root)
    outside = tmp_path / "outside-parent.h5"
    outside.touch()
    payload["episodes"][0]["shard"] = "../outside-parent.h5"
    with pytest.raises(ValueError, match="shard.*outside manifest root"):
        schema.validate_manifest(payload, root)


def test_manifest_rejects_symlink_shard_escape(tmp_path):
    root = tmp_path / "root"
    payload = make_manifest(root)
    outside = tmp_path / "outside-symlink.h5"
    outside.touch()
    link = root / "escape.h5"
    link.symlink_to(outside)
    payload["episodes"][0]["shard"] = link.name
    with pytest.raises(ValueError, match="shard.*outside manifest root"):
        schema.validate_manifest(payload, root)


def test_load_manifest_reads_json_and_validates_relative_to_parent(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    payload = make_manifest(tmp_path)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded_payload, records = schema.load_manifest(manifest_path)
    assert loaded_payload == payload
    assert records[0].shard_path == (tmp_path / "shard_00000.h5").resolve()
    assert records[0].caption == "pick up the red mug"


def test_validate_episode_group_accepts_matching_hdf5_data(tmp_path):
    record = schema.validate_manifest(make_manifest(tmp_path), tmp_path)[0]
    with h5py.File(record.shard_path, "w") as file:
        group = make_episode_group(file, record)
        schema.validate_episode_group(group, record)


def test_validate_episode_group_requires_scalar_datasets_not_attributes(tmp_path):
    record = schema.validate_manifest(make_manifest(tmp_path), tmp_path)[0]
    with h5py.File(record.shard_path, "w") as file:
        group = make_episode_group(file, record)
        for name in ("caption", "domain", "episode_index", "length"):
            value = group[name][()]
            del group[name]
            group.attrs[name] = value
        with pytest.raises(ValueError, match="caption"):
            schema.validate_episode_group(group, record)


@pytest.mark.parametrize(
    ("dataset", "value"),
    [
        ("caption", "wrong caption"),
        ("domain", "wrong_domain"),
        ("episode_index", np.int64(11)),
        ("length", np.int64(4)),
    ],
)
def test_validate_episode_group_rejects_wrong_scalar_metadata(tmp_path, dataset, value):
    record = schema.validate_manifest(make_manifest(tmp_path), tmp_path)[0]
    with h5py.File(record.shard_path, "w") as file:
        group = make_episode_group(file, record)
        del group[dataset]
        dtype = h5py.string_dtype("utf-8") if isinstance(value, str) else value.dtype
        group.create_dataset(dataset, data=value, dtype=dtype)
        with pytest.raises(ValueError, match=dataset):
            schema.validate_episode_group(group, record)


@pytest.mark.parametrize("dataset", ["caption", "domain", "episode_index", "length"])
def test_validate_episode_group_rejects_missing_scalar_metadata(tmp_path, dataset):
    record = schema.validate_manifest(make_manifest(tmp_path), tmp_path)[0]
    with h5py.File(record.shard_path, "w") as file:
        group = make_episode_group(file, record)
        del group[dataset]
        with pytest.raises(ValueError, match=dataset):
            schema.validate_episode_group(group, record)


@pytest.mark.parametrize(
    ("dataset", "value", "dtype"),
    [
        ("caption", ["pick up the red mug"], h5py.string_dtype("utf-8")),
        ("domain", np.bytes_("libero_goal"), "S16"),
        ("episode_index", np.int32(10), np.int32),
        ("length", [3], np.int64),
    ],
)
def test_validate_episode_group_rejects_wrong_scalar_shape_or_dtype(
    tmp_path, dataset, value, dtype
):
    record = schema.validate_manifest(make_manifest(tmp_path), tmp_path)[0]
    with h5py.File(record.shard_path, "w") as file:
        group = make_episode_group(file, record)
        del group[dataset]
        group.create_dataset(dataset, data=value, dtype=dtype)
        with pytest.raises(ValueError, match=dataset):
            schema.validate_episode_group(group, record)


def test_validate_episode_group_rejects_wrong_group_path(tmp_path):
    record = schema.validate_manifest(make_manifest(tmp_path), tmp_path)[0]
    with h5py.File(record.shard_path, "w") as file:
        group = file.create_group("episodes/wrong")
        with pytest.raises(ValueError, match="group path"):
            schema.validate_episode_group(group, record)


@pytest.mark.parametrize(
    ("dataset", "shape", "dtype"),
    [
        ("rgb_main", (3, 128, 256, 3), np.uint8),
        ("rgb_wrist", (3, 256, 256, 3), np.float32),
        ("action", (3, 7), np.float64),
        ("state", (3, 8), np.float64),
    ],
)
def test_validate_episode_group_rejects_wrong_tensor_shape_or_dtype(
    tmp_path, dataset, shape, dtype
):
    record = schema.validate_manifest(make_manifest(tmp_path), tmp_path)[0]
    with h5py.File(record.shard_path, "w") as file:
        group = make_episode_group(file, record)
        del group[dataset]
        group.create_dataset(dataset, shape=shape, dtype=dtype)
        with pytest.raises(ValueError, match=dataset):
            schema.validate_episode_group(group, record)


@pytest.mark.parametrize(
    ("dataset", "shape"),
    [
        ("action", (3,)),
        ("action", (3, 6)),
        ("action", (3, 7, 1)),
        ("state", (3,)),
        ("state", (3, 9)),
        ("state", (3, 8, 1)),
    ],
)
def test_validate_episode_group_rejects_wrong_control_rank_or_width(
    tmp_path, dataset, shape
):
    record = schema.validate_manifest(make_manifest(tmp_path), tmp_path)[0]
    with h5py.File(record.shard_path, "w") as file:
        group = make_episode_group(file, record)
        del group[dataset]
        group.create_dataset(dataset, shape=shape, dtype=np.float32)
        with pytest.raises(ValueError, match=dataset):
            schema.validate_episode_group(group, record)


def test_atomic_write_manifest_replaces_target_without_temp_files(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("old", encoding="utf-8")
    payload = make_manifest(tmp_path)
    schema.atomic_write_manifest(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert set(tmp_path.iterdir()) == {
        path,
        tmp_path / "shard_00000.h5",
    }


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


def test_atomic_write_manifest_removes_temp_file_when_serialization_fails(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("old", encoding="utf-8")
    payload = make_manifest(tmp_path)
    payload["not_json_serializable"] = {"a set"}
    with pytest.raises(TypeError, match="JSON serializable"):
        schema.atomic_write_manifest(path, payload)
    assert path.read_text(encoding="utf-8") == "old"
    assert set(tmp_path.iterdir()) == {path, tmp_path / "shard_00000.h5"}


CAMERAS = ("observation.images.image", "observation.images.wrist_image")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _synthetic_rgb(episode_index: int, length: int, camera_index: int) -> np.ndarray:
    values = np.arange(length * 8 * 10 * 3, dtype=np.uint32).reshape(length, 8, 10, 3)
    return ((values + episode_index * 17 + camera_index * 73) % 256).astype(np.uint8)


def _write_mp4(path: Path, frames: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=20)
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = "yuv420p"
        for array in frames:
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def make_tiny_lerobot_domain(
    data_root: Path,
    domain: str,
    *,
    episode_indexes: tuple[int, ...] = (2, 0, 1),
    length: int = 3,
    write_mp4: bool = False,
    task_lists: dict[int, list[object]] | None = None,
) -> Path:
    domain_root = data_root / domain
    meta = domain_root / "meta"
    _write_jsonl(
        meta / "tasks.jsonl",
        [
            {"task_index": 0, "task": f"perform {domain} task zero"},
            {"task_index": 1, "task": f"perform {domain} task one"},
        ],
    )
    _write_jsonl(
        meta / "episodes.jsonl",
        [
            {
                "episode_index": episode_index,
                "tasks": (task_lists or {}).get(
                    episode_index,
                    [
                        f"perform {domain} task {'zero' if episode_index % 2 == 0 else 'one'}"
                    ],
                ),
                "length": length,
            }
            for episode_index in episode_indexes
        ],
    )
    meta.joinpath("info.json").write_text(
        json.dumps({"chunks_size": 1000, "total_chunks": 1}),
        encoding="utf-8",
    )
    for episode_index in episode_indexes:
        parquet = (
            domain_root / "data/chunk-000" / f"episode_{episode_index:06d}.parquet"
        )
        parquet.parent.mkdir(parents=True, exist_ok=True)
        action = [
            np.arange(7, dtype=np.float32) + episode_index + step / 10
            for step in range(length)
        ]
        state = [
            np.arange(8, dtype=np.float32) + episode_index + step / 5
            for step in range(length)
        ]
        pd.DataFrame({"action": action, "observation.state": state}).to_parquet(parquet)
        for camera_index, camera in enumerate(CAMERAS):
            video = (
                domain_root
                / "videos/chunk-000"
                / camera
                / f"episode_{episode_index:06d}.mp4"
            )
            frames = _synthetic_rgb(episode_index, length, camera_index)
            if write_mp4:
                _write_mp4(video, frames)
            else:
                video.parent.mkdir(parents=True, exist_ok=True)
                video.touch()
    return data_root


def make_tiny_predecoded_cache(
    cache_root: Path,
    data_root: Path,
    domain: str,
    episode_indexes: tuple[int, ...],
    *,
    length: int = 3,
) -> Path:
    for episode_index in episode_indexes:
        for camera_index, camera in enumerate(CAMERAS):
            destination = (
                cache_root
                / domain
                / "videos/chunk-000"
                / camera
                / f"episode_{episode_index:06d}.npy"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(
                destination,
                _synthetic_rgb(episode_index, length, camera_index),
                allow_pickle=False,
            )
    return cache_root


def make_convert_args(
    *,
    data_root: Path,
    domains: list[str],
    output_root: Path,
    predecoded_root: Path | None,
    max_episodes: int | None = None,
    episodes_per_shard: int = 2,
    compression: str = "none",
    overwrite: bool = False,
) -> Namespace:
    return Namespace(
        data_root=[str(data_root)],
        domains=domains,
        predecoded_root=(None if predecoded_root is None else str(predecoded_root)),
        output_root=str(output_root),
        max_episodes=max_episodes,
        episodes_per_shard=episodes_per_shard,
        compression=compression,
        resize_microbatch=2,
        overwrite=overwrite,
    )


def test_resize_rgb_uint8_matches_float_resize_with_quantization_bound():
    frames = _synthetic_rgb(0, 5, 0)
    actual = converter.resize_rgb_uint8(frames, size=(256, 256), microbatch=2)
    reference = tvf.resize(
        torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0,
        [256, 256],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    restored = torch.from_numpy(actual).permute(0, 3, 1, 2).float() / 255.0
    assert actual.shape == (5, 256, 256, 3)
    assert actual.dtype == np.uint8
    assert (restored - reference).abs().max() <= 0.5 / 255.0 + 1e-6


@pytest.mark.parametrize(
    "frames",
    [
        np.empty((0, 8, 10, 3), dtype=np.uint8),
        np.empty((2, 8, 10), dtype=np.uint8),
        np.empty((2, 8, 10, 4), dtype=np.uint8),
        np.empty((2, 8, 10, 3), dtype=np.float32),
    ],
)
def test_resize_rgb_uint8_rejects_invalid_source_arrays(frames):
    with pytest.raises(ValueError, match="RGB"):
        converter.resize_rgb_uint8(frames)


@pytest.mark.parametrize("microbatch", [0, -1, True])
def test_resize_rgb_uint8_rejects_invalid_microbatch(microbatch):
    with pytest.raises(ValueError, match="microbatch"):
        converter.resize_rgb_uint8(_synthetic_rgb(0, 2, 0), microbatch=microbatch)


def test_discovery_is_deterministic_resolves_captions_and_deduplicates_pairs(
    tmp_path,
):
    source = make_tiny_lerobot_domain(tmp_path / "source", "domain_b")
    make_tiny_lerobot_domain(source, "domain_a", episode_indexes=(3, 1))
    cache = make_tiny_predecoded_cache(
        tmp_path / "cache", source, "domain_b", (2, 0, 1)
    )
    make_tiny_predecoded_cache(cache, source, "domain_a", (3, 1))

    episodes = converter.discover_source_episodes(
        [source, source, source],
        ["domain_b", "domain_b", "domain_a"],
        predecoded_root=cache,
    )

    assert [episode.key for episode in episodes] == [
        "domain_b:000000",
        "domain_b:000001",
        "domain_b:000002",
        "domain_a:000001",
        "domain_a:000003",
    ]
    assert episodes[0].caption == "perform domain_b task zero"
    assert episodes[1].caption == "perform domain_b task one"
    assert (
        episodes[0].main_cache_path
        == (
            cache
            / "domain_b/videos/chunk-000/observation.images.image/episode_000000.npy"
        ).resolve()
    )


def test_discovery_applies_max_episodes_after_ordering(tmp_path):
    source = make_tiny_lerobot_domain(tmp_path / "source", "domain_b")
    make_tiny_lerobot_domain(source, "domain_a", episode_indexes=(3, 1))
    cache = make_tiny_predecoded_cache(
        tmp_path / "cache", source, "domain_b", (2, 0, 1)
    )
    make_tiny_predecoded_cache(cache, source, "domain_a", (3, 1))
    episodes = converter.discover_source_episodes(
        [source],
        ["domain_b", "domain_a"],
        predecoded_root=cache,
        max_episodes=4,
    )
    assert [episode.key for episode in episodes] == [
        "domain_b:000000",
        "domain_b:000001",
        "domain_b:000002",
        "domain_a:000001",
    ]


def test_discovery_rejects_conflicting_roots_with_same_final_key(tmp_path):
    first = make_tiny_lerobot_domain(tmp_path / "first", "domain", episode_indexes=(0,))
    second = make_tiny_lerobot_domain(
        tmp_path / "second", "domain", episode_indexes=(0,)
    )
    first_cache = make_tiny_predecoded_cache(tmp_path / "cache", first, "domain", (0,))
    make_tiny_predecoded_cache(first_cache, second, "domain", (0,))
    with pytest.raises(ValueError, match="duplicate final episode key"):
        converter.discover_source_episodes(
            [first, second], ["domain", "domain"], predecoded_root=first_cache
        )


@pytest.mark.parametrize("missing", ["parquet", "main", "wrist"])
def test_discovery_rejects_missing_required_episode_inputs(tmp_path, missing):
    source = make_tiny_lerobot_domain(
        tmp_path / "source", "domain", episode_indexes=(0,)
    )
    cache = make_tiny_predecoded_cache(tmp_path / "cache", source, "domain", (0,))
    if missing == "parquet":
        source.joinpath("domain/data/chunk-000/episode_000000.parquet").unlink()
    else:
        camera = CAMERAS[0 if missing == "main" else 1]
        cache.joinpath(f"domain/videos/chunk-000/{camera}/episode_000000.npy").unlink()
    with pytest.raises(FileNotFoundError, match=f"{missing}|parquet|camera"):
        converter.discover_source_episodes([source], ["domain"], predecoded_root=cache)


def test_discovery_rejects_zero_or_multiple_tasks(tmp_path):
    for tasks in ([], ["perform domain task zero", "perform domain task one"]):
        source = make_tiny_lerobot_domain(
            tmp_path / f"source-{len(tasks)}",
            "domain",
            episode_indexes=(0,),
            task_lists={0: tasks},
        )
        cache = make_tiny_predecoded_cache(
            tmp_path / f"cache-{len(tasks)}", source, "domain", (0,)
        )
        with pytest.raises(ValueError, match="exactly one task"):
            converter.discover_source_episodes(
                [source], ["domain"], predecoded_root=cache
            )


@pytest.mark.parametrize(
    ("tasks", "message"),
    [
        (["unknown task text"], "unknown task text"),
        ([""], "non-empty task text"),
        ([0], "non-empty task text"),
        ([None], "non-empty task text"),
    ],
)
def test_discovery_requires_one_known_non_empty_task_text(tmp_path, tasks, message):
    source = make_tiny_lerobot_domain(
        tmp_path / "source",
        "domain",
        episode_indexes=(0,),
        task_lists={0: tasks},
    )
    cache = make_tiny_predecoded_cache(tmp_path / "cache", source, "domain", (0,))
    with pytest.raises(ValueError, match=message):
        converter.discover_source_episodes([source], ["domain"], predecoded_root=cache)


@pytest.mark.parametrize(
    ("column", "replacement", "message"),
    [
        ("action", [np.zeros(6, np.float32)] * 3, "action"),
        ("observation.state", [np.zeros(9, np.float32)] * 3, "state"),
        ("action", [np.zeros(7, np.float32)] * 2, "row count"),
    ],
)
def test_converter_rejects_wrong_control_width_or_row_count(
    tmp_path, column, replacement, message
):
    source = make_tiny_lerobot_domain(
        tmp_path / "source", "domain", episode_indexes=(0,)
    )
    cache = make_tiny_predecoded_cache(tmp_path / "cache", source, "domain", (0,))
    parquet = source / "domain/data/chunk-000/episode_000000.parquet"
    frame = pd.read_parquet(parquet)
    if len(replacement) != len(frame):
        frame = frame.iloc[: len(replacement)].copy()
    frame[column] = replacement
    frame.to_parquet(parquet)
    with pytest.raises(ValueError, match=message):
        converter.convert_dataset(
            make_convert_args(
                data_root=source,
                domains=["domain"],
                output_root=tmp_path / "out",
                predecoded_root=cache,
            )
        )
    assert not (tmp_path / "out/manifest.json").exists()


def test_converter_reports_non_numeric_control_conversion_context(tmp_path):
    source = make_tiny_lerobot_domain(
        tmp_path / "source", "domain", episode_indexes=(0,)
    )
    cache = make_tiny_predecoded_cache(tmp_path / "cache", source, "domain", (0,))
    parquet = source / "domain/data/chunk-000/episode_000000.parquet"
    frame = pd.read_parquet(parquet)
    frame["action"] = [["not-a-number"] * 7 for _ in range(len(frame))]
    frame.to_parquet(parquet)

    with pytest.raises(ValueError, match="invalid action values") as error:
        converter.convert_dataset(
            make_convert_args(
                data_root=source,
                domains=["domain"],
                output_root=tmp_path / "out",
                predecoded_root=cache,
            )
        )

    message = str(error.value)
    assert "domain=domain" in message
    assert "episode=0" in message
    assert str(parquet) in message


@pytest.mark.parametrize("compression", ["none", "lzf"])
def test_converter_writes_atomic_shards_and_exact_manifest(tmp_path, compression):
    source = make_tiny_lerobot_domain(tmp_path / "source", "domain")
    cache = make_tiny_predecoded_cache(tmp_path / "cache", source, "domain", (2, 0, 1))
    output = tmp_path / f"out-{compression}"
    report = converter.convert_dataset(
        make_convert_args(
            data_root=source,
            domains=["domain"],
            output_root=output,
            predecoded_root=cache,
            max_episodes=3,
            episodes_per_shard=2,
            compression=compression,
        )
    )
    assert report == {"episodes": 3, "shards": 2, "compression": compression}
    payload, records = schema.load_manifest(output / "manifest.json")
    assert payload["compression"] == compression
    assert payload["source_roots"] == [str(source.resolve())]
    assert [record.episode_index for record in records] == [0, 1, 2]
    assert len({record.shard_path for record in records}) == 2
    assert not list(output.glob("*.tmp*"))
    for shard_path in {record.shard_path for record in records}:
        with h5py.File(shard_path, "r") as shard:
            assert shard.attrs["schema_version"] == schema.SCHEMA_VERSION
            assert shard.attrs["compression"] == compression
            assert shard.attrs["image_height"] == 256
            assert shard.attrs["image_width"] == 256
            assert list(shard.attrs["camera_names"].astype(str)) == [
                "main",
                "wrist",
            ]
            for record in records:
                if record.shard_path != shard_path:
                    continue
                group = shard[record.group]
                schema.validate_episode_group(group, record)
                expected_compression = None if compression == "none" else "lzf"
                assert group["rgb_main"].compression == expected_compression
                assert group["rgb_main"].chunks == (1, 256, 256, 3)
                assert group["rgb_wrist"].chunks == (1, 256, 256, 3)
                assert group["rgb_wrist"].compression == expected_compression
                assert group["action"].chunks == (min(64, record.length), 7)
                assert group["action"].compression is None
                assert group["state"].chunks == (min(64, record.length), 8)
                assert group["state"].compression is None
                for scalar in ("caption", "domain", "episode_index", "length"):
                    assert group[scalar].chunks is None
                    assert group[scalar].compression is None


def test_converter_manifest_records_all_configured_source_roots(tmp_path):
    first = make_tiny_lerobot_domain(
        tmp_path / "first", "domain_a", episode_indexes=(0,)
    )
    second = make_tiny_lerobot_domain(
        tmp_path / "second", "domain_b", episode_indexes=(0,)
    )
    cache = make_tiny_predecoded_cache(tmp_path / "cache", first, "domain_a", (0,))
    make_tiny_predecoded_cache(cache, second, "domain_b", (0,))
    args = make_convert_args(
        data_root=first,
        domains=["domain_a", "domain_b"],
        output_root=tmp_path / "out",
        predecoded_root=cache,
        max_episodes=1,
    )
    args.data_root = [str(first), str(second)]
    converter.convert_dataset(args)
    payload, _ = schema.load_manifest(tmp_path / "out/manifest.json")
    assert payload["source_roots"] == [
        str(first.resolve()),
        str(second.resolve()),
    ]


def test_converter_strict_cache_requires_exact_frame_count(tmp_path):
    source = make_tiny_lerobot_domain(
        tmp_path / "source", "domain", episode_indexes=(0,)
    )
    cache = make_tiny_predecoded_cache(
        tmp_path / "cache", source, "domain", (0,), length=2
    )
    with pytest.raises(ValueError, match="frame count"):
        converter.convert_dataset(
            make_convert_args(
                data_root=source,
                domains=["domain"],
                output_root=tmp_path / "out",
                predecoded_root=cache,
            )
        )


def test_converter_decodes_mp4_when_cache_is_absent(tmp_path):
    source = make_tiny_lerobot_domain(
        tmp_path / "source",
        "domain",
        episode_indexes=(0,),
        write_mp4=True,
    )
    output = tmp_path / "out"
    converter.convert_dataset(
        make_convert_args(
            data_root=source,
            domains=["domain"],
            output_root=output,
            predecoded_root=None,
        )
    )
    _, records = schema.load_manifest(output / "manifest.json")
    assert len(records) == 1
    with h5py.File(records[0].shard_path, "r") as shard:
        assert shard[records[0].group]["rgb_main"].shape == (3, 256, 256, 3)


def test_converter_failure_removes_new_shards_temps_and_manifest(tmp_path, monkeypatch):
    source = make_tiny_lerobot_domain(tmp_path / "source", "domain")
    cache = make_tiny_predecoded_cache(tmp_path / "cache", source, "domain", (2, 0, 1))
    output = tmp_path / "out"
    real_validate = schema.validate_episode_group
    calls = 0

    def fail_in_second_shard(group, record):
        nonlocal calls
        calls += 1
        if record.episode_index == 2:
            raise ValueError("injected shard validation failure")
        return real_validate(group, record)

    monkeypatch.setattr(schema, "validate_episode_group", fail_in_second_shard)
    with pytest.raises(ValueError, match="injected shard validation failure"):
        converter.convert_dataset(
            make_convert_args(
                data_root=source,
                domains=["domain"],
                output_root=output,
                predecoded_root=cache,
                episodes_per_shard=2,
            )
        )
    assert not (output / "manifest.json").exists()
    assert not list(output.glob("*.h5"))
    assert not list(output.glob("*.tmp*"))


def test_converter_reuses_valid_matching_output_and_guards_fingerprint(
    tmp_path,
):
    source = make_tiny_lerobot_domain(
        tmp_path / "source", "domain", episode_indexes=(0,)
    )
    cache = make_tiny_predecoded_cache(tmp_path / "cache", source, "domain", (0,))
    output = tmp_path / "out"
    args = make_convert_args(
        data_root=source,
        domains=["domain"],
        output_root=output,
        predecoded_root=cache,
    )
    converter.convert_dataset(args)
    _, records = schema.load_manifest(output / "manifest.json")
    shard = records[0].shard_path
    before = shard.stat().st_mtime_ns
    assert converter.convert_dataset(args) == {
        "episodes": 1,
        "shards": 1,
        "compression": "none",
    }
    assert shard.stat().st_mtime_ns == before

    lzf_args = make_convert_args(
        data_root=source,
        domains=["domain"],
        output_root=output,
        predecoded_root=cache,
        compression="lzf",
    )
    with pytest.raises(ValueError, match="fingerprint"):
        converter.convert_dataset(lzf_args)
    lzf_args.overwrite = True
    converter.convert_dataset(lzf_args)
    payload, _ = schema.load_manifest(output / "manifest.json")
    assert payload["compression"] == "lzf"


def test_published_reuse_rejects_coordinated_compression_tamper(tmp_path):
    source = make_tiny_lerobot_domain(
        tmp_path / "source", "domain", episode_indexes=(0,)
    )
    cache = make_tiny_predecoded_cache(tmp_path / "cache", source, "domain", (0,))
    output = tmp_path / "out"
    args = make_convert_args(
        data_root=source,
        domains=["domain"],
        output_root=output,
        predecoded_root=cache,
        compression="none",
    )
    converter.convert_dataset(args)
    manifest_path = output / "manifest.json"
    payload, records = schema.load_manifest(manifest_path)
    record = records[0]
    payload["compression"] = "lzf"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with h5py.File(record.shard_path, "r+") as shard:
        shard.attrs["compression"] = "lzf"
    for dataset_name in ("rgb_main", "rgb_wrist"):
        _tamper_dataset_layout(
            record.shard_path,
            record.group,
            dataset_name,
            chunks=(1, 256, 256, 3),
            compression="lzf",
        )

    with pytest.raises(ValueError, match="compression"):
        converter.convert_dataset(args)


def test_published_reuse_rejects_source_roots_tamper(tmp_path):
    source = make_tiny_lerobot_domain(
        tmp_path / "source", "domain", episode_indexes=(0,)
    )
    cache = make_tiny_predecoded_cache(tmp_path / "cache", source, "domain", (0,))
    output = tmp_path / "out"
    args = make_convert_args(
        data_root=source,
        domains=["domain"],
        output_root=output,
        predecoded_root=cache,
    )
    converter.convert_dataset(args)
    manifest_path = output / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["source_roots"] = [str((tmp_path / "other-source").resolve())]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source_roots"):
        converter.convert_dataset(args)


def test_published_reuse_rejects_coordinated_episode_caption_tamper(tmp_path):
    source = make_tiny_lerobot_domain(
        tmp_path / "source", "domain", episode_indexes=(0,)
    )
    cache = make_tiny_predecoded_cache(tmp_path / "cache", source, "domain", (0,))
    output = tmp_path / "out"
    args = make_convert_args(
        data_root=source,
        domains=["domain"],
        output_root=output,
        predecoded_root=cache,
    )
    converter.convert_dataset(args)
    manifest_path = output / "manifest.json"
    payload, records = schema.load_manifest(manifest_path)
    record = records[0]
    payload["episodes"][0]["caption"] = "coordinated tampered caption"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with h5py.File(record.shard_path, "r+") as shard:
        group = shard[record.group]
        del group["caption"]
        group.create_dataset(
            "caption",
            data="coordinated tampered caption",
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    with pytest.raises(ValueError, match="episodes.*caption"):
        converter.convert_dataset(args)


def test_overwrite_matching_published_output_creates_fresh_generation(tmp_path):
    source = make_tiny_lerobot_domain(
        tmp_path / "source", "domain", episode_indexes=(0,)
    )
    cache = make_tiny_predecoded_cache(tmp_path / "cache", source, "domain", (0,))
    output = tmp_path / "out"
    args = make_convert_args(
        data_root=source,
        domains=["domain"],
        output_root=output,
        predecoded_root=cache,
    )
    converter.convert_dataset(args)
    _, old_records = schema.load_manifest(output / "manifest.json")
    old_shard = old_records[0].shard_path
    old_identity = (old_shard.stat().st_ino, old_shard.stat().st_mtime_ns)

    args.overwrite = True
    converter.convert_dataset(args)
    _, new_records = schema.load_manifest(output / "manifest.json")
    new_shard = new_records[0].shard_path

    assert new_shard != old_shard
    assert old_shard.exists()
    assert (old_shard.stat().st_ino, old_shard.stat().st_mtime_ns) == old_identity
    assert new_shard.exists()


def _tamper_dataset_layout(
    shard_path: Path,
    group_path: str,
    dataset_name: str,
    *,
    chunks,
    compression,
) -> None:
    with h5py.File(shard_path, "r+") as shard:
        group = shard[group_path]
        data = group[dataset_name][()]
        dtype = group[dataset_name].dtype
        del group[dataset_name]
        group.create_dataset(
            dataset_name,
            data=data,
            dtype=dtype,
            chunks=chunks,
            compression=compression,
        )


@pytest.mark.parametrize(
    ("dataset_name", "chunks", "compression", "message"),
    [
        ("rgb_main", (2, 256, 256, 3), None, "rgb_main.*chunks"),
        ("rgb_wrist", (1, 256, 256, 3), "lzf", "rgb_wrist.*compression"),
        ("action", (1, 7), None, "action.*chunks"),
        ("state", (1, 8), None, "state.*chunks"),
    ],
)
def test_converter_reuse_rejects_tampered_dataset_storage_layout(
    tmp_path, dataset_name, chunks, compression, message
):
    source = make_tiny_lerobot_domain(
        tmp_path / "source", "domain", episode_indexes=(0,)
    )
    cache = make_tiny_predecoded_cache(tmp_path / "cache", source, "domain", (0,))
    output = tmp_path / "out"
    args = make_convert_args(
        data_root=source,
        domains=["domain"],
        output_root=output,
        predecoded_root=cache,
    )
    converter.convert_dataset(args)
    _, records = schema.load_manifest(output / "manifest.json")
    record = records[0]
    _tamper_dataset_layout(
        record.shard_path,
        record.group,
        dataset_name,
        chunks=chunks,
        compression=compression,
    )

    with pytest.raises(ValueError, match=message) as error:
        converter.convert_dataset(args)
    assert str(record.shard_path) in str(error.value)


def test_converter_resumes_valid_orphan_shards_without_rewriting_them(tmp_path):
    source = make_tiny_lerobot_domain(tmp_path / "source", "domain")
    cache = make_tiny_predecoded_cache(tmp_path / "cache", source, "domain", (2, 0, 1))
    output = tmp_path / "out"
    args = make_convert_args(
        data_root=source,
        domains=["domain"],
        output_root=output,
        predecoded_root=cache,
        episodes_per_shard=2,
    )
    converter.convert_dataset(args)
    _, records = schema.load_manifest(output / "manifest.json")
    shards = sorted({record.shard_path for record in records})
    reused = shards[0]
    reused_identity = (reused.stat().st_ino, reused.stat().st_mtime_ns)
    (output / "manifest.json").unlink()
    shards[1].unlink()

    assert converter.convert_dataset(args) == {
        "episodes": 3,
        "shards": 2,
        "compression": "none",
    }
    _, resumed_records = schema.load_manifest(output / "manifest.json")
    assert len({record.shard_path for record in resumed_records}) == 2
    assert (reused.stat().st_ino, reused.stat().st_mtime_ns) == reused_identity


def test_overwrite_valid_orphan_creates_fresh_generation(tmp_path):
    source = make_tiny_lerobot_domain(
        tmp_path / "source", "domain", episode_indexes=(0,)
    )
    cache = make_tiny_predecoded_cache(tmp_path / "cache", source, "domain", (0,))
    output = tmp_path / "out"
    args = make_convert_args(
        data_root=source,
        domains=["domain"],
        output_root=output,
        predecoded_root=cache,
    )
    converter.convert_dataset(args)
    _, old_records = schema.load_manifest(output / "manifest.json")
    old_shard = old_records[0].shard_path
    old_identity = (old_shard.stat().st_ino, old_shard.stat().st_mtime_ns)
    (output / "manifest.json").unlink()

    args.overwrite = True
    converter.convert_dataset(args)
    _, new_records = schema.load_manifest(output / "manifest.json")
    new_shard = new_records[0].shard_path

    assert new_shard != old_shard
    assert old_shard.exists()
    assert (old_shard.stat().st_ino, old_shard.stat().st_mtime_ns) == old_identity
    assert new_shard.exists()


def test_manifest_failure_keeps_reused_orphan_and_cleans_new_shard(
    tmp_path, monkeypatch
):
    source = make_tiny_lerobot_domain(tmp_path / "source", "domain")
    cache = make_tiny_predecoded_cache(tmp_path / "cache", source, "domain", (2, 0, 1))
    output = tmp_path / "out"
    args = make_convert_args(
        data_root=source,
        domains=["domain"],
        output_root=output,
        predecoded_root=cache,
        episodes_per_shard=2,
    )
    converter.convert_dataset(args)
    _, records = schema.load_manifest(output / "manifest.json")
    shards = sorted({record.shard_path for record in records})
    reused = shards[0]
    reused_identity = (reused.stat().st_ino, reused.stat().st_mtime_ns)
    (output / "manifest.json").unlink()
    shards[1].unlink()

    def fail_manifest(*_args, **_kwargs):
        raise OSError("injected manifest publication failure")

    monkeypatch.setattr(schema, "atomic_write_manifest", fail_manifest)
    with pytest.raises(OSError, match="manifest publication failure"):
        converter.convert_dataset(args)

    assert not (output / "manifest.json").exists()
    assert (reused.stat().st_ino, reused.stat().st_mtime_ns) == reused_identity
    assert set(output.glob("*.h5")) == {reused}
    assert not list(output.glob("*.tmp*"))


def test_invalid_orphan_shard_is_rejected_without_overwrite(tmp_path):
    source = make_tiny_lerobot_domain(
        tmp_path / "source", "domain", episode_indexes=(0,)
    )
    cache = make_tiny_predecoded_cache(tmp_path / "cache", source, "domain", (0,))
    output = tmp_path / "out"
    args = make_convert_args(
        data_root=source,
        domains=["domain"],
        output_root=output,
        predecoded_root=cache,
    )
    converter.convert_dataset(args)
    _, records = schema.load_manifest(output / "manifest.json")
    record = records[0]
    (output / "manifest.json").unlink()
    _tamper_dataset_layout(
        record.shard_path,
        record.group,
        "action",
        chunks=(1, 7),
        compression=None,
    )

    with pytest.raises(ValueError, match="invalid orphan shard") as error:
        converter.convert_dataset(args)
    assert str(record.shard_path) in str(error.value)
    assert record.shard_path.exists()
    assert not (output / "manifest.json").exists()


def test_overwrite_replaces_invalid_orphan_with_fresh_generation(tmp_path):
    source = make_tiny_lerobot_domain(
        tmp_path / "source", "domain", episode_indexes=(0,)
    )
    cache = make_tiny_predecoded_cache(tmp_path / "cache", source, "domain", (0,))
    output = tmp_path / "out"
    args = make_convert_args(
        data_root=source,
        domains=["domain"],
        output_root=output,
        predecoded_root=cache,
    )
    converter.convert_dataset(args)
    _, old_records = schema.load_manifest(output / "manifest.json")
    old_record = old_records[0]
    (output / "manifest.json").unlink()
    _tamper_dataset_layout(
        old_record.shard_path,
        old_record.group,
        "action",
        chunks=(1, 7),
        compression=None,
    )
    args.overwrite = True

    converter.convert_dataset(args)
    _, new_records = schema.load_manifest(output / "manifest.json")
    assert new_records[0].shard_path != old_record.shard_path
    assert old_record.shard_path.exists()
    assert new_records[0].shard_path.exists()


def test_failed_overwrite_keeps_previous_manifest_and_shards_readable(
    tmp_path, monkeypatch
):
    source = make_tiny_lerobot_domain(
        tmp_path / "source", "domain", episode_indexes=(0,)
    )
    cache = make_tiny_predecoded_cache(tmp_path / "cache", source, "domain", (0,))
    output = tmp_path / "out"
    original_args = make_convert_args(
        data_root=source,
        domains=["domain"],
        output_root=output,
        predecoded_root=cache,
    )
    converter.convert_dataset(original_args)
    original_payload, original_records = schema.load_manifest(output / "manifest.json")

    def fail_manifest(*_args, **_kwargs):
        raise OSError("injected overwrite manifest failure")

    monkeypatch.setattr(schema, "atomic_write_manifest", fail_manifest)
    replacement_args = make_convert_args(
        data_root=source,
        domains=["domain"],
        output_root=output,
        predecoded_root=cache,
        compression="lzf",
        overwrite=True,
    )
    with pytest.raises(OSError, match="overwrite manifest failure"):
        converter.convert_dataset(replacement_args)

    payload, records = schema.load_manifest(output / "manifest.json")
    assert payload == original_payload
    assert [record.shard_path for record in records] == [
        record.shard_path for record in original_records
    ]
    converter._validate_published_output(output / "manifest.json")


def test_shard_build_failure_names_target_shard_and_cleans_partial_output(
    tmp_path, monkeypatch
):
    source = make_tiny_lerobot_domain(
        tmp_path / "source", "domain", episode_indexes=(0,)
    )
    cache = make_tiny_predecoded_cache(tmp_path / "cache", source, "domain", (0,))
    output = tmp_path / "out"

    def fail_episode(*_args, **_kwargs):
        raise ValueError("injected episode write failure")

    monkeypatch.setattr(converter, "_write_episode_group", fail_episode)
    with pytest.raises(ValueError, match="episode write failure") as error:
        converter.convert_dataset(
            make_convert_args(
                data_root=source,
                domains=["domain"],
                output_root=output,
                predecoded_root=cache,
            )
        )
    assert str(output) in str(error.value)
    assert "shard_00000" in str(error.value)
    assert not list(output.glob("*.h5"))
    assert not list(output.glob("*.tmp*"))
    assert not (output / "manifest.json").exists()


def test_converter_rejects_more_than_32_episodes_per_shard(tmp_path):
    args = make_convert_args(
        data_root=tmp_path / "source",
        domains=["domain"],
        output_root=tmp_path / "out",
        predecoded_root=tmp_path / "cache",
        episodes_per_shard=33,
    )
    with pytest.raises(ValueError, match="episodes_per_shard"):
        converter.convert_dataset(args)


def make_reader_fixture(
    tmp_path: Path,
    *,
    episodes: int = 1,
    one_shard_per_episode: bool = False,
) -> tuple[Path, Path]:
    root = tmp_path / "reader"
    root.mkdir(parents=True, exist_ok=True)
    records = []
    shard_count = episodes if one_shard_per_episode else 1
    for shard_index in range(shard_count):
        shard_path = root / f"shard_{shard_index:05d}.h5"
        with h5py.File(shard_path, "w") as shard:
            episode_indexes = (
                [shard_index] if one_shard_per_episode else range(episodes)
            )
            for episode_index in episode_indexes:
                key = f"domain:{episode_index:06d}"
                group = shard.create_group(f"episodes/{key}")
                string_dtype = h5py.string_dtype(encoding="utf-8")
                group.create_dataset(
                    "caption", data=f"caption {episode_index}", dtype=string_dtype
                )
                group.create_dataset("domain", data="domain", dtype=string_dtype)
                group.create_dataset(
                    "episode_index", data=episode_index, dtype=np.int64
                )
                group.create_dataset("length", data=50, dtype=np.int64)
                group.create_dataset(
                    "rgb_main",
                    shape=(50, 256, 256, 3),
                    dtype=np.uint8,
                    chunks=(1, 256, 256, 3),
                    fillvalue=10 + episode_index,
                )
                group.create_dataset(
                    "rgb_wrist",
                    shape=(50, 256, 256, 3),
                    dtype=np.uint8,
                    chunks=(1, 256, 256, 3),
                    fillvalue=110 + episode_index,
                )
                group.create_dataset(
                    "action",
                    data=np.arange(50 * 7, dtype=np.float32).reshape(50, 7),
                )
                group.create_dataset(
                    "state",
                    data=np.arange(50 * 8, dtype=np.float32).reshape(50, 8),
                )
                records.append(
                    {
                        "key": key,
                        "shard": shard_path.name,
                        "group": f"episodes/{key}",
                        "caption": f"caption {episode_index}",
                        "domain": "domain",
                        "episode_index": episode_index,
                        "length": 50,
                    }
                )

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
        "compression": "none",
        "source_roots": [str(tmp_path / "source")],
        "datasets": DATASET_DECLARATIONS,
        "converter_fingerprint": "b" * 64,
        "episodes": records,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    stats_path = root / "stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "domain_eef": {"mean": [1.0] * 7, "std": [2.0] * 7},
                "domain_state_eef": {"mean": [3.0] * 8, "std": [4.0] * 8},
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, stats_path


def make_reader(tmp_path: Path, **kwargs) -> LiberoFastWAMHDF5Dataset:
    manifest_path, stats_path = make_reader_fixture(
        tmp_path, **kwargs.pop("fixture", {})
    )
    return LiberoFastWAMHDF5Dataset(
        manifest_path=manifest_path,
        stat_file=stats_path,
        **kwargs,
    )


def test_hdf5_dataset_constructor_opens_no_shards(tmp_path, monkeypatch):
    manifest_path, stats_path = make_reader_fixture(tmp_path)

    def fail_open(*args, **kwargs):
        raise AssertionError("constructor opened an HDF5 shard")

    monkeypatch.setattr(hdf5_dataset.h5py, "File", fail_open)
    dataset = LiberoFastWAMHDF5Dataset(
        manifest_path=manifest_path,
        stat_file=stats_path,
    )
    assert len(dataset) == 1
    assert dataset._handles == {}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_fps", 30),
        ("valid_cam", ["observation.images.wrist_image", "observation.images.image"]),
        ("chunk", 8),
        ("action_chunk", 32),
        ("n_previous", 3),
        ("action_type", "relative"),
        ("action_space", "joint"),
        ("ignore_seek", True),
    ],
)
def test_hdf5_dataset_rejects_non_fixed_arguments(tmp_path, field, value):
    manifest_path, stats_path = make_reader_fixture(tmp_path)
    with pytest.raises(ValueError, match=field):
        LiberoFastWAMHDF5Dataset(
            manifest_path=manifest_path,
            stat_file=stats_path,
            **{field: value},
        )


@pytest.mark.parametrize(
    ("key", "field", "value", "message"),
    [
        ("domain_eef", "mean", [0.0] * 6, "domain_eef.*mean.*7"),
        ("domain_eef", "std", [1.0] * 8, "domain_eef.*std.*7"),
        ("domain_state_eef", "mean", [0.0] * 7, "domain_state_eef.*mean.*8"),
        ("domain_state_eef", "std", [1.0] * 9, "domain_state_eef.*std.*8"),
    ],
)
def test_hdf5_dataset_rejects_wrong_statistics_width(
    tmp_path, key, field, value, message
):
    manifest_path, stats_path = make_reader_fixture(tmp_path)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats[key][field] = value
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        LiberoFastWAMHDF5Dataset(
            manifest_path=manifest_path,
            stat_file=stats_path,
        )


def test_hdf5_dataset_returns_fixed_normalized_sample_in_camera_order(tmp_path):
    dataset = make_reader(
        tmp_path,
        fix_epiidx=0,
        fix_sidx=12,
        fix_mem_idx=[1, 4, 8, 11],
    )
    sample = dataset[0]
    assert sample["video"].shape == (3, 2, 13, 256, 256)
    assert sample["video"].dtype == torch.float32
    assert sample["actions"].shape == (40, 7)
    assert sample["actions"].dtype == torch.float32
    assert sample["state"].shape == (1, 8)
    assert sample["state"].dtype == torch.float32
    assert sample["caption"] == "caption 0"
    torch.testing.assert_close(
        sample["video"][:, 0],
        torch.full((3, 13, 256, 256), 10 / 255.0 * 2.0 - 1.0),
    )
    torch.testing.assert_close(
        sample["video"][:, 1],
        torch.full((3, 13, 256, 256), 110 / 255.0 * 2.0 - 1.0),
    )
    expected_action_indexes = [1, 4, 8, 11, *range(12, 48)]
    raw_actions = torch.arange(50 * 7, dtype=torch.float32).reshape(50, 7)
    expected_actions = (raw_actions[expected_action_indexes] - 1.0) / (2.0 + 1e-6)
    torch.testing.assert_close(sample["actions"], expected_actions)
    raw_state = torch.arange(50 * 8, dtype=torch.float32).reshape(50, 8)
    torch.testing.assert_close(sample["state"], (raw_state[[11]] - 3.0) / (4.0 + 1e-6))
    assert dataset.get_frame_indexes(50) == (
        [1, 4, 8, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47],
        expected_action_indexes,
    )


class TrackingRows:
    def __init__(self):
        self.shape = (6, 2)
        self.calls = []
        self.array = np.arange(12, dtype=np.float32).reshape(6, 2)

    def __getitem__(self, indexes):
        self.calls.append(np.asarray(indexes).copy())
        return self.array[indexes]


def test_read_rows_preserves_order_repeats_and_clips_without_full_read():
    rows = TrackingRows()
    actual = read_rows_preserving_order(rows, [4, 1, 4, -2, 9], length=6)
    np.testing.assert_array_equal(actual, rows.array[[4, 1, 4, 0, 5]])
    assert len(rows.calls) == 1
    np.testing.assert_array_equal(rows.calls[0], [0, 1, 4, 5])


@pytest.mark.parametrize("indexes", [[], [[1, 2]], [1.5], [True]])
def test_read_rows_rejects_invalid_indexes(indexes):
    with pytest.raises((TypeError, ValueError), match="indexes"):
        read_rows_preserving_order(TrackingRows(), indexes, length=6)


@pytest.mark.parametrize("mode", ["uniform", "random"])
def test_hdf5_dataset_training_sampling_invariants(tmp_path, mode):
    dataset = make_reader(tmp_path, previous_pick_mode=mode)
    random.seed(7)
    np.random.seed(7)
    frame_indexes, action_indexes = dataset.get_frame_indexes(50)
    assert len(frame_indexes) == 13
    assert len(action_indexes) == 40
    assert all(0 <= index < 50 for index in frame_indexes + action_indexes)
    assert frame_indexes[:4] == action_indexes[:4]
    assert frame_indexes[4:] == action_indexes[4:][3::4]


@pytest.mark.parametrize(
    ("fix_sidx", "fix_mem_idx", "message"),
    [
        (1, None, "together"),
        (None, [1, 2, 3, 4], "together"),
        (1, [1, 2, 3], "length 4"),
    ],
)
def test_hdf5_dataset_rejects_invalid_fixed_indexes(
    tmp_path, fix_sidx, fix_mem_idx, message
):
    with pytest.raises(ValueError, match=message):
        make_reader(tmp_path, fix_sidx=fix_sidx, fix_mem_idx=fix_mem_idx)


def test_hdf5_dataset_supports_normal_negative_dataset_indexes(tmp_path):
    dataset = make_reader(tmp_path, fixture={"episodes": 2})
    assert dataset[-1]["caption"] == "caption 1"
    with pytest.raises(IndexError):
        _ = dataset[-3]


def test_hdf5_dataset_lru_reuses_and_evicts_closed_handle(tmp_path):
    dataset = make_reader(
        tmp_path,
        max_open_shards=2,
        fixture={"episodes": 3, "one_shard_per_episode": True},
    )
    dataset[0]
    first_path = next(iter(dataset._handles))
    first = dataset._handles[first_path]
    dataset[0]
    assert dataset._handles[first_path] is first
    dataset[1]
    dataset[2]
    assert len(dataset._handles) == 2
    assert first.id.valid == 0


def test_hdf5_dataset_close_is_idempotent_and_reopens_invalid_handle(tmp_path):
    dataset = make_reader(tmp_path)
    dataset[0]
    path, first = next(iter(dataset._handles.items()))
    first.close()
    dataset[0]
    assert dataset._handles[path] is not first
    assert dataset._handles[path].id.valid == 1
    dataset.close()
    dataset.close()
    assert dataset._handles == {}


def test_hdf5_dataset_pickle_drops_live_handles(tmp_path):
    dataset = make_reader(tmp_path)
    dataset[0]
    handle = next(iter(dataset._handles.values()))
    restored = pickle.loads(pickle.dumps(dataset))
    assert len(dataset._handles) == 1
    assert handle.id.valid == 1
    assert restored._handles == {}


def test_hdf5_dataset_pid_change_closes_inherited_handles(tmp_path, monkeypatch):
    dataset = make_reader(tmp_path)
    dataset[0]
    inherited = next(iter(dataset._handles.values()))
    parent_pid = os.getpid()
    monkeypatch.setattr(hdf5_dataset.os, "getpid", lambda: parent_pid + 1000)
    dataset[0]
    replacement = next(iter(dataset._handles.values()))
    assert inherited.id.valid == 0
    assert replacement is not inherited
    assert replacement.id.valid == 1


@pytest.mark.parametrize("failure", ["open", "missing_group", "corrupt"])
def test_hdf5_dataset_failures_include_full_read_context(tmp_path, failure):
    dataset = make_reader(
        tmp_path,
        fixture={"episodes": 2},
        fix_sidx=12,
        fix_mem_idx=[1, 4, 8, 11],
    )
    record = dataset.records[0]
    if failure == "open":
        record.shard_path.unlink()
    else:
        with h5py.File(record.shard_path, "a") as shard:
            if failure == "missing_group":
                del shard[record.group]
            else:
                del shard[f"{record.group}/state"]

    with pytest.raises(Exception) as error:
        dataset[0]
    message = str(error.value)
    assert "worker=main" in message
    assert str(record.shard_path) in message
    assert record.key in message
    assert "frame_indexes=" in message
    assert "action_indexes=" in message
    assert "caption 1" not in message
