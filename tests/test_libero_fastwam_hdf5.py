import copy
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import pickle
import random
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import av
import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

from ge_act.data import libero_fastwam_hdf5_schema as schema
from ge_act.data import libero_fastwam_hdf5_dataset as hdf5_dataset
from ge_act.data.libero_fastwam_hdf5_dataset import (
    LiberoFastWAMHDF5Dataset,
    read_rows_preserving_order,
)
from ge_act.data.lerobot_like_dataset import CustomLeRobotDataset
from ge_act.scripts import convert_libero_fastwam_hdf5 as converter
from ge_act.scripts import benchmark_libero_fastwam_hdf5 as benchmark


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

GE_ACT_ROOT = Path(__file__).resolve().parents[1] / "ge_act"
ORIGINAL_CONFIG = (
    GE_ACT_ROOT / "configs/ltx_model/libero/video_model_libero_fastwam_siglip2.yaml"
)
HDF5_CONFIG = (
    GE_ACT_ROOT
    / "configs/ltx_model/libero/video_model_libero_fastwam_siglip2_hdf5.yaml"
)
ORIGINAL_LOADER = GE_ACT_ROOT / "data/lerobot_like_dataset.py"
ORIGINAL_PREFLIGHT = GE_ACT_ROOT / "scripts/preflight_ltx_siglip2.py"
ORIGINAL_LAUNCHER = GE_ACT_ROOT / "scripts/train_ltx_siglip2.sh"
HDF5_PREFLIGHT = GE_ACT_ROOT / "scripts/preflight_libero_fastwam_hdf5.py"
HDF5_LAUNCHER = GE_ACT_ROOT / "scripts/train_ltx_siglip2_hdf5.sh"

HDF5_DATA_BLOCK = {
    "manifest_path": (
        "/data/user/jhe724/junjie/datasets/LIBERO-fastwam-hdf5/manifest.json"
    ),
    "stat_file": "configs/ltx_model/libero/libero_fastwam_mix.json",
    "source_fps": 20,
    "sample_n_frames": 500,
    "valid_cam": [
        "observation.images.image",
        "observation.images.wrist_image",
    ],
    "chunk": 9,
    "action_chunk": 36,
    "n_previous": 4,
    "previous_pick_mode": "random",
    "action_type": "absolute",
    "action_space": "eef",
}


def _load_python_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_hdf5_preflight():
    return _load_python_module(HDF5_PREFLIGHT, "hdf5_preflight_under_test")


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


def test_converter_loads_fastwam_chw_predecoded_cache(tmp_path):
    source = make_tiny_lerobot_domain(
        tmp_path / "source", "domain", episode_indexes=(0,)
    )
    cache = make_tiny_predecoded_cache(
        tmp_path / "cache", source, "domain", (0,)
    )
    for camera in CAMERAS:
        cache_path = (
            cache
            / "domain"
            / "videos/chunk-000"
            / camera
            / "episode_000000.npy"
        )
        frames = np.load(cache_path, allow_pickle=False)
        np.save(
            cache_path,
            frames.transpose(0, 3, 1, 2),
            allow_pickle=False,
        )

    episode = converter.discover_source_episodes(
        [source], ["domain"], predecoded_root=cache
    )[0]
    actual = converter._load_rgb_source(episode, "main")

    assert actual.shape == (3, 8, 10, 3)
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, _synthetic_rgb(0, 3, 0))


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
                group["rgb_main"][-1] = 49
                group["rgb_wrist"][-1] = 149
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
        [1, 4, 8, 11, 12, 16, 20, 24, 28, 32, 36, 40, 44],
        expected_action_indexes,
    )


def test_hdf5_fixed_sampling_matches_original_loader(tmp_path):
    dataset = make_reader(
        tmp_path,
        fix_sidx=12,
        fix_mem_idx=[1, 4, 8, 11],
    )
    original = object.__new__(CustomLeRobotDataset)
    original.fix_sidx = 12
    original.fix_mem_idx = [1, 4, 8, 11]
    original.action_chunk = 36
    original.video_temporal_stride = 4
    assert dataset.get_frame_indexes(50) == original.get_frame_indexes(50)


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


def test_read_rows_clips_uint64_above_int64_before_casting():
    rows = TrackingRows()
    indexes = np.asarray([np.uint64(2**63)], dtype=np.uint64)
    actual = read_rows_preserving_order(rows, indexes, length=6)
    np.testing.assert_array_equal(actual, rows.array[[5]])
    np.testing.assert_array_equal(rows.calls[0], [5])


def test_read_by_indexes_clips_uint64_above_int64_before_casting(tmp_path):
    dataset = make_reader(tmp_path)
    huge = np.uint64(2**63)
    sample = dataset.read_by_indexes(
        0,
        np.asarray([huge], dtype=np.uint64),
        np.asarray([0, 1, 2, huge], dtype=np.uint64),
    )
    torch.testing.assert_close(
        sample["video"][:, 0, 0],
        torch.full((3, 256, 256), 49 / 255.0 * 2.0 - 1.0),
    )
    raw_actions = torch.arange(50 * 7, dtype=torch.float32).reshape(50, 7)
    torch.testing.assert_close(
        sample["actions"][-1], (raw_actions[-1] - 1.0) / (2.0 + 1e-6)
    )
    raw_state = torch.arange(50 * 8, dtype=torch.float32).reshape(50, 8)
    torch.testing.assert_close(sample["state"], (raw_state[[-1]] - 3.0) / (4.0 + 1e-6))


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_hdf5_preflight_fixture(tmp_path: Path) -> dict:
    dataset_root = tmp_path / "dataset"
    manifest_path = dataset_root / "manifest.json"
    payload = make_manifest(dataset_root)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    stat_file = tmp_path / "stats.json"
    stat_file.write_text("{}", encoding="utf-8")
    ltx_root = tmp_path / "ltx"
    for component in ("tokenizer", "text_encoder", "vae"):
        (ltx_root / component).mkdir(parents=True, exist_ok=True)
    diffusion_checkpoint = tmp_path / "base.safetensors"
    diffusion_checkpoint.touch()
    siglip_root = tmp_path / "siglip"
    siglip_root.mkdir()
    (siglip_root / "model.safetensors").touch()
    output_parent = tmp_path / "outputs"
    output_parent.mkdir()

    config = yaml.safe_load(HDF5_CONFIG.read_text(encoding="utf-8"))
    for split in ("train", "val"):
        config["data"][split]["manifest_path"] = str(manifest_path)
        config["data"][split]["stat_file"] = str(stat_file)
    config["pretrained_model_name_or_path"] = str(ltx_root)
    config["diffusion_model"]["model_path"] = str(diffusion_checkpoint)
    config["semantic_plan"]["model_name_or_path"] = str(siglip_root)
    config["output_dir"] = str(output_parent / "run")
    return config


def test_hdf5_training_entrypoints_exist():
    assert HDF5_CONFIG.is_file()
    assert HDF5_PREFLIGHT.is_file()
    assert HDF5_LAUNCHER.is_file()


def test_hdf5_yaml_only_changes_allowed_active_config_sections():
    original = yaml.safe_load(ORIGINAL_CONFIG.read_text(encoding="utf-8"))
    actual = yaml.safe_load(HDF5_CONFIG.read_text(encoding="utf-8"))
    expected = copy.deepcopy(original)
    expected.update(
        {
            "tracker_name": "ltx_siglip2_hdf5_trainer",
            "output_dir": (
                "/data/user/jhe724/junjie/outputs/libero_fastwam_ltx_siglip2_hdf5"
            ),
            "train_data_class_path": "data/libero_fastwam_hdf5_dataset.py",
            "train_data_class": "LiberoFastWAMHDF5Dataset",
            "val_data_class_path": "data/libero_fastwam_hdf5_dataset.py",
            "val_data_class": "LiberoFastWAMHDF5Dataset",
            "data": {
                "train": dict(HDF5_DATA_BLOCK, train_dataset=True),
                "val": dict(HDF5_DATA_BLOCK, train_dataset=False),
            },
        }
    )
    assert actual == expected


def test_original_hdf5_alternative_protected_files_are_unchanged():
    assert _sha256(ORIGINAL_CONFIG) == (
        "14fd689abc9813cd962886776c5a89c06c036a3920247cb078cca9b84003daad"
    )
    assert _sha256(ORIGINAL_LOADER) == (
        "35dbcaa7746344f789d1be26a5b67b323296c4d48702aa260abde51c409261a4"
    )
    assert _sha256(ORIGINAL_LAUNCHER) == (
        "fdfc2ea518af07badbf036f83dcfd9f803b3d712ba76288b59ca5e1253fb3bc9"
    )
    assert _sha256(ORIGINAL_PREFLIGHT) == (
        "240fd97a2550450c15b2f19b91d264900af7e2ac06666b290a7c61764fda2d9f"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_fps", 30, "source_fps"),
        ("sample_n_frames", 499, "sample_n_frames"),
        ("chunk", 8, "chunk"),
        ("action_chunk", 32, "action_chunk"),
        ("n_previous", 3, "n_previous"),
        ("previous_pick_mode", "uniform", "previous_pick_mode"),
        ("action_type", "relative", "action_type"),
        ("action_space", "joint", "action_space"),
        ("train_dataset", False, "train_dataset"),
    ],
)
def test_hdf5_preflight_rejects_non_fixed_train_contract(
    tmp_path, field, value, message
):
    preflight = _load_hdf5_preflight()
    config = _write_hdf5_preflight_fixture(tmp_path)
    config["data"]["train"][field] = value
    errors = preflight.collect_hdf5_preflight_errors(
        config, world_size=8, check_paths=False
    )
    assert any(message in error and "train" in error for error in errors)


def test_hdf5_preflight_rejects_wrong_camera_order_in_both_splits(tmp_path):
    preflight = _load_hdf5_preflight()
    config = _write_hdf5_preflight_fixture(tmp_path)
    wrong_order = [
        "observation.images.wrist_image",
        "observation.images.image",
    ]
    config["data"]["train"]["valid_cam"] = wrong_order
    config["data"]["val"]["valid_cam"] = wrong_order
    errors = preflight.collect_hdf5_preflight_errors(
        config, world_size=8, check_paths=False
    )
    assert any("train valid_cam" in error for error in errors)
    assert any("val valid_cam" in error for error in errors)


@pytest.mark.parametrize(
    ("split", "key"),
    [
        ("train", "data_roots"),
        ("train", "predecoded_video_root"),
        ("train", "require_predecoded"),
        ("train", "sample_size"),
        ("train", "state_key"),
        ("train", "ignore_seek"),
        ("val", "domains"),
        ("val", "preprocess"),
        ("val", "random_crop"),
        ("val", "action_key"),
    ],
)
def test_hdf5_preflight_rejects_old_loader_keys(tmp_path, split, key):
    preflight = _load_hdf5_preflight()
    config = _write_hdf5_preflight_fixture(tmp_path)
    config["data"][split][key] = "forbidden"
    errors = preflight.collect_hdf5_preflight_errors(
        config, world_size=8, check_paths=False
    )
    assert any(split in error and key in error for error in errors)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda config: config.update(train_data_class="WrongDataset"),
            "train_data_class",
        ),
        (
            lambda config: config.update(
                val_data_class_path="data/lerobot_like_dataset.py"
            ),
            "val_data_class_path",
        ),
        (
            lambda config: config["data"]["val"].update(train_dataset=True),
            "val train_dataset",
        ),
        (
            lambda config: config["semantic_plan"].update(enabled=False),
            "semantic_plan.enabled",
        ),
        (
            lambda config: config["semantic_plan"].update(
                keyframe_indices=[0, 2, 5, 8]
            ),
            "keyframes",
        ),
        (
            lambda config: config["semantic_plan"].update(
                keyframe_indices=[0, 3, 5, 8.0]
            ),
            "keyframes",
        ),
        (
            lambda config: config["semantic_plan"].update(keyframe_indices="bad"),
            "keyframes",
        ),
        (
            lambda config: config["semantic_plan"].update(
                keyframe_indices=[0, 3, "five", 8]
            ),
            "keyframes",
        ),
        (
            lambda config: config["semantic_plan"].update(tokens_per_frame=81),
            "256 tokens",
        ),
        (
            lambda config: config["diffusion_model"]["config"].update(
                semantic_plan_in_dim=768
            ),
            "feature width",
        ),
        (
            lambda config: config["diffusion_model"]["config"].update(
                semantic_plan_cross_attention_blocks=list(range(27))
            ),
            "all 28",
        ),
        (lambda config: config.update(batch_size=7), "global batch"),
        (lambda config: config.update(train_steps=20_000), "train_steps"),
        (lambda config: config.update(save_steps=[30_000]), "save_steps"),
        (
            lambda config: config.update(gradient_checkpointing=False),
            "gradient checkpointing",
        ),
        (
            lambda config: config.update(train_mode="action_only"),
            "train_mode",
        ),
        (
            lambda config: config["deepspeed"]["zero_optimization"].update(stage=3),
            "DeepSpeed ZeRO stage",
        ),
    ],
)
def test_hdf5_preflight_rejects_training_semantic_and_model_mismatch(
    tmp_path, mutation, message
):
    preflight = _load_hdf5_preflight()
    config = _write_hdf5_preflight_fixture(tmp_path)
    mutation(config)
    errors = preflight.collect_hdf5_preflight_errors(
        config, world_size=8, check_paths=False
    )
    assert any(message in error for error in errors)


def test_hdf5_preflight_rejects_train_val_manifest_mismatch(tmp_path):
    preflight = _load_hdf5_preflight()
    config = _write_hdf5_preflight_fixture(tmp_path)
    config["data"]["val"]["manifest_path"] = str(tmp_path / "other.json")
    errors = preflight.collect_hdf5_preflight_errors(
        config, world_size=8, check_paths=False
    )
    assert any("same manifest" in error for error in errors)


def test_hdf5_preflight_check_paths_false_does_no_discovery_or_io(
    tmp_path, monkeypatch
):
    preflight = _load_hdf5_preflight()
    config = _write_hdf5_preflight_fixture(tmp_path)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("check_paths=False performed filesystem discovery")

    monkeypatch.setattr(preflight.importlib.util, "find_spec", unexpected)
    monkeypatch.setattr(preflight, "load_manifest", unexpected)
    monkeypatch.setattr(preflight.shutil, "disk_usage", unexpected)
    monkeypatch.setattr(preflight.os, "access", unexpected)
    assert (
        preflight.collect_hdf5_preflight_errors(config, world_size=8, check_paths=False)
        == []
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("camera_names", ["wrist", "main"], "camera_names"),
        ("compression", "gzip", "compression"),
        (
            "datasets",
            dict(DATASET_DECLARATIONS, action={"width": 8, "dtype": "float32"}),
            "datasets",
        ),
    ],
)
def test_hdf5_preflight_reports_manifest_contract_errors(
    tmp_path, monkeypatch, field, value, message
):
    config = _write_hdf5_preflight_fixture(tmp_path)
    manifest_path = Path(config["data"]["train"]["manifest_path"])
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload[field] = value
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    preflight = _load_hdf5_preflight()
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda _name: object())
    errors = preflight.collect_hdf5_preflight_errors(
        config,
        world_size=8,
        check_paths=True,
        minimum_free_gb=0.0,
    )
    assert any("manifest" in error and message in error for error in errors)


def test_hdf5_preflight_reports_missing_and_unsafe_manifest_shards(
    tmp_path, monkeypatch
):
    config = _write_hdf5_preflight_fixture(tmp_path)
    manifest_path = Path(config["data"]["train"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.with_name(manifest["episodes"][0]["shard"]).unlink()
    preflight = _load_hdf5_preflight()
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda _name: object())
    errors = preflight.collect_hdf5_preflight_errors(
        config,
        world_size=8,
        check_paths=True,
        minimum_free_gb=0.0,
    )
    assert any("missing HDF5 shard" in error for error in errors)

    outside = tmp_path / "outside.h5"
    outside.touch()
    manifest["episodes"][0]["shard"] = "../outside.h5"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    preflight = _load_hdf5_preflight()
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda _name: object())
    errors = preflight.collect_hdf5_preflight_errors(
        config,
        world_size=8,
        check_paths=True,
        minimum_free_gb=0.0,
    )
    assert any("outside manifest root" in error for error in errors)


def test_hdf5_preflight_reports_invalid_manifest_json_without_raising(
    tmp_path, monkeypatch
):
    config = _write_hdf5_preflight_fixture(tmp_path)
    Path(config["data"]["train"]["manifest_path"]).write_text(
        "{broken", encoding="utf-8"
    )
    preflight = _load_hdf5_preflight()
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda _name: object())
    errors = preflight.collect_hdf5_preflight_errors(
        config,
        world_size=8,
        check_paths=True,
        minimum_free_gb=0.0,
    )
    assert any("manifest" in error and "JSON" in error for error in errors)


def test_hdf5_preflight_loads_shared_manifest_once(tmp_path, monkeypatch):
    preflight = _load_hdf5_preflight()
    config = _write_hdf5_preflight_fixture(tmp_path)
    real_load = preflight.load_manifest
    calls = []

    def tracking_load(path):
        calls.append(Path(path))
        return real_load(path)

    monkeypatch.setattr(preflight, "load_manifest", tracking_load)
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda _name: object())
    errors = preflight.collect_hdf5_preflight_errors(
        config,
        world_size=8,
        check_paths=True,
        minimum_free_gb=0.0,
    )
    assert errors == []
    assert calls == [Path(config["data"]["train"]["manifest_path"])]


@pytest.mark.parametrize(
    ("remove", "message"),
    [
        ("stat", "normalization statistics"),
        ("ltx_component", "LTX component"),
        ("diffusion", "base diffusion checkpoint"),
        ("siglip", "SigLIP2 checkpoint"),
    ],
)
def test_hdf5_preflight_reports_missing_required_paths(
    tmp_path, monkeypatch, remove, message
):
    preflight = _load_hdf5_preflight()
    config = _write_hdf5_preflight_fixture(tmp_path)
    if remove == "stat":
        Path(config["data"]["train"]["stat_file"]).unlink()
    elif remove == "ltx_component":
        Path(config["pretrained_model_name_or_path"], "vae").rmdir()
    elif remove == "diffusion":
        Path(config["diffusion_model"]["model_path"]).unlink()
    else:
        Path(
            config["semantic_plan"]["model_name_or_path"], "model.safetensors"
        ).unlink()
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda _name: object())
    errors = preflight.collect_hdf5_preflight_errors(
        config,
        world_size=8,
        check_paths=True,
        minimum_free_gb=0.0,
    )
    assert any(message in error for error in errors)


def test_hdf5_preflight_requires_h5py_and_reports_low_output_disk(
    tmp_path, monkeypatch
):
    preflight = _load_hdf5_preflight()
    config = _write_hdf5_preflight_fixture(tmp_path)
    monkeypatch.setattr(
        preflight.importlib.util,
        "find_spec",
        lambda name: None if name == "h5py" else object(),
    )
    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda _path: Namespace(total=1024**3, used=1024**3, free=0),
    )
    errors = preflight.collect_hdf5_preflight_errors(
        config,
        world_size=8,
        check_paths=True,
        minimum_free_gb=100.0,
    )
    assert any("missing Python module: h5py" in error for error in errors)
    assert any("100.0 GiB" in error for error in errors)


def _write_command_stub(path: Path, command: str) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"printf '{command}' >> \"$INVOCATION_LOG\"\n"
        'printf \' %q\' "$@" >> "$INVOCATION_LOG"\n'
        "printf '\\n' >> \"$INVOCATION_LOG\"\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _run_hdf5_launcher(
    tmp_path: Path, *, args=(), env_overrides=None, cwd: Path | None = None
) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_command_stub(bin_dir / "python", "python")
    _write_command_stub(bin_dir / "torchrun", "torchrun")
    invocation_log = tmp_path / "invocations.log"
    environment = os.environ.copy()
    for name in (
        "CONFIG",
        "NUM_PROCESSES",
        "NPROC_PER_NODE",
        "NNODES",
        "NODE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "PATH": f"{bin_dir}:{environment['PATH']}",
            "INVOCATION_LOG": str(invocation_log),
        }
    )
    if env_overrides:
        environment.update(env_overrides)
    subprocess.run(
        ["bash", str(HDF5_LAUNCHER), *map(str, args)],
        check=True,
        cwd=tmp_path if cwd is None else cwd,
        env=environment,
        text=True,
        capture_output=True,
    )
    return invocation_log.read_text(encoding="utf-8").splitlines()


def test_hdf5_launcher_executes_only_new_preflight_then_torchrun(tmp_path):
    config = tmp_path / "explicit.yaml"
    lines = _run_hdf5_launcher(
        tmp_path,
        args=(tmp_path / "ignored-positional.yaml",),
        env_overrides={"CONFIG": str(config), "NUM_PROCESSES": "4"},
    )
    assert len(lines) == 2
    assert lines[0].startswith("python scripts/preflight_libero_fastwam_hdf5.py")
    assert f"--config {config!s}" in lines[0]
    assert "--world-size 4" in lines[0]
    assert lines[1].startswith("torchrun --standalone --nnodes=1 --nproc_per_node=4")
    assert f"main.py --config_file {config!s}" in lines[1]
    assert all("predecode" not in line for line in lines)
    assert all("preflight_ltx_siglip2.py" not in line for line in lines)


def test_hdf5_launcher_supports_nproc_fallback_and_multinode_overrides(tmp_path):
    positional = tmp_path / "positional.yaml"
    lines = _run_hdf5_launcher(
        tmp_path,
        args=(positional,),
        env_overrides={
            "NPROC_PER_NODE": "4",
            "NNODES": "2",
            "NODE_RANK": "1",
            "MASTER_ADDR": "10.0.0.8",
            "MASTER_PORT": "29600",
        },
    )
    assert f"--config {positional!s}" in lines[0]
    assert "--world-size 8" in lines[0]
    assert "--standalone" not in lines[1]
    assert "--nnodes=2" in lines[1]
    assert "--nproc_per_node=4" in lines[1]
    assert "--node_rank=1" in lines[1]
    assert "--master_addr=10.0.0.8" in lines[1]
    assert "--master_port=29600" in lines[1]


def test_hdf5_launcher_defaults_to_hdf5_config_and_eight_processes(tmp_path):
    lines = _run_hdf5_launcher(tmp_path)
    assert f"--config {HDF5_CONFIG!s}" in lines[0]
    assert "--world-size 8" in lines[0]
    assert lines[1].startswith("torchrun --standalone --nnodes=1 --nproc_per_node=8")
    assert f"main.py --config_file {HDF5_CONFIG!s}" in lines[1]


@pytest.mark.parametrize("config_source", ["positional", "environment"])
def test_hdf5_launcher_resolves_relative_config_from_callers_working_directory(
    tmp_path, config_source
):
    caller = GE_ACT_ROOT.parent
    if config_source == "positional":
        relative_config = Path("ge_act/configs/ltx_model/libero") / HDF5_CONFIG.name
        expected = HDF5_CONFIG
        args = (relative_config,)
        env_overrides = None
    else:
        fixture_config = tmp_path / "relative-fixture.yaml"
        fixture_config.touch()
        relative_config = Path(os.path.relpath(fixture_config, caller))
        expected = fixture_config.resolve()
        args = ()
        env_overrides = {"CONFIG": str(relative_config)}

    lines = _run_hdf5_launcher(
        tmp_path,
        args=args,
        env_overrides=env_overrides,
        cwd=caller,
    )

    assert f"--config {expected}" in lines[0]
    assert f"main.py --config_file {expected}" in lines[1]


def test_hdf5_launcher_multinode_requires_explicit_master_addr(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as error:
        _run_hdf5_launcher(
            tmp_path,
            env_overrides={"NNODES": "2", "NUM_PROCESSES": "4"},
        )
    assert "MASTER_ADDR" in error.value.stderr
    invocation_log = tmp_path / "invocations.log"
    assert not invocation_log.exists() or not invocation_log.read_text(encoding="utf-8")


def test_hdf5_launcher_dynamic_import_constructs_real_dataset(tmp_path):
    manifest_path, stat_file = make_reader_fixture(tmp_path / "fixture")
    config = yaml.safe_load(HDF5_CONFIG.read_text(encoding="utf-8"))
    for split in ("train", "val"):
        config["data"][split]["manifest_path"] = str(manifest_path)
        config["data"][split]["stat_file"] = str(stat_file)
    config_path = tmp_path / "integration.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_command_stub(bin_dir / "python", "python")
    torchrun = bin_dir / "torchrun"
    torchrun.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "config=''\n"
        "while (($#)); do\n"
        '  if [[ "$1" == \'--config_file\' ]]; then config="$2"; shift 2; else shift; fi\n'
        "done\n"
        '[[ -n "$config" ]]\n'
        '"$REAL_PYTHON" -c \'import json, os, sys; '
        "from pathlib import Path; import yaml; "
        "from utils import import_custom_class; "
        "config = yaml.safe_load(Path(sys.argv[1]).read_text()); "
        'dataset_class = import_custom_class(config["train_data_class"], '
        'config["train_data_class_path"]); '
        'dataset = dataset_class(**config["data"]["train"]); '
        'result = {"class": dataset.__class__.__name__, "length": len(dataset), '
        '"cwd": os.getcwd(), "config": str(Path(sys.argv[1]).resolve()), '
        '"pythonpath": os.environ.get("PYTHONPATH", "")}; '
        'Path(os.environ["PROBE_RESULT"]).write_text(json.dumps(result)); '
        'dataset.close()\' "$config"\n',
        encoding="utf-8",
    )
    torchrun.chmod(0o755)
    probe_result = tmp_path / "probe.json"
    environment = os.environ.copy()
    environment.update(
        {
            "CONFIG": str(config_path),
            "PATH": f"{bin_dir}:{environment['PATH']}",
            "INVOCATION_LOG": str(tmp_path / "invocations.log"),
            "PROBE_RESULT": str(probe_result),
            "PYTHONPATH": "/existing/sentinel",
            "REAL_PYTHON": sys.executable,
        }
    )

    subprocess.run(
        ["bash", str(HDF5_LAUNCHER)],
        check=True,
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
    )

    result = json.loads(probe_result.read_text(encoding="utf-8"))
    assert result["class"] == "LiberoFastWAMHDF5Dataset"
    assert result["length"] == 1
    assert Path(result["cwd"]) == GE_ACT_ROOT
    assert Path(result["config"]) == config_path.resolve()
    python_paths = result["pythonpath"].split(os.pathsep)
    assert str(GE_ACT_ROOT.parent) in [
        str(Path(path).resolve()) for path in python_paths
    ]
    assert "/existing/sentinel" in python_paths


def test_hdf5_preflight_missing_h5py_is_clean_cli_error(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    blocker.joinpath("sitecustomize.py").write_text(
        "import builtins\n"
        "import importlib.util\n"
        "_real_import = builtins.__import__\n"
        "_real_find_spec = importlib.util.find_spec\n"
        "def _blocked_import(name, *args, **kwargs):\n"
        "    if name == 'h5py' or name.startswith('h5py.'):\n"
        "        raise ModuleNotFoundError(\"No module named 'h5py'\")\n"
        "    return _real_import(name, *args, **kwargs)\n"
        "def _blocked_find_spec(name, *args, **kwargs):\n"
        "    if name == 'h5py':\n"
        "        return None\n"
        "    return _real_find_spec(name, *args, **kwargs)\n"
        "builtins.__import__ = _blocked_import\n"
        "importlib.util.find_spec = _blocked_find_spec\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(blocker), str(GE_ACT_ROOT.parent)))

    result = subprocess.run(
        [
            sys.executable,
            str(HDF5_PREFLIGHT),
            "--config",
            str(HDF5_CONFIG),
            "--minimum-free-gb",
            "0",
        ],
        cwd=GE_ACT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "missing Python module: h5py" in combined
    assert "Traceback" not in combined


def _benchmark_sample(*, main=0.0, wrist=0.5):
    video = torch.empty((3, 2, 2, 2, 2), dtype=torch.float32)
    video[:, 0].fill_(main)
    video[:, 1].fill_(wrist)
    return {
        "video": video,
        "actions": torch.arange(14, dtype=torch.float32).reshape(2, 7),
        "caption": "benchmark caption",
        "state": torch.arange(8, dtype=torch.float32).reshape(1, 8),
    }


def test_benchmark_parity_accepts_uint8_rounding_bound_per_camera():
    old = _benchmark_sample()
    new = copy.deepcopy(old)
    new["video"][:, 0].add_(1.0 / 255.0)
    new["video"][:, 1].sub_(1.0 / 255.0)

    report = benchmark.compare_samples(old, new)

    assert report["exact_fields"] == [
        "actions",
        "state",
        "caption",
        "shape",
        "dtype",
    ]
    assert report["normalized_rgb_error"]["main"] <= 1 / 255 + 1e-6
    assert report["normalized_rgb_error"]["wrist"] <= 1 / 255 + 1e-6
    assert report["max_normalized_rgb_error"] <= 1 / 255 + 1e-6


def test_benchmark_parity_rejects_apparent_camera_swap():
    old = _benchmark_sample(main=-0.75, wrist=0.75)
    new = copy.deepcopy(old)
    new["video"] = old["video"].flip(1)
    with pytest.raises(AssertionError, match="camera.*order|camera.*swap"):
        benchmark.compare_samples(old, new)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda sample: sample.pop("state"), "keys"),
        (lambda sample: sample.update(extra=1), "keys"),
        (
            lambda sample: sample.update(video=sample["video"][:, :, :1]),
            "shape",
        ),
        (
            lambda sample: sample.update(video=sample["video"].to(torch.float64)),
            "dtype",
        ),
        (lambda sample: sample.update(caption="wrong"), "caption"),
        (lambda sample: sample["actions"].add_(1), "actions"),
        (lambda sample: sample["state"].add_(1), "state"),
        (
            lambda sample: sample["video"][:, 0].add_(1 / 255 + 2e-6),
            "main.*RGB|RGB.*main",
        ),
    ],
)
def test_benchmark_parity_rejects_contract_mismatch(mutation, message):
    old = _benchmark_sample()
    new = copy.deepcopy(old)
    mutation(new)
    with pytest.raises(AssertionError, match=message):
        benchmark.compare_samples(old, new)


def test_benchmark_compression_winner_prefers_lzf_within_five_percent():
    assert benchmark.choose_compression({"none": 100.0, "lzf": 97.0}) == "lzf"
    assert benchmark.choose_compression({"none": 100.0, "lzf": 95.0}) == "lzf"
    assert benchmark.choose_compression({"none": 100.0, "lzf": 94.9}) == "none"
    assert benchmark.choose_compression({"none": 100.0, "lzf": 140.0}) == "lzf"


@pytest.mark.parametrize(
    "results",
    [
        {},
        {"none": 1.0},
        {"none": 1.0, "lzf": 0.0},
        {"none": -1.0, "lzf": 1.0},
        {"none": float("nan"), "lzf": 1.0},
        {"none": 1.0, "lzf": float("inf")},
        {"none": True, "lzf": 1.0},
    ],
)
def test_benchmark_compression_winner_rejects_invalid_metrics(results):
    with pytest.raises(ValueError, match="none|lzf|finite|positive"):
        benchmark.choose_compression(results)


def _old_episode_record(domain, episode_index, caption="caption", length=50):
    stem = f"episode_{episode_index:06d}"
    return [
        f"/source/{domain}/videos/chunk-000/{{}}/{stem}.mp4",
        None,
        f"/source/{domain}/data/chunk-000/{stem}.parquet",
        domain,
        "",
        None,
        caption,
        length,
    ]


def _hdf5_episode_record(domain, episode_index, caption="caption", length=50):
    key = f"{domain}:{episode_index:06d}"
    return schema.EpisodeRecord(
        key=key,
        shard_path=Path("/manifest/shard.h5"),
        group=f"episodes/{key}",
        caption=caption,
        domain=domain,
        episode_index=episode_index,
        length=length,
    )


def test_benchmark_maps_pairs_in_manifest_order_and_limits_episodes():
    old = Namespace(
        dataset=[
            _old_episode_record("domain", 1, "one"),
            _old_episode_record("domain", 0, "zero"),
        ]
    )
    new = Namespace(
        records=[
            _hdf5_episode_record("domain", 0, "zero"),
            _hdf5_episode_record("domain", 1, "one"),
        ]
    )

    pairs = benchmark.map_episode_pairs(old, new, episode_limit=1)

    assert [(pair.domain, pair.episode_index) for pair in pairs] == [("domain", 0)]
    assert pairs[0].old_index == 1
    assert pairs[0].hdf5_index == 0


@pytest.mark.parametrize("problem", ["duplicate", "missing", "caption", "length"])
def test_benchmark_mapping_rejects_bad_identity_or_metadata(problem):
    old_records = [_old_episode_record("domain", 0)]
    new_records = [_hdf5_episode_record("domain", 0)]
    if problem == "duplicate":
        old_records.append(_old_episode_record("domain", 0))
    elif problem == "missing":
        new_records = [_hdf5_episode_record("domain", 1)]
    elif problem == "caption":
        new_records[0] = _hdf5_episode_record("domain", 0, caption="wrong")
    else:
        new_records[0] = _hdf5_episode_record("domain", 0, length=49)

    with pytest.raises(ValueError, match=problem):
        benchmark.map_episode_pairs(
            Namespace(dataset=old_records),
            Namespace(records=new_records),
            episode_limit=8,
        )


class _BenchmarkFakeOldDataset:
    def __init__(self):
        self.dataset = [_old_episode_record("domain", 0)]
        self.fix_sidx = None
        self.fix_mem_idx = None
        self.action_chunk = 36
        self.video_temporal_stride = 4
        self.item_calls = 0
        self.batch_calls = 0

    def get_frame_indexes(self, length):
        return CustomLeRobotDataset.get_frame_indexes(self, length)

    def get_batch(self, index):
        self.batch_calls += 1
        frame_indexes, action_indexes = self.get_frame_indexes(50)
        sample = _benchmark_sample()
        sample["actions"] = torch.tensor(action_indexes, dtype=torch.float32)[:, None]
        sample["state"] = torch.tensor(frame_indexes[:1], dtype=torch.float32)[:, None]
        return tuple(sample[key] for key in ("video", "actions", "caption", "state"))

    def __getitem__(self, index):
        self.item_calls += 1
        raise AssertionError("old __getitem__ fallback must not be used")


class _BenchmarkFakeHDF5Dataset:
    def __init__(self):
        self.records = [_hdf5_episode_record("domain", 0)]
        self.fix_sidx = None
        self.fix_mem_idx = None
        self.action_chunk = 36
        self.video_temporal_stride = 4
        self.read_calls = []
        self.closed = False

    def get_frame_indexes(self, length):
        return LiberoFastWAMHDF5Dataset.get_frame_indexes(self, length)

    def read_by_indexes(self, index, frame_indexes, action_indexes):
        self.read_calls.append((index, list(frame_indexes), list(action_indexes)))
        sample = _benchmark_sample()
        sample["actions"] = torch.tensor(action_indexes, dtype=torch.float32)[:, None]
        sample["state"] = torch.tensor(frame_indexes[:1], dtype=torch.float32)[:, None]
        return sample

    def close(self):
        self.closed = True


def test_benchmark_fixed_wrappers_share_indexes_and_bypass_old_fallback():
    old = _BenchmarkFakeOldDataset()
    new = _BenchmarkFakeHDF5Dataset()
    pairs = benchmark.map_episode_pairs(old, new, episode_limit=1)
    plans = benchmark.build_sample_plans(old, new, pairs)
    old_view = benchmark.DeterministicOldView(old, plans, sample_count=2)
    new_view = benchmark.DeterministicHDF5View(new, plans, sample_count=2)

    old_sample = old_view[1]
    new_sample = new_view[1]

    assert old.item_calls == 0
    assert old.batch_calls == 1
    assert new.read_calls[0][1:] == (
        plans[0].frame_indexes,
        plans[0].action_indexes,
    )
    benchmark.compare_samples(old_sample, new_sample)


def test_benchmark_percentile_and_rss_helpers(tmp_path, monkeypatch):
    assert benchmark.percentile([4.0, 1.0, 3.0, 2.0], 50) == 2.5
    assert benchmark.percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)

    status = tmp_path / "status"
    missing = tmp_path / "missing-status"
    status.write_text("Name:\ttest\nVmRSS:\t123 kB\n", encoding="utf-8")
    monkeypatch.setattr(benchmark, "_psutil", None)
    monkeypatch.setattr(
        benchmark,
        "_proc_status_path",
        lambda pid: status if pid == 123 else missing,
    )
    assert benchmark.read_process_rss_bytes(123) == 123 * 1024
    assert benchmark.aggregate_worker_rss_bytes([123, 999]) == 123 * 1024
    status.unlink()
    assert benchmark.read_process_rss_bytes(123) == 0


def test_benchmark_atomic_json_success_and_failure_cleanup(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    benchmark.atomic_write_json(output, {"value": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}
    assert list(tmp_path.iterdir()) == [output]

    with pytest.raises(TypeError):
        benchmark.atomic_write_json(output, {"bad": {1}})
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}
    assert list(tmp_path.iterdir()) == [output]

    monkeypatch.setattr(
        benchmark.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )
    with pytest.raises(OSError, match="replace failed"):
        benchmark.atomic_write_json(output, {"value": 2})
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}
    assert list(tmp_path.iterdir()) == [output]


def _make_real_benchmark_datasets(tmp_path, *, compression="none"):
    source = make_tiny_lerobot_domain(
        tmp_path / "source",
        "domain",
        episode_indexes=(0,),
        length=50,
    )
    cache = make_tiny_predecoded_cache(
        tmp_path / "cache",
        source,
        "domain",
        (0,),
        length=50,
    )
    output = tmp_path / f"hdf5-{compression}"
    converter.convert_dataset(
        make_convert_args(
            data_root=source,
            domains=["domain"],
            output_root=output,
            predecoded_root=cache,
            compression=compression,
        )
    )
    stats = tmp_path / "stats.json"
    stats.write_text(
        json.dumps(
            {
                "domain_eef": {"mean": [1.0] * 7, "std": [2.0] * 7},
                "domain_state_eef": {"mean": [3.0] * 8, "std": [4.0] * 8},
            }
        ),
        encoding="utf-8",
    )
    old = CustomLeRobotDataset(
        data_roots=[str(source)],
        domains=["domain"],
        sample_size=[256, 256],
        sample_n_frames=500,
        source_fps=20,
        preprocess="resize",
        valid_cam=list(CAMERAS),
        chunk=9,
        action_chunk=36,
        n_previous=4,
        previous_pick_mode="random",
        random_crop=False,
        predecoded_video_root=str(cache),
        require_predecoded=True,
        action_type="absolute",
        action_space="eef",
        ignore_seek=False,
        train_dataset=True,
        action_key="action",
        state_key="observation.state",
        stat_file=str(stats),
    )
    new = LiberoFastWAMHDF5Dataset(
        manifest_path=output / "manifest.json",
        stat_file=stats,
    )
    return old, new, output / "manifest.json", stats, source, cache


def test_benchmark_real_dataloaders_exact_accounting_and_cleanup(tmp_path):
    old, new, *_ = _make_real_benchmark_datasets(tmp_path)
    pairs = benchmark.map_episode_pairs(old, new, episode_limit=1)
    plans = benchmark.build_sample_plans(old, new, pairs)
    benchmark.compare_samples(
        benchmark.DeterministicOldView(old, plans, sample_count=1)[0],
        benchmark.DeterministicHDF5View(new, plans, sample_count=1)[0],
    )
    results = benchmark.run_throughput_benchmarks(
        old,
        new,
        plans,
        workers=[0, 2],
        sample_count=3,
        batch_size=1,
        warmup_batches=1,
        measure_batches=2,
        prefetch_factor=2,
    )

    assert [result["execution_order"] for result in results] == [
        ["old", "hdf5"],
        ["hdf5", "old"],
    ]
    for worker_result in results:
        assert set(worker_result["backends"]) == {"old", "hdf5"}
        for result in worker_result["backends"].values():
            assert result["observed_batches"] == 2
            assert result["observed_samples"] == 2
            assert len(result["batch_seconds"]) == 2
            assert result["samples_per_second"] > 0
            assert result["median_batch_seconds"] > 0
            assert result["p95_batch_seconds"] >= result["median_batch_seconds"]
            assert len(result["worker_pids"]) == worker_result["workers"]
            assert result["workers_shutdown"] is True
            assert (
                result["peak_aggregate_worker_rss_bytes"]
                >= result["aggregate_worker_rss_bytes"]
            )
            assert result["rss_sample_count"] >= 2
            assert result["read_bytes"] >= 0
            assert result["read_chars"] >= 0
            assert result["cpu_seconds"] >= 0
            assert result["cpu_utilization_percent"] >= 0
            assert result["counter_method"]
    assert new._handles == {}


def test_benchmark_insufficient_stream_fails_and_closes_dataset():
    old = _BenchmarkFakeOldDataset()
    new = _BenchmarkFakeHDF5Dataset()
    plans = benchmark.build_sample_plans(
        old,
        new,
        benchmark.map_episode_pairs(old, new, episode_limit=1),
    )
    view = benchmark.DeterministicHDF5View(new, plans, sample_count=1)
    with pytest.raises(RuntimeError, match="insufficient.*measurement"):
        benchmark.measure_dataloader(
            view,
            workers=0,
            batch_size=1,
            warmup_batches=1,
            measure_batches=1,
            prefetch_factor=2,
        )
    assert new.closed is True


def _write_benchmark_configs(tmp_path, *, compression="none"):
    old, new, manifest, stats, source, cache = _make_real_benchmark_datasets(
        tmp_path / "fixture", compression=compression
    )
    old.close() if hasattr(old, "close") else None
    new.close()
    relative_stats = os.path.relpath(stats, GE_ACT_ROOT)
    common = {
        "source_fps": 20,
        "sample_n_frames": 500,
        "valid_cam": list(CAMERAS),
        "chunk": 9,
        "action_chunk": 36,
        "n_previous": 4,
        "previous_pick_mode": "random",
        "action_type": "absolute",
        "action_space": "eef",
    }
    old_config = {
        "train_data_class_path": "data/lerobot_like_dataset.py",
        "train_data_class": "CustomLeRobotDataset",
        "data": {
            "train": {
                **common,
                "data_roots": [str(source)],
                "domains": ["domain"],
                "sample_size": [256, 256],
                "preprocess": "resize",
                "predecoded_video_root": str(cache),
                "require_predecoded": True,
                "random_crop": False,
                "ignore_seek": False,
                "train_dataset": True,
                "action_key": "action",
                "state_key": "observation.state",
                "stat_file": relative_stats,
            }
        },
    }
    new_config = {
        "train_data_class_path": "data/libero_fastwam_hdf5_dataset.py",
        "train_data_class": "LiberoFastWAMHDF5Dataset",
        "data": {
            "train": {
                **common,
                "manifest_path": "/must/be/overridden/manifest.json",
                "stat_file": relative_stats,
                "train_dataset": True,
            }
        },
    }
    old_path = tmp_path / "old.yaml"
    new_path = tmp_path / "hdf5.yaml"
    old_path.write_text(yaml.safe_dump(old_config), encoding="utf-8")
    new_path.write_text(yaml.safe_dump(new_config), encoding="utf-8")
    return old_path, new_path, manifest


def _run_benchmark_cli(
    tmp_path,
    old_config,
    hdf5_config,
    manifest,
    output,
    *,
    episodes=1,
    mode="parity",
    compression="none",
    compare_report=None,
    workers=(0,),
):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(GE_ACT_ROOT.parent), environment.get("PYTHONPATH", ""))
    )
    command = [
        sys.executable,
        "-m",
        "ge_act.scripts.benchmark_libero_fastwam_hdf5",
        "--old-config",
        str(old_config),
        "--hdf5-config",
        str(hdf5_config),
        "--hdf5-manifest",
        str(manifest),
        "--output-json",
        str(output),
        "--mode",
        mode,
        "--episodes",
        str(episodes),
        "--samples",
        "1",
        "--workers",
        *(str(worker) for worker in workers),
        "--batch-size",
        "1",
        "--warmup-batches",
        "0",
        "--measure-batches",
        "1",
        "--run-label",
        "warm",
        "--compression",
        compression,
    ]
    if compare_report is not None:
        command.extend(("--compare-report", str(compare_report)))
    return subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
    )


def test_benchmark_cli_help_and_config_resolution_from_different_cwd(tmp_path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(GE_ACT_ROOT.parent)
    help_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ge_act.scripts.benchmark_libero_fastwam_hdf5",
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert help_result.returncode == 0
    assert "--hdf5-manifest" in help_result.stdout

    old_config, hdf5_config, manifest = _write_benchmark_configs(tmp_path)
    output = tmp_path / "success.json"
    before = list(sys.path)
    result = _run_benchmark_cli(tmp_path, old_config, hdf5_config, manifest, output)
    assert list(sys.path) == before
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["parity"]["passed"] is True
    assert report["mapping"]["selected_pairs"] == 1
    assert report["run_label"] == "warm"
    assert report["filesystem"]["manifest"]
    assert report["filesystem"]["old_predecoded_video_root"]


def test_benchmark_dataset_construction_restores_sys_path(tmp_path, monkeypatch):
    old_config, hdf5_config, manifest = _write_benchmark_configs(tmp_path)
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    before = list(sys.path)

    old, _ = benchmark.construct_train_dataset(old_config)
    new, _ = benchmark.construct_train_dataset(hdf5_config, manifest_override=manifest)

    assert list(sys.path) == before
    assert len(old) == len(new) == 1
    new.close()


def test_benchmark_cli_parity_failure_writes_report_and_exits_nonzero(tmp_path):
    old_config, hdf5_config, manifest = _write_benchmark_configs(tmp_path)
    _, records = schema.load_manifest(manifest)
    with h5py.File(records[0].shard_path, "r+") as shard:
        shard[records[0].group]["rgb_main"][:] = 255
    output = tmp_path / "failure.json"

    result = _run_benchmark_cli(tmp_path, old_config, hdf5_config, manifest, output)

    assert result.returncode != 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["parity"]["passed"] is False
    assert len(report["parity"]["failures"]) == 1
    assert report["parity"]["pairs"][0]["domain"] == "domain"


def test_benchmark_cli_lzf_both_merges_none_report(tmp_path):
    none_old, none_hdf5, none_manifest = _write_benchmark_configs(
        tmp_path / "shared", compression="none"
    )
    none_report = tmp_path / "none.json"
    none_result = _run_benchmark_cli(
        tmp_path,
        none_old,
        none_hdf5,
        none_manifest,
        none_report,
        mode="both",
        compression="none",
        workers=(0, 2, 4, 8),
    )
    assert none_result.returncode == 0, none_result.stdout + none_result.stderr

    lzf_old, lzf_hdf5, lzf_manifest = _write_benchmark_configs(
        tmp_path / "shared", compression="lzf"
    )
    lzf_report = tmp_path / "lzf.json"
    lzf_result = _run_benchmark_cli(
        tmp_path,
        lzf_old,
        lzf_hdf5,
        lzf_manifest,
        lzf_report,
        mode="both",
        compression="lzf",
        compare_report=none_report,
        workers=(0, 2, 4, 8),
    )
    assert lzf_result.returncode == 0, lzf_result.stdout + lzf_result.stderr
    report = json.loads(lzf_report.read_text(encoding="utf-8"))
    assert report["parity"]["passed"] is True
    assert [entry["workers"] for entry in report["throughput"]] == [0, 2, 4, 8]
    assert all(
        entry["backends"]["hdf5"]["observed_batches"] == 1
        for entry in report["throughput"]
    )
    selection = report["compression_selection"]
    assert set(selection["by_compression"]) == {"none", "lzf"}
    assert selection["selected_format"] in {"none", "lzf"}
    assert selection["threshold"] == 0.95


def test_benchmark_mapping_requires_exact_requested_episode_count():
    old = Namespace(dataset=[_old_episode_record("domain", 0)])
    new = Namespace(records=[_hdf5_episode_record("domain", 0)])
    with pytest.raises(ValueError, match="requested=2.*available=1"):
        benchmark.map_episode_pairs(old, new, episode_limit=2)


def test_benchmark_cli_rejects_partial_episode_selection(tmp_path):
    old_config, hdf5_config, manifest = _write_benchmark_configs(tmp_path)
    output = tmp_path / "partial.json"
    result = _run_benchmark_cli(
        tmp_path, old_config, hdf5_config, manifest, output, episodes=2
    )
    assert result.returncode != 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert "requested=2" in report["fatal_error"]
    assert "available=1" in report["fatal_error"]


def test_benchmark_peak_rss_monitor_captures_transient_and_hwm(monkeypatch):
    samples = iter([100, 900, 200, 200])
    monkeypatch.setattr(
        benchmark,
        "aggregate_worker_rss_bytes",
        lambda _pids: next(samples, 200),
    )
    monkeypatch.setattr(
        benchmark,
        "aggregate_worker_peak_rss_bytes",
        lambda _pids: 700,
    )
    monitor = benchmark.PeakWorkerRSSMonitor([1, 2], interval_seconds=0.001)
    monitor.start()
    time.sleep(0.01)
    result = monitor.stop()
    assert result["peak_aggregate_worker_rss_bytes"] == 900
    assert result["worker_peak_rss_hwm_bytes"] == 700
    assert result["rss_sample_count"] >= 3
    assert "poll" in result["peak_rss_method"]


def test_benchmark_proc_peak_rss_and_counters_fallback(tmp_path, monkeypatch):
    status = tmp_path / "status"
    io_path = tmp_path / "io"
    stat_path = tmp_path / "stat"
    status.write_text("VmRSS:\t12 kB\nVmHWM:\t34 kB\n", encoding="utf-8")
    io_path.write_text("rchar: 1234\nread_bytes: 5678\n", encoding="utf-8")
    fields = ["R"] + ["0"] * 10 + ["200", "100"] + ["0"] * 20
    stat_path.write_text(f"99 (worker name) {' '.join(fields)}\n", encoding="utf-8")
    monkeypatch.setattr(benchmark, "_psutil", None)
    monkeypatch.setattr(benchmark, "_proc_status_path", lambda _pid: status)
    monkeypatch.setattr(benchmark, "_proc_io_path", lambda _pid: io_path)
    monkeypatch.setattr(benchmark, "_proc_stat_path", lambda _pid: stat_path)
    monkeypatch.setattr(benchmark.os, "sysconf", lambda _name: 100)

    assert benchmark.read_process_peak_rss_bytes(99) == 34 * 1024
    counters = benchmark.read_process_counters(99)
    assert counters == {
        "read_bytes": 5678,
        "read_chars": 1234,
        "cpu_seconds": 3.0,
        "method": "proc",
    }
    status.unlink()
    io_path.unlink()
    stat_path.unlink()
    assert benchmark.read_process_peak_rss_bytes(99) == 0
    assert benchmark.read_process_counters(99)["method"] == "vanished"


def test_benchmark_counter_delta_reports_one_core_convention():
    result = benchmark.counter_delta(
        {
            "read_bytes": 100,
            "read_chars": 1000,
            "cpu_seconds": 4.0,
            "method": "proc",
        },
        {
            "read_bytes": 500,
            "read_chars": 1800,
            "cpu_seconds": 6.5,
            "method": "proc",
        },
        elapsed_seconds=1.0,
    )
    assert result["read_bytes"] == 400
    assert result["read_chars"] == 800
    assert result["cpu_seconds"] == 2.5
    assert result["cpu_core_equivalents"] == 2.5
    assert result["cpu_utilization_percent"] == 250.0
    assert result["counter_method"] == "proc"
    assert "100%" in result["cpu_utilization_caveat"]


def _compression_report(compression, speed, workers=2, run_label="warm"):
    return {
        "host": "benchmark-host",
        "git": {"sha": "a" * 40, "dirty": False, "fingerprint": None},
        "compression": compression,
        "run_label": run_label,
        "arguments": {
            "episodes": 64,
            "samples": 1024,
            "workers": [0, 2, 4, 8],
            "batch_size": 8,
            "warmup_batches": 20,
            "measure_batches": 100,
            "prefetch_factor": 4,
        },
        "parity": {"passed": True},
        "mapping": {
            "pairs": [
                {
                    "domain": "domain",
                    "episode_index": 0,
                    "old_index": 0,
                    "hdf5_index": 0,
                    "caption": "pick up the bowl",
                    "length": 50,
                }
            ]
        },
        "filesystem": {
            "manifest": "nfs",
            "old_data": {"/shared/source": "nfs"},
            "old_predecoded_video_root": "nfs",
            "hdf5_shards": {
                f"/{compression}/shard_0.h5": "nfs",
                f"/{compression}/shard_1.h5": "nfs",
            },
        },
        "throughput": [
            {
                "workers": workers,
                "backends": {"hdf5": {"samples_per_second": speed}},
            }
        ],
    }


def _add_compression_parity_pairs(report):
    report["parity"]["pairs"] = [
        {
            "domain": "domain",
            "episode_index": 0,
            "old_index": 0,
            "hdf5_index": 0,
            "frame_indexes": [1, 2],
            "action_indexes": [1, 2, 3],
        }
    ]
    return report


def test_benchmark_merge_compression_reports_selects_with_95_percent_rule():
    current = _add_compression_parity_pairs(_compression_report("lzf", 96.0, workers=4))
    other = _add_compression_parity_pairs(_compression_report("none", 100.0, workers=2))
    selection = benchmark.merge_compression_reports(
        current,
        other,
    )
    assert selection == {
        "by_compression": {
            "none": {"best_samples_per_second": 100.0, "workers": 2},
            "lzf": {"best_samples_per_second": 96.0, "workers": 4},
        },
        "selected_format": "lzf",
        "selected_workers": 4,
        "threshold": 0.95,
    }


@pytest.mark.parametrize("problem", ["same", "parity", "label", "arguments"])
def test_benchmark_merge_compression_reports_rejects_incomparable_runs(problem):
    current = _add_compression_parity_pairs(_compression_report("lzf", 96.0))
    other = _add_compression_parity_pairs(_compression_report("none", 100.0))
    if problem == "same":
        other["compression"] = "lzf"
    elif problem == "parity":
        other["parity"]["passed"] = False
    elif problem == "label":
        other["run_label"] = "cold"
    else:
        other["arguments"]["measure_batches"] = 99
    with pytest.raises(ValueError, match="compression|parity|run_label|arguments"):
        benchmark.merge_compression_reports(current, other)


@pytest.mark.parametrize(
    "problem", ["host", "sha", "dirty", "mapping", "parity_pair", "filesystem"]
)
def test_benchmark_merge_rejects_execution_context_mismatch(problem):
    current = _add_compression_parity_pairs(_compression_report("lzf", 96.0))
    other = _add_compression_parity_pairs(_compression_report("none", 100.0))
    if problem == "host":
        other["host"] = "other-host"
    elif problem == "sha":
        other["git"]["sha"] = "b" * 40
    elif problem == "dirty":
        other["git"]["dirty"] = True
    elif problem == "mapping":
        other["mapping"]["pairs"][0]["old_index"] = 1
    elif problem == "parity_pair":
        other["parity"]["pairs"][0]["frame_indexes"] = [2, 3]
    else:
        other["filesystem"]["manifest"] = "ext4"
    with pytest.raises(ValueError, match="host|git|mapping|parity|filesystem"):
        benchmark.merge_compression_reports(current, other)


def test_benchmark_merge_rejects_missing_git_sha():
    current = _add_compression_parity_pairs(_compression_report("lzf", 96.0))
    other = _add_compression_parity_pairs(_compression_report("none", 100.0))
    other["git"]["sha"] = None
    with pytest.raises(ValueError, match="git.*sha"):
        benchmark.merge_compression_reports(current, other)


def test_benchmark_merge_rejects_missing_git_fingerprint_field():
    current = _add_compression_parity_pairs(_compression_report("lzf", 96.0))
    other = _add_compression_parity_pairs(_compression_report("none", 100.0))
    other["git"].pop("fingerprint")
    with pytest.raises(ValueError, match="git.*fingerprint"):
        benchmark.merge_compression_reports(current, other)


def test_benchmark_merge_accepts_matching_dirty_tree_fingerprint():
    current = _add_compression_parity_pairs(_compression_report("lzf", 96.0))
    other = _add_compression_parity_pairs(_compression_report("none", 100.0))
    for report in (current, other):
        report["git"] = {
            "sha": "a" * 40,
            "dirty": True,
            "fingerprint": "b" * 64,
        }

    selection = benchmark.merge_compression_reports(current, other)

    assert selection["selected_format"] == "lzf"


@pytest.mark.parametrize(
    ("dirty", "fingerprint"),
    [
        (False, "b" * 64),
        (True, None),
        (True, "not-a-sha256"),
    ],
)
def test_benchmark_merge_rejects_invalid_dirty_tree_fingerprint(dirty, fingerprint):
    current = _add_compression_parity_pairs(_compression_report("lzf", 96.0))
    other = _add_compression_parity_pairs(_compression_report("none", 100.0))
    for report in (current, other):
        report["git"] = {
            "sha": "a" * 40,
            "dirty": dirty,
            "fingerprint": fingerprint,
        }

    with pytest.raises(ValueError, match="git.*fingerprint"):
        benchmark.merge_compression_reports(current, other)


def test_benchmark_merge_rejects_different_dirty_tree_fingerprint():
    current = _add_compression_parity_pairs(_compression_report("lzf", 96.0))
    other = _add_compression_parity_pairs(_compression_report("none", 100.0))
    current["git"] = {
        "sha": "a" * 40,
        "dirty": True,
        "fingerprint": "b" * 64,
    }
    other["git"] = {
        "sha": "a" * 40,
        "dirty": True,
        "fingerprint": "c" * 64,
    }

    with pytest.raises(ValueError, match="git.*match"):
        benchmark.merge_compression_reports(current, other)


@pytest.mark.parametrize(
    ("field", "change"),
    [
        ("caption", "place the bowl"),
        ("length", 49),
        pytest.param("caption", None, id="missing-caption"),
        pytest.param("length", None, id="missing-length"),
    ],
)
def test_benchmark_merge_rejects_mapping_metadata_mismatch(field, change):
    current = _add_compression_parity_pairs(_compression_report("lzf", 96.0))
    other = _add_compression_parity_pairs(_compression_report("none", 100.0))
    pair = other["mapping"]["pairs"][0]
    if change is None:
        pair.pop(field)
    else:
        pair[field] = change

    with pytest.raises(ValueError, match="mapping"):
        benchmark.merge_compression_reports(current, other)


def _run_git(repo, *arguments):
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _make_benchmark_git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init")
    _run_git(repo, "config", "user.email", "benchmark@example.com")
    _run_git(repo, "config", "user.name", "Benchmark Test")
    (repo / ".gitignore").write_text("ignored.bin\n", encoding="utf-8")
    (repo / "tracked.bin").write_bytes(b"tracked\x00base")
    _run_git(repo, "add", ".gitignore", "tracked.bin")
    _run_git(repo, "commit", "-m", "initial")
    return repo


def _assert_dirty_fingerprint(metadata):
    assert metadata["dirty"] is True
    assert len(metadata["fingerprint"]) == 64
    assert set(metadata["fingerprint"]) <= set("0123456789abcdef")


def test_benchmark_git_metadata_fingerprints_staged_and_unstaged_binary_diff(
    tmp_path, monkeypatch
):
    repo = _make_benchmark_git_repo(tmp_path)
    monkeypatch.setattr(benchmark, "REPOSITORY_ROOT", repo)
    clean = benchmark._git_metadata()
    assert clean == {
        "sha": _run_git(repo, "rev-parse", "HEAD"),
        "dirty": False,
        "fingerprint": None,
    }

    tracked = repo / "tracked.bin"
    tracked.write_bytes(b"staged\x00binary")
    _run_git(repo, "add", "tracked.bin")
    staged = benchmark._git_metadata()
    _assert_dirty_fingerprint(staged)
    assert benchmark._git_metadata() == staged

    tracked.write_bytes(b"staged\x00binary\x00plus-unstaged")
    staged_and_unstaged = benchmark._git_metadata()
    _assert_dirty_fingerprint(staged_and_unstaged)
    assert staged_and_unstaged["fingerprint"] != staged["fingerprint"]


def test_benchmark_git_metadata_streams_untracked_identity_mode_and_content(
    tmp_path, monkeypatch
):
    repo = _make_benchmark_git_repo(tmp_path)
    monkeypatch.setattr(benchmark, "REPOSITORY_ROOT", repo)
    ignored = repo / "ignored.bin"
    ignored.write_bytes(b"not part of the dirty tree")
    assert benchmark._git_metadata()["dirty"] is False

    payload = repo / "payload.bin"
    payload.write_bytes(b"a" * (2 * 1024 * 1024 + 17))
    payload.chmod(0o600)
    first = benchmark._git_metadata()
    _assert_dirty_fingerprint(first)
    assert benchmark._git_metadata() == first

    payload.write_bytes(b"a" * (2 * 1024 * 1024 + 16) + b"b")
    content_changed = benchmark._git_metadata()
    assert content_changed["fingerprint"] != first["fingerprint"]

    payload.chmod(0o700)
    mode_changed = benchmark._git_metadata()
    assert mode_changed["fingerprint"] != content_changed["fingerprint"]

    renamed = repo / "renamed.bin"
    payload.rename(renamed)
    path_changed = benchmark._git_metadata()
    assert path_changed["fingerprint"] != mode_changed["fingerprint"]

    renamed.unlink()
    renamed.symlink_to("target-a")
    symlink_a = benchmark._git_metadata()
    assert symlink_a["fingerprint"] != path_changed["fingerprint"]
    renamed.unlink()
    renamed.symlink_to("target-b")
    symlink_b = benchmark._git_metadata()
    assert symlink_b["fingerprint"] != symlink_a["fingerprint"]


class _GateProbeDataset(torch.utils.data.Dataset):
    def __init__(self, count, reads):
        self.count = count
        self.reads = reads

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        with self.reads.get_lock():
            self.reads.value += 1
        return _benchmark_sample()


class _ExplodingGateDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.closed = False

    def __len__(self):
        return 4

    def __getitem__(self, _index):
        raise RuntimeError("worker getitem exploded")

    def close(self):
        self.closed = True


class _TrackingReadyQueue:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.closed = False
        self.joined = False

    def put(self, value):
        return self.wrapped.put(value)

    def get(self, *args, **kwargs):
        return self.wrapped.get(*args, **kwargs)

    def close(self):
        self.closed = True
        return self.wrapped.close()

    def join_thread(self):
        self.joined = True
        return self.wrapped.join_thread()


class _TrackingProcessContext:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.queues = []

    def Event(self):
        return self.wrapped.Event()

    def Queue(self):
        queue = _TrackingReadyQueue(self.wrapped.Queue())
        self.queues.append(queue)
        return queue


class _TrackingRSSMonitor(benchmark.PeakWorkerRSSMonitor):
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_calls = 0
        self.__class__.instances.append(self)

    def stop(self):
        self.stop_calls += 1
        return super().stop()


def test_benchmark_resource_counters_gate_prefetch_and_cover_full_stream(monkeypatch):
    context = mp.get_context()
    reads = context.Value("i", 0)
    calls = []

    def counters(_pids):
        calls.append(reads.value)
        return {
            "read_bytes": reads.value * 10,
            "read_chars": reads.value * 20,
            "cpu_seconds": float(reads.value),
            "method": "fake",
        }

    monkeypatch.setattr(benchmark, "aggregate_process_counters", counters)
    result = benchmark.measure_dataloader(
        _GateProbeDataset(8, reads),
        workers=2,
        batch_size=2,
        warmup_batches=1,
        measure_batches=1,
        prefetch_factor=2,
    )
    assert calls == [0, 8]
    assert result["counter_window"] == "full_stream_including_warmup_measurement_drain"
    assert result["resource_observed_batches"] == 4
    assert result["resource_observed_samples"] == 8
    assert result["drain_batches"] == 2
    assert result["drain_samples"] == 4
    assert result["resource_elapsed_seconds"] > 0


def test_benchmark_worker_getitem_error_cleans_up_without_deadlock(monkeypatch):
    actual_context = mp.get_context()
    tracking_context = _TrackingProcessContext(actual_context)
    monkeypatch.setattr(
        benchmark,
        "multiprocessing",
        Namespace(get_context=lambda: tracking_context),
    )
    _TrackingRSSMonitor.instances = []
    monkeypatch.setattr(benchmark, "PeakWorkerRSSMonitor", _TrackingRSSMonitor)
    dataset = _ExplodingGateDataset()
    child_pids_before = {process.pid for process in mp.active_children()}

    start = time.perf_counter()
    with pytest.raises(RuntimeError, match="worker getitem exploded"):
        benchmark.measure_dataloader(
            dataset,
            workers=2,
            batch_size=1,
            warmup_batches=1,
            measure_batches=1,
            prefetch_factor=2,
        )
    elapsed = time.perf_counter() - start

    assert elapsed < 10.0
    assert dataset.closed is True
    assert len(tracking_context.queues) == 1
    assert tracking_context.queues[0].closed is True
    assert tracking_context.queues[0].joined is True
    assert len(_TrackingRSSMonitor.instances) == 1
    monitor = _TrackingRSSMonitor.instances[0]
    assert monitor.stop_calls == 1
    assert monitor._thread is not None and not monitor._thread.is_alive()
    assert {process.pid for process in mp.active_children()} <= child_pids_before


@pytest.mark.parametrize("workers", [4, 8])
def test_benchmark_lightweight_high_worker_shutdown(workers):
    old = _BenchmarkFakeOldDataset()
    new = _BenchmarkFakeHDF5Dataset()
    plans = benchmark.build_sample_plans(
        old, new, benchmark.map_episode_pairs(old, new, episode_limit=1)
    )
    result = benchmark.measure_dataloader(
        benchmark.DeterministicHDF5View(new, plans, sample_count=workers),
        workers=workers,
        batch_size=workers,
        warmup_batches=0,
        measure_batches=1,
        prefetch_factor=1,
    )
    assert result["workers_shutdown"] is True
    assert len(result["worker_pids"]) == workers
    assert "peak_aggregate_worker_rss_bytes" in result
    assert "read_bytes" in result
    assert "cpu_utilization_percent" in result
