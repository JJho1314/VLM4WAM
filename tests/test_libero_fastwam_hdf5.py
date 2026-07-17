import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from ge_act.data import libero_fastwam_hdf5_schema as schema


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
    group.create_dataset(
        "rgb_main", shape=(record.length, 256, 256, 3), dtype=np.uint8
    )
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
def test_validate_episode_group_rejects_wrong_scalar_metadata(
    tmp_path, dataset, value
):
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
