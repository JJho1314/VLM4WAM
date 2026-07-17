"""Versioned manifest and HDF5 shard validation for LIBERO FastWAM data."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCHEMA_VERSION = 1
FIXED_CONTRACT = {
    "camera_names": ["main", "wrist"],
    "image_size": [256, 256],
    "source_fps": 20,
    "n_previous": 4,
    "chunk": 9,
    "action_chunk": 36,
    "action_type": "absolute",
    "action_space": "eef",
}
DATASET_DECLARATIONS = {
    "rgb_main": {"shape_tail": [256, 256, 3], "dtype": "uint8"},
    "rgb_wrist": {"shape_tail": [256, 256, 3], "dtype": "uint8"},
    "action": {"width": 7, "dtype": "float32"},
    "state": {"width": 8, "dtype": "float32"},
}
MANIFEST_FIELDS = {
    "schema_version",
    "compression",
    "source_roots",
    "datasets",
    "converter_fingerprint",
    "episodes",
} | set(FIXED_CONTRACT)
EPISODE_FIELDS = {
    "key",
    "shard",
    "group",
    "caption",
    "domain",
    "episode_index",
    "length",
}


@dataclass(frozen=True)
class EpisodeRecord:
    key: str
    shard_path: Path
    group: str
    caption: str
    domain: str
    episode_index: int
    length: int


def load_manifest(path: Path) -> tuple[dict[str, Any], list[EpisodeRecord]]:
    """Load and validate a manifest relative to its containing directory."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    records = validate_manifest(payload, path.parent)
    return payload, records


def _matches_exact_structure(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(
            _matches_exact_structure(actual[key], value)
            for key, value in expected.items()
        )
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _matches_exact_structure(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    return actual == expected


def validate_manifest(payload: Any, root: Path) -> list[EpisodeRecord]:
    """Validate the canonical LIBERO contract and return typed records."""
    if type(payload) is not dict:
        raise ValueError("manifest payload must be a dict")

    actual_fields = set(payload)
    if actual_fields != MANIFEST_FIELDS:
        missing = sorted(MANIFEST_FIELDS - actual_fields)
        unexpected = sorted(actual_fields - MANIFEST_FIELDS)
        details = []
        if missing:
            details.append(f"missing fields {missing!r}")
        if unexpected:
            details.append(f"unexpected fields {unexpected!r}")
        raise ValueError(f"manifest has {', '.join(details)}")

    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")

    for field, expected in FIXED_CONTRACT.items():
        actual = payload[field]
        if not _matches_exact_structure(actual, expected):
            raise ValueError(f"{field} must be {expected!r}, got {actual!r}")

    compression = payload.get("compression")
    if type(compression) is not str or compression not in ("none", "lzf"):
        raise ValueError("compression must be 'none' or 'lzf'")

    source_roots = payload.get("source_roots")
    if (
        type(source_roots) is not list
        or not source_roots
        or any(type(item) is not str or not item for item in source_roots)
    ):
        raise ValueError("source_roots must be a non-empty list of non-empty strings")

    datasets = payload.get("datasets")
    if not _matches_exact_structure(datasets, DATASET_DECLARATIONS):
        raise ValueError(
            f"datasets must be exactly {DATASET_DECLARATIONS!r}, got {datasets!r}"
        )

    fingerprint = payload.get("converter_fingerprint")
    if type(fingerprint) is not str or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise ValueError(
            "converter_fingerprint must be a 64-character lowercase hexadecimal "
            "SHA-256 string"
        )

    raw_records = payload.get("episodes")
    if type(raw_records) is not list or not raw_records:
        raise ValueError("episodes must be a non-empty list")

    root = Path(root).resolve()
    seen: set[str] = set()
    records = []
    for index, item in enumerate(raw_records):
        if type(item) is not dict:
            raise ValueError(f"episode {index} must be a dict")
        actual_fields = set(item)
        if actual_fields != EPISODE_FIELDS:
            missing = sorted(EPISODE_FIELDS - actual_fields)
            unexpected = sorted(actual_fields - EPISODE_FIELDS)
            details = []
            if missing:
                details.append(f"missing fields {missing!r}")
            if unexpected:
                details.append(f"unexpected fields {unexpected!r}")
            raise ValueError(f"episode {index} has {', '.join(details)}")

        for field in ("key", "shard", "group", "caption", "domain"):
            if type(item[field]) is not str:
                raise ValueError(f"episode {index} field {field} must be a string")
        for field in ("key", "shard", "group", "caption", "domain"):
            if not item[field]:
                raise ValueError(f"episode {index} field {field} must not be empty")
        for field in ("episode_index", "length"):
            if type(item[field]) is not int:
                raise ValueError(f"episode {index} field {field} must be an integer")

        key = item["key"]
        if key in seen:
            raise ValueError(f"duplicate episode key: {key}")
        seen.add(key)

        expected_group = f"episodes/{key}"
        if item["group"] != expected_group:
            raise ValueError(
                f"episode {index} field group must be {expected_group!r}, "
                f"got {item['group']!r}"
            )

        episode_index = item["episode_index"]
        if episode_index < 0:
            raise ValueError(
                f"episode {index} field episode_index must be non-negative"
            )
        length = item["length"]
        if length < 2:
            raise ValueError(f"episode {key} has invalid length {length}")

        shard_reference = Path(item["shard"])
        if shard_reference.is_absolute():
            raise ValueError(
                f"episode {index} shard path must be relative: {item['shard']!r}"
            )
        shard_path = (root / shard_reference).resolve()
        try:
            shard_path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"episode {index} shard resolves outside manifest root: "
                f"{item['shard']!r}"
            ) from error
        if not shard_path.is_file():
            raise FileNotFoundError(f"missing HDF5 shard: {shard_path}")

        records.append(
            EpisodeRecord(
                key=key,
                shard_path=shard_path,
                group=item["group"],
                caption=item["caption"],
                domain=item["domain"],
                episode_index=episode_index,
                length=length,
            )
        )
    return records


def _require_dataset(group: h5py.Group, name: str) -> h5py.Dataset:
    if name not in group or not isinstance(group[name], h5py.Dataset):
        raise ValueError(f"missing episode dataset: {name}")
    return group[name]


def _validate_string_scalar(
    group: h5py.Group, name: str, expected: str
) -> None:
    dataset = _require_dataset(group, name)
    string_info = h5py.check_string_dtype(dataset.dtype)
    if dataset.shape != () or string_info is None or string_info.encoding != "utf-8":
        raise ValueError(f"{name} must be a UTF-8 scalar dataset")
    actual = dataset.asstr()[()]
    if actual != expected:
        raise ValueError(f"{name} must be {expected!r}, got {actual!r}")


def _validate_int64_scalar(group: h5py.Group, name: str, expected: int) -> None:
    dataset = _require_dataset(group, name)
    if dataset.shape != () or dataset.dtype != np.dtype("int64"):
        raise ValueError(f"{name} must be an int64 scalar dataset")
    actual = dataset[()]
    if actual != expected:
        raise ValueError(f"{name} must be {expected!r}, got {actual!r}")


def validate_episode_group(group: h5py.Group, record: EpisodeRecord) -> None:
    """Validate one HDF5 episode group against its manifest record."""
    actual_group = group.name.removeprefix("/")
    if actual_group != record.group:
        raise ValueError(
            f"group path must be {record.group!r}, got {actual_group!r}"
        )

    _validate_string_scalar(group, "caption", record.caption)
    _validate_string_scalar(group, "domain", record.domain)
    _validate_int64_scalar(group, "episode_index", record.episode_index)
    _validate_int64_scalar(group, "length", record.length)

    expected_shapes = {
        "rgb_main": (record.length, 256, 256, 3),
        "rgb_wrist": (record.length, 256, 256, 3),
        "action": (record.length, 7),
        "state": (record.length, 8),
    }
    expected_dtypes = {
        "rgb_main": np.dtype("uint8"),
        "rgb_wrist": np.dtype("uint8"),
        "action": np.dtype("float32"),
        "state": np.dtype("float32"),
    }
    for name, expected_shape in expected_shapes.items():
        dataset = _require_dataset(group, name)
        expected_dtype = expected_dtypes[name]
        if dataset.shape != expected_shape or dataset.dtype != expected_dtype:
            raise ValueError(
                f"{name} must have shape {expected_shape} and dtype "
                f"{expected_dtype}, got shape {dataset.shape} and dtype "
                f"{dataset.dtype}"
            )


def atomic_write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Durably write JSON and atomically replace the target manifest."""
    path = Path(path)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
