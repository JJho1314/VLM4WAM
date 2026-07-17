"""Versioned manifest and HDF5 shard validation for LIBERO FastWAM data."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class EpisodeRecord:
    key: str
    shard_path: Path
    group: str
    domain: str
    episode_index: int
    length: int


def load_manifest(path: Path) -> list[EpisodeRecord]:
    """Load and validate a manifest relative to its containing directory."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return validate_manifest(payload, path.parent)


def validate_manifest(payload: dict[str, Any], root: Path) -> list[EpisodeRecord]:
    """Validate the fixed LIBERO contract and return typed episode records."""
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    for field, expected in FIXED_CONTRACT.items():
        if payload.get(field) != expected:
            raise ValueError(
                f"{field} must be {expected!r}, got {payload.get(field)!r}"
            )
    compression = payload.get("compression")
    if compression not in ("none", "lzf"):
        raise ValueError("compression must be 'none' or 'lzf'")
    raw_records = payload.get("episodes")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("episodes must be a non-empty list")

    root = Path(root)
    seen: set[str] = set()
    records = []
    for item in raw_records:
        key = str(item["key"])
        if key in seen:
            raise ValueError(f"duplicate episode key: {key}")
        seen.add(key)
        shard_path = root / str(item["shard"])
        if not shard_path.is_file():
            raise FileNotFoundError(f"missing HDF5 shard: {shard_path}")
        length = int(item["length"])
        if length < 2:
            raise ValueError(f"episode {key} has invalid length {length}")
        records.append(
            EpisodeRecord(
                key=key,
                shard_path=shard_path,
                group=str(item["group"]),
                domain=str(item["domain"]),
                episode_index=int(item["episode_index"]),
                length=length,
            )
        )
    return records


def validate_episode_group(group: Any, record: EpisodeRecord) -> None:
    """Validate one HDF5 episode group against its manifest record."""
    expected_metadata = {
        "key": record.key,
        "domain": record.domain,
        "episode_index": record.episode_index,
        "length": record.length,
    }
    for field, expected in expected_metadata.items():
        if field not in group.attrs:
            raise ValueError(f"missing episode metadata attribute: {field}")
        actual = group.attrs[field]
        if isinstance(actual, bytes):
            actual = actual.decode("utf-8")
        if actual != expected:
            raise ValueError(
                f"episode metadata {field} must be {expected!r}, got {actual!r}"
            )

    expected_rgb_shape = (record.length, 256, 256, 3)
    for name in ("rgb_main", "rgb_wrist"):
        if name not in group:
            raise ValueError(f"missing episode dataset: {name}")
        dataset = group[name]
        if dataset.shape != expected_rgb_shape or dataset.dtype != np.dtype("uint8"):
            raise ValueError(
                f"{name} must have shape {expected_rgb_shape} and dtype uint8, "
                f"got shape {dataset.shape} and dtype {dataset.dtype}"
            )

    for name in ("action", "state"):
        if name not in group:
            raise ValueError(f"missing episode dataset: {name}")
        dataset = group[name]
        if (
            len(dataset.shape) < 1
            or dataset.shape[0] != record.length
            or dataset.dtype != np.dtype("float32")
        ):
            raise ValueError(
                f"{name} must have leading dimension {record.length} and dtype "
                f"float32, got shape {dataset.shape} and dtype {dataset.dtype}"
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
