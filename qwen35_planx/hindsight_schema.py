"""Atomic shard and immutable memmap schema for video-hindsight targets."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from qwen35_planx.config import (
    CAMERA_NAMES,
    HindsightCacheMetadata,
    PlanGeometry,
)
from qwen35_planx.hashing import sha256_file, sha256_json
from qwen35_planx.hindsight_data import HindsightWindowRecord


_FORMAT_VERSION = 1
_ARRAY_SPECS: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {
    "codes": (np.dtype(np.uint16), (2, 4, 729)),
    "relevance_q": (np.dtype(np.uint8), (2, 4, 3, 729)),
    "relevance_scale": (np.dtype(np.float16), (2, 4, 3)),
    "confidence": (np.dtype(np.float16), (2, 4, 3)),
    "flow": (np.dtype(np.float16), (2, 3, 729, 3)),
    "phrase_embeddings": (np.dtype(np.float16), (3, 1152)),
}
_SHARD_FIELDS = set(_ARRAY_SPECS) | {
    "format_version",
    "metadata_json",
    "records_jsonl",
    "camera_names_json",
}
_MANIFEST_FIELDS = {
    "format_version",
    "metadata",
    "cache_hash",
    "split_hash",
    "teacher_hash",
    "camera_names",
    "num_samples",
    "index",
    "arrays",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _encode_text(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8).copy()


def _decode_text(value: np.ndarray, *, name: str) -> str:
    if value.ndim != 1 or value.dtype != np.uint8:
        raise ValueError(f"{name} must be a one-dimensional uint8 payload")
    try:
        return value.tobytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} must contain UTF-8") from exc


def _teacher_hash(metadata: HindsightCacheMetadata) -> str:
    return sha256_json(
        {
            "ta_tok_hash": metadata.ta_tok_hash,
            "siglip2_hash": metadata.siglip2_hash,
            "dinov3_hash": metadata.dinov3_hash,
        }
    )


def _validate_records(
    records: Sequence[HindsightWindowRecord],
    *,
    require_one_episode: bool = True,
) -> list[HindsightWindowRecord]:
    result = list(records)
    if not result or any(
        not isinstance(record, HindsightWindowRecord) for record in result
    ):
        raise ValueError("records must contain HindsightWindowRecord values")
    sample_ids = [record.sample_id for record in result]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("duplicate sample_id within trajectory shard")
    if sample_ids != sorted(sample_ids):
        raise ValueError("trajectory shard records must be ordered by sample_id")
    episode_keys = {record.episode_key for record in result}
    if require_one_episode and len(episode_keys) != 1:
        raise ValueError("a trajectory shard must contain exactly one episode")
    if any(tuple(record.camera_names) != CAMERA_NAMES for record in result):
        raise ValueError("camera_names must use canonical main/wrist order")
    return result


def _validate_and_convert_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    count: int,
    require_storage_dtypes: bool,
) -> dict[str, np.ndarray]:
    if set(arrays) != set(_ARRAY_SPECS):
        missing = sorted(set(_ARRAY_SPECS) - set(arrays))
        unexpected = sorted(set(arrays) - set(_ARRAY_SPECS))
        raise ValueError(
            f"array fields mismatch: missing={missing!r}, unexpected={unexpected!r}"
        )

    result: dict[str, np.ndarray] = {}
    for name, (storage_dtype, tail_shape) in _ARRAY_SPECS.items():
        array = np.asarray(arrays[name])
        expected_shape = (count, *tail_shape)
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {array.shape}"
            )
        if require_storage_dtypes and array.dtype != storage_dtype:
            raise ValueError(
                f"{name} must have dtype {storage_dtype}, got {array.dtype}"
            )
        result[name] = array

    codes = result["codes"]
    if codes.dtype.kind not in "iu" or codes.dtype.kind == "b":
        raise ValueError("codes must contain integers")
    if codes.size and (
        int(codes.min()) < 0 or int(codes.max()) >= PlanGeometry().visual_vocab_size
    ):
        raise ValueError("codes must be in the released TA-Tok vocabulary range")

    relevance_q = result["relevance_q"]
    if relevance_q.dtype.kind not in "iu" or relevance_q.dtype.kind == "b":
        raise ValueError("relevance_q must contain integers")
    if relevance_q.size and (
        int(relevance_q.min()) < 0 or int(relevance_q.max()) > 255
    ):
        raise ValueError("relevance_q must be in [0, 255]")
    if np.any(np.asarray(relevance_q, dtype=np.uint64).sum(axis=-1) == 0):
        raise ValueError("relevance_q maps must contain positive mass")

    for name in (
        "relevance_scale",
        "confidence",
        "flow",
        "phrase_embeddings",
    ):
        array = result[name]
        if array.dtype.kind != "f" or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain finite floating-point values")
    if np.any(result["relevance_scale"] <= 0):
        raise ValueError("relevance_scale must be positive")
    if np.any(result["confidence"] < 0) or np.any(result["confidence"] > 1):
        raise ValueError("confidence must be in [0, 1]")

    return {
        name: np.asarray(result[name], dtype=spec[0])
        for name, spec in _ARRAY_SPECS.items()
    }


def _records_jsonl(records: Sequence[HindsightWindowRecord]) -> str:
    return "".join(_canonical_json(record.to_dict()) + "\n" for record in records)


def _parse_records_jsonl(
    payload: str,
    *,
    require_one_episode: bool = True,
) -> list[HindsightWindowRecord]:
    records: list[HindsightWindowRecord] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid records_jsonl line {line_number}: {exc}"
            ) from exc
        records.append(HindsightWindowRecord.from_dict(item))
    return _validate_records(records, require_one_episode=require_one_episode)


class HindsightShardWriter:
    """Validate and atomically publish one trajectory-level ``.npz`` shard."""

    def __init__(
        self,
        path: Path | str,
        *,
        metadata: HindsightCacheMetadata,
    ) -> None:
        self.path = Path(path)
        if self.path.suffix != ".npz":
            raise ValueError("hindsight shard path must end in .npz")
        if not isinstance(metadata, HindsightCacheMetadata):
            raise TypeError("metadata must be HindsightCacheMetadata")
        self.metadata = metadata

    def write(
        self,
        records: Sequence[HindsightWindowRecord],
        *,
        codes: np.ndarray,
        relevance_q: np.ndarray,
        relevance_scale: np.ndarray,
        confidence: np.ndarray,
        flow: np.ndarray,
        phrase_embeddings: np.ndarray,
    ) -> Path:
        ordered_records = _validate_records(records)
        arrays = _validate_and_convert_arrays(
            {
                "codes": codes,
                "relevance_q": relevance_q,
                "relevance_scale": relevance_scale,
                "confidence": confidence,
                "flow": flow,
                "phrase_embeddings": phrase_embeddings,
            },
            count=len(ordered_records),
            require_storage_dtypes=False,
        )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as handle:
                np.savez(
                    handle,
                    format_version=np.asarray(_FORMAT_VERSION, dtype=np.int64),
                    metadata_json=_encode_text(
                        _canonical_json(self.metadata.to_dict())
                    ),
                    records_jsonl=_encode_text(_records_jsonl(ordered_records)),
                    camera_names_json=_encode_text(_canonical_json(list(CAMERA_NAMES))),
                    **arrays,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return self.path


@dataclass(frozen=True)
class _LoadedShard:
    records: tuple[HindsightWindowRecord, ...]
    arrays: Mapping[str, np.ndarray]


def _load_shard(
    path: Path,
    *,
    expected_metadata: HindsightCacheMetadata,
) -> _LoadedShard:
    try:
        archive_context = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"failed to load hindsight shard {path}: {exc}") from exc
    with archive_context as archive:
        actual_fields = set(archive.files)
        if actual_fields != _SHARD_FIELDS:
            missing = sorted(_SHARD_FIELDS - actual_fields)
            unexpected = sorted(actual_fields - _SHARD_FIELDS)
            raise ValueError(
                f"shard fields mismatch: missing={missing!r}, unexpected={unexpected!r}"
            )
        version = archive["format_version"]
        if (
            version.shape != ()
            or version.dtype != np.int64
            or int(version) != _FORMAT_VERSION
        ):
            raise ValueError(f"shard format_version must be {_FORMAT_VERSION}")
        try:
            metadata_payload = json.loads(
                _decode_text(archive["metadata_json"], name="metadata_json")
            )
        except json.JSONDecodeError as exc:
            raise ValueError("metadata_json must contain valid JSON") from exc
        metadata = HindsightCacheMetadata.from_dict(metadata_payload)
        if metadata != expected_metadata:
            raise ValueError("shard metadata does not match finalized cache metadata")
        try:
            cameras = json.loads(
                _decode_text(archive["camera_names_json"], name="camera_names_json")
            )
        except json.JSONDecodeError as exc:
            raise ValueError("camera_names_json must contain valid JSON") from exc
        if cameras != list(CAMERA_NAMES):
            raise ValueError("shard camera order must be canonical main/wrist")
        records = _parse_records_jsonl(
            _decode_text(archive["records_jsonl"], name="records_jsonl")
        )
        arrays = _validate_and_convert_arrays(
            {name: archive[name] for name in _ARRAY_SPECS},
            count=len(records),
            require_storage_dtypes=True,
        )
        copied = {name: np.array(value, copy=True) for name, value in arrays.items()}
    return _LoadedShard(records=tuple(records), arrays=copied)


def _write_index(path: Path, records: Sequence[HindsightWindowRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_canonical_json(record.to_dict()) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def finalize_hindsight_cache(
    cache_dir: Path | str,
    *,
    shard_paths: Sequence[Path | str],
    metadata: HindsightCacheMetadata,
) -> dict[str, Any]:
    """Validate all shards and atomically publish a finalized memmap cache."""

    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        raise FileExistsError(f"cache directory already exists: {cache_dir}")
    if not isinstance(metadata, HindsightCacheMetadata):
        raise TypeError("metadata must be HindsightCacheMetadata")
    paths = sorted((Path(path) for path in shard_paths), key=lambda path: str(path))
    if not paths:
        raise ValueError("shard_paths must not be empty")

    rows: list[tuple[HindsightWindowRecord, Mapping[str, np.ndarray], int]] = []
    seen_ids: set[str] = set()
    for path in paths:
        shard = _load_shard(path, expected_metadata=metadata)
        for index, record in enumerate(shard.records):
            if record.sample_id in seen_ids:
                raise ValueError(f"duplicate sample_id: {record.sample_id}")
            seen_ids.add(record.sample_id)
            rows.append((record, shard.arrays, index))
    rows.sort(key=lambda item: item[0].sample_id)
    records = [row[0] for row in rows]

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            dir=cache_dir.parent,
            prefix=f".{cache_dir.name}.",
            suffix=".tmp",
        )
    )
    try:
        array_manifest: dict[str, dict[str, Any]] = {}
        for name, (dtype, tail_shape) in _ARRAY_SPECS.items():
            filename = f"{name}.npy"
            path = temporary_dir / filename
            destination = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=dtype,
                shape=(len(rows), *tail_shape),
            )
            for output_index, (_, source_arrays, source_index) in enumerate(rows):
                destination[output_index] = source_arrays[name][source_index]
            destination.flush()
            del destination
            array_manifest[name] = {
                "filename": filename,
                "dtype": dtype.name,
                "shape": [len(rows), *tail_shape],
                "sha256": sha256_file(path),
            }

        index_path = temporary_dir / "index.jsonl"
        _write_index(index_path, records)
        index_manifest = {
            "filename": index_path.name,
            "records": len(records),
            "sha256": sha256_file(index_path),
        }
        manifest: dict[str, Any] = {
            "format_version": _FORMAT_VERSION,
            "metadata": metadata.to_dict(),
            "cache_hash": metadata.cache_hash,
            "split_hash": metadata.window_manifest_hash,
            "teacher_hash": _teacher_hash(metadata),
            "camera_names": list(CAMERA_NAMES),
            "num_samples": len(records),
            "index": index_manifest,
            "arrays": array_manifest,
        }
        manifest_path = temporary_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(
                manifest,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        for path in temporary_dir.iterdir():
            path.chmod(0o444)
        _fsync_directory(temporary_dir)
        if cache_dir.exists():
            raise FileExistsError(f"cache directory already exists: {cache_dir}")
        os.replace(temporary_dir, cache_dir)
        _fsync_directory(cache_dir.parent)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return manifest


@dataclass(frozen=True)
class HindsightSample:
    record: HindsightWindowRecord
    codes: torch.Tensor
    relevance: torch.Tensor
    confidence: torch.Tensor
    flow: torch.Tensor
    phrase_embeddings: torch.Tensor


class HindsightCache:
    """Read-only view over a fully published hindsight cache."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        metadata: HindsightCacheMetadata,
        records: Sequence[HindsightWindowRecord],
        arrays: Mapping[str, np.ndarray],
        manifest: Mapping[str, Any],
    ) -> None:
        self.cache_dir = cache_dir
        self.metadata = metadata
        self.records = tuple(records)
        self._arrays = dict(arrays)
        self.manifest = dict(manifest)

    @property
    def codes(self) -> np.ndarray:
        return self._arrays["codes"]

    @classmethod
    def open(
        cls,
        cache_dir: Path | str,
        *,
        expected_metadata: HindsightCacheMetadata | None = None,
        expected_split_hash: str | None = None,
        expected_teacher_hash: str | None = None,
    ) -> HindsightCache:
        cache_dir = Path(cache_dir)
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                f"hindsight cache manifest is missing; cache is not finalized: "
                f"{manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid hindsight cache manifest: {exc}") from exc
        if type(manifest) is not dict or set(manifest) != _MANIFEST_FIELDS:
            raise ValueError("hindsight cache manifest fields are invalid")
        if manifest["format_version"] != _FORMAT_VERSION:
            raise ValueError(f"cache format_version must be {_FORMAT_VERSION}")
        if manifest["camera_names"] != list(CAMERA_NAMES):
            raise ValueError("cache camera order must be canonical main/wrist")
        metadata = HindsightCacheMetadata.from_dict(manifest["metadata"])
        if manifest["cache_hash"] != metadata.cache_hash:
            raise ValueError("cache metadata hash mismatch")
        if manifest["split_hash"] != metadata.window_manifest_hash:
            raise ValueError("cache split hash does not match metadata")
        if manifest["teacher_hash"] != _teacher_hash(metadata):
            raise ValueError("cache teacher hash does not match metadata")
        if expected_metadata is not None and metadata != expected_metadata:
            raise ValueError("cache metadata does not match expected metadata")
        if (
            expected_split_hash is not None
            and manifest["split_hash"] != expected_split_hash
        ):
            raise ValueError("cache split hash does not match expected split hash")
        if (
            expected_teacher_hash is not None
            and manifest["teacher_hash"] != expected_teacher_hash
        ):
            raise ValueError("cache teacher hash does not match expected teacher hash")

        index_description = manifest["index"]
        if (
            type(index_description) is not dict
            or set(index_description) != {"filename", "records", "sha256"}
            or index_description["filename"] != "index.jsonl"
        ):
            raise ValueError("cache index manifest is invalid")
        index_path = cache_dir / index_description["filename"]
        if (
            not index_path.is_file()
            or sha256_file(index_path) != index_description["sha256"]
        ):
            raise ValueError("cache index is missing or corrupted")
        records = _parse_records_jsonl(
            index_path.read_text(encoding="utf-8"),
            require_one_episode=False,
        )
        num_samples = manifest["num_samples"]
        if (
            type(num_samples) is not int
            or num_samples <= 0
            or len(records) != num_samples
            or index_description["records"] != num_samples
        ):
            raise ValueError("cache sample count does not match index")

        array_descriptions = manifest["arrays"]
        if type(array_descriptions) is not dict or set(array_descriptions) != set(
            _ARRAY_SPECS
        ):
            raise ValueError("cache array manifest is invalid")
        arrays: dict[str, np.ndarray] = {}
        try:
            for name, (dtype, tail_shape) in _ARRAY_SPECS.items():
                description = array_descriptions[name]
                expected_description_fields = {
                    "filename",
                    "dtype",
                    "shape",
                    "sha256",
                }
                if (
                    type(description) is not dict
                    or set(description) != expected_description_fields
                    or description["filename"] != f"{name}.npy"
                    or description["dtype"] != dtype.name
                    or description["shape"] != [num_samples, *tail_shape]
                ):
                    raise ValueError(f"cache {name} manifest is invalid")
                path = cache_dir / description["filename"]
                if not path.is_file() or sha256_file(path) != description["sha256"]:
                    raise ValueError(f"cache {name} is missing or corrupted")
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if array.dtype != dtype or array.shape != (
                    num_samples,
                    *tail_shape,
                ):
                    raise ValueError(f"cache {name} has invalid dtype or shape")
                arrays[name] = array
            _validate_and_convert_arrays(
                arrays,
                count=num_samples,
                require_storage_dtypes=True,
            )
        except Exception:
            for array in arrays.values():
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()
            raise
        return cls(
            cache_dir=cache_dir,
            metadata=metadata,
            records=records,
            arrays=arrays,
            manifest=manifest,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> HindsightSample:
        if type(index) is not int:
            raise TypeError("cache index must be an integer")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"cache index out of range: {index}")

        relevance_q = torch.from_numpy(
            np.array(self._arrays["relevance_q"][index], copy=True)
        ).to(torch.float32)
        relevance_scale = torch.from_numpy(
            np.array(self._arrays["relevance_scale"][index], copy=True)
        ).to(torch.float32)
        relevance = torch.clamp(
            relevance_q / relevance_scale.unsqueeze(-1),
            min=0,
        )
        relevance = relevance / relevance.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(relevance.dtype).tiny
        )
        return HindsightSample(
            record=self.records[index],
            codes=torch.from_numpy(np.array(self._arrays["codes"][index], copy=True)),
            relevance=relevance,
            confidence=torch.from_numpy(
                np.array(self._arrays["confidence"][index], copy=True)
            ),
            flow=torch.from_numpy(np.array(self._arrays["flow"][index], copy=True)),
            phrase_embeddings=torch.from_numpy(
                np.array(self._arrays["phrase_embeddings"][index], copy=True)
            ),
        )

    def close(self) -> None:
        arrays = getattr(self, "_arrays", {})
        for array in arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        arrays.clear()

    def __enter__(self) -> HindsightCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
