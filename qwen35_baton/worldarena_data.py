"""Strict WorldArena source validation and deterministic RGB data adapters."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from qwen35_baton.config import exact_half_even_round


_FRAME_COUNT = 121
_IMAGE_SIZE = 256
_SOURCE_REPOSITORY = "worldarena2026-robotwin-data"
_SPLITS = frozenset(("train", "validation"))
_CACHE_REQUIRED_RECORD_FIELDS = frozenset(
    (
        "episode_id",
        "hdf5_path",
        "source_dataset_root",
        "source_video_path",
        "source_video_relative_path",
        "source_video_sha256",
        "split",
        "task",
        "instruction",
        "frame_count",
        "source_frame_count",
    )
)
_CACHE_OPTIONAL_FIELD_PAIRS = (
    ("actions_16d_path", "actions_16d_sha256"),
    ("intrinsic_path", "intrinsic_sha256"),
    ("extrinsic_path", "extrinsic_sha256"),
)


@dataclass(frozen=True)
class WorldArenaRecord:
    """One validated episode from the WorldArena training-only release."""

    episode_id: str
    task_name: str
    instruction: str
    video_path: Path
    actions_16d_path: Path | None
    intrinsic_path: Path | None
    extrinsic_path: Path | None
    dataset_root: Path | None = None
    source_video_relative_path: str | None = None


def _is_official_path(path: Path) -> bool:
    return any(part.casefold() == "official_episodes" for part in path.parts)


def _is_forbidden_source_root(path: Path) -> bool:
    forbidden = {
        "official_episodes",
        "official_validation",
        "official_val",
        "official_test",
        "worldarena_validation",
        "worldarena_test",
        "validation",
        "test",
    }
    for part in path.parts:
        normalized = part.casefold().replace("-", "_")
        if normalized in forbidden:
            return True
        official_validation_or_test = normalized.startswith("official_") and (
            "validation" in normalized
            or "test" in normalized
            or "episode" in normalized
        )
        worldarena_validation_or_test = normalized.startswith("worldarena_") and (
            "validation" in normalized or "test" in normalized
        )
        if official_validation_or_test or worldarena_validation_or_test:
            return True
    return False


def _validate_training_video_provenance(
    *,
    dataset_root: Any,
    video_path: Any,
    relative_path: Any,
    episode_id: str,
) -> tuple[Path, str]:
    """Prove the video is the canonical training asset under its source root."""

    if not isinstance(dataset_root, (str, os.PathLike)):
        raise ValueError("training provenance requires source_dataset_root")
    raw_root = Path(dataset_root).expanduser()
    root = raw_root.resolve()
    if not raw_root.is_absolute() or raw_root != root:
        raise ValueError("source_dataset_root must be an absolute canonical path")
    if root.name != _SOURCE_REPOSITORY or _is_forbidden_source_root(root):
        raise ValueError("source_dataset_root is not the canonical training repository")
    expected_relative = f"episodes/{episode_id}/video.mp4"
    if relative_path != expected_relative:
        raise ValueError("source video provenance must be exactly " + expected_relative)
    if not isinstance(video_path, (str, os.PathLike)):
        raise ValueError("training provenance requires source_video_path")
    expected_video = root / "episodes" / episode_id / "video.mp4"
    resolved_expected_video = expected_video.resolve()
    if resolved_expected_video != expected_video:
        raise ValueError(
            "canonical training provenance must not traverse source symlinks"
        )
    raw_video = Path(video_path).expanduser()
    resolved_video = raw_video.resolve()
    if (
        not raw_video.is_absolute()
        or raw_video != expected_video
        or raw_video != resolved_video
    ):
        raise ValueError(
            "source_video_path does not match canonical training provenance"
        )
    if _is_forbidden_source_root(resolved_video):
        raise ValueError("source_video_path resolves to a forbidden dataset area")
    try:
        resolved_video.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "source video provenance escapes source_dataset_root"
        ) from error
    return root, expected_relative


def _require_under_root(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"source path is outside dataset root: {path}") from error
    return resolved


def localize_source_path(
    source_path: str | os.PathLike[str],
    *,
    dataset_root: str | os.PathLike[str],
    required: bool = False,
) -> Path:
    """Resolve publisher paths under ``dataset_root`` and fail closed.

    Absolute paths are accepted only for the publisher's stale
    ``.../training_data/...`` prefix. The suffix after ``training_data`` is
    re-rooted locally; arbitrary host-absolute paths are never trusted.
    """

    if not isinstance(source_path, (str, os.PathLike)):
        raise TypeError("source path must be a string or path-like value")
    raw = os.fspath(source_path)
    if not raw or not raw.strip():
        raise ValueError("source path must be nonblank")
    if not isinstance(required, bool):
        raise TypeError("required must be a boolean")

    root = Path(dataset_root).expanduser().resolve()
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        components = candidate.parts
        training_positions = [
            index
            for index, component in enumerate(components)
            if component == "training_data"
        ]
        if not training_positions:
            raise ValueError(
                "absolute source paths must contain the publisher training_data prefix"
            )
        suffix = components[training_positions[-1] + 1 :]
        if not suffix:
            raise ValueError("training_data source path must contain a relative suffix")
        candidate = root.joinpath(*suffix)
    else:
        candidate = root / candidate

    resolved = _require_under_root(candidate, root)
    if _is_official_path(resolved):
        raise ValueError(f"official episode paths are forbidden: {source_path}")
    if required and not resolved.is_file():
        raise FileNotFoundError(f"required source path does not exist: {resolved}")
    return resolved


def _optional_localized_path(
    value: Any, *, dataset_root: Path, field: str
) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a path string when present")
    if not value.strip():
        raise ValueError(f"{field} must be nonblank when present")
    return localize_source_path(value, dataset_root=dataset_root, required=True)


def load_worldarena_source_manifest(
    path: str | os.PathLike[str], dataset_root: str | os.PathLike[str]
) -> tuple[WorldArenaRecord, ...]:
    """Load the publisher JSONL while rejecting leakage and stale paths."""

    manifest_path = Path(path).expanduser().resolve()
    root = Path(dataset_root).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"WorldArena source manifest does not exist: {manifest_path}"
        )
    if not root.is_dir():
        raise FileNotFoundError(f"WorldArena dataset root does not exist: {root}")

    records: list[WorldArenaRecord] = []
    episode_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"source manifest line {line_number} is blank")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"source manifest line {line_number} is invalid JSON"
                ) from error
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"source manifest line {line_number} must be a JSON object"
                )

            video_value = payload.get("video")
            if not isinstance(video_value, str) or not video_value.strip():
                raise ValueError(
                    f"source manifest line {line_number} requires nonblank video"
                )
            prompt = payload.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"source manifest line {line_number} requires nonblank prompt"
                )
            video_path = localize_source_path(
                video_value, dataset_root=root, required=True
            )
            episode_id = video_path.parent.name
            if not episode_id:
                raise ValueError(
                    f"source manifest line {line_number} has no episode directory"
                )
            if episode_id in episode_ids:
                raise ValueError(f"duplicate WorldArena episode_id: {episode_id}")
            task_name = episode_id.split("__episode", 1)[0]
            if not task_name:
                raise ValueError(f"episode {episode_id!r} has no task prefix")
            source_video_relative_path = video_path.relative_to(root).as_posix()
            _validate_training_video_provenance(
                dataset_root=root,
                video_path=video_path,
                relative_path=source_video_relative_path,
                episode_id=episode_id,
            )

            records.append(
                WorldArenaRecord(
                    episode_id=episode_id,
                    task_name=task_name,
                    instruction=prompt,
                    video_path=video_path,
                    actions_16d_path=_optional_localized_path(
                        payload.get("action_path"),
                        dataset_root=root,
                        field="action_path",
                    ),
                    intrinsic_path=_optional_localized_path(
                        payload.get("intrinsic_path"),
                        dataset_root=root,
                        field="intrinsic_path",
                    ),
                    extrinsic_path=_optional_localized_path(
                        payload.get("extrinsic_path"),
                        dataset_root=root,
                        field="extrinsic_path",
                    ),
                    dataset_root=root,
                    source_video_relative_path=source_video_relative_path,
                )
            )
            episode_ids.add(episode_id)
    if not records:
        raise ValueError("WorldArena source manifest must contain at least one record")
    return tuple(records)


def future_frame_indices(
    current_index: int, frame_count: int = _FRAME_COUNT
) -> tuple[int, int, int, int]:
    """Return four normalized, strictly future frame indices."""

    if (
        type(frame_count) is not int
        or type(current_index) is not int
        or frame_count < 5
        or current_index < 0
        or current_index > frame_count - 5
    ):
        raise ValueError("current index must leave four strictly future frames")
    last = frame_count - 1
    future = tuple(
        current_index + exact_half_even_round((step + 1) * (last - current_index), 4)
        for step in range(4)
    )
    if len(set(future)) != 4 or tuple(sorted(future)) != future:
        raise ValueError("normalized future indices must be unique and ordered")
    return future  # type: ignore[return-value]


def canonical_source_frame_indices(
    source_frame_count: int,
) -> tuple[int, ...]:
    """Map the canonical 121-frame timeline across an entire source MP4."""

    if type(source_frame_count) is not int or source_frame_count < 1:
        raise ValueError("source video must contain at least one decodable frame")
    last = source_frame_count - 1
    return tuple(
        exact_half_even_round(index * last, _FRAME_COUNT - 1)
        for index in range(_FRAME_COUNT)
    )


def _validate_seed(seed: int) -> None:
    if type(seed) is not int:
        raise ValueError("seed must be an integer")


def _validate_split(split: str) -> None:
    if not isinstance(split, str) or split not in _SPLITS:
        raise ValueError("split must be 'train' or 'validation'")


def _validate_episode_id(episode_id: str) -> None:
    if (
        not episode_id
        or episode_id in {".", ".."}
        or Path(episode_id).name != episode_id
        or "/" in episode_id
        or "\\" in episode_id
        or "\x00" in episode_id
    ):
        raise ValueError("episode_id must be a single nonblank directory name")


def _source_indices(
    *, seed: int, epoch: int, episode_id: str, split: str, frame_count: int
) -> tuple[int, int, int, int, int]:
    if split == "train":
        payload = f"{seed}:{epoch}:{episode_id}"
    else:
        payload = f"{seed}:{episode_id}"
    current = int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
    ) % (frame_count - 4)
    return (current, *future_frame_indices(current, frame_count))


def _decodable_frame_count(path: Path) -> int:
    """Probe actual decodable frames without trusting container metadata."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"failed to open WorldArena video: {path}")
    count = 0
    try:
        while capture.grab():
            count += 1
    finally:
        capture.release()
    if count < 1:
        raise ValueError(f"WorldArena video has no decodable frames: {path}")
    return count


def _record_metadata(
    record: WorldArenaRecord, *, source_frame_count: int
) -> dict[str, Any]:
    dataset_root, relative_path = _validate_training_video_provenance(
        dataset_root=record.dataset_root,
        video_path=record.video_path,
        relative_path=record.source_video_relative_path,
        episode_id=record.episode_id,
    )
    metadata: dict[str, Any] = {
        "source_dataset_root": str(dataset_root),
        "source_video_path": str(record.video_path),
        "source_video_relative_path": relative_path,
        "task": record.task_name,
        "source_frame_count": source_frame_count,
    }
    for key, value in (
        ("actions_16d_path", record.actions_16d_path),
        ("intrinsic_path", record.intrinsic_path),
        ("extrinsic_path", record.extrinsic_path),
    ):
        if value is not None:
            metadata[key] = str(value)
    return metadata


def _rgb_sample(
    rgb: np.ndarray,
    *,
    instruction: str,
    episode_id: str,
    source_indices: tuple[int, int, int, int, int],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if rgb.shape != (5, _IMAGE_SIZE, _IMAGE_SIZE, 3) or rgb.dtype != np.uint8:
        raise ValueError("selected RGB must be uint8 [5,256,256,3]")
    chw = torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2)))
    return {
        "current_images": chw[0].unsqueeze(0),
        "future_images": chw[1:].unsqueeze(0),
        "camera_names": ("head",),
        "instruction": instruction,
        "suite": "worldarena",
        "episode_key": episode_id,
        "source_indices": source_indices,
        "metadata": dict(metadata),
    }


class WorldArenaMP4Dataset(Dataset[dict[str, Any]]):
    """Correctness dataset that reads the five selected frames from MP4."""

    def __init__(
        self,
        records: Sequence[WorldArenaRecord],
        *,
        seed: int,
        split: str = "train",
    ) -> None:
        _validate_seed(seed)
        _validate_split(split)
        self.records = _validate_records(records)
        self.seed = seed
        self.split = split
        self._shared_epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("dataset epoch must be a non-negative integer")
        self._shared_epoch.fill_(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        indices = _source_indices(
            seed=self.seed,
            epoch=int(self._shared_epoch.item()),
            episode_id=record.episode_id,
            split=self.split,
            frame_count=_FRAME_COUNT,
        )
        source_frame_count = _decodable_frame_count(record.video_path)
        capture = cv2.VideoCapture(str(record.video_path))
        if not capture.isOpened():
            raise ValueError(f"failed to open WorldArena video: {record.video_path}")
        try:
            canonical_to_source = canonical_source_frame_indices(source_frame_count)
            frames: list[np.ndarray] = []
            for source_index in (canonical_to_source[index] for index in indices):
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, source_index):
                    raise ValueError(
                        f"failed to seek frame {source_index}: {record.video_path}"
                    )
                ok, bgr = capture.read()
                if not ok or bgr is None:
                    raise ValueError(
                        f"failed to decode frame {source_index}: {record.video_path}"
                    )
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(
                    cv2.resize(
                        rgb,
                        (_IMAGE_SIZE, _IMAGE_SIZE),
                        interpolation=cv2.INTER_AREA,
                    )
                )
        finally:
            capture.release()
        return _rgb_sample(
            np.stack(frames).astype(np.uint8, copy=False),
            instruction=record.instruction,
            episode_id=record.episode_id,
            source_indices=indices,
            metadata=_record_metadata(record, source_frame_count=source_frame_count),
        )


def _resolve_hdf5_path(value: Any, cache_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("cached record requires nonblank hdf5_path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("cached hdf5_path must be relative to the manifest")
    resolved = _require_under_root(cache_root / relative, cache_root)
    if _is_official_path(resolved):
        raise ValueError("official episode paths are forbidden in cache manifests")
    if not resolved.is_file():
        raise FileNotFoundError(f"cached HDF5 file does not exist: {resolved}")
    return resolved


def _cached_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    source_frame_count = payload.get("source_frame_count")
    if type(source_frame_count) is not int or source_frame_count < 1:
        raise ValueError("cached record requires positive source_frame_count")
    metadata: dict[str, Any] = {
        "task": payload["task"],
        "source_frame_count": source_frame_count,
    }
    for key in (
        "source_video_path",
        "source_dataset_root",
        "source_video_relative_path",
        "actions_16d_path",
        "intrinsic_path",
        "extrinsic_path",
        "source_video_sha256",
        "actions_16d_sha256",
        "intrinsic_sha256",
        "extrinsic_sha256",
    ):
        value = payload.get(key)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ValueError(f"cached record field {key} must be nonblank")
            if key.endswith("_path") and _is_official_path(Path(value)):
                raise ValueError("official paths are forbidden in cache metadata")
            metadata[key] = value
    return metadata


def _validate_sha256(value: Any, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"cached record field {field} must be lowercase SHA-256")


def _validated_rgb_dataset(
    handle: h5py.File,
    *,
    shard: Path,
    split: str,
    episode_id: str,
) -> h5py.Dataset:
    prefix = f"WorldArena {split} shard {episode_id!r} ({shard})"
    if "rgb" not in handle:
        raise ValueError(f"{prefix} has no rgb dataset")
    rgb = handle["rgb"]
    if not isinstance(rgb, h5py.Dataset):
        raise ValueError(f"{prefix} rgb must be an HDF5 dataset")
    if rgb.shape != (_FRAME_COUNT, _IMAGE_SIZE, _IMAGE_SIZE, 3):
        raise ValueError(f"{prefix} rgb must have shape [121,256,256,3]")
    if rgb.dtype != np.dtype(np.uint8):
        raise ValueError(f"{prefix} rgb must have uint8 dtype")
    if rgb.chunks != (1, _IMAGE_SIZE, _IMAGE_SIZE, 3):
        raise ValueError(f"{prefix} rgb must be chunked one frame at a time")
    if rgb.compression != "lzf":
        raise ValueError(f"{prefix} rgb must use LZF compression")
    return rgb


@dataclass(frozen=True)
class WorldArenaCacheAudit:
    """Metadata-only schema audit counts for a complete published cache."""

    record_count: int
    train_count: int
    validation_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "record_count": self.record_count,
            "train_count": self.train_count,
            "validation_count": self.validation_count,
        }


class WorldArenaHDF5Dataset(Dataset[dict[str, Any]]):
    """Epoch-deterministic WorldArena cache reader with one head camera."""

    def __init__(
        self,
        manifest_path: str | os.PathLike[str],
        *,
        seed: int,
        split: str = "train",
    ) -> None:
        _validate_seed(seed)
        _validate_split(split)
        path = Path(manifest_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"WorldArena cache manifest does not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("WorldArena cache manifest is invalid JSON") from error
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("records"), list
        ):
            raise ValueError("WorldArena cache manifest requires a records list")
        if set(payload) != {"version", "source_repository", "records"}:
            raise ValueError(
                "WorldArena cache manifest has an invalid top-level schema"
            )
        if payload["version"] != 1:
            raise ValueError("WorldArena cache manifest version must be 1")
        if payload["source_repository"] != _SOURCE_REPOSITORY:
            raise ValueError(
                "WorldArena cache manifest source_repository is not training-only"
            )
        if not payload["records"]:
            raise ValueError("WorldArena cache manifest records must be nonempty")

        cache_root = path.parent.resolve()
        parsed: list[dict[str, Any]] = []
        seen: set[str] = set()
        hdf5_paths: set[Path] = set()
        for item in payload["records"]:
            if not isinstance(item, Mapping):
                raise ValueError("WorldArena cached records must be JSON objects")
            allowed_fields = _CACHE_REQUIRED_RECORD_FIELDS.union(
                field for pair in _CACHE_OPTIONAL_FIELD_PAIRS for field in pair
            )
            item_fields = set(item)
            if not _CACHE_REQUIRED_RECORD_FIELDS.issubset(
                item_fields
            ) or not item_fields.issubset(allowed_fields):
                raise ValueError("WorldArena cached record has an invalid schema")
            for path_field, hash_field in _CACHE_OPTIONAL_FIELD_PAIRS:
                if (path_field in item) != (hash_field in item):
                    raise ValueError(
                        f"cached record must pair {path_field} with {hash_field}"
                    )
            episode_id = item.get("episode_id")
            instruction = item.get("instruction")
            task = item.get("task")
            item_split = item.get("split")
            if not isinstance(episode_id, str) or not episode_id:
                raise ValueError("cached record requires nonblank episode_id")
            _validate_episode_id(episode_id)
            if episode_id in seen:
                raise ValueError(f"duplicate cached episode_id: {episode_id}")
            seen.add(episode_id)
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("cached record requires nonblank instruction")
            if not isinstance(task, str) or not task:
                raise ValueError("cached record requires nonblank task")
            expected_task = episode_id.split("__episode", 1)[0]
            if task != expected_task:
                raise ValueError("cached record task does not match episode_id")
            _validate_split(item_split)
            if item.get("frame_count") != _FRAME_COUNT:
                raise ValueError("cached records must declare exactly 121 frames")
            source_video_path = item["source_video_path"]
            if not isinstance(source_video_path, str) or not source_video_path:
                raise ValueError("cached record requires nonblank source_video_path")
            if _is_official_path(Path(source_video_path)):
                raise ValueError("official paths are forbidden in cache metadata")
            if Path(source_video_path).parent.name != episode_id:
                raise ValueError("source_video_path does not match episode_id")
            _validate_training_video_provenance(
                dataset_root=item["source_dataset_root"],
                video_path=source_video_path,
                relative_path=item["source_video_relative_path"],
                episode_id=episode_id,
            )
            _validate_sha256(item["source_video_sha256"], field="source_video_sha256")
            for _, hash_field in _CACHE_OPTIONAL_FIELD_PAIRS:
                if hash_field in item:
                    _validate_sha256(item[hash_field], field=hash_field)
            expected_hdf5 = f"episodes/{episode_id}.h5"
            if item.get("hdf5_path") != expected_hdf5:
                raise ValueError("cached hdf5_path does not match episode_id")
            parsed_record = dict(item)
            parsed_record["hdf5_path"] = _resolve_hdf5_path(
                item.get("hdf5_path"), cache_root
            )
            if parsed_record["hdf5_path"] in hdf5_paths:
                raise ValueError("duplicate cached hdf5_path")
            hdf5_paths.add(parsed_record["hdf5_path"])
            parsed_record["metadata"] = _cached_metadata(item)
            if item_split == split:
                parsed.append(parsed_record)
        if not parsed:
            raise ValueError(f"WorldArena cache has no records for split {split!r}")
        self.manifest_path = path
        self.records = tuple(parsed)
        self.seed = seed
        self.split = split
        self._shared_epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("dataset epoch must be a non-negative integer")
        self._shared_epoch.fill_(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        indices = _source_indices(
            seed=self.seed,
            epoch=int(self._shared_epoch.item()),
            episode_id=record["episode_id"],
            split=self.split,
            frame_count=record["frame_count"],
        )
        try:
            handle_context = h5py.File(record["hdf5_path"], "r")
        except OSError as error:
            raise ValueError(
                f"WorldArena {self.split} shard {record['episode_id']!r} "
                f"is not valid HDF5: {record['hdf5_path']}"
            ) from error
        with handle_context as handle:
            rgb_dataset = _validated_rgb_dataset(
                handle,
                shard=record["hdf5_path"],
                split=self.split,
                episode_id=record["episode_id"],
            )
            rgb = np.asarray(rgb_dataset[list(indices)], dtype=np.uint8)
        return _rgb_sample(
            rgb,
            instruction=record["instruction"],
            episode_id=record["episode_id"],
            source_indices=indices,
            metadata=record["metadata"],
        )


def audit_worldarena_hdf5_cache(
    manifest_path: str | os.PathLike[str],
) -> WorldArenaCacheAudit:
    """Inspect every referenced shard schema without reading RGB frame chunks."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"WorldArena cache manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("WorldArena cache manifest is invalid JSON") from error
    records = payload.get("records") if isinstance(payload, Mapping) else None
    if not isinstance(records, list) or not records:
        # Reuse the strict reader for the canonical schema-level diagnostic.
        WorldArenaHDF5Dataset(path, seed=0, split="train")
        raise AssertionError("strict WorldArena manifest validation is unreachable")
    declared_splits = {
        record.get("split") for record in records if isinstance(record, Mapping)
    }
    if not declared_splits or not declared_splits.issubset(_SPLITS):
        WorldArenaHDF5Dataset(path, seed=0, split="train")
        raise AssertionError("strict WorldArena split validation is unreachable")

    counts = Counter()
    audited = 0
    for split in ("train", "validation"):
        if split not in declared_splits:
            continue
        dataset = WorldArenaHDF5Dataset(path, seed=0, split=split)
        counts[split] = len(dataset.records)
        for record in dataset.records:
            shard = record["hdf5_path"]
            try:
                handle_context = h5py.File(shard, "r")
            except OSError as error:
                raise ValueError(
                    f"WorldArena {split} shard {record['episode_id']!r} "
                    f"is not valid HDF5: {shard}"
                ) from error
            with handle_context as handle:
                _validated_rgb_dataset(
                    handle,
                    shard=shard,
                    split=split,
                    episode_id=record["episode_id"],
                )
            audited += 1
    if audited != len(records):
        raise ValueError("WorldArena cache audit did not cover every manifest record")
    return WorldArenaCacheAudit(
        record_count=audited,
        train_count=counts["train"],
        validation_count=counts["validation"],
    )


def _validate_records(
    records: Sequence[WorldArenaRecord],
) -> tuple[WorldArenaRecord, ...]:
    if isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence of WorldArenaRecord values")
    validated = tuple(records)
    if not validated:
        raise ValueError("at least one WorldArena record is required")
    seen: set[str] = set()
    for record in validated:
        if not isinstance(record, WorldArenaRecord):
            raise TypeError("records must contain only WorldArenaRecord values")
        for name, value in (
            ("episode_id", record.episode_id),
            ("task_name", record.task_name),
            ("instruction", record.instruction),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"WorldArena record {name} must be nonblank")
        if record.episode_id in seen:
            raise ValueError(f"duplicate WorldArena episode_id: {record.episode_id}")
        _validate_episode_id(record.episode_id)
        if record.video_path.parent.name != record.episode_id:
            raise ValueError("WorldArena record episode_id must match video parent")
        expected_task = record.episode_id.split("__episode", 1)[0]
        if record.task_name != expected_task:
            raise ValueError("WorldArena record task_name must match episode_id")
        _validate_training_video_provenance(
            dataset_root=record.dataset_root,
            video_path=record.video_path,
            relative_path=record.source_video_relative_path,
            episode_id=record.episode_id,
        )
        seen.add(record.episode_id)
        if _is_official_path(record.video_path):
            raise ValueError("official episode paths are forbidden")
        if not record.video_path.is_file():
            raise FileNotFoundError(
                f"WorldArena video does not exist: {record.video_path}"
            )
        for optional in (
            record.actions_16d_path,
            record.intrinsic_path,
            record.extrinsic_path,
        ):
            if optional is not None:
                if _is_official_path(optional):
                    raise ValueError("official episode paths are forbidden")
                if not optional.is_file():
                    raise FileNotFoundError(
                        f"WorldArena optional metadata does not exist: {optional}"
                    )
    return validated


def _decode_video(record: WorldArenaRecord) -> tuple[np.ndarray, int]:
    capture = cv2.VideoCapture(str(record.video_path))
    if not capture.isOpened():
        raise ValueError(f"failed to open WorldArena video: {record.video_path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(
                cv2.resize(
                    rgb,
                    (_IMAGE_SIZE, _IMAGE_SIZE),
                    interpolation=cv2.INTER_AREA,
                )
            )
    finally:
        capture.release()
    source_frame_count = len(frames)
    if source_frame_count < 1:
        raise ValueError(
            f"WorldArena video has no decodable frames: {record.video_path}"
        )
    canonical = canonical_source_frame_indices(source_frame_count)
    return (
        np.stack([frames[index] for index in canonical]).astype(np.uint8, copy=False),
        source_frame_count,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _source_manifest_record(
    record: WorldArenaRecord,
    *,
    hdf5_path: str,
    split: str,
    source_frame_count: int,
) -> dict[str, Any]:
    dataset_root, relative_path = _validate_training_video_provenance(
        dataset_root=record.dataset_root,
        video_path=record.video_path,
        relative_path=record.source_video_relative_path,
        episode_id=record.episode_id,
    )
    payload: dict[str, Any] = {
        "episode_id": record.episode_id,
        "hdf5_path": hdf5_path,
        "source_dataset_root": str(dataset_root),
        "source_video_path": str(record.video_path),
        "source_video_relative_path": relative_path,
        "source_video_sha256": _sha256_file(record.video_path),
        "split": split,
        "task": record.task_name,
        "instruction": record.instruction,
        "frame_count": _FRAME_COUNT,
        "source_frame_count": source_frame_count,
    }
    for path_key, hash_key, value in (
        ("actions_16d_path", "actions_16d_sha256", record.actions_16d_path),
        ("intrinsic_path", "intrinsic_sha256", record.intrinsic_path),
        ("extrinsic_path", "extrinsic_sha256", record.extrinsic_path),
    ):
        if value is not None:
            payload[path_key] = str(value)
            payload[hash_key] = _sha256_file(value)
    return payload


def _validation_split(seed: int, episode_id: str, fraction: float) -> str:
    value = (
        int.from_bytes(
            hashlib.sha256(f"{seed}:{episode_id}".encode("utf-8")).digest()[:8], "big"
        )
        / 2**64
    )
    return "validation" if value < fraction else "train"


def predecode_worldarena(
    records: Sequence[WorldArenaRecord],
    *,
    output_root: str | os.PathLike[str],
    seed: int,
    validation_fraction: float = 0.1,
) -> Path:
    """Decode each episode to LZF HDF5 and atomically publish cache metadata."""

    _validate_seed(seed)
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not 0.0 <= float(validation_fraction) <= 1.0
    ):
        raise ValueError("validation_fraction must be between 0 and 1")
    fraction = float(validation_fraction)
    validated = sorted(_validate_records(records), key=lambda item: item.episode_id)
    output = Path(output_root).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise ValueError("output_root is nonempty or is not a directory")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    )
    try:
        episodes_root = staging / "episodes"
        episodes_root.mkdir()
        manifest_records: list[dict[str, Any]] = []
        for record in validated:
            final_path = _require_under_root(
                episodes_root / f"{record.episode_id}.h5", episodes_root
            )
            temporary = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
            try:
                frames, source_frame_count = _decode_video(record)
                with h5py.File(temporary, "w") as handle:
                    handle.create_dataset(
                        "rgb",
                        data=frames,
                        dtype=np.uint8,
                        chunks=(1, _IMAGE_SIZE, _IMAGE_SIZE, 3),
                        compression="lzf",
                    )
                    handle.flush()
                _fsync_file(temporary)
                os.replace(temporary, final_path)
            finally:
                temporary.unlink(missing_ok=True)
            relative_hdf5 = final_path.relative_to(staging).as_posix()
            manifest_records.append(
                _source_manifest_record(
                    record,
                    hdf5_path=relative_hdf5,
                    split=_validation_split(seed, record.episode_id, fraction),
                    source_frame_count=source_frame_count,
                )
            )
        _fsync_directory(episodes_root)

        staged_manifest = staging / "manifest.json"
        _atomic_json(
            staged_manifest,
            {
                "version": 1,
                "source_repository": _SOURCE_REPOSITORY,
                "records": manifest_records,
            },
        )
        split_counts = Counter(record["split"] for record in manifest_records)
        task_counts = Counter(record["task"] for record in manifest_records)
        _atomic_json(
            staging / "stats.json",
            {
                "train_count": split_counts["train"],
                "validation_count": split_counts["validation"],
                "task_counts": dict(sorted(task_counts.items())),
                "image_size": [_IMAGE_SIZE, _IMAGE_SIZE],
                "frame_count": _FRAME_COUNT,
                "source_frame_counts": {
                    record["episode_id"]: record["source_frame_count"]
                    for record in manifest_records
                },
                "seed": seed,
                "source_repository": _SOURCE_REPOSITORY,
                "manifest_sha256": _sha256_file(staged_manifest),
            },
        )
        _fsync_directory(staging)
        os.replace(staging, output)
        _fsync_directory(output.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output / "manifest.json"


__all__ = (
    "WorldArenaHDF5Dataset",
    "WorldArenaMP4Dataset",
    "WorldArenaRecord",
    "canonical_source_frame_indices",
    "future_frame_indices",
    "load_worldarena_source_manifest",
    "localize_source_path",
    "predecode_worldarena",
)
