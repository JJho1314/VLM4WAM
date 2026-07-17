#!/usr/bin/env python3
"""Convert fixed-contract LIBERO FastWAM episodes into immutable HDF5 shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import av
import h5py
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

from ge_act.data import libero_fastwam_hdf5_schema as schema


MAIN_CAMERA = "observation.images.image"
WRIST_CAMERA = "observation.images.wrist_image"
CAMERA_PATHS = {"main": MAIN_CAMERA, "wrist": WRIST_CAMERA}
IMAGE_SIZE = (256, 256)


@dataclass(frozen=True)
class SourceEpisode:
    """All immutable source paths and metadata needed to convert one episode."""

    key: str
    source_root: Path
    domain: str
    episode_index: int
    length: int
    caption: str
    parquet_path: Path
    main_video_path: Path
    wrist_video_path: Path
    main_cache_path: Path | None
    wrist_cache_path: Path | None


def _load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON metadata {path}: {error}") from error


def _load_jsonl(path: Path) -> list[Any]:
    records: list[Any] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSONL metadata {path}:{line_number}: {error}"
                    ) from error
    except OSError as error:
        raise ValueError(f"cannot read JSONL metadata {path}: {error}") from error
    return records


def _aligned_source_specs(
    data_roots: Sequence[str | Path], domains: Sequence[str]
) -> list[tuple[Path, str]]:
    roots = [Path(root).expanduser().resolve() for root in data_roots]
    domain_list = list(domains)
    if len(roots) == 1 and len(domain_list) > 1:
        roots *= len(domain_list)
    if not roots or not domain_list or len(roots) != len(domain_list):
        raise ValueError("data_roots and domains must be non-empty and aligned")

    result: list[tuple[Path, str]] = []
    seen: set[tuple[Path, str]] = set()
    for root, domain in zip(roots, domain_list):
        if type(domain) is not str or not domain:
            raise ValueError("domains must contain non-empty strings")
        pair = (root, domain)
        if pair not in seen:
            seen.add(pair)
            result.append(pair)
    return result


def _cache_path_for_video(
    video_path: Path, source_root: Path, predecoded_root: Path
) -> Path:
    try:
        relative = video_path.relative_to(source_root)
    except ValueError as error:
        raise ValueError(
            f"video path is outside source root: video={video_path}, "
            f"source_root={source_root}"
        ) from error
    return (predecoded_root / relative).with_suffix(".npy").resolve()


def discover_source_episodes(
    data_roots: Sequence[str | Path],
    domains: Sequence[str],
    *,
    predecoded_root: str | Path | None = None,
    max_episodes: int | None = None,
) -> list[SourceEpisode]:
    """Discover unique episodes in caller domain order and numeric episode order."""
    if max_episodes is not None and (type(max_episodes) is not int or max_episodes < 1):
        raise ValueError("max_episodes must be a positive integer or None")
    cache_root = (
        None
        if predecoded_root is None
        else Path(predecoded_root).expanduser().resolve()
    )
    discovered: list[SourceEpisode] = []
    owners: dict[str, Path] = {}

    for source_root, domain in _aligned_source_specs(data_roots, domains):
        domain_root = source_root / domain
        meta_root = domain_root / "meta"
        info = _load_json(meta_root / "info.json")
        if type(info) is not dict:
            raise ValueError(f"info metadata must be a dict: {meta_root / 'info.json'}")
        chunks_size = info.get("chunks_size")
        if type(chunks_size) is not int or chunks_size < 1:
            raise ValueError(
                f"chunks_size must be a positive integer: {meta_root / 'info.json'}"
            )

        task_texts: set[str] = set()
        for item in _load_jsonl(meta_root / "tasks.jsonl"):
            if type(item) is not dict:
                raise ValueError(f"task record must be a dict: domain={domain}")
            task = item.get("task")
            if type(task) is not str or not task:
                raise ValueError(
                    f"invalid task metadata: domain={domain}, record={item!r}"
                )
            if task in task_texts:
                raise ValueError(f"duplicate task text: domain={domain}, task={task!r}")
            task_texts.add(task)

        raw_episodes = _load_jsonl(meta_root / "episodes.jsonl")
        try:
            raw_episodes.sort(key=lambda item: item["episode_index"])
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"invalid episode metadata in {meta_root / 'episodes.jsonl'}"
            ) from error

        for item in raw_episodes:
            if type(item) is not dict:
                raise ValueError(f"episode record must be a dict: domain={domain}")
            episode_index = item.get("episode_index")
            length = item.get("length")
            tasks = item.get("tasks")
            if type(episode_index) is not int or episode_index < 0:
                raise ValueError(
                    f"invalid episode_index: domain={domain}, record={item!r}"
                )
            if type(length) is not int or length < 2:
                raise ValueError(
                    f"invalid episode length: domain={domain}, "
                    f"episode={episode_index}, length={length!r}"
                )
            if type(tasks) is not list or len(tasks) != 1:
                raise ValueError(
                    f"episode must have exactly one task: domain={domain}, "
                    f"episode={episode_index}, tasks={tasks!r}"
                )
            task_text = tasks[0]
            if type(task_text) is not str or not task_text:
                raise ValueError(
                    f"episode task must be non-empty task text: domain={domain}, "
                    f"episode={episode_index}, task={task_text!r}"
                )
            if task_text not in task_texts:
                raise ValueError(
                    f"unknown task text: domain={domain}, episode={episode_index}, "
                    f"task={task_text!r}"
                )

            key = f"{domain}:{episode_index:06d}"
            previous_root = owners.get(key)
            if previous_root is not None:
                raise ValueError(
                    f"duplicate final episode key {key} from conflicting roots "
                    f"{previous_root} and {source_root}"
                )
            owners[key] = source_root

            chunk_index = episode_index // chunks_size
            parquet_path = (
                domain_root
                / "data"
                / f"chunk-{chunk_index:03d}"
                / f"episode_{episode_index:06d}.parquet"
            ).resolve()
            if not parquet_path.is_file():
                raise FileNotFoundError(
                    f"missing parquet: domain={domain}, episode={episode_index}, "
                    f"path={parquet_path}"
                )

            videos: dict[str, Path] = {}
            caches: dict[str, Path | None] = {}
            for camera_role, camera_name in CAMERA_PATHS.items():
                video_path = (
                    domain_root
                    / "videos"
                    / f"chunk-{chunk_index:03d}"
                    / camera_name
                    / f"episode_{episode_index:06d}.mp4"
                ).resolve()
                videos[camera_role] = video_path
                if cache_root is None:
                    caches[camera_role] = None
                    required_path = video_path
                else:
                    cache_path = _cache_path_for_video(
                        video_path, source_root, cache_root
                    )
                    caches[camera_role] = cache_path
                    required_path = cache_path
                if not required_path.is_file():
                    source_kind = (
                        "camera cache" if cache_root is not None else "camera video"
                    )
                    raise FileNotFoundError(
                        f"missing {camera_role} {source_kind}: domain={domain}, "
                        f"episode={episode_index}, camera={camera_name}, "
                        f"path={required_path}"
                    )

            discovered.append(
                SourceEpisode(
                    key=key,
                    source_root=source_root,
                    domain=domain,
                    episode_index=episode_index,
                    length=length,
                    caption=task_text,
                    parquet_path=parquet_path,
                    main_video_path=videos["main"],
                    wrist_video_path=videos["wrist"],
                    main_cache_path=caches["main"],
                    wrist_cache_path=caches["wrist"],
                )
            )

    if not discovered:
        raise ValueError("no source episodes discovered")
    if max_episodes is not None:
        discovered = discovered[:max_episodes]
    return discovered


def _validate_rgb(frames: np.ndarray, *, source: str) -> None:
    if (
        not isinstance(frames, np.ndarray)
        or frames.ndim != 4
        or len(frames) == 0
        or frames.shape[-1] != 3
        or frames.dtype != np.uint8
    ):
        shape = getattr(frames, "shape", None)
        dtype = getattr(frames, "dtype", None)
        raise ValueError(
            f"invalid RGB source {source}: expected nonempty [T,H,W,3] uint8, "
            f"got shape={shape}, dtype={dtype}"
        )


def resize_rgb_uint8(
    frames: np.ndarray,
    *,
    size: tuple[int, int] = IMAGE_SIZE,
    microbatch: int = 16,
) -> np.ndarray:
    """Resize RGB frames through the training float-bilinear reference path."""
    _validate_rgb(frames, source="array")
    if type(microbatch) is not int or microbatch < 1:
        raise ValueError("microbatch must be a positive integer")
    if (
        type(size) not in (tuple, list)
        or len(size) != 2
        or any(type(value) is not int or value < 1 for value in size)
    ):
        raise ValueError("size must contain two positive integers")

    resized_batches: list[np.ndarray] = []
    for start in range(0, len(frames), microbatch):
        batch = np.array(frames[start : start + microbatch], copy=True)
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float().div_(255.0)
        resized = tvf.resize(
            tensor,
            list(size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        quantized = (
            resized.mul(255.0)
            .round_()
            .clamp_(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )
        resized_batches.append(quantized)
    return np.concatenate(resized_batches, axis=0)


def decode_rgb_video(video_path: str | Path) -> np.ndarray:
    """Sequentially decode one MP4 into an RGB uint8 array."""
    path = Path(video_path)
    frames: list[np.ndarray] = []
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError("video has no video stream")
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray(format="rgb24"))
    except Exception as error:
        raise ValueError(f"cannot decode camera video {path}: {error}") from error
    if not frames:
        raise ValueError(f"camera video has no decoded frames: {path}")
    result = np.stack(frames).astype(np.uint8, copy=False)
    _validate_rgb(result, source=str(path))
    return result


def _load_rgb_source(episode: SourceEpisode, camera_role: str) -> np.ndarray:
    cache_path = getattr(episode, f"{camera_role}_cache_path")
    video_path = getattr(episode, f"{camera_role}_video_path")
    if cache_path is None:
        frames = decode_rgb_video(video_path)
        source_path = video_path
    else:
        try:
            frames = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"cannot read {camera_role} camera cache: domain={episode.domain}, "
                f"episode={episode.episode_index}, path={cache_path}: {error}"
            ) from error
        source_path = cache_path
    _validate_rgb(frames, source=str(source_path))
    if len(frames) != episode.length:
        raise ValueError(
            f"camera frame count mismatch: domain={episode.domain}, "
            f"episode={episode.episode_index}, camera={camera_role}, "
            f"path={source_path}, expected={episode.length}, got={len(frames)}"
        )
    return frames


def _load_controls(episode: SourceEpisode) -> tuple[np.ndarray, np.ndarray]:
    try:
        frame = pd.read_parquet(
            episode.parquet_path, columns=["action", "observation.state"]
        )
    except Exception as error:
        raise ValueError(
            f"cannot read control parquet: domain={episode.domain}, "
            f"episode={episode.episode_index}, path={episode.parquet_path}: {error}"
        ) from error
    if len(frame) != episode.length:
        raise ValueError(
            f"control row count mismatch: domain={episode.domain}, "
            f"episode={episode.episode_index}, path={episode.parquet_path}, "
            f"expected={episode.length}, got={len(frame)}"
        )

    arrays: dict[str, np.ndarray] = {}
    for output_name, column, width in (
        ("action", "action", 7),
        ("state", "observation.state", 8),
    ):
        try:
            array = np.stack(frame[column].to_list())
        except Exception as error:
            raise ValueError(
                f"invalid {output_name} column: domain={episode.domain}, "
                f"episode={episode.episode_index}, path={episode.parquet_path}: "
                f"{error}"
            ) from error
        expected_shape = (episode.length, width)
        if array.shape != expected_shape:
            raise ValueError(
                f"invalid {output_name} shape: domain={episode.domain}, "
                f"episode={episode.episode_index}, path={episode.parquet_path}, "
                f"expected={expected_shape}, got={array.shape}"
            )
        arrays[output_name] = np.ascontiguousarray(array, dtype=np.float32)
    return arrays["action"], arrays["state"]


def _normalize_arg_paths(value: Any, *, field: str) -> list[Path]:
    if isinstance(value, (str, Path)):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} must be a non-empty path list")
    return [Path(item).expanduser().resolve() for item in value]


def _converter_inputs(args: Any) -> SimpleNamespace:
    roots_value = getattr(args, "data_root", None)
    if roots_value is None:
        roots_value = getattr(args, "data_roots", None)
    roots = _normalize_arg_paths(roots_value, field="data_root")
    domains = getattr(args, "domains", None)
    if not isinstance(domains, (list, tuple)) or not domains:
        raise ValueError("domains must be a non-empty list")
    output_value = getattr(args, "output_root", None)
    if output_value is None:
        raise ValueError("output_root is required")
    compression = getattr(args, "compression", "none")
    if type(compression) is not str or compression not in ("none", "lzf"):
        raise ValueError("compression must be 'none' or 'lzf'")
    episodes_per_shard = getattr(args, "episodes_per_shard", 32)
    if (
        type(episodes_per_shard) is not int
        or episodes_per_shard < 1
        or episodes_per_shard > 32
    ):
        raise ValueError("episodes_per_shard must be an integer in 1..32")
    microbatch = getattr(args, "resize_microbatch", 16)
    if type(microbatch) is not int or microbatch < 1:
        raise ValueError("resize_microbatch must be a positive integer")
    max_episodes = getattr(args, "max_episodes", None)
    if max_episodes is not None and (type(max_episodes) is not int or max_episodes < 1):
        raise ValueError("max_episodes must be a positive integer or None")
    predecoded_value = getattr(args, "predecoded_root", None)
    return SimpleNamespace(
        data_roots=roots,
        domains=list(domains),
        predecoded_root=(
            None
            if predecoded_value is None
            else Path(predecoded_value).expanduser().resolve()
        ),
        output_root=Path(output_value).expanduser().resolve(),
        max_episodes=max_episodes,
        episodes_per_shard=episodes_per_shard,
        compression=compression,
        resize_microbatch=microbatch,
        overwrite=bool(getattr(args, "overwrite", False)),
    )


def _unique_source_roots(data_roots: Iterable[Path]) -> list[str]:
    roots: list[str] = []
    seen: set[Path] = set()
    for source_root in data_roots:
        if source_root not in seen:
            seen.add(source_root)
            roots.append(str(source_root))
    return roots


def _fingerprint(config: SimpleNamespace, episodes: Sequence[SourceEpisode]) -> str:
    canonical = {
        "schema_version": schema.SCHEMA_VERSION,
        "contract": schema.FIXED_CONTRACT,
        "datasets": schema.DATASET_DECLARATIONS,
        "source_specs": [
            [str(root), domain]
            for root, domain in _aligned_source_specs(config.data_roots, config.domains)
        ],
        "predecoded_root": (
            None if config.predecoded_root is None else str(config.predecoded_root)
        ),
        "compression": config.compression,
        "episodes_per_shard": config.episodes_per_shard,
        "max_episodes": config.max_episodes,
        "resize_microbatch": config.resize_microbatch,
        "camera_paths": CAMERA_PATHS,
        "episodes": [
            {
                "key": episode.key,
                "length": episode.length,
                "caption": episode.caption,
                "parquet_path": str(episode.parquet_path),
                "main_source": str(episode.main_cache_path or episode.main_video_path),
                "wrist_source": str(
                    episode.wrist_cache_path or episode.wrist_video_path
                ),
            }
            for episode in episodes
        ],
    }
    serialized = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _record_for(episode: SourceEpisode, shard_path: Path) -> schema.EpisodeRecord:
    return schema.EpisodeRecord(
        key=episode.key,
        shard_path=shard_path.resolve(),
        group=f"episodes/{episode.key}",
        caption=episode.caption,
        domain=episode.domain,
        episode_index=episode.episode_index,
        length=episode.length,
    )


def _camera_names_attr() -> np.ndarray:
    return np.asarray(["main", "wrist"], dtype=h5py.string_dtype(encoding="utf-8"))


def _read_camera_names_attribute(file: h5py.File) -> list[str]:
    if "camera_names" not in file.attrs:
        raise ValueError("shard is missing camera_names attribute")
    values = np.atleast_1d(file.attrs["camera_names"]).tolist()
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _validate_shard(
    shard_path: Path,
    records: Sequence[schema.EpisodeRecord],
    *,
    compression: str,
    fingerprint: str,
) -> None:
    with h5py.File(shard_path, "r") as file:
        expected_attrs = {
            "schema_version": schema.SCHEMA_VERSION,
            "compression": compression,
            "image_height": IMAGE_SIZE[0],
            "image_width": IMAGE_SIZE[1],
            "converter_fingerprint": fingerprint,
        }
        for name, expected in expected_attrs.items():
            actual = file.attrs.get(name)
            if actual != expected:
                raise ValueError(
                    f"shard {shard_path} attribute {name} must be "
                    f"{expected!r}, got {actual!r}"
                )
        camera_names = _read_camera_names_attribute(file)
        if camera_names != ["main", "wrist"]:
            raise ValueError(
                f"shard {shard_path} camera_names must be ['main', 'wrist'], "
                f"got {camera_names!r}"
            )
        if "episodes" not in file or not isinstance(file["episodes"], h5py.Group):
            raise ValueError(f"shard {shard_path} is missing episodes group")
        actual_groups = set(file["episodes"].keys())
        expected_groups = {record.key for record in records}
        if actual_groups != expected_groups:
            raise ValueError(
                f"shard {shard_path} episode groups mismatch: expected "
                f"{sorted(expected_groups)!r}, got {sorted(actual_groups)!r}"
            )
        for record in records:
            group = file[record.group]
            try:
                schema.validate_episode_group(group, record)
            except Exception as error:
                raise ValueError(
                    f"shard {shard_path} group {record.group} is invalid: {error}"
                ) from error

            expected_rgb_compression = None if compression == "none" else "lzf"
            storage_contract = {
                "rgb_main": ((1, 256, 256, 3), expected_rgb_compression),
                "rgb_wrist": ((1, 256, 256, 3), expected_rgb_compression),
                "action": ((min(64, record.length), 7), None),
                "state": ((min(64, record.length), 8), None),
            }
            for dataset_name, (
                expected_chunks,
                expected_compression,
            ) in storage_contract.items():
                dataset = group[dataset_name]
                if dataset.chunks != expected_chunks:
                    raise ValueError(
                        f"shard {shard_path} group {record.group} dataset "
                        f"{dataset_name} chunks must be {expected_chunks!r}, "
                        f"got {dataset.chunks!r}"
                    )
                if dataset.compression != expected_compression:
                    raise ValueError(
                        f"shard {shard_path} group {record.group} dataset "
                        f"{dataset_name} compression must be "
                        f"{expected_compression!r}, got {dataset.compression!r}"
                    )
            for dataset_name in ("caption", "domain", "episode_index", "length"):
                dataset = group[dataset_name]
                if dataset.chunks is not None or dataset.compression is not None:
                    raise ValueError(
                        f"shard {shard_path} group {record.group} scalar "
                        f"{dataset_name} must be contiguous and uncompressed, got "
                        f"chunks={dataset.chunks!r}, compression="
                        f"{dataset.compression!r}"
                    )


def _write_episode_group(
    file: h5py.File,
    episode: SourceEpisode,
    record: schema.EpisodeRecord,
    *,
    compression: str,
    resize_microbatch: int,
) -> None:
    main = resize_rgb_uint8(
        _load_rgb_source(episode, "main"), microbatch=resize_microbatch
    )
    wrist = resize_rgb_uint8(
        _load_rgb_source(episode, "wrist"), microbatch=resize_microbatch
    )
    action, state = _load_controls(episode)
    group = file.create_group(record.group)
    hdf5_compression = None if compression == "none" else "lzf"
    rgb_options = {
        "chunks": (1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        "compression": hdf5_compression,
    }
    group.create_dataset("rgb_main", data=main, dtype=np.uint8, **rgb_options)
    group.create_dataset("rgb_wrist", data=wrist, dtype=np.uint8, **rgb_options)
    group.create_dataset(
        "action",
        data=action,
        dtype=np.float32,
        chunks=(min(64, episode.length), 7),
    )
    group.create_dataset(
        "state",
        data=state,
        dtype=np.float32,
        chunks=(min(64, episode.length), 8),
    )
    string_dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset("caption", data=episode.caption, dtype=string_dtype)
    group.create_dataset("domain", data=episode.domain, dtype=string_dtype)
    group.create_dataset("episode_index", data=episode.episode_index, dtype=np.int64)
    group.create_dataset("length", data=episode.length, dtype=np.int64)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_shard_atomic(
    shard_path: Path,
    episodes: Sequence[SourceEpisode],
    records: Sequence[schema.EpisodeRecord],
    *,
    compression: str,
    resize_microbatch: int,
    fingerprint: str,
) -> None:
    temporary = shard_path.parent / (
        f".{shard_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        with h5py.File(temporary, "w") as file:
            file.attrs["schema_version"] = schema.SCHEMA_VERSION
            file.attrs["compression"] = compression
            file.attrs["image_height"] = IMAGE_SIZE[0]
            file.attrs["image_width"] = IMAGE_SIZE[1]
            file.attrs["camera_names"] = _camera_names_attr()
            file.attrs["converter_fingerprint"] = fingerprint
            file.create_group("episodes")
            for episode, record in zip(episodes, records):
                _write_episode_group(
                    file,
                    episode,
                    record,
                    compression=compression,
                    resize_microbatch=resize_microbatch,
                )
            file.flush()
        _fsync_file(temporary)
        _validate_shard(
            temporary,
            records,
            compression=compression,
            fingerprint=fingerprint,
        )
        os.replace(temporary, shard_path)
    except Exception as error:
        raise ValueError(f"failed to build shard {shard_path}: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_payload(
    config: SimpleNamespace,
    episodes: Sequence[SourceEpisode],
    records: Sequence[schema.EpisodeRecord],
    fingerprint: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": schema.SCHEMA_VERSION,
        **schema.FIXED_CONTRACT,
        "compression": config.compression,
        "source_roots": _unique_source_roots(config.data_roots),
        "datasets": schema.DATASET_DECLARATIONS,
        "converter_fingerprint": fingerprint,
        "episodes": [],
    }
    for record in records:
        payload["episodes"].append(
            {
                "key": record.key,
                "shard": record.shard_path.name,
                "group": record.group,
                "caption": record.caption,
                "domain": record.domain,
                "episode_index": record.episode_index,
                "length": record.length,
            }
        )
    return payload


def _validate_published_output(
    manifest_path: Path, *, expected_fingerprint: str | None = None
) -> tuple[dict[str, Any], list[schema.EpisodeRecord]]:
    payload, records = schema.load_manifest(manifest_path)
    if (
        expected_fingerprint is not None
        and payload["converter_fingerprint"] != expected_fingerprint
    ):
        raise ValueError(
            "converter fingerprint mismatch: "
            f"existing={payload['converter_fingerprint']}, "
            f"requested={expected_fingerprint}"
        )
    records_by_shard: dict[Path, list[schema.EpisodeRecord]] = {}
    for record in records:
        records_by_shard.setdefault(record.shard_path, []).append(record)
    for shard_path, shard_records in records_by_shard.items():
        _validate_shard(
            shard_path,
            shard_records,
            compression=payload["compression"],
            fingerprint=payload["converter_fingerprint"],
        )
    return payload, records


def _make_shard_plan(
    output_root: Path,
    generation: str,
    episodes: Sequence[SourceEpisode],
    episodes_per_shard: int,
) -> tuple[
    list[Path],
    list[list[SourceEpisode]],
    list[list[schema.EpisodeRecord]],
    list[schema.EpisodeRecord],
]:
    shard_count = (len(episodes) + episodes_per_shard - 1) // episodes_per_shard
    shard_paths = [
        output_root / f"shard_{index:05d}_{generation}.h5"
        for index in range(shard_count)
    ]
    episode_batches: list[list[SourceEpisode]] = []
    record_batches: list[list[schema.EpisodeRecord]] = []
    all_records: list[schema.EpisodeRecord] = []
    for shard_index, shard_path in enumerate(shard_paths):
        start = shard_index * episodes_per_shard
        batch = list(episodes[start : start + episodes_per_shard])
        records = [_record_for(episode, shard_path) for episode in batch]
        episode_batches.append(batch)
        record_batches.append(records)
        all_records.extend(records)
    return shard_paths, episode_batches, record_batches, all_records


def convert_dataset(args: Any) -> dict[str, int | str]:
    """Convert source episodes and atomically publish a validated manifest."""
    config = _converter_inputs(args)
    episodes = discover_source_episodes(
        config.data_roots,
        config.domains,
        predecoded_root=config.predecoded_root,
        max_episodes=config.max_episodes,
    )
    fingerprint = _fingerprint(config, episodes)
    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"

    fresh_generation_required = False
    if manifest_path.exists():
        try:
            existing_payload, existing_records = _validate_published_output(
                manifest_path
            )
        except Exception:
            if not config.overwrite:
                raise
            fresh_generation_required = True
        else:
            if existing_payload["converter_fingerprint"] != fingerprint:
                if not config.overwrite:
                    raise ValueError(
                        "converter fingerprint mismatch: "
                        f"existing={existing_payload['converter_fingerprint']}, "
                        f"requested={fingerprint}; pass --overwrite to replace"
                    )
                fresh_generation_required = True
            else:
                return {
                    "episodes": len(existing_records),
                    "shards": len({record.shard_path for record in existing_records}),
                    "compression": existing_payload["compression"],
                }

    generation = fingerprint[:12]
    if fresh_generation_required:
        generation = f"{generation}_{uuid.uuid4().hex[:8]}"
    shard_paths, episode_batches, record_batches, all_records = _make_shard_plan(
        output_root,
        generation,
        episodes,
        config.episodes_per_shard,
    )

    reused_shards: set[Path] = set()
    if not manifest_path.exists() and not fresh_generation_required:
        invalid_orphan: tuple[Path, Exception] | None = None
        for shard_path, record_batch in zip(shard_paths, record_batches):
            if not shard_path.exists():
                continue
            try:
                _validate_shard(
                    shard_path,
                    record_batch,
                    compression=config.compression,
                    fingerprint=fingerprint,
                )
            except Exception as error:
                invalid_orphan = (shard_path, error)
                break
            reused_shards.add(shard_path)

        if invalid_orphan is not None:
            invalid_path, validation_error = invalid_orphan
            if not config.overwrite:
                raise ValueError(
                    f"invalid orphan shard {invalid_path}: {validation_error}; "
                    "pass --overwrite to create a fresh immutable generation"
                ) from validation_error
            generation = f"{fingerprint[:12]}_{uuid.uuid4().hex[:8]}"
            (
                shard_paths,
                episode_batches,
                record_batches,
                all_records,
            ) = _make_shard_plan(
                output_root,
                generation,
                episodes,
                config.episodes_per_shard,
            )
            reused_shards.clear()

    payload = _manifest_payload(config, episodes, all_records, fingerprint)
    created_shards: list[Path] = []
    try:
        for shard_path, episode_batch, record_batch in zip(
            shard_paths, episode_batches, record_batches
        ):
            if shard_path in reused_shards:
                continue
            _write_shard_atomic(
                shard_path,
                episode_batch,
                record_batch,
                compression=config.compression,
                resize_microbatch=config.resize_microbatch,
                fingerprint=fingerprint,
            )
            created_shards.append(shard_path)

        validated_records = schema.validate_manifest(payload, output_root)
        by_shard: dict[Path, list[schema.EpisodeRecord]] = {}
        for record in validated_records:
            by_shard.setdefault(record.shard_path, []).append(record)
        for shard_path, records in by_shard.items():
            _validate_shard(
                shard_path,
                records,
                compression=config.compression,
                fingerprint=fingerprint,
            )
        schema.atomic_write_manifest(manifest_path, payload)
    except Exception:
        for shard_path in created_shards:
            shard_path.unlink(missing_ok=True)
        raise

    return {
        "episodes": len(episodes),
        "shards": len(shard_paths),
        "compression": config.compression,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        action="append",
        required=True,
        help="LeRobot source root; repeat for aligned roots if needed",
    )
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--predecoded-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--episodes-per-shard", type=int, default=32)
    parser.add_argument("--compression", choices=("none", "lzf"), default="none")
    parser.add_argument("--resize-microbatch", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = convert_dataset(args)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
