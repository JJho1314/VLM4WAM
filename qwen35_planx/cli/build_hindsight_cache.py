#!/usr/bin/env python3
"""Build, finalize, and audit the sharded grounded hindsight cache."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Sequence

import numpy as np
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F

from qwen35_planx.config import CAMERA_NAMES, HindsightCacheMetadata
from qwen35_planx.hashing import sha256_file, sha256_json
from qwen35_planx.hindsight_builder import (
    HindsightTarget,
    HindsightTargetBuilder,
    build_counterfactual_vocabulary,
)
from qwen35_planx.hindsight_data import (
    HindsightWindowRecord,
    read_full_trajectory,
)
from qwen35_planx.hindsight_schema import (
    PHRASE_ROLES,
    HindsightCache,
    HindsightShardWriter,
    finalize_hindsight_cache,
)
from qwen35_planx.instruction import parse_libero_instruction


_FORMAT_VERSION = 1
_WINDOW_FILES = ("hindsight_train.jsonl", "hindsight_val.jsonl")
_WINDOW_CONTRACT_FIELDS = (
    "format_version",
    "camera_names",
    "num_keyframes",
    "ge_act_future_indices",
    "action_chunk",
    "chunk",
    "n_previous",
    "video_temporal_stride",
    "split_seed",
    "window_stride",
    "sample_n_frames",
)
_WINDOW_ENVELOPE_FIELDS = set(_WINDOW_CONTRACT_FIELDS) | {
    "contract_hash",
    "hdf5_manifest",
    "hdf5_manifest_hash",
    "window_manifest_hash",
    "files",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _fsync_path(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parse_jsonl(path: Path) -> list[HindsightWindowRecord]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid window JSONL line {line_number}: {error}"
            ) from error
        records.append(HindsightWindowRecord.from_dict(payload))
    return records


def _load_window_manifest_envelope(
    path: Path | str,
    *,
    expected_hdf5_manifest: Path | str | None = None,
) -> tuple[dict[str, Any], list[HindsightWindowRecord]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"window manifest does not exist: {path}")
    if path.suffix != ".json":
        raise ValueError(
            "window manifest must be the canonical Task-3 JSON envelope, "
            "not standalone JSONL"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid window manifest JSON: {error}") from error
    if not isinstance(payload, dict) or set(payload) != _WINDOW_ENVELOPE_FIELDS:
        raise ValueError("window manifest is not the canonical Task-3 envelope")

    contract = {name: payload[name] for name in _WINDOW_CONTRACT_FIELDS}
    expected_fixed = {
        "format_version": 1,
        "camera_names": ["main", "wrist"],
        "num_keyframes": 4,
        "ge_act_future_indices": [0, 3, 5, 8],
        "action_chunk": 36,
        "chunk": 9,
        "n_previous": 4,
        "video_temporal_stride": 4,
    }
    for name, expected in expected_fixed.items():
        if contract[name] != expected:
            raise ValueError(
                f"window manifest contract {name} must be {expected!r}"
            )
    for name in ("split_seed", "window_stride", "sample_n_frames"):
        if type(contract[name]) is not int:
            raise ValueError(f"window manifest contract {name} must be an integer")
    if contract["window_stride"] <= 0 or contract["sample_n_frames"] <= 36:
        raise ValueError("window manifest stride/sample length contract is invalid")
    if payload["contract_hash"] != sha256_json(contract):
        raise ValueError("window manifest contract_hash mismatch")

    referenced_hdf5 = Path(payload["hdf5_manifest"])
    if not referenced_hdf5.is_absolute() or not referenced_hdf5.is_file():
        raise ValueError(
            "window manifest hdf5_manifest must be an existing absolute path"
        )
    if expected_hdf5_manifest is not None and (
        referenced_hdf5.resolve() != Path(expected_hdf5_manifest).resolve()
    ):
        raise ValueError("window manifest references a different HDF5 manifest")
    if payload["hdf5_manifest_hash"] != sha256_file(referenced_hdf5):
        raise ValueError("window manifest hdf5_manifest_hash mismatch")

    files = payload["files"]
    if not isinstance(files, dict) or set(files) != set(_WINDOW_FILES):
        raise ValueError(
            "window manifest files must reference exactly the train/val JSONL files"
        )
    records = []
    root = path.parent.resolve()
    for name in _WINDOW_FILES:
        description = files[name]
        if (
            not isinstance(description, dict)
            or set(description) != {"records", "sha256"}
            or type(description["records"]) is not int
            or description["records"] < 0
        ):
            raise ValueError(f"window manifest file metadata is invalid for {name}")
        child = (root / name).resolve()
        if child.parent != root or not child.is_file():
            raise ValueError(f"window manifest referenced file is missing: {name}")
        if description["sha256"] != sha256_file(child):
            raise ValueError(f"window manifest referenced SHA-256 mismatch: {name}")
        split_records = _parse_jsonl(child)
        if len(split_records) != description["records"]:
            raise ValueError(f"window manifest referenced record count mismatch: {name}")
        expected_split = name.removeprefix("hindsight_").removesuffix(".jsonl")
        if any(record.split != expected_split for record in split_records):
            raise ValueError(
                f"window record split does not match referenced filename: {name}"
            )
        records.extend(split_records)
    if not records:
        raise ValueError("window manifest must contain at least one record")
    records.sort(key=lambda record: record.sample_id)
    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("window manifest contains duplicate sample IDs")
    if payload["window_manifest_hash"] != sha256_json(
        [record.to_dict() for record in records]
    ):
        raise ValueError("window manifest canonical window_manifest_hash mismatch")
    return payload, records


def load_window_records(
    path: Path | str,
    *,
    expected_hdf5_manifest: Path | str | None = None,
) -> list[HindsightWindowRecord]:
    """Load only the hash-bound canonical Task-3 window-manifest envelope."""

    _, records = _load_window_manifest_envelope(
        path,
        expected_hdf5_manifest=expected_hdf5_manifest,
    )
    return records


def _artifact_hash(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(f"local artifact does not exist: {path}")
    entries = []
    for child in sorted(
        (candidate for candidate in path.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(path).as_posix(),
    ):
        entries.append(
            {
                "path": child.relative_to(path).as_posix(),
                "bytes": child.stat().st_size,
                "sha256": sha256_file(child),
            }
        )
    if not entries:
        raise ValueError(f"local artifact directory is empty: {path}")
    return sha256_json(entries)


def build_cache_metadata(
    *,
    hdf5_manifest: Path | str,
    window_manifest: Path | str,
    records: Sequence[HindsightWindowRecord],
    ta_checkpoint: Path | str,
    siglip_model: Path | str,
    dinov3_model: Path | str,
    microbatch_size: int,
) -> HindsightCacheMetadata:
    """Bind cache provenance to every input and preprocessing implementation."""

    package_root = Path(__file__).resolve().parents[1]
    preprocessing = {
        "format_version": _FORMAT_VERSION,
        "microbatch_size": microbatch_size,
        "modules": {
            name: sha256_file(package_root / name)
            for name in (
                "hindsight_builder.py",
                "siglip_relevance.py",
                "temporal_grounding.py",
            )
        },
    }
    envelope, authoritative_records = _load_window_manifest_envelope(
        window_manifest,
        expected_hdf5_manifest=hdf5_manifest,
    )
    if list(records) != authoritative_records:
        raise ValueError("records differ from authoritative window manifest")
    return HindsightCacheMetadata(
        format_version=HindsightCacheMetadata.FORMAT_VERSION,
        hdf5_manifest_hash=envelope["hdf5_manifest_hash"],
        window_manifest_hash=envelope["window_manifest_hash"],
        instruction_parser_hash=sha256_file(package_root / "instruction.py"),
        ta_tok_hash=_artifact_hash(Path(ta_checkpoint)),
        siglip2_hash=_artifact_hash(Path(siglip_model)),
        dinov3_hash=_artifact_hash(Path(dinov3_model)),
        preprocessing_hash=sha256_json(preprocessing),
    )


def _quantize_relevance(relevance: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    maps = relevance.detach().float().cpu().numpy()
    maximum = maps.max(axis=-1)
    scale = np.minimum(
        np.float32(np.finfo(np.float16).max),
        np.divide(
            np.float32(255),
            maximum,
            out=np.full_like(maximum, np.finfo(np.float16).max),
            where=maximum > 0,
        ),
    )
    quantized = np.rint(maps * scale[..., None]).clip(0, 255).astype(np.uint8)
    empty = quantized.sum(axis=-1) == 0
    for index in np.argwhere(empty):
        index_tuple = tuple(int(value) for value in index)
        position = int(np.argmax(maps[index_tuple]))
        quantized[index_tuple + (position,)] = 1
    return quantized, scale.astype(np.float32)


def _target_arrays(targets: Sequence[HindsightTarget]) -> dict[str, np.ndarray]:
    relevance_q = []
    relevance_scale = []
    for target in targets:
        quantized, scale = _quantize_relevance(target.relevance)
        relevance_q.append(quantized)
        relevance_scale.append(scale)
    return {
        "codes": np.stack(
            [target.codes.numpy() for target in targets]
        ).astype(np.int64),
        "relevance_q": np.stack(relevance_q),
        "relevance_scale": np.stack(relevance_scale),
        "confidence": np.stack(
            [target.confidence.numpy() for target in targets]
        ).astype(np.float32),
        "flow": np.stack(
            [target.flow.numpy() for target in targets]
        ).astype(np.float32),
        "phrase_embeddings": np.stack(
            [target.phrase_embeddings.numpy() for target in targets]
        ).astype(np.float32),
    }


def _shard_name(episode_key: str) -> str:
    return f"trajectory-{sha256_json(episode_key)[:20]}.npz"


def _collect_shard_assignment_errors(
    *,
    records: Sequence[HindsightWindowRecord],
    output: Path | str,
    shard_index: int,
    num_shards: int,
) -> list[str]:
    if type(shard_index) is not int or type(num_shards) is not int:
        return ["shard_index and num_shards must be integers"]
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        return ["shard assignment must satisfy 0 <= index < num_shards"]
    output = Path(output)
    episode_keys = sorted({record.episode_key for record in records})
    collisions = [
        output / _shard_name(episode_key)
        for assignment, episode_key in enumerate(episode_keys)
        if assignment % num_shards == shard_index
        and (output / _shard_name(episode_key)).exists()
    ]
    return [
        f"assigned trajectory shard already exists: {path}"
        for path in collisions
    ]


def build_shards(
    *,
    hdf5_manifest: Path | str,
    window_manifest: Path | str,
    output: Path | str,
    shard_index: int,
    num_shards: int,
    builder: HindsightTargetBuilder,
    metadata: HindsightCacheMetadata,
) -> list[Path]:
    """Build assigned trajectory shards; HDF5 is opened read-only by contract."""

    from ge_act.data.libero_fastwam_hdf5_schema import load_manifest

    if type(shard_index) is not int or type(num_shards) is not int:
        raise TypeError("shard_index and num_shards must be integers")
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("shard assignment must satisfy 0 <= index < num_shards")
    envelope, records = _load_window_manifest_envelope(
        window_manifest,
        expected_hdf5_manifest=hdf5_manifest,
    )
    if metadata.hdf5_manifest_hash != envelope["hdf5_manifest_hash"]:
        raise ValueError("metadata does not bind the authoritative HDF5 manifest hash")
    if metadata.window_manifest_hash != envelope["window_manifest_hash"]:
        raise ValueError("metadata does not bind the supplied window manifest")
    _, episodes = load_manifest(Path(hdf5_manifest))
    episode_lookup = {episode.key: episode for episode in episodes}
    windows_by_episode: dict[str, list[HindsightWindowRecord]] = defaultdict(list)
    for record in records:
        if record.episode_key not in episode_lookup:
            raise ValueError(
                f"window references unknown HDF5 episode: {record.episode_key}"
            )
        if record.caption != episode_lookup[record.episode_key].caption:
            raise ValueError(
                f"window caption does not match HDF5 episode: {record.episode_key}"
            )
        windows_by_episode[record.episode_key].append(record)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    written = []
    episode_keys = sorted(windows_by_episode)
    for assignment, episode_key in enumerate(episode_keys):
        if assignment % num_shards != shard_index:
            continue
        shard_path = output / _shard_name(episode_key)
        if shard_path.exists():
            raise FileExistsError(f"trajectory shard already exists: {shard_path}")
        trajectory = read_full_trajectory(episode_lookup[episode_key])
        episode_windows = sorted(
            windows_by_episode[episode_key],
            key=lambda record: record.sample_id,
        )
        targets = [
            builder.build_window(trajectory, record)
            for record in episode_windows
        ]
        HindsightShardWriter(shard_path, metadata=metadata).write(
            episode_windows,
            **_target_arrays(targets),
        )
        written.append(shard_path)
    return written


def _metadata_from_shard(path: Path) -> HindsightCacheMetadata:
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive:
            raise ValueError(f"shard metadata is missing: {path}")
        payload = json.loads(archive["metadata_json"].tobytes().decode("utf-8"))
    return HindsightCacheMetadata.from_dict(payload)


def _derive_phrase_table(
    cache: HindsightCache,
) -> tuple[dict[str, list[str]], dict[str, torch.Tensor]]:
    vocabulary = build_counterfactual_vocabulary(cache.records)
    values: dict[str, dict[str, list[torch.Tensor]]] = {
        role: defaultdict(list) for role in PHRASE_ROLES
    }
    for index, record in enumerate(cache.records):
        if record.split != "train":
            continue
        fields = parse_libero_instruction(record.caption)
        sample = cache[index]
        for role_index, role in enumerate(PHRASE_ROLES):
            phrase = getattr(fields, role)
            if phrase:
                values[role][phrase].append(
                    sample.phrase_embeddings[role_index].float()
                )

    vocabulary_by_role = {
        "source": list(vocabulary.sources),
        "target": list(vocabulary.targets),
        "action": list(vocabulary.actions),
    }
    tensors = {}
    for role in PHRASE_ROLES:
        rows = []
        for phrase in vocabulary_by_role[role]:
            embeddings = values[role].get(phrase)
            if not embeddings:
                raise ValueError(f"missing train embedding for {role} phrase {phrase!r}")
            rows.append(F.normalize(torch.stack(embeddings).mean(dim=0), dim=-1))
        tensors[role] = (
            torch.stack(rows).to(torch.float16).contiguous()
            if rows
            else torch.empty((0, 1152), dtype=torch.float16)
        )
    return vocabulary_by_role, tensors


def _phrase_table(
    cache_dir: Path,
    records: Sequence[HindsightWindowRecord],
) -> None:
    with HindsightCache.open(cache_dir) as cache:
        if tuple(records) != cache.records:
            raise ValueError("phrase table records do not match finalized cache")
        vocabulary_by_role, tensors = _derive_phrase_table(cache)
        cache_hash = cache.cache_hash

    tensor_path = cache_dir / "phrase_embeddings.safetensors"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=cache_dir,
        prefix=".phrase_embeddings.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_tensor = Path(temporary_name)
    try:
        save_file(tensors, temporary_tensor)
        os.replace(temporary_tensor, tensor_path)
        _fsync_path(tensor_path)
    finally:
        temporary_tensor.unlink(missing_ok=True)
    payload = {
        "format_version": _FORMAT_VERSION,
        "split": "train",
        "cache_hash": cache_hash,
        "phrase_roles": list(PHRASE_ROLES),
        "phrases": vocabulary_by_role,
        "embedding_file": tensor_path.name,
        "embedding_sha256": sha256_file(tensor_path),
        "embedding_tensors": {
            role: {
                "dtype": "float16",
                "shape": list(tensors[role].shape),
            }
            for role in PHRASE_ROLES
        },
    }
    vocabulary_path = cache_dir / "phrase_vocabulary.json"
    with vocabulary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tensor_path.chmod(0o444)
    vocabulary_path.chmod(0o444)
    _fsync_path(cache_dir)


def load_phrase_embedding_table(
    cache_dir: Path | str,
) -> tuple[dict[str, list[str]], dict[str, torch.Tensor]]:
    """Load and rederive the train table so it is bound to cache contents."""

    cache_dir = Path(cache_dir)
    vocabulary_path = cache_dir / "phrase_vocabulary.json"
    tensor_path = cache_dir / "phrase_embeddings.safetensors"
    if not vocabulary_path.is_file() or not tensor_path.is_file():
        raise ValueError("cache phrase vocabulary/table is missing")
    try:
        payload = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cache phrase vocabulary is invalid: {error}") from error
    expected_fields = {
        "format_version",
        "split",
        "cache_hash",
        "phrase_roles",
        "phrases",
        "embedding_file",
        "embedding_sha256",
        "embedding_tensors",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("cache phrase vocabulary fields are invalid")
    if (
        payload["format_version"] != _FORMAT_VERSION
        or payload["split"] != "train"
        or payload["phrase_roles"] != list(PHRASE_ROLES)
        or payload["embedding_file"] != tensor_path.name
        or payload["embedding_sha256"] != sha256_file(tensor_path)
    ):
        raise ValueError("cache phrase vocabulary contract/hash mismatch")
    tensors = load_file(tensor_path)
    if set(tensors) != set(PHRASE_ROLES):
        raise ValueError("cache phrase embedding tensor roles are invalid")
    with HindsightCache.open(cache_dir) as cache:
        if payload["cache_hash"] != cache.cache_hash:
            raise ValueError("phrase embedding table is bound to a different cache")
        expected_vocabulary, expected_tensors = _derive_phrase_table(cache)
    if payload["phrases"] != expected_vocabulary:
        raise ValueError("phrase embedding table content differs from train vocabulary")
    expected_descriptions = {
        role: {
            "dtype": "float16",
            "shape": list(expected_tensors[role].shape),
        }
        for role in PHRASE_ROLES
    }
    if payload["embedding_tensors"] != expected_descriptions:
        raise ValueError("phrase embedding table tensor metadata is invalid")
    for role in PHRASE_ROLES:
        if (
            tensors[role].dtype != torch.float16
            or not bool(torch.isfinite(tensors[role]).all())
            or not torch.equal(tensors[role], expected_tensors[role])
        ):
            raise ValueError(
                f"phrase embedding table content differs from cache for role {role}"
            )
    return expected_vocabulary, tensors


def _write_build_diagnostics(cache_dir: Path) -> None:
    with HindsightCache.open(cache_dir) as cache:
        body = {
            "format_version": _FORMAT_VERSION,
            "policy": "fail_closed_no_discard",
            "cache_hash": cache.cache_hash,
            "index_sha256": cache.manifest["index"]["sha256"],
            "validated_trajectory_ids": sorted(
                {record.episode_key for record in cache.records}
            ),
            "discarded_trajectory_ids": [],
            "non_finite_trajectory_ids": [],
        }
    payload = {**body, "diagnostics_hash": sha256_json(body)}
    path = cache_dir / "build_diagnostics.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def _load_build_diagnostics(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "build_diagnostics.json"
    if not path.is_file():
        raise ValueError("cache build diagnostics are missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cache build diagnostics are invalid: {error}") from error
    fields = {
        "format_version",
        "policy",
        "cache_hash",
        "index_sha256",
        "validated_trajectory_ids",
        "discarded_trajectory_ids",
        "non_finite_trajectory_ids",
        "diagnostics_hash",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("cache build diagnostics fields are invalid")
    body = {
        name: payload[name]
        for name in fields
        if name != "diagnostics_hash"
    }
    if payload["diagnostics_hash"] != sha256_json(body):
        raise ValueError("cache build diagnostics hash mismatch")
    with HindsightCache.open(cache_dir) as cache:
        expected_ids = sorted({record.episode_key for record in cache.records})
        valid = (
            payload["format_version"] == _FORMAT_VERSION
            and payload["policy"] == "fail_closed_no_discard"
            and payload["cache_hash"] == cache.cache_hash
            and payload["index_sha256"] == cache.manifest["index"]["sha256"]
            and payload["validated_trajectory_ids"] == expected_ids
            and payload["discarded_trajectory_ids"] == []
            and payload["non_finite_trajectory_ids"] == []
        )
    if not valid:
        raise ValueError(
            "cache build diagnostics violate the successful-cache invariant"
        )
    return payload


def finalize_cache(
    *,
    window_manifest: Path | str,
    shard_root: Path | str,
    output: Path | str,
) -> dict[str, Any]:
    """Atomically publish the strict cache plus its train-only phrase table."""

    records = load_window_records(window_manifest)
    shard_paths = sorted(Path(shard_root).glob("*.npz"))
    if not shard_paths:
        raise ValueError(f"no completed trajectory shards found in: {shard_root}")
    metadata = _metadata_from_shard(shard_paths[0])
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"cache output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            dir=output.parent,
            prefix=f".{output.name}.task5.",
            suffix=".tmp",
        )
    )
    stage.rmdir()
    try:
        manifest = finalize_hindsight_cache(
            stage,
            shard_paths=shard_paths,
            metadata=metadata,
            expected_records=records,
        )
        _phrase_table(stage, records)
        _write_build_diagnostics(stage)
        _fsync_path(stage)
        os.replace(stage, output)
        _fsync_path(output.parent)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def _counterfactual_margins(cache: Path) -> dict[str, float]:
    _, tensors = load_phrase_embedding_table(cache)
    margins = {}
    for role in PHRASE_ROLES:
        values = F.normalize(tensors[role].float(), dim=-1)
        if len(values) < 2:
            margins[role] = 0.0
            continue
        similarities = values @ values.T
        similarities.fill_diagonal_(-torch.inf)
        margins[role] = float((1 - similarities.amax(dim=-1)).mean())
    return margins


def audit_cache(
    *,
    cache: Path | str,
    samples: int,
    output: Path | str,
) -> dict[str, Any]:
    """Compute deterministic teacher/cache health metrics."""

    if type(samples) is not int or samples <= 0:
        raise ValueError("samples must be a positive integer")
    cache = Path(cache)
    margins = _counterfactual_margins(cache)
    diagnostics = _load_build_diagnostics(cache)
    with HindsightCache.open(cache) as opened:
        count = min(samples, len(opened))
        indices = np.linspace(0, len(opened) - 1, count, dtype=np.int64)
        entropy_values: dict[tuple[int, int], list[float]] = defaultdict(list)
        support_values: dict[tuple[int, int], list[float]] = defaultdict(list)
        confidence_values: dict[tuple[int, int], list[float]] = defaultdict(list)
        flow_values: dict[int, list[float]] = defaultdict(list)
        code_values: dict[int, list[np.ndarray]] = defaultdict(list)
        for raw_index in indices:
            sample = opened[int(raw_index)]
            relevance = sample.relevance.float()
            entropy = -(
                relevance
                * torch.log(relevance.clamp_min(torch.finfo(torch.float32).tiny))
            ).sum(dim=-1)
            support = torch.exp(entropy) / relevance.shape[-1]
            for camera_index in range(2):
                for role_index in range(3):
                    entropy_values[camera_index, role_index].extend(
                        entropy[camera_index, :, role_index].tolist()
                    )
                    support_values[camera_index, role_index].extend(
                        support[camera_index, :, role_index].tolist()
                    )
                    confidence_values[camera_index, role_index].extend(
                        sample.confidence[camera_index, :, role_index].float().tolist()
                    )
                flow_values[camera_index].extend(
                    sample.flow[camera_index, :, :, 2].float().flatten().tolist()
                )
                code_values[camera_index].append(
                    sample.codes[camera_index].numpy().reshape(-1)
                )

        per_camera_phrase = {}
        per_camera = {}
        for camera_index, camera_name in enumerate(CAMERA_NAMES):
            per_camera_phrase[camera_name] = {}
            for role_index, role in enumerate(PHRASE_ROLES):
                confidence = np.asarray(
                    confidence_values[camera_index, role_index],
                    dtype=np.float64,
                )
                per_camera_phrase[camera_name][role] = {
                    "valid_confidence_ratio": float(np.mean(confidence > 0)),
                    "effective_support": float(
                        np.mean(support_values[camera_index, role_index])
                    ),
                    "map_entropy": float(
                        np.mean(entropy_values[camera_index, role_index])
                    ),
                    "counterfactual_margin": margins[role],
                }
            codes = np.concatenate(code_values[camera_index])
            unique = int(np.unique(codes).size)
            per_camera[camera_name] = {
                "temporal_cycle_confidence": float(
                    np.mean(flow_values[camera_index])
                ),
                "ta_code_usage": {
                    "unique_codes": unique,
                    "vocabulary_ratio": unique / 65_536,
                },
            }
        metrics = {
            "format_version": _FORMAT_VERSION,
            "cache_hash": opened.cache_hash,
            "samples_audited": count,
            "per_camera_phrase": per_camera_phrase,
            "per_camera": per_camera,
            "validated_trajectory_ids": diagnostics["validated_trajectory_ids"],
            "discarded_trajectory_ids": diagnostics["discarded_trajectory_ids"],
            "non_finite_trajectory_ids": diagnostics[
                "non_finite_trajectory_ids"
            ],
        }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metrics


def _load_production_builder(arguments: argparse.Namespace) -> HindsightTargetBuilder:
    from qwen35_planx.official_ta_tok import ReleasedTATok
    from qwen35_planx.siglip_relevance import SiglipRelevanceTeacher
    from qwen35_planx.temporal_grounding import DinoTemporalTeacher

    records = load_window_records(
        arguments.window_manifest,
        expected_hdf5_manifest=arguments.hdf5_manifest,
    )
    vocabulary = build_counterfactual_vocabulary(records)
    device = torch.device(
        f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
        if torch.cuda.is_available()
        else "cpu"
    )
    with torch.device(device):
        ta_tokenizer = ReleasedTATok.from_checkpoint(
            arguments.ta_checkpoint,
            siglip_model_path=arguments.siglip_model,
        )
        siglip_teacher = SiglipRelevanceTeacher.from_pretrained(
            arguments.siglip_model,
            local_files_only=True,
        )
        dino_teacher = DinoTemporalTeacher.from_pretrained(
            arguments.dinov3_model,
            local_files_only=True,
        )
    ta_tokenizer.to(device)
    siglip_teacher.model.to(device)
    dino_teacher.model.to(device)
    return HindsightTargetBuilder.from_components(
        ta_tokenizer=ta_tokenizer,
        siglip_teacher=siglip_teacher,
        dino_teacher=dino_teacher,
        vocabulary=vocabulary,
        microbatch_size=arguments.microbatch_size,
    )


def _build_command(arguments: argparse.Namespace) -> int:
    from qwen35_planx.cli.preflight import (
        collect_hindsight_cache_preflight_errors,
    )

    errors = collect_hindsight_cache_preflight_errors(
        hdf5_manifest=arguments.hdf5_manifest,
        window_manifest=arguments.window_manifest,
        ta_checkpoint=arguments.ta_checkpoint,
        siglip_model=arguments.siglip_model,
        dinov3_model=arguments.dinov3_model,
        output_dir=arguments.output,
        require_new_output=False,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    records = load_window_records(
        arguments.window_manifest,
        expected_hdf5_manifest=arguments.hdf5_manifest,
    )
    shard_index = arguments.shard_index
    if shard_index is None:
        shard_index = int(os.environ.get("RANK", "0"))
    assignment_errors = _collect_shard_assignment_errors(
        records=records,
        output=arguments.output,
        shard_index=shard_index,
        num_shards=arguments.num_shards,
    )
    if assignment_errors:
        for error in assignment_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    metadata = build_cache_metadata(
        hdf5_manifest=arguments.hdf5_manifest,
        window_manifest=arguments.window_manifest,
        records=records,
        ta_checkpoint=arguments.ta_checkpoint,
        siglip_model=arguments.siglip_model,
        dinov3_model=arguments.dinov3_model,
        microbatch_size=arguments.microbatch_size,
    )
    builder = _load_production_builder(arguments)
    paths = build_shards(
        hdf5_manifest=arguments.hdf5_manifest,
        window_manifest=arguments.window_manifest,
        output=arguments.output,
        shard_index=shard_index,
        num_shards=arguments.num_shards,
        builder=builder,
        metadata=metadata,
    )
    print(_canonical_json({"shards": [str(path) for path in paths]}))
    return 0


def _finalize_command(arguments: argparse.Namespace) -> int:
    manifest = finalize_cache(
        window_manifest=arguments.window_manifest,
        shard_root=arguments.shard_root,
        output=arguments.output,
    )
    print(_canonical_json({"cache_hash": manifest["cache_hash"]}))
    return 0


def _audit_command(arguments: argparse.Namespace) -> int:
    metrics = audit_cache(
        cache=arguments.cache,
        samples=arguments.samples,
        output=arguments.output,
    )
    print(_canonical_json(metrics))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--hdf5-manifest", type=Path, required=True)
    build.add_argument("--window-manifest", type=Path, required=True)
    build.add_argument("--ta-checkpoint", type=Path, required=True)
    build.add_argument("--siglip-model", type=Path, required=True)
    build.add_argument("--dinov3-model", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--shard-index", type=int)
    build.add_argument("--num-shards", type=int, required=True)
    build.add_argument("--microbatch-size", type=int, default=16)
    build.set_defaults(handler=_build_command)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--window-manifest", type=Path, required=True)
    finalize.add_argument("--shard-root", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.set_defaults(handler=_finalize_command)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--cache", type=Path, required=True)
    audit.add_argument("--samples", type=int, default=128)
    audit.add_argument("--output", type=Path, required=True)
    audit.set_defaults(handler=_audit_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
