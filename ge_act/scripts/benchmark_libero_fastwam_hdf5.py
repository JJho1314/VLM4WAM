#!/usr/bin/env python3
"""Parity and CPU/DataLoader benchmark for the optional LIBERO HDF5 backend."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

try:
    import psutil as _psutil
except ModuleNotFoundError:  # pragma: no cover - exercised through fallback tests
    _psutil = None


GE_ACT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = GE_ACT_ROOT.parent
EXPECTED_SAMPLE_KEYS = {"video", "actions", "caption", "state"}
EXACT_FIELDS = ["actions", "state", "caption", "shape", "dtype"]
RGB_ERROR_BOUND = 1.0 / 255.0 + 1e-6
EPISODE_PATTERN = re.compile(r"episode_(\d{6})(?:\.[^/]*)?$")


class SampleComparisonError(AssertionError):
    """Assertion carrying machine-readable comparison details."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


def _assert_sample_tensor(value: Any, field: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise SampleComparisonError(f"{field} must be a torch.Tensor")
    return value


def compare_samples(old: dict, new: dict) -> dict[str, Any]:
    """Compare one old/HDF5 sample at the normalized model-input boundary."""
    if type(old) is not dict or type(new) is not dict:
        raise SampleComparisonError("samples must be dict objects")
    if set(old) != EXPECTED_SAMPLE_KEYS or set(new) != EXPECTED_SAMPLE_KEYS:
        raise SampleComparisonError(
            "sample keys must be exactly video/actions/caption/state; "
            f"old={sorted(old)} new={sorted(new)}"
        )

    for field in ("video", "actions", "state"):
        old_tensor = _assert_sample_tensor(old[field], f"old {field}")
        new_tensor = _assert_sample_tensor(new[field], f"new {field}")
        if old_tensor.shape != new_tensor.shape:
            raise SampleComparisonError(
                f"{field} shape mismatch: {tuple(old_tensor.shape)} != "
                f"{tuple(new_tensor.shape)}"
            )
        if old_tensor.dtype != new_tensor.dtype:
            raise SampleComparisonError(
                f"{field} dtype mismatch: {old_tensor.dtype} != {new_tensor.dtype}"
            )

    if old["caption"] != new["caption"]:
        raise SampleComparisonError(
            f"caption mismatch: {old['caption']!r} != {new['caption']!r}"
        )
    for field in ("actions", "state"):
        if not torch.equal(old[field], new[field]):
            raise SampleComparisonError(f"{field} values are not exactly identical")

    old_video = old["video"]
    new_video = new["video"]
    if old_video.ndim < 2 or old_video.shape[1] != 2:
        raise SampleComparisonError(
            "video shape must contain exactly two cameras at dimension 1"
        )
    errors = {
        "main": float((old_video[:, 0] - new_video[:, 0]).abs().max().item()),
        "wrist": float((old_video[:, 1] - new_video[:, 1]).abs().max().item()),
    }
    maximum = max(errors.values())
    details = {
        "normalized_rgb_error": errors,
        "max_normalized_rgb_error": maximum,
        "exact_fields": EXACT_FIELDS.copy(),
    }
    if not all(math.isfinite(value) for value in errors.values()):
        raise SampleComparisonError(
            "RGB comparison produced a non-finite error", details
        )

    swapped_errors = {
        "main_to_wrist": float((old_video[:, 0] - new_video[:, 1]).abs().max().item()),
        "wrist_to_main": float((old_video[:, 1] - new_video[:, 0]).abs().max().item()),
    }
    if maximum > RGB_ERROR_BOUND and max(swapped_errors.values()) <= RGB_ERROR_BOUND:
        details["swapped_normalized_rgb_error"] = swapped_errors
        raise SampleComparisonError(
            "camera order failure: apparent main/wrist camera swap", details
        )
    for camera, error in errors.items():
        if error > RGB_ERROR_BOUND:
            raise SampleComparisonError(
                f"{camera} normalized RGB error {error:.9g} exceeds "
                f"{RGB_ERROR_BOUND:.9g}",
                details,
            )
    return details


def choose_compression(results: dict[str, float]) -> str:
    """Choose LZF when it is no more than five percent slower than none."""
    if type(results) is not dict:
        raise ValueError("compression results must be a dict")
    values: dict[str, float] = {}
    for name in ("none", "lzf"):
        if name not in results or isinstance(results[name], bool):
            raise ValueError(f"compression result {name} must be finite and positive")
        try:
            value = float(results[name])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"compression result {name} must be finite and positive"
            ) from error
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"compression result {name} must be finite and positive")
        values[name] = value
    return "lzf" if values["lzf"] >= 0.95 * values["none"] else "none"


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Write one JSON report atomically and remove temporary files on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def percentile(values: Sequence[float], q: float) -> float:
    """Return the linear percentile used in benchmark summaries."""
    if not values:
        raise ValueError("percentile values must not be empty")
    if isinstance(q, bool) or not math.isfinite(float(q)) or not 0 <= float(q) <= 100:
        raise ValueError("percentile q must be finite and in [0, 100]")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("percentile values must be a finite one-dimensional sequence")
    return float(np.percentile(array, float(q), method="linear"))


def _proc_status_path(pid: int) -> Path:
    return Path("/proc") / str(pid) / "status"


def read_process_rss_bytes(pid: int) -> int:
    """Read RSS through psutil, falling back to /proc; vanished PIDs return zero."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("pid must be a positive integer")
    if _psutil is not None:
        try:
            return int(_psutil.Process(pid).memory_info().rss)
        except (_psutil.NoSuchProcess, _psutil.ZombieProcess, ProcessLookupError):
            return 0
        except (_psutil.AccessDenied, PermissionError):
            pass
    try:
        status = _proc_status_path(pid).read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return 0
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            fields = line.split()
            if len(fields) >= 2:
                return int(fields[1]) * 1024
    return 0


def aggregate_worker_rss_bytes(pids: Sequence[int]) -> int:
    return sum(read_process_rss_bytes(int(pid)) for pid in pids)


@dataclass(frozen=True)
class EpisodePair:
    domain: str
    episode_index: int
    caption: str
    length: int
    old_index: int
    hdf5_index: int


@dataclass(frozen=True)
class DeterministicSamplePlan:
    pair: EpisodePair
    fix_sidx: int
    fix_mem_idx: list[int]
    frame_indexes: list[int]
    action_indexes: list[int]


def _episode_index_from_path(path: Any, field: str) -> int:
    if type(path) is not str:
        raise ValueError(f"old episode {field} path must be a string")
    match = EPISODE_PATTERN.search(path)
    if match is None:
        raise ValueError(f"old episode {field} path is not canonical: {path!r}")
    return int(match.group(1))


def _old_episode_metadata(record: Any, index: int) -> tuple[str, int, str, int]:
    if not isinstance(record, (list, tuple)) or len(record) < 8:
        raise ValueError(f"old episode record {index} is malformed")
    video_index = _episode_index_from_path(record[0], "video")
    parquet_index = _episode_index_from_path(record[2], "parquet")
    if video_index != parquet_index:
        raise ValueError(
            f"old episode record {index} video/parquet episode index mismatch"
        )
    domain, caption, length = record[3], record[6], record[7]
    if type(domain) is not str or not domain:
        raise ValueError(f"old episode record {index} domain is invalid")
    if type(caption) is not str or not caption:
        raise ValueError(f"old episode record {index} caption is invalid")
    if type(length) is not int or length < 2:
        raise ValueError(f"old episode record {index} length is invalid")
    return domain, video_index, caption, length


def map_episode_pairs(
    old_dataset: Any,
    hdf5_dataset: Any,
    *,
    episode_limit: int,
) -> list[EpisodePair]:
    """Map HDF5 manifest order onto canonical old dataset identities."""
    if type(episode_limit) is not int or episode_limit <= 0:
        raise ValueError("episode_limit must be a positive integer")
    old_records = getattr(old_dataset, "dataset", None)
    new_records = getattr(hdf5_dataset, "records", None)
    if not isinstance(old_records, list) or not isinstance(new_records, list):
        raise ValueError("datasets must expose old dataset and HDF5 records lists")

    old_by_identity: dict[tuple[str, int], tuple[int, str, int]] = {}
    for old_index, record in enumerate(old_records):
        domain, episode_index, caption, length = _old_episode_metadata(
            record, old_index
        )
        identity = (domain, episode_index)
        if identity in old_by_identity:
            raise ValueError(f"duplicate old episode pair: {identity!r}")
        old_by_identity[identity] = (old_index, caption, length)

    seen_new: set[tuple[str, int]] = set()
    pairs: list[EpisodePair] = []
    for hdf5_index, record in enumerate(new_records):
        identity = (record.domain, record.episode_index)
        if identity in seen_new:
            raise ValueError(f"duplicate HDF5 episode pair: {identity!r}")
        seen_new.add(identity)
        if identity not in old_by_identity:
            raise ValueError(f"missing old episode pair for HDF5 record: {identity!r}")
        old_index, old_caption, old_length = old_by_identity[identity]
        if old_caption != record.caption:
            raise ValueError(
                f"caption mismatch for pair {identity!r}: "
                f"{old_caption!r} != {record.caption!r}"
            )
        if old_length != record.length:
            raise ValueError(
                f"length mismatch for pair {identity!r}: "
                f"{old_length} != {record.length}"
            )
        pairs.append(
            EpisodePair(
                domain=record.domain,
                episode_index=record.episode_index,
                caption=record.caption,
                length=record.length,
                old_index=old_index,
                hdf5_index=hdf5_index,
            )
        )
        if len(pairs) == episode_limit:
            break
    if not pairs:
        raise ValueError("missing mapped episode pairs")
    return pairs


def _fixed_inputs(length: int) -> tuple[int, list[int]]:
    if type(length) is not int or length < 2:
        raise ValueError("episode length must be at least two")
    maximum = length - 1
    return min(12, maximum), [min(index, maximum) for index in (1, 4, 8, 11)]


def build_sample_plans(
    old_dataset: Any,
    hdf5_dataset: Any,
    pairs: Sequence[EpisodePair],
) -> list[DeterministicSamplePlan]:
    """Build fixed indexes and prove both samplers produce identical indexes."""
    plans = []
    old_state = (
        getattr(old_dataset, "fix_sidx", None),
        getattr(old_dataset, "fix_mem_idx", None),
    )
    new_state = (
        getattr(hdf5_dataset, "fix_sidx", None),
        getattr(hdf5_dataset, "fix_mem_idx", None),
    )
    try:
        for pair in pairs:
            fix_sidx, fix_mem_idx = _fixed_inputs(pair.length)
            old_dataset.fix_sidx = fix_sidx
            old_dataset.fix_mem_idx = fix_mem_idx
            hdf5_dataset.fix_sidx = fix_sidx
            hdf5_dataset.fix_mem_idx = fix_mem_idx
            old_indexes = old_dataset.get_frame_indexes(pair.length)
            new_indexes = hdf5_dataset.get_frame_indexes(pair.length)
            old_indexes = (list(old_indexes[0]), list(old_indexes[1]))
            new_indexes = (list(new_indexes[0]), list(new_indexes[1]))
            if old_indexes != new_indexes:
                raise AssertionError(
                    "fixed frame/action indexes differ for "
                    f"{pair.domain}:{pair.episode_index:06d}: "
                    f"old={old_indexes} hdf5={new_indexes}"
                )
            plans.append(
                DeterministicSamplePlan(
                    pair=pair,
                    fix_sidx=fix_sidx,
                    fix_mem_idx=fix_mem_idx,
                    frame_indexes=old_indexes[0],
                    action_indexes=old_indexes[1],
                )
            )
    finally:
        old_dataset.fix_sidx, old_dataset.fix_mem_idx = old_state
        hdf5_dataset.fix_sidx, hdf5_dataset.fix_mem_idx = new_state
    return plans


class _DeterministicView(Dataset):
    """Each forked worker owns its mutable fixed-index dataset state."""

    def __init__(
        self, dataset: Any, plans: Sequence[DeterministicSamplePlan], sample_count: int
    ):
        if not plans:
            raise ValueError("plans must not be empty")
        if type(sample_count) is not int or sample_count <= 0:
            raise ValueError("sample_count must be a positive integer")
        self.dataset = dataset
        self.plans = list(plans)
        self.sample_count = sample_count

    def __len__(self) -> int:
        return self.sample_count

    def _plan(self, index: int) -> DeterministicSamplePlan:
        if type(index) is not int or index < 0 or index >= self.sample_count:
            raise IndexError(index)
        return self.plans[index % len(self.plans)]

    def close(self) -> None:
        close = getattr(self.dataset, "close", None)
        if callable(close):
            close()


class DeterministicOldView(_DeterministicView):
    def __getitem__(self, index: int) -> dict[str, Any]:
        plan = self._plan(index)
        self.dataset.fix_sidx = plan.fix_sidx
        self.dataset.fix_mem_idx = plan.fix_mem_idx
        try:
            video, actions, caption, state = self.dataset.get_batch(plan.pair.old_index)
        except Exception as error:
            raise RuntimeError(
                "old loader read failed for "
                f"{plan.pair.domain}:{plan.pair.episode_index:06d} "
                f"old_index={plan.pair.old_index}: {error}"
            ) from error
        return {
            "video": video,
            "actions": actions,
            "caption": caption,
            "state": state,
        }


class DeterministicHDF5View(_DeterministicView):
    def __getitem__(self, index: int) -> dict[str, Any]:
        plan = self._plan(index)
        try:
            return self.dataset.read_by_indexes(
                plan.pair.hdf5_index,
                plan.frame_indexes,
                plan.action_indexes,
            )
        except Exception as error:
            raise RuntimeError(
                "HDF5 loader read failed for "
                f"{plan.pair.domain}:{plan.pair.episode_index:06d} "
                f"hdf5_index={plan.pair.hdf5_index}: {error}"
            ) from error


def run_parity(
    old_dataset: Any,
    hdf5_dataset: Any,
    plans: Sequence[DeterministicSamplePlan],
) -> dict[str, Any]:
    old_view = DeterministicOldView(old_dataset, plans, len(plans))
    new_view = DeterministicHDF5View(hdf5_dataset, plans, len(plans))
    checked = []
    failures = []
    try:
        for index, plan in enumerate(plans):
            entry: dict[str, Any] = {
                "domain": plan.pair.domain,
                "episode_index": plan.pair.episode_index,
                "old_index": plan.pair.old_index,
                "hdf5_index": plan.pair.hdf5_index,
                "frame_indexes": plan.frame_indexes,
                "action_indexes": plan.action_indexes,
            }
            try:
                entry.update(compare_samples(old_view[index], new_view[index]))
                entry["passed"] = True
            except Exception as error:
                entry.update(getattr(error, "details", {}))
                entry["passed"] = False
                entry["error"] = str(error)
                failures.append(
                    {
                        "domain": plan.pair.domain,
                        "episode_index": plan.pair.episode_index,
                        "old_index": plan.pair.old_index,
                        "hdf5_index": plan.pair.hdf5_index,
                        "error": str(error),
                    }
                )
            checked.append(entry)
    finally:
        old_view.close()
        new_view.close()
    return {
        "passed": not failures,
        "exact_fields": EXACT_FIELDS.copy(),
        "checked_pairs": len(checked),
        "pairs": checked,
        "failures": failures,
    }


def _batch_size(batch: Any) -> int:
    if isinstance(batch, dict):
        video = batch.get("video")
        if isinstance(video, torch.Tensor) and video.ndim:
            return int(video.shape[0])
        caption = batch.get("caption")
        if isinstance(caption, (list, tuple)):
            return len(caption)
    raise RuntimeError("cannot determine observed DataLoader batch size")


def _next_batch(iterator: Iterator[Any], phase: str, index: int) -> Any:
    try:
        return next(iterator)
    except StopIteration as error:
        raise RuntimeError(
            f"insufficient samples during {phase} batch {index}"
        ) from error


def measure_dataloader(
    dataset: Dataset,
    *,
    workers: int,
    batch_size: int,
    warmup_batches: int,
    measure_batches: int,
    prefetch_factor: int,
) -> dict[str, Any]:
    """Measure a fresh CPU-only DataLoader and synchronously clean it up."""
    if type(workers) is not int or workers < 0:
        raise ValueError("workers must be a non-negative integer")
    for name, value, minimum in (
        ("batch_size", batch_size, 1),
        ("warmup_batches", warmup_batches, 0),
        ("measure_batches", measure_batches, 1),
        ("prefetch_factor", prefetch_factor, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=prefetch_factor if workers > 0 else None,
    )
    iterator = None
    worker_processes = []
    result: dict[str, Any] | None = None
    try:
        iterator = iter(loader)
        worker_processes = list(getattr(iterator, "_workers", []) or [])
        worker_pids = [process.pid for process in worker_processes]
        for index in range(warmup_batches):
            _next_batch(iterator, "warmup", index)

        durations = []
        observed_samples = 0
        for index in range(measure_batches):
            start = time.perf_counter()
            batch = _next_batch(iterator, "measurement", index)
            durations.append(time.perf_counter() - start)
            observed_samples += _batch_size(batch)
        total_seconds = sum(durations)
        if total_seconds <= 0:
            raise RuntimeError("measured DataLoader duration must be positive")
        result = {
            "workers": workers,
            "batch_size": batch_size,
            "warmup_batches": warmup_batches,
            "requested_measure_batches": measure_batches,
            "observed_batches": len(durations),
            "observed_samples": observed_samples,
            "batch_seconds": durations,
            "total_measure_seconds": total_seconds,
            "samples_per_second": observed_samples / total_seconds,
            "median_batch_seconds": percentile(durations, 50),
            "p95_batch_seconds": percentile(durations, 95),
            "worker_pids": worker_pids,
            "aggregate_worker_rss_bytes": aggregate_worker_rss_bytes(worker_pids),
            "main_process_rss_bytes": read_process_rss_bytes(os.getpid()),
            "workers_shutdown": False,
        }
    finally:
        if iterator is not None:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                shutdown()
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
        if result is not None:
            result["workers_shutdown"] = all(
                not process.is_alive() for process in worker_processes
            )
    if result is None:  # pragma: no cover - exceptions leave through the try block
        raise RuntimeError("DataLoader benchmark did not produce a result")
    return result


@contextmanager
def _ge_act_import_context() -> Iterator[None]:
    original = list(sys.path)
    try:
        for path in (str(REPOSITORY_ROOT), str(GE_ACT_ROOT)):
            if path not in sys.path:
                sys.path.insert(0, path)
        yield
    finally:
        sys.path[:] = original


def _load_class(class_name: str, source: str) -> type:
    with _ge_act_import_context():
        if not source.endswith(".py") and not os.path.isabs(source):
            module = importlib.import_module(source)
        else:
            path = Path(source)
            if not path.is_absolute():
                path = GE_ACT_ROOT / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"dataset class source not found: {path}")
            digest = hashlib.sha256(str(path).encode()).hexdigest()[:12]
            module_name = f"_ge_act_benchmark_{path.stem}_{digest}"
            module = sys.modules.get(module_name)
            if module is None:
                spec = importlib.util.spec_from_file_location(module_name, path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot import dataset source: {path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    sys.modules.pop(module_name, None)
                    raise
        dataset_class = getattr(module, class_name, None)
        if not isinstance(dataset_class, type):
            raise ImportError(f"class {class_name!r} not found in {source!r}")
        return dataset_class


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if type(payload) is not dict:
        raise ValueError(f"config must contain a mapping: {path}")
    return payload


def construct_train_dataset(
    config_path: str | Path,
    *,
    manifest_override: str | Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Construct one train dataset without changing the YAML on disk."""
    config_path = Path(config_path).expanduser().resolve()
    config = _load_yaml(config_path)
    class_name = config.get("train_data_class")
    class_source = config.get("train_data_class_path")
    data = config.get("data")
    if type(class_name) is not str or type(class_source) is not str:
        raise ValueError(f"config lacks train dataset class fields: {config_path}")
    if type(data) is not dict or type(data.get("train")) is not dict:
        raise ValueError(f"config lacks data.train mapping: {config_path}")
    arguments = copy.deepcopy(data["train"])
    stat_path = arguments.get("stat_file")
    if type(stat_path) is str and not Path(stat_path).is_absolute():
        arguments["stat_file"] = str((GE_ACT_ROOT / stat_path).resolve())
    if manifest_override is not None:
        arguments["manifest_path"] = str(Path(manifest_override).expanduser().resolve())
    dataset_class = _load_class(class_name, class_source)
    return dataset_class(**arguments), config


def load_benchmark_datasets(
    old_config: str | Path,
    hdf5_config: str | Path,
    hdf5_manifest: str | Path,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    old_dataset, old_payload = construct_train_dataset(old_config)
    try:
        hdf5_dataset, hdf5_payload = construct_train_dataset(
            hdf5_config, manifest_override=hdf5_manifest
        )
    except Exception:
        close = getattr(old_dataset, "close", None)
        if callable(close):
            close()
        raise
    return old_dataset, hdf5_dataset, old_payload, hdf5_payload


def run_throughput_benchmarks(
    old_dataset: Any,
    hdf5_dataset: Any,
    plans: Sequence[DeterministicSamplePlan],
    *,
    workers: Sequence[int],
    sample_count: int,
    batch_size: int,
    warmup_batches: int,
    measure_batches: int,
    prefetch_factor: int,
) -> list[dict[str, Any]]:
    """Alternate backend order and ensure their worker lifetimes never overlap."""
    output = []
    for worker_position, worker_count in enumerate(workers):
        order = ["old", "hdf5"] if worker_position % 2 == 0 else ["hdf5", "old"]
        entry: dict[str, Any] = {
            "workers": worker_count,
            "execution_order": order,
            "backends": {},
        }
        for backend in order:
            if backend == "old":
                view = DeterministicOldView(old_dataset, plans, sample_count)
            else:
                close = getattr(hdf5_dataset, "close", None)
                if callable(close):
                    close()
                view = DeterministicHDF5View(hdf5_dataset, plans, sample_count)
            entry["backends"][backend] = measure_dataloader(
                view,
                workers=worker_count,
                batch_size=batch_size,
                warmup_batches=warmup_batches,
                measure_batches=measure_batches,
                prefetch_factor=prefetch_factor,
            )
        output.append(entry)
    return output


def _filesystem_type(path: str | Path) -> str:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        target = target.parent
    if _psutil is not None:
        candidates = []
        for partition in _psutil.disk_partitions(all=True):
            try:
                target.relative_to(Path(partition.mountpoint).resolve())
            except ValueError:
                continue
            candidates.append((len(partition.mountpoint), partition.fstype))
        if candidates:
            return max(candidates)[1] or "unknown"
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return "unknown"
    candidates = []
    for line in lines:
        fields = line.split()
        if "-" not in fields:
            continue
        separator = fields.index("-")
        mountpoint = Path(fields[4].replace("\\040", " "))
        try:
            target.relative_to(mountpoint)
        except ValueError:
            continue
        candidates.append((len(str(mountpoint)), fields[separator + 1]))
    return max(candidates)[1] if candidates else "unknown"


def _git_metadata() -> dict[str, Any]:
    def command(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip()

    try:
        return {
            "sha": command("rev-parse", "HEAD"),
            "dirty": bool(command("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-config", type=Path, required=True)
    parser.add_argument("--hdf5-config", type=Path, required=True)
    parser.add_argument("--hdf5-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("parity", "throughput", "both"), default="both"
    )
    parser.add_argument("--episodes", type=_positive_int, default=64)
    parser.add_argument("--samples", type=_positive_int, default=1024)
    parser.add_argument(
        "--workers", type=_non_negative_int, nargs="+", default=[0, 2, 4, 8]
    )
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--warmup-batches", type=_non_negative_int, default=20)
    parser.add_argument("--measure-batches", type=_positive_int, default=100)
    parser.add_argument("--prefetch-factor", type=_positive_int, default=4)
    parser.add_argument("--run-label", choices=("cold", "warm"), required=True)
    parser.add_argument("--compression", choices=("none", "lzf"), required=True)
    return parser


def _base_report(args: argparse.Namespace) -> dict[str, Any]:
    old_config = args.old_config.expanduser().resolve()
    hdf5_config = args.hdf5_config.expanduser().resolve()
    manifest = args.hdf5_manifest.expanduser().resolve()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "git": _git_metadata(),
        "paths": {
            "old_config": str(old_config),
            "hdf5_config": str(hdf5_config),
            "hdf5_manifest": str(manifest),
            "output_json": str(args.output_json.expanduser().resolve()),
        },
        "mode": args.mode,
        "compression": args.compression,
        "run_label": args.run_label,
        "cache_state_claim": None,
        "arguments": {
            "episodes": args.episodes,
            "samples": args.samples,
            "workers": args.workers,
            "batch_size": args.batch_size,
            "warmup_batches": args.warmup_batches,
            "measure_batches": args.measure_batches,
            "prefetch_factor": args.prefetch_factor,
        },
        "filesystem": {
            "manifest": _filesystem_type(manifest),
            "old_data": {},
            "old_predecoded_video_root": None,
            "hdf5_shards": {},
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = _base_report(args)
    old_dataset = None
    hdf5_dataset = None
    exit_code = 0
    try:
        old_dataset, hdf5_dataset, old_config, _ = load_benchmark_datasets(
            args.old_config, args.hdf5_config, args.hdf5_manifest
        )
        manifest_payload = getattr(hdf5_dataset, "manifest", None)
        if type(manifest_payload) is not dict:
            raise ValueError("HDF5 dataset does not expose its validated manifest")
        if manifest_payload.get("compression") != args.compression:
            raise ValueError(
                "CLI compression does not match manifest compression: "
                f"{args.compression!r} != {manifest_payload.get('compression')!r}"
            )
        old_train_config = old_config.get("data", {}).get("train", {})
        source_roots = old_train_config.get("data_roots", [])
        report["filesystem"]["old_data"] = {
            str(Path(path).expanduser().resolve()): _filesystem_type(path)
            for path in dict.fromkeys(source_roots)
        }
        predecoded_root = old_train_config.get("predecoded_video_root")
        if type(predecoded_root) is str:
            report["filesystem"]["old_predecoded_video_root"] = _filesystem_type(
                predecoded_root
            )
        report["filesystem"]["hdf5_shards"] = {
            str(record.shard_path): _filesystem_type(record.shard_path)
            for record in hdf5_dataset.records
        }
        pairs = map_episode_pairs(
            old_dataset, hdf5_dataset, episode_limit=args.episodes
        )
        plans = build_sample_plans(old_dataset, hdf5_dataset, pairs)
        report["mapping"] = {
            "selected_pairs": len(pairs),
            "requested_episodes": args.episodes,
            "manifest_episode_count": len(hdf5_dataset.records),
            "pairs": [
                {
                    "domain": pair.domain,
                    "episode_index": pair.episode_index,
                    "old_index": pair.old_index,
                    "hdf5_index": pair.hdf5_index,
                    "caption": pair.caption,
                    "length": pair.length,
                }
                for pair in pairs
            ],
        }
        if args.mode in ("parity", "both"):
            report["parity"] = run_parity(old_dataset, hdf5_dataset, plans)
            if not report["parity"]["passed"]:
                exit_code = 1
        if args.mode in ("throughput", "both"):
            report["throughput"] = run_throughput_benchmarks(
                old_dataset,
                hdf5_dataset,
                plans,
                workers=args.workers,
                sample_count=args.samples,
                batch_size=args.batch_size,
                warmup_batches=args.warmup_batches,
                measure_batches=args.measure_batches,
                prefetch_factor=args.prefetch_factor,
            )
    except Exception as error:
        report["fatal_error"] = f"{type(error).__name__}: {error}"
        exit_code = 1
    finally:
        for dataset in (old_dataset, hdf5_dataset):
            close = getattr(dataset, "close", None)
            if callable(close):
                close()
    atomic_write_json(args.output_json, report)
    if exit_code:
        print(f"LIBERO HDF5 benchmark failed; report: {args.output_json}")
    else:
        print(f"LIBERO HDF5 benchmark passed; report: {args.output_json}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
