"""Independent persistent-handle reader for fixed LIBERO FastWAM HDF5 shards."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from ge_act.data.libero_fastwam_hdf5_schema import (
    EpisodeRecord,
    load_manifest,
    validate_episode_group,
)


FIXED_CAMERAS = [
    "observation.images.image",
    "observation.images.wrist_image",
]
BATON_SAMPLING_ALGORITHM = "libero_fastwam_hdf5_stateless_sha256"
BATON_SAMPLING_VERSION = 1


def _clip_integer_indexes(
    indexes: Sequence[int] | np.ndarray,
    *,
    length: int,
    name: str,
) -> np.ndarray:
    requested = np.asarray(indexes)
    if requested.ndim != 1 or requested.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D integer sequence")
    if requested.dtype.kind not in "iu" or requested.dtype.kind == "b":
        raise TypeError(f"{name} must be a non-empty 1-D integer sequence")
    if type(length) is not int or length <= 0 or length > np.iinfo(np.int64).max:
        raise ValueError(
            f"length must be an integer in [1, {np.iinfo(np.int64).max}], "
            f"got {length!r}"
        )

    if requested.dtype.kind == "u":
        clipped = np.minimum(requested, length - 1)
        return clipped.astype(np.int64, copy=False)
    signed = requested.astype(np.int64, copy=False)
    return np.clip(signed, 0, length - 1)


def read_rows_preserving_order(
    dataset: Any,
    indexes: Sequence[int] | np.ndarray,
    *,
    length: int | None = None,
) -> np.ndarray:
    """Read sorted unique rows once, then restore clipped order and repeats."""
    dataset_length = int(dataset.shape[0])
    if length is None:
        length = dataset_length
    if type(length) is not int or length <= 0 or length > dataset_length:
        raise ValueError(
            f"length must be an integer in [1, {dataset_length}], got {length!r}"
        )

    clipped = _clip_integer_indexes(indexes, length=length, name="indexes")
    unique_rows = np.unique(clipped)
    values = np.asarray(dataset[unique_rows])
    gather = np.searchsorted(unique_rows, clipped)
    return values[gather]


class LiberoFastWAMHDF5Dataset(Dataset):
    """Fixed-contract LIBERO dataset backed by immutable HDF5 episode shards."""

    def __init__(
        self,
        manifest_path: str | Path,
        stat_file: str | Path,
        source_fps: int = 20,
        sample_n_frames: int = 500,
        valid_cam: Sequence[str] = FIXED_CAMERAS,
        chunk: int = 9,
        action_chunk: int = 36,
        n_previous: int = 4,
        previous_pick_mode: str = "random",
        action_type: str = "absolute",
        action_space: str = "eef",
        train_dataset: bool = True,
        fix_epiidx: int | None = None,
        fix_sidx: int | None = None,
        fix_mem_idx: Sequence[int] | None = None,
        max_open_shards: int = 8,
        ignore_seek: bool = False,
        baton_sampling_algorithm: str | None = None,
        baton_sampling_version: int | None = None,
        baton_sampling_seed: int | None = None,
    ) -> None:
        self._validate_fixed_arguments(
            source_fps=source_fps,
            valid_cam=valid_cam,
            chunk=chunk,
            action_chunk=action_chunk,
            n_previous=n_previous,
            action_type=action_type,
            action_space=action_space,
            ignore_seek=ignore_seek,
        )
        if type(sample_n_frames) is not int or sample_n_frames <= action_chunk:
            raise ValueError("sample_n_frames must be an integer greater than 36")
        if previous_pick_mode not in ("random", "uniform"):
            raise ValueError("previous_pick_mode must be 'random' or 'uniform'")
        if type(train_dataset) is not bool:
            raise ValueError("train_dataset must be a bool")
        if type(max_open_shards) is not int or max_open_shards <= 0:
            raise ValueError("max_open_shards must be a positive integer")
        self._validate_fixed_indexes(fix_sidx, fix_mem_idx)
        if fix_epiidx is not None and type(fix_epiidx) is not int:
            raise ValueError("fix_epiidx must be an integer or None")
        sampling_values = (
            baton_sampling_algorithm,
            baton_sampling_version,
            baton_sampling_seed,
        )
        if any(value is not None for value in sampling_values):
            if not all(value is not None for value in sampling_values):
                raise ValueError(
                    "Baton stateless sampling fields must be provided together"
                )
            if baton_sampling_algorithm != BATON_SAMPLING_ALGORITHM:
                raise ValueError(
                    "baton_sampling_algorithm must be "
                    f"{BATON_SAMPLING_ALGORITHM!r}"
                )
            if (
                type(baton_sampling_version) is not int
                or baton_sampling_version != BATON_SAMPLING_VERSION
            ):
                raise ValueError(
                    f"baton_sampling_version must be {BATON_SAMPLING_VERSION}"
                )
            if type(baton_sampling_seed) is not int or baton_sampling_seed < 0:
                raise ValueError(
                    "baton_sampling_seed must be a non-negative integer"
                )

        self.manifest_path = Path(manifest_path)
        self.manifest, self.records = load_manifest(self.manifest_path)
        self.stat_file = Path(stat_file)
        self.action_mean, self.action_std, self.state_mean, self.state_std = (
            self._load_statistics(self.stat_file, self.records)
        )

        self.source_fps = source_fps
        self.sample_n_frames = sample_n_frames
        self.valid_cam = list(valid_cam)
        self.chunk = chunk
        self.action_chunk = action_chunk
        self.n_previous = n_previous
        self.video_temporal_stride = action_chunk // chunk
        self.previous_pick_mode = previous_pick_mode
        self.action_type = action_type
        self.action_space = action_space
        self.train_dataset = train_dataset
        self.fix_epiidx = fix_epiidx
        self.fix_sidx = fix_sidx
        self.fix_mem_idx = None if fix_mem_idx is None else list(fix_mem_idx)
        self.max_open_shards = max_open_shards
        self.ignore_seek = ignore_seek
        self._baton_sampling_contract = (
            None
            if baton_sampling_algorithm is None
            else {
                "algorithm": baton_sampling_algorithm,
                "version": baton_sampling_version,
                "seed": baton_sampling_seed,
            }
        )
        self._baton_sampling_epoch = (
            None
            if self._baton_sampling_contract is None
            else torch.zeros((), dtype=torch.int64).share_memory_()
        )

        self._handles: OrderedDict[Path, h5py.File] = OrderedDict()
        self._handle_pid: int | None = os.getpid()

    @staticmethod
    def _validate_fixed_arguments(**arguments: Any) -> None:
        expected = {
            "source_fps": 20,
            "valid_cam": FIXED_CAMERAS,
            "chunk": 9,
            "action_chunk": 36,
            "n_previous": 4,
            "action_type": "absolute",
            "action_space": "eef",
            "ignore_seek": False,
        }
        for field, expected_value in expected.items():
            actual = arguments[field]
            if type(expected_value) is list:
                matches = (
                    type(actual) in (list, tuple) and list(actual) == expected_value
                )
            else:
                matches = (
                    type(actual) is type(expected_value) and actual == expected_value
                )
            if not matches:
                raise ValueError(
                    f"{field} must be fixed at {expected_value!r}, got {actual!r}"
                )

    @staticmethod
    def _validate_fixed_indexes(
        fix_sidx: int | None, fix_mem_idx: Sequence[int] | None
    ) -> None:
        if (fix_sidx is None) != (fix_mem_idx is None):
            raise ValueError("fix_sidx and fix_mem_idx must be provided together")
        if fix_sidx is None:
            return
        if type(fix_sidx) is not int:
            raise ValueError("fix_sidx must be an integer")
        if type(fix_mem_idx) not in (list, tuple) or len(fix_mem_idx) != 4:
            raise ValueError("fix_mem_idx must be an integer sequence of length 4")
        if any(type(index) is not int for index in fix_mem_idx):
            raise ValueError("fix_mem_idx must be an integer sequence of length 4")

    @staticmethod
    def _load_statistics(
        path: Path, records: Sequence[EpisodeRecord]
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if type(payload) is not dict:
            raise ValueError("statistics payload must be a dict")

        action_mean: dict[str, torch.Tensor] = {}
        action_std: dict[str, torch.Tensor] = {}
        state_mean: dict[str, torch.Tensor] = {}
        state_std: dict[str, torch.Tensor] = {}
        for domain in dict.fromkeys(record.domain for record in records):
            action_key = f"{domain}_eef"
            state_key = f"{domain}_state_eef"
            action_mean[domain] = LiberoFastWAMHDF5Dataset._stat_tensor(
                payload, action_key, "mean", 7
            )
            action_std[domain] = (
                LiberoFastWAMHDF5Dataset._stat_tensor(payload, action_key, "std", 7)
                + 1e-6
            )
            state_mean[domain] = LiberoFastWAMHDF5Dataset._stat_tensor(
                payload, state_key, "mean", 8
            )
            state_std[domain] = (
                LiberoFastWAMHDF5Dataset._stat_tensor(payload, state_key, "std", 8)
                + 1e-6
            )
        return action_mean, action_std, state_mean, state_std

    @staticmethod
    def _stat_tensor(
        payload: dict[str, Any], key: str, field: str, width: int
    ) -> torch.Tensor:
        section = payload.get(key)
        if type(section) is not dict:
            raise ValueError(f"statistics must contain dict {key}")
        values = section.get(field)
        if type(values) is not list or len(values) != width:
            raise ValueError(f"statistics {key} {field} must have width {width}")
        if any(type(value) not in (int, float) for value in values):
            raise ValueError(f"statistics {key} {field} must contain numbers")
        array = np.asarray(values, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"statistics {key} {field} must contain finite numbers")
        return torch.from_numpy(array).unsqueeze(0)

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_index(self, index: int) -> int:
        if type(index) is not int:
            raise TypeError("dataset index must be an integer")
        if index < 0:
            index += len(self.records)
        if index < 0 or index >= len(self.records):
            raise IndexError(f"dataset index out of range: {index}")
        return index

    @property
    def baton_sampling_contract(self) -> dict[str, Any] | None:
        if self._baton_sampling_contract is None:
            return None
        return dict(self._baton_sampling_contract)

    @property
    def baton_sampling_epoch(self) -> int | None:
        if self._baton_sampling_epoch is None:
            return None
        return int(self._baton_sampling_epoch.item())

    def set_epoch(self, epoch: int) -> None:
        if self._baton_sampling_epoch is None:
            return
        if type(epoch) is not int or epoch < 0:
            raise ValueError("Baton sampling epoch must be a non-negative integer")
        self._baton_sampling_epoch.fill_(epoch)

    def _item_rngs(
        self,
        *,
        index: int,
        record: EpisodeRecord,
    ) -> tuple[random.Random, np.random.Generator]:
        contract = self._baton_sampling_contract
        epoch = self.baton_sampling_epoch
        if contract is None or epoch is None:
            raise RuntimeError("Baton stateless sampling contract is disabled")
        identity = {
            "algorithm": contract["algorithm"],
            "version": contract["version"],
            "seed": contract["seed"],
            "epoch": epoch,
            "dataset_index": index,
            "record_key": record.key,
            "domain": record.domain,
            "episode_index": record.episode_index,
            "length": record.length,
        }
        material = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        python_seed = int.from_bytes(
            hashlib.sha256(material + b"\\0python").digest(),
            byteorder="big",
        )
        numpy_seed = int.from_bytes(
            hashlib.sha256(material + b"\\0numpy").digest(),
            byteorder="big",
        )
        return random.Random(python_seed), np.random.default_rng(numpy_seed)

    def get_frame_indexes(
        self,
        total_frames: int,
        *,
        python_rng: random.Random | None = None,
        numpy_rng: np.random.Generator | None = None,
    ) -> tuple[list[int], list[int]]:
        """Match the fixed LIBERO history/future sampling contract."""
        if type(total_frames) is not int or total_frames < 2:
            raise ValueError("total_frames must be an integer of at least 2")

        if self.fix_sidx is not None:
            action_future = list(
                range(self.fix_sidx, self.fix_sidx + self.action_chunk)
            )
            frame_future = action_future[:: self.video_temporal_stride]
            action_future = np.clip(action_future, 0, total_frames - 1).tolist()
            frame_future = np.clip(frame_future, 0, total_frames - 1).tolist()
            memories = np.clip(self.fix_mem_idx, 0, total_frames - 1).tolist()
            return memories + frame_future, memories + action_future

        chunk_end = (
            random.randint(self.action_chunk, total_frames + self.action_chunk)
            if python_rng is None
            else python_rng.randint(
                self.action_chunk,
                total_frames + self.action_chunk,
            )
        )
        indexes_start = max(-self.n_previous, chunk_end - self.sample_n_frames)
        indexes = np.arange(indexes_start, chunk_end)
        indexes = np.clip(indexes, 1, total_frames - 1).tolist()
        video_end = indexes[-self.action_chunk :]
        memory_candidates = indexes[: -self.action_chunk]
        if len(memory_candidates) < self.n_previous - 1:
            memory_candidates = [1] * (self.n_previous - 1) + memory_candidates

        if self.previous_pick_mode == "uniform":
            memories = [
                memory_candidates[int(index)]
                for index in np.linspace(
                    0, len(memory_candidates) - 1, self.n_previous
                ).tolist()
            ]
        else:
            choice_rng = np.random if numpy_rng is None else numpy_rng
            choices = choice_rng.choice(
                list(range(0, len(memory_candidates) - 1)),
                size=self.n_previous - 1,
                replace=False,
            )
            memories = [memory_candidates[index] for index in sorted(choices.tolist())]
            memories.append(memory_candidates[-1])

        frame_indexes = (
            memories
            + video_end[self.video_temporal_stride - 1 :: self.video_temporal_stride]
        )
        action_indexes = memories + video_end
        return frame_indexes, action_indexes

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        selected = self.fix_epiidx if self.fix_epiidx is not None else index
        selected = self._resolve_index(selected)
        record = self.records[selected]
        if self._baton_sampling_contract is None:
            frame_indexes, action_indexes = self.get_frame_indexes(record.length)
        else:
            python_rng, numpy_rng = self._item_rngs(
                index=selected,
                record=record,
            )
            frame_indexes, action_indexes = self.get_frame_indexes(
                record.length,
                python_rng=python_rng,
                numpy_rng=numpy_rng,
            )
        return self.read_by_indexes(selected, frame_indexes, action_indexes)

    def read_by_indexes(
        self,
        index: int,
        frame_indexes: Sequence[int],
        action_indexes: Sequence[int],
    ) -> dict[str, torch.Tensor | str]:
        """Read one manifest record by explicit, clipped row indexes."""
        selected = self._resolve_index(index)
        record = self.records[selected]
        frame_list = self._validated_index_list(
            frame_indexes, "frame_indexes", record.length
        )
        action_list = self._validated_index_list(
            action_indexes, "action_indexes", record.length
        )
        if len(action_list) < self.n_previous:
            raise ValueError(
                f"action_indexes must contain at least {self.n_previous} rows"
            )

        try:
            handle = self._get_handle(record.shard_path)
            if record.group not in handle:
                raise KeyError(f"missing episode group {record.group}")
            group = handle[record.group]
            validate_episode_group(group, record)
            main = read_rows_preserving_order(
                group["rgb_main"], frame_list, length=record.length
            )
            wrist = read_rows_preserving_order(
                group["rgb_wrist"], frame_list, length=record.length
            )
            actions = read_rows_preserving_order(
                group["action"], action_list, length=record.length
            )
            states = read_rows_preserving_order(
                group["state"], action_list, length=record.length
            )
            caption = group["caption"].asstr()[()]
        except Exception as error:
            worker = self._worker_label()
            raise RuntimeError(
                "HDF5 read failed: "
                f"worker={worker} shard={record.shard_path} episode={record.key} "
                f"frame_indexes={frame_list} action_indexes={action_list}: {error}"
            ) from error

        video = torch.from_numpy(np.stack([main, wrist], axis=0))
        video = video.permute(4, 0, 1, 2, 3).contiguous().to(torch.float32)
        video = video.div_(255.0).sub_(0.5).div_(0.5)
        action = torch.from_numpy(np.asarray(actions, dtype=np.float32))
        state_sequence = torch.from_numpy(np.asarray(states, dtype=np.float32))
        state = state_sequence[self.n_previous - 1 : self.n_previous]
        action = (action - self.action_mean[record.domain]) / self.action_std[
            record.domain
        ]
        state = (state - self.state_mean[record.domain]) / self.state_std[record.domain]
        return {
            "video": video,
            "actions": action,
            "caption": caption,
            "state": state,
        }

    @staticmethod
    def _validated_index_list(
        indexes: Sequence[int], name: str, length: int
    ) -> list[int]:
        return _clip_integer_indexes(indexes, length=length, name=name).tolist()

    def _get_handle(self, path: Path) -> h5py.File:
        current_pid = os.getpid()
        if self._handle_pid != current_pid:
            self.close()
            self._handle_pid = current_pid

        handle = self._handles.pop(path, None)
        if handle is not None and not handle.id.valid:
            handle.close()
            handle = None
        if handle is None:
            handle = h5py.File(path, "r")
        self._handles[path] = handle
        while len(self._handles) > self.max_open_shards:
            _, evicted = self._handles.popitem(last=False)
            evicted.close()
        return handle

    @staticmethod
    def _worker_label() -> int | str:
        info = get_worker_info()
        return info.id if info is not None else "main"

    def close(self) -> None:
        handles = getattr(self, "_handles", None)
        if handles is None:
            return
        while handles:
            _, handle = handles.popitem(last=False)
            try:
                handle.close()
            except Exception:
                pass

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handles"] = OrderedDict()
        state["_handle_pid"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
