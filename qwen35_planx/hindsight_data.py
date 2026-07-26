"""Deterministic GE-Act HDF5 windows for offline hindsight teachers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from ge_act.data.libero_fastwam_hdf5_schema import (
    EpisodeRecord,
    load_manifest,
    validate_episode_group,
)
from qwen35_planx.config import CAMERA_NAMES, PlanGeometry
from qwen35_planx.libero_data import trajectory_split


_N_PREVIOUS = 4
_ACTION_CHUNK = 36
_VIDEO_TEMPORAL_STRIDE = 4
_VIDEO_CHUNK = 9


def _integer_tuple(values: Sequence[int], *, name: str, length: int) -> tuple[int, ...]:
    array = np.asarray(values)
    if (
        array.ndim != 1
        or len(array) != length
        or array.dtype.kind not in "iu"
        or array.dtype.kind == "b"
    ):
        raise ValueError(f"{name} must contain exactly {length} integers")
    result = tuple(int(value) for value in array)
    if any(value < 0 for value in result):
        raise ValueError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True)
class HindsightWindowRecord:
    """One explicit, reproducible GE-Act planner window."""

    sample_id: str
    episode_key: str
    split: str
    caption: str
    current_index: int
    future_indices: tuple[int, int, int, int]
    frame_indices: Sequence[int]
    action_indices: Sequence[int]
    camera_names: tuple[str, str] = CAMERA_NAMES

    def __post_init__(self) -> None:
        if not self.sample_id or not self.episode_key:
            raise ValueError("sample_id and episode_key must not be empty")
        if self.split not in {"train", "val"}:
            raise ValueError(f"invalid split: {self.split!r}")
        if type(self.caption) is not str or not self.caption:
            raise ValueError("caption must be a non-empty string")
        if tuple(self.camera_names) != CAMERA_NAMES:
            raise ValueError("camera_names must use canonical main/wrist order")

        frames = _integer_tuple(self.frame_indices, name="frame_indices", length=13)
        actions = _integer_tuple(self.action_indices, name="action_indices", length=40)
        futures = _integer_tuple(self.future_indices, name="future_indices", length=4)
        expected_futures = tuple(
            frames[4 + index] for index in PlanGeometry().ge_act_future_indices
        )
        if self.current_index != frames[_N_PREVIOUS - 1]:
            raise ValueError("current_index must be the final memory frame")
        if futures != expected_futures:
            raise ValueError("future_indices must select the exact GE-Act keyframes")
        if frames[:_N_PREVIOUS] != actions[:_N_PREVIOUS]:
            raise ValueError("frame/action memory indices must match")
        if frames[_N_PREVIOUS:] != actions[_N_PREVIOUS + 3 :: 4]:
            raise ValueError("frame/action future indices must use temporal stride 4")

        object.__setattr__(self, "frame_indices", frames)
        object.__setattr__(self, "action_indices", actions)
        object.__setattr__(self, "future_indices", futures)
        object.__setattr__(self, "camera_names", tuple(self.camera_names))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frame_indices"] = list(self.frame_indices)
        payload["action_indices"] = list(self.action_indices)
        payload["future_indices"] = list(self.future_indices)
        payload["camera_names"] = list(self.camera_names)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HindsightWindowRecord:
        required = {
            "sample_id",
            "episode_key",
            "split",
            "caption",
            "current_index",
            "future_indices",
            "frame_indices",
            "action_indices",
            "camera_names",
        }
        if type(payload) is not dict or set(payload) != required:
            raise ValueError(
                "window record fields must be exactly " + ", ".join(sorted(required))
            )
        return cls(**{name: payload[name] for name in required})


@dataclass(frozen=True)
class HDF5Trajectory:
    """A validated full dual-camera HDF5 trajectory."""

    rgb: np.ndarray
    actions: np.ndarray
    states: np.ndarray

    def __post_init__(self) -> None:
        if (
            self.rgb.ndim != 5
            or self.rgb.shape[0] != 2
            or self.rgb.shape[2:] != (256, 256, 3)
            or self.rgb.dtype != np.uint8
        ):
            raise ValueError("rgb must have shape [2,T,256,256,3] and dtype uint8")
        length = self.rgb.shape[1]
        if self.actions.shape != (length, 7) or self.actions.dtype != np.float32:
            raise ValueError("actions must have shape [T,7] and dtype float32")
        if self.states.shape != (length, 8) or self.states.dtype != np.float32:
            raise ValueError("states must have shape [T,8] and dtype float32")


def _uniform_window_indices(
    *,
    start_index: int,
    total_frames: int,
    sample_n_frames: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Reproduce GE-Act uniform history selection without global RNG state."""

    chunk_end = start_index + _ACTION_CHUNK
    indexes_start = max(-_N_PREVIOUS, chunk_end - sample_n_frames)
    indexes = np.arange(indexes_start, chunk_end, dtype=np.int64)
    indexes = np.clip(indexes, 1, total_frames - 1).tolist()
    video_end = indexes[-_ACTION_CHUNK:]
    memory_candidates = indexes[:-_ACTION_CHUNK]
    if len(memory_candidates) < _N_PREVIOUS - 1:
        memory_candidates = [1] * (_N_PREVIOUS - 1) + memory_candidates
    memories = [
        memory_candidates[int(index)]
        for index in np.linspace(0, len(memory_candidates) - 1, _N_PREVIOUS).tolist()
    ]
    frames = memories + video_end[_VIDEO_TEMPORAL_STRIDE - 1 :: _VIDEO_TEMPORAL_STRIDE]
    actions = memories + video_end
    return tuple(frames), tuple(actions)


def build_fixed_windows(
    manifest_path: Path | str,
    *,
    split_seed: int,
    window_stride: int = _ACTION_CHUNK,
    sample_n_frames: int = 500,
) -> list[HindsightWindowRecord]:
    """Build sorted, trajectory-local windows with explicit GE-Act indices."""

    if type(split_seed) is not int:
        raise TypeError("split_seed must be an integer")
    if type(window_stride) is not int or window_stride <= 0:
        raise ValueError("window_stride must be a positive integer")
    if type(sample_n_frames) is not int or sample_n_frames <= _ACTION_CHUNK:
        raise ValueError("sample_n_frames must be an integer greater than 36")

    manifest, records = load_manifest(Path(manifest_path))
    expected_contract = {
        "n_previous": _N_PREVIOUS,
        "chunk": _VIDEO_CHUNK,
        "action_chunk": _ACTION_CHUNK,
        "camera_names": list(CAMERA_NAMES),
    }
    for field, expected in expected_contract.items():
        if manifest[field] != expected:
            raise ValueError(
                f"HDF5 manifest {field} must be {expected!r}, got {manifest[field]!r}"
            )

    windows: list[HindsightWindowRecord] = []
    geometry = PlanGeometry()
    for record in sorted(records, key=lambda item: item.key):
        split = trajectory_split(record.key, seed=split_seed)
        for start_index in range(0, record.length, window_stride):
            frame_indices, action_indices = _uniform_window_indices(
                start_index=start_index,
                total_frames=record.length,
                sample_n_frames=sample_n_frames,
            )
            windows.append(
                HindsightWindowRecord(
                    sample_id=f"{record.key}:{start_index:09d}",
                    episode_key=record.key,
                    split=split,
                    caption=record.caption,
                    current_index=frame_indices[_N_PREVIOUS - 1],
                    future_indices=tuple(
                        frame_indices[4 + index]
                        for index in geometry.ge_act_future_indices
                    ),
                    frame_indices=frame_indices,
                    action_indices=action_indices,
                )
            )
    return windows


def read_full_trajectory(record: EpisodeRecord) -> HDF5Trajectory:
    """Validate an HDF5 episode group before reading its complete trajectory."""

    if not isinstance(record, EpisodeRecord):
        raise TypeError("record must be an EpisodeRecord")
    with h5py.File(record.shard_path, "r") as handle:
        if record.group not in handle:
            raise ValueError(f"missing episode group: {record.group}")
        group = handle[record.group]
        validate_episode_group(group, record)
        main = np.asarray(group["rgb_main"][:], dtype=np.uint8)
        wrist = np.asarray(group["rgb_wrist"][:], dtype=np.uint8)
        actions = np.asarray(group["action"][:], dtype=np.float32)
        states = np.asarray(group["state"][:], dtype=np.float32)
    return HDF5Trajectory(
        rgb=np.stack((main, wrist), axis=0),
        actions=actions,
        states=states,
    )
