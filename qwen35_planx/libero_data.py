"""Trajectory-safe LIBERO data discovery over predecoded RGB caches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

from qwen35_planx.config import (
    CAMERA_KEYS,
    CAMERA_NAMES,
    PlanGeometry,
)


def trajectory_split(trajectory_id: str, *, seed: int) -> str:
    """Assign an entire trajectory to a deterministic 95/5 train/val split."""

    bucket = int.from_bytes(
        hashlib.sha256(f"{seed}:{trajectory_id}".encode("utf-8")).digest()[:8],
        "big",
    )
    return "val" if bucket % 20 == 0 else "train"


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str
    suite: str
    episode_index: int
    instruction: str
    num_frames: int
    split: str
    camera_cache_paths: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.trajectory_id:
            raise ValueError("trajectory_id must not be empty")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if self.split not in {"train", "val"}:
            raise ValueError(f"invalid split: {self.split!r}")
        if tuple(self.camera_cache_paths) != CAMERA_NAMES:
            raise ValueError(
                "camera_cache_paths must use canonical main/wrist order"
            )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["camera_cache_paths"] = dict(self.camera_cache_paths)
        return payload


@dataclass(frozen=True)
class FrameRecord:
    trajectory_id: str
    suite: str
    split: str
    instruction: str
    camera: str
    frame_index: int
    cache_path: str

    def __post_init__(self) -> None:
        if self.camera not in CAMERA_NAMES:
            raise ValueError(f"unknown camera: {self.camera!r}")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PlannerWindowRecord:
    trajectory_id: str
    suite: str
    split: str
    instruction: str
    num_frames: int
    current_index: int
    future_indices: tuple[int, ...]
    camera_cache_paths: tuple[str, str]

    def __post_init__(self) -> None:
        geometry = PlanGeometry()
        if self.current_index < 0:
            raise ValueError("current_index must be non-negative")
        if not self.future_indices or max(self.future_indices) >= self.num_frames:
            raise ValueError(
                "planner window extends past the end of the trajectory"
            )
        expected = tuple(
            self.current_index + offset
            for offset in geometry.future_frame_offsets
        )
        if tuple(self.future_indices) != expected:
            raise ValueError(
                f"future_indices must use exact offsets "
                f"{geometry.future_frame_offsets}, got {self.future_indices}"
            )
        if len(self.camera_cache_paths) != len(CAMERA_NAMES):
            raise ValueError("camera_cache_paths must contain main and wrist")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _load_rgb_cache(path: Path, *, camera: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing {camera} RGB cache: {path}")
    try:
        frames = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"failed to load {camera} RGB cache {path}: {exc}") from exc
    if (
        frames.ndim != 4
        or frames.shape[-1] != 3
        or frames.dtype != np.uint8
        or frames.shape[0] == 0
    ):
        raise ValueError(
            f"invalid {camera} RGB cache {path}: expected nonempty "
            f"[T,H,W,3] uint8, got shape={frames.shape}, dtype={frames.dtype}"
        )
    return frames


def _find_camera_cache(
    *,
    predecoded_root: Path,
    domain: str,
    camera_name: str,
    camera_key: str,
    episode_index: int,
) -> Path:
    episode_name = f"episode_{episode_index:06d}.npy"
    root = predecoded_root / domain / "videos"
    matches = sorted(root.glob(f"chunk-*/{camera_key}/{episode_name}"))
    if not matches:
        raise FileNotFoundError(
            f"missing {camera_name} RGB cache for {domain}:{episode_index}: "
            f"{root}/chunk-*/{camera_key}/{episode_name}"
        )
    if len(matches) != 1:
        raise ValueError(
            f"multiple {camera_name} RGB caches for {domain}:{episode_index}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def _read_episode_rows(path: Path) -> Iterator[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing episode metadata: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {path}:{line_number}: {exc}"
                ) from exc
            required = {"episode_index", "tasks", "length"}
            missing = required.difference(row)
            if missing:
                raise ValueError(
                    f"missing fields in {path}:{line_number}: "
                    + ", ".join(sorted(missing))
                )
            yield row


def discover_trajectories(
    *,
    dataset_roots: Sequence[Path | str],
    domains: Sequence[str],
    predecoded_root: Path | str,
    split_seed: int,
) -> list[TrajectoryRecord]:
    """Discover and validate both predecoded cameras for every episode."""

    if not dataset_roots or len(dataset_roots) != len(domains):
        raise ValueError("dataset_roots and domains must be nonempty aligned lists")

    predecoded_root = Path(predecoded_root)
    records: list[TrajectoryRecord] = []
    seen_ids: set[str] = set()
    for dataset_root_value, domain in zip(dataset_roots, domains):
        dataset_root = Path(dataset_root_value)
        metadata_path = dataset_root / domain / "meta" / "episodes.jsonl"
        for row in _read_episode_rows(metadata_path):
            episode_index = int(row["episode_index"])
            trajectory_id = f"{domain}:{episode_index:06d}"
            if trajectory_id in seen_ids:
                raise ValueError(f"duplicate trajectory_id: {trajectory_id}")
            seen_ids.add(trajectory_id)

            tasks = row["tasks"]
            if not isinstance(tasks, list) or len(tasks) != 1 or not tasks[0]:
                raise ValueError(
                    f"{trajectory_id} must contain exactly one instruction"
                )
            expected_frames = int(row["length"])
            paths: dict[str, str] = {}
            frame_counts: dict[str, int] = {}
            for camera_name, camera_key in zip(CAMERA_NAMES, CAMERA_KEYS):
                path = _find_camera_cache(
                    predecoded_root=predecoded_root,
                    domain=domain,
                    camera_name=camera_name,
                    camera_key=camera_key,
                    episode_index=episode_index,
                )
                frames = _load_rgb_cache(path, camera=camera_name)
                paths[camera_name] = str(path.resolve())
                frame_counts[camera_name] = int(frames.shape[0])

            if len(set(frame_counts.values())) != 1:
                raise ValueError(
                    f"{trajectory_id} camera frame count mismatch: {frame_counts}"
                )
            actual_frames = frame_counts["main"]
            if actual_frames != expected_frames:
                raise ValueError(
                    f"{trajectory_id} metadata/cache frame count mismatch: "
                    f"metadata={expected_frames}, cache={actual_frames}"
                )

            records.append(
                TrajectoryRecord(
                    trajectory_id=trajectory_id,
                    suite=domain,
                    episode_index=episode_index,
                    instruction=str(tasks[0]),
                    num_frames=actual_frames,
                    split=trajectory_split(trajectory_id, seed=split_seed),
                    camera_cache_paths=paths,
                )
            )
    return sorted(records, key=lambda record: record.trajectory_id)


def load_predecoded_frames(
    record: TrajectoryRecord,
) -> dict[str, np.ndarray]:
    """Memory-map a trajectory's two camera caches in canonical order."""

    loaded = {
        camera: _load_rgb_cache(
            Path(record.camera_cache_paths[camera]), camera=camera
        )
        for camera in CAMERA_NAMES
    }
    counts = {camera: len(frames) for camera, frames in loaded.items()}
    if len(set(counts.values())) != 1 or counts["main"] != record.num_frames:
        raise ValueError(
            f"{record.trajectory_id} camera frame count changed: {counts}"
        )
    return loaded


def iter_all_camera_frames(
    record: TrajectoryRecord,
) -> Iterator[FrameRecord]:
    """Yield every frame from both cameras without crossing trajectories."""

    for frame_index in range(record.num_frames):
        for camera in CAMERA_NAMES:
            yield FrameRecord(
                trajectory_id=record.trajectory_id,
                suite=record.suite,
                split=record.split,
                instruction=record.instruction,
                camera=camera,
                frame_index=frame_index,
                cache_path=record.camera_cache_paths[camera],
            )


def iter_planner_windows(
    record: TrajectoryRecord,
    *,
    stride: int = 10,
    max_windows: int = 16,
    geometry: PlanGeometry | None = None,
) -> Iterator[PlannerWindowRecord]:
    """Yield bounded K4 windows using the exact production future offsets."""

    if stride <= 0:
        raise ValueError("stride must be positive")
    if max_windows <= 0:
        raise ValueError("max_windows must be positive")
    geometry = geometry or PlanGeometry()
    last_offset = geometry.future_frame_offsets[-1]
    last_current_index = record.num_frames - last_offset - 1
    if last_current_index < 0:
        return

    for window_number, current_index in enumerate(
        range(0, last_current_index + 1, stride)
    ):
        if window_number >= max_windows:
            break
        yield PlannerWindowRecord(
            trajectory_id=record.trajectory_id,
            suite=record.suite,
            split=record.split,
            instruction=record.instruction,
            num_frames=record.num_frames,
            current_index=current_index,
            future_indices=tuple(
                current_index + offset
                for offset in geometry.future_frame_offsets
            ),
            camera_cache_paths=tuple(
                record.camera_cache_paths[camera] for camera in CAMERA_NAMES
            ),
        )
