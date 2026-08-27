"""Training utilities and atomic artifacts for the domain TA-Tok."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import save_file
from torch.utils.data import Dataset

from qwen35_planx.config import CAMERA_NAMES, TATokMetadata
from qwen35_planx.hashing import sha256_file


def load_frame_manifest(
    path: Path | str,
    *,
    expected_split: str,
) -> list[dict[str, object]]:
    """Load and validate one split's predecoded frame records."""

    if expected_split not in {"train", "val"}:
        raise ValueError(f"invalid expected_split: {expected_split!r}")
    path = Path(path)
    rows: list[dict[str, object]] = []
    required = {
        "trajectory_id",
        "suite",
        "split",
        "instruction",
        "camera",
        "frame_index",
        "cache_path",
    }
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = required.difference(row)
            if missing:
                raise ValueError(
                    f"{path}:{line_number} missing fields: "
                    + ", ".join(sorted(missing))
                )
            if row["split"] != expected_split:
                raise ValueError(
                    f"{path}:{line_number} split is {row['split']!r}, "
                    f"expected {expected_split!r}"
                )
            if row["camera"] not in CAMERA_NAMES:
                raise ValueError(
                    f"{path}:{line_number} has invalid camera "
                    f"{row['camera']!r}"
                )
            if int(row["frame_index"]) < 0:
                raise ValueError(
                    f"{path}:{line_number} frame_index must be non-negative"
                )
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"frame manifest is empty: {path}")
    return rows


class FrameManifestDataset(Dataset[dict[str, torch.Tensor]]):
    """Random-access frame dataset backed by episode-level NumPy memmaps."""

    def __init__(self, rows: Sequence[Mapping[str, object]]) -> None:
        self.rows = [dict(row) for row in rows]
        self._arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _array(self, path: str) -> np.ndarray:
        if path not in self._arrays:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                array.ndim != 4
                or array.shape[-1] != 3
                or array.dtype != np.uint8
            ):
                raise ValueError(
                    f"invalid predecoded RGB cache {path}: "
                    f"shape={array.shape}, dtype={array.dtype}"
                )
            self._arrays[path] = array
        return self._arrays[path]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        cache_path = str(row["cache_path"])
        frame_index = int(row["frame_index"])
        array = self._array(cache_path)
        if frame_index >= len(array):
            raise IndexError(
                f"frame_index {frame_index} exceeds cache length {len(array)}"
            )
        image = np.asarray(array[frame_index])
        image_tensor = (
            torch.from_numpy(image.copy())
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(255.0)
        )
        return {
            "image": image_tensor,
            "camera_id": torch.tensor(
                CAMERA_NAMES.index(str(row["camera"])), dtype=torch.long
            ),
            "frame_index": torch.tensor(frame_index, dtype=torch.long),
        }


def validate_split_isolation(
    train_rows: Iterable[Mapping[str, object]],
    val_rows: Iterable[Mapping[str, object]],
) -> None:
    train_ids = {str(row["trajectory_id"]) for row in train_rows}
    val_ids = {str(row["trajectory_id"]) for row in val_rows}
    overlap = sorted(train_ids.intersection(val_ids))
    if overlap:
        raise ValueError(
            "train/validation trajectory overlap: " + ", ".join(overlap[:10])
        )


def summarize_validation(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    """Average scalar validation reports per camera and overall."""

    by_camera: dict[str, list[Mapping[str, object]]] = {
        camera: [] for camera in CAMERA_NAMES
    }
    for row in rows:
        camera = str(row["camera"])
        if camera not in by_camera:
            raise ValueError(f"unknown validation camera: {camera}")
        by_camera[camera].append(row)
    missing = [camera for camera, items in by_camera.items() if not items]
    if missing:
        raise ValueError(
            "validation report is missing camera(s): " + ", ".join(missing)
        )

    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key != "camera" and isinstance(value, (int, float))
        }
    )

    def average(items: Sequence[Mapping[str, object]]) -> dict[str, float]:
        return {
            name: sum(float(item[name]) for item in items if name in item)
            / sum(1 for item in items if name in item)
            for name in metric_names
            if any(name in item for item in items)
        }

    return {
        **{camera: average(items) for camera, items in by_camera.items()},
        "overall": average(rows),
    }


def validate_resume_anchors(
    expected: np.ndarray | Sequence[int],
    restored: np.ndarray | Sequence[int],
) -> None:
    expected_array = np.asarray(expected, dtype=np.int64)
    restored_array = np.asarray(restored, dtype=np.int64)
    if (
        expected_array.shape != restored_array.shape
        or not np.array_equal(expected_array, restored_array)
    ):
        raise ValueError("resume checkpoint anchor mapping does not match")


@dataclass(frozen=True)
class TATokCheckpoint:
    path: Path
    state_hash: str
    metadata: TATokMetadata


def _json_dump(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def save_ta_tok_checkpoint(
    *,
    output_dir: Path | str,
    step: int,
    state_dict: Mapping[str, torch.Tensor],
    metadata: TATokMetadata,
    anchor_token_ids: np.ndarray | Sequence[int],
    metrics: Mapping[str, object],
) -> TATokCheckpoint:
    """Atomically publish a teacher-free, hash-bound TA-Tok checkpoint."""

    if step < 0:
        raise ValueError("step must be non-negative")
    anchors = np.asarray(anchor_token_ids, dtype=np.int64)
    validate_resume_anchors(
        np.asarray(metadata.anchor_token_ids, dtype=np.int64), anchors
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"step_{step:06d}"
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".step_{step:06d}.", dir=output_dir)
    )
    try:
        filtered_state = {
            key: value.detach().cpu().contiguous()
            for key, value in state_dict.items()
            if not key.startswith("teacher.")
        }
        if not filtered_state:
            raise ValueError("checkpoint has no trainable TA-Tok tensors")
        state_path = temporary / "ta_tok.safetensors"
        save_file(filtered_state, state_path)
        state_hash = sha256_file(state_path)
        saved_metadata = replace(metadata, state_hash=state_hash)
        _json_dump(temporary / "metadata.json", saved_metadata.to_dict())
        with (temporary / "anchor_ids.npy").open("wb") as handle:
            np.save(handle, anchors, allow_pickle=False)
        _json_dump(temporary / "metrics.json", metrics)
        os.replace(temporary, destination)

        latest_temporary = output_dir / f".latest_checkpoint.{os.getpid()}.tmp"
        latest_temporary.write_text(destination.name + "\n", encoding="utf-8")
        os.replace(latest_temporary, output_dir / "latest_checkpoint.txt")
        return TATokCheckpoint(
            path=destination,
            state_hash=state_hash,
            metadata=saved_metadata,
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
