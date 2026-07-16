from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
GE_ACT_ROOT = REPO_ROOT / "ge_act"


def test_stable_libero_source_files_are_vendored() -> None:
    required = [
        "data/__init__.py",
        "data/lerobot_like_dataset.py",
        "data/libero_dataset.py",
        "data/utils/__init__.py",
        "data/utils/statistics.py",
        "data/utils/utils.py",
        "experiments/__init__.py",
        "experiments/eval_libero.py",
        "experiments/eval_libero_plus.py",
        "experiments/eval_libero_official.py",
        "experiments/eval_libero_plus_official.py",
        "configs/ltx_model/libero/action_model_libero_official_eval.yaml",
        "configs/ltx_model/libero/action_model_libero_official_localeval.yaml",
        "scripts/train.sh",
        "scripts/predecode_lerobot_videos.py",
        "scripts/sbatch_train_ltx_siglip2_hpc3.sh",
        "requirements.txt",
    ]

    missing = [path for path in required if not (GE_ACT_ROOT / path).is_file()]
    assert not missing, f"Missing stable GE-Act files: {missing}"


def test_lerobot_dataset_keeps_two_camera_video_contract() -> None:
    sys.path.insert(0, str(GE_ACT_ROOT))
    try:
        module = importlib.import_module("data.lerobot_like_dataset")
    finally:
        sys.path.pop(0)

    dataset_cls = module.CustomLeRobotDataset
    assert dataset_cls.__name__ == "CustomLeRobotDataset"
    assert hasattr(dataset_cls, "seek_mp4")
    assert hasattr(dataset_cls, "get_frame_indexes")
    assert hasattr(dataset_cls, "normalize_video")


def test_lerobot_dataset_accepts_source_fps_metadata() -> None:
    sys.path.insert(0, str(GE_ACT_ROOT))
    try:
        module = importlib.import_module("data.lerobot_like_dataset")
    finally:
        sys.path.pop(0)

    dataset = module.CustomLeRobotDataset(
        data_roots=[],
        domains=[],
        source_fps=20,
        chunk=9,
        action_chunk=36,
        n_previous=4,
        random_crop=False,
    )

    assert dataset.source_fps == 20
    assert dataset.video_temporal_stride == 4


def test_predecoded_rgb_preserves_order_repeats_and_clamps(tmp_path: Path) -> None:
    sys.path.insert(0, str(GE_ACT_ROOT))
    try:
        module = importlib.import_module("data.lerobot_like_dataset")
    finally:
        sys.path.pop(0)

    frames = np.arange(4 * 2 * 3 * 3, dtype=np.uint8).reshape(4, 2, 3, 3)
    path = tmp_path / "episode.npy"
    np.save(path, frames)

    actual = module.load_predecoded_rgb(path, [2, 0, 2, 99, -4])

    np.testing.assert_array_equal(actual, frames[[2, 0, 2, 3, 0]])


def test_predecoded_rgb_rejects_missing_file(tmp_path: Path) -> None:
    sys.path.insert(0, str(GE_ACT_ROOT))
    try:
        module = importlib.import_module("data.lerobot_like_dataset")
    finally:
        sys.path.pop(0)

    with pytest.raises(FileNotFoundError, match="predecoded RGB cache"):
        module.load_predecoded_rgb(tmp_path / "missing.npy", [0])


def test_lerobot_dataset_reads_camera_frames_from_mirrored_cache(tmp_path: Path) -> None:
    sys.path.insert(0, str(GE_ACT_ROOT))
    try:
        module = importlib.import_module("data.lerobot_like_dataset")
    finally:
        sys.path.pop(0)

    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    cache_path = (
        cache_root
        / "domain/videos/chunk-000/observation.images.image/episode_000001.npy"
    )
    cache_path.parent.mkdir(parents=True)
    frames = np.arange(3 * 2 * 2 * 3, dtype=np.uint8).reshape(3, 2, 2, 3)
    np.save(cache_path, frames)
    dataset = module.CustomLeRobotDataset(
        data_roots=[str(data_root)],
        domains=[],
        predecoded_video_root=str(cache_root),
        require_predecoded=True,
        chunk=1,
        action_chunk=1,
        n_previous=1,
        random_crop=False,
    )
    video_template = str(
        data_root / "domain/videos/chunk-000/{}/episode_000001.mp4"
    )

    (video,) = dataset.seek_mp4(
        video_template, ["observation.images.image"], [2, 0, 2]
    )

    expected = frames[[2, 0, 2]].transpose(3, 0, 1, 2) / 255.0
    np.testing.assert_allclose(video.numpy(), expected, rtol=0, atol=1e-7)


def test_libero_plus_reuses_the_stable_libero_inference_class() -> None:
    source = (GE_ACT_ROOT / "experiments/eval_libero_plus.py").read_text()
    assert "from experiments.eval_libero import InferenceLibero" in source
    assert "LIBERO_PLUS_ROOT" in source
