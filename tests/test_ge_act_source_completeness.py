from __future__ import annotations

import importlib
import sys
from pathlib import Path


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


def test_libero_plus_reuses_the_stable_libero_inference_class() -> None:
    source = (GE_ACT_ROOT / "experiments/eval_libero_plus.py").read_text()
    assert "from experiments.eval_libero import InferenceLibero" in source
    assert "LIBERO_PLUS_ROOT" in source
