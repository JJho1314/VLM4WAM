from __future__ import annotations

import importlib.util
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def precompute_module():
    return _load_script_module(
        "fastwam_precompute_frames_for_test",
        "third_party/FastWAM/scripts/precompute_frames.py",
    )


@pytest.fixture
def verify_module():
    return _load_script_module(
        "fastwam_verify_frame_cache_for_test",
        "third_party/FastWAM/scripts/verify_frame_cache.py",
    )


def test_resolve_decoder_backend_auto_falls_back_to_pyav(
    monkeypatch: pytest.MonkeyPatch,
    precompute_module,
):
    monkeypatch.setattr(
        precompute_module.importlib.util,
        "find_spec",
        lambda name: None,
    )

    assert precompute_module.resolve_decoder_backend("auto") == "pyav"


def test_resolve_decoder_backend_rejects_missing_explicit_torchcodec(
    monkeypatch: pytest.MonkeyPatch,
    precompute_module,
):
    monkeypatch.setattr(
        precompute_module.importlib.util,
        "find_spec",
        lambda name: None,
    )

    with pytest.raises(RuntimeError, match="torchcodec is unavailable"):
        precompute_module.resolve_decoder_backend("torchcodec")


def test_decode_all_frames_pyav_returns_uint8_nchw(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    precompute_module,
):
    arrays = [
        np.array(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            ],
            dtype=np.uint8,
        ),
        np.array(
            [
                [[21, 22, 23], [24, 25, 26], [27, 28, 29]],
                [[30, 31, 32], [33, 34, 35], [36, 37, 38]],
            ],
            dtype=np.uint8,
        ),
    ]

    class FakeFrame:
        def __init__(self, array: np.ndarray):
            self.array = array

        def to_ndarray(self, *, format: str) -> np.ndarray:
            assert format == "rgb24"
            return self.array.copy()

    class FakeContainer:
        def __init__(self):
            self.stream = SimpleNamespace(
                average_rate=Fraction(20, 1),
                base_rate=None,
            )
            self.streams = SimpleNamespace(video=[self.stream])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def decode(self, stream):
            assert stream is self.stream
            return [FakeFrame(array) for array in arrays]

    fake_av = SimpleNamespace(open=lambda path: FakeContainer())
    monkeypatch.setitem(sys.modules, "av", fake_av)

    frames, fps = precompute_module.decode_all_frames_pyav(
        tmp_path / "episode.mp4"
    )

    assert frames.dtype == torch.uint8
    assert frames.shape == (2, 3, 2, 3)
    assert fps == 20.0
    assert frames[0, :, 0, 0].tolist() == [1, 2, 3]
    assert frames[1, :, 1, 2].tolist() == [36, 37, 38]


def test_apply_dataset_overrides_updates_hydra_node(
    tmp_path: Path,
    verify_module,
):
    suite = tmp_path / "suite"
    stats = tmp_path / "stats.json"
    node = OmegaConf.create(
        {
            "dataset_dirs": ["old"],
            "pretrained_norm_stats": None,
        }
    )

    verify_module.apply_dataset_overrides(
        node,
        [str(suite)],
        str(stats),
    )

    assert list(node.dataset_dirs) == [str(suite.resolve())]
    assert node.pretrained_norm_stats == str(stats.resolve())
