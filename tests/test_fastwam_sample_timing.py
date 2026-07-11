from __future__ import annotations

import importlib.util
import logging
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
FASTWAM_SRC = ROOT / "third_party/FastWAM/src"
LEROBOT_DIR = FASTWAM_SRC / "fastwam/datasets/lerobot"
BASE_DATASET_PATH = LEROBOT_DIR / "base_lerobot_dataset.py"
ROBOT_DATASET_PATH = LEROBOT_DIR / "robot_video_dataset.py"
FPS_ABSENT = object()


def _install_package(monkeypatch, name: str, path: Path) -> None:
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    monkeypatch.setitem(sys.modules, name, package)


def _install_fastwam_packages(monkeypatch) -> None:
    _install_package(monkeypatch, "fastwam", FASTWAM_SRC / "fastwam")
    _install_package(
        monkeypatch,
        "fastwam.datasets",
        FASTWAM_SRC / "fastwam/datasets",
    )
    _install_package(monkeypatch, "fastwam.datasets.lerobot", LEROBOT_DIR)
    _install_package(
        monkeypatch,
        "fastwam.datasets.lerobot.lerobot",
        LEROBOT_DIR / "lerobot",
    )
    _install_package(
        monkeypatch,
        "fastwam.datasets.lerobot.processors",
        LEROBOT_DIR / "processors",
    )
    _install_package(
        monkeypatch,
        "fastwam.datasets.lerobot.utils",
        LEROBOT_DIR / "utils",
    )
    _install_package(monkeypatch, "fastwam.utils", FASTWAM_SRC / "fastwam/utils")

    logging_module = types.ModuleType("fastwam.utils.logging_config")
    logging_module.get_logger = logging.getLogger
    monkeypatch.setitem(
        sys.modules,
        "fastwam.utils.logging_config",
        logging_module,
    )


def _load_source_module(monkeypatch, name: str, path: Path):
    monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _load_base_dataset_module(monkeypatch, *, raw_fps=20.0):
    _install_fastwam_packages(monkeypatch)

    fps_values = (
        list(raw_fps)
        if isinstance(raw_fps, (list, tuple))
        else [raw_fps]
    )
    metadata_count = 0

    lerobot_module = types.ModuleType(
        "fastwam.datasets.lerobot.lerobot.lerobot_dataset"
    )

    class FakeMetadata:
        def __init__(self, repo_id, root):
            nonlocal metadata_count
            del root
            self.repo_id = repo_id
            self.fps = fps_values[metadata_count]
            metadata_count += 1
            self.total_episodes = 1

    class FakeMultiLeRobotDataset:
        def __init__(self, **_kwargs):
            episode_index = {
                "from": torch.tensor([0]),
                "to": torch.tensor([1]),
            }
            self._datasets = [
                types.SimpleNamespace(episode_data_index=episode_index)
            ]
            self.num_frames = 1
            self.num_episodes = 1

    lerobot_module.LeRobotDatasetMetadata = FakeMetadata
    lerobot_module.MultiLeRobotDataset = FakeMultiLeRobotDataset
    monkeypatch.setitem(
        sys.modules,
        "fastwam.datasets.lerobot.lerobot.lerobot_dataset",
        lerobot_module,
    )

    processor_module = types.ModuleType(
        "fastwam.datasets.lerobot.processors.base_processor"
    )
    processor_module.BaseProcessor = object
    monkeypatch.setitem(
        sys.modules,
        "fastwam.datasets.lerobot.processors.base_processor",
        processor_module,
    )
    return _load_source_module(
        monkeypatch,
        "fastwam.datasets.lerobot.base_lerobot_dataset",
        BASE_DATASET_PATH,
    )


def _load_robot_dataset_module(
    monkeypatch,
    *,
    raw_fps=20.0,
    provides_validated_fps=FPS_ABSENT,
):
    _install_fastwam_packages(monkeypatch)

    omegaconf = types.ModuleType("omegaconf")

    class DictConfig(dict):
        pass

    omegaconf.DictConfig = DictConfig
    omegaconf.OmegaConf = types.SimpleNamespace(
        to_container=lambda value, resolve=True: value
    )
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf)

    hydra = types.ModuleType("hydra")
    hydra_utils = types.ModuleType("hydra.utils")
    hydra_utils.instantiate = lambda config: config
    hydra.utils = hydra_utils
    monkeypatch.setitem(sys.modules, "hydra", hydra)
    monkeypatch.setitem(sys.modules, "hydra.utils", hydra_utils)

    accelerate = types.ModuleType("accelerate")
    accelerate.PartialState = lambda: types.SimpleNamespace(is_main_process=True)
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)

    torchvision = types.ModuleType("torchvision")
    torchvision_transforms = types.ModuleType("torchvision.transforms")
    torchvision_functional = types.ModuleType("torchvision.transforms.functional")
    torchvision_functional.InterpolationMode = types.SimpleNamespace(
        BILINEAR="bilinear"
    )
    torchvision_functional.resize = (
        lambda tensor, size, interpolation=None, antialias=None: tensor
    )
    torchvision_transforms.functional = torchvision_functional
    torchvision.transforms = torchvision_transforms
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", torchvision_transforms)
    monkeypatch.setitem(
        sys.modules,
        "torchvision.transforms.functional",
        torchvision_functional,
    )

    base_module = types.ModuleType(
        "fastwam.datasets.lerobot.base_lerobot_dataset"
    )

    class FakeBaseLerobotDataset:
        task_text = "pick up the cup"

        def __init__(self, *, global_sample_stride, **_kwargs):
            if raw_fps is not FPS_ABSENT:
                self.fps = raw_fps
            self.global_sample_stride = global_sample_stride
            self.processor = None
            self.samples = [
                {
                    "pixel_values": torch.zeros(33, 3, 2, 2),
                    "image_is_pad": torch.zeros(33, dtype=torch.bool),
                    "action": torch.zeros(32, 7),
                    "proprio": torch.zeros(33, 8),
                    "instruction": self.task_text,
                    "action_is_pad": torch.zeros(32, dtype=torch.bool),
                    "proprio_is_pad": torch.zeros(33, dtype=torch.bool),
                }
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return dict(self.samples[int(index)])

        def _set_return_images(self, _flag):
            return None

        def set_processor(self, processor):
            self.processor = processor
            return self

        def get_dataset_stats(self, _processor):
            return {}

    if provides_validated_fps is not FPS_ABSENT:
        FakeBaseLerobotDataset.provides_validated_fps = provides_validated_fps

    base_module.BaseLerobotDataset = FakeBaseLerobotDataset
    monkeypatch.setitem(
        sys.modules,
        "fastwam.datasets.lerobot.base_lerobot_dataset",
        base_module,
    )

    video_utils = types.ModuleType(
        "fastwam.datasets.lerobot.lerobot.datasets.video_utils"
    )
    video_utils.set_frame_cache_dir = lambda _path: None
    monkeypatch.setitem(
        sys.modules,
        "fastwam.datasets.lerobot.lerobot.datasets.video_utils",
        video_utils,
    )

    normalizer = types.ModuleType("fastwam.datasets.lerobot.utils.normalizer")
    normalizer.save_dataset_stats_to_json = lambda *_args, **_kwargs: None
    normalizer.load_dataset_stats_from_json = lambda *_args, **_kwargs: {}
    monkeypatch.setitem(
        sys.modules,
        "fastwam.datasets.lerobot.utils.normalizer",
        normalizer,
    )

    dataset_utils = types.ModuleType("fastwam.datasets.dataset_utils")

    class IdentityTransform:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, value):
            return value

    dataset_utils.ResizeSmallestSideAspectPreserving = IdentityTransform
    dataset_utils.CenterCrop = IdentityTransform
    dataset_utils.Normalize = IdentityTransform
    monkeypatch.setitem(sys.modules, "fastwam.datasets.dataset_utils", dataset_utils)

    return _load_source_module(
        monkeypatch,
        "fastwam.datasets.lerobot.robot_video_dataset",
        ROBOT_DATASET_PATH,
    )


def _load_actual_base_and_robot_modules(monkeypatch):
    monkeypatch.syspath_prepend(str(FASTWAM_SRC))
    importlib.import_module("fastwam.datasets.lerobot")

    lerobot_module = types.ModuleType(
        "fastwam.datasets.lerobot.lerobot.lerobot_dataset"
    )

    class IntegrationMetadata:
        def __init__(self, repo_id, root):
            del root
            self.repo_id = repo_id
            self.fps = 20.0
            self.total_episodes = 1

    class IntegrationMultiLeRobotDataset:
        def __init__(self, **_kwargs):
            episode_index = {
                "from": torch.tensor([0]),
                "to": torch.tensor([1]),
            }
            self._datasets = [
                types.SimpleNamespace(episode_data_index=episode_index)
            ]
            self.num_frames = 1
            self.num_episodes = 1
            self.sample = {
                "task": "pick up the cup",
                "observation.images": torch.zeros(33, 3, 2, 2),
                "observation.state": torch.zeros(33, 8),
                "action": torch.zeros(32, 7),
                "observation.images_is_pad": torch.zeros(
                    33,
                    dtype=torch.bool,
                ),
                "observation.state_is_pad": torch.zeros(
                    33,
                    dtype=torch.bool,
                ),
                "action_is_pad": torch.zeros(32, dtype=torch.bool),
            }

        def __len__(self):
            return self.num_frames

        def __getitem__(self, index):
            assert int(index) == 0
            return dict(self.sample)

        def set_during_training(self, _flag):
            return None

    lerobot_module.LeRobotDatasetMetadata = IntegrationMetadata
    lerobot_module.MultiLeRobotDataset = IntegrationMultiLeRobotDataset
    monkeypatch.setitem(
        sys.modules,
        "fastwam.datasets.lerobot.lerobot.lerobot_dataset",
        lerobot_module,
    )

    video_utils = types.ModuleType(
        "fastwam.datasets.lerobot.lerobot.datasets.video_utils"
    )
    video_utils.set_frame_cache_dir = lambda _path: None
    monkeypatch.setitem(
        sys.modules,
        "fastwam.datasets.lerobot.lerobot.datasets.video_utils",
        video_utils,
    )

    base_module = _load_source_module(
        monkeypatch,
        "fastwam.datasets.lerobot.base_lerobot_dataset",
        BASE_DATASET_PATH,
    )
    robot_module = _load_source_module(
        monkeypatch,
        "fastwam.datasets.lerobot.robot_video_dataset",
        ROBOT_DATASET_PATH,
    )
    return base_module, robot_module


class IntegrationProcessor:
    def set_normalizer_from_stats(self, _stats):
        return None

    def train(self):
        return None

    def eval(self):
        return None

    def preprocess(self, sample):
        return {
            "pixel_values": sample["images"]["default"],
            "image_is_pad": sample["image_is_pad"],
            "action": sample["action"]["default"],
            "proprio": sample["state"]["default"],
            "instruction": sample["task"],
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["state_is_pad"],
        }


def _shape_meta():
    return {"images": [], "state": [], "action": []}


def test_base_dataset_preserves_raw_fps_and_normalizes_global_stride(monkeypatch):
    module = _load_base_dataset_module(monkeypatch, raw_fps=20)

    dataset = module.BaseLerobotDataset(
        dataset_dirs=["libero"],
        shape_meta=_shape_meta(),
        obs_size=1,
        action_size=0,
        val_set_proportion=0,
        global_sample_stride=np.int64(2),
    )

    assert dataset.fps == pytest.approx(20.0)
    assert type(dataset.fps) is float
    assert dataset.global_sample_stride == 2
    assert type(dataset.global_sample_stride) is int
    assert module.BaseLerobotDataset.provides_validated_fps is True


def test_base_dataset_validates_every_metadata_fps_before_comparison(monkeypatch):
    module = _load_base_dataset_module(monkeypatch, raw_fps=[1.0, True])

    with pytest.raises(TypeError, match="fps.*real number"):
        module.BaseLerobotDataset(
            dataset_dirs=["valid", "invalid"],
            shape_meta=_shape_meta(),
            obs_size=1,
            action_size=0,
            val_set_proportion=0,
        )


def test_base_dataset_reports_normalized_fps_mismatch(monkeypatch):
    module = _load_base_dataset_module(monkeypatch, raw_fps=[20, 30.0])

    with pytest.raises(ValueError, match="same fps.*20.0.*30.0"):
        module.BaseLerobotDataset(
            dataset_dirs=["twenty", "thirty"],
            shape_meta=_shape_meta(),
            obs_size=1,
            action_size=0,
            val_set_proportion=0,
        )


@pytest.mark.parametrize("raw_fps", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_base_dataset_rejects_nonpositive_or_nonfinite_fps(
    monkeypatch,
    raw_fps,
):
    module = _load_base_dataset_module(monkeypatch, raw_fps=raw_fps)

    with pytest.raises(ValueError, match="fps.*positive.*finite"):
        module.BaseLerobotDataset(
            dataset_dirs=["libero"],
            shape_meta=_shape_meta(),
            obs_size=1,
            action_size=0,
            val_set_proportion=0,
        )


@pytest.mark.parametrize("global_stride", [True, 1.0, 1.5, "1", None])
def test_base_dataset_rejects_non_integral_global_stride(
    monkeypatch,
    global_stride,
):
    module = _load_base_dataset_module(monkeypatch)

    with pytest.raises(TypeError, match="global_sample_stride.*integer"):
        module.BaseLerobotDataset(
            dataset_dirs=["libero"],
            shape_meta=_shape_meta(),
            obs_size=1,
            action_size=0,
            val_set_proportion=0,
            global_sample_stride=global_stride,
        )


@pytest.mark.parametrize("global_stride", [0, -1])
def test_base_dataset_rejects_nonpositive_global_stride(
    monkeypatch,
    global_stride,
):
    module = _load_base_dataset_module(monkeypatch)

    with pytest.raises(ValueError, match="global_sample_stride.*positive"):
        module.BaseLerobotDataset(
            dataset_dirs=["libero"],
            shape_meta=_shape_meta(),
            obs_size=1,
            action_size=0,
            val_set_proportion=0,
            global_sample_stride=global_stride,
        )


def test_effective_video_fps_uses_both_sampling_strides(monkeypatch):
    module = _load_robot_dataset_module(monkeypatch)

    assert module.compute_effective_video_fps(
        raw_fps=20.0,
        global_sample_stride=2,
        action_video_freq_ratio=4,
    ) == pytest.approx(2.5)
    assert module.compute_effective_video_fps(
        raw_fps=20.0,
        global_sample_stride=1,
        action_video_freq_ratio=4,
    ) == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("raw_fps", "global_stride", "ratio"),
    [
        (0.0, 1, 4),
        (-20.0, 1, 4),
        (math.inf, 1, 4),
        (-math.inf, 1, 4),
        (math.nan, 1, 4),
        (20.0, 0, 4),
        (20.0, -1, 4),
        (20.0, 1, 0),
        (20.0, 1, -4),
    ],
)
def test_effective_video_fps_rejects_invalid_numeric_inputs(
    monkeypatch,
    raw_fps,
    global_stride,
    ratio,
):
    module = _load_robot_dataset_module(monkeypatch)

    with pytest.raises(ValueError, match="positive|finite"):
        module.compute_effective_video_fps(
            raw_fps=raw_fps,
            global_sample_stride=global_stride,
            action_video_freq_ratio=ratio,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("raw_fps", True),
        ("raw_fps", "20"),
        ("global_sample_stride", True),
        ("global_sample_stride", 1.0),
        ("global_sample_stride", 1.5),
        ("action_video_freq_ratio", True),
        ("action_video_freq_ratio", 4.0),
        ("action_video_freq_ratio", 1.5),
    ],
)
def test_effective_video_fps_rejects_invalid_types(monkeypatch, name, value):
    module = _load_robot_dataset_module(monkeypatch)
    kwargs = {
        "raw_fps": 20.0,
        "global_sample_stride": 1,
        "action_video_freq_ratio": 4,
    }
    kwargs[name] = value

    with pytest.raises(TypeError, match=name):
        module.compute_effective_video_fps(**kwargs)


@pytest.mark.parametrize(
    ("name", "value", "error_type"),
    [
        ("global_sample_stride", True, TypeError),
        ("global_sample_stride", 1.0, TypeError),
        ("global_sample_stride", 0, ValueError),
        ("action_video_freq_ratio", True, TypeError),
        ("action_video_freq_ratio", 4.0, TypeError),
        ("action_video_freq_ratio", 0, ValueError),
    ],
)
def test_robot_video_dataset_validates_sampling_inputs_at_init(
    monkeypatch,
    tmp_path,
    name,
    value,
    error_type,
):
    module = _load_robot_dataset_module(monkeypatch)
    kwargs = {
        "dataset_dirs": [str(tmp_path / "lerobot")],
        "shape_meta": _shape_meta(),
        "num_frames": 33,
        "global_sample_stride": 1,
        "action_video_freq_ratio": 4,
        "video_size": [2, 2],
        "text_embedding_cache_dir": str(tmp_path / "text"),
    }
    kwargs[name] = value

    with pytest.raises(error_type, match=name):
        module.RobotVideoDataset(**kwargs)


@pytest.mark.parametrize("raw_fps", [0.0, -1.0, math.inf, math.nan])
def test_robot_video_dataset_rejects_present_but_invalid_base_fps(
    monkeypatch,
    tmp_path,
    raw_fps,
):
    module = _load_robot_dataset_module(monkeypatch, raw_fps=raw_fps)

    with pytest.raises(ValueError, match="raw_fps.*positive.*finite"):
        module.RobotVideoDataset(
            dataset_dirs=[str(tmp_path / "lerobot")],
            shape_meta=_shape_meta(),
            num_frames=33,
            global_sample_stride=1,
            action_video_freq_ratio=4,
            video_size=[2, 2],
            text_embedding_cache_dir=str(tmp_path / "text"),
        )


def test_robot_video_dataset_rejects_present_none_fps(monkeypatch, tmp_path):
    module = _load_robot_dataset_module(monkeypatch, raw_fps=None)

    with pytest.raises(TypeError, match="raw_fps.*real number"):
        module.RobotVideoDataset(
            dataset_dirs=[str(tmp_path / "lerobot")],
            shape_meta=_shape_meta(),
            num_frames=33,
            global_sample_stride=1,
            action_video_freq_ratio=4,
            video_size=[2, 2],
            text_embedding_cache_dir=str(tmp_path / "text"),
        )


@pytest.mark.parametrize("capability_marker", [True, False])
def test_marked_base_missing_fps_fails_instead_of_degrading(
    monkeypatch,
    tmp_path,
    capability_marker,
):
    module = _load_robot_dataset_module(
        monkeypatch,
        raw_fps=FPS_ABSENT,
        provides_validated_fps=capability_marker,
    )

    with pytest.raises(RuntimeError, match="validated FPS.*missing `fps`"):
        module.RobotVideoDataset(
            dataset_dirs=[str(tmp_path / "lerobot")],
            shape_meta=_shape_meta(),
            num_frames=33,
            global_sample_stride=1,
            action_video_freq_ratio=4,
            video_size=[2, 2],
            text_embedding_cache_dir=str(tmp_path / "text"),
        )


def test_robot_video_dataset_warns_when_injected_base_has_no_fps(
    monkeypatch,
    tmp_path,
    caplog,
):
    module = _load_robot_dataset_module(monkeypatch, raw_fps=FPS_ABSENT)

    with caplog.at_level(logging.WARNING):
        dataset = module.RobotVideoDataset(
            dataset_dirs=[str(tmp_path / "lerobot")],
            shape_meta=_shape_meta(),
            num_frames=33,
            global_sample_stride=1,
            action_video_freq_ratio=4,
            video_size=[2, 2],
            text_embedding_cache_dir=str(tmp_path / "text"),
        )

    assert dataset.video_fps is None
    assert "sampled video timing is unavailable" in caplog.text
    monkeypatch.setattr(
        dataset,
        "_get_cached_text_context",
        lambda _prompt: (
            torch.zeros(2, 8),
            torch.ones(2, dtype=torch.bool),
        ),
    )
    assert "video_fps" not in dataset[0]


def test_robot_video_sample_emits_fps_and_raw_instruction(monkeypatch, tmp_path):
    module = _load_robot_dataset_module(monkeypatch)
    dataset = module.RobotVideoDataset(
        dataset_dirs=[str(tmp_path / "lerobot")],
        shape_meta=_shape_meta(),
        num_frames=33,
        global_sample_stride=1,
        action_video_freq_ratio=4,
        video_size=[2, 2],
        text_embedding_cache_dir=str(tmp_path / "text"),
    )
    monkeypatch.setattr(
        dataset,
        "_get_cached_text_context",
        lambda _prompt: (
            torch.zeros(2, 8),
            torch.ones(2, dtype=torch.bool),
        ),
    )

    sample = dataset[0]

    assert dataset.lerobot_dataset.fps == pytest.approx(20.0)
    assert dataset.global_sample_stride == 1
    assert dataset.action_video_freq_ratio == 4
    assert dataset.video_fps == pytest.approx(5.0)
    assert sample["video_fps"].ndim == 0
    assert sample["video_fps"].dtype == torch.float32
    assert sample["video_fps"].item() == pytest.approx(5.0)
    assert sample["instruction"] == dataset.lerobot_dataset.task_text
    assert sample["prompt"] == (
        "A video recorded from a robot's point of view executing the following "
        "instruction: pick up the cup"
    )


def test_actual_base_robot_pipeline_emits_collatable_libero_timing(
    monkeypatch,
    tmp_path,
):
    base_module, robot_module = _load_actual_base_and_robot_modules(monkeypatch)
    stats_path = tmp_path / "stats.json"
    stats_path.write_text("{}", encoding="utf-8")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.setattr(robot_module, "_get_work_dir", lambda: str(work_dir))
    dataset = robot_module.RobotVideoDataset(
        dataset_dirs=[str(tmp_path / "lerobot")],
        shape_meta=robot_module.OmegaConf.create(
            {
                "images": [
                    {"key": "default", "raw_shape": [3, 2, 2]},
                ],
                "state": [{"key": "default", "raw_shape": 8}],
                "action": [{"key": "default", "raw_shape": 7}],
            }
        ),
        num_frames=33,
        global_sample_stride=1,
        action_video_freq_ratio=4,
        video_size=[2, 2],
        processor=IntegrationProcessor(),
        pretrained_norm_stats=str(stats_path),
        text_embedding_cache_dir=str(tmp_path / "text"),
    )
    monkeypatch.setattr(
        dataset,
        "_get_cached_text_context",
        lambda _prompt: (
            torch.zeros(2, 8),
            torch.ones(2, dtype=torch.bool),
        ),
    )

    sample = dataset[0]
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))

    assert base_module.BaseLerobotDataset.provides_validated_fps is True
    assert type(dataset.lerobot_dataset) is base_module.BaseLerobotDataset
    assert dataset.lerobot_dataset.fps == pytest.approx(20.0)
    assert sample["video_fps"].shape == torch.Size([])
    assert sample["video_fps"].dtype == torch.float32
    assert sample["video_fps"].item() == pytest.approx(5.0)
    assert batch["video_fps"].shape == (1,)
    assert batch["video_fps"].dtype == torch.float32
    assert batch["video_fps"].item() == pytest.approx(5.0)
    assert sample["instruction"] == "pick up the cup"
    assert sample["prompt"].endswith("instruction: pick up the cup")

    dataset.override_instruction = "move to the drawer"
    override_sample = dataset[0]
    assert override_sample["instruction"] == "move to the drawer"
    assert override_sample["prompt"].endswith("instruction: move to the drawer")
