import importlib
import json
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
FASTWAM_SRC = REPO_ROOT / "third_party/FastWAM/src"
FASTWAM_ROOT = REPO_ROOT / "third_party/FastWAM"


def _install_fastwam_stubs(monkeypatch):
    if str(FASTWAM_SRC) not in sys.path:
        monkeypatch.syspath_prepend(str(FASTWAM_SRC))

    omegaconf = types.ModuleType("omegaconf")

    class DictConfig(dict):
        pass

    omegaconf.DictConfig = DictConfig
    omegaconf.OmegaConf = types.SimpleNamespace(
        to_container=lambda value, resolve=True: value,
    )
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf)

    hydra = types.ModuleType("hydra")
    hydra_utils = types.ModuleType("hydra.utils")
    hydra_utils.instantiate = lambda cfg: cfg
    hydra.utils = hydra_utils
    monkeypatch.setitem(sys.modules, "hydra", hydra)
    monkeypatch.setitem(sys.modules, "hydra.utils", hydra_utils)

    accelerate = types.ModuleType("accelerate")
    accelerate.PartialState = lambda: types.SimpleNamespace(is_main_process=True)
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)

    tv = types.ModuleType("torchvision")
    tv_transforms = types.ModuleType("torchvision.transforms")
    tv_functional = types.ModuleType("torchvision.transforms.functional")
    tv_functional.InterpolationMode = types.SimpleNamespace(BILINEAR="bilinear")
    tv_functional.resize = lambda tensor, size, interpolation=None, antialias=None: tensor
    tv_transforms.functional = tv_functional
    tv.transforms = tv_transforms
    monkeypatch.setitem(sys.modules, "torchvision", tv)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", tv_transforms)
    monkeypatch.setitem(sys.modules, "torchvision.transforms.functional", tv_functional)

    video_utils = types.ModuleType("fastwam.datasets.lerobot.lerobot.datasets.video_utils")
    video_utils.set_frame_cache_dir = lambda _path: None
    monkeypatch.setitem(
        sys.modules,
        "fastwam.datasets.lerobot.lerobot.datasets.video_utils",
        video_utils,
    )

    dataset_utils = types.ModuleType("fastwam.datasets.dataset_utils")

    class _IdentityTransform:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, x):
            return x

    dataset_utils.ResizeSmallestSideAspectPreserving = _IdentityTransform
    dataset_utils.CenterCrop = _IdentityTransform
    dataset_utils.Normalize = _IdentityTransform
    monkeypatch.setitem(sys.modules, "fastwam.datasets.dataset_utils", dataset_utils)

    normalizer = types.ModuleType("fastwam.datasets.lerobot.utils.normalizer")
    normalizer.save_dataset_stats_to_json = lambda *_args, **_kwargs: None
    normalizer.load_dataset_stats_from_json = lambda *_args, **_kwargs: {}
    monkeypatch.setitem(sys.modules, "fastwam.datasets.lerobot.utils.normalizer", normalizer)

    misc = types.ModuleType("fastwam.utils.misc")
    misc.get_work_dir = lambda: str(REPO_ROOT)
    pytorch_utils = types.ModuleType("fastwam.utils.pytorch_utils")
    fastwam_utils = importlib.import_module("fastwam.utils")
    monkeypatch.setattr(fastwam_utils, "misc", misc, raising=False)
    monkeypatch.setattr(fastwam_utils, "pytorch_utils", pytorch_utils, raising=False)
    monkeypatch.setitem(sys.modules, "fastwam.utils.misc", misc)
    monkeypatch.setitem(sys.modules, "fastwam.utils.pytorch_utils", pytorch_utils)

    base_module = types.ModuleType("fastwam.datasets.lerobot.base_lerobot_dataset")

    class FakeBaseLerobotDataset:
        def __init__(self, *args, **kwargs):
            self.samples = [
                {
                    "pixel_values": torch.zeros(5, 3, 2, 2),
                    "image_is_pad": torch.zeros(5, dtype=torch.bool),
                    "action": torch.zeros(4, 7),
                    "proprio": torch.zeros(5, 8),
                    "instruction": "pick up the cup",
                    "action_is_pad": torch.zeros(4, dtype=torch.bool),
                    "proprio_is_pad": torch.zeros(5, dtype=torch.bool),
                }
            ]
            self.processor = None

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = dict(self.samples[int(idx)])
            sample["idx"] = int(idx)
            return sample

        def _set_return_images(self, _flag):
            pass

        def set_processor(self, processor):
            self.processor = processor
            return self

        def get_dataset_stats(self, _processor):
            return {}

    base_module.BaseLerobotDataset = FakeBaseLerobotDataset
    monkeypatch.setitem(sys.modules, "fastwam.datasets.lerobot.base_lerobot_dataset", base_module)


def test_fastwam_libero_cosmos_data_config_is_vendored():
    assert (FASTWAM_ROOT / "configs/data/libero_2cam_cosmos.yaml").is_file()


def test_robot_video_dataset_loads_semantic_plan_manifest(tmp_path, monkeypatch):
    _install_fastwam_stubs(monkeypatch)
    module = importlib.import_module("fastwam.datasets.lerobot.robot_video_dataset")

    semantic_dir = tmp_path / "semantic"
    semantic_dir.mkdir()
    plan = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    torch.save({"semantic_plan": plan}, semantic_dir / "sample_a.pt")
    manifest = {
        "sample_id": "sample_a",
        "idx": 0,
        "video_frame_indices": [10, 12, 14, 16, 18],
        "future_frame_indices": [12, 18],
    }
    (semantic_dir / "manifest_0.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    dataset = module.RobotVideoDataset(
        dataset_dirs=[str(tmp_path / "lerobot")],
        shape_meta={"images": [], "action": [], "state": []},
        num_frames=5,
        action_video_freq_ratio=1,
        video_size=[2, 2],
        text_embedding_cache_dir=str(tmp_path / "text"),
        semantic_plan_dir=str(semantic_dir),
        semantic_plan_manifest="manifest_0.jsonl",
        semantic_plan_source="file",
        semantic_plan_dim=4,
        semantic_plan_max_tokens=3,
    )
    monkeypatch.setattr(
        dataset,
        "_get_cached_text_context",
        lambda _prompt: (torch.zeros(2, 8), torch.ones(2, dtype=torch.bool)),
    )

    sample = dataset[0]

    assert len(dataset) == 1
    assert torch.equal(sample["semantic_plan"], plan)
    assert torch.allclose(sample["semantic_plan_times"], torch.tensor([0.25, 1.0]))
    assert sample["semantic_plan_meta"]["sample_id"] == "sample_a"


class _DummyVideoExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = types.SimpleNamespace(blocks=nn.ModuleList([nn.Linear(1, 1)]))


class _DummyActionExpert(nn.Module):
    action_dim = 7

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(1, 1)])


class _CaptureCoupling:
    def __init__(self):
        self.semantic_plan = None
        self.semantic_plan_times = None

    def forward(self, model, noisy_latents, _t_v, noisy_action, _t_a, _crossattn_emb):
        self.semantic_plan = getattr(model, "_current_semantic_plan", None)
        self.semantic_plan_times = getattr(model, "_current_semantic_plan_times", None)
        return torch.zeros_like(noisy_latents), torch.zeros_like(noisy_action)


def test_fastwam_cosmos_training_loss_routes_semantic_plan(monkeypatch):
    if str(FASTWAM_SRC) not in sys.path:
        monkeypatch.syspath_prepend(str(FASTWAM_SRC))
    module = importlib.import_module("fastwam.models.cosmos.fastwam_cosmos")

    model = module.FastWAMCosmos(
        video_expert=_DummyVideoExpert(),
        action_expert=_DummyActionExpert(),
        vae=None,
        vae_encode_fn=lambda _name, _vae, video, device: torch.zeros(video.shape[0], 16, 2, 2, 2),
        qwen_dim=8,
        crossattn_dim=8,
        coupling="mot",
        device="cpu",
        torch_dtype=torch.float32,
        semantic_plan_dim=4,
        semantic_plan_max_tokens=3,
        semantic_plan_num_keyframes=2,
    )
    capture = _CaptureCoupling()
    model._coupling = capture
    sample = {
        "video": torch.zeros(2, 3, 4, 2, 2),
        "context": torch.zeros(2, 2, 8),
        "action": torch.zeros(2, 3, 7),
        "proprio": torch.zeros(2, 3, 8),
        "semantic_plan": torch.ones(2, 3, 4),
        "semantic_plan_times": torch.tensor([[0.25, 1.0], [0.25, 1.0]]),
        "video_fps": 5.0,
    }

    loss, _metrics = model.training_loss(sample)

    assert torch.isfinite(loss)
    assert capture.semantic_plan is not None
    assert capture.semantic_plan.shape == (2, 3, 4)
    assert torch.equal(capture.semantic_plan_times, sample["semantic_plan_times"])


def test_video_expert_prepare_returns_semantic_context(monkeypatch):
    if str(FASTWAM_SRC) not in sys.path:
        monkeypatch.syspath_prepend(str(FASTWAM_SRC))
    module = importlib.import_module("fastwam.models.cosmos.video_expert")

    class FakeNet:
        rope_enable_fps_modulation = False
        concat_padding_mask = False
        use_crossattn_projection = False

        def prepare_embedded_sequence(self, x, fps=None, padding_mask=None):
            batch, _channels, time, height, width = x.shape
            return torch.zeros(batch, time, height, width, 8), None, None

        def t_embedder(self, timesteps):
            batch, time = timesteps.shape
            return torch.zeros(batch, time, 8), None

        def t_embedding_norm(self, t_emb):
            return t_emb

        def prepare_semantic_plan_context(self, semantic_plan, latent_shape_T_H_W=None, fps=None, semantic_plan_times_B_N=None):
            assert latent_shape_T_H_W == (2, 2, 2)
            assert semantic_plan_times_B_N is not None
            return semantic_plan + 1.0, torch.ones(semantic_plan.shape[1], 1, 1, 8)

    expert = module.CosmosVideoExpert(FakeNet())
    plan = torch.zeros(1, 3, 8)
    times = torch.tensor([[0.0, 0.5, 1.0]])

    state = expert.prepare(
        torch.zeros(1, 16, 2, 2, 2),
        torch.zeros(1),
        torch.zeros(1, 2, 8),
        semantic_plan_B_L_D=plan,
        semantic_plan_times_B_N=times,
    )

    assert torch.equal(state["semantic_plan_crossattn"], plan + 1.0)
    assert state["semantic_plan_rope"].shape[0] == 3
