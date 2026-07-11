from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import math
import sys
import types
from numbers import Real
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
FASTWAM_ROOT = ROOT / "third_party/FastWAM"
FASTWAM_SRC = FASTWAM_ROOT / "src"
EVAL_PATH = FASTWAM_ROOT / "experiments/libero/eval_libero_single.py"

if str(FASTWAM_SRC) not in sys.path:
    sys.path.insert(0, str(FASTWAM_SRC))

from fastwam.models.cosmos.couplings.agra import AGRACoupling  # noqa: E402
from fastwam.models.cosmos.couplings.cross_attn import CrossAttnCoupling  # noqa: E402
from fastwam.models.cosmos.couplings.mot import MoTCoupling  # noqa: E402
from fastwam.models.cosmos.fastwam_cosmos import FastWAMCosmos  # noqa: E402


class _RecordingVideoExpert:
    def __init__(self):
        self.net = types.SimpleNamespace(blocks=[])
        self.calls: list[tuple[str, object]] = []

    @staticmethod
    def _prepared(tokens, context):
        batch = tokens.shape[0]
        return {
            "tokens": tokens,
            "rope": None,
            "t_emb": tokens.new_zeros(batch, 1, tokens.shape[-1]),
            "crossattn": context,
            "THW": (1, 1, 1),
            "adaln_lora": None,
            "semantic_plan_crossattn": None,
            "semantic_plan_rope": None,
        }

    def prepare(self, noisy_latents, _timestep, context, **kwargs):
        self.calls.append(("prepare", kwargs.get("fps")))
        tokens = noisy_latents.new_zeros(noisy_latents.shape[0], 1, 2)
        return self._prepared(tokens, context)

    def finalize(self, tokens, *_args):
        return tokens

    def forward_standalone(self, noisy_latents, *_args, **kwargs):
        self.calls.append(("forward_standalone", kwargs.get("fps")))
        batch = noisy_latents.shape[0]
        return noisy_latents, noisy_latents.new_zeros(batch, 1, 2)

    def forward_foresight(self, noisy_latents, *_args, **kwargs):
        self.calls.append(("forward_foresight", kwargs.get("fps")))
        batch = noisy_latents.shape[0]
        return [noisy_latents.new_zeros(batch, 1, 2)]


class _RecordingActionExpert(nn.Module):
    action_dim = 1

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()

    def prepare(self, noisy_action, _timestep, context, _video_net):
        batch = noisy_action.shape[0]
        return {
            "tokens": noisy_action.new_zeros(batch, 1, 2),
            "rope": None,
            "t_emb": noisy_action.new_zeros(batch, 1, 2),
            "crossattn": context,
            "THW": (1, 1, 1),
            "adaln_lora": None,
        }

    def finalize(self, tokens):
        return tokens

    def forward_cross_attn(self, noisy_action, *_args):
        return noisy_action

    def forward(self, noisy_action, *_args):
        return torch.zeros_like(noisy_action)


def _coupling_case(name: str, *, semantic: bool):
    fps = torch.tensor([5.0, 5.0])
    video_expert = _RecordingVideoExpert()
    action_expert = _RecordingActionExpert()
    model = types.SimpleNamespace(
        video_expert=video_expert,
        action_expert=action_expert,
        action_head=action_expert,
        video_feat_proj=nn.Identity(),
        feature_layer=-1,
        mot_bidirectional=False,
        _mot_o0_latent=None,
        _mot_cond_frames=1,
        _agra_o0_latent=None,
        _agra_proprio0=torch.zeros(2, 1),
        agra_video_layers=[0],
        agra_video_projs=nn.ModuleList([nn.Linear(2, 2, bias=False)]),
        train_video_scheduler=types.SimpleNamespace(num_train_timesteps=1000),
        _current_semantic_plan=(torch.zeros(2, 4, 2) if semantic else None),
        _current_semantic_plan_times=(
            torch.tensor([[0.25, 0.5, 0.75, 1.0]]).expand(2, -1)
            if semantic
            else None
        ),
        _current_video_fps=fps,
    )
    coupling = {
        "mot": MoTCoupling(),
        "cross_attn": CrossAttnCoupling(),
        "agra": AGRACoupling(),
    }[name]
    inputs = (
        torch.zeros(2, 1, 1, 1, 1),
        torch.zeros(2),
        torch.zeros(2, 1, 1),
        torch.zeros(2),
        torch.zeros(2, 1, 2),
    )
    return coupling, model, inputs, fps


@pytest.mark.parametrize("semantic", [False, True])
@pytest.mark.parametrize(
    ("coupling_name", "expected_methods"),
    [
        ("mot", ["prepare"]),
        ("cross_attn", ["forward_standalone"]),
        ("agra", ["forward_standalone", "forward_foresight"]),
    ],
)
def test_every_training_coupling_routes_the_exact_sampled_fps_object(
    coupling_name,
    expected_methods,
    semantic,
):
    coupling, model, inputs, fps = _coupling_case(
        coupling_name,
        semantic=semantic,
    )

    coupling.forward(model, *inputs)

    assert [method for method, _ in model.video_expert.calls] == expected_methods
    assert all(value is fps for _, value in model.video_expert.calls)


class _UnitScheduler:
    num_train_timesteps = 1000

    @staticmethod
    def build_inference_schedule(_steps, device, dtype, shift_override=None):
        del shift_override
        return (
            torch.ones(1, device=device, dtype=dtype),
            torch.ones(1, device=device, dtype=dtype),
        )

    @staticmethod
    def step(_prediction, _delta, sample):
        return sample


def test_agra_inference_routes_the_exact_sampled_fps_object():
    model = FastWAMCosmos.__new__(FastWAMCosmos)
    nn.Module.__init__(model)
    fps = torch.tensor([5.0])
    video_expert = _RecordingVideoExpert()
    model.video_expert = video_expert
    model.action_expert = _RecordingActionExpert()
    model.agra_video_layers = [0]
    model.agra_video_projs = nn.ModuleList([nn.Linear(2, 2, bias=False)])
    model.train_video_scheduler = _UnitScheduler()
    model.train_action_scheduler = _UnitScheduler()
    model.device = "cpu"
    model.torch_dtype = torch.float32
    model.coupling = "agra"
    model.proprio_encoder = None
    model.text_proj = nn.Identity()
    model._current_semantic_plan = None
    model._current_semantic_plan_times = None
    model._current_video_fps = None
    model._vae_encode = lambda video: video.new_zeros(video.shape[0], 1, 1, 1, 1)

    def prepare_semantic(self, sample, _current_rgb):
        self._current_video_fps = sample["video_fps"]

    model._prepare_semantic_condition = types.MethodType(prepare_semantic, model)

    model._infer_action_impl(
        input_image=torch.zeros(1, 3, 2, 2),
        context=torch.zeros(1, 1, 2),
        action_horizon=1,
        num_inference_steps=1,
        num_video_frames=1,
        video_fps=fps,
    )

    assert [method for method, _ in video_expert.calls] == ["forward_foresight"]
    assert video_expert.calls[0][1] is fps
    assert video_expert.calls[0][1] is model._current_video_fps


def _load_sample_timing_helpers():
    path = ROOT / "tests/test_fastwam_sample_timing.py"
    spec = importlib.util.spec_from_file_location(
        "fastwam_sample_timing_helpers_for_source_mode",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataset_kwargs(tmp_path, **overrides):
    kwargs = {
        "dataset_dirs": [str(tmp_path / "lerobot")],
        "shape_meta": {"images": [], "state": [], "action": []},
        "num_frames": 33,
        "global_sample_stride": 1,
        "action_video_freq_ratio": 4,
        "video_size": [2, 2],
        "text_embedding_cache_dir": str(tmp_path / "text"),
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize("source", ["unknown", "FILE", True, 1, None])
def test_dataset_rejects_invalid_source_before_base_construction(
    monkeypatch,
    tmp_path,
    source,
):
    helpers = _load_sample_timing_helpers()
    module = helpers._load_robot_dataset_module(monkeypatch)
    base_calls = []
    monkeypatch.setattr(
        module,
        "BaseLerobotDataset",
        lambda **kwargs: base_calls.append(kwargs),
    )

    with pytest.raises(ValueError, match="semantic_plan_source"):
        module.RobotVideoDataset(
            **_dataset_kwargs(tmp_path, semantic_plan_source=source)
        )

    assert base_calls == []


@pytest.mark.parametrize("source", ["none", "online"])
@pytest.mark.parametrize(
    "paths",
    [
        {"semantic_plan_dir": "plans"},
        {"semantic_plan_manifest": "manifest.jsonl"},
        {
            "semantic_plan_dir": "plans",
            "semantic_plan_manifest": "manifest.jsonl",
        },
    ],
)
def test_non_file_sources_reject_file_paths_before_base_construction(
    monkeypatch,
    tmp_path,
    source,
    paths,
):
    helpers = _load_sample_timing_helpers()
    module = helpers._load_robot_dataset_module(monkeypatch)
    base_calls = []
    monkeypatch.setattr(
        module,
        "BaseLerobotDataset",
        lambda **kwargs: base_calls.append(kwargs),
    )

    with pytest.raises(ValueError, match="requires semantic_plan_dir.*null"):
        module.RobotVideoDataset(
            **_dataset_kwargs(
                tmp_path,
                semantic_plan_source=source,
                **paths,
            )
        )

    assert base_calls == []


@pytest.mark.parametrize(
    "paths",
    [
        {},
        {"semantic_plan_dir": "plans"},
        {"semantic_plan_manifest": "manifest.jsonl"},
    ],
)
def test_file_source_requires_both_paths_before_base_construction(
    monkeypatch,
    tmp_path,
    paths,
):
    helpers = _load_sample_timing_helpers()
    module = helpers._load_robot_dataset_module(monkeypatch)
    base_calls = []
    monkeypatch.setattr(
        module,
        "BaseLerobotDataset",
        lambda **kwargs: base_calls.append(kwargs),
    )

    with pytest.raises(ValueError, match="file.*requires both"):
        module.RobotVideoDataset(
            **_dataset_kwargs(
                tmp_path,
                semantic_plan_source="file",
                **paths,
            )
        )

    assert base_calls == []


@pytest.mark.parametrize("source", ["none", "online"])
def test_non_file_sources_never_attach_offline_or_zero_plans(
    monkeypatch,
    tmp_path,
    source,
):
    helpers = _load_sample_timing_helpers()
    module = helpers._load_robot_dataset_module(monkeypatch)
    dataset = module.RobotVideoDataset(
        **_dataset_kwargs(
            tmp_path,
            semantic_plan_source=source,
            semantic_plan_default_to_zero=True,
            semantic_plan_dim=1024,
            semantic_plan_max_tokens=1024,
        )
    )
    sample = {}

    dataset._attach_semantic_plan(sample, None, 0)

    assert dataset.semantic_plan_source == source
    assert "semantic_plan" not in sample
    assert "semantic_plan_times" not in sample


def test_file_source_preserves_manifest_loading(monkeypatch, tmp_path):
    helpers = _load_sample_timing_helpers()
    module = helpers._load_robot_dataset_module(monkeypatch)
    semantic_dir = tmp_path / "semantic"
    semantic_dir.mkdir()
    plan = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    torch.save({"semantic_plan": plan}, semantic_dir / "sample_a.pt")
    (semantic_dir / "manifest.jsonl").write_text(
        '{"sample_id":"sample_a","idx":0,"future_frame_indices":[2,8],'
        '"video_frame_indices":[0,2,4,6,8]}\n',
        encoding="utf-8",
    )
    dataset = module.RobotVideoDataset(
        **_dataset_kwargs(
            tmp_path,
            semantic_plan_source="file",
            semantic_plan_dir=str(semantic_dir),
            semantic_plan_manifest="manifest.jsonl",
            semantic_plan_dim=4,
            semantic_plan_max_tokens=3,
        )
    )
    monkeypatch.setattr(
        dataset,
        "_get_cached_text_context",
        lambda _prompt: (torch.zeros(2, 8), torch.ones(2, dtype=torch.bool)),
    )

    sample = dataset[0]

    assert dataset.semantic_plan_source == "file"
    assert torch.equal(sample["semantic_plan"], plan)


def test_cosmos_model_and_data_configs_use_exact_online_planner_contract():
    model = yaml.safe_load(
        (FASTWAM_ROOT / "configs/model/fastwam_cosmos.yaml").read_text()
    )
    expected_model = {
        "semantic_plan_context": True,
        "semantic_plan_in_dim": 1024,
        "semantic_plan_hidden_dim": 2048,
        "semantic_plan_num_keyframes": 4,
        "semantic_plan_source_num_keyframes": 4,
        "semantic_plan_spatial_grid": 16,
        "semantic_plan_max_tokens": 1024,
        "semantic_plan_coord_hidden_dim": 256,
        "semantic_plan_use_rope": True,
        "semantic_plan_cross_attention_blocks": None,
        "online_semantic_planner": False,
        "online_semantic_planner_code_dir": (
            "scripts/qwen3_vl_semantic_planner/lingbot_dino_4b"
        ),
        "online_semantic_planner_checkpoint": None,
        "semantic_plan_initial_depth_gate": 0.1,
    }
    for key, value in expected_model.items():
        assert model[key] == value

    data = yaml.safe_load(
        (FASTWAM_ROOT / "configs/data/libero_2cam_cosmos.yaml").read_text()
    )["train"]
    assert data["semantic_plan_source"] == "none"
    assert data["semantic_plan_dir"] is None
    assert data["semantic_plan_manifest"] is None
    assert data["semantic_plan_dim"] == 1024
    assert data["semantic_plan_max_tokens"] == 1024
    assert data["semantic_plan_default_to_zero"] is False


def test_online_task_composes_exact_checkpoint_source_and_eval_fps(
    monkeypatch,
    tmp_path,
):
    checkpoint = tmp_path / "planner"
    monkeypatch.setenv("FASTWAM_PLANNER_CHECKPOINT", str(checkpoint))
    with initialize_config_dir(
        config_dir=str((FASTWAM_ROOT / "configs").resolve()),
        version_base=None,
    ):
        config = compose(
            config_name="train",
            overrides=["task=libero_cosmos_2cam224_online_dino_depth"],
        )

    assert config.model.online_semantic_planner is True
    assert config.model.online_semantic_planner_checkpoint == str(checkpoint)
    assert config.data.train.semantic_plan_source == "online"
    assert config.EVALUATION.video_fps == pytest.approx(5.0)
    assert OmegaConf.to_container(config, resolve=True)["model"][
        "semantic_plan_num_keyframes"
    ] == 4


def test_sim_libero_online_model_keys_are_all_accepted_by_cosmos_factory(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("FASTWAM_PLANNER_CHECKPOINT", str(tmp_path / "planner"))
    with initialize_config_dir(
        config_dir=str((FASTWAM_ROOT / "configs").resolve()),
        version_base=None,
    ):
        config = compose(
            config_name="sim_libero",
            overrides=["task=libero_cosmos_2cam224_online_dino_depth"],
        )

    factory = importlib.import_module(
        "fastwam.models.cosmos.runtime"
    ).create_fastwam_cosmos
    parameters = inspect.signature(factory).parameters
    model_config = OmegaConf.to_container(config.model, resolve=True)
    unsupported = sorted(
        key
        for key in model_config
        if key != "_target_" and key not in parameters
    )

    assert unsupported == []
    assert config.model.load_text_encoder is True
    assert config.model.skip_dit_load_from_pretrain is True
    assert config.model.action_dit_pretrained_path is None


def test_sim_libero_online_config_lightweight_instantiates_from_fastwam_cwd(
    monkeypatch,
    tmp_path,
):
    helpers = _load_online_planner_test_helpers()
    checkpoint = tmp_path / "planner"
    helpers._write_fake_checkpoint(checkpoint)
    monkeypatch.setenv("FASTWAM_PLANNER_CHECKPOINT", str(checkpoint))
    with initialize_config_dir(
        config_dir=str((FASTWAM_ROOT / "configs").resolve()),
        version_base=None,
    ):
        config = compose(
            config_name="sim_libero",
            overrides=["task=libero_cosmos_2cam224_online_dino_depth"],
        )
    config.model.vae = None
    config.model.video_dit_pretrained_path = "unused"

    runtime = importlib.import_module("fastwam.models.cosmos.runtime")
    capture = {}

    class Video:
        net = types.SimpleNamespace(
            model_channels=8,
            blocks=[types.SimpleNamespace(self_attn=types.SimpleNamespace(n_heads=1))],
        )

    class Action(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            capture["action"] = kwargs
            self.blocks = nn.ModuleList([nn.Linear(1, 1)])

        def copy_init_from_video(self, _net):
            return None

    class Model:
        proprio_encoder = None

        def __init__(self, **kwargs):
            capture["model"] = kwargs
            self.dit = nn.Linear(1, 1)

    monkeypatch.setattr(
        runtime.CosmosVideoExpert,
        "from_pretrained",
        lambda **kwargs: capture.setdefault("video", kwargs) and Video(),
    )
    monkeypatch.setattr(runtime, "CosmosActionExpert", Action)
    monkeypatch.setattr(runtime, "FastWAMCosmos", Model)
    monkeypatch.setattr(
        runtime,
        "load_online_semantic_planner",
        lambda **kwargs: capture.setdefault("loader", kwargs) and object(),
    )
    monkeypatch.chdir(FASTWAM_ROOT)

    model = instantiate(config.model, model_dtype=torch.float32, device="cpu")

    assert isinstance(model, Model)
    assert capture["loader"]["code_dir"] == str(
        (
            ROOT
            / "scripts/qwen3_vl_semantic_planner/lingbot_dino_4b"
        ).resolve()
    )
    assert capture["loader"]["checkpoint_dir"] == str(checkpoint.resolve())
    assert capture["model"]["online_semantic_planner"] is not None


def _load_online_planner_test_helpers():
    path = ROOT / "tests/test_fastwam_online_semantic_planner.py"
    spec = importlib.util.spec_from_file_location(
        "fastwam_online_planner_helpers_for_timing_routing",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_relative_planner_code_dir_resolves_from_fastwam_working_directory(
    monkeypatch,
    tmp_path,
):
    helpers = _load_online_planner_test_helpers()
    loader = helpers._load_loader_module()
    checkpoint = tmp_path / "checkpoint"
    helpers._write_fake_checkpoint(checkpoint)
    monkeypatch.chdir(FASTWAM_ROOT)

    module_path, checkpoint_path = loader.validate_online_semantic_planner_paths(
        code_dir="scripts/qwen3_vl_semantic_planner/lingbot_dino_4b",
        checkpoint_dir=str(checkpoint),
    )

    assert module_path == (
        ROOT
        / "scripts/qwen3_vl_semantic_planner/lingbot_dino_4b"
        / "dino_depth_plan_provider.py"
    ).resolve()
    assert checkpoint_path == checkpoint.resolve()


def test_relative_planner_code_dir_honors_explicit_vlm4wam_root(
    monkeypatch,
    tmp_path,
):
    helpers = _load_online_planner_test_helpers()
    loader = helpers._load_loader_module()
    vlm_root = tmp_path / "explicit-vlm-root"
    relative_code = Path("custom/planner")
    code_dir = vlm_root / relative_code
    checkpoint = tmp_path / "checkpoint"
    helpers._write_fake_provider(code_dir)
    helpers._write_fake_checkpoint(checkpoint)
    working_dir = tmp_path / "working-dir"
    working_dir.mkdir()
    # An explicit root is an override, so it must beat a stale checkout in cwd.
    helpers._write_fake_provider(working_dir / relative_code)
    monkeypatch.chdir(working_dir)
    monkeypatch.setenv("VLM4WAM_ROOT", str(vlm_root))

    module_path, _ = loader.validate_online_semantic_planner_paths(
        code_dir=str(relative_code),
        checkpoint_dir=str(checkpoint),
    )

    assert module_path == (code_dir / "dino_depth_plan_provider.py").resolve()


def _load_eval_helper(name: str, extra_namespace=None):
    source = EVAL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(EVAL_PATH))
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    for item in ast.walk(node):
        if isinstance(item, ast.arg):
            item.annotation = None
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            item.returns = None
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"inspect": inspect, "math": math, "Real": Real, "torch": torch}
    if extra_namespace is not None:
        namespace.update(extra_namespace)
    exec(compile(module, str(EVAL_PATH), "exec"), namespace)
    return namespace[name]


class _OnlineInferenceAPI:
    def infer_action(self, *, prompt, instruction=None, video_fps=None):
        del prompt, instruction, video_fps


class _LegacyInferenceAPI:
    def infer_action(self, *, prompt):
        del prompt


class _InstructionOnlyInferenceAPI:
    def infer_action(self, *, prompt, instruction=None):
        del prompt, instruction


class _VideoFpsOnlyInferenceAPI:
    def infer_action(self, *, prompt, video_fps=None):
        del prompt, video_fps


def test_eval_helper_passes_raw_instruction_and_explicit_five_fps():
    helper = _load_eval_helper("_add_online_semantic_inference_inputs")
    config = OmegaConf.create(
        {
            "model": {"online_semantic_planner": True},
            "EVALUATION": {"video_fps": 5.0},
        }
    )
    kwargs = {"prompt": "formatted prompt"}

    result = helper(
        kwargs,
        task_description="raw libero task",
        model=_OnlineInferenceAPI(),
        cfg=config,
    )

    assert result is kwargs
    assert result["instruction"] == "raw libero task"
    assert result["instruction"] != result["prompt"]
    assert result["video_fps"] == pytest.approx(5.0)


def test_eval_helper_fails_before_online_call_without_explicit_fps():
    helper = _load_eval_helper("_add_online_semantic_inference_inputs")
    config = OmegaConf.create(
        {
            "model": {"online_semantic_planner": True},
            "EVALUATION": {},
        }
    )

    with pytest.raises(ValueError, match="EVALUATION.video_fps"):
        helper(
            {"prompt": "formatted prompt"},
            task_description="raw libero task",
            model=_OnlineInferenceAPI(),
            cfg=config,
        )


@pytest.mark.parametrize(
    "model",
    [_InstructionOnlyInferenceAPI(), _VideoFpsOnlyInferenceAPI()],
)
def test_eval_helper_rejects_partial_online_inference_apis(model):
    helper = _load_eval_helper("_add_online_semantic_inference_inputs")
    config = OmegaConf.create(
        {
            "model": {"online_semantic_planner": True},
            "EVALUATION": {"video_fps": 5.0},
        }
    )
    kwargs = {"prompt": "formatted prompt"}

    with pytest.raises(ValueError, match="instruction.*video_fps.*together"):
        helper(
            kwargs,
            task_description="raw libero task",
            model=model,
            cfg=config,
        )

    assert kwargs == {"prompt": "formatted prompt"}


@pytest.mark.parametrize(
    "video_fps",
    [True, "5.0", 0, -1.0, math.nan, math.inf, -math.inf],
)
def test_eval_helper_rejects_invalid_explicit_video_fps(video_fps):
    helper = _load_eval_helper("_add_online_semantic_inference_inputs")
    config = OmegaConf.create(
        {
            "model": {"online_semantic_planner": True},
            "EVALUATION": {"video_fps": video_fps},
        }
    )

    with pytest.raises(ValueError, match="finite positive real"):
        helper(
            {"prompt": "formatted prompt"},
            task_description="raw libero task",
            model=_OnlineInferenceAPI(),
            cfg=config,
        )


@pytest.mark.parametrize(
    ("model", "online"),
    [(_LegacyInferenceAPI(), True), (_OnlineInferenceAPI(), False)],
)
def test_eval_helper_leaves_unsupported_or_offline_apis_unchanged(model, online):
    helper = _load_eval_helper("_add_online_semantic_inference_inputs")
    config = OmegaConf.create(
        {
            "model": {"online_semantic_planner": online},
            "EVALUATION": {},
        }
    )
    kwargs = {"prompt": "formatted prompt"}

    assert helper(
        kwargs,
        task_description="raw libero task",
        model=model,
        cfg=config,
    ) == {"prompt": "formatted prompt"}


def _eval_config(*, visualize_future_video=False):
    return OmegaConf.create(
        {
            "model": {"online_semantic_planner": True},
            "data": {"train": {"num_frames": 33, "action_video_freq_ratio": 4}},
            "EVALUATION": {
                "video_fps": 5.0,
                "visualize_future_video": visualize_future_video,
                "num_inference_steps": 1,
                "negative_prompt": "",
                "text_cfg_scale": 1.0,
                "sigma_shift": None,
                "rand_device": "cpu",
                "tiled": False,
            },
            "seed": 7,
            "eval_num_inference_steps": 1,
        }
    )


class _CallThroughOnlineModel:
    torch_dtype = torch.float32

    def __init__(self):
        self.infer_action_calls = []
        self.infer_joint_calls = []

    def infer_action(
        self,
        *,
        prompt,
        instruction=None,
        video_fps=None,
        num_video_frames=None,
        **kwargs,
    ):
        self.infer_action_calls.append(
            {
                "prompt": prompt,
                "instruction": instruction,
                "video_fps": video_fps,
                "num_video_frames": num_video_frames,
                **kwargs,
            }
        )
        return {"action": torch.zeros(2, 2)}

    def infer_joint(self, **kwargs):
        self.infer_joint_calls.append(kwargs)
        return {"action": torch.zeros(2, 2), "video": [object()]}


def _load_predict_action_chunk(observation_calls):
    try:
        validate_mode = _load_eval_helper(
            "_validate_online_semantic_eval_mode"
        )
    except StopIteration:
        def validate_mode(_cfg):
            return None
    add_inputs = _load_eval_helper("_add_online_semantic_inference_inputs")

    def obs_to_model_input(*_args, **_kwargs):
        observation_calls.append(True)
        return torch.zeros(1, 3, 2, 2), torch.zeros(1, 1), {"camera": []}

    namespace = {
        "DEFAULT_PROMPT": (
            "A video recorded from a robot's point of view executing the "
            "following instruction: {task}"
        ),
        "_validate_online_semantic_eval_mode": validate_mode,
        "_add_online_semantic_inference_inputs": add_inputs,
        "_obs_to_model_input": obs_to_model_input,
        "_get_num_video_frames": lambda _cfg: 9,
        "_select_predicted_future_frames": lambda frames, _cfg: frames,
        "_denormalize_action": lambda action, _processor: action.unsqueeze(0).numpy(),
        "invert_gripper_action": lambda action: action,
        "np": np,
    }
    return _load_eval_helper("_predict_action_chunk", namespace)


def test_predict_action_chunk_calls_online_api_with_raw_task_and_five_fps():
    observation_calls = []
    predict = _load_predict_action_chunk(observation_calls)
    model = _CallThroughOnlineModel()

    predict(
        {},
        "raw libero task",
        model,
        object(),
        _eval_config(),
        action_horizon=2,
        input_w=2,
        input_h=2,
        model_device="cpu",
    )

    assert observation_calls == [True]
    assert model.infer_joint_calls == []
    assert len(model.infer_action_calls) == 1
    call = model.infer_action_calls[0]
    assert call["instruction"] == "raw libero task"
    assert call["video_fps"] == pytest.approx(5.0)
    assert call["prompt"].endswith("instruction: raw libero task")
    assert call["prompt"] != call["instruction"]


def test_predict_action_chunk_rejects_online_future_video_before_any_call():
    observation_calls = []
    predict = _load_predict_action_chunk(observation_calls)
    model = _CallThroughOnlineModel()

    with pytest.raises(ValueError, match="online semantic.*infer_joint"):
        predict(
            {},
            "raw libero task",
            model,
            object(),
            _eval_config(visualize_future_video=True),
            action_horizon=2,
            input_w=2,
            input_h=2,
            model_device="cpu",
        )

    assert observation_calls == []
    assert model.infer_action_calls == []
    assert model.infer_joint_calls == []
