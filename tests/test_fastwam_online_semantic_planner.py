from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
FASTWAM_SRC = ROOT / "third_party/FastWAM/src"
LOADER_PATH = (
    FASTWAM_SRC / "fastwam/models/cosmos/online_semantic_planner.py"
)

if str(FASTWAM_SRC) not in sys.path:
    sys.path.insert(0, str(FASTWAM_SRC))


class _FakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(1, 1)])
        self.model_channels = 8


class _FakeVideoExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _FakeNet()
        self.fusion_scale = nn.Parameter(torch.tensor(0.25))
        self.fusion_inputs = None

    def fuse_semantic_plan(self, dino_plan, depth_plan):
        self.fusion_inputs = (dino_plan, depth_plan)
        return dino_plan + self.fusion_scale * depth_plan


class _FakeActionExpert(nn.Module):
    action_dim = 2

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(1, 1)])


class _CaptureCoupling:
    def __init__(self):
        self.calls = 0

    def forward(
        self,
        model,
        noisy_latents,
        _t_v,
        noisy_action,
        _t_a,
        _crossattn_emb,
    ):
        self.calls += 1
        semantic = model._current_semantic_plan
        semantic_term = (
            noisy_latents.new_zeros(())
            if semantic is None
            else semantic.to(noisy_latents.dtype).mean()
        )
        return (
            torch.zeros_like(noisy_latents) + semantic_term,
            torch.zeros_like(noisy_action),
        )


class _PlanResult:
    def __init__(self, batch: int, tokens: int = 3, dim: int = 4):
        self.dino_plan = torch.ones(
            batch, tokens, dim, requires_grad=True
        )
        self.depth_plan = torch.full(
            (batch, tokens, dim), 2.0, requires_grad=True
        )
        self.semantic_plan_times = torch.tensor(
            [[0.25, 0.5, 0.75, 1.0]], dtype=torch.float32
        ).expand(batch, -1)


class _FakeProvider:
    def __init__(self, *, tokens: int = 3, dim: int = 4):
        self.tokens = tokens
        self.dim = dim
        self.calls = []
        self.last_result = None

    def predict(self, images, instructions):
        self.calls.append((images.detach().clone(), list(instructions)))
        self.last_result = _PlanResult(images.shape[0], self.tokens, self.dim)
        return self.last_result


@pytest.fixture
def cosmos_module():
    return importlib.import_module("fastwam.models.cosmos.fastwam_cosmos")


def _make_model(cosmos_module, *, provider=None, vae_calls=None):
    if vae_calls is None:
        vae_calls = []

    def encode(_name, _vae, video, device):
        del device
        vae_calls.append(video.detach().clone())
        return torch.zeros(
            video.shape[0], 4, 2 if video.shape[2] > 1 else 1, 2, 2
        )

    model = cosmos_module.FastWAMCosmos(
        video_expert=_FakeVideoExpert(),
        action_expert=_FakeActionExpert(),
        vae=None,
        vae_encode_fn=encode,
        qwen_dim=8,
        crossattn_dim=8,
        coupling="mot",
        device="cpu",
        torch_dtype=torch.float32,
        semantic_plan_dim=4,
        semantic_plan_max_tokens=3,
        semantic_plan_num_keyframes=4,
        online_semantic_planner=provider,
    )
    model._coupling = _CaptureCoupling()
    return model


def _training_sample(batch: int = 2):
    video = torch.linspace(-1.0, 1.0, batch * 3 * 9 * 2 * 2).reshape(
        batch, 3, 9, 2, 2
    )
    return {
        "video": video,
        "context": torch.zeros(batch, 2, 8),
        "action": torch.zeros(batch, 3, 2),
        "instruction": [f"task {index}" for index in range(batch)],
        "video_fps": torch.full((batch,), 5.0),
    }


def _load_loader_module():
    spec = importlib.util.spec_from_file_location(
        "fastwam_online_semantic_planner_test", LOADER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fake_provider(code_dir: Path) -> None:
    code_dir.mkdir(parents=True)
    (code_dir / "dino_depth_plan_provider.py").write_text(
        "class FrozenDinoDepthPlanProvider:\n"
        "    @classmethod\n"
        "    def from_checkpoint(cls, checkpoint_dir, *, device, dtype):\n"
        "        return {\"checkpoint\": str(checkpoint_dir), "
        "\"device\": device, \"dtype\": dtype}\n",
        encoding="utf-8",
    )


def test_loader_imports_provider_without_reusing_a_colliding_module(
    monkeypatch, tmp_path
):
    loader = _load_loader_module()
    code_dir = tmp_path / "planner/lingbot_dino_4b"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    _write_fake_provider(code_dir)
    sentinel = types.ModuleType("fastwam_dino_depth_plan_provider")
    monkeypatch.setitem(sys.modules, "fastwam_dino_depth_plan_provider", sentinel)

    result = loader.load_online_semantic_planner(
        code_dir=str(code_dir),
        checkpoint_dir=str(checkpoint_dir),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert result["checkpoint"] == str(checkpoint_dir.resolve())
    assert result["device"] == torch.device("cpu")
    assert result["dtype"] is torch.float32
    assert sys.modules["fastwam_dino_depth_plan_provider"] is sentinel


@pytest.mark.parametrize("missing", ["code", "checkpoint"])
def test_loader_preflight_reports_missing_paths_before_import(tmp_path, missing):
    loader = _load_loader_module()
    code_dir = tmp_path / "planner/lingbot_dino_4b"
    checkpoint_dir = tmp_path / "checkpoint"
    if missing != "code":
        _write_fake_provider(code_dir)
    if missing != "checkpoint":
        checkpoint_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="online planner"):
        loader.validate_online_semantic_planner_paths(
            code_dir=str(code_dir), checkpoint_dir=str(checkpoint_dir)
        )


def test_loader_wraps_external_import_failure_with_provider_path(tmp_path):
    loader = _load_loader_module()
    code_dir = tmp_path / "planner/lingbot_dino_4b"
    checkpoint_dir = tmp_path / "checkpoint"
    code_dir.mkdir(parents=True)
    checkpoint_dir.mkdir()
    module_path = code_dir / "dino_depth_plan_provider.py"
    module_path.write_text("raise RuntimeError('boom')\n", encoding="utf-8")

    with pytest.raises(ImportError, match=str(module_path)):
        loader.load_online_semantic_planner(
            code_dir=str(code_dir),
            checkpoint_dir=str(checkpoint_dir),
            device="cpu",
            dtype=torch.float32,
        )


def test_training_calls_provider_once_with_current_normalized_rgb_and_raw_task(
    cosmos_module,
):
    provider = _FakeProvider()
    model = _make_model(cosmos_module, provider=provider)
    sample = _training_sample()

    loss, _metrics = model.training_loss(sample)
    loss.backward()

    assert len(provider.calls) == 1
    images, instructions = provider.calls[0]
    assert torch.equal(images, sample["video"][:, :, 0])
    assert instructions == sample["instruction"]
    assert model._current_semantic_plan.shape == (2, 3, 4)
    assert model._current_semantic_plan_times.shape == (2, 4)
    assert torch.equal(model._current_video_fps, sample["video_fps"])
    assert model.video_expert.fusion_scale.grad is not None
    dino_input, depth_input = model.video_expert.fusion_inputs
    assert not dino_input.requires_grad
    assert not depth_input.requires_grad
    assert provider.last_result.dino_plan.grad is None
    assert provider.last_result.depth_plan.grad is None


def test_module_provider_is_outside_state_parameters_and_train_traversal(
    cosmos_module,
):
    class ModuleProvider(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

    provider = ModuleProvider()
    model = _make_model(cosmos_module, provider=provider)
    model.eval()

    assert model._online_semantic_planner is provider
    assert provider.training
    assert all(parameter is not provider.weight for parameter in model.parameters())
    assert provider not in list(model.modules())
    assert not any("online_semantic_planner" in key for key in model.state_dict())
    assert not any("online_semantic_planner" in key for key in model.state_payload())


def test_online_and_offline_semantics_are_mutually_exclusive(cosmos_module):
    model = _make_model(cosmos_module, provider=_FakeProvider())
    sample = _training_sample()
    sample["semantic_plan"] = torch.zeros(2, 3, 4)
    sample["semantic_plan_times"] = torch.tensor(
        [[0.25, 0.5, 0.75, 1.0]]
    ).expand(2, -1)

    with pytest.raises(ValueError, match="mutually exclusive"):
        model.training_loss(sample)


@pytest.mark.parametrize("missing", ["instruction", "video_fps"])
def test_online_semantics_require_instruction_and_fps(cosmos_module, missing):
    model = _make_model(cosmos_module, provider=_FakeProvider())
    sample = _training_sample()
    del sample[missing]

    with pytest.raises(KeyError, match=missing):
        model.training_loss(sample)


@pytest.mark.parametrize(
    "instruction",
    ["", "   ", ["task 0"], ["task 0", 4], ["task 0", ""]],
)
def test_online_semantics_validate_batched_raw_instruction(
    cosmos_module, instruction
):
    model = _make_model(cosmos_module, provider=_FakeProvider())
    sample = _training_sample()
    sample["instruction"] = instruction

    with pytest.raises(ValueError, match="instruction"):
        model.training_loss(sample)


def test_offline_semantic_fields_must_be_complete_and_require_fps(cosmos_module):
    model = _make_model(cosmos_module)
    sample = _training_sample()
    del sample["instruction"]
    sample["semantic_plan"] = torch.zeros(2, 3, 4)

    with pytest.raises(ValueError, match="provided together"):
        model.training_loss(sample)

    sample["semantic_plan_times"] = torch.tensor(
        [[0.25, 0.5, 0.75, 1.0]]
    ).expand(2, -1)
    del sample["video_fps"]
    with pytest.raises(KeyError, match="video_fps"):
        model.training_loss(sample)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("semantic_plan", torch.zeros(2, 2, 4), "semantic_plan must have shape"),
        (
            "semantic_plan_times",
            torch.zeros(2, 3),
            "semantic_plan_times must have shape",
        ),
        (
            "semantic_plan_times",
            torch.tensor([[0.25, 0.5, 0.5, 1.0]]).expand(2, -1),
            "strictly increasing",
        ),
        (
            "semantic_plan_times",
            torch.tensor([[-0.1, 0.5, 0.75, 1.0]]).expand(2, -1),
            r"\[0, 1\]",
        ),
        (
            "semantic_plan",
            torch.full((2, 3, 4), float("nan")),
            "non-finite",
        ),
    ],
)
def test_offline_semantic_tensors_are_strictly_validated(
    cosmos_module, field, value, message
):
    model = _make_model(cosmos_module)
    sample = _training_sample()
    del sample["instruction"]
    sample["semantic_plan"] = torch.zeros(2, 3, 4)
    sample["semantic_plan_times"] = torch.tensor(
        [[0.25, 0.5, 0.75, 1.0]]
    ).expand(2, -1)
    sample[field] = value

    with pytest.raises(ValueError, match=message):
        model.training_loss(sample)


@pytest.mark.parametrize(
    "fps",
        [
            torch.ones(2, 1),
            torch.tensor([5.0]),
            torch.tensor([5.0, 0.0]),
            torch.tensor([5.0, float("nan")]),
            torch.tensor([True, True]),
            True,
        ],
)
def test_video_fps_shape_and_values_are_strictly_validated(cosmos_module, fps):
    model = _make_model(cosmos_module, provider=_FakeProvider())
    sample = _training_sample()
    sample["video_fps"] = fps

    with pytest.raises(ValueError, match="video_fps"):
        model.training_loss(sample)


def test_legacy_call_without_semantics_does_not_require_fps(cosmos_module):
    model = _make_model(cosmos_module)
    sample = _training_sample()
    del sample["instruction"]
    del sample["video_fps"]

    loss, _metrics = model.training_loss(sample)

    assert torch.isfinite(loss)
    assert model._current_semantic_plan is None
    assert model._current_semantic_plan_times is None
    assert model._current_video_fps is None


def test_each_call_resets_transient_state_including_after_validation_error(
    cosmos_module,
):
    provider = _FakeProvider()
    model = _make_model(cosmos_module, provider=provider)
    model.training_loss(_training_sample())
    object.__setattr__(model, "_online_semantic_planner", None)
    legacy = _training_sample()
    del legacy["instruction"]
    del legacy["video_fps"]
    model.training_loss(legacy)
    assert model._current_semantic_plan is None
    assert model._current_semantic_plan_times is None
    assert model._current_video_fps is None

    object.__setattr__(model, "_online_semantic_planner", provider)
    bad = _training_sample()
    bad["video_fps"] = torch.tensor([0.0, 5.0])
    with pytest.raises(ValueError):
        model.training_loss(bad)
    assert model._current_semantic_plan is None
    assert model._current_semantic_plan_times is None
    assert model._current_video_fps is None


def test_inference_routes_batched_image_instruction_and_fps_before_one_vae_encode(
    cosmos_module,
):
    provider = _FakeProvider()
    vae_calls = []
    model = _make_model(cosmos_module, provider=provider, vae_calls=vae_calls)
    image = torch.linspace(-1.0, 1.0, 2 * 3 * 2 * 2).reshape(2, 3, 2, 2)

    result = model.infer_action(
        input_image=image,
        context=torch.zeros(2, 2, 8),
        instruction=["open drawer", "pick mug"],
        video_fps=torch.tensor([5.0, 4.0]),
        action_horizon=3,
        num_inference_steps=1,
        rand_device="cpu",
    )

    assert len(provider.calls) == 1
    assert torch.equal(provider.calls[0][0], image)
    assert provider.calls[0][1] == ["open drawer", "pick mug"]
    assert len(vae_calls) == 1
    assert torch.equal(vae_calls[0], image.unsqueeze(2))
    assert result["action"].shape == (2, 3, 2)
    assert torch.equal(model._current_video_fps, torch.tensor([5.0, 4.0]))


def test_online_inference_rejects_direct_offline_plan(cosmos_module):
    model = _make_model(cosmos_module, provider=_FakeProvider())
    with pytest.raises(ValueError, match="mutually exclusive"):
        model.infer_action(
            input_image=torch.zeros(1, 3, 2, 2),
            context=torch.zeros(1, 2, 8),
            instruction="task",
            video_fps=5.0,
            semantic_plan=torch.zeros(1, 3, 4),
            semantic_plan_times=torch.tensor([[0.25, 0.5, 0.75, 1.0]]),
            num_inference_steps=1,
        )


def test_online_times_must_match_the_exact_four_fastwam_keyframes(cosmos_module):
    provider = _FakeProvider()
    model = _make_model(cosmos_module, provider=provider)
    original_predict = provider.predict

    def wrong_times(images, instructions):
        result = original_predict(images, instructions)
        result.semantic_plan_times = torch.tensor(
            [[0.2, 0.5, 0.75, 1.0]]
        ).expand(images.shape[0], -1)
        return result

    provider.predict = wrong_times
    with pytest.raises(ValueError, match=r"\[0.25, 0.5, 0.75, 1.0\]"):
        model.training_loss(_training_sample())


def test_runtime_preflights_paths_and_exact_geometry_before_heavy_allocation(
    monkeypatch, tmp_path
):
    runtime = importlib.import_module("fastwam.models.cosmos.runtime")
    calls = []
    monkeypatch.setattr(
        runtime.CosmosVideoExpert,
        "from_pretrained",
        lambda **kwargs: calls.append(kwargs),
    )

    with pytest.raises(FileNotFoundError, match="online planner"):
        runtime.create_fastwam_cosmos(
            video_dit_pretrained_path="unused",
            online_semantic_planner=True,
            online_semantic_planner_code_dir=str(tmp_path / "missing-code"),
            online_semantic_planner_checkpoint=str(tmp_path / "missing-checkpoint"),
            semantic_plan_context=True,
            semantic_plan_in_dim=1024,
            semantic_plan_max_tokens=1024,
            semantic_plan_num_keyframes=4,
            semantic_plan_source_num_keyframes=4,
            semantic_plan_spatial_grid=16,
            device="cpu",
        )
    assert calls == []

    code_dir = tmp_path / "code"
    checkpoint_dir = tmp_path / "checkpoint"
    _write_fake_provider(code_dir)
    checkpoint_dir.mkdir()
    with pytest.raises(ValueError, match="semantic_plan_context"):
        runtime.create_fastwam_cosmos(
            video_dit_pretrained_path="unused",
            online_semantic_planner=True,
            online_semantic_planner_code_dir=str(code_dir),
            online_semantic_planner_checkpoint=str(checkpoint_dir),
            semantic_plan_context=False,
            semantic_plan_in_dim=1024,
            semantic_plan_max_tokens=1024,
            semantic_plan_num_keyframes=4,
            semantic_plan_source_num_keyframes=4,
            semantic_plan_spatial_grid=16,
            device="cpu",
        )
    assert calls == []


@pytest.mark.parametrize("invalid", [1024.5, "1024", True])
def test_runtime_rejects_coercible_but_non_integer_online_geometry(
    monkeypatch, tmp_path, invalid
):
    runtime = importlib.import_module("fastwam.models.cosmos.runtime")
    code_dir = tmp_path / "code"
    checkpoint_dir = tmp_path / "checkpoint"
    _write_fake_provider(code_dir)
    checkpoint_dir.mkdir()
    calls = []
    monkeypatch.setattr(
        runtime.CosmosVideoExpert,
        "from_pretrained",
        lambda **kwargs: calls.append(kwargs),
    )

    with pytest.raises(ValueError, match="exact K4 dense geometry"):
        runtime.create_fastwam_cosmos(
            video_dit_pretrained_path="unused",
            online_semantic_planner=True,
            online_semantic_planner_code_dir=str(code_dir),
            online_semantic_planner_checkpoint=str(checkpoint_dir),
            semantic_plan_context=True,
            semantic_plan_in_dim=invalid,
            semantic_plan_max_tokens=1024,
            semantic_plan_num_keyframes=4,
            semantic_plan_source_num_keyframes=4,
            semantic_plan_spatial_grid=16,
            device="cpu",
        )
    assert calls == []


@pytest.mark.parametrize("invalid", [3.5, "3", True, -1])
def test_model_geometry_is_never_silently_truncated(cosmos_module, invalid):
    with pytest.raises(ValueError, match="semantic_plan_max_tokens"):
        cosmos_module.FastWAMCosmos(
            video_expert=_FakeVideoExpert(),
            action_expert=_FakeActionExpert(),
            vae=None,
            vae_encode_fn=lambda *_args, **_kwargs: None,
            qwen_dim=8,
            crossattn_dim=8,
            coupling="mot",
            device="cpu",
            torch_dtype=torch.float32,
            semantic_plan_dim=4,
            semantic_plan_max_tokens=invalid,
            semantic_plan_num_keyframes=4,
        )


def test_runtime_enables_fusion_and_passes_unregistered_provider(monkeypatch, tmp_path):
    runtime = importlib.import_module("fastwam.models.cosmos.runtime")
    code_dir = tmp_path / "planner/lingbot_dino_4b"
    checkpoint_dir = tmp_path / "checkpoint"
    _write_fake_provider(code_dir)
    checkpoint_dir.mkdir()
    provider = object()
    capture = {}

    class Video:
        def __init__(self):
            self.net = types.SimpleNamespace(
                model_channels=8,
                blocks=[types.SimpleNamespace(self_attn=types.SimpleNamespace(n_heads=1))],
            )

    class Action(nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(1, 1)])

        def copy_init_from_video(self, _net):
            return None

    class Model:
        proprio_encoder = None

        def __init__(self, **kwargs):
            capture["model"] = kwargs
            self.dit = nn.Identity()

    def video_factory(**kwargs):
        capture["video"] = kwargs
        return Video()

    monkeypatch.setattr(runtime.CosmosVideoExpert, "from_pretrained", video_factory)
    monkeypatch.setattr(runtime, "CosmosActionExpert", Action)
    monkeypatch.setattr(runtime, "FastWAMCosmos", Model)
    monkeypatch.setattr(
        runtime,
        "load_online_semantic_planner",
        lambda **kwargs: capture.setdefault("loader", kwargs) and provider,
    )

    runtime.create_fastwam_cosmos(
        video_dit_pretrained_path="video.pt",
        online_semantic_planner=True,
        online_semantic_planner_code_dir=str(code_dir),
        online_semantic_planner_checkpoint=str(checkpoint_dir),
        semantic_plan_context=True,
        semantic_plan_in_dim=1024,
        semantic_plan_hidden_dim=2048,
        semantic_plan_max_tokens=1024,
        semantic_plan_num_keyframes=4,
        semantic_plan_source_num_keyframes=4,
        semantic_plan_spatial_grid=16,
        semantic_plan_initial_depth_gate=0.2,
        model_dtype=torch.float32,
        device="cpu",
    )

    assert capture["video"]["semantic_plan_fusion_enabled"] is True
    assert capture["video"]["semantic_plan_feature_dim"] == 1024
    assert capture["video"]["semantic_plan_fusion_max_tokens"] == 1024
    assert capture["video"]["semantic_plan_initial_depth_gate"] == pytest.approx(0.2)
    assert capture["model"]["online_semantic_planner"] is provider
    assert capture["model"]["semantic_plan_dim"] == 1024
    assert capture["model"]["semantic_plan_max_tokens"] == 1024
    assert capture["model"]["semantic_plan_num_keyframes"] == 4
    assert capture["loader"]["device"] == "cpu"
