import importlib.util
import json
from pathlib import Path
import types

import pytest
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SMOKE_PATH = (
    ROOT
    / "third_party/FastWAM/scripts"
    / "smoke_online_dino_depth_semantic_plan.py"
)


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location(
        "fastwam_checkpoint_smoke_under_test",
        SMOKE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_cli(tmp_path, *, video_fps="5.0"):
    return [
        "--planner-checkpoint",
        "planner",
        "--config-dir",
        "configs",
        "--config-name",
        "train",
        "--device",
        "cuda:0",
        "--instruction",
        "open the middle drawer",
        "--image",
        "current.png",
        "--video-fps",
        video_fps,
    ]


def test_online_smoke_script_requires_real_checkpoint_and_config():
    source = SMOKE_PATH.read_text(encoding="utf-8")

    assert "--planner-checkpoint" in source
    assert "--config-dir" in source
    assert "--config-name" in source
    assert "--task" in source
    assert "--device" in source
    assert "--instruction" in source
    assert "--image" in source
    assert "--video-fps" in source
    assert "torch.inference_mode()" in source


def test_smoke_module_import_is_dependency_light():
    module = _load_smoke_module()

    assert "torch" not in module.__dict__
    assert "hydra" not in module.__dict__
    assert "Image" not in module.__dict__


def test_parse_args_resolves_all_paths_before_runtime_chdir(monkeypatch, tmp_path):
    module = _load_smoke_module()
    monkeypatch.chdir(tmp_path)

    args = module.parse_args(
        _required_cli(tmp_path)
        + [
            "--video-dit-checkpoint",
            "weights/dit.pt",
            "--vae-checkpoint",
            "weights/vae.pt",
            "--text-cache-dir",
            "text-cache",
        ]
    )

    assert args.planner_checkpoint == (tmp_path / "planner").resolve()
    assert args.config_dir == (tmp_path / "configs").resolve()
    assert args.image == (tmp_path / "current.png").resolve()
    assert args.video_dit_checkpoint == (tmp_path / "weights/dit.pt").resolve()
    assert args.vae_checkpoint == (tmp_path / "weights/vae.pt").resolve()
    assert args.text_cache_dir == (tmp_path / "text-cache").resolve()
    assert args.task == "libero_cosmos_2cam224_online_dino_depth"
    assert args.num_inference_steps == 1
    assert args.action_horizon == 1


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_parse_args_rejects_nonpositive_or_nonfinite_fps(tmp_path, value):
    module = _load_smoke_module()

    with pytest.raises(SystemExit):
        module.parse_args(_required_cli(tmp_path, video_fps=value))


@pytest.mark.parametrize("value", [True, False, 0, -1, float("nan"), float("inf")])
def test_runtime_fps_validation_rejects_bool_nonpositive_and_nonfinite(value):
    module = _load_smoke_module()

    with pytest.raises(ValueError, match="positive finite"):
        module.validate_video_fps(value)


@pytest.mark.parametrize("flag", ["--num-inference-steps", "--action-horizon"])
@pytest.mark.parametrize("value", ["0", "-1", "1.5"])
def test_parse_args_rejects_nonpositive_integer_runtime_overrides(
    tmp_path,
    flag,
    value,
):
    module = _load_smoke_module()

    with pytest.raises(SystemExit):
        module.parse_args(_required_cli(tmp_path) + [flag, value])


@pytest.mark.parametrize(
    "flag",
    ["--config-name", "--task", "--device", "--instruction"],
)
def test_parse_args_rejects_blank_runtime_strings(tmp_path, flag):
    module = _load_smoke_module()
    cli = _required_cli(tmp_path)
    if flag == "--task":
        cli += [flag, "   "]
    else:
        cli[cli.index(flag) + 1] = "   "

    with pytest.raises(SystemExit):
        module.parse_args(cli)


def test_validate_checkpoint_uses_exported_provider_contract(monkeypatch, tmp_path):
    module = _load_smoke_module()
    checkpoint = tmp_path / "planner"
    checkpoint.mkdir()
    metadata = {"contract": "strict-k4"}
    (checkpoint / "planner_meta.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    events = []

    provider = types.SimpleNamespace(
        validate_checkpoint_files=lambda path: events.append(
            ("files", Path(path))
        )
        or Path(path).resolve(),
        validate_planner_metadata=lambda payload: events.append(
            ("metadata", payload)
        )
        or object(),
    )
    monkeypatch.setattr(module, "_load_provider_module", lambda: provider)

    resolved = module.validate_checkpoint(checkpoint)

    assert resolved == checkpoint.resolve()
    assert events == [
        ("files", checkpoint),
        ("metadata", metadata),
    ]


def test_validate_checkpoint_rejects_non_object_metadata_before_contract(
    monkeypatch,
    tmp_path,
):
    module = _load_smoke_module()
    checkpoint = tmp_path / "planner"
    checkpoint.mkdir()
    (checkpoint / "planner_meta.json").write_text("[]", encoding="utf-8")
    provider = types.SimpleNamespace(
        validate_checkpoint_files=lambda path: Path(path).resolve(),
        validate_planner_metadata=lambda _payload: pytest.fail(
            "non-object metadata must not reach the provider contract"
        ),
    )
    monkeypatch.setattr(module, "_load_provider_module", lambda: provider)

    with pytest.raises(ValueError, match="metadata.*object"):
        module.validate_checkpoint(checkpoint)


def test_validate_cli_paths_checks_config_image_and_optional_overrides(tmp_path):
    module = _load_smoke_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "train.yaml").touch()
    image = tmp_path / "current.png"
    image.touch()
    dit = tmp_path / "dit.pt"
    dit.touch()
    vae = tmp_path / "vae.pt"
    vae.touch()
    text_cache = tmp_path / "text-cache"
    text_cache.mkdir()
    args = types.SimpleNamespace(
        config_dir=config_dir,
        config_name="train",
        image=image,
        video_dit_checkpoint=dit,
        vae_checkpoint=vae,
        text_cache_dir=text_cache,
    )

    module.validate_cli_paths(args)

    args.video_dit_checkpoint = text_cache
    with pytest.raises(FileNotFoundError, match="video DiT checkpoint"):
        module.validate_cli_paths(args)


def test_load_config_composes_task_and_forces_online_checkpoint_overrides(tmp_path):
    module = _load_smoke_module()
    checkpoint = (tmp_path / "planner").resolve()
    dit = (tmp_path / "dit.pt").resolve()
    vae = (tmp_path / "vae.pt").resolve()
    args = types.SimpleNamespace(
        config_dir=(ROOT / "third_party/FastWAM/configs").resolve(),
        config_name="train",
        task="libero_cosmos_2cam224_online_dino_depth",
        planner_checkpoint=checkpoint,
        video_dit_checkpoint=dit,
        vae_checkpoint=vae,
    )

    cfg = module.load_config(args)

    assert cfg.model.online_semantic_planner is True
    assert cfg.model.online_semantic_planner_checkpoint == str(checkpoint)
    assert cfg.model.online_semantic_planner_code_dir == str(
        module.PLANNER_CODE_DIR.resolve()
    )
    assert cfg.model.video_dit_pretrained_path == str(dit)
    assert cfg.model.vae.vae_pth == str(vae)
    assert cfg.data.train.semantic_plan_source == "online"
    assert cfg.EVALUATION.video_fps == pytest.approx(5.0)


def test_runtime_preflight_runs_before_hydra_model_allocation(monkeypatch):
    module = _load_smoke_module()
    events = []
    cfg = types.SimpleNamespace(
        mixed_precision="bf16",
        model=types.SimpleNamespace(
            online_semantic_planner_code_dir="/planner/code",
            online_semantic_planner_checkpoint="/planner/checkpoint",
        ),
    )
    args = types.SimpleNamespace(device="cuda:0")

    monkeypatch.setattr(
        module,
        "preflight_fastwam_runtime",
        lambda config: events.append(("preflight", config)),
    )

    class Model:
        def eval(self):
            events.append(("eval", self))
            return self

    hydra = types.ModuleType("hydra")
    hydra_utils = types.ModuleType("hydra.utils")

    def instantiate(model_cfg, **kwargs):
        events.append(("instantiate", model_cfg, kwargs))
        return Model()

    hydra_utils.instantiate = instantiate
    hydra.utils = hydra_utils
    monkeypatch.setitem(__import__("sys").modules, "hydra", hydra)
    monkeypatch.setitem(__import__("sys").modules, "hydra.utils", hydra_utils)

    model = module.create_fastwam_cosmos(cfg, args)

    assert isinstance(model, Model)
    assert events[0] == ("preflight", cfg)
    assert events[1][0] == "instantiate"
    assert events[1][1] is cfg.model
    assert events[1][2] == {
        "model_dtype": torch.bfloat16,
        "device": "cuda:0",
    }
    assert events[2][0] == "eval"


def test_runtime_preflight_uses_configured_code_and_checkpoint(monkeypatch):
    module = _load_smoke_module()
    calls = []
    validator = lambda **kwargs: calls.append(kwargs) or (Path("code"), Path("ckpt"))
    monkeypatch.setattr(module, "_load_fastwam_runtime_validator", lambda: validator)
    cfg = types.SimpleNamespace(
        model=types.SimpleNamespace(
            online_semantic_planner_code_dir="/planner/code",
            online_semantic_planner_checkpoint="/planner/checkpoint",
        )
    )

    result = module.preflight_fastwam_runtime(cfg)

    assert result == (Path("code"), Path("ckpt"))
    assert calls == [
        {
            "code_dir": "/planner/code",
            "checkpoint_dir": "/planner/checkpoint",
        }
    ]


def test_load_rgb_tensor_returns_finite_normalized_bchw(tmp_path):
    module = _load_smoke_module()
    image_path = tmp_path / "current.png"
    image = Image.new("RGB", (2, 1))
    image.putdata([(0, 127, 255), (255, 0, 127)])
    image.save(image_path)

    tensor = module.load_rgb_tensor(image_path, device="cpu")

    assert tensor.shape == (1, 3, 1, 2)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    assert tensor.min().item() == pytest.approx(-1.0)
    assert tensor.max().item() == pytest.approx(1.0)
    module.validate_image_tensor(tensor, expected_hw=(1, 2))
    with pytest.raises(ValueError, match="configured video size"):
        module.validate_image_tensor(tensor, expected_hw=(2, 2))


def test_preflight_model_assets_validates_composed_dit_and_vae_paths(tmp_path):
    module = _load_smoke_module()
    dit = tmp_path / "dit.pt"
    vae = tmp_path / "vae.pt"
    dit.touch()
    vae.touch()
    cfg = types.SimpleNamespace(
        model=types.SimpleNamespace(
            video_dit_pretrained_path=str(dit),
            vae=types.SimpleNamespace(vae_pth=str(vae)),
        )
    )

    module.preflight_model_assets(cfg)

    dit.unlink()
    with pytest.raises(FileNotFoundError, match="video DiT checkpoint"):
        module.preflight_model_assets(cfg)


def test_text_cache_is_canonical_and_prompt_entry_is_preflighted(
    monkeypatch,
    tmp_path,
):
    module = _load_smoke_module()
    text_cache = tmp_path / "text-cache"
    text_cache.mkdir()
    instruction = "open the middle drawer"
    prompt = module.format_prompt(instruction)
    expected_entry = module.text_cache_entry(text_cache, prompt, context_len=128)
    expected_entry.touch()
    cfg = types.SimpleNamespace(
        data=types.SimpleNamespace(
            train=types.SimpleNamespace(
                text_embedding_cache_dir="./relative/cache",
                context_len=128,
            )
        )
    )
    args = types.SimpleNamespace(text_cache_dir=text_cache)

    resolved = module.configure_text_cache(args, cfg, prompt)

    assert resolved == text_cache.resolve()
    assert __import__("os").environ["FASTWAM_TEXT_CACHE_DIR"] == str(
        text_cache.resolve()
    )
    assert __import__("os").environ["FASTWAM_TEXT_CONTEXT_LEN"] == "128"

    expected_entry.unlink()
    with pytest.raises(FileNotFoundError, match="prompt text embedding"):
        module.configure_text_cache(args, cfg, prompt)


class _SmokeModel:
    def __init__(self, *, fused_shape=(1, 1024, 1024), action=None, error=None):
        self.video_expert = types.SimpleNamespace(
            semantic_plan_fusion=torch.nn.Identity()
        )
        self.fused_shape = fused_shape
        self.action = torch.ones(1, 7) if action is None else action
        self.error = error
        self.calls = []

    def infer_action(self, **kwargs):
        assert not torch.is_grad_enabled()
        self.calls.append(kwargs)
        self.video_expert.semantic_plan_fusion(torch.zeros(self.fused_shape))
        if self.error is not None:
            raise self.error
        return {"action": self.action}


def _smoke_args(tmp_path):
    return types.SimpleNamespace(
        planner_checkpoint=(tmp_path / "planner").resolve(),
        instruction="open the middle drawer",
        video_fps=5.0,
        num_inference_steps=1,
        action_horizon=1,
    )


def test_run_smoke_calls_infer_once_and_removes_fusion_hook(tmp_path):
    module = _load_smoke_module()
    model = _SmokeModel()
    args = _smoke_args(tmp_path)
    image = torch.zeros(1, 3, 224, 448)

    summary = module.run_smoke(model, image, args)

    assert len(model.calls) == 1
    assert model.calls[0] == {
        "input_image": image,
        "instruction": args.instruction,
        "prompt": module.format_prompt(args.instruction),
        "video_fps": 5.0,
        "num_inference_steps": 1,
        "action_horizon": 1,
    }
    assert model.video_expert.semantic_plan_fusion._forward_hooks == {}
    assert summary == {
        "planner_checkpoint": str(args.planner_checkpoint),
        "fused_plan_shape": (1, 1024, 1024),
        "action_shape": (1, 7),
        "video_fps": 5.0,
    }


def test_run_smoke_removes_hook_when_inference_raises(tmp_path):
    module = _load_smoke_module()
    model = _SmokeModel(error=RuntimeError("inference failed"))

    with pytest.raises(RuntimeError, match="inference failed"):
        module.run_smoke(model, torch.zeros(1, 3, 2, 2), _smoke_args(tmp_path))

    assert model.video_expert.semantic_plan_fusion._forward_hooks == {}
    assert len(model.calls) == 1


def test_run_smoke_fails_fast_when_semantic_fusion_is_missing(tmp_path):
    module = _load_smoke_module()
    model = types.SimpleNamespace(
        video_expert=types.SimpleNamespace(semantic_plan_fusion=None)
    )

    with pytest.raises(RuntimeError, match="semantic fusion is missing"):
        module.run_smoke(model, torch.zeros(1, 3, 2, 2), _smoke_args(tmp_path))


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (_SmokeModel(fused_shape=(1, 512, 1024)), "fused plan shape"),
        (
            _SmokeModel(action=torch.tensor([[float("nan")]])),
            "empty or non-finite",
        ),
        (_SmokeModel(action=torch.empty(0, 7)), "empty or non-finite"),
    ],
)
def test_run_smoke_rejects_invalid_fusion_or_action(tmp_path, model, message):
    module = _load_smoke_module()

    with pytest.raises(RuntimeError, match=message):
        module.run_smoke(model, torch.zeros(1, 3, 2, 2), _smoke_args(tmp_path))

    assert model.video_expert.semantic_plan_fusion._forward_hooks == {}


def test_main_preflights_then_prints_one_json_summary(
    monkeypatch,
    capsys,
    tmp_path,
):
    module = _load_smoke_module()
    args = _smoke_args(tmp_path)
    args.config_dir = tmp_path / "configs"
    args.config_name = "train"
    args.task = module.DEFAULT_TASK
    args.device = "cpu"
    args.image = tmp_path / "current.png"
    args.video_dit_checkpoint = None
    args.vae_checkpoint = None
    args.text_cache_dir = None
    cfg = types.SimpleNamespace(
        data=types.SimpleNamespace(
            train=types.SimpleNamespace(video_size=[224, 448])
        )
    )
    image = torch.zeros(1, 3, 224, 448)
    model = object()
    summary = {
        "planner_checkpoint": str(args.planner_checkpoint),
        "fused_plan_shape": (1, 1024, 1024),
        "action_shape": (1, 7),
        "video_fps": 5.0,
    }
    events = []
    monkeypatch.setattr(module, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(
        module,
        "validate_cli_paths",
        lambda value: events.append(("cli", value)),
    )
    monkeypatch.setattr(
        module,
        "validate_checkpoint",
        lambda path: events.append(("checkpoint", path)) or path,
    )
    monkeypatch.setattr(
        module,
        "load_config",
        lambda value: events.append(("config", value)) or cfg,
    )
    monkeypatch.setattr(
        module,
        "preflight_model_assets",
        lambda value: events.append(("assets", value)),
    )
    monkeypatch.setattr(
        module,
        "configure_text_cache",
        lambda a, c, p: events.append(("text", a, c, p)),
    )
    monkeypatch.setattr(
        module,
        "load_rgb_tensor",
        lambda path, device: events.append(("image", path, device)) or image,
    )
    monkeypatch.setattr(
        module,
        "validate_image_tensor",
        lambda value, expected_hw: events.append(
            ("geometry", value, expected_hw)
        ),
    )
    monkeypatch.setattr(
        module,
        "create_fastwam_cosmos",
        lambda c, a: events.append(("model", c, a)) or model,
    )
    monkeypatch.setattr(
        module,
        "run_smoke",
        lambda m, i, a: events.append(("infer", m, i, a)) or summary,
    )

    module.main([])

    assert [event[0] for event in events] == [
        "cli",
        "checkpoint",
        "config",
        "assets",
        "text",
        "image",
        "geometry",
        "model",
        "infer",
    ]
    output = capsys.readouterr().out.strip().splitlines()
    assert len(output) == 1
    assert json.loads(output[0]) == {
        "planner_checkpoint": str(args.planner_checkpoint),
        "fused_plan_shape": [1, 1024, 1024],
        "action_shape": [1, 7],
        "video_fps": 5.0,
    }
