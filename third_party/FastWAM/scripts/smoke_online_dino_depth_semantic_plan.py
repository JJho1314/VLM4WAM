#!/usr/bin/env python3
"""Run one checkpoint-backed FastWAM Cosmos online-planner inference."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import sys


DEFAULT_TASK = "libero_cosmos_2cam224_online_dino_depth"
REPO_ROOT = Path(__file__).resolve().parents[3]
FASTWAM_ROOT = Path(__file__).resolve().parents[1]
FASTWAM_SRC = FASTWAM_ROOT / "src"
PLANNER_CODE_DIR = REPO_ROOT / "scripts/qwen3_vl_semantic_planner/lingbot_dino_4b"
DEFAULT_COSMOS_REPO = REPO_ROOT / "third_party/cosmos-predict2.5"


def _resolved_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _positive_finite_float(value: str) -> float:
    try:
        parsed = validate_video_fps(float(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive finite number") from error
    return parsed


def validate_video_fps(value) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"video FPS must be a positive finite number, got {value!r}")
    return float(value)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _non_empty_string(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise argparse.ArgumentTypeError("expected a non-empty string")
    return value


def _default_cosmos_repo() -> Path:
    configured = os.environ.get("COSMOS_REPO")
    if configured is not None and configured.strip():
        return _resolved_path(configured)
    return DEFAULT_COSMOS_REPO.resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-checkpoint", type=_resolved_path, required=True)
    parser.add_argument("--config-dir", type=_resolved_path, required=True)
    parser.add_argument("--config-name", type=_non_empty_string, required=True)
    parser.add_argument("--task", type=_non_empty_string, default=DEFAULT_TASK)
    parser.add_argument("--device", type=_non_empty_string, required=True)
    parser.add_argument("--instruction", type=_non_empty_string, required=True)
    parser.add_argument("--image", type=_resolved_path, required=True)
    parser.add_argument("--video-fps", type=_positive_finite_float, required=True)
    parser.add_argument("--video-dit-checkpoint", type=_resolved_path)
    parser.add_argument("--vae-checkpoint", type=_resolved_path)
    parser.add_argument("--text-cache-dir", type=_resolved_path)
    parser.add_argument(
        "--cosmos-repo",
        type=_resolved_path,
        default=_default_cosmos_repo(),
    )
    parser.add_argument(
        "--num-inference-steps",
        type=_positive_int,
        default=1,
    )
    parser.add_argument("--action-horizon", type=_positive_int, default=1)
    return parser.parse_args(argv)


def _load_provider_module():
    _ensure_runtime_paths()
    provider_path = PLANNER_CODE_DIR / "dino_depth_plan_provider.py"
    if not provider_path.is_file():
        raise FileNotFoundError(f"planner provider file not found: {provider_path}")
    path_digest = hashlib.sha256(str(provider_path).encode("utf-8")).hexdigest()[:16]
    module_name = f"_fastwam_smoke_planner_provider_{path_digest}"
    spec = importlib.util.spec_from_file_location(module_name, provider_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create planner provider spec for {provider_path}")
    provider = importlib.util.module_from_spec(spec)
    had_previous = module_name in sys.modules
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = provider
    try:
        spec.loader.exec_module(provider)
    except Exception as error:
        if had_previous:
            sys.modules[module_name] = previous
        else:
            sys.modules.pop(module_name, None)
        raise ImportError(
            f"failed to import planner provider from {provider_path}: {error}"
        ) from error
    return provider


def validate_checkpoint(checkpoint_dir: str | Path) -> Path:
    """Validate the public planner export contract before importing FastWAM."""
    provider = _load_provider_module()
    checkpoint = provider.validate_checkpoint_files(checkpoint_dir)
    metadata_path = checkpoint / "planner_meta.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid planner metadata {metadata_path}: {error}"
        ) from error
    if not isinstance(metadata, dict):
        raise ValueError(
            f"invalid planner metadata {metadata_path}: expected an object"
        )
    provider.validate_planner_metadata(metadata)
    return checkpoint.resolve()


def _config_file_candidates(config_dir: Path, config_name: str):
    requested = config_dir / config_name
    if requested.suffix in {".yaml", ".yml"}:
        return (requested,)
    return (requested.with_suffix(".yaml"), requested.with_suffix(".yml"))


def validate_cli_paths(args) -> None:
    """Reject invalid runtime assets before Hydra or model imports."""
    if not args.cosmos_repo.is_dir():
        raise FileNotFoundError(
            f"Cosmos repository directory not found: {args.cosmos_repo}"
        )
    cosmos_package = args.cosmos_repo / "cosmos_predict2"
    cosmos_marker = cosmos_package / "__init__.py"
    if not cosmos_package.is_dir() or not cosmos_marker.is_file():
        raise FileNotFoundError(
            "Cosmos repository must contain an importable cosmos_predict2 "
            f"package with {cosmos_marker}"
        )
    if not args.config_dir.is_dir():
        raise FileNotFoundError(f"Hydra config directory not found: {args.config_dir}")
    config_candidates = _config_file_candidates(args.config_dir, args.config_name)
    if not any(path.is_file() for path in config_candidates):
        raise FileNotFoundError(
            "Hydra config not found; expected one of: "
            + ", ".join(str(path) for path in config_candidates)
        )
    if not args.image.is_file():
        raise FileNotFoundError(f"input RGB image not found: {args.image}")
    for label, path in (
        ("video DiT checkpoint", args.video_dit_checkpoint),
        ("VAE checkpoint", args.vae_checkpoint),
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.text_cache_dir is not None and not args.text_cache_dir.is_dir():
        raise FileNotFoundError(
            f"text embedding cache directory not found: {args.text_cache_dir}"
        )


def _ensure_runtime_paths(cosmos_repo: str | Path | None = None) -> None:
    for path in (REPO_ROOT, FASTWAM_SRC):
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    if cosmos_repo is not None:
        resolved_cosmos_repo = str(Path(cosmos_repo).expanduser().resolve())
        sys.path[:] = [
            entry
            for entry in sys.path
            if str(Path(entry or ".").expanduser().resolve()) != resolved_cosmos_repo
        ]
        sys.path.insert(0, resolved_cosmos_repo)
    importlib.invalidate_caches()


def load_config(args):
    """Compose the selected task and apply canonical smoke-only overrides."""
    _ensure_runtime_paths()
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(
        config_dir=str(args.config_dir),
        version_base=None,
    ):
        cfg = compose(
            config_name=args.config_name,
            overrides=[f"task={args.task}"],
        )
    OmegaConf.update(cfg, "model.online_semantic_planner", True)
    OmegaConf.update(
        cfg,
        "model.online_semantic_planner_checkpoint",
        str(args.planner_checkpoint),
    )
    OmegaConf.update(
        cfg,
        "model.online_semantic_planner_code_dir",
        str(PLANNER_CODE_DIR.resolve()),
    )
    if args.video_dit_checkpoint is not None:
        OmegaConf.update(
            cfg,
            "model.video_dit_pretrained_path",
            str(args.video_dit_checkpoint),
        )
    if args.vae_checkpoint is not None:
        OmegaConf.update(
            cfg,
            "model.vae.vae_pth",
            str(args.vae_checkpoint),
        )
    return cfg


def _load_fastwam_runtime_validator():
    _ensure_runtime_paths()
    from fastwam.models.cosmos.online_semantic_planner import (
        validate_online_semantic_planner_paths,
    )

    return validate_online_semantic_planner_paths


def preflight_fastwam_runtime(cfg):
    """Run FastWAM's complete path/geometry gate before Cosmos allocation."""
    validator = _load_fastwam_runtime_validator()
    return validator(
        code_dir=str(cfg.model.online_semantic_planner_code_dir),
        checkpoint_dir=str(cfg.model.online_semantic_planner_checkpoint),
    )


def _model_dtype(mixed_precision: str):
    import torch

    precision = str(mixed_precision).strip().lower()
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(
        f"unsupported mixed_precision {mixed_precision!r}; expected no, fp16, or bf16"
    )


def create_fastwam_cosmos(cfg, args):
    """Preflight and instantiate the real Hydra model on the requested device."""
    _ensure_runtime_paths(args.cosmos_repo)
    preflight_fastwam_runtime(cfg)
    from hydra.utils import instantiate

    model = instantiate(
        cfg.model,
        model_dtype=_model_dtype(cfg.mixed_precision),
        device=args.device,
    )
    return model.eval()


def _resolve_config_path(value: str | Path) -> Path:
    expanded = Path(os.path.expandvars(str(value))).expanduser()
    if not expanded.is_absolute():
        expanded = FASTWAM_ROOT / expanded
    return expanded.resolve()


def preflight_model_assets(cfg) -> None:
    """Validate model assets selected by the composed config."""
    video_checkpoint = _resolve_config_path(cfg.model.video_dit_pretrained_path)
    if not video_checkpoint.is_file():
        raise FileNotFoundError(f"video DiT checkpoint not found: {video_checkpoint}")
    vae = cfg.model.vae
    if vae is not None:
        vae_checkpoint = _resolve_config_path(vae.vae_pth)
        if not vae_checkpoint.is_file():
            raise FileNotFoundError(f"VAE checkpoint not found: {vae_checkpoint}")


def format_prompt(instruction: str) -> str:
    return (
        "A video recorded from a robot's point of view executing "
        f"the following instruction: {instruction}"
    )


def text_cache_entry(
    cache_dir: str | Path,
    prompt: str,
    *,
    context_len: int,
) -> Path:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return Path(cache_dir) / (f"{digest}.t5_len{context_len}.wan22ti2v5b.pt")


def configure_text_cache(args, cfg, prompt: str) -> Path:
    """Pin cwd-independent text-cache semantics and validate this prompt."""
    configured = args.text_cache_dir
    if configured is None:
        configured = os.environ.get("FASTWAM_TEXT_CACHE_DIR")
    if configured is None:
        configured = cfg.data.train.text_embedding_cache_dir
    cache_dir = _resolve_config_path(configured)
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"text embedding cache directory not found: {cache_dir}"
        )
    context_len = cfg.data.train.context_len
    if (
        isinstance(context_len, bool)
        or not isinstance(context_len, int)
        or context_len <= 0
    ):
        raise ValueError(
            f"text context length must be a positive integer, got {context_len!r}"
        )
    entry = text_cache_entry(cache_dir, prompt, context_len=context_len)
    if not entry.is_file():
        raise FileNotFoundError(f"prompt text embedding not found: {entry}")
    os.environ["FASTWAM_TEXT_CACHE_DIR"] = str(cache_dir)
    os.environ["FASTWAM_TEXT_CONTEXT_LEN"] = str(context_len)
    return cache_dir


def load_rgb_tensor(image_path: str | Path, *, device: str):
    """Load one RGB observation as finite normalized BCHW float32."""
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(image_path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) <= 0:
        raise ValueError(
            f"RGB image must have shape [H, W, 3], got {tuple(array.shape)}"
        )
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32)
    tensor = tensor.mul(2.0 / 255.0).sub(1.0)
    validate_image_tensor(tensor)
    return tensor


def validate_image_tensor(image, *, expected_hw=None) -> None:
    import torch

    if image.ndim != 4 or tuple(image.shape[:2]) != (1, 3):
        raise ValueError(
            f"input image must have shape [1, 3, H, W], got {tuple(image.shape)}"
        )
    if image.shape[2] <= 0 or image.shape[3] <= 0:
        raise ValueError(
            f"input image has empty spatial dimensions: {tuple(image.shape)}"
        )
    if not torch.isfinite(image).all():
        raise ValueError("input image contains non-finite values")
    if image.min().item() < -1.0001 or image.max().item() > 1.0001:
        raise ValueError("input image must be normalized to [-1, 1]")
    if expected_hw is not None and tuple(image.shape[-2:]) != tuple(expected_hw):
        raise ValueError(
            "input image does not match configured video size: "
            f"expected {tuple(expected_hw)}, got {tuple(image.shape[-2:])}"
        )


def configured_video_hw(cfg) -> tuple[int, int]:
    raw_size = cfg.data.train.video_size
    try:
        height, width = raw_size
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"data.train.video_size must be [height, width], got {raw_size!r}"
        ) from error
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (height, width)
    ):
        raise ValueError(
            f"data.train.video_size must contain positive integers, got {raw_size!r}"
        )
    return int(height), int(width)


def validate_device(device: str) -> None:
    import torch

    try:
        target = torch.device(device)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid torch device {device!r}") from error
    if target.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    if target.index is not None and target.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device index out of range: {device}; "
            f"available devices={torch.cuda.device_count()}"
        )


def run_smoke(model, image, args) -> dict:
    """Run exactly one inference while proving online fusion was exercised."""
    import torch

    video_fps = validate_video_fps(args.video_fps)
    video_expert = getattr(model, "video_expert", None)
    fusion = getattr(video_expert, "semantic_plan_fusion", None)
    if fusion is None or not callable(getattr(fusion, "register_forward_hook", None)):
        raise RuntimeError("online semantic fusion is missing from the video expert")

    captured_shapes = []

    def capture_fused_plan(_module, _inputs, output):
        shape = getattr(output, "shape", None)
        captured_shapes.append(None if shape is None else tuple(shape))

    handle = fusion.register_forward_hook(capture_fused_plan)
    try:
        with torch.inference_mode():
            result = model.infer_action(
                input_image=image,
                instruction=args.instruction,
                prompt=format_prompt(args.instruction),
                video_fps=video_fps,
                num_inference_steps=args.num_inference_steps,
                action_horizon=args.action_horizon,
            )
    finally:
        handle.remove()

    if captured_shapes != [(1, 1024, 1024)]:
        observed = captured_shapes[0] if len(captured_shapes) == 1 else captured_shapes
        raise RuntimeError(f"unexpected fused plan shape: {observed}")
    if not isinstance(result, dict) or "action" not in result:
        raise RuntimeError("infer_action must return a dictionary containing action")
    actions = result["action"]
    if (
        not isinstance(actions, torch.Tensor)
        or actions.ndim < 2
        or actions.shape[-2] <= 0
        or actions.numel() == 0
        or not bool(torch.isfinite(actions).all().item())
    ):
        raise RuntimeError("action output is empty or non-finite")
    return {
        "planner_checkpoint": str(Path(args.planner_checkpoint).resolve()),
        "fused_plan_shape": captured_shapes[0],
        "action_shape": tuple(actions.shape),
        "video_fps": video_fps,
    }


def main(argv=None):
    args = parse_args(argv)
    validate_cli_paths(args)
    validate_device(args.device)
    args.planner_checkpoint = validate_checkpoint(args.planner_checkpoint)
    cfg = load_config(args)
    preflight_model_assets(cfg)
    prompt = format_prompt(args.instruction)
    configure_text_cache(args, cfg, prompt)
    image = load_rgb_tensor(args.image, device=args.device)
    validate_image_tensor(image, expected_hw=configured_video_hw(cfg))
    model = create_fastwam_cosmos(cfg, args)
    summary = run_smoke(model, image, args)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
