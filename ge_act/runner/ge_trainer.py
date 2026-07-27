import hashlib
import math
import os
import random
import stat

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Sequence

from datetime import datetime, timedelta
import argparse
import json
import shutil

from yaml import Loader, load
from tqdm import tqdm
import torch
from torch import distributed as dist
from einops import rearrange
from copy import deepcopy
import transformers
import logging

# ----------------------------------------------------
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

# ----------------------------------------------------
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)

# ----------------------------------------------------
from utils.model_utils import load_condition_models, load_latent_models, load_vae_models, load_diffusion_model, count_model_parameters, unwrap_model
from utils.model_utils import forward_pass
from utils.optimizer_utils import get_optimizer
from utils.memory_utils import get_memory_statistics, free_memory

# ----------------------------------------------------
from torch.utils.tensorboard import SummaryWriter
from utils import init_logging, import_custom_class, save_video

# ----------------------------------------------------
from utils.data_utils import get_latents, get_text_conditions, gen_noise_from_condition_frame_latent, randn_tensor, apply_color_jitter_to_video

# ----------------------------------------------------
from utils.extra_utils import act_metric
from models.ltx_models.semantic_conditioning import (
    OnlineSiglip2SemanticEncoder,
    build_patch_center_positions,
    build_semantic_plan_times,
    select_future_keyframes,
)
from models.ltx_models.baton_semantic_planner import (
    FrozenDualCameraBatonPlanner,
)
from models.ltx_models.vlm_semantic_planner import (
    FrozenDualCameraVLMPlanner,
    configure_qwen_top_layers_for_joint_training,
    load_qwen35_grounded_provider,
)
from data.libero_fastwam_hdf5_dataset import (
    BATON_SAMPLING_ALGORITHM,
    BATON_SAMPLING_VERSION,
)
from qwen35_baton.hashing import sha256_artifact, sha256_file, sha256_json
from qwen35_baton.teacher import FrozenSiglip2Teacher

LOG_LEVEL = "INFO"
# LOG_LEVEL = "DEBUG"
logger = get_logger("wm_runner")
logger.setLevel(LOG_LEVEL)


BATON_TEACHER_SOURCE = "qwen35_baton_teacher"
BATON_PREDICTION_SOURCE = "qwen35_baton_prediction"
BATON_SOURCES = frozenset(
    (BATON_TEACHER_SOURCE, BATON_PREDICTION_SOURCE)
)
BATON_FUTURE_INDICES = (0, 3, 5, 8)


@dataclass(frozen=True)
class BatonConditioningComponents:
    """Mutually exclusive frozen condition provider owned outside GE-Act."""

    source: str
    teacher: FrozenSiglip2Teacher | None
    planner: FrozenDualCameraBatonPlanner | None

    def __post_init__(self) -> None:
        if self.source not in BATON_SOURCES:
            raise ValueError(f"unknown Baton conditioning source: {self.source!r}")
        expected_teacher = self.source == BATON_TEACHER_SOURCE
        if (self.teacher is not None) != expected_teacher:
            raise ValueError("Baton teacher ownership does not match source")
        if (self.planner is not None) == expected_teacher:
            raise ValueError("Baton planner ownership does not match source")


@dataclass(frozen=True)
class BatonSemanticCondition:
    """Complete full-grid Baton tensors passed to one GE-Act forward."""

    tokens: torch.Tensor
    times: torch.Tensor
    positions: torch.Tensor
    mask: None = None
    relevance: None = None


@dataclass(frozen=True)
class BatonValidationSelection:
    source: str
    mode: str
    tokens: torch.Tensor
    condition_mask: torch.Tensor


_BATON_FORBIDDEN_FIELD_KINDS = {
    "hindsight_cache": "hindsight cache",
    "cache_dir": "hindsight cache",
    "cache_path": "hindsight cache",
    "cached_targets": "hindsight cache",
    "planner_aux_loss": "planner auxiliary loss",
    "planner_aux_weight": "planner auxiliary loss",
    "qwen_ge_gradient_scale": "planner auxiliary loss",
    "qwen_top_lr": "planner auxiliary loss",
    "qwen_vision_lr": "planner auxiliary loss",
    "relevance": "relevance",
    "semantic_plan_relevance": "relevance",
    "use_relevance": "relevance",
    "mask": "mask",
    "semantic_plan_mask": "mask",
    "token_mask": "mask",
}


def _config_mapping(config: Any) -> Mapping[str, Any]:
    if isinstance(config, argparse.Namespace):
        return vars(config)
    if not isinstance(config, Mapping):
        raise TypeError("Baton config must be a mapping or argparse namespace")
    return config


def _walk_config_fields(
    value: Any,
    *,
    prefix: str = "",
):
    if not isinstance(value, Mapping):
        return
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        yield path, str(key), child
        yield from _walk_config_fields(child, prefix=path)


def _reject_forbidden_baton_fields(config: Mapping[str, Any]) -> None:
    for path, key, value in _walk_config_fields(config):
        key_lower = key.lower()
        kind = _BATON_FORBIDDEN_FIELD_KINDS.get(key_lower)
        if kind is None and "hindsight" in key_lower:
            kind = "hindsight cache"
        if kind is None and "aux" in key_lower and "planner" in key_lower:
            kind = "planner auxiliary loss"
        if kind is not None and value is not None:
            raise ValueError(f"Baton configs reject {kind} field {path}")


def _validated_baton_semantic_config(
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    _reject_forbidden_baton_fields(config)
    semantic = config.get("semantic_plan", config)
    if not isinstance(semantic, Mapping):
        raise TypeError("semantic_plan must be a mapping")
    source = semantic.get("source")
    if source not in BATON_SOURCES:
        raise ValueError(
            "Baton semantic_plan.source must be exactly "
            "qwen35_baton_teacher or qwen35_baton_prediction"
        )
    if semantic.get("enabled", True) is not True:
        raise ValueError("Baton semantic_plan.enabled must be true")
    if semantic.get("keyframe_indices") != list(BATON_FUTURE_INDICES):
        raise ValueError("Baton keyframe_indices must be [0, 3, 5, 8]")
    if semantic.get("tokens_per_frame") != 256:
        raise ValueError("Baton tokens_per_frame must be 256")
    if semantic.get("feature_dim") != 1024:
        raise ValueError("Baton feature_dim must be 1024")

    teacher_only = {"frame_microbatch_size"}
    planner_only = {
        "planner_checkpoint",
        "expected_planner_topology",
        "qwen_model_path",
        "qwen_tokenizer_path",
        "qwen_processor_path",
    }
    if source == BATON_TEACHER_SOURCE:
        ambiguous = sorted(planner_only.intersection(semantic))
        if ambiguous:
            raise ValueError(
                "Baton teacher source rejects planner fields: "
                + ", ".join(ambiguous)
            )
        required = {
            "siglip2_model_path",
            "siglip2_config_hash",
            "siglip2_artifact_hash",
            "teacher_preprocessing_hash",
        }
    else:
        ambiguous = sorted(teacher_only.intersection(semantic))
        if ambiguous:
            raise ValueError(
                "Baton prediction source rejects teacher fields: "
                + ", ".join(ambiguous)
            )
        required = planner_only | {"siglip2_model_path"}
    missing = sorted(
        field
        for field in required
        if not isinstance(semantic.get(field), str)
        or not semantic[field].strip()
    )
    if missing:
        raise ValueError(
            f"{source} is missing required local artifact fields: "
            + ", ".join(missing)
        )
    return semantic, source


def validate_baton_siglip2_provenance(
    semantic: Mapping[str, Any],
) -> None:
    """Validate Stage-2's local SigLIP tree before constructing its teacher."""

    model_path = Path(str(semantic["siglip2_model_path"])).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"local SigLIP2 model directory does not exist: {model_path}"
        )
    from qwen35_baton.cli.preflight import _siglip_geometry

    geometry = _siglip_geometry(model_path)
    if geometry != {"image_size": 256, "patch_size": 16, "hidden_size": 1024}:
        raise ValueError("SigLIP2 geometry differs from the 256/16/1024 contract")
    config_hash = sha256_file(model_path / "config.json")
    artifact_hash = sha256_artifact(model_path)
    if config_hash != semantic["siglip2_config_hash"]:
        raise ValueError("SigLIP2 config hash mismatch")
    if artifact_hash != semantic["siglip2_artifact_hash"]:
        raise ValueError("SigLIP2 artifact hash mismatch")
    if artifact_hash != semantic["teacher_preprocessing_hash"]:
        raise ValueError("SigLIP2 preprocessing hash mismatch")


def prepare_baton_conditioning(
    config: Mapping[str, Any] | argparse.Namespace,
    dataset: Any,
    device: torch.device | str,
    dtype: torch.dtype,
) -> BatonConditioningComponents:
    """Construct exactly one frozen Baton source after fail-closed validation."""

    config_mapping = _config_mapping(config)
    semantic, source = _validated_baton_semantic_config(config_mapping)
    if getattr(dataset, "hindsight_cache", None) is not None:
        raise ValueError("Baton conditioning rejects hindsight cache datasets")
    target_device = torch.device(device)
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Baton conditioning dtype must be float32 or bfloat16")

    if source == BATON_TEACHER_SOURCE:
        validate_baton_siglip2_provenance(semantic)
        teacher = FrozenSiglip2Teacher(
            semantic["siglip2_model_path"],
            device=target_device,
            dtype=dtype,
            frame_microbatch_size=int(
                semantic.get("frame_microbatch_size", 32)
            ),
        )
        freeze_conditioning_modules(teacher)
        return BatonConditioningComponents(
            source=source,
            teacher=teacher,
            planner=None,
        )

    planner = FrozenDualCameraBatonPlanner.from_checkpoint(
        semantic["planner_checkpoint"],
        qwen_model_path=semantic["qwen_model_path"],
        qwen_tokenizer_path=semantic["qwen_tokenizer_path"],
        qwen_processor_path=semantic["qwen_processor_path"],
        siglip2_model_path=semantic["siglip2_model_path"],
        expected_planner_topology=semantic["expected_planner_topology"],
        device=target_device,
        torch_dtype=dtype,
    )
    freeze_conditioning_modules(planner)
    return BatonConditioningComponents(
        source=source,
        teacher=None,
        planner=planner,
    )


def _normalized_video_to_uint8(images: torch.Tensor) -> torch.Tensor:
    if not images.dtype.is_floating_point:
        if images.dtype != torch.uint8:
            raise TypeError("Baton RGB must be uint8 or normalized floating point")
        return images
    if not bool(torch.isfinite(images).all()):
        raise ValueError("Baton RGB must be finite")
    if bool(((images < -1) | (images > 1)).any()):
        raise ValueError("normalized Baton RGB must be in [-1,1]")
    return images.add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8)


def build_baton_semantic_condition(
    components: BatonConditioningComponents,
    semantic_config: Mapping[str, Any],
    video: torch.Tensor,
    instructions: Sequence[str],
    *,
    n_previous: int,
    num_future_frames: int,
    num_latent_frames: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> BatonSemanticCondition:
    """Build teacher or predicted full grids with exact times and positions."""

    if not isinstance(components, BatonConditioningComponents):
        raise TypeError("components must be BatonConditioningComponents")
    semantic, source = _validated_baton_semantic_config(semantic_config)
    if source != components.source:
        raise ValueError("Baton components and semantic config source differ")
    if (
        not isinstance(video, torch.Tensor)
        or video.ndim != 6
        or tuple(video.shape[1:3]) != (3, 2)
    ):
        raise ValueError("video must have shape [B,3,2,T,H,W]")
    batch_size = int(video.shape[0])
    if (
        isinstance(instructions, (str, bytes))
        or not isinstance(instructions, Sequence)
        or len(instructions) != batch_size
        or any(type(value) is not str or not value.strip() for value in instructions)
    ):
        raise ValueError("instructions must be one nonblank string per sample")
    if type(n_previous) is not int or n_previous <= 0:
        raise ValueError("n_previous must be a positive integer")

    target_device = torch.device(device)
    if source == BATON_TEACHER_SOURCE:
        if video.shape[3] < n_previous + num_future_frames:
            raise ValueError("teacher video does not contain the full future clip")
        future = video[:, :, :, n_previous : n_previous + num_future_frames]
        keyframes = select_future_keyframes(
            rearrange(future, "b c v t h w -> b v t c h w"),
            indices=BATON_FUTURE_INDICES,
        ).contiguous()
        if components.teacher is None:
            raise RuntimeError("Baton teacher component is missing")
        with torch.no_grad():
            tokens = components.teacher.encode_future(keyframes)
        positions = build_patch_center_positions(
            batch_size,
            2,
            4,
            device=tokens.device,
        )
    else:
        if video.shape[3] <= n_previous - 1:
            raise ValueError("video does not contain the last current observation")
        if components.planner is None:
            raise RuntimeError("Baton planner component is missing")
        current = video[:, :, :, n_previous - 1].permute(0, 2, 1, 3, 4)
        current = _normalized_video_to_uint8(current).contiguous()
        with torch.no_grad():
            plan = components.planner.predict(current, tuple(instructions))
        if getattr(plan, "future_indices", None) != BATON_FUTURE_INDICES:
            raise ValueError("Baton provider future_indices must be (0, 3, 5, 8)")
        if getattr(plan, "relevance", None) is not None:
            raise ValueError("Baton provider must not return relevance")
        for field in ("mask", "token_mask", "key_mask"):
            if getattr(plan, field, None) is not None:
                raise ValueError("Baton provider must not return token masks")
        tokens = getattr(plan, "tokens", None)
        positions = getattr(plan, "positions_xy", None)

    expected_tokens = (batch_size, 2, 4, 256, 1024)
    if (
        not isinstance(tokens, torch.Tensor)
        or tuple(tokens.shape) != expected_tokens
        or not tokens.dtype.is_floating_point
        or not bool(torch.isfinite(tokens).all())
    ):
        raise ValueError("Baton tokens must be finite floating [B,2,4,256,1024]")
    expected_positions = build_patch_center_positions(
        batch_size,
        2,
        4,
        device=positions.device if isinstance(positions, torch.Tensor) else None,
    )
    if (
        not isinstance(positions, torch.Tensor)
        or positions.dtype != torch.float32
        or not torch.equal(positions, expected_positions)
    ):
        raise ValueError("Baton positions must be exact float32 patch centers")
    tokens = tokens.detach().to(device=target_device, dtype=dtype)
    positions = positions.detach().to(device=target_device, dtype=torch.float32)
    times = build_semantic_plan_times(
        batch_size=batch_size,
        n_view=2,
        n_previous=n_previous,
        num_future_frames=num_future_frames,
        num_latent_frames=num_latent_frames,
        indices=BATON_FUTURE_INDICES,
        device=target_device,
        dtype=torch.float32,
    )
    return BatonSemanticCondition(
        tokens=tokens,
        times=times,
        positions=positions,
    )


def apply_baton_validation_mode(
    components: BatonConditioningComponents,
    *,
    tokens: torch.Tensor,
    mode: str,
    batch_size: int,
    n_view: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> BatonValidationSelection:
    expected = (
        "teacher"
        if components.source == BATON_TEACHER_SOURCE
        else "prediction"
    )
    if mode not in (expected, "semantic_disabled"):
        raise ValueError(
            f"{components.source} validation mode must be {expected} "
            "or semantic_disabled"
        )
    value = 0.0 if mode == "semantic_disabled" else 1.0
    condition_mask = torch.full(
        (batch_size * n_view,),
        value,
        device=device,
        dtype=dtype,
    )
    return BatonValidationSelection(
        source=components.source,
        mode=mode,
        tokens=tokens,
        condition_mask=condition_mask,
    )


def baton_validation_metric_name(source: str, mode: str, metric: str) -> str:
    if source not in BATON_SOURCES:
        raise ValueError(f"unknown Baton metric source: {source}")
    if metric not in {"video", "action"}:
        raise ValueError("Baton validation metric must be video or action")
    return f"validation/{source}/{mode}/{metric}"


def forward_baton_ge_act(
    model: torch.nn.Module,
    condition: BatonSemanticCondition,
    *,
    semantic_condition_mask: torch.Tensor,
    diffusion_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Forward one complete Baton grid through the normal GE-Act model."""

    if not isinstance(condition, BatonSemanticCondition):
        raise TypeError("condition must be a BatonSemanticCondition")
    if tuple(condition.tokens.shape[1:]) != (2, 4, 256, 1024):
        raise ValueError("Baton forward requires [B,2,4,256,1024] tokens")
    if (
        tuple(condition.positions.shape)
        != (*condition.tokens.shape[:4], 2)
        or condition.mask is not None
        or condition.relevance is not None
    ):
        raise ValueError("Baton forward requires positions and no mask/relevance")
    if tuple(semantic_condition_mask.shape) != (
        condition.tokens.shape[0] * condition.tokens.shape[1],
    ):
        raise ValueError("Baton gate requires one value per sample camera")
    semantic_fields = {
        "semantic_plan",
        "semantic_plan_times",
        "semantic_plan_positions",
        "semantic_plan_mask",
        "semantic_plan_relevance",
        "semantic_condition_mask",
    }
    overlap = semantic_fields & set(diffusion_kwargs)
    if overlap:
        raise ValueError(
            "Baton forward constructs semantic fields: "
            + ", ".join(sorted(overlap))
        )
    return forward_pass(
        model=model,
        semantic_plan=condition.tokens,
        semantic_plan_times=condition.times,
        semantic_plan_positions=condition.positions,
        semantic_plan_mask=None,
        semantic_plan_relevance=None,
        semantic_condition_mask=semantic_condition_mask,
        **dict(diffusion_kwargs),
    )


def compute_effective_video_fps(data_config: Dict[str, Any], default_source_fps: float = 30.0) -> float:
    """Convert source control FPS to the sampled video FPS used by GE-Act."""

    source_fps = float(data_config.get("source_fps", default_source_fps))
    chunk = int(data_config["chunk"])
    action_chunk = int(data_config["action_chunk"])
    if chunk <= 0 or action_chunk % chunk != 0:
        raise ValueError("action_chunk must be an integer multiple of chunk")
    return source_fps / (action_chunk // chunk)


def build_deepspeed_batch_config(
    deepspeed_config: Dict[str, Any],
    *,
    per_device_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int,
) -> Dict[str, Any]:
    config = dict(deepspeed_config)
    config["train_batch_size"] = (
        per_device_batch_size * world_size * gradient_accumulation_steps
    )
    # Accelerate defaults this field to 1 when an explicit DeepSpeed config is
    # supplied, even if DeepSpeedPlugin receives a different constructor value.
    config["gradient_accumulation_steps"] = gradient_accumulation_steps
    return config


def compute_ltx_latent_frames(
    raw_future_frames: int,
    temporal_compression_ratio: int,
    n_previous: int,
) -> int:
    """LTX temporal VAE keeps the first frame and then compresses intervals."""

    if raw_future_frames < 1:
        raise ValueError("at least one future frame is required")
    return (raw_future_frames - 1) // temporal_compression_ratio + 1 + n_previous


def select_training_future_video(
    video: torch.Tensor,
    *,
    n_previous: int,
    chunk: int,
    return_action: bool,
    return_video: bool,
) -> torch.Tensor:
    """Keep real future RGB for joint training; synthesize length only for action-only."""

    future_video = video[:, :, n_previous:]
    if return_action and not return_video:
        repeats = [1] * future_video.ndim
        repeats[2] = chunk
        future_video = future_video[:, :, :1].repeat(*repeats)
    return future_video


def sample_semantic_condition_mask(
    batch_size: int,
    n_view: int,
    dropout_probability: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one semantic keep/drop decision per scene and share it across views."""

    if not 0 <= dropout_probability <= 1:
        raise ValueError("semantic dropout probability must be in [0, 1]")
    keep = torch.rand(batch_size, device=device, generator=generator) >= dropout_probability
    return keep.to(dtype=dtype).repeat_interleave(n_view)


def build_vlm_semantic_condition(
    provider,
    video: torch.Tensor,
    instructions,
    *,
    n_previous: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict from the last current observation without exposing future RGB."""

    if video.ndim != 6 or video.shape[1] != 3 or video.shape[2] != 2:
        raise ValueError(f"video must be [B,3,2,T,H,W], got {tuple(video.shape)}")
    current_index = int(n_previous) - 1
    if current_index < 0 or current_index >= video.shape[3]:
        raise ValueError(
            f"n_previous={n_previous} selects invalid current index {current_index}"
        )
    current_images = video[:, :, :, current_index].permute(0, 2, 1, 3, 4)
    plan = provider.predict(current_images.contiguous(), instructions)
    expected_tokens = (video.shape[0], 2, 1, 256, 1024)
    expected_times = (video.shape[0] * 2, 1)
    if tuple(plan.semantic_tokens.shape) != expected_tokens:
        raise RuntimeError(
            "VLM semantic tokens must have shape "
            f"{expected_tokens}, got {tuple(plan.semantic_tokens.shape)}"
        )
    if tuple(plan.times.shape) != expected_times:
        raise RuntimeError(
            f"VLM semantic times must have shape {expected_times}, "
            f"got {tuple(plan.times.shape)}"
        )
    return plan.semantic_tokens, plan.times


def freeze_conditioning_modules(*components) -> None:
    """Explicitly freeze and eval all condition-only neural networks."""

    for component in components:
        if component is None:
            continue
        module = getattr(component, "model", component)
        if isinstance(module, torch.nn.Module):
            module.requires_grad_(False)
            module.eval()


def _is_semantic_parameter(name: str) -> bool:
    return name.startswith("semantic_") or ".semantic_" in name


def build_optimizer_parameter_groups(
    model: torch.nn.Module,
    train_mode: str,
    base_lr: float,
    semantic_lr: float,
    *,
    action_lr: float | None = None,
    provider: torch.nn.Module | None = None,
    qwen_top_lr: float | None = None,
    qwen_vision_lr: float | None = None,
    qwen_ownership: Any | None = None,
    baton_source: str | None = None,
) -> List[Dict[str, Any]]:
    """Apply GE-Act train-mode filtering and split semantic parameters by LR."""

    if baton_source is not None:
        if baton_source not in BATON_SOURCES:
            raise ValueError(f"unknown Baton optimizer source: {baton_source}")
        if provider is not None or qwen_ownership is not None:
            raise ValueError("frozen Baton conditioning cannot own optimizer state")
        if train_mode not in ("all", "action_full"):
            raise ValueError("Baton curricula require video and action training")
        if action_lr is None:
            raise ValueError("Baton optimizer requires action_lr")

        by_identifier: dict[int, tuple[str, torch.nn.Parameter]] = {}
        named_parameters = model.named_parameters
        try:
            aliases = named_parameters(remove_duplicate=False)
        except TypeError:
            aliases = named_parameters()
        for name, parameter in aliases:
            if "action_" in name:
                owner = "action_expert"
            elif _is_semantic_parameter(name):
                owner = "semantic_adapter"
            else:
                owner = "ltx_video"
            previous = by_identifier.get(id(parameter))
            if previous is not None and previous[0] != owner:
                raise RuntimeError(
                    "tied GE-Act parameter aliases cross optimizer owners: "
                    f"{previous[0]} and {owner}"
                )
            by_identifier[id(parameter)] = (owner, parameter)
            parameter.requires_grad_(True)

        grouped_parameters = {
            "ltx_video": [],
            "action_expert": [],
            "semantic_adapter": [],
        }
        for owner, parameter in by_identifier.values():
            grouped_parameters[owner].append(parameter)
        empty = [
            name
            for name, parameters in grouped_parameters.items()
            if not parameters
        ]
        if empty:
            raise ValueError(
                "Baton optimizer groups must all be nonempty: "
                + ", ".join(empty)
            )
        grouped_ids = [
            id(parameter)
            for parameters in grouped_parameters.values()
            for parameter in parameters
        ]
        trainable_ids = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }
        if len(grouped_ids) != len(set(grouped_ids)):
            raise RuntimeError("Baton optimizer parameter groups contain duplicates")
        if set(grouped_ids) != trainable_ids:
            raise RuntimeError(
                "Baton optimizer groups do not exhaust trainable GE-Act parameters"
            )
        learning_rates = {
            "ltx_video": float(base_lr),
            "action_expert": float(action_lr),
            "semantic_adapter": float(semantic_lr),
        }
        return [
            {
                "name": name,
                "params": grouped_parameters[name],
                "lr": learning_rates[name],
            }
            for name in (
                "ltx_video",
                "action_expert",
                "semantic_adapter",
            )
        ]

    if provider is None:
        base_parameters = []
        semantic_parameters = []
        for name, parameter in model.named_parameters():
            is_action = "action_" in name
            if train_mode == "action_only":
                trainable = is_action
            elif train_mode == "video_only":
                trainable = not is_action
            elif train_mode in ("all", "action_full"):
                trainable = True
            else:
                raise NotImplementedError(f"unknown train mode: {train_mode}")
            parameter.requires_grad_(trainable)
            if not trainable:
                continue
            if _is_semantic_parameter(name):
                semantic_parameters.append(parameter)
            else:
                base_parameters.append(parameter)

        groups = []
        if base_parameters:
            groups.append({"name": "base_ltx", "params": base_parameters, "lr": base_lr})
        if semantic_parameters:
            groups.append({"name": "semantic", "params": semantic_parameters, "lr": semantic_lr})
        return groups

    if qwen_ownership is None:
        raise ValueError("grounded provider requires explicit Qwen module ownership")
    if action_lr is None or qwen_top_lr is None or qwen_vision_lr is None:
        raise ValueError("all five grounded optimizer learning rates are required")

    base_parameters = []
    action_parameters = []
    semantic_parameters = []
    for name, parameter in model.named_parameters():
        is_action = "action_" in name
        if train_mode == "action_only":
            trainable = is_action
        elif train_mode == "video_only":
            trainable = not is_action
        elif train_mode in ("all", "action_full"):
            trainable = True
        else:
            raise NotImplementedError(f"unknown train mode: {train_mode}")
        parameter.requires_grad_(trainable)
        if not trainable:
            continue
        if is_action:
            action_parameters.append(parameter)
        elif _is_semantic_parameter(name):
            semantic_parameters.append(parameter)
        else:
            base_parameters.append(parameter)

    def owned_parameter_ids(modules: Any, *, label: str) -> set[int]:
        if not isinstance(modules, tuple) or not modules:
            raise ValueError(f"{label} ownership must contain explicit modules")
        identifiers: set[int] = set()
        for module in modules:
            if not isinstance(module, torch.nn.Module):
                raise TypeError(f"{label} ownership must contain torch modules")
            identifiers.update(id(parameter) for parameter in module.parameters())
        return identifiers

    top_ids = owned_parameter_ids(
        getattr(qwen_ownership, "top_language_layers", None),
        label="qwen_top8",
    )
    vision_ids = owned_parameter_ids(
        getattr(qwen_ownership, "vision_modules", None),
        label="qwen_vision",
    )
    semantic_ids = owned_parameter_ids(
        getattr(qwen_ownership, "semantic_modules", None),
        label="semantic_adapter",
    )
    ownership_sets = (top_ids, vision_ids, semantic_ids)
    if any(
        left.intersection(right)
        for index, left in enumerate(ownership_sets)
        for right in ownership_sets[index + 1 :]
    ):
        raise RuntimeError("explicit Qwen module ownership overlaps")

    qwen_top_parameters = []
    qwen_vision_parameters = []
    provider_semantic_parameters = []
    unowned_provider_parameters = []
    for name, parameter in provider.named_parameters():
        if not parameter.requires_grad:
            continue
        identifier = id(parameter)
        if identifier in top_ids:
            qwen_top_parameters.append(parameter)
        elif identifier in vision_ids:
            qwen_vision_parameters.append(parameter)
        elif identifier in semantic_ids:
            provider_semantic_parameters.append(parameter)
        else:
            unowned_provider_parameters.append(name)
    if unowned_provider_parameters:
        raise ValueError(
            "unowned trainable grounded provider parameters: "
            + ", ".join(unowned_provider_parameters)
        )
    semantic_parameters.extend(provider_semantic_parameters)

    grouped = (
        ("ltx_video", base_parameters, float(base_lr)),
        ("action_expert", action_parameters, float(action_lr)),
        ("semantic_adapter", semantic_parameters, float(semantic_lr)),
        ("qwen_top8", qwen_top_parameters, float(qwen_top_lr)),
        ("qwen_vision", qwen_vision_parameters, float(qwen_vision_lr)),
    )
    empty = [name for name, parameters, _ in grouped if not parameters]
    if empty:
        raise ValueError(
            "grounded optimizer groups must all be nonempty: "
            + ", ".join(empty)
        )
    parameter_ids = [
        id(parameter)
        for _, parameters, _ in grouped
        for parameter in parameters
    ]
    trainable_ids = {
        id(parameter)
        for module in (model, provider)
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    if len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError("grounded optimizer parameter groups contain duplicates")
    if set(parameter_ids) != trainable_ids:
        raise RuntimeError(
            "grounded optimizer groups do not exhaust trainable parameters"
        )
    return [
        {"name": name, "params": parameters, "lr": learning_rate}
        for name, parameters, learning_rate in grouped
    ]


def compute_joint_loss(
    *,
    loss_video,
    loss_action,
    planner_loss,
    action_loss_scale: float,
    planner_aux_weight: float,
):
    """Combine video, action, and unscaled planner auxiliary objectives."""

    return (
        loss_video
        + float(action_loss_scale) * loss_action
        + float(planner_aux_weight) * planner_loss
    )


def build_grounded_semantic_condition(
    provider: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    training: bool,
    qwen_gradient_scale: float,
):
    """Teacher-force cached targets for training, generate otherwise, then compress."""

    from qwen35_planx.compression import compress_grounded_plan
    from qwen35_planx.planner_dataset import CachedPlannerTargets
    from qwen35_planx.provider import scale_gradient

    current_images = batch.get("current_images")
    captions = batch.get("caption")
    if not isinstance(current_images, torch.Tensor):
        raise TypeError("grounded batch current_images must be a tensor")
    if not isinstance(captions, (list, tuple)):
        raise TypeError("grounded batch caption must be a sequence")
    instructions = [str(caption) for caption in captions]
    if training:
        required = {
            "target_codes",
            "target_relevance",
            "target_relevance_confidence",
            "target_flow",
            "target_phrase_embeddings",
        }
        missing = sorted(required.difference(batch))
        if missing:
            raise ValueError(
                "grounded training batch is missing cached targets: "
                + ", ".join(missing)
            )
        targets = CachedPlannerTargets(
            codes=batch["target_codes"].long(),
            relevance=batch["target_relevance"],
            relevance_confidence=batch["target_relevance_confidence"],
            flow=batch["target_flow"],
            phrase_embeddings=batch["target_phrase_embeddings"],
        )
        planner_output = provider.teacher_force(
            current_images,
            instructions,
            targets=targets,
        )
        boundary_scale = float(qwen_gradient_scale)
        planner_loss = planner_output.loss
    else:
        planner_output = provider.generate(current_images, instructions)
        boundary_scale = 1.0
        planner_loss = None
    fused = provider.fuse(
        planner_output,
        qwen_gradient_scale=boundary_scale,
    )
    relevance = scale_gradient(
        planner_output.relevance,
        boundary_scale,
    )
    return compress_grounded_plan(fused, relevance), planner_loss


def generate_grounded_condition_for_validation(
    provider: torch.nn.Module,
    batch: Mapping[str, Any],
    *,
    qwen_gradient_scale: float,
):
    """Generate without gradients and restore the provider's prior mode."""

    was_training = provider.training
    provider.eval()
    try:
        with torch.no_grad():
            condition, _ = build_grounded_semantic_condition(
                provider,
                batch,
                training=False,
                qwen_gradient_scale=qwen_gradient_scale,
            )
            return condition
    finally:
        provider.train(was_training)


@dataclass(frozen=True)
class JointGroundedForwardOutput:
    latents: Mapping[str, torch.Tensor]
    planner_loss: torch.Tensor
    semantic_condition: Any


class JointGroundedTrainingModel(torch.nn.Module):
    """One registered module for DeepSpeed containing both trainable models."""

    def __init__(
        self,
        *,
        diffusion_model: torch.nn.Module,
        semantic_provider: torch.nn.Module,
    ) -> None:
        super().__init__()
        if not isinstance(diffusion_model, torch.nn.Module):
            raise TypeError("diffusion_model must be a torch module")
        if not isinstance(semantic_provider, torch.nn.Module):
            raise TypeError("semantic_provider must be a torch module")
        self.diffusion_model = diffusion_model
        self.semantic_provider = semantic_provider

    def forward(
        self,
        *,
        grounded_batch: Mapping[str, Any],
        qwen_gradient_scale: float,
        **diffusion_kwargs,
    ) -> JointGroundedForwardOutput:
        semantic_condition, planner_loss = build_grounded_semantic_condition(
            self.semantic_provider,
            grounded_batch,
            training=True,
            qwen_gradient_scale=qwen_gradient_scale,
        )
        reference = diffusion_kwargs.get("noisy_latents")
        if not isinstance(reference, torch.Tensor):
            raise TypeError("joint forward requires noisy_latents")
        from qwen35_planx.compression import CompressedSemanticPlan

        aligned_condition = CompressedSemanticPlan(
            tokens=semantic_condition.tokens.to(
                device=reference.device,
                dtype=reference.dtype,
            ),
            positions=semantic_condition.positions.to(
                device=reference.device,
                dtype=torch.float32,
            ),
            mask=semantic_condition.mask.to(device=reference.device),
            relevance=semantic_condition.relevance.to(
                device=reference.device,
                dtype=reference.dtype,
            ),
            source_indices=semantic_condition.source_indices.to(
                device=reference.device,
            ),
        )
        for field in (
            "semantic_plan",
            "semantic_plan_positions",
            "semantic_plan_mask",
            "semantic_plan_relevance",
        ):
            if field in diffusion_kwargs:
                raise ValueError(
                    f"joint forward constructs {field}; callers must not supply it"
                )
        latents = forward_pass(
            model=self.diffusion_model,
            semantic_plan=aligned_condition.tokens,
            semantic_plan_positions=aligned_condition.positions,
            semantic_plan_mask=aligned_condition.mask,
            semantic_plan_relevance=aligned_condition.relevance,
            **diffusion_kwargs,
        )["latents"]
        return JointGroundedForwardOutput(
            latents=latents,
            planner_loss=planner_loss,
            semantic_condition=aligned_condition,
        )


class EpochSeededRandomSampler(torch.utils.data.Sampler[int]):
    """Reconstruct each epoch permutation from an immutable seed and epoch."""

    def __init__(self, data_source, *, seed: int) -> None:
        if type(seed) is not int:
            raise TypeError("sampler seed must be an integer")
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("sampler epoch must be a non-negative integer")
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(
            torch.randperm(
                len(self.data_source),
                generator=generator,
            ).tolist()
        )

    def __len__(self) -> int:
        return len(self.data_source)


@dataclass(frozen=True)
class TrainingCursor:
    """Exact next-microbatch position for deterministic joint resume."""

    global_step: int
    epoch: int
    consumed_microbatches: int
    microbatches_per_epoch: int
    sampler_seed: int

    def __post_init__(self) -> None:
        for field in (
            "global_step",
            "epoch",
            "consumed_microbatches",
            "microbatches_per_epoch",
            "sampler_seed",
        ):
            if type(getattr(self, field)) is not int:
                raise TypeError(f"training cursor {field} must be an integer")
        if self.global_step < 0 or self.epoch < 0:
            raise ValueError("training cursor step and epoch must be non-negative")
        if self.microbatches_per_epoch <= 0:
            raise ValueError(
                "training cursor microbatches_per_epoch must be positive"
            )
        if not 0 <= self.consumed_microbatches < self.microbatches_per_epoch:
            raise ValueError(
                "training cursor consumed_microbatches must identify the next "
                "batch within the epoch"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "consumed_microbatches": self.consumed_microbatches,
            "microbatches_per_epoch": self.microbatches_per_epoch,
            "sampler_seed": self.sampler_seed,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "TrainingCursor":
        expected = {
            "global_step",
            "epoch",
            "consumed_microbatches",
            "microbatches_per_epoch",
            "sampler_seed",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("joint checkpoint training cursor is invalid")
        return cls(**payload)


def advance_training_cursor(
    cursor: TrainingCursor,
    *,
    epoch: int,
    consumed_microbatches: int,
    global_step: int,
) -> TrainingCursor:
    """Normalize an exhausted epoch to the next epoch's first microbatch."""

    if consumed_microbatches == cursor.microbatches_per_epoch:
        epoch += 1
        consumed_microbatches = 0
    return TrainingCursor(
        global_step=global_step,
        epoch=epoch,
        consumed_microbatches=consumed_microbatches,
        microbatches_per_epoch=cursor.microbatches_per_epoch,
        sampler_seed=cursor.sampler_seed,
    )


def set_dataloader_epoch(
    dataloader,
    *,
    epoch: int,
    sampler_seed: int,
) -> None:
    """Set both permutation epoch and private worker-base-seed generator."""

    set_epoch = getattr(dataloader, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(epoch)
    else:
        sampler = getattr(dataloader, "sampler", None)
        sampler_set_epoch = getattr(sampler, "set_epoch", None)
        if callable(sampler_set_epoch):
            sampler_set_epoch(epoch)
    generator = getattr(dataloader, "generator", None)
    if isinstance(generator, torch.Generator):
        generator.manual_seed(sampler_seed + epoch)
    dataset = getattr(dataloader, "dataset", None)
    if dataset is not None:
        _set_wrapped_dataset_epoch(dataset, epoch=epoch)


def _dataset_children(dataset: Any) -> tuple[Any, ...]:
    children: list[Any] = []
    child = getattr(dataset, "dataset", None)
    if child is not None and child is not dataset:
        children.append(child)
    base = getattr(dataset, "base", None)
    if (
        base is not None
        and base is not dataset
        and all(base is not existing for existing in children)
    ):
        children.append(base)
    nested = getattr(dataset, "datasets", None)
    if isinstance(nested, (list, tuple)):
        children.extend(
            item
            for item in nested
            if item is not dataset
            and all(item is not existing for existing in children)
        )
    return tuple(children)


def _set_wrapped_dataset_epoch(
    dataset: Any,
    *,
    epoch: int,
    _seen: set[int] | None = None,
) -> None:
    seen = set() if _seen is None else _seen
    identifier = id(dataset)
    if identifier in seen:
        return
    seen.add(identifier)
    set_epoch = getattr(dataset, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(epoch)
    for child in _dataset_children(dataset):
        _set_wrapped_dataset_epoch(child, epoch=epoch, _seen=seen)


def _baton_dataset_sampling_contracts(
    dataset: Any,
    *,
    _seen: set[int] | None = None,
) -> list[dict[str, Any]]:
    seen = set() if _seen is None else _seen
    identifier = id(dataset)
    if identifier in seen:
        return []
    seen.add(identifier)
    children = _dataset_children(dataset)
    if children:
        contracts: list[dict[str, Any]] = []
        for child in children:
            contracts.extend(
                _baton_dataset_sampling_contracts(child, _seen=seen)
            )
        return contracts
    contract = getattr(dataset, "baton_sampling_contract", None)
    return [contract] if isinstance(contract, dict) else []


def validate_baton_dataset_sampling_contract(
    dataset: Any,
    *,
    expected_seed: int,
) -> dict[str, Any]:
    """Require every wrapped Baton dataset leaf to share one stateless contract."""

    expected = {
        "algorithm": BATON_SAMPLING_ALGORITHM,
        "version": BATON_SAMPLING_VERSION,
        "seed": expected_seed,
    }
    contracts = _baton_dataset_sampling_contracts(dataset)
    if not contracts or any(contract != expected for contract in contracts):
        raise ValueError(
            "Baton training requires the exact stateless window sampling "
            f"contract {expected!r}"
        )
    return expected


def prepare_joint_training_components(
    accelerator,
    joint_model: JointGroundedTrainingModel,
    optimizer,
    dataloader,
    scheduler,
):
    """Prepare one DeepSpeed-compatible model plus all mutable training state."""

    return accelerator.prepare(
        joint_model,
        optimizer,
        dataloader,
        scheduler,
    )


def _joint_model_topology(model: JointGroundedTrainingModel) -> list[dict[str, Any]]:
    if not isinstance(model, JointGroundedTrainingModel):
        raise TypeError("joint_model must be a JointGroundedTrainingModel")
    state = model.state_dict()
    names = tuple(state)
    if not any(name.startswith("diffusion_model.") for name in names):
        raise ValueError("joint model has no diffusion state")
    if not any(name.startswith("semantic_provider.") for name in names):
        raise ValueError("joint model has no semantic provider state")
    return [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in state.items()
    ]


def _joint_topology_hash(model: JointGroundedTrainingModel) -> str:
    from qwen35_planx.hashing import sha256_json

    return sha256_json(_joint_model_topology(model))


def save_joint_training_checkpoint(
    accelerator,
    output_dir: str | Path,
    *,
    cursor: TrainingCursor,
    joint_model: JointGroundedTrainingModel,
) -> Path:
    """Atomically publish Accelerate state containing diffusion and provider."""

    if not isinstance(cursor, TrainingCursor):
        raise TypeError("joint checkpoint requires a TrainingCursor")
    root = Path(output_dir)
    destination = root / f"step_{cursor.global_step:06d}"
    staging = root / f".{destination.name}.incomplete"
    if getattr(accelerator, "is_main_process", True):
        root.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    if destination.exists():
        raise FileExistsError(f"joint checkpoint already exists: {destination}")
    if staging.exists():
        raise FileExistsError(
            f"incomplete joint checkpoint already exists: {staging}"
        )
    if getattr(accelerator, "is_main_process", True):
        staging.mkdir()
    try:
        accelerator.wait_for_everyone()
        accelerator.save_state(str(staging))
        accelerator.wait_for_everyone()
        if getattr(accelerator, "is_main_process", True):
            metadata = {
                "format_version": 2,
                "model_children": [
                    "diffusion_model",
                    "semantic_provider",
                ],
                "topology_hash": _joint_topology_hash(joint_model),
                "cursor": cursor.to_dict(),
            }
            (staging / "joint_state.json").write_text(
                json.dumps(
                    metadata,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(staging, destination)
        accelerator.wait_for_everyone()
    except Exception:
        if getattr(accelerator, "is_main_process", True):
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def load_joint_training_checkpoint(
    accelerator,
    checkpoint_dir: str | Path,
    *,
    joint_model: JointGroundedTrainingModel,
    expected_microbatches_per_epoch: int,
    expected_sampler_seed: int,
) -> TrainingCursor:
    """Validate both child topologies before restoring Accelerate state."""

    checkpoint = Path(checkpoint_dir)
    metadata_path = checkpoint / "joint_state.json"
    if not metadata_path.is_file():
        raise ValueError(
            f"joint checkpoint is incomplete: missing {metadata_path}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("joint checkpoint metadata is invalid") from error
    expected_fields = {
        "format_version",
        "model_children",
        "topology_hash",
        "cursor",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_fields:
        raise ValueError("joint checkpoint metadata fields are invalid")
    if (
        metadata["format_version"] != 2
        or metadata["model_children"]
        != ["diffusion_model", "semantic_provider"]
        or metadata["topology_hash"] != _joint_topology_hash(joint_model)
    ):
        raise ValueError("joint checkpoint is incompatible with runtime models")
    try:
        cursor = TrainingCursor.from_dict(metadata["cursor"])
    except (TypeError, ValueError) as error:
        raise ValueError("joint checkpoint training cursor is invalid") from error
    if (
        cursor.microbatches_per_epoch != expected_microbatches_per_epoch
        or cursor.sampler_seed != expected_sampler_seed
    ):
        raise ValueError("joint checkpoint data cursor differs from runtime")
    accelerator.load_state(str(checkpoint))
    accelerator.wait_for_everyone()
    return cursor


def _module_topology_hash(model: torch.nn.Module) -> str:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("diffusion_model must be a torch module")
    return _state_dict_topology_hash(model.state_dict())


def _state_dict_topology_hash(
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    return sha256_json(
        [
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in sorted(state_dict.items())
        ]
    )


_SAFETENSORS_TORCH_DTYPES = {
    "BOOL": "torch.bool",
    "U8": "torch.uint8",
    "I8": "torch.int8",
    "I16": "torch.int16",
    "I32": "torch.int32",
    "I64": "torch.int64",
    "F16": "torch.float16",
    "BF16": "torch.bfloat16",
    "F32": "torch.float32",
    "F64": "torch.float64",
}


def _diffusion_snapshot_topology_hash(diffusion_dir: Path) -> str:
    from safetensors import safe_open
    from utils.model_utils import resolve_checkpoint_files

    topology: list[dict[str, Any]] = []
    for checkpoint_file in resolve_checkpoint_files(diffusion_dir):
        with safe_open(
            str(checkpoint_file),
            framework="pt",
            device="cpu",
        ) as handle:
            for name in handle.keys():
                tensor_slice = handle.get_slice(name)
                raw_dtype = tensor_slice.get_dtype()
                dtype = _SAFETENSORS_TORCH_DTYPES.get(raw_dtype)
                if dtype is None:
                    raise ValueError(
                        f"unsupported diffusion snapshot dtype: {raw_dtype}"
                    )
                topology.append(
                    {
                        "name": name,
                        "shape": list(tensor_slice.get_shape()),
                        "dtype": dtype,
                    }
                )
    topology.sort(key=lambda item: item["name"])
    if not topology:
        raise ValueError("diffusion snapshot topology is empty")
    return sha256_json(topology)


def _diffusion_snapshot_files(diffusion_dir: Path) -> dict[str, str]:
    files = {
        path.relative_to(diffusion_dir).as_posix(): sha256_file(path)
        for path in sorted(diffusion_dir.rglob("*"))
        if path.is_file()
    }
    if not files:
        raise ValueError("diffusion snapshot is empty")
    return files


def _validated_diffusion_file_manifest(
    manifest: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(manifest, Mapping) or not manifest:
        raise ValueError("Stage-3 diffusion file manifest is invalid")
    validated: dict[str, str] = {}
    for name, digest in manifest.items():
        relative = Path(name) if type(name) is str else None
        if (
            relative is None
            or not name
            or relative.is_absolute()
            or name != relative.as_posix()
            or any(part in ("", ".", "..") for part in relative.parts)
            or not _is_concrete_sha256(digest)
        ):
            raise ValueError("Stage-3 diffusion file manifest is invalid")
        validated[name] = digest
    if not any(name.endswith(".safetensors") for name in validated):
        raise ValueError("Stage-3 diffusion manifest has no tensor snapshot")
    return validated


def _regular_snapshot_file_set(snapshot_path: Path) -> set[str]:
    if snapshot_path.is_symlink() or not snapshot_path.is_dir():
        raise ValueError("Stage-3 snapshot root must be a regular directory")
    names: set[str] = set()
    for directory, child_directories, filenames in os.walk(
        snapshot_path,
        topdown=True,
        followlinks=False,
    ):
        directory_path = Path(directory)
        for child in child_directories:
            child_path = directory_path / child
            if child_path.is_symlink() or not child_path.is_dir():
                raise ValueError(
                    "Stage-3 snapshot contains a non-regular directory"
                )
        for filename in filenames:
            file_path = directory_path / filename
            if file_path.is_symlink() or not file_path.is_file():
                raise ValueError(
                    "Stage-3 snapshot contains a non-regular file"
                )
            names.add(file_path.relative_to(snapshot_path).as_posix())
    return names


def _read_regular_snapshot_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Stage-3 snapshot contains a non-regular file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def strict_load_baton_stage3_diffusion_model(
    diffusion_model: torch.nn.Module,
    diffusion_dir: str | Path,
    *,
    expected_snapshot_topology_hash: str,
    expected_diffusion_files: Mapping[str, str],
) -> torch.nn.Module:
    """Hash and deserialize identical snapshot bytes before one strict mutation."""

    if not isinstance(diffusion_model, torch.nn.Module):
        raise TypeError("Stage-3 diffusion model must be a torch module")
    if not _is_concrete_sha256(expected_snapshot_topology_hash):
        raise ValueError("Stage-3 expected snapshot topology hash is invalid")
    expected_files = _validated_diffusion_file_manifest(
        expected_diffusion_files
    )
    snapshot_path = Path(diffusion_dir)
    if _module_topology_hash(diffusion_model) != expected_snapshot_topology_hash:
        raise ValueError(
            "Stage-3 runtime model topology differs from the Stage-2 snapshot"
        )
    if _regular_snapshot_file_set(snapshot_path) != set(expected_files):
        raise ValueError("Stage-3 snapshot file set differs from its manifest")
    from safetensors import SafetensorError
    from safetensors.torch import load as load_safetensors

    state_dict: dict[str, torch.Tensor] = {}
    try:
        for relative_name, expected_digest in sorted(expected_files.items()):
            payload = _read_regular_snapshot_file(
                snapshot_path / relative_name
            )
            if hashlib.sha256(payload).hexdigest() != expected_digest:
                raise ValueError(
                    "Stage-3 snapshot artifact hash differs from its manifest"
                )
            if not relative_name.endswith(".safetensors"):
                continue
            shard = load_safetensors(payload)
            duplicate_keys = set(state_dict).intersection(shard)
            if duplicate_keys:
                raise ValueError(
                    "Stage-3 snapshot contains duplicate tensor keys"
                )
            state_dict.update(
                {
                    name: value.clone()
                    for name, value in shard.items()
                }
            )
            del shard
            del payload
    except (OSError, SafetensorError) as error:
        raise ValueError("Stage-3 snapshot artifact is invalid") from error
    if _regular_snapshot_file_set(snapshot_path) != set(expected_files):
        raise ValueError("Stage-3 snapshot file set changed during loading")
    if _state_dict_topology_hash(state_dict) != expected_snapshot_topology_hash:
        raise ValueError("Stage-3 snapshot topology differs from its envelope")
    try:
        diffusion_model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError("Stage-3 strict diffusion state loading failed") from error
    return diffusion_model


_BATON_STATE_FIELDS = {
    "format_version",
    "checkpoint_kind",
    "model_children",
    "source",
    "topology_hash",
    "snapshot_topology_hash",
    "cursor",
    "accelerator_files",
    "diffusion_subdir",
    "diffusion_files",
    "training_provenance",
}
_BATON_SAMPLING_PROVENANCE_FIELDS = {
    "window_sampling_algorithm",
    "window_sampling_version",
    "window_sampling_seed",
    "window_sampling_topology_hash",
}
_BATON_TEACHER_PROVENANCE_FIELDS = {
    "hdf5_manifest_hash",
    "siglip2_config_hash",
    "siglip2_artifact_hash",
    "teacher_preprocessing_hash",
} | _BATON_SAMPLING_PROVENANCE_FIELDS
_BATON_PREDICTION_PROVENANCE_FIELDS = {
    "hdf5_manifest_hash",
    "planner_manifest_hash",
    "planner_topology_hash",
    "qwen_config_hash",
    "tokenizer_hash",
    "processor_hash",
    "input_template_hash",
    "siglip2_config_hash",
    "siglip2_artifact_hash",
    "teacher_preprocessing_hash",
} | _BATON_SAMPLING_PROVENANCE_FIELDS


@dataclass(frozen=True)
class _LoadedBatonTrainingCheckpoint:
    """One fully verified, immutable view of a Baton checkpoint envelope."""

    checkpoint: Path
    source: str
    topology_hash: str
    snapshot_topology_hash: str
    cursor: TrainingCursor
    diffusion_model_dir: Path
    diffusion_files: Mapping[str, str]
    training_provenance: Mapping[str, Any]


def _is_concrete_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and value != "0" * 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_baton_training_provenance(
    source: str,
    provenance: Any,
) -> dict[str, Any]:
    expected_fields = (
        _BATON_TEACHER_PROVENANCE_FIELDS
        if source == BATON_TEACHER_SOURCE
        else _BATON_PREDICTION_PROVENANCE_FIELDS
    )
    if type(provenance) is not dict or set(provenance) != expected_fields:
        raise ValueError("Baton checkpoint training provenance is invalid")
    hash_fields = expected_fields - {
        "window_sampling_algorithm",
        "window_sampling_version",
        "window_sampling_seed",
    }
    if not all(_is_concrete_sha256(provenance[field]) for field in hash_fields):
        raise ValueError("Baton checkpoint training provenance is invalid")
    sampling_contract = {
        "algorithm": provenance["window_sampling_algorithm"],
        "version": provenance["window_sampling_version"],
        "seed": provenance["window_sampling_seed"],
    }
    if (
        sampling_contract["algorithm"] != BATON_SAMPLING_ALGORITHM
        or type(sampling_contract["version"]) is not int
        or sampling_contract["version"] != BATON_SAMPLING_VERSION
        or type(sampling_contract["seed"]) is not int
        or sampling_contract["seed"] < 0
        or provenance["window_sampling_topology_hash"]
        != sha256_json(sampling_contract)
    ):
        raise ValueError("Baton checkpoint training provenance is invalid")
    return dict(provenance)


def _accelerator_state_files(checkpoint: Path) -> dict[str, str]:
    files = {
        path.relative_to(checkpoint).as_posix(): sha256_file(path)
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file()
        and path.name != "baton_state.json"
        and "diffusion_model" not in path.relative_to(checkpoint).parts
    }
    if not files:
        raise ValueError("Baton checkpoint has no Accelerator state")
    return files


def _load_baton_training_metadata(
    checkpoint_dir: str | Path,
) -> _LoadedBatonTrainingCheckpoint:
    checkpoint = Path(checkpoint_dir)
    metadata_path = checkpoint / "baton_state.json"
    if not metadata_path.is_file():
        raise ValueError(
            f"Baton checkpoint is incomplete: missing {metadata_path}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Baton checkpoint metadata is invalid") from error
    if type(metadata) is not dict or set(metadata) != _BATON_STATE_FIELDS:
        raise ValueError("Baton checkpoint metadata fields are invalid")
    if (
        metadata["format_version"] != 1
        or metadata["checkpoint_kind"] != "ge_act_baton"
        or metadata["model_children"] != ["diffusion_model"]
        or metadata["source"] not in BATON_SOURCES
        or type(metadata["topology_hash"]) is not str
        or type(metadata["snapshot_topology_hash"]) is not str
        or metadata["diffusion_subdir"] != "diffusion_model"
        or type(metadata["training_provenance"]) is not dict
    ):
        raise ValueError("Baton checkpoint envelope is incompatible")
    provenance = _validated_baton_training_provenance(
        metadata["source"],
        metadata["training_provenance"],
    )
    try:
        cursor = TrainingCursor.from_dict(metadata["cursor"])
    except (TypeError, ValueError) as error:
        raise ValueError("Baton checkpoint training cursor is invalid") from error
    if provenance["window_sampling_seed"] != cursor.sampler_seed:
        raise ValueError(
            "Baton checkpoint sampling provenance differs from training cursor"
        )
    expected_accelerator_files = metadata["accelerator_files"]
    if (
        type(expected_accelerator_files) is not dict
        or not expected_accelerator_files
    ):
        raise ValueError("Baton checkpoint Accelerator manifest is invalid")
    try:
        actual_accelerator_files = _accelerator_state_files(checkpoint)
    except (OSError, ValueError) as error:
        raise ValueError("Baton checkpoint Accelerator state is incomplete") from error
    if actual_accelerator_files != expected_accelerator_files:
        raise ValueError("Baton checkpoint Accelerator state hash mismatch")
    diffusion_dir = checkpoint / metadata["diffusion_subdir"]
    if not diffusion_dir.is_dir():
        raise ValueError("Baton checkpoint diffusion snapshot is missing")
    expected_files = metadata["diffusion_files"]
    if (
        type(expected_files) is not dict
        or not expected_files
        or not all(
            type(name) is str
            and name
            and type(digest) is str
            and len(digest) == 64
            for name, digest in expected_files.items()
        )
    ):
        raise ValueError("Baton checkpoint diffusion artifact manifest is invalid")
    try:
        actual_files = _diffusion_snapshot_files(diffusion_dir)
    except (OSError, ValueError) as error:
        raise ValueError("Baton checkpoint diffusion artifact is invalid") from error
    if actual_files != expected_files:
        raise ValueError("Baton checkpoint diffusion artifact hash mismatch")
    try:
        snapshot_topology_hash = _diffusion_snapshot_topology_hash(
            diffusion_dir
        )
    except (OSError, ValueError) as error:
        raise ValueError("Baton checkpoint snapshot topology is invalid") from error
    if snapshot_topology_hash != metadata["snapshot_topology_hash"]:
        raise ValueError("Baton checkpoint snapshot topology mismatch")
    return _LoadedBatonTrainingCheckpoint(
        checkpoint=checkpoint,
        source=metadata["source"],
        topology_hash=metadata["topology_hash"],
        snapshot_topology_hash=metadata["snapshot_topology_hash"],
        cursor=cursor,
        diffusion_model_dir=diffusion_dir,
        diffusion_files=MappingProxyType(dict(expected_files)),
        training_provenance=MappingProxyType(dict(provenance)),
    )


def save_baton_training_checkpoint(
    accelerator,
    output_dir: str | Path,
    *,
    cursor: TrainingCursor,
    diffusion_model: torch.nn.Module,
    source: str,
    training_provenance: Mapping[str, Any],
) -> Path:
    """Atomically save mutable GE-Act state and a deployable diffusion snapshot."""

    if not isinstance(cursor, TrainingCursor):
        raise TypeError("Baton checkpoint requires a TrainingCursor")
    if source not in BATON_SOURCES:
        raise ValueError("Baton checkpoint source is invalid")
    validated_provenance = _validated_baton_training_provenance(
        source,
        training_provenance,
    )
    if validated_provenance["window_sampling_seed"] != cursor.sampler_seed:
        raise ValueError(
            "Baton checkpoint sampling provenance differs from training cursor"
        )
    root = Path(output_dir)
    destination = root / f"step_{cursor.global_step:06d}"
    staging = root / f".{destination.name}.incomplete"
    is_main = getattr(accelerator, "is_main_process", True)
    if is_main:
        root.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    if destination.exists():
        raise FileExistsError(f"Baton checkpoint already exists: {destination}")
    if staging.exists():
        raise FileExistsError(
            f"incomplete Baton checkpoint already exists: {staging}"
        )
    if is_main:
        staging.mkdir()
    try:
        accelerator.wait_for_everyone()
        accelerator.save_state(str(staging))
        accelerator.wait_for_everyone()
        if is_main:
            diffusion_dir = staging / "diffusion_model"
            diffusion_model.save_pretrained(
                diffusion_dir,
                safe_serialization=True,
            )
            metadata = {
                "format_version": 1,
                "checkpoint_kind": "ge_act_baton",
                "model_children": ["diffusion_model"],
                "source": source,
                "topology_hash": _module_topology_hash(diffusion_model),
                "snapshot_topology_hash": (
                    _diffusion_snapshot_topology_hash(diffusion_dir)
                ),
                "cursor": cursor.to_dict(),
                "accelerator_files": _accelerator_state_files(staging),
                "diffusion_subdir": "diffusion_model",
                "diffusion_files": _diffusion_snapshot_files(diffusion_dir),
                "training_provenance": validated_provenance,
            }
            (staging / "baton_state.json").write_text(
                json.dumps(
                    metadata,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(staging, destination)
        accelerator.wait_for_everyone()
    except Exception:
        if is_main:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def load_baton_training_checkpoint(
    accelerator,
    checkpoint_dir: str | Path,
    *,
    diffusion_model: torch.nn.Module,
    expected_source: str,
    expected_microbatches_per_epoch: int,
    expected_sampler_seed: int,
    expected_training_provenance: Mapping[str, Any],
) -> TrainingCursor:
    """Validate the complete Baton envelope before restoring Accelerate state."""

    loaded = _load_baton_training_metadata(checkpoint_dir)
    if loaded.source != expected_source:
        raise ValueError("Baton checkpoint conditioning source mismatch")
    if loaded.topology_hash != _module_topology_hash(diffusion_model):
        raise ValueError("Baton checkpoint runtime topology mismatch")
    expected_provenance = _validated_baton_training_provenance(
        expected_source,
        dict(expected_training_provenance),
    )
    if loaded.training_provenance != expected_provenance:
        raise ValueError("Baton checkpoint training provenance mismatch")
    if (
        loaded.cursor.microbatches_per_epoch
        != expected_microbatches_per_epoch
        or loaded.cursor.sampler_seed != expected_sampler_seed
    ):
        raise ValueError("Baton checkpoint data cursor mismatch")
    accelerator.load_state(str(loaded.checkpoint))
    accelerator.wait_for_everyone()
    return loaded.cursor


def _validate_loaded_baton_stage2_checkpoint(
    loaded: _LoadedBatonTrainingCheckpoint,
    *,
    expected_topology_hash: str,
    expected_hdf5_manifest_hash: str,
    expected_semantic_provenance: Mapping[str, str] | None = None,
) -> _LoadedBatonTrainingCheckpoint:
    """Purely validate every Stage-2 invariant on one sealed load result."""

    if not isinstance(loaded, _LoadedBatonTrainingCheckpoint):
        raise TypeError("Stage-2 validation requires one loaded Baton envelope")
    if loaded.source != BATON_TEACHER_SOURCE:
        raise ValueError(
            "Stage-3 initialization requires a Stage-2 teacher checkpoint"
        )
    if loaded.cursor.global_step != 20_000:
        raise ValueError("Stage-2 initialization checkpoint must be step 20000")
    if loaded.snapshot_topology_hash != expected_topology_hash:
        raise ValueError("Stage-2 initialization topology mismatch")
    provenance = loaded.training_provenance
    if provenance.get("hdf5_manifest_hash") != expected_hdf5_manifest_hash:
        raise ValueError("Stage-2 initialization dataset provenance mismatch")
    semantic_fields = (
        "siglip2_config_hash",
        "siglip2_artifact_hash",
        "teacher_preprocessing_hash",
    )
    if expected_semantic_provenance is not None:
        if set(expected_semantic_provenance) != set(semantic_fields):
            raise ValueError(
                "Stage-2 expected semantic provenance is incomplete"
            )
        for field in semantic_fields:
            if provenance.get(field) != expected_semantic_provenance[field]:
                raise ValueError(
                    f"Stage-1/Stage-2 semantic target mismatch for {field}"
                )
    else:
        for field in semantic_fields:
            if not _is_concrete_sha256(provenance.get(field)):
                raise ValueError(
                    f"Stage-2 initialization provenance is missing {field}"
                )
    if (
        loaded.diffusion_model_dir
        != loaded.checkpoint / "diffusion_model"
        or not loaded.diffusion_files
        or not all(
            type(name) is str
            and name
            and _is_concrete_sha256(digest)
            for name, digest in loaded.diffusion_files.items()
        )
    ):
        raise ValueError("Stage-2 diffusion file envelope is invalid")
    return loaded


def validate_baton_stage2_checkpoint_envelope(
    checkpoint_dir: str | Path,
    *,
    expected_topology_hash: str,
    expected_hdf5_manifest_hash: str,
) -> Path:
    """Validate a completed teacher curriculum before Stage-3 allocation."""

    loaded = _load_baton_training_metadata(checkpoint_dir)
    return _validate_loaded_baton_stage2_checkpoint(
        loaded,
        expected_topology_hash=expected_topology_hash,
        expected_hdf5_manifest_hash=expected_hdf5_manifest_hash,
    ).diffusion_model_dir


@dataclass(frozen=True)
class BatonStage3ArtifactChain:
    """Validated immutable Stage-1/Stage-2 boundary for Stage-3 allocation."""

    diffusion_model_dir: Path
    diffusion_files: Mapping[str, str]
    stage1_metadata: Any
    stage2_training_provenance: Mapping[str, Any]


def validate_baton_stage3_artifact_chain(
    *,
    stage2_checkpoint: str | Path,
    expected_stage2_topology_hash: str,
    planner_checkpoint: str | Path,
    expected_planner_topology: str | Path | Mapping[str, Any],
    expected_hdf5_manifest_hash: str,
) -> BatonStage3ArtifactChain:
    """Bind Stage-2 teacher targets exactly to trusted Stage-1 metadata."""

    from qwen35_baton.provider import (
        _validate_checkpoint_envelope,
        _validate_trusted_planner_topology,
    )

    planner_path = Path(planner_checkpoint).expanduser().resolve()
    stage1_metadata = _validate_checkpoint_envelope(planner_path)
    _validate_trusted_planner_topology(
        stage1_metadata,
        checkpoint=planner_path,
        expected=expected_planner_topology,
    )
    if stage1_metadata.hdf5_manifest_hash != expected_hdf5_manifest_hash:
        raise ValueError(
            "Stage-1 planner checkpoint dataset provenance mismatch"
        )
    loaded_stage2 = _load_baton_training_metadata(stage2_checkpoint)
    validated_stage2 = _validate_loaded_baton_stage2_checkpoint(
        loaded_stage2,
        expected_topology_hash=expected_stage2_topology_hash,
        expected_hdf5_manifest_hash=expected_hdf5_manifest_hash,
        expected_semantic_provenance={
            field: getattr(stage1_metadata, field)
            for field in (
                "siglip2_config_hash",
                "siglip2_artifact_hash",
                "teacher_preprocessing_hash",
            )
        },
    )
    return BatonStage3ArtifactChain(
        diffusion_model_dir=validated_stage2.diffusion_model_dir,
        diffusion_files=validated_stage2.diffusion_files,
        stage1_metadata=stage1_metadata,
        stage2_training_provenance=(
            validated_stage2.training_provenance
        ),
    )


def build_baton_training_provenance(
    config: Mapping[str, Any],
    *,
    stage1_metadata: Any | None = None,
) -> dict[str, Any]:
    """Bind a Baton checkpoint to its local dataset and frozen source."""

    config_mapping = _config_mapping(config)
    semantic, source = _validated_baton_semantic_config(config_mapping)
    manifest_path = Path(
        config_mapping.get("data", {})
        .get("train", {})
        .get("manifest_path", "")
    )
    if not manifest_path.is_file():
        raise ValueError(
            f"Baton training HDF5 manifest is missing: {manifest_path}"
        )
    provenance = {
        "hdf5_manifest_hash": sha256_file(manifest_path),
    }
    train_data = config_mapping.get("data", {}).get("train", {})
    sampling_contract = {
        "algorithm": train_data.get("baton_sampling_algorithm"),
        "version": train_data.get("baton_sampling_version"),
        "seed": train_data.get("baton_sampling_seed"),
    }
    if sampling_contract != {
        "algorithm": BATON_SAMPLING_ALGORITHM,
        "version": BATON_SAMPLING_VERSION,
        "seed": config_mapping.get("seed"),
    }:
        raise ValueError("Baton stateless window sampling config is invalid")
    provenance.update(
        {
            "window_sampling_algorithm": sampling_contract["algorithm"],
            "window_sampling_version": sampling_contract["version"],
            "window_sampling_seed": sampling_contract["seed"],
            "window_sampling_topology_hash": sha256_json(
                sampling_contract
            ),
        }
    )
    if source == BATON_TEACHER_SOURCE:
        provenance.update(
            {
                field: semantic[field]
                for field in (
                    "siglip2_config_hash",
                    "siglip2_artifact_hash",
                    "teacher_preprocessing_hash",
                )
            }
        )
        return provenance
    planner_checkpoint = Path(semantic["planner_checkpoint"])
    planner_manifest = planner_checkpoint / "manifest.json"
    if not planner_manifest.is_file():
        raise ValueError("Baton planner provenance files are missing")
    if stage1_metadata is None:
        from qwen35_baton.provider import (
            _validate_checkpoint_envelope,
            _validate_trusted_planner_topology,
        )

        stage1_metadata = _validate_checkpoint_envelope(
            planner_checkpoint.resolve()
        )
        _validate_trusted_planner_topology(
            stage1_metadata,
            checkpoint=planner_checkpoint.resolve(),
            expected=semantic["expected_planner_topology"],
        )
    if stage1_metadata.hdf5_manifest_hash != provenance[
        "hdf5_manifest_hash"
    ]:
        raise ValueError("Baton planner checkpoint differs from HDF5 manifest")
    provenance.update(
        {
            "planner_manifest_hash": sha256_file(planner_manifest),
            "planner_topology_hash": stage1_metadata.planner_topology_hash,
            "qwen_config_hash": stage1_metadata.qwen_config_hash,
            "tokenizer_hash": stage1_metadata.tokenizer_hash,
            "processor_hash": stage1_metadata.processor_hash,
            "input_template_hash": stage1_metadata.input_template_hash,
            "siglip2_config_hash": stage1_metadata.siglip2_config_hash,
            "siglip2_artifact_hash": stage1_metadata.siglip2_artifact_hash,
            "teacher_preprocessing_hash": (
                stage1_metadata.teacher_preprocessing_hash
            ),
        }
    )
    return provenance


def should_save_checkpoint(global_step: int, args: argparse.Namespace) -> bool:
    explicit_steps = getattr(args, "save_steps", None)
    if explicit_steps:
        return global_step in {int(step) for step in explicit_steps}
    return global_step > 0 and global_step % int(args.steps_to_save) == 0


class State:
    # Training state
    seed: int = None
    model_name: str = None
    accelerator: Accelerator = None
    weight_dtype: torch.dtype = None
    train_epochs: int = None
    train_steps: int = None
    overwrote_max_train_steps: bool = False
    num_trainable_parameters: int = 0
    learning_rate: float = None
    train_batch_size: int = None
    generator: torch.Generator = None

    # Hub state
    repo_id: str = None
    # Artifacts state
    output_dir: str = None



class Trainer:

    def __init__(self, config_file, to_log=True, output_dir=None) -> None:
        
        cd = load(open(config_file, "r"), Loader=Loader)
        self.raw_config = deepcopy(cd)
        args = argparse.Namespace(**cd)
        args.lr = float(args.lr)
        args.semantic_lr = float(getattr(args, "semantic_lr", args.lr))
        args.action_lr = float(getattr(args, "action_lr", args.lr))
        args.qwen_top_lr = float(getattr(args, "qwen_top_lr", 1e-6))
        args.qwen_vision_lr = float(getattr(args, "qwen_vision_lr", 5e-7))
        args.planner_aux_weight = float(
            getattr(args, "planner_aux_weight", 0.25)
        )
        args.qwen_ge_gradient_scale = float(
            getattr(args, "qwen_ge_gradient_scale", 0.1)
        )
        args.epsilon = float(args.epsilon)
        args.weight_decay = float(args.weight_decay)

        self.args = args

        if output_dir is not None:
            self.args.output_dir = output_dir

        if not self.args.load_weights:
            print('You are not loading the pretrained weights, please check the code.')
        self.state = State()

        self.tokenizer = None
        self.text_encoder = None
        self.diffusion_model = None
        self.unet = None
        self.vae = None
        self.scheduler = None
        self.semantic_encoder = None
        self.semantic_planner = None
        self.baton_components = None
        self.baton_training_provenance = None
        self.baton_stage3_artifact_chain = None
        self.grounded_provider = None
        self.grounded_training_enabled = False
        self.qwen_ownership = None
        self.joint_model = None
        self.sampler_seed = int(self.args.seed) if self.args.seed is not None else 0
        self.resume_cursor = None
        self.current_cursor = None
        self.video_frame_rate = compute_effective_video_fps(self.args.data["train"])

        self._init_distributed()
        self._init_logging()
        self._init_directories_and_repositories()

        self.state.model_name = self.args.model_name

        current_time = datetime.now()
        start_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        if self.state.accelerator.is_main_process:

            self.save_folder = os.path.join(self.args.output_dir, start_time)
            if getattr(self.args, "sub_folder", False):
                self.save_folder = os.path.join(self.args.output_dir, self.args.sub_folder)
            os.makedirs(self.save_folder, exist_ok=True)

            args_dict = vars(deepcopy(self.args))
            for k, v in args_dict.items():
                args_dict[k] = str(v)
            with open(os.path.join(self.save_folder, 'config.json'), "w") as file:
                json.dump(args_dict, file, indent=4, sort_keys=False)
            
            if to_log:
                self.writer = SummaryWriter(log_dir=self.save_folder)
            else:
                self.writer = None

            save_folder_bytes = self.save_folder.encode()
            folder_len_tensor = torch.tensor([len(save_folder_bytes)], device=self.state.accelerator.device)
            dist.broadcast(folder_len_tensor, src=0)
            folder_tensor = torch.ByteTensor(list(save_folder_bytes)).to(self.state.accelerator.device)
            dist.broadcast(folder_tensor, src=0)
        else:
            folder_len_tensor = torch.tensor([0], device=self.state.accelerator.device)
            dist.broadcast(folder_len_tensor, src=0)
            folder_tensor = torch.empty(folder_len_tensor.item(), dtype=torch.uint8, device=self.state.accelerator.device)
            dist.broadcast(folder_tensor, src=0)
            self.save_folder = bytes(folder_tensor.tolist()).decode()

        init_logging(self.save_folder, rank=self.state.accelerator.process_index)


    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to

        if getattr(self.args, "use_deepspeed", False):
            per_device_bs = self.args.batch_size
            world_size = int(os.environ.get("WORLD_SIZE", 1))  # 或 self.args.world_size
            grad_accum = self.args.gradient_accumulation_steps

            self.args.deepspeed = build_deepspeed_batch_config(
                self.args.deepspeed,
                per_device_batch_size=per_device_bs,
                world_size=world_size,
                gradient_accumulation_steps=grad_accum,
            )
            ds_plugin = DeepSpeedPlugin(
                hf_ds_config=self.args.deepspeed,
                gradient_accumulation_steps=grad_accum
            )
        else:
            ds_plugin = None

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
            deepspeed_plugin=ds_plugin,
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.state.accelerator = accelerator

        if self.args.seed is not None:
            self.state.seed = self.args.seed
            set_seed(self.args.seed)

        weight_dtype = torch.float32
        if self.state.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.state.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            
        self.state.weight_dtype = weight_dtype


    def _init_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.state.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized Trainer")
        logger.info(self.state.accelerator.state, main_process_only=False)
        

    def _init_directories_and_repositories(self):
        if self.state.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)
            self.state.output_dir = self.args.output_dir


    def prepare_dataset(self) -> None:

        logger.info(f"Training Dataset: {self.args.train_data_class}")
        train_dataset_class = import_custom_class(
            self.args.train_data_class, self.args.train_data_class_path
        )
        self.train_dataset = train_dataset_class(**self.args.data['train'])

        self.train_sampler = EpochSeededRandomSampler(
            self.train_dataset,
            seed=self.sampler_seed,
        )
        self.train_dataloader_generator = torch.Generator()
        self.train_dataloader_generator.manual_seed(self.sampler_seed)
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            sampler=self.train_sampler,
            batch_size=self.args.batch_size,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=getattr(self.args, "pin_memory", False),
            persistent_workers=self.args.dataloader_num_workers > 0,
            prefetch_factor=(
                getattr(self.args, "dataloader_prefetch_factor", 2)
                if self.args.dataloader_num_workers > 0
                else None
            ),
            multiprocessing_context=None,
            generator=self.train_dataloader_generator,
        )
        logger.info(f">>>>>>>>>>>>>Total Train Eps: {len(self.train_dataset)}<<<<<<<<<<<<<<<<<<\n")


        if 'val' in self.args.data:
            self.prepare_val_dataset()


    def prepare_val_dataset(self) -> None:
        if not hasattr(self.args, "val_data_class"):
            self.args.val_data_class = self.args.train_data_class
        logger.info(f"Validation Dataset: {self.args.val_data_class}")

        val_dataset_class = import_custom_class(
            self.args.val_data_class, self.args.val_data_class_path
        )
        self.val_dataset = val_dataset_class(**self.args.data['val'])

        self.val_index = []
        for _ in range(self.args.batch_size):
            self.val_index.append(random.randint(0, len(self.val_dataset)-1))
        if self.state.accelerator.is_main_process:
            with open(os.path.join(self.save_folder, 'idx.txt'), "w") as file:
                file.write(", ".join(map(str, self.val_index)))

        subset = torch.utils.data.Subset(self.val_dataset, self.val_index)
        self.val_dataloader = torch.utils.data.DataLoader(
            subset, batch_size=self.args.batch_size, shuffle=getattr(self.args, "val_shuffle", False)
        )
        logger.info(f">>>>>>>>>>>>>Total Validatoin Eps: {len(self.val_dataset)}<<<<<<<<<<<<<<<<<<\n")


    def prepare_models(self):

        logger.info("Initializing models")
        device = self.state.accelerator.device
        dtype = self.state.weight_dtype
        semantic_config = getattr(self.args, "semantic_plan", {})
        semantic_source = semantic_config.get("source")
        if semantic_source in BATON_SOURCES:
            # Baton provenance and source ownership fail closed before any
            # tokenizer, VAE, or LTX allocation in direct Trainer use.
            validate_baton_dataset_sampling_contract(
                self.train_dataset,
                expected_seed=self.sampler_seed,
            )
            if semantic_source == BATON_PREDICTION_SOURCE:
                manifest_path = Path(
                    self.raw_config["data"]["train"]["manifest_path"]
                )
                if not manifest_path.is_file():
                    raise ValueError(
                        f"Stage-3 HDF5 manifest is missing: {manifest_path}"
                    )
                self.baton_stage3_artifact_chain = (
                    validate_baton_stage3_artifact_chain(
                        stage2_checkpoint=self.raw_config.get(
                            "stage2_init_checkpoint",
                            "",
                        ),
                        expected_stage2_topology_hash=self.raw_config.get(
                            "stage2_init_topology_hash",
                            "",
                        ),
                        planner_checkpoint=semantic_config.get(
                            "planner_checkpoint",
                            "",
                        ),
                        expected_planner_topology=semantic_config.get(
                            "expected_planner_topology",
                            "",
                        ),
                        expected_hdf5_manifest_hash=sha256_file(
                            manifest_path
                        ),
                    )
                )
                stage2_model_dir = (
                    self.baton_stage3_artifact_chain.diffusion_model_dir
                )
                configured_model_path = Path(
                    self.raw_config["diffusion_model"]["model_path"]
                )
                if configured_model_path != stage2_model_dir:
                    raise ValueError(
                        "Stage-3 diffusion model path differs from the "
                        "validated Stage-2 checkpoint"
                    )
                self.args.diffusion_model["model_path"] = str(stage2_model_dir)
            self.baton_components = prepare_baton_conditioning(
                self.raw_config,
                self.train_dataset,
                device=device,
                dtype=dtype,
            )
            self.baton_training_provenance = (
                build_baton_training_provenance(
                    self.raw_config,
                    stage1_metadata=(
                        self.baton_stage3_artifact_chain.stage1_metadata
                        if self.baton_stage3_artifact_chain is not None
                        else None
                    ),
                )
            )

        ### Load Tokenizer
        tokenizer_class = import_custom_class(
            self.args.tokenizer_class, getattr(self.args, "tokenizer_class_path", "transformers")
        )
        textenc_class = import_custom_class(
            self.args.textenc_class, getattr(self.args, "textenc_class_path", "transformers")
        )
        cond_models = load_condition_models(
            tokenizer_class, textenc_class,
            self.args.pretrained_model_name_or_path if not hasattr(self.args, "tokenizer_pretrained_model_name_or_path") else self.args.tokenizer_pretrained_model_name_or_path,
            load_weights=True
        )
        self.tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
        self.text_encoder = text_encoder.to(device, dtype=dtype).eval()
        self.text_encoder.requires_grad_(False)
        self.text_uncond = get_text_conditions(self.tokenizer, self.text_encoder, prompt="")
        self.uncond_prompt_embeds = self.text_uncond['prompt_embeds']
        self.uncond_prompt_attention_mask = self.text_uncond['prompt_attention_mask']

        ### Load VAE
        vae_class = import_custom_class(
            self.args.vae_class, getattr(self.args, "vae_class_path", "transformers")
        )
        if getattr(self.args, 'vae_path', False):
            self.vae = load_vae_models(vae_class, self.args.vae_path).to(device, dtype=dtype).eval()
        else:
            self.vae = load_latent_models(vae_class, self.args.pretrained_model_name_or_path)["vae"].to(device, dtype=dtype).eval()
        if isinstance(self.vae.latents_mean, List):
            self.vae.latents_mean = torch.FloatTensor(self.vae.latents_mean)
        if isinstance(self.vae.latents_std, List):
            self.vae.latents_std = torch.FloatTensor(self.vae.latents_std)
        if self.vae is not None:
            self.vae.requires_grad_(False)
            if self.args.enable_slicing:
                self.vae.enable_slicing()
            if self.args.enable_tiling:
                self.vae.enable_tiling()
        self.SPATIAL_DOWN_RATIO = self.vae.spatial_compression_ratio
        self.TEMPORAL_DOWN_RATIO = self.vae.temporal_compression_ratio
        logger.info(f'SPATIAL_DOWN_RATIO of VAE :{self.SPATIAL_DOWN_RATIO}')
        logger.info(f'TEMPORAL_DOWN_RATIO of VAE :{self.TEMPORAL_DOWN_RATIO}')


        ### Load Diffusion Model
        diffusion_model_class = import_custom_class(
            self.args.diffusion_model_class, getattr(self.args, "diffusion_model_class_path", "transformers")
        )
        if semantic_source == BATON_PREDICTION_SOURCE:
            if self.baton_stage3_artifact_chain is None:
                raise RuntimeError(
                    "Stage-3 artifact chain was lost before LTX allocation"
                )
            self.diffusion_model = diffusion_model_class(
                **self.args.diffusion_model["config"]
            ).to(device=device, dtype=dtype)
            strict_load_baton_stage3_diffusion_model(
                self.diffusion_model,
                self.baton_stage3_artifact_chain.diffusion_model_dir,
                expected_snapshot_topology_hash=self.raw_config[
                    "stage2_init_topology_hash"
                ],
                expected_diffusion_files=(
                    self.baton_stage3_artifact_chain.diffusion_files
                ),
            )
        else:
            self.diffusion_model = load_diffusion_model(
                model_cls=diffusion_model_class,
                model_dir=self.args.diffusion_model['model_path'],
                load_weights=self.args.load_weights and getattr(self.args, "load_diffusion_model_weights", True),
                **self.args.diffusion_model['config']
            ).to(device, dtype=dtype)
        total_params = count_model_parameters(self.diffusion_model)
        logger.info(f'Total parameters for transformer model:{total_params}')

        if semantic_config.get("enabled", False):
            semantic_source = semantic_config.get("source", "gt_siglip2")
            if semantic_source in BATON_SOURCES:
                if self.baton_components is None:
                    raise RuntimeError("Baton conditioning was lost before LTX setup")
            elif semantic_source == "gt_siglip2":
                self.semantic_encoder = OnlineSiglip2SemanticEncoder(
                    semantic_config["model_name_or_path"],
                    device=device,
                    dtype=dtype,
                    frame_microbatch_size=int(semantic_config.get("frame_microbatch_size", 32)),
                    expected_tokens=int(semantic_config.get("tokens_per_frame", 256)),
                    expected_feature_dim=int(semantic_config.get("feature_dim", 1024)),
                )
            elif semantic_source == "vlm_planner":
                self.semantic_planner = FrozenDualCameraVLMPlanner.from_checkpoint(
                    semantic_config["planner_checkpoint"],
                    device=device,
                    dtype=dtype,
                )
            elif semantic_source == "qwen35_grounded":
                cache = getattr(self.train_dataset, "cache", None)
                cache_hash = getattr(cache, "cache_hash", None)
                cache_dir = getattr(self.train_dataset, "hindsight_cache", None)
                if not isinstance(cache_hash, str) or not cache_hash:
                    raise ValueError(
                        "qwen35_grounded requires a validated hindsight dataset"
                    )
                if cache_dir is None:
                    raise ValueError(
                        "qwen35_grounded dataset must expose hindsight_cache"
                    )
                condition_dim = int(
                    self.args.diffusion_model["config"][
                        "semantic_plan_in_dim"
                    ]
                )
                self.grounded_provider = load_qwen35_grounded_provider(
                    semantic_config["planner_checkpoint"],
                    hindsight_cache_hash=cache_hash,
                    cache_dir=cache_dir,
                    dataset=self.train_dataset,
                    condition_dim=condition_dim,
                    device=device,
                    dtype=dtype,
                )
                self.grounded_training_enabled = True
            else:
                raise ValueError(f"unknown semantic_plan.source: {semantic_source}")


        ### Load Diffuser Scheduler
        diffusion_scheduler_class = import_custom_class(
            self.args.diffusion_scheduler_class, getattr(self.args, "diffusion_scheduler_class_path", "diffusers")
        )
        if hasattr(self.args, "diffusion_scheduler_args"):
            self.scheduler = diffusion_scheduler_class(**self.args.diffusion_scheduler_args)
        else:
            self.scheduler = diffusion_scheduler_class()

        ### Import Inference Pipeline Class
        self.pipeline_class = import_custom_class(
            self.args.pipeline_class, getattr(self.args, "pipeline_class_path", "diffusers")
        )


    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")
        
        freeze_conditioning_modules(
            self.text_encoder,
            self.vae,
            self.semantic_encoder,
            getattr(self.semantic_planner, "wrapper", None),
            getattr(self.baton_components, "teacher", None),
            getattr(self.baton_components, "planner", None),
        )

        if torch.backends.mps.is_available() and self.state.weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        if self.args.gradient_checkpointing:
            self.diffusion_model.enable_gradient_checkpointing()
            if self.grounded_provider is not None:
                backbone = self.grounded_provider.planner.backbone
                enable = getattr(backbone, "gradient_checkpointing_enable", None)
                if callable(enable):
                    try:
                        enable(
                            gradient_checkpointing_kwargs={
                                "use_reentrant": False,
                            }
                        )
                    except TypeError:
                        enable()

        if self.grounded_provider is not None:
            self.qwen_ownership = (
                configure_qwen_top_layers_for_joint_training(
                    self.grounded_provider,
                    top_language_layers=8,
                )
            )

        # Enable TF32 for faster training on Ampere GPUs: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True


    def prepare_optimizer(self):
        logger.info("Initializing optimizer and lr scheduler")

        train_mode = self.args.train_mode

        self.state.train_epochs = self.args.train_epochs
        self.state.train_steps = self.args.train_steps

        # Make sure the trainable params are in float32
        if self.args.mixed_precision == "fp16":
            models = [self.diffusion_model]
            if self.grounded_provider is not None:
                models.append(self.grounded_provider)
            cast_training_params(models, dtype=torch.float32)

        self.state.learning_rate = self.args.lr
        semantic_learning_rate = self.args.semantic_lr
        action_learning_rate = self.args.action_lr
        qwen_top_learning_rate = self.args.qwen_top_lr
        qwen_vision_learning_rate = self.args.qwen_vision_lr
        if self.args.scale_lr:
            lr_scale = (
                self.args.gradient_accumulation_steps
                * self.args.batch_size
                * self.state.accelerator.num_processes
            )
            self.state.learning_rate *= lr_scale
            semantic_learning_rate *= lr_scale
            action_learning_rate *= lr_scale
            qwen_top_learning_rate *= lr_scale
            qwen_vision_learning_rate *= lr_scale

        params_to_optimize = build_optimizer_parameter_groups(
            self.diffusion_model,
            train_mode=train_mode,
            base_lr=self.state.learning_rate,
            semantic_lr=semantic_learning_rate,
            action_lr=action_learning_rate,
            provider=self.grounded_provider,
            qwen_top_lr=qwen_top_learning_rate,
            qwen_vision_lr=qwen_vision_learning_rate,
            qwen_ownership=self.qwen_ownership,
            baton_source=(
                self.baton_components.source
                if self.baton_components is not None
                else None
            ),
        )
        trainable_params = [
            parameter for group in params_to_optimize for parameter in group["params"]
        ]
        num_trainable_params = sum(parameter.numel() for parameter in trainable_params)
        logger.info(f'Total trainable parameters: {num_trainable_params}')
        logger.info(
            "Optimizer groups: %s",
            {group["name"]: {"lr": group["lr"], "params": sum(p.numel() for p in group["params"])} for group in params_to_optimize},
        )
        self.state.num_trainable_parameters = sum(p.numel() for p in trainable_params)

        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.lr,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_8bit = self.args.optimizer_8bit,
            use_torchao = self.args.optimizer_torchao,
        )

        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.state.train_steps is None:
            self.state.train_steps = self.state.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.state.accelerator.num_processes,
            num_training_steps=self.state.train_steps * self.state.accelerator.num_processes,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        

    def prepare_for_training(self):
        if not self.grounded_training_enabled:
            self.diffusion_model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.state.accelerator.prepare(
                self.diffusion_model, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
            if self.baton_components is not None:
                microbatches_per_epoch = len(self.train_dataloader)
                resume_from = getattr(
                    self.args,
                    "resume_from_checkpoint",
                    None,
                )
                if resume_from:
                    self.resume_cursor = load_baton_training_checkpoint(
                        self.state.accelerator,
                        resume_from,
                        diffusion_model=unwrap_model(
                            self.state.accelerator,
                            self.diffusion_model,
                        ),
                        expected_source=self.baton_components.source,
                        expected_microbatches_per_epoch=(
                            microbatches_per_epoch
                        ),
                        expected_sampler_seed=self.sampler_seed,
                        expected_training_provenance=(
                            self.baton_training_provenance
                        ),
                    )
                else:
                    self.resume_cursor = TrainingCursor(
                        global_step=0,
                        epoch=0,
                        consumed_microbatches=0,
                        microbatches_per_epoch=microbatches_per_epoch,
                        sampler_seed=self.sampler_seed,
                    )
                self.current_cursor = self.resume_cursor
            return
        if self.grounded_provider is None:
            raise RuntimeError("grounded provider was lost before joint preparation")
        self.joint_model = JointGroundedTrainingModel(
            diffusion_model=self.diffusion_model,
            semantic_provider=self.grounded_provider,
        )
        (
            self.joint_model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = prepare_joint_training_components(
            self.state.accelerator,
            self.joint_model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        )
        self.diffusion_model = self.joint_model
        self.grounded_provider = None
        microbatches_per_epoch = len(self.train_dataloader)
        resume_from = getattr(self.args, "resume_from_checkpoint", None)
        if resume_from:
            unwrapped = unwrap_model(
                self.state.accelerator,
                self.joint_model,
            )
            self.resume_cursor = load_joint_training_checkpoint(
                self.state.accelerator,
                resume_from,
                joint_model=unwrapped,
                expected_microbatches_per_epoch=microbatches_per_epoch,
                expected_sampler_seed=self.sampler_seed,
            )
        else:
            self.resume_cursor = TrainingCursor(
                global_step=0,
                epoch=0,
                consumed_microbatches=0,
                microbatches_per_epoch=microbatches_per_epoch,
                sampler_seed=self.sampler_seed,
            )
        self.current_cursor = self.resume_cursor

    def _unwrapped_diffusion_model(self, accelerator):
        model = unwrap_model(accelerator, self.diffusion_model)
        if isinstance(model, JointGroundedTrainingModel):
            return model.diffusion_model
        return model

    def _unwrapped_grounded_provider(self, accelerator):
        model = unwrap_model(accelerator, self.diffusion_model)
        if not isinstance(model, JointGroundedTrainingModel):
            raise RuntimeError("grounded provider requires a prepared joint model")
        return model.semantic_provider


    def prepare_trackers(self):
        logger.info("Initializing trackers")
        tracker_name = self.args.tracker_name or "model_train"
        self.state.accelerator.init_trackers(tracker_name, config=self.args.__dict__)


    def train(self):
        logger.info("Starting training")
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.train_batch_size = (
            self.args.batch_size * self.state.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.train_dataset),
            "train epochs": self.state.train_epochs,
            "train steps": self.state.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.train_dataloader),
            "train batch size": self.state.train_batch_size,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")
        
        if self.grounded_training_enabled or self.baton_components is not None:
            if not isinstance(self.resume_cursor, TrainingCursor):
                raise RuntimeError("resumable training cursor was not prepared")
            cursor = self.resume_cursor
        else:
            cursor = TrainingCursor(
                global_step=0,
                epoch=0,
                consumed_microbatches=0,
                microbatches_per_epoch=len(self.train_dataloader),
                sampler_seed=self.sampler_seed,
            )
        global_step = cursor.global_step
        first_epoch = cursor.epoch
        initial_global_step = global_step
        progress_bar = tqdm(
            range(0, self.state.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.state.accelerator.is_local_main_process,
        )

        accelerator = self.state.accelerator
        weight_dtype = self.state.weight_dtype
        scheduler_sigmas = self.scheduler.sigmas.clone().to(device=accelerator.device, dtype=weight_dtype)
        for epoch in range(first_epoch, self.state.train_epochs):
            if global_step >= self.state.train_steps:
                break

            logger.debug(f"Starting epoch ({epoch + 1}/{self.state.train_epochs})")

            self.diffusion_model.train()

            running_loss = 0.0
            set_dataloader_epoch(
                self.train_dataloader,
                epoch=epoch,
                sampler_seed=cursor.sampler_seed,
            )
            skipped_microbatches = (
                cursor.consumed_microbatches if epoch == first_epoch else 0
            )
            epoch_dataloader = accelerator.skip_first_batches(
                self.train_dataloader,
                num_batches=skipped_microbatches,
            )
            for step, batch in enumerate(epoch_dataloader):
                absolute_microbatch = skipped_microbatches + step + 1
                logger.debug(f"Starting step {step + 1}")
                logs = {}
                with accelerator.accumulate(self.diffusion_model):
                    
                    video = batch['video']

                    # shape: {b, c, v, t, h, w}; ranging from -1 to 1
                    video = video.to(accelerator.device, dtype=weight_dtype).contiguous()
                    batch_size, c, n_view, _, h, w = video.shape
                    mem_size = self.args.data['train']['n_previous']
                    baton_video = (
                        video
                        if self.baton_components is not None
                        else None
                    )
                    semantic_keyframes = None
                    planner_semantic_plan = None
                    planner_semantic_times = None
                    planner_aux_loss = None
                    if self.semantic_encoder is not None:
                        semantic_config = self.args.semantic_plan
                        semantic_future = rearrange(
                            video[:, :, :, mem_size:],
                            'b c v t h w -> b v t c h w',
                        )
                        semantic_keyframes = select_future_keyframes(
                            semantic_future,
                            indices=tuple(semantic_config.get("keyframe_indices", (0, 3, 5, 8))),
                        ).contiguous()
                    elif self.semantic_planner is not None:
                        planner_semantic_plan, planner_semantic_times = (
                            build_vlm_semantic_condition(
                                self.semantic_planner,
                                video,
                                batch['caption'],
                                n_previous=mem_size,
                            )
                        )
                    video = rearrange(video, 'b c v t h w -> (b v) c t h w')

                    # here we use color jitter to the video, with different views or different batches different jitter
                    if self.args.use_color_jitter:
                        video = apply_color_jitter_to_video(video)

                    mem = video[:,:,:mem_size]
                    future_video = select_training_future_video(
                        video,
                        n_previous=mem_size,
                        chunk=self.args.data["train"]["chunk"],
                        return_action=self.args.return_action,
                        return_video=self.args.return_video,
                    )

                    # get the shape params
                    _, _, raw_frames, raw_height, raw_width = future_video.shape

                    latent_frames = compute_ltx_latent_frames(
                        raw_frames,
                        temporal_compression_ratio=self.TEMPORAL_DOWN_RATIO,
                        n_previous=mem_size,
                    )
                    latent_height = raw_height // self.SPATIAL_DOWN_RATIO
                    latent_width = raw_width // self.SPATIAL_DOWN_RATIO

                    semantic_plan = None
                    semantic_plan_times = None
                    semantic_plan_positions = None
                    semantic_plan_mask = None
                    semantic_plan_relevance = None
                    semantic_condition_mask = None
                    if self.baton_components is not None:
                        baton_condition = build_baton_semantic_condition(
                            self.baton_components,
                            self.args.semantic_plan,
                            baton_video,
                            batch["caption"],
                            n_previous=mem_size,
                            num_future_frames=raw_frames,
                            num_latent_frames=latent_frames,
                            device=accelerator.device,
                            dtype=weight_dtype,
                        )
                        semantic_plan = baton_condition.tokens
                        semantic_plan_times = baton_condition.times
                        semantic_plan_positions = baton_condition.positions
                        semantic_plan_mask = baton_condition.mask
                        semantic_plan_relevance = baton_condition.relevance
                    elif self.semantic_encoder is not None:
                        semantic_plan = self.semantic_encoder.encode(semantic_keyframes)
                        semantic_plan_times = build_semantic_plan_times(
                            batch_size=batch_size,
                            n_view=n_view,
                            n_previous=mem_size,
                            num_future_frames=raw_frames,
                            num_latent_frames=latent_frames,
                            indices=tuple(self.args.semantic_plan.get("keyframe_indices", (0, 3, 5, 8))),
                            device=accelerator.device,
                        )
                    elif self.semantic_planner is not None:
                        semantic_plan = planner_semantic_plan.to(
                            device=accelerator.device,
                            dtype=weight_dtype,
                        )
                        semantic_plan_times = planner_semantic_times.to(
                            device=accelerator.device,
                            dtype=torch.float32,
                        )
                    elif self.grounded_training_enabled:
                        semantic_plan_times = build_semantic_plan_times(
                            batch_size=batch_size,
                            n_view=n_view,
                            n_previous=mem_size,
                            num_future_frames=raw_frames,
                            num_latent_frames=latent_frames,
                            indices=tuple(
                                self.args.semantic_plan.get(
                                    "keyframe_indices",
                                    (0, 3, 5, 8),
                                )
                            ),
                            device=accelerator.device,
                        )
                    if semantic_plan is not None or self.grounded_training_enabled:
                        semantic_condition_mask = sample_semantic_condition_mask(
                            batch_size=batch_size,
                            n_view=n_view,
                            dropout_probability=float(self.args.semantic_plan.get("dropout", 0.15)),
                            device=accelerator.device,
                            dtype=weight_dtype,
                        )

                    dropout_factor = torch.rand(batch_size).to(accelerator.device, dtype=weight_dtype)
                    dropout_mask_prompt = dropout_factor < self.args.caption_dropout_p
                    dropout_mask_prompt = dropout_mask_prompt.unsqueeze(1).unsqueeze(2)

                    # In action training with noisy_video, the future clip is replaced by pure
                    # noise (ss=1) and carries no video loss, so its VAE encode (~32% of a step,
                    # here of a repeated dummy frame) is pure waste — skip it.
                    _skip_future_encode = (
                        getattr(self.args, "noisy_video", False)
                        and self.args.train_mode in ("action_only", "action_full")
                    )
                    mem_latents, future_video_latents = get_latents(
                        self.vae, mem, future_video, encode_future=not _skip_future_encode
                    )

                    mem_latents = rearrange(mem_latents, '(b v m) (h w) c -> (b v) c m h w', b=batch_size, m=mem_size, h=latent_height)
                    future_video_latents = rearrange(future_video_latents, '(b v) (f h w) c -> (b v) c f h w',b=batch_size,h=latent_height,w=latent_width)
                    latents = torch.cat((mem_latents, future_video_latents), dim=2)

                    video_attention_mask = None
                    latents = rearrange(latents, 'bv c f h w -> bv (f h w) c')

                    captions = batch['caption']
                    text_conds = get_text_conditions(self.tokenizer,self.text_encoder,captions)
                    prompt_embeds = text_conds['prompt_embeds']
                    prompt_attention_mask = text_conds['prompt_attention_mask']
                    prompt_embeds = self.uncond_prompt_embeds.repeat(batch_size,1,1)*dropout_mask_prompt + \
                                    prompt_embeds*~dropout_mask_prompt

                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    action_weights = compute_density_for_timestep_sampling(
                        weighting_scheme=self.args.flow_weighting_scheme,
                        batch_size=batch_size,
                        logit_mean=self.args.flow_logit_mean,
                        logit_std=self.args.flow_logit_std,
                        mode_scale=self.args.flow_mode_scale,
                    )
                    # 0-1, 0 -> most noisy, 1 -> almost clean
                    action_indices = (action_weights * self.scheduler.config.num_train_timesteps).long()
                    action_sigmas = scheduler_sigmas[action_indices]
                    action_timesteps = (action_sigmas * 1000.0).long()

                    if self.args.return_action and self.args.noisy_video:
                        weights = torch.full_like(action_weights, 0.0).unsqueeze(1).repeat(1,n_view)
                    else:
                        weights = action_weights.unsqueeze(1).repeat(1,n_view)

                    weights = rearrange(weights, 'b v -> (b v)')
                    indices = (weights * self.scheduler.config.num_train_timesteps).long()
                    sigmas = scheduler_sigmas[indices]
                    timesteps = (sigmas * 1000.0).long()

                    if self.args.return_action:
                        if getattr(self.args, "add_state", False):
                            # NOTE add states from the batch:
                            act_state = batch['state']
                            if act_state.shape[1] != 1:
                                act_state = act_state[:, mem_size-1:mem_size]
                            act_state = act_state.to(accelerator.device, dtype=weight_dtype).contiguous()
                        else:
                            act_state = None
                            

                        actions = batch['actions'][:, -self.args.data['train']['action_chunk']:].to(accelerator.device, dtype=weight_dtype).contiguous()   # shape b,t,c
                        noise_actions = randn_tensor(actions.shape, device=accelerator.device, dtype=weight_dtype)

                        # here we get action_timesteps, shape (b,) originally, target shape (b, l) 
                        action_timesteps = action_timesteps.unsqueeze(-1).repeat(1, actions.shape[1])
                        action_ss= action_sigmas.reshape(-1, 1, 1).repeat(1, 1, actions.shape[-1])

                        noisy_actions = (1.0 - action_ss) * actions + action_ss * noise_actions

                        action_weights = compute_loss_weighting_for_sd3(
                            weighting_scheme=self.args.flow_weighting_scheme, sigmas=action_sigmas
                        ).reshape(-1, 1, 1).repeat(1, 1, actions.size(-1))
                    else:
                        actions = None
                        action_timesteps = None
                        noisy_actions = None
                        act_state = None

                    # shape:  bv, l, c and bv, l
                    noise, conditioning_mask, cond_indicator = gen_noise_from_condition_frame_latent(
                        mem_latents, latent_frames, latent_height, latent_width, noise_to_condition_frames=self.args.noise_to_first_frame
                    )  # set initial frames noise to 0
                    if self.args.pixel_wise_timestep:
                        # shape: bv, thw
                        timesteps = timesteps.unsqueeze(-1) * (1 - conditioning_mask)
                    else:
                        # shape: bv, t
                        timesteps = timesteps.unsqueeze(-1) * (1 - cond_indicator)

                    # shape: bv,1,c
                    ss = sigmas.reshape(-1, 1, 1).repeat(1, 1, latents.size(-1))
                    if self.args.return_action and self.args.noisy_video:
                        ss = torch.full_like(ss, 1.0)

                    noisy_latents = (1.0 - ss) * latents + ss * noise

                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss, shape bv,1,c
                    weights = compute_loss_weighting_for_sd3(
                        weighting_scheme=self.args.flow_weighting_scheme, sigmas=sigmas
                    ).reshape(-1, 1, 1).repeat(1, 1, latents.size(-1))

                    diffusion_forward_kwargs = dict(
                        timesteps=timesteps, 
                        noisy_latents=noisy_latents,
                        prompt_embeds=prompt_embeds, 
                        prompt_attention_mask=prompt_attention_mask,
                        num_frames=latent_frames,
                        height=latent_height,
                        width=latent_width,
                        n_view=n_view,
                        action_states=noisy_actions,
                        action_timestep=action_timesteps,
                        return_video=self.args.return_video or self.args.return_action,
                        return_action=self.args.return_action,
                        video_attention_mask=video_attention_mask,
                        history_action_state=act_state,
                        condition_mask=conditioning_mask,
                        frame_rate=self.video_frame_rate,
                        temporal_compression_ratio=self.TEMPORAL_DOWN_RATIO,
                        spatial_compression_ratio=self.SPATIAL_DOWN_RATIO,
                        semantic_plan=semantic_plan,
                        semantic_plan_times=semantic_plan_times,
                        semantic_plan_positions=semantic_plan_positions,
                        semantic_plan_mask=semantic_plan_mask,
                        semantic_plan_relevance=semantic_plan_relevance,
                        semantic_condition_mask=semantic_condition_mask,
                    )
                    if self.grounded_training_enabled:
                        for field in (
                            "semantic_plan",
                            "semantic_plan_positions",
                            "semantic_plan_mask",
                            "semantic_plan_relevance",
                        ):
                            diffusion_forward_kwargs.pop(field)
                        joint_output = self.diffusion_model(
                            grounded_batch=batch,
                            qwen_gradient_scale=self.args.qwen_ge_gradient_scale,
                            **diffusion_forward_kwargs,
                        )
                        pred_all = joint_output.latents
                        planner_aux_loss = joint_output.planner_loss
                    elif self.baton_components is not None:
                        for field in (
                            "semantic_plan",
                            "semantic_plan_times",
                            "semantic_plan_positions",
                            "semantic_plan_mask",
                            "semantic_plan_relevance",
                            "semantic_condition_mask",
                        ):
                            diffusion_forward_kwargs.pop(field)
                        pred_all = forward_baton_ge_act(
                            self.diffusion_model,
                            baton_condition,
                            semantic_condition_mask=semantic_condition_mask,
                            diffusion_kwargs=diffusion_forward_kwargs,
                        )["latents"]
                    else:
                        pred_all = forward_pass(
                            model=self.diffusion_model,
                            **diffusion_forward_kwargs,
                        )["latents"]

                    if self.args.train_mode == 'all' or self.args.train_mode == 'video_only':
                        pred = pred_all['video']
                        target = noise - latents
                        loss_video = weights.float() * (pred.float() - target.float()).pow(2)
                        loss_video = loss_video * (1 - conditioning_mask.unsqueeze(-1).repeat(1, 1, loss_video.size(-1)))
                        # Average loss across channel dimension
                        loss_video = loss_video.mean(list(range(1, loss_video.ndim)))
                        # Average loss across batch dimension
                        loss_video = loss_video.mean()
                    else:
                        loss_video = 0.

                    if self.args.train_mode == 'all' or self.args.train_mode == 'action_only' or self.args.train_mode == 'action_full':
                        target_action = noise_actions - actions
                        loss_action = action_weights.float() * (pred_all['action'].float() - target_action.float()).pow(2)    # shape b,l,c
                        loss_action = loss_action.mean()
                    else:
                        loss_action = 0.
                    action_loss_scale = getattr(self.args, "action_loss_scale", 1.0)

                    if planner_aux_loss is None:
                        loss = loss_video + action_loss_scale * loss_action
                    else:
                        loss = compute_joint_loss(
                            loss_video=loss_video,
                            loss_action=loss_action,
                            planner_loss=planner_aux_loss,
                            action_loss_scale=action_loss_scale,
                            planner_aux_weight=self.args.planner_aux_weight,
                        )

                    assert not torch.isnan(loss), "NaN loss detected"
                    accelerator.backward(loss)
                    if accelerator.sync_gradients and accelerator.distributed_type != DistributedType.DEEPSPEED:
                        grad_norm = accelerator.clip_grad_norm_(self.diffusion_model.parameters(), self.args.max_grad_norm)
                        logs["grad_norm"] = grad_norm
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                

                loss = accelerator.reduce(loss.detach(), reduction='mean')
                if self.args.train_mode == 'all' or self.args.train_mode == 'action_only' or self.args.train_mode == 'action_full':
                    loss_action = accelerator.reduce(loss_action.detach(), reduction='mean')
                if self.args.train_mode == 'all' or self.args.train_mode == 'video_only':
                    loss_video = accelerator.reduce(loss_video.detach(), reduction='mean')

                running_loss += loss.item()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                cursor = advance_training_cursor(
                    cursor,
                    epoch=epoch,
                    consumed_microbatches=absolute_microbatch,
                    global_step=global_step,
                )
                self.current_cursor = cursor

                logs = {"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                if self.baton_components is not None:
                    train_mode_label = (
                        "teacher"
                        if self.baton_components.source == BATON_TEACHER_SOURCE
                        else "prediction"
                    )
                    logs[
                        f"train/{self.baton_components.source}/"
                        f"{train_mode_label}/video"
                    ] = loss_video.detach().item()
                    logs[
                        f"train/{self.baton_components.source}/"
                        f"{train_mode_label}/action"
                    ] = loss_action.detach().item()
                progress_bar.set_postfix(logs)
                accelerator.log(logs, step=global_step)

                if global_step >= self.state.train_steps:
                    logger.info(">>> max train step reached")
                    break

                if global_step % self.args.steps_to_log == 0:
                    if accelerator.is_main_process:
                        if self.writer is not None:
                            self.writer.add_scalar("Training Loss", loss.item(), global_step)
                            if self.args.train_mode == 'all' or self.args.train_mode == 'action_only' or self.args.train_mode == 'action_full':
                                self.writer.add_scalar("Action loss", loss_action.mean().item(), global_step)
                            if self.args.train_mode == 'all' or self.args.train_mode == 'video_only':
                                self.writer.add_scalar("Video loss", loss_video.item(), global_step)

                # global_step > 0 守卫: step 0 时 0 % N == 0 恒真会误触发 (grad_accum>1 时首步 global_step 停在 0)
                if accelerator.sync_gradients and global_step > 0 and global_step % self.args.steps_to_val == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        model_save_dir = os.path.join(self.save_folder,f'Validation_step_{global_step}')
                        if self.baton_components is None:
                            self.validate(accelerator, model_save_dir, global_step, n_view=n_view, n_chunk=1)
                        else:
                            self.validate_baton_modes(
                                accelerator,
                                model_save_dir,
                                global_step,
                                n_view=n_view,
                                n_chunk=1,
                            )

                
                if accelerator.sync_gradients and should_save_checkpoint(global_step, self.args):
                    accelerator.wait_for_everyone()
                    if self.grounded_training_enabled:
                        save_joint_training_checkpoint(
                            accelerator,
                            self.save_folder,
                            cursor=cursor,
                            joint_model=unwrap_model(
                                accelerator,
                                self.diffusion_model,
                            ),
                        )
                    elif self.baton_components is not None:
                        save_baton_training_checkpoint(
                            accelerator,
                            self.save_folder,
                            cursor=cursor,
                            diffusion_model=unwrap_model(
                                accelerator,
                                self.diffusion_model,
                            ),
                            source=self.baton_components.source,
                            training_provenance=(
                                self.baton_training_provenance
                            ),
                        )
                    elif accelerator.is_main_process:
                        model_to_save = unwrap_model(accelerator, self.diffusion_model)

                        model_save_dir = os.path.join(self.save_folder,f'step_{global_step}')
                        model_to_save.save_pretrained(model_save_dir, safe_serialization=True)
                        del  model_to_save
                        
            if cursor.epoch == epoch and global_step < self.state.train_steps:
                cursor = TrainingCursor(
                    global_step=global_step,
                    epoch=epoch + 1,
                    consumed_microbatches=0,
                    microbatches_per_epoch=cursor.microbatches_per_epoch,
                    sampler_seed=cursor.sampler_seed,
                )
                self.current_cursor = cursor
            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

            if accelerator.is_main_process and self.writer is not None:
                avg_loss = running_loss / len(self.train_dataloader)
                self.writer.add_scalar("Average Training Loss", avg_loss, epoch)

        accelerator.wait_for_everyone()
        if self.grounded_training_enabled:
            final_checkpoint = Path(self.save_folder) / f"step_{global_step:06d}"
            if not final_checkpoint.exists():
                save_joint_training_checkpoint(
                    accelerator,
                    self.save_folder,
                    cursor=cursor,
                    joint_model=unwrap_model(
                        accelerator,
                        self.diffusion_model,
                    ),
                )
        elif self.baton_components is not None:
            final_checkpoint = Path(self.save_folder) / f"step_{global_step:06d}"
            if not final_checkpoint.exists():
                save_baton_training_checkpoint(
                    accelerator,
                    self.save_folder,
                    cursor=cursor,
                    diffusion_model=unwrap_model(
                        accelerator,
                        self.diffusion_model,
                    ),
                    source=self.baton_components.source,
                    training_provenance=self.baton_training_provenance,
                )
        elif accelerator.is_main_process:
            self.diffusion_model = unwrap_model(accelerator, self.diffusion_model)

            model_save_dir = os.path.join(self.save_folder,f'step_{global_step}')
            self.diffusion_model.save_pretrained(model_save_dir, safe_serialization=True)

        del self.diffusion_model, self.scheduler
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()


    def validate_baton_modes(
        self,
        accelerator,
        model_save_dir,
        global_step,
        *,
        n_view: int,
        n_chunk: int,
        to_log: bool = True,
    ) -> list[dict[str, Any]]:
        """Evaluate enabled/disabled gates from one batch and RNG state."""

        if self.baton_components is None:
            raise RuntimeError("paired Baton validation requires a Baton source")
        batch = next(iter(self.val_dataloader))
        batch_size = 1
        mem_size = self.args.data["train"]["n_previous"]
        raw_future_frames = self.args.data["train"]["chunk"]
        latent_num_frames = compute_ltx_latent_frames(
            raw_future_frames,
            temporal_compression_ratio=self.TEMPORAL_DOWN_RATIO,
            n_previous=mem_size,
        )
        baton_condition = build_baton_semantic_condition(
            self.baton_components,
            self.args.semantic_plan,
            batch["video"][:batch_size],
            batch["caption"][:batch_size],
            n_previous=mem_size,
            num_future_frames=raw_future_frames,
            num_latent_frames=latent_num_frames,
            device=accelerator.device,
            dtype=self.state.weight_dtype,
        )
        base_generator = torch.Generator(
            device=accelerator.device,
        ).manual_seed(42)
        generator_state = base_generator.get_state()
        results: list[dict[str, Any]] = []
        reference_trace = None
        for mode in self.args.semantic_plan.get(
            "validation_modes",
            [self.args.semantic_plan["validation_mode"]],
        ):
            generator = torch.Generator(device=accelerator.device)
            generator.set_state(generator_state)
            result = self.validate(
                accelerator,
                os.path.join(model_save_dir, str(mode)),
                global_step,
                n_view=n_view,
                n_chunk=n_chunk,
                to_log=to_log,
                semantic_mode=str(mode),
                batch=batch,
                baton_condition=baton_condition,
                inference_generator=generator,
                capture_validation_trace=True,
            )
            trace = result.get("validation_trace")
            if type(trace) is not dict:
                raise RuntimeError("Baton validation pipeline omitted its trace")
            if reference_trace is None:
                reference_trace = trace
            else:
                if set(trace) != set(reference_trace):
                    raise RuntimeError("paired Baton validation trace fields differ")
                for field, expected in reference_trace.items():
                    actual = trace[field]
                    if (
                        not isinstance(expected, torch.Tensor)
                        or not isinstance(actual, torch.Tensor)
                        or not torch.equal(actual, expected)
                    ):
                        raise RuntimeError(
                            "paired Baton validation stochastic input "
                            f"differs: {field}"
                        )
            results.append(result)
        return results


    def validate(
        self,
        accelerator,
        model_save_dir,
        global_step,
        n_view=1,
        n_chunk=30,
        image=None,
        prompt=None,
        cap=None,
        path=None,
        gt_actions=None,
        to_log=True,
        semantic_plan=None,
        semantic_plan_times=None,
        semantic_plan_positions=None,
        semantic_plan_mask=None,
        semantic_plan_relevance=None,
        semantic_condition_mask=None,
        semantic_mode=None,
        batch=None,
        baton_condition=None,
        inference_generator=None,
        capture_validation_trace=False,
    ):

        os.makedirs(model_save_dir,exist_ok=True)

        pipe = self.pipeline_class(
            self.scheduler, self.vae, self.text_encoder, self.tokenizer,
            self._unwrapped_diffusion_model(accelerator) if accelerator is not None else self.diffusion_model
        )

        if batch is None:
            batch = next(iter(self.val_dataloader))
        image = batch['video'][:,:,:,:self.args.data['train']['n_previous']].clone()  # shape b,c,v,t,h,w 
        prompt = batch['caption']
        gt_video = batch['video']
        b, c, v, t, h, w = image.shape
        negative_prompt = ''

        batch_size = 1

        image = image[:batch_size]

        if (
            self.baton_components is not None
            or self.semantic_encoder is not None
            or self.semantic_planner is not None
            or self.grounded_training_enabled
        ):
            semantic_mode = semantic_mode or self.args.semantic_plan.get("validation_mode", "gt")
            if self.baton_components is not None:
                mem_size = self.args.data['train']['n_previous']
                raw_future_frames = self.args.data['train']['chunk']
                latent_num_frames = compute_ltx_latent_frames(
                    raw_future_frames,
                    temporal_compression_ratio=self.TEMPORAL_DOWN_RATIO,
                    n_previous=mem_size,
                )
                if baton_condition is None:
                    baton_condition = build_baton_semantic_condition(
                        self.baton_components,
                        self.args.semantic_plan,
                        gt_video[:batch_size],
                        prompt[:batch_size],
                        n_previous=mem_size,
                        num_future_frames=raw_future_frames,
                        num_latent_frames=latent_num_frames,
                        device=accelerator.device,
                        dtype=self.state.weight_dtype,
                    )
                selection = apply_baton_validation_mode(
                    self.baton_components,
                    tokens=baton_condition.tokens,
                    mode=semantic_mode,
                    batch_size=batch_size,
                    n_view=v,
                    device=accelerator.device,
                    dtype=self.state.weight_dtype,
                )
                semantic_plan = selection.tokens
                semantic_plan_times = baton_condition.times
                semantic_plan_positions = baton_condition.positions
                semantic_plan_mask = baton_condition.mask
                semantic_plan_relevance = baton_condition.relevance
                semantic_condition_mask = selection.condition_mask
            elif semantic_mode == "gt":
                if self.semantic_encoder is None:
                    raise ValueError(
                        "GT semantic validation requires semantic_plan.source=gt_siglip2"
                    )
                mem_size = self.args.data['train']['n_previous']
                raw_future_frames = self.args.data['train']['chunk']
                semantic_future = rearrange(
                    gt_video[:batch_size, :, :, mem_size:mem_size + raw_future_frames],
                    'b c v t h w -> b v t c h w',
                ).to(accelerator.device, dtype=self.state.weight_dtype)
                semantic_plan = self.semantic_encoder.encode(
                    select_future_keyframes(
                        semantic_future,
                        indices=tuple(self.args.semantic_plan.get("keyframe_indices", (0, 3, 5, 8))),
                    )
                )
                latent_num_frames = compute_ltx_latent_frames(
                    raw_future_frames,
                    temporal_compression_ratio=self.TEMPORAL_DOWN_RATIO,
                    n_previous=mem_size,
                )
                semantic_plan_times = build_semantic_plan_times(
                    batch_size=batch_size,
                    n_view=v,
                    n_previous=mem_size,
                    num_future_frames=raw_future_frames,
                    num_latent_frames=latent_num_frames,
                    indices=tuple(self.args.semantic_plan.get("keyframe_indices", (0, 3, 5, 8))),
                    device=accelerator.device,
                )
                semantic_condition_mask = torch.ones(
                    batch_size * v,
                    device=accelerator.device,
                    dtype=self.state.weight_dtype,
                )
            elif semantic_mode == "planner":
                if (
                    self.semantic_planner is None
                    and not self.grounded_training_enabled
                ):
                    raise ValueError(
                        "planner semantic validation requires a planner source"
                    )
                if self.grounded_training_enabled:
                    provider = self._unwrapped_grounded_provider(accelerator)
                    grounded_condition = (
                        generate_grounded_condition_for_validation(
                            provider,
                            {
                                "current_images": batch[
                                    "current_images"
                                ][:batch_size],
                                "caption": prompt[:batch_size],
                            },
                            qwen_gradient_scale=(
                                self.args.qwen_ge_gradient_scale
                            ),
                        )
                    )
                    semantic_plan = grounded_condition.tokens.to(
                        device=accelerator.device,
                        dtype=self.state.weight_dtype,
                    )
                    semantic_plan_positions = (
                        grounded_condition.positions.to(
                            device=accelerator.device,
                            dtype=torch.float32,
                        )
                    )
                    semantic_plan_mask = grounded_condition.mask.to(
                        device=accelerator.device,
                    )
                    semantic_plan_relevance = (
                        grounded_condition.relevance.to(
                            device=accelerator.device,
                            dtype=self.state.weight_dtype,
                        )
                    )
                    mem_size = self.args.data['train']['n_previous']
                    raw_future_frames = self.args.data['train']['chunk']
                    latent_num_frames = compute_ltx_latent_frames(
                        raw_future_frames,
                        temporal_compression_ratio=self.TEMPORAL_DOWN_RATIO,
                        n_previous=mem_size,
                    )
                    semantic_plan_times = build_semantic_plan_times(
                        batch_size=batch_size,
                        n_view=v,
                        n_previous=mem_size,
                        num_future_frames=raw_future_frames,
                        num_latent_frames=latent_num_frames,
                        indices=tuple(
                            self.args.semantic_plan.get(
                                "keyframe_indices",
                                (0, 3, 5, 8),
                            )
                        ),
                        device=accelerator.device,
                    )
                else:
                    semantic_plan, semantic_plan_times = build_vlm_semantic_condition(
                        self.semantic_planner,
                        gt_video[:batch_size],
                        prompt[:batch_size],
                        n_previous=self.args.data['train']['n_previous'],
                    )
                    semantic_plan = semantic_plan.to(
                        device=accelerator.device,
                        dtype=self.state.weight_dtype,
                    )
                    semantic_plan_times = semantic_plan_times.to(
                        device=accelerator.device,
                        dtype=torch.float32,
                    )
                semantic_condition_mask = torch.ones(
                    batch_size * v,
                    device=accelerator.device,
                    dtype=self.state.weight_dtype,
                )
            elif semantic_mode == "external":
                if semantic_plan is None or semantic_plan_times is None:
                    raise ValueError("external semantic validation requires plan tensors and times")
            elif semantic_mode == "none":
                semantic_plan = None
                semantic_plan_times = None
                semantic_plan_positions = None
                semantic_plan_mask = None
                semantic_plan_relevance = None
                semantic_condition_mask = None
            else:
                raise ValueError(f"unknown semantic validation mode: {semantic_mode}")

        image = rearrange(image, 'b c v t h w -> (b v) c t h w')
        num_denois_steps = self.args.num_inference_step

        if self.args.return_action and getattr(self.args, "add_state", False):
            history_action_state = batch['state'][:batch_size]
            if history_action_state.shape[1] > 1:
                history_action_state = history_action_state[:, self.args.data['train']['n_previous']-1:self.args.data['train']['n_previous'], :]
            history_action_state = history_action_state.contiguous()
        else:
            history_action_state = None

        preds = pipe.infer(
            image=image,
            prompt=prompt[:batch_size],
            negative_prompt=negative_prompt,
            num_inference_steps=num_denois_steps,
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            guidance_scale=1.0,
            height=h,
            width=w,
            n_view=v,
            return_action=self.args.return_action,
            n_prev=self.args.data['train']['n_previous'],
            chunk=(self.args.data['train']['chunk']-1)//self.TEMPORAL_DOWN_RATIO+1,
            return_video=self.args.return_video,
            noise_seed=42,
            action_chunk=self.args.data['train']['action_chunk'],
            history_action_state = history_action_state,
            pixel_wise_timestep = self.args.pixel_wise_timestep,
            num_frames=self.args.data['train']['chunk'],  # Cosmos infer uses num_frames (future pixel frames = chunk); LTX ignores via kwargs
            postprocess_video=False,  # keep raw (b v) c t h w tensor so preds['video'] works; LTX ignores via kwargs
            n_chunk=n_chunk,
            action_dim=self.args.diffusion_model["config"]["action_in_channels"] if self.args.return_action else None,
            frame_rate=self.video_frame_rate,
            semantic_plan=semantic_plan,
            semantic_plan_times=semantic_plan_times,
            semantic_plan_positions=semantic_plan_positions,
            semantic_plan_mask=semantic_plan_mask,
            semantic_plan_relevance=semantic_plan_relevance,
            semantic_condition_mask=semantic_condition_mask,
            generator=inference_generator,
            capture_validation_trace=capture_validation_trace,
        )[0]

        if self.baton_components is None:
            cap = 'Validation'
        else:
            cap = (
                f"Validation_{self.baton_components.source}_"
                f"{semantic_mode}"
            )
        fps = int(getattr(self.args, "basic_fps", 30) / (self.args.data['train']['action_chunk'] // self.args.data['train']['chunk']))
        save_video(rearrange(gt_video[0].data.cpu(), 'c v t h w -> c t h (v w)', v=n_view), os.path.join(model_save_dir, f'{cap}_gt.mp4'), fps=fps)

        if self.args.return_video:
            video = preds['video'].data.cpu()
            save_video(rearrange(video, '(b v) c t h w -> b c t h (v w)', v=n_view)[0], os.path.join(model_save_dir, f'{cap}.mp4'), fps=fps)
            if self.baton_components is not None and to_log:
                predicted = rearrange(
                    video,
                    "(b v) c t h w -> b c v t h w",
                    b=batch_size,
                    v=n_view,
                )
                target = gt_video[
                    :batch_size,
                    :,
                    :,
                    self.args.data['train']['n_previous'] :,
                ].float().cpu()
                common_frames = min(predicted.shape[3], target.shape[3])
                video_mse = (
                    predicted[:, :, :, :common_frames].float()
                    - target[:, :, :, :common_frames]
                ).square().mean()
                if self.writer is not None:
                    self.writer.add_scalar(
                        baton_validation_metric_name(
                            self.baton_components.source,
                            str(semantic_mode),
                            "video",
                        ),
                        video_mse.item(),
                        global_step,
                    )

        if to_log and self.writer is not None:
            self.writer.add_text(f'step_{global_step}/{cap} prompt:', prompt[0], global_step)

        if self.args.return_action:
            # shape t, c
            gt_actions = batch['actions'][:, -self.args.data['train']['action_chunk']:]
            action_dim = gt_actions.shape[-1]

            action_logs = act_metric(
                preds['action'][:,:,:action_dim].detach().cpu().to(torch.float).numpy()[:batch_size],
                gt_actions[:,:,:action_dim].detach().cpu().to(torch.float).numpy()[:batch_size],
                prefix=(
                    baton_validation_metric_name(
                        self.baton_components.source,
                        str(semantic_mode),
                        "action",
                    )
                    if self.baton_components is not None
                    else cap
                ),
                start_stop_interval=[(0,1),(1,9),(9,25),(25,self.args.data['train']['action_chunk'])]
            )

            if to_log:
                for key, value in action_logs.items():
                    if self.writer is not None:
                        self.writer.add_scalar(key, value, global_step)
        return {
            "batch": batch,
            "semantic_condition_mask": semantic_condition_mask,
            "validation_trace": preds.get("validation_trace"),
        }
