#!/usr/bin/env python3
"""Fine-tune Qwen3-VL-4B as a semantic planner — LINGBOT-DINO variant (full lingbot-vla-v2 recipe).

Independent 4B line (2B CoVT + tasktoken untouched). Differs from the SigLIP scripts:
  * base VLM = Qwen3-VL-4B extracted from robbyant/lingbot-vla-v2-6b (--model-path → extracted dir);
  * target  = DINO-video (1024-d, 256 tokens/keyframe) from lingbot's teacher, online, NOT SigLIP;
  * head    = faithful TaskTokenResampler (LingbotDinoPlanHead), warm-startable from the 6b
              future_video_align_head (--head-warmstart-ckpt);
  * loss    = plain MSE (set via loss weights: mse=1, others=0).
Plan = num_keyframes × 256 tokens (grid_size=16 ⇒ 16²=256), e.g. 5×256 = [B, 1280, 1024].

NOTE: this plan lives in DINO-video space, so the Cosmos WM must be retrained to consume it — a
separate downstream step (not covered here). See lingbot_dino_4b/LINGBOT_DINO_SPEC.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# Per-rank torch.compile / Triton cache dirs. With N ranks sharing ONE cache dir, concurrent
# flex_attention kernel compiles race on the same artifact (FileNotFoundError on
# triton_tem_fused_*.cubin). torchrun exports LOCAL_RANK before this script runs, so scope the
# cache per rank BEFORE importing torch (inductor/triton read these env vars at compile time).
_local_rank = os.environ.get("LOCAL_RANK")
if _local_rank is not None:
    for _cache_var in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        _base = os.environ.get(_cache_var)
        if _base:
            os.environ[_cache_var] = os.path.join(_base, f"rank{_local_rank}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:  # package import for GE-Act provider; flat fallback for script entry point
    from .qwen3vl_wrapper import (
        load_qwen3vl_model_and_processor,
        move_qwen_inputs_to_device,
    )
except ImportError:  # pragma: no cover - exercised by the production script entry point
    from qwen3vl_wrapper import (
        load_qwen3vl_model_and_processor,
        move_qwen_inputs_to_device,
    )

# 4B-specific modules live in the lingbot_dino_4b/ subpackage; add it to the path (flat import,
# matching how lingbot_dino_head imports lingbot_resampler).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lingbot_dino_4b"))
# 2B DINOv3+DA3 line reuses the same head/query; only its teacher encoders differ (lazy-imported).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dinov3_da3_2b"))
from lingbot_dino_head import LingbotDinoPlanHead  # noqa: E402
from dino_video_target import DinoVideoTargetEncoder  # noqa: E402
from depth_target import DepthTargetEncoder  # noqa: E402
try:  # package import in tests/libraries; flat fallback when launched as a script
    from .ge_act_dual_camera import (  # type: ignore[import-not-found]
        DualCameraPlannerCollator,
        GEActDualCameraPlannerDataset,
    )
except ImportError:  # pragma: no cover - exercised by the production script entry point
    from ge_act_dual_camera import (  # noqa: E402
        DualCameraPlannerCollator,
        GEActDualCameraPlannerDataset,
    )
try:  # package import for GE-Act provider; flat fallback for script entry point
    from .distributed_runtime import (  # type: ignore[import-not-found]
        accumulation_context,
        build_accelerator,
        checkpoint_module,
        is_deepspeed,
        is_optimizer_update,
        should_save_periodic_checkpoint,
        validate_runtime_contract,
    )
except ImportError:  # pragma: no cover - exercised by the production script entry point
    from distributed_runtime import (  # noqa: E402
        accumulation_context,
        build_accelerator,
        checkpoint_module,
        is_deepspeed,
        is_optimizer_update,
        should_save_periodic_checkpoint,
        validate_runtime_contract,
    )

import contextlib  # noqa: E402


HEAD_FILES = {
    "plan_head": "plan_head.pt",
    "depth_head": "depth_head.pt",
    "current_plan_head": "current_plan_head.pt",
    "current_depth_head": "current_depth_head.pt",
}

DUAL_CAMERA_EXPORT_METADATA = {
    "planner_input_layout": "separate_camera_images",
    "camera_names": ["main", "wrist"],
    "num_camera_views": 2,
    "camera_head_sharing": "shared_head_per_view_image_context",
    "semantic_output_layout": "batch_view_token_feature",
    "semantic_teacher": "siglip2-large-patch16-256",
    "future_keyframe_offsets": [8],
}


def build_dual_camera_export_metadata(
    *,
    future_keyframe_offsets: Sequence[int],
    num_keyframes: int,
    target_tokens_per_keyframe: int,
    planner_token_count: int,
) -> dict[str, Any]:
    """Build the camera/export contract from the trained planner geometry."""
    offsets = [int(offset) for offset in future_keyframe_offsets]
    keyframes = int(num_keyframes)
    tokens_per_keyframe = int(target_tokens_per_keyframe)
    plan_tokens = int(planner_token_count)
    if (
        keyframes < 1
        or len(offsets) != keyframes
        or any(offset <= 0 for offset in offsets)
        or any(left >= right for left, right in zip(offsets, offsets[1:]))
    ):
        raise ValueError(
            "future_keyframe_offsets must contain one strictly increasing positive "
            f"offset per keyframe, got {offsets} for K={keyframes}"
        )
    if tokens_per_keyframe <= 0 or plan_tokens <= 0:
        raise ValueError("target and planner token counts must be positive")
    metadata = {
        "planner_input_layout": "separate_camera_images",
        "camera_names": ["main", "wrist"],
        "num_camera_views": 2,
        "camera_head_sharing": "shared_head_per_view_image_context",
        "semantic_output_layout": (
            "batch_view_token_feature"
            if keyframes == 1
            else "batch_view_keyframe_token_feature"
        ),
        "semantic_teacher": "siglip2-large-patch16-256",
        "future_keyframe_offsets": offsets,
    }
    if keyframes > 1:
        metadata.update(
            {
                "num_keyframes": keyframes,
                "target_tokens_per_keyframe": tokens_per_keyframe,
                "planner_token_count": plan_tokens,
            }
        )
    return metadata

_INITIALIZATION_METADATA = {
    "use_current_alignment": True,
    "independent_modality_task_tokens": True,
    "num_task_tokens": 64,
    "num_latent_per_keyframe": 64,
    "branch_latent_per_keyframe": 64,
    "num_keyframes": 1,
    "grid_size": 16,
    "semantic_dim": 1024,
    "target_len": 256,
    "target_tokens": 256,
    "video_target_type": "siglip2",
    "has_depth_head": True,
    "depth_grid_size": 16,
    "depth_feature_dim": 2048,
    "depth_target_type": "da3",
    "plan_head_type": "lingbot_dino",
    "sequence_length": 9,
    "keyframe_offsets": [8],
    "latent_len": 4 * 64,
    "total_unique_latent_per_keyframe": 4 * 64,
    "query_layout": (
        "current_dino_64_then_current_depth_64_then_"
        "future_dino_64_then_future_depth_64"
    ),
}


def _metadata_value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return type(actual) is bool and actual == expected
    if isinstance(expected, int):
        return type(actual) is int and actual == expected
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _metadata_value_matches(item, expected_item)
                for item, expected_item in zip(actual, expected, strict=True)
            )
        )
    return actual == expected


def validate_dual_camera_export_metadata(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Reject exports that cannot preserve separate main/wrist view context."""
    if not isinstance(metadata, dict):
        raise ValueError("dual-camera planner metadata must be a dictionary")
    common = {
        key: value
        for key, value in DUAL_CAMERA_EXPORT_METADATA.items()
        if key not in ("semantic_output_layout", "future_keyframe_offsets")
    }
    for field, expected in common.items():
        actual = metadata.get(field)
        if not _metadata_value_matches(actual, expected):
            raise ValueError(
                f"incompatible dual-camera metadata field {field}: "
                f"expected {expected!r}, got {actual!r}"
            )
    offsets = metadata.get("future_keyframe_offsets")
    if not isinstance(offsets, list) or not offsets:
        raise ValueError(
            "incompatible dual-camera metadata field future_keyframe_offsets: "
            f"got {offsets!r}"
        )
    keyframes = len(offsets)
    if keyframes == 1:
        expected_fields = {
            "semantic_output_layout": "batch_view_token_feature",
            "future_keyframe_offsets": [8],
        }
        if metadata.get("num_keyframes", 1) != 1:
            raise ValueError(
                "incompatible dual-camera metadata field num_keyframes: expected 1, "
                f"got {metadata.get('num_keyframes')!r}"
            )
    elif keyframes == 4:
        expected_fields = {
            "semantic_output_layout": "batch_view_keyframe_token_feature",
            "future_keyframe_offsets": [2, 4, 6, 8],
            "num_keyframes": 4,
            "target_tokens_per_keyframe": 256,
            "planner_token_count": 384,
        }
        if metadata.get("sequence_length", 9) != 9:
            raise ValueError(
                "incompatible dual-camera metadata field sequence_length: expected 9, "
                f"got {metadata.get('sequence_length')!r}"
            )
    else:
        raise ValueError(
            "incompatible dual-camera metadata field future_keyframe_offsets: "
            f"expected explicit K1 or K4 geometry, got {offsets!r}"
        )
    for field, expected in expected_fields.items():
        actual = metadata.get(field)
        if not _metadata_value_matches(actual, expected):
            raise ValueError(
                f"incompatible dual-camera metadata field {field}: "
                f"expected {expected!r}, got {actual!r}"
            )
    legacy_input_frame = metadata.get("planner_input_frame")
    if legacy_input_frame not in (None, "separate_camera_images"):
        raise ValueError(
            "incompatible dual-camera metadata field planner_input_frame: "
            "expected 'separate_camera_images', "
            f"got {legacy_input_frame!r}"
        )
    return metadata


def _validate_planner_initialization_metadata(metadata: dict[str, Any]) -> None:
    for field, expected in _INITIALIZATION_METADATA.items():
        actual = metadata.get(field)
        if not _metadata_value_matches(actual, expected):
            raise ValueError(
                f"incompatible planner initialization metadata field {field}: "
                f"expected {expected!r}, got {actual!r}"
            )

    plan_token_ids = metadata.get("plan_token_ids")
    if (
        not isinstance(plan_token_ids, list)
        or len(plan_token_ids) != 4 * 64
        or any(type(token_id) is not int or token_id < 0 for token_id in plan_token_ids)
        or len(set(plan_token_ids)) != 4 * 64
    ):
        raise ValueError(
            "incompatible planner initialization metadata field plan_token_ids: "
            "expected 256 unique non-negative integers"
        )
    expected_token_strings = [f"<|sem_plan_{index}|>" for index in range(4 * 64)]
    if metadata.get("plan_token_strings") != expected_token_strings:
        raise ValueError(
            "incompatible planner initialization metadata field plan_token_strings: "
            "expected four independent groups of 64 ordered plan tokens"
        )


def load_planner_initialization(
    wrapper: nn.Module,
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """Strict-load the four legacy alignment heads into an existing wrapper."""
    checkpoint_dir = Path(checkpoint_dir)
    required_files = ["planner_meta.json", *HEAD_FILES.values()]
    missing = [name for name in required_files if not (checkpoint_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete planner initialization {checkpoint_dir}: missing {missing}"
        )

    try:
        metadata = json.loads(
            (checkpoint_dir / "planner_meta.json").read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(
            f"invalid planner initialization metadata in {checkpoint_dir}"
        ) from error
    if not isinstance(metadata, dict):
        raise ValueError("planner initialization metadata must be a dictionary")
    _validate_planner_initialization_metadata(metadata)

    wrapper_fields = {
        "use_current_alignment": True,
        "independent_modality_task_tokens": True,
        "use_depth": True,
        "plan_head_type": "lingbot_dino",
        "num_task_tokens": 64,
        "num_latent_per_keyframe": 64,
        "branch_latent_per_keyframe": 64,
        "num_keyframes": 1,
        "target_len": 256,
        "latent_len": 4 * 64,
    }
    for field, expected in wrapper_fields.items():
        actual = getattr(wrapper, field, None)
        if not _metadata_value_matches(actual, expected):
            raise ValueError(
                f"incompatible planner wrapper field {field}: "
                f"expected {expected!r}, got {actual!r}"
            )

    wrapper_plan_token_ids = getattr(wrapper, "plan_token_ids", None)
    if (
        not isinstance(wrapper_plan_token_ids, list)
        or metadata["plan_token_ids"] != wrapper_plan_token_ids
    ):
        raise ValueError(
            "incompatible planner initialization metadata field plan_token_ids: "
            "checkpoint token IDs must exactly match the wrapper in order and value"
        )

    head_geometry = {
        "plan_head": 1024,
        "depth_head": 2048,
        "current_plan_head": 1024,
        "current_depth_head": 2048,
    }
    for attribute, expected_dim_out in head_geometry.items():
        head = getattr(wrapper, attribute, None)
        if head is None:
            raise ValueError(f"planner wrapper is missing required head {attribute}")
        expected_fields = {
            "num_keyframes": 1,
            "num_latent_per_keyframe": 64,
            "num_backbone_tokens": 256,
            "dim_out": expected_dim_out,
        }
        for field, expected in expected_fields.items():
            actual = getattr(head, field, None)
            if not _metadata_value_matches(actual, expected):
                raise ValueError(
                    f"incompatible planner wrapper field {attribute}.{field}: "
                    f"expected {expected!r}, got {actual!r}"
                )
        query_embs = getattr(head, "query_embs", None)
        if (
            not torch.is_tensor(query_embs)
            or query_embs.ndim != 2
            or query_embs.shape[0] != 256
        ):
            shape = tuple(query_embs.shape) if torch.is_tensor(query_embs) else None
            raise ValueError(
                f"incompatible planner wrapper field {attribute}.query_embs: "
                f"expected [256, hidden], got {shape}"
            )

    loaded = []
    for attribute, filename in HEAD_FILES.items():
        head = getattr(wrapper, attribute, None)
        state = torch.load(
            checkpoint_dir / filename,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(state, dict):
            raise ValueError(f"planner head file {filename} must contain a state dictionary")
        head.load_state_dict(state, strict=True)
        loaded.append(attribute)
    return {"loaded_heads": loaded, "source": str(checkpoint_dir)}


def _bidirectional_prefix_mask(config, input_embeds, attention_mask, cache_position,
                               past_key_values, position_ids=None,
                               or_mask_function=None, and_mask_function=None):
    """Full bidirectional attention over the (padded) prefix, replacing the causal mask.

    Official LingBot-VLA v2 runs the planner prefix bidirectionally; HF Qwen3-VL 4.57 builds
    its causal mask via ``masking_utils.create_causal_mask``.  We swap in an all-visible 4D
    additive mask that still zeroes out padded keys (works because the model is loaded with
    ``attn_implementation="sdpa"``)."""
    b, s = input_embeds.shape[0], input_embeds.shape[1]
    dtype, device = input_embeds.dtype, input_embeds.device
    m = torch.zeros(b, 1, s, s, dtype=dtype, device=device)
    if attention_mask is not None:
        keep = attention_mask.to(device=device, dtype=torch.bool)  # [b, s] 1=keep, 0=pad
        m = m.masked_fill(~keep[:, None, None, :], torch.finfo(dtype).min)
    return m


@contextlib.contextmanager
def _bidirectional_prefix():
    from transformers.models.qwen3_vl import modeling_qwen3_vl as _q3
    _orig = _q3.create_causal_mask
    _q3.create_causal_mask = _bidirectional_prefix_mask
    try:
        yield
    finally:
        _q3.create_causal_mask = _orig


def _load_lingbot_head_state(src_6b_dir: Path) -> dict:
    """Stream all LingBot current/future DINO/depth alignment heads and queries."""
    from collections import defaultdict

    from safetensors import safe_open

    src = Path(src_6b_dir)
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    head_names = (
        "current_video_align_head",
        "future_video_align_head",
        "depth_align_head",
        "future_depth_align_head",
    )
    query_names = (
        "current_video_align_embs",
        "future_video_align_embs",
        "depth_align_embs",
        "future_depth_align_embs",
    )
    want = [
        key
        for key in index
        if any(f"{name}." in key for name in head_names)
        or key.endswith(query_names)
    ]
    by_file: dict[str, list[str]] = defaultdict(list)
    for k in want:
        by_file[index[k]].append(k)
    state: dict[str, torch.Tensor] = {}
    for fname, keys in by_file.items():
        with safe_open(src / fname, framework="pt", device="cpu") as sf:
            for k in keys:
                state[k] = sf.get_tensor(k)
    return state

PLAN_TOKEN = "<|sem_plan|>"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer, got {value}"
        )
    return parsed


def validate_dataset_source_selection(
    *,
    dataset_root: Path | None,
    fastwam_data_config: Path | None,
    ge_act_data_config: Path | None,
) -> str:
    sources = {
        "legacy": dataset_root,
        "fastwam": fastwam_data_config,
        "ge_act": ge_act_data_config,
    }
    selected = [name for name, value in sources.items() if value is not None]
    if len(selected) != 1:
        raise ValueError(
            "exactly one of --dataset-root, --fastwam-data-config, or "
            "--ge-act-data-config is required"
        )
    return selected[0]


GE_ACT_DUAL_CAMERA_TRAIN_CONTRACT = {
    "valid_cam": [
        "observation.images.image",
        "observation.images.wrist_image",
    ],
    "source_fps": 20,
    "chunk": 9,
    "action_chunk": 36,
    "n_previous": 4,
    "ignore_seek": False,
}


def validate_ge_act_dual_camera_train_config(train_config: dict[str, Any]) -> None:
    """Reject GE-Act configs whose view order or temporal endpoints differ."""
    for field, expected in GE_ACT_DUAL_CAMERA_TRAIN_CONTRACT.items():
        actual = train_config.get(field, False if field == "ignore_seek" else None)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                "GE-Act dual-camera planner requires "
                f"data.train.{field}={expected!r}, got {actual!r}"
            )


def load_ge_act_dual_camera_planner_dataset(
    config_path: str | Path,
    *,
    future_offsets: Sequence[int] = (8,),
) -> GEActDualCameraPlannerDataset:
    import yaml

    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.Loader)
    if not isinstance(config, dict):
        raise ValueError(f"GE-Act config must be a mapping: {config_path}")
    try:
        class_name = str(config["train_data_class"])
        class_source = str(config["train_data_class_path"])
        train_config = dict(config["data"]["train"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "GE-Act config must define train_data_class, train_data_class_path, "
            "and mapping data.train"
        ) from error
    validate_ge_act_dual_camera_train_config(train_config)

    ge_act_root = Path(__file__).resolve().parents[1] / "ge_act"
    source_path = Path(class_source)
    if source_path.suffix == ".py" and not source_path.is_absolute():
        class_source = str(ge_act_root / source_path)
    ge_act_root_string = str(ge_act_root)
    previous_sys_path = list(sys.path)
    previous_working_directory = Path.cwd()
    try:
        if ge_act_root_string not in sys.path:
            sys.path.insert(0, ge_act_root_string)
        from ge_act.utils import import_custom_class

        dataset_class = import_custom_class(class_name, class_source)
        # GE-Act runners construct datasets from the ge_act directory, and their
        # YAML train split may contain paths such as configs/ltx_model/....
        os.chdir(ge_act_root)
        dataset = dataset_class(**train_config)
    finally:
        os.chdir(previous_working_directory)
        sys.path[:] = previous_sys_path
    return GEActDualCameraPlannerDataset(
        dataset,
        n_previous=4,
        future_offsets=future_offsets,
    )


def flatten_camera_teacher_frames(frames: torch.Tensor) -> torch.Tensor:
    if frames.ndim != 5 or frames.shape[-1] != 3:
        raise ValueError(
            f"camera teacher frames must be [B,V,H,W,3], got {tuple(frames.shape)}"
        )
    batch_size, num_views, height, width, _channels = frames.shape
    return frames.permute(0, 1, 4, 2, 3).reshape(
        batch_size * num_views,
        3,
        height,
        width,
    ).contiguous()


def restore_camera_teacher_features(
    encoded: torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
) -> torch.Tensor:
    expected = int(batch_size) * int(num_views)
    if encoded.ndim < 2 or encoded.shape[0] != expected:
        raise ValueError(
            "encoded camera teacher features must have leading dimension "
            f"B*V={expected}, got {tuple(encoded.shape)}"
        )
    return encoded.reshape(int(batch_size), int(num_views), *encoded.shape[1:])


def encode_dual_camera_teacher_targets(
    current_camera_images: torch.Tensor,
    future_camera_images: torch.Tensor,
    *,
    appearance_encoder: Any,
    depth_encoder: Any,
) -> dict[str, torch.Tensor]:
    """Encode one current/future frame pair without losing the view dimension."""
    for name, frames in (
        ("current", current_camera_images),
        ("future", future_camera_images),
    ):
        if frames.ndim != 5 or frames.shape[1] != 2 or frames.shape[-1] != 3:
            raise ValueError(
                f"{name} camera teacher frames must be [B,2,H,W,3], "
                f"got {tuple(frames.shape)}"
            )
        if not torch.isfinite(frames).all():
            raise ValueError(f"{name} camera teacher frames must be finite")
        if frames.min() < -1.0001 or frames.max() > 1.0001:
            raise ValueError(
                f"{name} camera teacher frames must be normalized to [-1,1]"
            )
    if current_camera_images.shape != future_camera_images.shape:
        raise ValueError(
            "current and future camera teacher frames must have the same shape per frame, got "
            f"{tuple(current_camera_images.shape)} and "
            f"{tuple(future_camera_images.shape)}"
        )

    batch_size, num_views = current_camera_images.shape[:2]
    current_bv = (
        flatten_camera_teacher_frames(current_camera_images).float() + 1.0
    ).mul_(0.5).clamp_(0.0, 1.0)
    future_bv = (
        flatten_camera_teacher_frames(future_camera_images).float() + 1.0
    ).mul_(0.5).clamp_(0.0, 1.0)
    with torch.no_grad():
        current_appearance, future_appearance = (
            appearance_encoder.encode_current_and_future(current_bv, future_bv)
        )
        current_depth, future_depth = depth_encoder.encode_current_and_future(
            current_bv,
            future_bv,
        )

    expected_shapes = {
        "appearance current": (batch_size * num_views, 256, 1024),
        "appearance future": (batch_size * num_views, 256, 1024),
        "depth current": (batch_size * num_views, 256, 2048),
        "depth future": (batch_size * num_views, 256, 2048),
    }
    features = {
        "appearance current": current_appearance,
        "appearance future": future_appearance,
        "depth current": current_depth,
        "depth future": future_depth,
    }
    for name, encoded in features.items():
        expected = expected_shapes[name]
        if tuple(encoded.shape) != expected:
            raise ValueError(
                f"{name} teacher features must be {expected}, got {tuple(encoded.shape)}"
            )

    def restore(encoded: torch.Tensor) -> torch.Tensor:
        return restore_camera_teacher_features(
            encoded,
            batch_size=batch_size,
            num_views=num_views,
        ).float()

    return {
        "current_dino_labels": restore(current_appearance),
        "semantic_plan_labels": restore(future_appearance),
        "current_depth_labels": restore(current_depth),
        "depth_plan_labels": restore(future_depth),
    }


def encode_dual_camera_future_targets(
    current_camera_images: torch.Tensor,
    future_camera_images: torch.Tensor,
    *,
    appearance_encoder: Any,
    depth_encoder: Any,
) -> dict[str, torch.Tensor]:
    """Encode K future keyframes while preserving sample, view, and time order.

    Inputs use ``[B,V,K,H,W,3]`` and teacher tokens are concatenated along the
    token axis within each view. WSA depth keeps its explicit backbone-layer axis.
    """
    if (
        current_camera_images.ndim != 5
        or current_camera_images.shape[1] != 2
        or current_camera_images.shape[-1] != 3
    ):
        raise ValueError(
            "current camera teacher frames must be [B,2,H,W,3], got "
            f"{tuple(current_camera_images.shape)}"
        )
    if (
        future_camera_images.ndim != 6
        or future_camera_images.shape[1] != 2
        or future_camera_images.shape[2] < 1
        or future_camera_images.shape[-1] != 3
    ):
        raise ValueError(
            "future camera teacher frames must be [B,2,K,H,W,3] with K>=1, got "
            f"{tuple(future_camera_images.shape)}"
        )
    expected_current_shape = (
        future_camera_images.shape[0],
        future_camera_images.shape[1],
        future_camera_images.shape[3],
        future_camera_images.shape[4],
        future_camera_images.shape[5],
    )
    if tuple(current_camera_images.shape) != tuple(expected_current_shape):
        raise ValueError(
            "current and future camera teacher frames must have the same shape "
            "per frame, got "
            f"{tuple(current_camera_images.shape)} and "
            f"{tuple(future_camera_images.shape)}"
        )
    for name, frames in (
        ("current", current_camera_images),
        ("future", future_camera_images),
    ):
        if not torch.isfinite(frames).all():
            raise ValueError(f"{name} camera teacher frames must be finite")
        if frames.min() < -1.0001 or frames.max() > 1.0001:
            raise ValueError(
                f"{name} camera teacher frames must be normalized to [-1,1]"
            )

    batch_size, num_views, num_keyframes = future_camera_images.shape[:3]

    def normalize(frames: torch.Tensor) -> torch.Tensor:
        return (frames.float() + 1.0).mul_(0.5).clamp_(0.0, 1.0)

    current_bv = normalize(flatten_camera_teacher_frames(current_camera_images))
    keyframes_bv = [
        normalize(flatten_camera_teacher_frames(future_camera_images[:, :, index]))
        for index in range(num_keyframes)
    ]
    with torch.no_grad():
        semantic = appearance_encoder.encode_future_keyframes(
            current_bv,
            keyframes_bv,
            effective_fps=None,
        )
        depth = depth_encoder.encode_future_keyframes(keyframes_bv)

    expected_rows = batch_size * num_views
    expected_tokens = num_keyframes * 256
    if (
        semantic.ndim != 3
        or semantic.shape[0] != expected_rows
        or semantic.shape[1] != expected_tokens
    ):
        raise ValueError(
            "future appearance teacher features must be "
            f"[B*V,{expected_tokens},D], got {tuple(semantic.shape)}"
        )
    if depth.ndim not in (3, 4) or depth.shape[0] != expected_rows:
        raise ValueError(
            "future depth teacher features must be [B*V,K*256,D] or "
            f"[B*V,L,K*256,D], got {tuple(depth.shape)}"
        )
    if depth.shape[-2] != expected_tokens:
        raise ValueError(
            f"future depth teacher token count must be {expected_tokens}, "
            f"got {depth.shape[-2]}"
        )

    semantic = semantic.reshape(
        batch_size,
        num_views,
        expected_tokens,
        semantic.shape[-1],
    ).float()
    if depth.ndim == 4:
        depth = depth.reshape(
            batch_size,
            num_views,
            depth.shape[1],
            expected_tokens,
            depth.shape[-1],
        ).float()
    else:
        depth = depth.reshape(
            batch_size,
            num_views,
            expected_tokens,
            depth.shape[-1],
        ).float()
    return {
        "semantic_plan_labels": semantic,
        "depth_plan_labels": depth,
    }


def configure_gradient_checkpointing(model: nn.Module, *, enabled: bool) -> None:
    if not enabled:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        return
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--init-planner-checkpoint",
        type=Path,
        default=None,
        help=(
            "initialize the Qwen model, processor, token embeddings, and exactly four "
            "alignment heads from an exported planner checkpoint"
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--fastwam-data-config", type=Path, default=None)
    parser.add_argument("--ge-act-data-config", type=Path, default=None)
    parser.add_argument("--fastwam-dataset-dir", action="append", default=[])
    parser.add_argument(
        "--fastwam-text-embedding-cache-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--fastwam-pretrained-norm-stats",
        type=Path,
        default=None,
    )
    parser.add_argument("--plan-label-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    # Online labels (cosmos-style episode sampling): instead of fixed precomputed .pt windows,
    # sample a random sequence-length window per stem per epoch from frame_ranges.json and
    # encode the SigLIP2 targets on the fly — aligns the planner's window distribution with
    # the world model's episode sampling and makes the keyframe scheme an env-free switch.
    parser.add_argument("--online-plan-labels", action="store_true")
    parser.add_argument("--frame-ranges-json", type=Path, default=None)
    parser.add_argument("--siglip2-encoder-path", type=Path, default=None)
    parser.add_argument("--sequence-length", type=int, default=49)
    parser.add_argument(
        "--keyframe-scheme",
        choices=["uniform", "late", "even_future"],
        default="uniform",
    )
    parser.add_argument("--keyframe-gamma", type=float, default=0.6)
    parser.add_argument("--online-grid-size", type=int, default=0, help="<=0 keeps the native SigLIP2 grid")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-one-window-per-stem", action="store_true")
    parser.add_argument("--episode-window-sampling", choices=["random", "round_robin"], default="random")
    parser.add_argument("--episode-window-seed", type=int, default=-1)
    parser.add_argument("--num-keyframes", type=int, default=6)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--semantic-dim", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--expected-global-batch", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument(
        "--plan-head-type", choices=["mlp", "baton_crossattn", "covt", "lingbot_dino"], default="lingbot_dino"
    )
    # lingbot_dino: online DINO-video teacher target + optional head warm-start from the 6b ckpt.
    parser.add_argument("--dino-teacher-ckpt", type=Path, default=None, help="dino_video/teacher_step_*.pth")
    parser.add_argument("--dino-teacher-config", type=Path, default=None, help="dino_video/config.yaml")
    parser.add_argument("--dino-input-size", type=int, default=256)
    # Auxiliary future-DEPTH alignment (lingbot-style): MoGe-2 -> MoRGBD depth-feature teacher + a
    # second warm-started TaskTokenResampler head, smooth_L1 loss. Off unless --use-depth.
    parser.add_argument("--use-depth", action="store_true", help="add lingbot future-depth alignment (aux)")
    parser.add_argument("--depth-moge-path", type=Path, default=None, help="MoGe-2 weights (moge-2-vitb-normal/model.pt)")
    parser.add_argument("--depth-morgbd-path", type=Path, default=None, help="LingBot-Depth/MoRGBD weights (6b/depth/model.pt)")
    parser.add_argument("--depth-input-size", type=int, default=256)
    parser.add_argument("--depth-grid-size", type=int, default=16, help="depth teacher token grid side (16 -> 256)")
    parser.add_argument("--depth-dim", type=int, default=1024, help="MoRGBD feature dim")
    # --- teacher swap for the 2B DINOv3+DA3 line (same query/head; different alignment targets) ---
    parser.add_argument("--video-target-type", choices=["dino_video", "dinov3", "siglip2"], default="dino_video",
                        help="video/appearance teacher: lingbot dino_video (default), Meta DINOv3, or SigLIP2")
    parser.add_argument("--dinov3-model-dir", type=Path, default=None,
                        help="DINOv3 HF snapshot dir (default: env DINOV3_MODEL_DIR)")
    parser.add_argument("--siglip2-model-dir", type=Path, default=None,
                        help="SigLIP2 HF snapshot dir (default: env SIGLIP2_MODEL_DIR)")
    parser.add_argument("--siglip2-input-size", type=int, default=384,
                        help="SigLIP2 input res; 384 = native (keeps interpolate_pos_encoding=False)")
    parser.add_argument("--siglip2-grid-size", type=int, default=16,
                        help="pool SigLIP2's 27x27 penultimate grid to this square grid (16 -> 256 tok, DA3-matched)")
    parser.add_argument("--depth-target-type", choices=["morgbd", "da3"], default="morgbd",
                        help="depth teacher: lingbot MoGe->MoRGBD (default) or Depth-Anything-3 encoder")
    parser.add_argument("--da3-ckpt-dir", type=Path, default=None,
                        help="DA3 HF-mixin ckpt dir (default: env DA3_CKPT_DIR)")
    parser.add_argument("--da3-process-res", type=int, default=224,
                        help="DA3 input square res (multiple of 14; 224 -> 16x16=256 tok)")
    parser.add_argument("--da3-align-strategy", choices=["last_layer", "wsa_multilayer"],
                        default="last_layer",
                        help="DA3 depth alignment: last encoder layer only (default) or WSA-style "
                             "multi-layer (per-layer cos+MSE(LayerNorm) over several backbone depths)")
    parser.add_argument("--da3-teacher-layers", type=str, default="11,15,19,23",
                        help="comma-separated absolute backbone layer numbers for wsa_multilayer "
                             "(must be a subset of the DA3 backbone out_layers)")
    parser.add_argument("--da3-layer-weights", type=str, default="1.0,1.2,1.4,1.6",
                        help="comma-separated per-layer loss weights for wsa_multilayer")
    parser.add_argument("--depth-loss-weight", type=float, default=0.004, help="lingbot future_depth_loss_weight")
    parser.add_argument(
        "--use-current-alignment",
        action="store_true",
        help="LingBot-native current+one-future DINO/depth alignment",
    )
    parser.add_argument(
        "--bidirectional-plan-attn",
        action="store_true",
        help="official-parity: run the planner prefix with bidirectional (non-causal) attention (needs retrain)",
    )
    parser.add_argument(
        "--independent-modality-task-tokens",
        action="store_true",
        help=(
            "use four private --num-task-tokens groups for current/future DINO/depth "
            "instead of sharing one group between modalities"
        ),
    )
    parser.add_argument("--num-task-tokens", type=_positive_int, default=8)
    parser.add_argument("--current-dino-loss-weight", type=float, default=0.004)
    parser.add_argument("--future-dino-loss-weight", type=float, default=0.004)
    parser.add_argument("--current-depth-loss-weight", type=float, default=0.004)
    parser.add_argument("--future-depth-loss-weight", type=float, default=0.004)
    parser.add_argument(
        "--head-warmstart-ckpt", type=Path, default=None,
        help="lingbot-vla-v2-6b dir: warm-start the head from future_video_align_head.*",
    )
    # CoVT-style bottleneck: the LM emits only num-latent-per-keyframe continuous latents per
    # keyframe (a compact "visual thought"), and a decoder reconstructs the dense SigLIP grid
    # from them. Decouples the LM sequence length from the target grid resolution.
    parser.add_argument("--num-latent-per-keyframe", type=int, default=4)
    parser.add_argument("--shared-latent-per-keyframe", type=_positive_int, default=32)
    parser.add_argument("--private-latent-per-keyframe", type=_positive_int, default=32)
    parser.add_argument("--plan-head-num-heads", type=int, default=16)
    parser.add_argument("--plan-head-dropout", type=float, default=0.0)
    parser.add_argument("--sem-mlp-hidden-size", type=int, default=0)
    # Loss recipe: plain per-token MSE regresses multimodal futures to their mean (norm
    # shrinkage + spatially blurred, non-discriminative plans). Direction (cosine) and
    # magnitude (relative norm) are supervised separately, InfoNCE supervises token-level
    # discriminativeness (the quantity sample_retrieval_top1 measures), and the dispersion
    # term penalizes predicting the per-sample mean at every token. The pre-fix recipe is
    # `--mse-loss-weight 1 --cosine-loss-weight 0.1` with the other weights at 0.
    parser.add_argument("--mse-loss-weight", type=float, default=1.0)
    parser.add_argument("--cosine-loss-weight", type=float, default=1.0)
    parser.add_argument("--norm-loss-weight", type=float, default=0.2)
    parser.add_argument("--variance-loss-weight", type=float, default=0.1)
    parser.add_argument("--infonce-loss-weight", type=float, default=0.1)
    parser.add_argument("--infonce-temperature", type=float, default=0.07)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--lr-schedule", choices=["cosine", "constant", "none"], default="cosine")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="enable non-reentrant gradient checkpointing (disabled by default)",
    )
    parser.add_argument("--freeze-vision", action="store_true", default=True)
    parser.add_argument("--no-freeze-vision", action="store_false", dest="freeze_vision")
    parser.add_argument("--freeze-lm-head", action="store_true", default=True)
    parser.add_argument("--no-freeze-lm-head", action="store_false", dest="freeze_lm_head")
    parser.add_argument("--train-plan-token-embedding", action="store_true", default=True)
    parser.add_argument("--no-train-plan-token-embedding", action="store_false", dest="train_plan_token_embedding")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--save-steps", type=_positive_int, default=500)
    parser.add_argument(
        "--save-start-step",
        type=_nonnegative_int,
        default=0,
        help="do not write periodic checkpoints before this optimizer step",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--log-steps", type=int, default=10)
    args = parser.parse_args()
    if args.model_path is None and args.init_planner_checkpoint is None:
        parser.error("one of --model-path or --init-planner-checkpoint is required")
    if (
        args.init_planner_checkpoint is not None
        and args.head_warmstart_ckpt is not None
    ):
        parser.error(
            "--init-planner-checkpoint is mutually exclusive with --head-warmstart-ckpt"
        )
    try:
        validate_dataset_source_selection(
            dataset_root=args.dataset_root,
            fastwam_data_config=args.fastwam_data_config,
            ge_act_data_config=args.ge_act_data_config,
        )
    except ValueError as error:
        parser.error(
            str(error)
        )
    if args.fastwam_dataset_dir and args.fastwam_data_config is None:
        parser.error("--fastwam-dataset-dir requires --fastwam-data-config")
    if (
        args.fastwam_text_embedding_cache_dir is not None
        and args.fastwam_data_config is None
    ):
        parser.error(
            "--fastwam-text-embedding-cache-dir requires --fastwam-data-config"
        )
    if (
        args.fastwam_pretrained_norm_stats is not None
        and args.fastwam_data_config is None
    ):
        parser.error(
            "--fastwam-pretrained-norm-stats requires --fastwam-data-config"
        )
    if args.fastwam_data_config is not None:
        if not args.online_plan_labels:
            parser.error("--fastwam-data-config requires --online-plan-labels")
        try:
            offsets = keyframe_offsets(
                args.sequence_length,
                args.num_keyframes,
                args.keyframe_scheme,
                args.keyframe_gamma,
            )
        except ValueError as error:
            parser.error(f"invalid FastWAM keyframe geometry: {error}")
        if args.sequence_length != 9:
            parser.error("FastWAM planner training requires sequence_length=9")
        if args.use_current_alignment:
            # the current-alignment head predicts exactly one future keyframe, at the horizon end
            if args.num_keyframes != 1 or offsets != [8]:
                parser.error(
                    "FastWAM current-alignment training requires num_keyframes=1 "
                    "and keyframe offsets [8]"
                )
        # future-only: any keyframe count the scheme can realize over the 9-frame window is
        # allowed (e.g. uniform K=3 -> [1,4,8], mirroring the WM's keyframe_indices sampling).
    loss_weights = (
        args.current_dino_loss_weight,
        args.future_dino_loss_weight,
        args.current_depth_loss_weight,
        args.future_depth_loss_weight,
    )
    if any(not math.isfinite(weight) or weight < 0 for weight in loss_weights):
        parser.error("current/future alignment loss weights must be finite and non-negative")
    if args.use_current_alignment and not (
        args.plan_head_type == "lingbot_dino" and args.use_depth
    ):
        parser.error("--use-current-alignment requires --plan-head-type lingbot_dino --use-depth")
    if args.independent_modality_task_tokens and not args.use_current_alignment:
        parser.error(
            "--independent-modality-task-tokens requires --use-current-alignment"
        )
    return args


def is_main(rank: int) -> bool:
    return rank == 0


def load_frame(video_path: Path, index: int) -> Image.Image:
    import decord

    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    index = min(max(int(index), 0), len(vr) - 1)
    return Image.fromarray(vr[index].asnumpy()).convert("RGB")


class SemanticPlanDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        label_dir: Path,
        num_keyframes: int,
        grid_size: int,
        max_samples: int = 0,
        sample_one_window_per_stem: bool = False,
        episode_window_sampling: str = "random",
        episode_window_seed: int = 0,
    ) -> None:
        self.dataset_root = dataset_root
        self.label_dir = label_dir
        self.num_keyframes = num_keyframes
        self.grid_size = grid_size
        self.sample_one_window_per_stem = sample_one_window_per_stem
        self.episode_window_sampling = episode_window_sampling
        self.episode_window_seed = int(episode_window_seed)
        self.epoch = 0
        manifest_paths = [label_dir / "manifest.jsonl"] if (label_dir / "manifest.jsonl").exists() else sorted(label_dir.glob("manifest*.jsonl"))
        entries = []
        if manifest_paths:
            for manifest_path in manifest_paths:
                for line in manifest_path.read_text().splitlines():
                    if line.strip():
                        entries.append(json.loads(line))
            entry_paths = [(entry, Path(entry["path"])) for entry in entries]
            entry_paths = [(entry, path) for entry, path in entry_paths if path.exists()]
            paths = [path for _, path in entry_paths]
        else:
            paths = sorted(label_dir.glob("*.pt"))
            entry_paths = []
        paths = [p for p in paths if p.exists()]
        if sample_one_window_per_stem:
            grouped: dict[str, list[Path]] = {}
            if entry_paths:
                for entry, path in entry_paths:
                    stem = str(entry.get("stem") or Path(entry.get("path", path)).stem.split("__r", 1)[0])
                    grouped.setdefault(stem, []).append(path)
            else:
                for path in paths:
                    stem = path.stem.split("__r", 1)[0]
                    grouped.setdefault(stem, []).append(path)
            self.groups = [sorted(grouped[k], key=lambda p: str(p)) for k in sorted(grouped)]
            if max_samples > 0:
                self.groups = self.groups[:max_samples]
            self.paths = []
        else:
            self.paths = paths
            if max_samples > 0:
                self.paths = self.paths[:max_samples]
            self.groups = []
        if sample_one_window_per_stem:
            if not self.groups:
                raise RuntimeError(f"No semantic plan label groups found under {label_dir}")
        elif not self.paths:
            raise RuntimeError(f"No semantic plan labels found under {label_dir}")

    def __len__(self) -> int:
        return len(self.groups) if self.sample_one_window_per_stem else len(self.paths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _choose_window_path(self, index: int) -> Path:
        group = self.groups[index]
        if len(group) == 1:
            return group[0]
        if self.episode_window_sampling == "round_robin":
            return group[self.epoch % len(group)]
        seed = (self.episode_window_seed + 1_000_003 * self.epoch + 9_176 * index) & 0xFFFFFFFF
        rng = random.Random(seed)
        return group[rng.randrange(len(group))]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.sample_one_window_per_stem:
            path = self._choose_window_path(index)
        else:
            path = self.paths[index]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        stem = payload["stem"]
        video_path = Path(payload.get("video_path") or self.dataset_root / "videos" / f"{stem}.mp4")
        prompt = payload.get("prompt") or "A robot manipulates the target object."
        first_frame_index = int(payload.get("first_frame_index", 0))
        image = load_frame(video_path, first_frame_index)
        plan = payload["semantic_plan"].float()
        expected = self.num_keyframes * self.grid_size * self.grid_size
        plan = plan.reshape(-1, plan.shape[-1])
        if plan.shape[0] != expected:
            raise RuntimeError(f"{path} has {plan.shape[0]} plan tokens, expected {expected}")
        return {
            "stem": stem,
            "sample_id": payload.get("sample_id", path.stem),
            "image": image,
            "prompt": prompt,
            "semantic_plan": plan,
            "feature_type": payload.get("feature_type", "unknown"),
        }


def keyframe_offsets(sequence_length: int, n: int, scheme: str, gamma: float) -> list[int]:
    """Window-relative keyframe positions; mirrors the world model's
    semantic_plan_conditioning.keyframe_indices (uniform over [1, T-1] excluding the
    conditioning frame; "late" = end-densified sqrt-warp with strictly increasing fixup)."""
    n = max(min(n, sequence_length), 1)
    if sequence_length <= 1:
        return [0] * n
    if scheme == "even_future":
        future_frames = sequence_length - 1
        if future_frames % n != 0:
            raise ValueError(
                f"even_future requires {future_frames} future frames "
                f"to be divisible by {n} keyframes"
            )
        step = future_frames // n
        return [step * (index + 1) for index in range(n)]
    if scheme == "late" and n > 1:
        u = torch.linspace(0.0, 1.0, n) ** float(gamma)
        idx = (1.0 + (sequence_length - 1 - 1) * u).round().long()
        idx = torch.clamp(idx, 1, sequence_length - 1)
        for i in range(1, n):
            if idx[i] <= idx[i - 1]:
                idx[i] = min(int(idx[i - 1]) + 1, sequence_length - 1)
        return idx.tolist()
    return torch.linspace(1, sequence_length - 1, n).round().long().tolist()


class OnlineSemanticPlanDataset(Dataset):
    """Cosmos-style episode sampling for planner training.

    Each item is a stem; every epoch a fresh stride-1 window of ``sequence_length`` frames
    is drawn from that stem's valid frame ranges (range picked ∝ its number of valid starts,
    start uniform within the range).  Returns the window's first frame (planner input) plus
    the raw keyframe frames; SigLIP2 targets are encoded on-GPU in the training loop with the
    offline builder's exact encode_images, so online targets match precomputed labels."""

    def __init__(
        self,
        dataset_root: Path,
        frame_ranges_json: Path,
        num_keyframes: int,
        sequence_length: int,
        keyframe_scheme: str,
        keyframe_gamma: float,
        max_samples: int = 0,
        seed: int = 0,
    ) -> None:
        from build_siglip2_semantic_plan_labels import load_frame_ranges

        self.dataset_root = dataset_root
        self.sequence_length = int(sequence_length)
        self.offsets = keyframe_offsets(self.sequence_length, num_keyframes, keyframe_scheme, keyframe_gamma)
        self.seed = int(seed)
        self.epoch = 0
        ranges = load_frame_ranges(frame_ranges_json)
        self.items: list[tuple[str, list[tuple[int, int, int]]]] = []
        for stem in sorted(ranges):
            fitting = []
            for range_start, range_end in ranges[stem]:
                n_starts = int(range_end) - int(range_start) - self.sequence_length + 1
                if n_starts > 0:
                    fitting.append((int(range_start), int(range_end), n_starts))
            if fitting:
                self.items.append((stem, fitting))
        if max_samples > 0:
            self.items = self.items[:max_samples]
        if not self.items:
            raise RuntimeError(f"No stems with a fitting {self.sequence_length}-frame window in {frame_ranges_json}")

    def __len__(self) -> int:
        return len(self.items)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np

        from build_siglip2_semantic_plan_labels import load_frames, read_meta

        stem, fitting = self.items[index]
        rng_seed = (self.seed + 1_000_003 * self.epoch + 9_176 * index) & 0xFFFFFFFF
        rng = random.Random(rng_seed)
        total = sum(n for _, _, n in fitting)
        pick = rng.randrange(total)
        for range_start, _range_end, n_starts in fitting:
            if pick < n_starts:
                start = range_start + pick
                break
            pick -= n_starts
        frame_indices = [start] + [start + o for o in self.offsets]
        video_path = self.dataset_root / "videos" / f"{stem}.mp4"
        frames = load_frames(video_path, frame_indices)
        keyframes = np.stack([np.asarray(img, dtype=np.uint8) for img in frames[1:]], axis=0)
        return {
            "stem": stem,
            "sample_id": f"{stem}__s{start:06d}",
            "image": frames[0],
            "prompt": read_meta(self.dataset_root, stem),
            "keyframe_images": torch.from_numpy(keyframes),  # (K, H, W, 3) uint8
            # current frame (frames[0]) as raw uint8 too — the DINO teacher needs it as the clip's
            # current/warmup frame (the PIL `image` above is consumed by the VLM processor).
            "current_image": torch.from_numpy(np.asarray(frames[0], dtype=np.uint8)),  # (H, W, 3) uint8
        }


def _resolve_fastwam_path(path: str | os.PathLike, *, base_dir: Path) -> str:
    expanded = Path(os.fspath(path)).expanduser()
    if not expanded.is_absolute():
        expanded = base_dir / expanded
    return str(expanded.resolve())


def prepare_fastwam_data_config(
    config_path: Path,
    *,
    dataset_dirs: Sequence[str] | None = None,
    text_embedding_cache_dir: Path | None = None,
    pretrained_norm_stats: Path | None = None,
):
    """Load FastWAM data config with explicit, cwd-independent data paths."""
    from omegaconf import OmegaConf

    resolved_config_path = Path(config_path).expanduser().resolve()
    fastwam_root = resolved_config_path.parents[2]
    caller_cwd = Path.cwd()
    data_config = OmegaConf.load(resolved_config_path)
    root_config = OmegaConf.create(
        {
            "data": OmegaConf.to_container(
                data_config,
                resolve=False,
            )
        }
    )
    train_config = root_config.data.train
    if dataset_dirs:
        train_config.dataset_dirs = [
            _resolve_fastwam_path(path, base_dir=caller_cwd)
            for path in dataset_dirs
        ]
    else:
        train_config.dataset_dirs = [
            _resolve_fastwam_path(path, base_dir=fastwam_root)
            for path in train_config.dataset_dirs
        ]
    selected_text_cache = (
        text_embedding_cache_dir
        if text_embedding_cache_dir is not None
        else train_config.get("text_embedding_cache_dir")
    )
    if selected_text_cache is not None:
        train_config.text_embedding_cache_dir = _resolve_fastwam_path(
            selected_text_cache,
            base_dir=(
                caller_cwd
                if text_embedding_cache_dir is not None
                else fastwam_root
            ),
        )
    selected_norm_stats = (
        pretrained_norm_stats
        if pretrained_norm_stats is not None
        else train_config.get("pretrained_norm_stats")
    )
    if selected_norm_stats is not None:
        train_config.pretrained_norm_stats = _resolve_fastwam_path(
            selected_norm_stats,
            base_dir=(
                caller_cwd
                if pretrained_norm_stats is not None
                else fastwam_root
            ),
        )
    return root_config


def preflight_fastwam_data_config(
    config_path: Path,
    *,
    dataset_dirs: Sequence[str] | None = None,
    text_embedding_cache_dir: Path | None = None,
    pretrained_norm_stats: Path | None = None,
):
    """Validate FastWAM assets and import its Hydra target without allocation."""
    from hydra.utils import get_class

    root_config = prepare_fastwam_data_config(
        config_path,
        dataset_dirs=dataset_dirs,
        text_embedding_cache_dir=text_embedding_cache_dir,
        pretrained_norm_stats=pretrained_norm_stats,
    )
    train_config = root_config.data.train
    selected_dataset_dirs = list(train_config.get("dataset_dirs") or [])
    if not selected_dataset_dirs:
        raise ValueError("FastWAM dataset directories are not configured")
    for dataset_dir in selected_dataset_dirs:
        dataset_path = Path(os.fspath(dataset_dir))
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"FastWAM dataset directory does not exist: {dataset_path}"
            )
        if not dataset_path.is_dir():
            raise ValueError(
                f"FastWAM dataset directory is not a directory: {dataset_path}"
            )

    text_cache_dir = train_config.get("text_embedding_cache_dir")
    if text_cache_dir is None:
        raise ValueError("FastWAM text-embedding cache is not configured")
    text_cache_path = Path(os.fspath(text_cache_dir))
    if not text_cache_path.exists():
        raise FileNotFoundError(
            f"FastWAM text-embedding cache does not exist: {text_cache_path}"
        )
    if not text_cache_path.is_dir():
        raise ValueError(
            f"FastWAM text-embedding cache is not a directory: {text_cache_path}"
        )

    norm_stats = train_config.get("pretrained_norm_stats")
    if norm_stats is not None:
        norm_stats_path = Path(os.fspath(norm_stats))
        if not norm_stats_path.exists():
            raise FileNotFoundError(
                "FastWAM pretrained normalization stats do not exist: "
                f"{norm_stats_path}"
            )
        if not norm_stats_path.is_file():
            raise ValueError(
                "FastWAM pretrained normalization stats are not a regular file: "
                f"{norm_stats_path}"
            )

    target = root_config.data.train.get("_target_")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("FastWAM train data config requires a non-empty _target_")
    get_class(target)
    return root_config


class FastWAMOnlinePlannerDataset(Dataset):
    def __init__(
        self,
        dataset,
        max_samples: int = 0,
        offsets: Sequence[int] | None = None,
    ):
        self.dataset = dataset
        self.max_samples = int(max_samples)
        self.offsets = [int(offset) for offset in (offsets or [2, 4, 6, 8])]
        if not self.offsets or any(offset <= 0 for offset in self.offsets):
            raise ValueError(
                f"FastWAM planner offsets must be positive, got {self.offsets}"
            )

    @classmethod
    def from_config(
        cls,
        config_path: Path,
        *,
        dataset_dirs: Sequence[str] | None = None,
        text_embedding_cache_dir: Path | None = None,
        pretrained_norm_stats: Path | None = None,
        max_samples: int = 0,
        offsets: Sequence[int] | None = None,
    ):
        from hydra.utils import instantiate

        root_config = prepare_fastwam_data_config(
            config_path,
            dataset_dirs=dataset_dirs,
            text_embedding_cache_dir=text_embedding_cache_dir,
            pretrained_norm_stats=pretrained_norm_stats,
        )
        dataset = instantiate(root_config.data.train)
        return cls(dataset, max_samples=max_samples, offsets=offsets)

    @classmethod
    def from_dataset(
        cls,
        dataset,
        max_samples: int = 0,
        offsets: Sequence[int] | None = None,
    ):
        return cls(dataset, max_samples=max_samples, offsets=offsets)

    def __len__(self):
        size = len(self.dataset)
        return min(size, self.max_samples) if self.max_samples > 0 else size

    def set_epoch(self, _epoch: int) -> None:
        return None

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        video = sample["video"]
        if video.ndim != 4 or tuple(video.shape[:2]) != (3, 9):
            raise ValueError(
                "FastWAM planner input must be [3, 9, H, W], got "
                f"{tuple(video.shape)}"
            )
        instruction = sample.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(
                "FastWAM planner sample needs a non-empty raw instruction"
            )
        if "video_fps" not in sample:
            raise KeyError(
                "FastWAM planner sample needs sampled video_fps to derive "
                "the DINO future-frame timing"
            )
        video_fps_tensor = torch.as_tensor(sample["video_fps"], dtype=torch.float32)
        if video_fps_tensor.numel() != 1:
            raise ValueError(
                "FastWAM planner video_fps must be scalar, got "
                f"shape={tuple(video_fps_tensor.shape)}"
            )
        video_fps = float(video_fps_tensor.item())
        if not math.isfinite(video_fps) or video_fps <= 0:
            raise ValueError(
                f"FastWAM planner video_fps must be positive and finite, got {video_fps}"
            )
        invalid_offsets = [offset for offset in self.offsets if offset >= video.shape[1]]
        if invalid_offsets:
            raise ValueError(
                "FastWAM planner offsets exceed the video horizon "
                f"T={video.shape[1]}: {invalid_offsets}"
            )
        selected = video[:, [0, *self.offsets]]
        selected = (
            ((selected.to(torch.float32) + 1.0) * 127.5)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 3, 0)
            .contiguous()
        )
        current = selected[0]
        return {
            "stem": f"fastwam_{index:09d}",
            "sample_id": f"fastwam_{index:09d}",
            "image": Image.fromarray(current.numpy()),
            "prompt": instruction,
            "keyframe_images": selected[1:],
            "current_image": current,
            # Each DINO clip represents [current, current, future@offset].  Its temporal
            # RoPE therefore sees the sampled-video rate divided by that future offset.
            "future_video_effective_fps": torch.tensor(
                [video_fps / float(offset) for offset in self.offsets],
                dtype=torch.float32,
            ),
        }


PLANNER_USER_TEMPLATE = (
    "You are a robot video semantic planner. Given the first frame and instruction, "
    "predict future spatial semantic plan tokens for the manipulation video.\n"
    "Instruction: {instruction}"
)


def build_planner_inputs(
    processor: Any,
    images: list[Any],
    instructions: list[Any],
    plan_sequence: str | list[str],
) -> Any:
    if len(images) != len(instructions):
        raise ValueError(
            f"images/instructions batch mismatch: {len(images)} != {len(instructions)}"
        )
    plan_text = plan_sequence if isinstance(plan_sequence, str) else "".join(plan_sequence)
    conversations = []
    for image, instruction in zip(images, instructions, strict=True):
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": PLANNER_USER_TEMPLATE.format(instruction=str(instruction)),
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": plan_text,
                },
            ]
        )
    texts = [
        processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        for conversation in conversations
    ]
    return processor(
        text=texts,
        images=list(images),
        padding=True,
        return_tensors="pt",
    )


@dataclass
class Collator:
    processor: Any
    plan_sequence: list[str]

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [x["image"] for x in batch]
        labels = None
        keyframes = None
        current = None
        future_video_effective_fps = None
        if "semantic_plan" in batch[0]:
            labels = torch.stack([x["semantic_plan"] for x in batch], dim=0)
        else:  # online mode: raw keyframes (+ current frame for the DINO teacher), encoded in the loop
            keyframes = torch.stack([x["keyframe_images"] for x in batch], dim=0)
            if "current_image" in batch[0]:
                current = torch.stack([x["current_image"] for x in batch], dim=0)
            if "future_video_effective_fps" in batch[0]:
                if not all("future_video_effective_fps" in item for item in batch):
                    raise ValueError(
                        "future_video_effective_fps must be present for every item in a batch"
                    )
                future_video_effective_fps = torch.stack(
                    [x["future_video_effective_fps"] for x in batch], dim=0
                )
        inputs = build_planner_inputs(
            self.processor,
            images,
            [item["prompt"] for item in batch],
            self.plan_sequence,
        )
        if labels is not None:
            inputs["semantic_plan_labels"] = labels
        if keyframes is not None:
            inputs["keyframe_images"] = keyframes
        if current is not None:
            inputs["current_image"] = current
        if future_video_effective_fps is not None:
            inputs["future_video_effective_fps"] = future_video_effective_fps
        inputs["stems"] = [x["stem"] for x in batch]
        return inputs


class MLPPlanHead(nn.Module):
    def __init__(self, hidden_size: int, semantic_dim: int, sem_mlp_hidden_size: int) -> None:
        super().__init__()
        if sem_mlp_hidden_size > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, sem_mlp_hidden_size),
                nn.GELU(),
                nn.Linear(sem_mlp_hidden_size, semantic_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, semantic_dim),
            )

    def forward(self, plan_hidden: torch.Tensor) -> torch.Tensor:
        return self.net(plan_hidden)


class BatonCrossAttentionPlanHead(nn.Module):
    """Video-only Baton-style semantic alignment tower.

    Baton uses learnable queries that cross-attend to MLLM planning hidden
    states, then a Sem-MLP projects to the perceptual feature space.  We keep
    the same video tower idea and omit the audio cross-modal tower.
    """

    def __init__(
        self,
        *,
        plan_len: int,
        hidden_size: int,
        semantic_dim: int,
        sem_mlp_hidden_size: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.query = nn.Parameter(torch.empty(plan_len, hidden_size))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.sem_mlp = MLPPlanHead(hidden_size, semantic_dim, sem_mlp_hidden_size)

    def forward(self, plan_hidden: torch.Tensor) -> torch.Tensor:
        batch = plan_hidden.shape[0]
        query = self.query.unsqueeze(0).expand(batch, -1, -1)
        context = self.context_norm(plan_hidden)
        attn_out, _ = self.cross_attn(
            self.query_norm(query),
            context,
            context,
            need_weights=False,
        )
        query = query + attn_out
        query = query + self.ffn(self.ffn_norm(query))
        return self.sem_mlp(query)


class CoVTLatentDecoderHead(nn.Module):
    """CoVT DINO-branch-style latent bottleneck decoder.

    Faithful to the reference DINO head in CoVT (arXiv 2511.19418,
    train/src/training/covt_qwen2_5_vl.py): the VLM emits a small budget of continuous
    thinking tokens (CoVT uses 4 for DINO); each is projected to the target feature dim
    and L2-normalized, then a bank of learnable grid queries (the full target token grid)
    cross-attends to them through a single MultiheadAttention whose output is *directly*
    the reconstructed dense feature map (regressed to the frozen encoder features by MSE).
    No residual / LayerNorm / extra MLP — CoVT keeps this head minimal.

    Reference (per single image):
        dino_projection   = nn.Linear(hidden, 1024)         # LM hidden -> DINO dim
        dino_query_vectors= nn.Parameter(randn(1025, 1024)) # full DINO grid
        dino_cross_attn   = nn.MultiheadAttention(1024, 8)
        kv   = F.normalize(dino_projection(hidden[<dino> positions]))   # [B, 4, 1024]
        pred = dino_cross_attn(query=dino_query_vectors, key=kv, value=kv)  # [B, 1025, 1024]

    Here the target is the per-keyframe SigLIP grid, so the same head is applied per
    keyframe: keyframe kf is reconstructed from *its own* num_latent_per_keyframe thinking
    tokens, with a keyframe embedding added to the shared grid queries.  The LM sequence
    length (num_keyframes*num_latent_per_keyframe) is decoupled from the target grid
    resolution (num_keyframes*grid_size**2).
    """

    def __init__(
        self,
        *,
        num_keyframes: int,
        num_latent_per_keyframe: int,
        grid_size: int,
        hidden_size: int,
        semantic_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if semantic_dim % num_heads != 0:
            raise ValueError(f"semantic_dim={semantic_dim} must be divisible by num_heads={num_heads}")
        if grid_size <= 0:
            raise ValueError(f"covt head needs a positive target grid_size, got {grid_size}")
        self.num_keyframes = int(num_keyframes)
        self.num_latent_per_keyframe = int(num_latent_per_keyframe)
        self.grid_size = int(grid_size)
        self.grid_tokens = self.grid_size * self.grid_size
        # Project LM hidden -> target feature dim, then the queries/attention live in that dim.
        self.latent_projection = nn.Linear(hidden_size, semantic_dim)
        self.grid_query = nn.Parameter(torch.randn(self.grid_tokens, semantic_dim))
        self.keyframe_embed = nn.Parameter(torch.zeros(self.num_keyframes, semantic_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=semantic_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, latent_hidden: torch.Tensor) -> torch.Tensor:
        # latent_hidden: (B, num_keyframes * num_latent_per_keyframe, H)
        batch = latent_hidden.shape[0]
        hidden = latent_hidden.shape[-1]
        expected = self.num_keyframes * self.num_latent_per_keyframe
        if latent_hidden.shape[1] != expected:
            raise RuntimeError(
                f"covt head expected {expected} latent tokens, got {latent_hidden.shape[1]}"
            )
        latents = latent_hidden.reshape(batch * self.num_keyframes, self.num_latent_per_keyframe, hidden)
        # CoVT: project thinking tokens to the target dim and L2-normalize before cross-attn.
        kv = F.normalize(self.latent_projection(latents), dim=-1)
        # Per-keyframe grid queries: (num_keyframes, grid_tokens, D) -> (B*num_keyframes, grid_tokens, D)
        query = self.grid_query.unsqueeze(0) + self.keyframe_embed.unsqueeze(1)
        query = query.unsqueeze(0).expand(batch, -1, -1, -1)
        query = query.reshape(batch * self.num_keyframes, self.grid_tokens, self.grid_query.shape[-1])
        attn_out, _ = self.cross_attn(query.to(kv.dtype), kv, kv, need_weights=False)
        return attn_out.reshape(batch, self.num_keyframes * self.grid_tokens, attn_out.shape[-1])


class PlanTokenEmbeddingInjector(nn.Module):
    """Inject compact learned query embeddings at plan-token positions."""

    def __init__(self, base_embedding: nn.Embedding, plan_token_ids: list[int]) -> None:
        super().__init__()
        if base_embedding.weight.ndim != 2:
            raise ValueError(
                "base embedding must be a matrix, got "
                f"shape={tuple(base_embedding.weight.shape)}"
            )
        plan_ids = [int(token_id) for token_id in plan_token_ids]
        if len(set(plan_ids)) != len(plan_ids):
            raise ValueError("plan_token_ids must be unique")
        num_embeddings = int(base_embedding.weight.shape[0])
        invalid = [token_id for token_id in plan_ids if not 0 <= token_id < num_embeddings]
        if invalid:
            raise ValueError(
                f"plan token ids outside embedding rows {num_embeddings}: {invalid}"
            )

        plan_ids_tensor = torch.tensor(
            plan_ids,
            device=base_embedding.weight.device,
            dtype=torch.long,
        )
        self.weight = nn.Parameter(
            base_embedding.weight.detach().index_select(0, plan_ids_tensor).clone()
        )
        id_to_row = torch.full(
            (num_embeddings,),
            -1,
            device=base_embedding.weight.device,
            dtype=torch.long,
        )
        id_to_row[plan_ids_tensor] = torch.arange(
            len(plan_ids),
            device=id_to_row.device,
            dtype=torch.long,
        )
        self.register_buffer("id_to_row", id_to_row, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        base_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        rows = self.id_to_row[input_ids]
        plan_mask = rows >= 0
        output = base_embeddings.clone()
        output[plan_mask] = self.weight[rows[plan_mask]].to(dtype=output.dtype)
        return output

    def forward_hook(
        self,
        _module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        return self(inputs[0], output)


class PlannerWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        hidden_size: int,
        semantic_dim: int,
        plan_token_ids: list[int],
        target_len: int,
        num_keyframes: int,
        grid_size: int,
        num_latent_per_keyframe: int = 4,
        plan_head_type: str = "mlp",
        plan_head_num_heads: int = 16,
        plan_head_dropout: float = 0.0,
        sem_mlp_hidden_size: int = 0,
        mse_loss_weight: float = 1.0,
        cosine_loss_weight: float = 1.0,
        norm_loss_weight: float = 0.2,
        variance_loss_weight: float = 0.1,
        infonce_loss_weight: float = 0.1,
        infonce_temperature: float = 0.07,
        use_depth: bool = False,
        depth_dim: int = 1024,
        depth_grid_size: int = 16,
        depth_loss_weight: float = 0.004,
        shared_latent_per_keyframe: int = 32,
        private_latent_per_keyframe: int = 32,
        use_current_alignment: bool = False,
        bidirectional_plan_attn: bool = False,
        independent_modality_task_tokens: bool = False,
        num_task_tokens: int = 8,
        current_dino_loss_weight: float = 0.004,
        future_dino_loss_weight: float = 0.004,
        current_depth_loss_weight: float = 0.004,
        future_depth_loss_weight: float = 0.004,
        train_plan_token_embedding: bool = False,
        share_head_query_embeddings: bool | None = None,
        da3_align_strategy: str = "last_layer",
        da3_num_layers: int = 1,
        da3_layer_weights: Sequence[float] | None = None,
        num_camera_views: int = 1,
    ) -> None:
        super().__init__()
        self.num_camera_views = int(num_camera_views)
        if self.num_camera_views not in (1, 2):
            raise ValueError("num_camera_views must be 1 or 2")
        if sem_mlp_hidden_size < 0:
            sem_mlp_hidden_size = hidden_size
        self.mse_loss_weight = float(mse_loss_weight)
        self.cosine_loss_weight = float(cosine_loss_weight)
        self.norm_loss_weight = float(norm_loss_weight)
        self.variance_loss_weight = float(variance_loss_weight)
        self.infonce_loss_weight = float(infonce_loss_weight)
        self.infonce_temperature = float(infonce_temperature)
        self.sem_mlp_hidden_size = int(sem_mlp_hidden_size)
        self.plan_head_type = plan_head_type
        self.plan_head_num_heads = int(plan_head_num_heads)
        self.plan_head_dropout = float(plan_head_dropout)
        self.num_keyframes = int(num_keyframes)
        self.use_current_alignment = bool(use_current_alignment)
        self.bidirectional_plan_attn = bool(bidirectional_plan_attn)
        self.independent_modality_task_tokens = bool(
            independent_modality_task_tokens
        )
        self.num_task_tokens = int(num_task_tokens)
        self.current_dino_loss_weight = float(current_dino_loss_weight)
        self.future_dino_loss_weight = float(future_dino_loss_weight)
        self.current_depth_loss_weight = float(current_depth_loss_weight)
        self.future_depth_loss_weight = float(future_depth_loss_weight)
        self.shared_latent_per_keyframe = int(shared_latent_per_keyframe)
        self.private_latent_per_keyframe = int(private_latent_per_keyframe)
        self.branch_latent_per_keyframe = (
            self.shared_latent_per_keyframe + self.private_latent_per_keyframe
        )
        self.total_unique_latent_per_keyframe = (
            self.shared_latent_per_keyframe + 2 * self.private_latent_per_keyframe
        )
        self.use_depth = bool(use_depth) and plan_head_type == "lingbot_dino"
        # DA3 depth alignment strategy. last_layer: single-layer smooth-L1 on the last encoder layer
        # (unchanged legacy path). wsa_multilayer: predict L stacked layers per token and score each
        # with cos + MSE(LayerNorm), per-layer weighted (WSA-style). The depth heads widen to
        # dim_out = depth_dim * L for both current-alignment and future-only planners.
        self.da3_align_strategy = str(da3_align_strategy)
        if self.da3_align_strategy not in ("last_layer", "wsa_multilayer"):
            raise ValueError(f"unknown da3_align_strategy: {self.da3_align_strategy!r}")
        self.da3_num_layers = int(da3_num_layers)
        self.depth_per_layer_dim = int(depth_dim)  # per-layer target width (2048), for reshaping
        self._da3_head_layer_mult = (
            self.da3_num_layers if self.da3_align_strategy == "wsa_multilayer" else 1
        )
        if self.da3_align_strategy == "wsa_multilayer":
            if not self.use_depth:
                raise ValueError(
                    "da3_align_strategy=wsa_multilayer requires lingbot_dino depth"
                )
            if self.da3_num_layers < 1:
                raise ValueError("da3_num_layers must be positive")
            _lw = list(da3_layer_weights) if da3_layer_weights else [1.0] * self.da3_num_layers
            if len(_lw) != self.da3_num_layers:
                raise ValueError(
                    f"da3_layer_weights ({len(_lw)}) must match da3_num_layers ({self.da3_num_layers})"
                )
            self.register_buffer(
                "da3_layer_weights",
                torch.tensor([float(x) for x in _lw], dtype=torch.float32),
                persistent=False,
            )
        else:
            self.da3_layer_weights = None
        if self.use_current_alignment and not self.use_depth:
            raise ValueError("current alignment requires lingbot_dino with depth")
        if self.use_current_alignment and self.num_keyframes != 1:
            raise ValueError("current alignment predicts exactly one future keyframe")
        if self.independent_modality_task_tokens and not self.use_current_alignment:
            raise ValueError(
                "independent modality task tokens require current alignment"
            )
        if self.use_depth and not self.use_current_alignment and (
            self.shared_latent_per_keyframe <= 0 or self.private_latent_per_keyframe <= 0
        ):
            raise ValueError(
                "DINO+depth mode requires positive shared/private latent counts"
            )
        if getattr(self, "use_current_alignment", False):
            self.branch_latent_per_keyframe = self.num_task_tokens
            self.total_unique_latent_per_keyframe = (
                4 if self.independent_modality_task_tokens else 2
            ) * self.num_task_tokens
            self.num_latent_per_keyframe = self.num_task_tokens
        else:
            self.num_latent_per_keyframe = (
                self.branch_latent_per_keyframe
                if self.use_depth
                else int(num_latent_per_keyframe)
            )
        # target_len = tokens the loss regresses (dense SigLIP grid). latent_len = <|sem_plan|>
        # tokens the LM actually emits. They differ only for the CoVT bottleneck head.
        self.target_len = int(target_len)
        if plan_head_type == "mlp":
            self.latent_len = int(target_len)
            self.plan_head = MLPPlanHead(hidden_size, semantic_dim, sem_mlp_hidden_size)
        elif plan_head_type == "baton_crossattn":
            self.latent_len = int(target_len)
            self.plan_head = BatonCrossAttentionPlanHead(
                plan_len=self.latent_len,
                hidden_size=hidden_size,
                semantic_dim=semantic_dim,
                sem_mlp_hidden_size=sem_mlp_hidden_size,
                num_heads=plan_head_num_heads,
                dropout=plan_head_dropout,
            )
        elif plan_head_type == "covt":
            self.latent_len = int(num_keyframes) * int(num_latent_per_keyframe)
            self.plan_head = CoVTLatentDecoderHead(
                num_keyframes=num_keyframes,
                num_latent_per_keyframe=num_latent_per_keyframe,
                grid_size=grid_size,
                hidden_size=hidden_size,
                semantic_dim=semantic_dim,
                num_heads=plan_head_num_heads,
                dropout=plan_head_dropout,
            )
        elif plan_head_type == "lingbot_dino":
            # lingbot-vla-v2 rich-KV head predicting DINO-video patches: shared TaskTokenResampler
            # (warm-startable from future_video_align_head) run per keyframe. grid_size**2 = the DINO
            # patch-token count per keyframe (16**2 = 256). semantic_dim is the DINO dim (1024).
            self.latent_len = (
                (4 if self.independent_modality_task_tokens else 2)
                * self.num_task_tokens
                if self.use_current_alignment
                else (
                    self.num_keyframes * self.total_unique_latent_per_keyframe
                    if self.use_depth
                    else self.num_keyframes * self.num_latent_per_keyframe
                )
            )
            self.plan_head = LingbotDinoPlanHead(
                num_keyframes=num_keyframes,
                num_latent_per_keyframe=self.num_latent_per_keyframe,
                num_backbone_tokens=int(grid_size) * int(grid_size),
                llm_hidden=hidden_size,
                dim_out=semantic_dim,
            )
        else:
            raise ValueError(f"Unsupported plan_head_type: {plan_head_type}")
        # Auxiliary future-DEPTH alignment head (lingbot-style, only for lingbot_dino): a second shared
        # TaskTokenResampler, warm-started from future_depth_align_head, reading the SAME image+latent
        # context and regressing LingBot-Depth features (MoGe-2 -> MoRGBD) with smooth_L1.
        self.depth_loss_weight = float(depth_loss_weight)
        self.depth_head = None
        self.current_plan_head = None
        self.current_depth_head = None
        _depth_head_dim_out = int(depth_dim) * self._da3_head_layer_mult
        if self.use_depth:
            self.depth_head = LingbotDinoPlanHead(
                num_keyframes=num_keyframes,
                num_latent_per_keyframe=self.branch_latent_per_keyframe,
                num_backbone_tokens=int(depth_grid_size) * int(depth_grid_size),
                llm_hidden=hidden_size,
                dim_out=_depth_head_dim_out,
            )
        if getattr(self, "use_current_alignment", False):
            self.current_plan_head = LingbotDinoPlanHead(
                num_keyframes=1,
                num_latent_per_keyframe=self.num_task_tokens,
                num_backbone_tokens=int(grid_size) * int(grid_size),
                llm_hidden=hidden_size,
                dim_out=semantic_dim,
            )
            self.current_depth_head = LingbotDinoPlanHead(
                num_keyframes=1,
                num_latent_per_keyframe=self.num_task_tokens,
                num_backbone_tokens=int(depth_grid_size) * int(depth_grid_size),
                llm_hidden=hidden_size,
                dim_out=_depth_head_dim_out,
            )
        self.plan_token_ids = [int(x) for x in plan_token_ids]
        self.model = model
        self.plan_embedding_injector = None
        can_pool_head_queries = bool(
            train_plan_token_embedding
            and self.plan_head_type == "lingbot_dino"
            and self.use_current_alignment
            and self.independent_modality_task_tokens
        )
        self.uses_pooled_head_query_embeddings = (
            can_pool_head_queries
            if share_head_query_embeddings is None
            else bool(share_head_query_embeddings)
        )
        if self.uses_pooled_head_query_embeddings and not can_pool_head_queries:
            raise ValueError(
                "shared head query embeddings require train_plan_token_embedding, "
                "lingbot_dino, current alignment, and independent modality task tokens"
            )
        if train_plan_token_embedding:
            base_embedding = model.get_input_embeddings()
            base_embedding.weight.requires_grad_(False)
            if self.uses_pooled_head_query_embeddings:
                plan_ids = torch.tensor(
                    self.plan_token_ids,
                    device=base_embedding.weight.device,
                    dtype=torch.long,
                )
                if len(set(self.plan_token_ids)) != len(self.plan_token_ids):
                    raise ValueError("plan_token_ids must be unique")
                invalid = [
                    token_id
                    for token_id in self.plan_token_ids
                    if not 0 <= token_id < base_embedding.weight.shape[0]
                ]
                if invalid:
                    raise ValueError(
                        "plan token ids outside embedding rows "
                        f"{base_embedding.weight.shape[0]}: {invalid}"
                    )
                id_to_row = torch.full(
                    (base_embedding.weight.shape[0],),
                    -1,
                    device=base_embedding.weight.device,
                    dtype=torch.long,
                )
                id_to_row[plan_ids] = torch.arange(
                    len(self.plan_token_ids),
                    device=id_to_row.device,
                    dtype=torch.long,
                )
                self.register_buffer(
                    "pooled_plan_token_id_to_row",
                    id_to_row,
                    persistent=False,
                )
                # The hook calls back into the four canonical alignment heads, so no fifth
                # independently trainable plan-token embedding table is registered.
                base_embedding.register_forward_hook(
                    self._pooled_head_query_embedding_forward_hook
                )
            else:
                self.plan_embedding_injector = PlanTokenEmbeddingInjector(
                    base_embedding,
                    self.plan_token_ids,
                )
                base_embedding.register_forward_hook(
                    self.plan_embedding_injector.forward_hook
                )
        # image-token id used by the lingbot_dino head to gather the LLM's image-token hiddens
        self.image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)

    def pooled_lingbot_prefix_queries(self) -> torch.Tensor:
        """Pool each canonical 256-query alignment bank into its 64-token LM prefix.

        The order is current-first, matching LingBot-VLA v2:
        current DINO, current depth, future DINO, future depth.
        """
        if not self.uses_pooled_head_query_embeddings:
            raise RuntimeError("pooled head query embeddings are not enabled")
        heads = (
            ("current_dino", self.current_plan_head),
            ("current_depth", self.current_depth_head),
            ("future_dino", self.plan_head),
            ("future_depth", self.depth_head),
        )
        pooled = []
        pool_size = None
        hidden_size = None
        for name, head in heads:
            if head is None or not hasattr(head, "query_embs"):
                raise RuntimeError(f"{name} head has no canonical query_embs bank")
            query_bank = head.query_embs
            if query_bank.ndim != 2:
                raise RuntimeError(
                    f"{name} query bank must be 2-D, got {tuple(query_bank.shape)}"
                )
            if query_bank.shape[0] % self.num_task_tokens != 0:
                raise RuntimeError(
                    f"{name} query count {query_bank.shape[0]} is not divisible by "
                    f"num_task_tokens={self.num_task_tokens}"
                )
            this_pool_size = query_bank.shape[0] // self.num_task_tokens
            if pool_size is None:
                pool_size = this_pool_size
                hidden_size = query_bank.shape[1]
            elif this_pool_size != pool_size or query_bank.shape[1] != hidden_size:
                raise RuntimeError("all four alignment query banks must share one geometry")
            pooled.append(
                query_bank.reshape(
                    self.num_task_tokens,
                    this_pool_size,
                    query_bank.shape[1],
                ).mean(dim=1)
            )
        result = torch.cat(pooled, dim=0)
        if result.shape[0] != len(self.plan_token_ids):
            raise RuntimeError(
                "pooled query count does not match plan token count: "
                f"{result.shape[0]} != {len(self.plan_token_ids)}"
            )
        return result

    def _pooled_head_query_embedding_forward_hook(
        self,
        _module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = inputs[0]
        rows = self.pooled_plan_token_id_to_row[input_ids]
        plan_mask = rows >= 0
        if not bool(plan_mask.any()):
            return output
        pooled = self.pooled_lingbot_prefix_queries().to(
            device=output.device,
            dtype=output.dtype,
        )
        injected = output.clone()
        injected[plan_mask] = pooled[rows[plan_mask]]
        return injected

    @classmethod
    def from_exported_checkpoint(
        cls,
        *,
        model: nn.Module,
        checkpoint_dir: str | Path,
        metadata: dict[str, Any],
    ) -> "PlannerWrapper":
        """Rebuild and freeze a planner from the files written by ``save_checkpoint``."""
        checkpoint_dir = Path(checkpoint_dir)
        num_camera_views = metadata.get("num_camera_views", 1)
        if type(num_camera_views) is not int or num_camera_views not in (1, 2):
            raise ValueError(
                "incompatible planner metadata field num_camera_views: "
                f"expected 1 or 2, got {num_camera_views!r}"
            )
        if num_camera_views == 2:
            validate_dual_camera_export_metadata(metadata)
        text_config = getattr(model.config, "text_config", model.config)
        hidden_size = int(text_config.hidden_size)
        query_embedding_source = metadata.get("query_embedding_source")
        if query_embedding_source not in (
            None,
            "pooled_head_query_banks",
            "independent_plan_token_embedding",
            "base_model_token_embedding",
        ):
            raise ValueError(
                "unsupported query_embedding_source in planner metadata: "
                f"{query_embedding_source!r}"
            )
        wrapper = cls(
            model=model,
            hidden_size=hidden_size,
            semantic_dim=int(metadata["semantic_dim"]),
            plan_token_ids=[int(value) for value in metadata["plan_token_ids"]],
            target_len=int(metadata["target_tokens"]),
            num_keyframes=int(metadata["num_keyframes"]),
            grid_size=int(metadata["grid_size"]),
            num_latent_per_keyframe=int(
                metadata["branch_latent_per_keyframe"]
            ),
            shared_latent_per_keyframe=int(
                metadata["shared_latent_per_keyframe"]
            ),
            private_latent_per_keyframe=int(
                metadata["private_latent_per_keyframe"]
            ),
            plan_head_type=str(metadata["plan_head_type"]),
            plan_head_num_heads=int(metadata["plan_head_num_heads"]),
            plan_head_dropout=float(metadata["plan_head_dropout"]),
            sem_mlp_hidden_size=int(metadata["sem_mlp_hidden_size"]),
            mse_loss_weight=float(metadata["mse_loss_weight"]),
            cosine_loss_weight=float(metadata["cosine_loss_weight"]),
            norm_loss_weight=float(metadata["norm_loss_weight"]),
            variance_loss_weight=float(metadata["variance_loss_weight"]),
            infonce_loss_weight=float(metadata["infonce_loss_weight"]),
            infonce_temperature=float(metadata["infonce_temperature"]),
            use_depth=True,
            depth_dim=int(metadata["depth_feature_dim"]),
            depth_grid_size=int(metadata["depth_grid_size"]),
            depth_loss_weight=float(metadata["depth_loss_weight"]),
            use_current_alignment=bool(metadata.get("use_current_alignment", False)),
            bidirectional_plan_attn=bool(metadata.get("bidirectional_plan_attn", False)),
            independent_modality_task_tokens=bool(
                metadata.get("independent_modality_task_tokens", False)
            ),
            num_task_tokens=int(metadata.get("num_task_tokens", 8)),
            current_dino_loss_weight=float(metadata.get("current_dino_loss_weight", 0.004)),
            future_dino_loss_weight=float(metadata.get("future_dino_loss_weight", 0.004)),
            current_depth_loss_weight=float(metadata.get("current_depth_loss_weight", 0.004)),
            future_depth_loss_weight=float(metadata.get("future_depth_loss_weight", 0.004)),
            train_plan_token_embedding=bool(
                metadata.get("train_plan_token_embedding", False)
            ),
            # Metadata-less exports predate true sharing and must retain their independent
            # plan-token embedding table when loaded.
            share_head_query_embeddings=(
                query_embedding_source == "pooled_head_query_banks"
            ),
            da3_align_strategy=str(metadata.get("da3_align_strategy", "last_layer")),
            da3_num_layers=int(metadata.get("da3_num_layers", 1)),
            da3_layer_weights=metadata.get("da3_layer_weights", None),
            num_camera_views=num_camera_views,
        )
        wrapper.plan_head.load_state_dict(
            torch.load(
                checkpoint_dir / "plan_head.pt",
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        wrapper.depth_head.load_state_dict(
            torch.load(
                checkpoint_dir / "depth_head.pt",
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        if wrapper.use_current_alignment:
            wrapper.current_plan_head.load_state_dict(
                torch.load(
                    checkpoint_dir / "current_plan_head.pt",
                    map_location="cpu",
                    weights_only=True,
                ),
                strict=True,
            )
            wrapper.current_depth_head.load_state_dict(
                torch.load(
                    checkpoint_dir / "current_depth_head.pt",
                    map_location="cpu",
                    weights_only=True,
                ),
                strict=True,
            )
        plan_embedding = torch.load(
            checkpoint_dir / "plan_token_embedding.pt",
            map_location="cpu",
            weights_only=True,
        )
        embedding_weight = model.get_input_embeddings().weight
        expected_embedding_shape = (
            len(wrapper.plan_token_ids),
            embedding_weight.shape[1],
        )
        if tuple(plan_embedding.shape) != expected_embedding_shape:
            raise ValueError(
                "incompatible plan token embedding shape: "
                f"expected {expected_embedding_shape}, "
                f"got {tuple(plan_embedding.shape)}"
            )
        plan_ids = torch.tensor(
            wrapper.plan_token_ids,
            device=embedding_weight.device,
            dtype=torch.long,
        )
        with torch.no_grad():
            if wrapper.uses_pooled_head_query_embeddings:
                # Validation-only in pooled mode: the loaded model already carries the
                # trained plan-token embeddings; nothing is copied from this snapshot here.
                # The re-pool runs through bf16 head-query banks, so upcasting to fp32 cannot
                # recover the ~1e-2 bf16 rounding drift vs the saved snapshot. Compare in fp32
                # with a bf16-appropriate tolerance and WARN (do not hard-fail) so benign drift
                # never blocks eval/viz reload, while a gross mismatch (wrong ckpt) still shows.
                expected_snapshot = wrapper.pooled_lingbot_prefix_queries().detach().float().cpu()
                saved_snapshot = plan_embedding.detach().float().cpu()
                if not torch.allclose(
                    expected_snapshot,
                    saved_snapshot,
                    rtol=2e-2,
                    atol=2e-2,
                ):
                    max_abs = (expected_snapshot - saved_snapshot).abs().max().item()
                    warnings.warn(
                        "plan token embedding snapshot differs from the pooled head query "
                        f"banks (max abs diff {max_abs:.4g}); proceeding since pooled mode "
                        "uses the model's own embeddings (validation-only)."
                    )
            else:
                embedding_weight.index_copy_(
                    0,
                    plan_ids,
                    plan_embedding.to(
                        device=embedding_weight.device,
                        dtype=embedding_weight.dtype,
                    ),
                )
            if wrapper.plan_embedding_injector is not None:
                wrapper.plan_embedding_injector.weight.copy_(
                    plan_embedding.to(
                        device=wrapper.plan_embedding_injector.weight.device,
                        dtype=wrapper.plan_embedding_injector.weight.dtype,
                    )
                )
        wrapper.eval()
        for parameter in wrapper.parameters():
            parameter.requires_grad_(False)
        return wrapper

    def collect_plan_hidden(self, hidden: torch.Tensor, input_ids: torch.Tensor, plan_len: int) -> torch.Tensor:
        ids = torch.as_tensor(self.plan_token_ids, device=input_ids.device)
        # Distinct latent tokens each appear once, in emit order (keyframe-major); the single
        # <|sem_plan|> token (mlp/baton) appears plan_len times. Both gather in sequence order.
        plan_mask = torch.isin(input_ids, ids)
        plan_hidden = []
        for b in range(input_ids.shape[0]):
            h = hidden[b, plan_mask[b]]
            if h.shape[0] != plan_len:
                raise RuntimeError(f"Found {h.shape[0]} plan tokens, expected {plan_len}")
            plan_hidden.append(h)
        return torch.stack(plan_hidden, dim=0)

    def collect_image_hidden(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Gather the LLM's image-token hidden states (rich-KV context for the lingbot_dino head)."""
        if self.image_token_id is None:
            raise RuntimeError("lingbot_dino head needs model.config.image_token_id, which is unset")
        mask = input_ids == int(self.image_token_id)
        counts = mask.sum(dim=1)
        n_img = int(counts[0].item())
        if n_img == 0 or not bool(torch.all(counts == counts[0])):
            raise RuntimeError(
                f"lingbot_dino head needs an equal, nonzero image-token count per batch item, got {counts.tolist()}"
            )
        batch, _, hidden_dim = hidden.shape
        return hidden[mask].reshape(batch, n_img, hidden_dim)

    def collect_image_hidden_by_view(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        num_views: int,
    ) -> torch.Tensor:
        if self.image_token_id is None:
            raise RuntimeError("lingbot_dino head needs model.config.image_token_id, which is unset")
        rows = []
        for batch_index in range(input_ids.shape[0]):
            positions = torch.nonzero(
                input_ids[batch_index] == int(self.image_token_id),
                as_tuple=False,
            ).flatten()
            split_points = torch.nonzero(
                positions[1:] != positions[:-1] + 1,
                as_tuple=False,
            ).flatten() + 1
            spans = torch.tensor_split(positions, split_points.cpu().tolist())
            if len(spans) != num_views or any(span.numel() == 0 for span in spans):
                raise RuntimeError(
                    f"expected {num_views} image-token spans, got {len(spans)}"
                )
            if len({int(span.numel()) for span in spans}) != 1:
                raise RuntimeError("dual-camera image-token spans must have equal length")
            rows.append(
                torch.stack(
                    [hidden[batch_index, span] for span in spans],
                    dim=0,
                )
            )
        return torch.stack(rows, dim=0)

    def _forward_hiddens(self, **inputs: Any) -> tuple[torch.Tensor | None, torch.Tensor]:
        """One VLM forward -> live ``(image_hidden|None, plan_hidden)`` tensors.

        Current alignment follows LingBot-VLA v2 and keeps the image-token gradient.  Callers
        detach the image context only when routing it into a future alignment head.
        """
        ctx = _bidirectional_prefix() if getattr(self, "bidirectional_plan_attn", False) else contextlib.nullcontext()
        with ctx:
            outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[-1]
        input_ids = inputs["input_ids"]
        plan_hidden = self.collect_plan_hidden(hidden, input_ids, self.latent_len)
        image_hidden = None
        if self.plan_head_type == "lingbot_dino":
            num_views = getattr(self, "num_camera_views", 1)
            if num_views == 1:
                image_hidden = self.collect_image_hidden(hidden, input_ids)
            else:
                image_hidden = self.collect_image_hidden_by_view(
                    hidden,
                    input_ids,
                    num_views=num_views,
                )
        return image_hidden, plan_hidden

    @staticmethod
    def image_hidden_for_alignment_branch(
        image_hidden: torch.Tensor,
        branch_name: str,
    ) -> torch.Tensor:
        if branch_name.startswith("current_"):
            return image_hidden
        if branch_name.startswith("future_"):
            return image_hidden.detach()
        raise ValueError(f"unknown current/future alignment branch: {branch_name}")

    def split_lingbot_query_hidden(
        self,
        plan_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, token_count, hidden_size = plan_hidden.shape
        expected = self.num_keyframes * self.total_unique_latent_per_keyframe
        if token_count != expected:
            raise RuntimeError(
                f"expected {expected} shared/private query tokens, got {token_count}"
            )
        grouped = plan_hidden.reshape(
            batch,
            self.num_keyframes,
            self.total_unique_latent_per_keyframe,
            hidden_size,
        )
        shared_end = self.shared_latent_per_keyframe
        dino_end = shared_end + self.private_latent_per_keyframe
        depth_end = dino_end + self.private_latent_per_keyframe
        shared = grouped[:, :, :shared_end]
        dino_private = grouped[:, :, shared_end:dino_end]
        depth_private = grouped[:, :, dino_end:depth_end]
        dino_hidden = torch.cat([shared, dino_private], dim=2)
        depth_hidden = torch.cat([shared, depth_private], dim=2)
        return (
            dino_hidden.reshape(
                batch,
                self.num_keyframes * self.branch_latent_per_keyframe,
                hidden_size,
            ),
            depth_hidden.reshape(
                batch,
                self.num_keyframes * self.branch_latent_per_keyframe,
                hidden_size,
            ),
        )

    def split_current_future_task_hidden(
        self,
        plan_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = 2 * int(self.num_task_tokens)
        if plan_hidden.ndim != 3 or plan_hidden.shape[1] != expected:
            raise RuntimeError(
                f"expected {expected} current/future task tokens, got "
                f"{tuple(plan_hidden.shape)}"
            )
        return (
            plan_hidden[:, : self.num_task_tokens],
            plan_hidden[:, self.num_task_tokens :],
        )

    def split_independent_current_future_task_hidden(
        self,
        plan_hidden: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        expected = 4 * int(self.num_task_tokens)
        if plan_hidden.ndim != 3 or plan_hidden.shape[1] != expected:
            raise RuntimeError(
                f"expected {expected} independent modality task tokens, got "
                f"{tuple(plan_hidden.shape)}"
            )
        width = self.num_task_tokens
        return {
            "current_dino": plan_hidden[:, 0 * width : 1 * width],
            "current_depth": plan_hidden[:, 1 * width : 2 * width],
            "future_dino": plan_hidden[:, 2 * width : 3 * width],
            "future_depth": plan_hidden[:, 3 * width : 4 * width],
        }

    def _reshape_depth_plan(self, x: torch.Tensor) -> torch.Tensor:
        """Depth-head output ``[...,tok,L*D]`` -> ``[...,tok,L,D]`` for WSA."""
        if self.da3_align_strategy != "wsa_multilayer":
            return x
        expected = self.da3_num_layers * self.depth_per_layer_dim
        if x.shape[-1] != expected:
            raise RuntimeError(
                f"WSA depth head width must be {expected}, got {x.shape[-1]}"
            )
        return x.reshape(
            *x.shape[:-1],
            self.da3_num_layers,
            self.depth_per_layer_dim,
        )

    def _reshape_depth_target(self, t: torch.Tensor) -> torch.Tensor:
        """Depth target ``[...,L,tok,D]`` -> ``[...,tok,L,D]`` for WSA."""
        if self.da3_align_strategy != "wsa_multilayer":
            return t
        if t.ndim < 4 or t.shape[-3] != self.da3_num_layers:
            raise RuntimeError(
                "WSA depth target must end in [L,tok,D] with "
                f"L={self.da3_num_layers}, got {tuple(t.shape)}"
            )
        return t.transpose(-3, -2).contiguous()

    def _wsa_layer_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """WSA per-layer depth alignment. pred/target: ``[...,tok,L,D]``.

        Per layer: cos_loss = (1 - <norm(pred), norm(target)>).mean over tokens; mse_loss over
        LayerNorm(pred) vs LayerNorm(target). layer_loss = (cos + mse) * weight; averaged over layers."""
        pred = pred.reshape(-1, *pred.shape[-3:])
        target = target.detach().reshape(-1, *target.shape[-3:])
        dim = pred.shape[-1]
        pred_n = F.normalize(pred, dim=-1)
        target_n = F.normalize(target, dim=-1)
        cos_per_layer = (1.0 - (pred_n * target_n).sum(-1)).mean(dim=(0, 1))  # [L]
        mse_per_layer = (
            F.layer_norm(pred, (dim,)) - F.layer_norm(target, (dim,))
        ).pow(2).mean(dim=(0, 1, 3))  # [L]
        weights = self.da3_layer_weights.to(device=pred.device, dtype=pred.dtype)
        layer_loss = (cos_per_layer + mse_per_layer) * weights  # [L]
        return layer_loss.mean(), cos_per_layer.mean(), mse_per_layer.mean()

    def compute_current_future_losses(
        self,
        plans: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        required = {
            "current_dino",
            "future_dino",
            "current_depth",
            "future_depth",
        }
        missing = sorted(required.difference(plans) | required.difference(targets))
        if missing:
            raise ValueError(f"missing current/future plan loss tensors: {missing}")
        current_dino = F.mse_loss(plans["current_dino"], targets["current_dino"])
        future_dino = F.mse_loss(plans["future_dino"], targets["future_dino"])
        extra_stats: dict[str, torch.Tensor] = {}
        if self.da3_align_strategy == "wsa_multilayer":
            current_depth, cd_cos, cd_mse = self._wsa_layer_loss(
                plans["current_depth"], targets["current_depth"]
            )
            future_depth, fd_cos, fd_mse = self._wsa_layer_loss(
                plans["future_depth"], targets["future_depth"]
            )
            extra_stats = {
                "current_depth_cos": cd_cos.detach(),
                "current_depth_lnmse": cd_mse.detach(),
                "future_depth_cos": fd_cos.detach(),
                "future_depth_lnmse": fd_mse.detach(),
            }
        else:
            current_depth = F.smooth_l1_loss(
                plans["current_depth"], targets["current_depth"]
            )
            future_depth = F.smooth_l1_loss(
                plans["future_depth"], targets["future_depth"]
            )
        weighted = {
            "current_dino_weighted": self.current_dino_loss_weight * current_dino,
            "future_dino_weighted": self.future_dino_loss_weight * future_dino,
            "current_depth_weighted": self.current_depth_loss_weight * current_depth,
            "future_depth_weighted": self.future_depth_loss_weight * future_depth,
        }
        return {
            "loss": sum(weighted.values()),
            "current_dino_mse": current_dino.detach(),
            "future_dino_mse": future_dino.detach(),
            "current_depth_smooth_l1": current_depth.detach(),
            "future_depth_smooth_l1": future_depth.detach(),
            **extra_stats,
            **{name: value.detach() for name, value in weighted.items()},
        }

    def predict_dino_depth_plan(
        self,
        **model_inputs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.plan_head_type != "lingbot_dino":
            raise RuntimeError(
                "DINO+depth prediction requires plan_head_type=lingbot_dino"
            )
        if self.depth_head is None:
            raise RuntimeError(
                "DINO+depth prediction requires a configured depth head"
            )
        if getattr(self, "use_current_alignment", False):
            plans = self.predict_current_future_plans(**model_inputs)
            return plans["future_dino"], plans["future_depth"]
        image_hidden, plan_hidden = self._forward_hiddens(**model_inputs)
        dino_hidden, depth_hidden = self.split_lingbot_query_hidden(plan_hidden)
        dino_dtype = next(self.plan_head.parameters()).dtype
        depth_dtype = next(self.depth_head.parameters()).dtype
        num_views = getattr(self, "num_camera_views", 1)
        if num_views == 1:
            image_hiddens = (image_hidden,)
        else:
            if image_hidden.ndim != 4 or image_hidden.shape[1] != num_views:
                raise RuntimeError(
                    f"expected image hidden [B,{num_views},N,H], got "
                    f"{tuple(image_hidden.shape)}"
                )
            image_hiddens = image_hidden.unbind(dim=1)
        dino_views = []
        depth_views = []
        for view_image_hidden in image_hiddens:
            future_image_hidden = view_image_hidden.detach()
            dino_views.append(
                self.plan_head(
                    future_image_hidden.to(dtype=dino_dtype),
                    dino_hidden.to(dtype=dino_dtype),
                ).float()
            )
            depth_views.append(
                self._reshape_depth_plan(
                    self.depth_head(
                        future_image_hidden.to(dtype=depth_dtype),
                        depth_hidden.to(dtype=depth_dtype),
                    ).float()
                )
            )
        dino_plan = dino_views[0] if num_views == 1 else torch.stack(dino_views, dim=1)
        depth_plan = depth_views[0] if num_views == 1 else torch.stack(depth_views, dim=1)
        depth_geometry = (
            depth_plan.shape[:-2]
            if self.da3_align_strategy == "wsa_multilayer"
            else depth_plan.shape[:-1]
        )
        if dino_plan.shape[:-1] != depth_geometry:
            raise RuntimeError(
                "DINO/depth head batch, view, or token mismatch: "
                f"{tuple(dino_plan.shape)} vs {tuple(depth_plan.shape)}"
            )
        return dino_plan, depth_plan

    def predict_current_future_plans(
        self,
        **model_inputs: Any,
    ) -> dict[str, torch.Tensor]:
        if not self.use_current_alignment:
            raise RuntimeError("current/future prediction requires current alignment mode")
        if self.current_plan_head is None or self.current_depth_head is None:
            raise RuntimeError("current alignment heads are not configured")
        image_hidden, task_hidden = self._forward_hiddens(**model_inputs)
        if self.independent_modality_task_tokens:
            hidden_by_branch = self.split_independent_current_future_task_hidden(
                task_hidden
            )
        else:
            current_hidden, future_hidden = self.split_current_future_task_hidden(
                task_hidden
            )
            hidden_by_branch = {
                "current_dino": current_hidden,
                "future_dino": future_hidden,
                "current_depth": current_hidden,
                "future_depth": future_hidden,
            }
        heads_and_hiddens = {
            "current_dino": (self.current_plan_head, hidden_by_branch["current_dino"]),
            "future_dino": (self.plan_head, hidden_by_branch["future_dino"]),
            "current_depth": (self.current_depth_head, hidden_by_branch["current_depth"]),
            "future_depth": (self.depth_head, hidden_by_branch["future_depth"]),
        }
        plans = {}
        num_views = getattr(self, "num_camera_views", 1)
        for name, (head, hidden) in heads_and_hiddens.items():
            head_dtype = next(head.parameters()).dtype
            head_hidden = hidden.to(dtype=head_dtype)
            if num_views == 1:
                image_hiddens = (image_hidden,)
            else:
                if image_hidden.ndim != 4 or image_hidden.shape[1] != num_views:
                    raise RuntimeError(
                        f"expected image hidden [B,{num_views},N,H], got "
                        f"{tuple(image_hidden.shape)}"
                    )
                image_hiddens = image_hidden.unbind(dim=1)
            view_plans = []
            for view_image_hidden in image_hiddens:
                branch_image_hidden = self.image_hidden_for_alignment_branch(
                    view_image_hidden,
                    name,
                )
                plan = head(
                    branch_image_hidden.to(dtype=head_dtype),
                    head_hidden,
                ).float()
                if name in ("current_depth", "future_depth"):
                    plan = self._reshape_depth_plan(plan)  # [B,tok,L*D] -> [B,tok,L,D] for wsa_multilayer
                view_plans.append(plan)
            plans[name] = (
                view_plans[0]
                if num_views == 1
                else torch.stack(view_plans, dim=1)
            )
        return plans

    def predict_semantic_plan(self, **inputs: Any) -> torch.Tensor:
        if self.plan_head_type == "lingbot_dino" and getattr(
            self, "use_current_alignment", False
        ):
            return self.predict_current_future_plans(**inputs)["future_dino"]
        if self.plan_head_type == "lingbot_dino" and self.depth_head is not None:
            return self.predict_dino_depth_plan(**inputs)[0]
        image_hidden, plan_hidden = self._forward_hiddens(**inputs)
        head_dtype = next(self.plan_head.parameters()).dtype
        if self.plan_head_type == "lingbot_dino":
            num_views = getattr(self, "num_camera_views", 1)
            image_hiddens = (
                (image_hidden,)
                if num_views == 1
                else image_hidden.unbind(dim=1)
            )
            plans = [
                self.plan_head(
                    view_hidden.detach().to(dtype=head_dtype),
                    plan_hidden.to(dtype=head_dtype),
                ).float()
                for view_hidden in image_hiddens
            ]
            return plans[0] if num_views == 1 else torch.stack(plans, dim=1)
        return self.plan_head(plan_hidden.to(dtype=head_dtype)).float()

    def compute_plan_losses(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        eps = 1e-6
        mse = F.mse_loss(pred, target)
        pred = pred.reshape(-1, *pred.shape[-2:])
        target = target.reshape(-1, *target.shape[-2:])
        cosine = 1.0 - F.cosine_similarity(pred.flatten(0, 1), target.flatten(0, 1), dim=-1).mean()

        pred_norm_B_L = pred.norm(dim=-1)
        target_norm_B_L = target.norm(dim=-1)
        norm_loss = ((pred_norm_B_L - target_norm_B_L) / (target_norm_B_L + eps)).pow(2).mean()

        # Per-sample dispersion of tokens around their mean: collapses to 0 when the head
        # predicts the (conditional) mean at every token.
        pred_disp_B = (pred - pred.mean(dim=1, keepdim=True)).norm(dim=-1).mean(dim=1)
        target_disp_B = (target - target.mean(dim=1, keepdim=True)).norm(dim=-1).mean(dim=1)
        variance_loss = ((pred_disp_B - target_disp_B) / (target_disp_B + eps)).pow(2).mean()

        infonce = pred.new_zeros(())
        if self.infonce_loss_weight > 0:
            pred_dir = F.normalize(pred, dim=-1)
            target_dir = F.normalize(target, dim=-1)
            logits_B_L_L = torch.bmm(pred_dir, target_dir.transpose(1, 2)) / self.infonce_temperature
            labels_L = torch.arange(pred.shape[1], device=pred.device)
            labels = labels_L.unsqueeze(0).expand(pred.shape[0], -1).reshape(-1)
            infonce = 0.5 * (
                F.cross_entropy(logits_B_L_L.flatten(0, 1), labels)
                + F.cross_entropy(logits_B_L_L.transpose(1, 2).flatten(0, 1), labels)
            )

        loss = (
            self.mse_loss_weight * mse
            + self.cosine_loss_weight * cosine
            + self.norm_loss_weight * norm_loss
            + self.variance_loss_weight * variance_loss
            + self.infonce_loss_weight * infonce
        )

        with torch.no_grad():
            pred_dir = F.normalize(pred, dim=-1)
            target_dir = F.normalize(target, dim=-1)
            sims_B_L_L = torch.bmm(pred_dir, target_dir.transpose(1, 2))
            hits = sims_B_L_L.argmax(dim=-1) == torch.arange(pred.shape[1], device=pred.device)
            token_retrieval = hits.float().mean()

        return {
            "loss": loss,
            "mse": mse.detach(),
            "cosine_loss": cosine.detach(),
            "norm_loss": norm_loss.detach(),
            "variance_loss": variance_loss.detach(),
            "infonce_loss": infonce.detach(),
            "pred_norm": pred_norm_B_L.mean().detach(),
            "target_norm": target_norm_B_L.mean().detach(),
            "norm_ratio": (pred_norm_B_L.mean() / target_norm_B_L.mean().clamp_min(eps)).detach(),
            "token_disp_ratio": (pred_disp_B.mean() / target_disp_B.mean().clamp_min(eps)).detach(),
            "token_retrieval_top1": token_retrieval,
        }

    def forward(
        self,
        semantic_plan_labels: torch.Tensor,
        depth_plan_labels: torch.Tensor | None = None,
        current_dino_labels: torch.Tensor | None = None,
        current_depth_labels: torch.Tensor | None = None,
        **inputs: Any,
    ) -> dict[str, torch.Tensor]:
        batch = semantic_plan_labels.shape[0]
        target_len = semantic_plan_labels.shape[-2]
        if target_len != self.target_len:
            raise RuntimeError(f"Batch has {target_len} target tokens, wrapper expects {self.target_len}")
        if self.plan_head_type == "lingbot_dino":
            if getattr(self, "use_current_alignment", False):
                if current_dino_labels is None or current_depth_labels is None:
                    raise ValueError(
                        "current alignment requires current DINO and depth labels"
                    )
                if depth_plan_labels is None:
                    raise ValueError("current alignment requires future depth labels")
                plans = self.predict_current_future_plans(**inputs)
                targets = {
                    "current_dino": current_dino_labels.to(
                        device=plans["current_dino"].device, dtype=torch.float32
                    ),
                    "future_dino": semantic_plan_labels.to(
                        device=plans["future_dino"].device, dtype=torch.float32
                    ),
                    "current_depth": self._reshape_depth_target(
                        current_depth_labels.to(
                            device=plans["current_depth"].device, dtype=torch.float32
                        )
                    ),
                    "future_depth": self._reshape_depth_target(
                        depth_plan_labels.to(
                            device=plans["future_depth"].device, dtype=torch.float32
                        )
                    ),
                }
                out = self.compute_current_future_losses(plans, targets)
                with torch.no_grad():
                    for name in (
                        "current_dino",
                        "future_dino",
                        "current_depth",
                        "future_depth",
                    ):
                        out[f"{name}_norm_ratio"] = (
                            plans[name].norm(dim=-1).mean()
                            / targets[name].norm(dim=-1).mean().clamp_min(1e-6)
                        ).detach()
                return out
            if self.depth_head is not None:
                pred, depth_pred = self.predict_dino_depth_plan(**inputs)
            else:
                pred = self.predict_semantic_plan(**inputs)
                depth_pred = None
            if pred.shape[0] != batch or pred.shape[-2] != self.target_len:
                raise RuntimeError(
                    f"Prediction batch/tokens {pred.shape[0]}/{pred.shape[-2]} != "
                    f"{batch}/{self.target_len}"
                )
            out = self.compute_plan_losses(pred, semantic_plan_labels.to(device=pred.device, dtype=torch.float32))
            if depth_pred is not None and depth_plan_labels is not None:
                depth_target = self._reshape_depth_target(
                    depth_plan_labels.to(device=pred.device, dtype=torch.float32)
                )
                if self.da3_align_strategy == "wsa_multilayer":
                    depth_l, depth_cos, depth_mse = self._wsa_layer_loss(
                        depth_pred,
                        depth_target,
                    )
                    out["depth_cos"] = depth_cos.detach()
                    out["depth_lnmse"] = depth_mse.detach()
                else:
                    depth_l = F.smooth_l1_loss(depth_pred, depth_target)
                out["loss"] = out["loss"] + self.depth_loss_weight * depth_l
                out["depth_smooth_l1"] = depth_l.detach()
                out["depth_norm_ratio"] = (
                    depth_pred.norm(dim=-1).mean() / depth_target.norm(dim=-1).mean().clamp_min(1e-6)
                ).detach()
            return out
        pred = self.predict_semantic_plan(**inputs)
        if pred.shape[0] != batch:
            raise RuntimeError(f"Prediction batch {pred.shape[0]} does not match labels batch {batch}")
        if pred.shape[1] != self.target_len:
            raise RuntimeError(f"Prediction has {pred.shape[1]} tokens, expected target_len {self.target_len}")
        target = semantic_plan_labels.to(device=pred.device, dtype=torch.float32)
        return self.compute_plan_losses(pred, target)


def apply_lora(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    if args.lora_r <= 0:
        return model
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def set_trainable(model: nn.Module, *, freeze_vision: bool, freeze_lm_head: bool, full_finetune: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(full_finetune)
    if freeze_vision and hasattr(model, "visual"):
        for p in model.visual.parameters():
            p.requires_grad_(False)
    if freeze_lm_head and hasattr(model, "lm_head"):
        for p in model.lm_head.parameters():
            p.requires_grad_(False)


def build_optimizer(wrapper: PlannerWrapper, args: argparse.Namespace) -> torch.optim.Optimizer:
    head_names = (
        "plan_head",
        "depth_head",
        "current_plan_head",
        "current_depth_head",
    )
    head_params = []
    for name in head_names:
        head = getattr(wrapper, name, None)
        if head is not None:
            head_params.extend(p for p in head.parameters() if p.requires_grad)
    other_params = [
        p
        for n, p in wrapper.named_parameters()
        if p.requires_grad
        and not any(n.startswith(f"{name}.") for name in head_names)
    ]
    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": args.head_lr, "weight_decay": args.weight_decay})
    return torch.optim.AdamW(groups)


def build_scheduler(
    optimizer: torch.optim.Optimizer, args: argparse.Namespace
) -> torch.optim.lr_scheduler.LambdaLR | None:
    """Linear warmup to the base LR, then cosine decay to ``min_lr_ratio`` of it.

    ``--warmup-steps`` was previously parsed but never consumed, leaving the run
    at a constant LR with no warmup. The multiplier is shared across param groups
    so the backbone (--lr) and plan head (--head-lr) decay proportionally.
    """
    if args.lr_schedule == "none":
        return None
    warmup = max(0, int(args.warmup_steps))
    total = max(1, int(args.max_steps))
    min_ratio = float(args.min_lr_ratio)

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if args.lr_schedule == "constant" or total <= warmup:
            return 1.0
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def count_trainable_parameters(module: nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in module.parameters():
        numel = param.numel()
        total += numel
        if param.requires_grad:
            trainable += numel
    return trainable, total


def validate_fastwam_export_contract(
    module: PlannerWrapper,
    args: argparse.Namespace,
) -> list[int]:
    """Reject any checkpoint geometry that the production FastWAM provider cannot load."""
    try:
        offsets = keyframe_offsets(
            sequence_length=int(args.sequence_length),
            n=int(args.num_keyframes),
            scheme=str(args.keyframe_scheme),
            gamma=float(args.keyframe_gamma),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            f"FastWAM planner export contract has invalid keyframe geometry: {error}"
        ) from error

    plan_token_ids = getattr(module, "plan_token_ids", ())
    try:
        plan_token_count = len(plan_token_ids)
        unique_plan_token_count = len({int(token_id) for token_id in plan_token_ids})
    except (TypeError, ValueError):
        plan_token_count = -1
        unique_plan_token_count = -1

    actual = {
        "use_depth": bool(getattr(args, "use_depth", False)),
        "sequence_length": getattr(args, "sequence_length", None),
        "num_keyframes": getattr(args, "num_keyframes", None),
        "wrapper_num_keyframes": getattr(module, "num_keyframes", None),
        "keyframe_scheme": str(getattr(args, "keyframe_scheme", "")),
        "keyframe_offsets": [int(offset) for offset in offsets],
        "grid_size": getattr(args, "grid_size", None),
        "semantic_dim": getattr(args, "semantic_dim", None),
        "depth_grid_size": getattr(args, "depth_grid_size", None),
        "depth_dim": getattr(args, "depth_dim", None),
        "target_len": getattr(module, "target_len", None),
        "shared_latent_per_keyframe": getattr(
            module, "shared_latent_per_keyframe", None
        ),
        "private_latent_per_keyframe": getattr(
            module, "private_latent_per_keyframe", None
        ),
        "branch_latent_per_keyframe": getattr(
            module, "branch_latent_per_keyframe", None
        ),
        "total_unique_latent_per_keyframe": getattr(
            module, "total_unique_latent_per_keyframe", None
        ),
        "num_latent_per_keyframe": getattr(
            module, "num_latent_per_keyframe", None
        ),
        "latent_len": getattr(module, "latent_len", None),
        "plan_token_count": plan_token_count,
        "unique_plan_token_count": unique_plan_token_count,
        "plan_head_type": getattr(module, "plan_head_type", None),
        "has_depth_head": getattr(module, "depth_head", None) is not None,
        "use_current_alignment": bool(
            getattr(module, "use_current_alignment", False)
        ),
        "bidirectional_plan_attn": bool(
            getattr(module, "bidirectional_plan_attn", False)
        ),
        "num_task_tokens": getattr(module, "num_task_tokens", 8),
        "has_current_plan_head": getattr(module, "current_plan_head", None)
        is not None,
        "has_current_depth_head": getattr(module, "current_depth_head", None)
        is not None,
        "independent_modality_task_tokens": bool(
            getattr(module, "independent_modality_task_tokens", False)
        ),
    }
    # Future-only geometry is derived from the configured keyframe count rather than pinned to
    # the K=4/even_future preset, so alternative plans (e.g. SigLIP2 uniform K=3 -> [1,4,8]) can
    # export. The module-vs-config checks below stay strict: the wrapper's K, the head's
    # target_len (K*grid^2) and latent_len (K*96) must still match what was configured.
    _lk = int(getattr(args, "num_keyframes", 4) or 4)
    _lg = int(getattr(args, "grid_size", 16) or 16)
    _l_unique_per_kf = 96
    legacy_expected = {
        "use_depth": True,
        "sequence_length": 9,
        "num_keyframes": _lk,
        "wrapper_num_keyframes": _lk,
        "keyframe_scheme": str(getattr(args, "keyframe_scheme", "even_future")),
        "keyframe_offsets": [int(offset) for offset in offsets],
        "grid_size": 16,
        "semantic_dim": 1024,
        "depth_grid_size": 16,
        "depth_dim": 1024,
        "target_len": _lk * _lg * _lg,
        "shared_latent_per_keyframe": 32,
        "private_latent_per_keyframe": 32,
        "branch_latent_per_keyframe": 64,
        "total_unique_latent_per_keyframe": _l_unique_per_kf,
        "num_latent_per_keyframe": 64,
        "latent_len": _lk * _l_unique_per_kf,
        "plan_token_count": _lk * _l_unique_per_kf,
        "unique_plan_token_count": _lk * _l_unique_per_kf,
        "plan_head_type": "lingbot_dino",
        "has_depth_head": True,
        "use_current_alignment": False,
        "num_task_tokens": 8,
        "has_current_plan_head": False,
        "has_current_depth_head": False,
        "independent_modality_task_tokens": False,
    }
    configured_task_tokens = actual["num_task_tokens"]
    expected_task_tokens = (
        configured_task_tokens
        if type(configured_task_tokens) is int and configured_task_tokens > 0
        else 8
    )
    current_expected = {
        "use_depth": True,
        "sequence_length": 9,
        "num_keyframes": 1,
        "wrapper_num_keyframes": 1,
        "keyframe_scheme": "even_future",
        "keyframe_offsets": [8],
        "grid_size": 16,
        "semantic_dim": 1024,
        "depth_grid_size": 16,
        "depth_dim": 1024,
        "target_len": 256,
        "shared_latent_per_keyframe": 32,
        "private_latent_per_keyframe": 32,
        "branch_latent_per_keyframe": expected_task_tokens,
        "total_unique_latent_per_keyframe": 2 * expected_task_tokens,
        "num_latent_per_keyframe": expected_task_tokens,
        "latent_len": 2 * expected_task_tokens,
        "plan_token_count": 2 * expected_task_tokens,
        "unique_plan_token_count": 2 * expected_task_tokens,
        "plan_head_type": "lingbot_dino",
        "has_depth_head": True,
        "use_current_alignment": True,
        "num_task_tokens": expected_task_tokens,
        "has_current_plan_head": True,
        "has_current_depth_head": True,
        "independent_modality_task_tokens": False,
    }
    independent_current_expected = {
        **current_expected,
        "total_unique_latent_per_keyframe": 4 * expected_task_tokens,
        "latent_len": 4 * expected_task_tokens,
        "plan_token_count": 4 * expected_task_tokens,
        "unique_plan_token_count": 4 * expected_task_tokens,
        "independent_modality_task_tokens": True,
    }
    if actual["use_current_alignment"]:
        expected = (
            independent_current_expected
            if actual["independent_modality_task_tokens"]
            else current_expected
        )
    else:
        expected = legacy_expected
    # The 2B DINOv3/DA3 line keeps the lingbot query/latent/keyframe contract but legitimately
    # changes the teacher feature dims (DINOv3 1280 / DA3 2048 vs the 1024 DINO-video); relax only those.
    if getattr(args, "video_target_type", "dino_video") != "dino_video":
        for _k in ("semantic_dim", "depth_dim"):
            if _k in expected:
                expected[_k] = actual[_k]
    mismatches = [
        f"{name}={actual[name]!r} (expected {expected_value!r})"
        for name, expected_value in expected.items()
        if actual[name] != expected_value
    ]
    if mismatches:
        raise ValueError(
            "FastWAM planner export contract mismatch: " + "; ".join(mismatches)
        )
    return offsets


def save_checkpoint(
    output_dir: Path,
    step: int,
    wrapper: PlannerWrapper,
    processor: Any,
    args: argparse.Namespace,
    rank: int,
) -> None:
    if not is_main(rank):
        return
    module = wrapper
    num_camera_views = int(getattr(module, "num_camera_views", 1))
    fastwam_data_config = getattr(args, "fastwam_data_config", None)
    fastwam_offsets = None
    if fastwam_data_config is not None:
        fastwam_offsets = validate_fastwam_export_contract(module, args)
    depth_head = getattr(module, "depth_head", None)
    current_plan_head = getattr(module, "current_plan_head", None)
    current_depth_head = getattr(module, "current_depth_head", None)
    ckpt = output_dir / f"step_{step:06d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    module.model.save_pretrained(ckpt / "qwen3vl_lora_or_model")
    processor.save_pretrained(ckpt / "processor")
    torch.save(module.plan_head.state_dict(), ckpt / "plan_head.pt")
    if depth_head is not None:
        torch.save(depth_head.state_dict(), ckpt / "depth_head.pt")
    if current_plan_head is not None:
        torch.save(current_plan_head.state_dict(), ckpt / "current_plan_head.pt")
    if current_depth_head is not None:
        torch.save(current_depth_head.state_dict(), ckpt / "current_depth_head.pt")
    plan_embedding_injector = getattr(module, "plan_embedding_injector", None)
    uses_pooled_head_queries = bool(
        getattr(module, "uses_pooled_head_query_embeddings", False)
    )
    if uses_pooled_head_queries:
        plan_embedding = module.pooled_lingbot_prefix_queries().detach().cpu()
    elif plan_embedding_injector is not None:
        plan_embedding = plan_embedding_injector.weight.detach().cpu()
    else:
        plan_ids = torch.as_tensor(module.plan_token_ids)
        plan_embedding = (
            module.model.get_input_embeddings().weight[plan_ids].detach().cpu()
        )
    torch.save(plan_embedding, ckpt / "plan_token_embedding.pt")
    feature_type = str(getattr(args, "sample_feature_type", "unknown"))
    summary_path = args.plan_label_dir / "summary.json" if args.plan_label_dir else None
    if summary_path is not None and summary_path.exists():
        try:
            feature_type = json.loads(summary_path.read_text()).get("feature_type", feature_type)
        except Exception:
            feature_type = feature_type
    meta = {
        "step": step,
        "plan_token": PLAN_TOKEN,
        "plan_token_ids": module.plan_token_ids,
        "num_keyframes": args.num_keyframes,
        "grid_size": args.grid_size,
        "semantic_dim": args.semantic_dim,
        "model_path": str(args.model_path),
        "objective": "continuous_semantic_blueprint_regression",
        "feature_type": feature_type,
        "plan_label_dir": str(args.plan_label_dir) if args.plan_label_dir else None,
        "sample_one_window_per_stem": bool(args.sample_one_window_per_stem),
        "online_plan_labels": bool(args.online_plan_labels),
        "keyframe_scheme": str(args.keyframe_scheme),
        "keyframe_gamma": float(args.keyframe_gamma),
        "sequence_length": int(args.sequence_length),
        "online_grid_size": int(args.online_grid_size),
        "siglip2_encoder_path": str(args.siglip2_encoder_path) if args.siglip2_encoder_path else None,
        "frame_ranges_json": str(args.frame_ranges_json) if args.frame_ranges_json else None,
        "plan_head_type": module.plan_head_type,
        "plan_head_num_heads": int(module.plan_head_num_heads),
        "plan_head_dropout": float(module.plan_head_dropout),
        "num_latent_per_keyframe": int(module.num_latent_per_keyframe),
        "latent_len": int(module.latent_len),
        "target_len": int(module.target_len),
        "sem_mlp_hidden_size": int(module.sem_mlp_hidden_size),
        "mse_loss_weight": float(module.mse_loss_weight),
        "cosine_loss_weight": float(module.cosine_loss_weight),
        "norm_loss_weight": float(module.norm_loss_weight),
        "variance_loss_weight": float(module.variance_loss_weight),
        "infonce_loss_weight": float(module.infonce_loss_weight),
        "infonce_temperature": float(module.infonce_temperature),
        "train_plan_token_embedding": bool(args.train_plan_token_embedding),
        "query_embedding_source": (
            "pooled_head_query_banks"
            if uses_pooled_head_queries
            else "independent_plan_token_embedding"
            if plan_embedding_injector is not None
            else "base_model_token_embedding"
        ),
        "query_pool_size": (
            int(module.plan_head.query_embs.shape[0] // module.num_task_tokens)
            if uses_pooled_head_queries
            else None
        ),
        "full_finetune": bool(args.full_finetune),
        "freeze_vision": bool(args.freeze_vision),
        "freeze_lm_head": bool(args.freeze_lm_head),
    }
    offsets = fastwam_offsets or keyframe_offsets(
        sequence_length=int(args.sequence_length),
        n=int(args.num_keyframes),
        scheme=str(args.keyframe_scheme),
        gamma=float(args.keyframe_gamma),
    )
    meta.update(
        {
            "sequence_length": int(args.sequence_length),
            "num_keyframes": int(args.num_keyframes),
            "grid_size": int(args.grid_size),
            "semantic_dim": int(args.semantic_dim),
            "video_target_type": str(getattr(args, "video_target_type", "dino_video")),
            "depth_target_type": str(getattr(args, "depth_target_type", "morgbd")),
            "target_tokens": int(module.target_len),
            "keyframe_offsets": [int(offset) for offset in offsets],
            "normalized_keyframe_times": [
                float(offset) / float(args.sequence_length - 1)
                for offset in offsets
            ],
            "has_depth_head": depth_head is not None,
            "depth_feature_dim": (
                int(args.depth_dim) if depth_head is not None else None
            ),
            "da3_align_strategy": str(getattr(module, "da3_align_strategy", "last_layer")),
            "da3_num_layers": int(getattr(module, "da3_num_layers", 1)),
            "da3_teacher_layers": (
                list(getattr(args, "da3_teacher_layers_resolved", []))
                if getattr(module, "da3_align_strategy", "last_layer") == "wsa_multilayer"
                else None
            ),
            "da3_layer_weights": (
                [float(x) for x in module.da3_layer_weights.tolist()]
                if getattr(module, "da3_layer_weights", None) is not None
                else None
            ),
            "depth_grid_size": int(args.depth_grid_size),
            "depth_loss_weight": float(module.depth_loss_weight),
            "use_current_alignment": bool(
                getattr(module, "use_current_alignment", False)
            ),
            "bidirectional_plan_attn": bool(
                getattr(module, "bidirectional_plan_attn", False)
            ),
            "num_task_tokens": int(getattr(module, "num_task_tokens", 8)),
            "independent_modality_task_tokens": bool(
                getattr(module, "independent_modality_task_tokens", False)
            ),
            "current_dino_loss_weight": float(
                getattr(module, "current_dino_loss_weight", 0.004)
            ),
            "future_dino_loss_weight": float(
                getattr(module, "future_dino_loss_weight", 0.004)
            ),
            "current_depth_loss_weight": float(
                getattr(module, "current_depth_loss_weight", 0.004)
            ),
            "future_depth_loss_weight": float(
                getattr(module, "future_depth_loss_weight", 0.004)
            ),
            "shared_latent_per_keyframe": int(
                module.shared_latent_per_keyframe
            ),
            "private_latent_per_keyframe": int(
                module.private_latent_per_keyframe
            ),
            "branch_latent_per_keyframe": int(
                module.branch_latent_per_keyframe
            ),
            "total_unique_latent_per_keyframe": int(
                module.total_unique_latent_per_keyframe
            ),
            "query_layout": (
                (
                    f"current_dino_{module.num_task_tokens}_then_"
                    f"current_depth_{module.num_task_tokens}_then_"
                    f"future_dino_{module.num_task_tokens}_then_"
                    f"future_depth_{module.num_task_tokens}"
                )
                if getattr(module, "independent_modality_task_tokens", False)
                else (
                    f"current_{module.num_task_tokens}_then_"
                    f"future_{module.num_task_tokens}__"
                    "dino_depth_shared_within_time"
                )
                if getattr(module, "use_current_alignment", False)
                else (
                    "keyframe_major__shared_dino_private_depth_private"
                    if depth_head is not None
                    else "keyframe_major__legacy_single_branch"
                )
            ),
            "plan_token_strings": [
                f"<|sem_plan_{index}|>" for index in range(module.latent_len)
            ],
            "token_order": "keyframe_major_row_major",
            "planner_input_frame": (
                "separate_camera_images"
                if num_camera_views == 2
                else (
                    "fastwam_current_multicamera_composite"
                    if fastwam_data_config is not None
                    else "legacy_single_current_frame"
                )
            ),
        }
    )
    if num_camera_views == 2:
        if module.target_len % module.num_keyframes != 0:
            raise ValueError(
                "dual-camera target_len must divide evenly across keyframes: "
                f"{module.target_len} / {module.num_keyframes}"
            )
        meta.update(
            build_dual_camera_export_metadata(
                future_keyframe_offsets=offsets,
                num_keyframes=module.num_keyframes,
                target_tokens_per_keyframe=(
                    module.target_len // module.num_keyframes
                ),
                planner_token_count=module.latent_len,
            )
        )
        validate_dual_camera_export_metadata(meta)
    (ckpt / "planner_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "latest_checkpoint.txt").write_text(str(ckpt), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.full_finetune and args.lora_r > 0:
        raise ValueError("--full-finetune is mutually exclusive with LoRA; set --lora-r 0.")
    if args.plan_head_type == "lingbot_dino" and args.use_depth and not args.use_current_alignment and (
        args.shared_latent_per_keyframe <= 0 or args.private_latent_per_keyframe <= 0
    ):
        raise ValueError(
            "DINO+depth mode requires positive --shared-latent-per-keyframe and "
            "--private-latent-per-keyframe"
        )
    if args.fastwam_data_config is not None:
        preflight_fastwam_data_config(
            args.fastwam_data_config,
            dataset_dirs=args.fastwam_dataset_dir,
            text_embedding_cache_dir=args.fastwam_text_embedding_cache_dir,
            pretrained_norm_stats=args.fastwam_pretrained_norm_stats,
        )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    accelerator = build_accelerator(grad_accum=args.grad_accum, dtype=args.dtype)
    runtime_contract = validate_runtime_contract(
        accelerator,
        per_device_batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        expected_global_batch=args.expected_global_batch,
    )
    rank = int(accelerator.process_index)
    world = int(accelerator.num_processes)
    device = accelerator.device
    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    "distributed_type": runtime_contract.distributed_type,
                    "world_size": runtime_contract.world_size,
                    "batch_size_per_gpu": runtime_contract.per_device_batch_size,
                    "gradient_accumulation_steps": runtime_contract.gradient_accumulation_steps,
                    "global_batch_size": runtime_contract.global_batch_size,
                    "zero_stage": runtime_contract.zero_stage,
                    "dtype": args.dtype,
                    "gradient_checkpointing": bool(args.gradient_checkpointing),
                }
            ),
            flush=True,
        )
    accelerator.wait_for_everyone()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    model_path = args.model_path
    processor_path = None
    if args.init_planner_checkpoint is not None:
        initialization_dir = Path(args.init_planner_checkpoint)
        model_path = initialization_dir / "qwen3vl_lora_or_model"
        processor_path = initialization_dir / "processor"
        missing = [
            str(path.name)
            for path in (model_path, processor_path)
            if not path.is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                f"incomplete planner initialization {initialization_dir}: "
                f"missing directories {missing}"
            )
        # Save metadata must identify the model actually used for this fresh run.
        args.model_path = model_path

    model, processor = load_qwen3vl_model_and_processor(
        model_path,
        processor_path=processor_path,
        device=None,
        dtype=args.dtype,
        attn_implementation="sdpa",
        local_files_only=True,
        eval_mode=False,
    )
    # DIAL-style distinct latent tokens (gr00t <|bridge_i|>): for the CoVT bottleneck the LM
    # emits a small, structured set of latents (num_keyframes x num_latent_per_keyframe), so
    # give each position its OWN learnable token/embedding instead of repeating one
    # <|sem_plan|> — identical inputs differentiated by position alone tend to collapse
    # (all latents become the same -> keyframes stop evolving). mlp/baton keep the single
    # repeated token (their latent_len = target_len = thousands of tokens).
    if args.plan_head_type == "lingbot_dino" and args.use_current_alignment:
        latent_len = (
            4 if args.independent_modality_task_tokens else 2
        ) * int(args.num_task_tokens)
        plan_token_strs = [f"<|sem_plan_{i}|>" for i in range(latent_len)]
        plan_sequence = list(plan_token_strs)
    elif (
        args.plan_head_type == "lingbot_dino"
        and args.use_depth
        and args.shared_latent_per_keyframe > 0
    ):
        latent_len = int(args.num_keyframes) * (
            int(args.shared_latent_per_keyframe)
            + 2 * int(args.private_latent_per_keyframe)
        )
        plan_token_strs = [f"<|sem_plan_{i}|>" for i in range(latent_len)]
        plan_sequence = list(plan_token_strs)
    elif args.plan_head_type in ("covt", "lingbot_dino"):
        latent_len = int(args.num_keyframes) * int(args.num_latent_per_keyframe)
        plan_token_strs = [f"<|sem_plan_{i}|>" for i in range(latent_len)]
        plan_sequence = list(plan_token_strs)
    else:
        plan_token_strs = [PLAN_TOKEN]
        plan_sequence = [PLAN_TOKEN] * (int(args.num_keyframes) * int(args.grid_size) * int(args.grid_size))
    new_tokens = [t for t in plan_token_strs if t not in processor.tokenizer.get_vocab()]
    if new_tokens:
        processor.tokenizer.add_tokens(new_tokens, special_tokens=True)
        model.resize_token_embeddings(len(processor.tokenizer))
    plan_token_ids = [processor.tokenizer.convert_tokens_to_ids(t) for t in plan_token_strs]
    model.config.use_cache = False
    configure_gradient_checkpointing(model, enabled=args.gradient_checkpointing)
    set_trainable(
        model,
        freeze_vision=args.freeze_vision,
        freeze_lm_head=args.freeze_lm_head,
        full_finetune=args.full_finetune,
    )
    model = apply_lora(model, args)

    sig_model = sig_processor = sig_encode = None
    dino_encoder = None
    depth_encoder = None
    if args.plan_head_type == "lingbot_dino":
        if not args.online_plan_labels:
            raise ValueError("lingbot_dino requires --online-plan-labels (online DINO-video targets)")
        if (
            args.frame_ranges_json is None
            and args.fastwam_data_config is None
            and args.ge_act_data_config is None
        ):
            args.frame_ranges_json = args.dataset_root / "frame_ranges.json"
        if args.video_target_type == "dinov3":
            from dinov3_target import Dinov3TargetEncoder  # noqa: E402
            _vkw = {"input_size": args.dino_input_size, "device": device}
            if args.dinov3_model_dir is not None:
                _vkw["model_dir"] = args.dinov3_model_dir
            dino_encoder = Dinov3TargetEncoder(**_vkw)
            args.sample_feature_type = "dinov3"
            if args.semantic_dim <= 0:
                args.semantic_dim = dino_encoder.feature_dim  # DINOv3 ViT-H+ -> 1280
        elif args.video_target_type == "siglip2":
            from siglip2_target import Siglip2TargetEncoder  # noqa: E402
            _vkw = {
                "input_size": args.siglip2_input_size,
                "grid_size": args.siglip2_grid_size,
                "device": device,
            }
            if args.siglip2_model_dir is not None:
                _vkw["model_dir"] = args.siglip2_model_dir
            dino_encoder = Siglip2TargetEncoder(**_vkw)
            args.sample_feature_type = "siglip2"
            if args.semantic_dim <= 0:
                args.semantic_dim = dino_encoder.feature_dim  # SigLIP2 so400m -> 1152
        else:
            dino_encoder = DinoVideoTargetEncoder(
                ckpt_path=args.dino_teacher_ckpt,
                config_path=args.dino_teacher_config,
                input_size=args.dino_input_size,
                device=device,
            )
            args.sample_feature_type = "dino_video"
            if args.semantic_dim <= 0:
                args.semantic_dim = 1024
        if args.use_depth:
            if args.depth_target_type == "da3":
                from depth_anything3_target import DepthAnything3TargetEncoder  # noqa: E402
                _dkw = {
                    "process_res": args.da3_process_res,
                    "device": device,
                    "align_strategy": args.da3_align_strategy,
                }
                if args.da3_ckpt_dir is not None:
                    _dkw["ckpt_dir"] = args.da3_ckpt_dir
                if args.da3_align_strategy == "wsa_multilayer":
                    if args.da3_teacher_layers:
                        _dkw["teacher_layers"] = [
                            int(x) for x in str(args.da3_teacher_layers).split(",") if x.strip()
                        ]
                    if args.da3_layer_weights:
                        _dkw["layer_weights"] = [
                            float(x) for x in str(args.da3_layer_weights).split(",") if x.strip()
                        ]
                depth_encoder = DepthAnything3TargetEncoder(**_dkw)
                args.depth_dim = depth_encoder.feature_dim  # DA3-Large cat_token -> 2048 (per layer)
                # provenance for wrapper construction + meta (last_layer -> 1 layer, weight [1.0])
                args.da3_num_layers = int(depth_encoder.num_layers)
                args.da3_teacher_layers_resolved = [int(x) for x in depth_encoder.teacher_layers]
                args.da3_layer_weights_resolved = [float(x) for x in depth_encoder.layer_weights]
                _da3_grid = args.da3_process_res // 14
                if _da3_grid * _da3_grid != args.depth_grid_size * args.depth_grid_size:
                    raise ValueError(
                        f"--da3-process-res {args.da3_process_res} yields {_da3_grid}x{_da3_grid} tokens/keyframe, "
                        f"but --depth-grid-size {args.depth_grid_size} expects {args.depth_grid_size}^2"
                    )
            else:
                if args.depth_moge_path is None or args.depth_morgbd_path is None:
                    raise ValueError("--use-depth (morgbd) requires --depth-moge-path and --depth-morgbd-path")
                depth_encoder = DepthTargetEncoder(
                    moge_path=args.depth_moge_path,
                    morgbd_path=args.depth_morgbd_path,
                    input_size=args.depth_input_size,
                    num_tokens=args.depth_grid_size * args.depth_grid_size,
                    device=device,
                )
    elif args.online_plan_labels:
        from build_siglip2_semantic_plan_labels import encode_images, load_siglip2, semantic_feature_type

        if args.siglip2_encoder_path is None:
            raise ValueError("--online-plan-labels requires --siglip2-encoder-path")
        if (
            args.frame_ranges_json is None
            and args.fastwam_data_config is None
            and args.ge_act_data_config is None
        ):
            args.frame_ranges_json = args.dataset_root / "frame_ranges.json"
        sig_model, sig_processor = load_siglip2(args.siglip2_encoder_path, device, "bf16")
        sig_model.requires_grad_(False)
        sig_encode = encode_images
        # Probe dim + tokens-per-keyframe with a dummy frame; validates the head grid early.
        probe = sig_encode(sig_model, sig_processor, [Image.new("RGB", (64, 64))], args.online_grid_size, device, torch.float32)
        if probe.shape[1] != args.grid_size * args.grid_size:
            raise ValueError(
                f"online grid {args.online_grid_size} yields {probe.shape[1]} tokens/keyframe, "
                f"but --grid-size {args.grid_size} expects {args.grid_size * args.grid_size}"
            )
        args.sample_feature_type = semantic_feature_type(args.online_grid_size)
        if args.semantic_dim <= 0:
            args.semantic_dim = int(probe.shape[-1])
    else:
        if args.plan_label_dir is None:
            raise ValueError("--plan-label-dir is required unless --online-plan-labels is set")
        # Probe a sample label for semantic dim + feature type (the latter is recorded in
        # planner_meta.json; the old summary.json path never exists for these label dirs).
        first = next(iter(sorted(args.plan_label_dir.glob("*.pt"))), None)
        if first is None:
            raise RuntimeError(f"No .pt labels under {args.plan_label_dir}")
        payload = torch.load(first, map_location="cpu", weights_only=False)
        args.sample_feature_type = str(payload.get("feature_type", "unknown"))
        if args.semantic_dim <= 0:
            args.semantic_dim = int(payload["semantic_plan"].shape[-1])

    hidden_size = int(model.config.text_config.hidden_size)
    wrapper = PlannerWrapper(
        model=model,
        hidden_size=hidden_size,
        semantic_dim=args.semantic_dim,
        plan_token_ids=plan_token_ids,
        target_len=args.num_keyframes * args.grid_size * args.grid_size,
        num_keyframes=args.num_keyframes,
        grid_size=args.grid_size,
        num_latent_per_keyframe=args.num_latent_per_keyframe,
        shared_latent_per_keyframe=args.shared_latent_per_keyframe,
        private_latent_per_keyframe=args.private_latent_per_keyframe,
        plan_head_type=args.plan_head_type,
        plan_head_num_heads=args.plan_head_num_heads,
        plan_head_dropout=args.plan_head_dropout,
        sem_mlp_hidden_size=args.sem_mlp_hidden_size,
        mse_loss_weight=args.mse_loss_weight,
        cosine_loss_weight=args.cosine_loss_weight,
        norm_loss_weight=args.norm_loss_weight,
        variance_loss_weight=args.variance_loss_weight,
        infonce_loss_weight=args.infonce_loss_weight,
        infonce_temperature=args.infonce_temperature,
        use_depth=args.use_depth,
        depth_dim=args.depth_dim,
        depth_grid_size=args.depth_grid_size,
        depth_loss_weight=args.depth_loss_weight,
        use_current_alignment=args.use_current_alignment,
        bidirectional_plan_attn=args.bidirectional_plan_attn,
        independent_modality_task_tokens=args.independent_modality_task_tokens,
        num_task_tokens=args.num_task_tokens,
        current_dino_loss_weight=args.current_dino_loss_weight,
        future_dino_loss_weight=args.future_dino_loss_weight,
        current_depth_loss_weight=args.current_depth_loss_weight,
        future_depth_loss_weight=args.future_depth_loss_weight,
        train_plan_token_embedding=args.train_plan_token_embedding,
        da3_align_strategy=getattr(args, "da3_align_strategy", "last_layer"),
        da3_num_layers=int(getattr(args, "da3_num_layers", 1)),
        da3_layer_weights=getattr(args, "da3_layer_weights_resolved", None),
        num_camera_views=2 if args.ge_act_data_config is not None else 1,
    )
    if args.init_planner_checkpoint is not None:
        initialization_report = load_planner_initialization(
            wrapper,
            args.init_planner_checkpoint,
        )
        if accelerator.is_main_process:
            print(
                json.dumps({"planner_initialization": initialization_report}),
                flush=True,
            )
    elif args.plan_head_type == "lingbot_dino" and args.head_warmstart_ckpt is not None:
        head_state = _load_lingbot_head_state(args.head_warmstart_ckpt)
        report = wrapper.plan_head.load_lingbot_warmstart(head_state, head_name="future_video_align_head")
        if wrapper.depth_head is not None:
            depth_report = wrapper.depth_head.load_lingbot_warmstart(head_state, head_name="future_depth_align_head")
        current_report = None
        current_depth_report = None
        if wrapper.current_plan_head is not None:
            current_report = wrapper.current_plan_head.load_lingbot_warmstart(
                head_state, head_name="current_video_align_head"
            )
        if wrapper.current_depth_head is not None:
            current_depth_report = wrapper.current_depth_head.load_lingbot_warmstart(
                head_state, head_name="depth_align_head"
            )
        if accelerator.is_main_process:
            print(json.dumps({"head_warmstart": report}), flush=True)
            if wrapper.depth_head is not None:
                print(json.dumps({"depth_head_warmstart": depth_report}), flush=True)
            if current_report is not None:
                print(json.dumps({"current_head_warmstart": current_report}), flush=True)
            if current_depth_report is not None:
                print(json.dumps({"current_depth_head_warmstart": current_depth_report}), flush=True)
    wrapper.to(device)
    wrapper.train()
    if accelerator.is_main_process:
        trainable, total = count_trainable_parameters(wrapper)
        print(
            json.dumps(
                {
                    "full_finetune": bool(args.full_finetune),
                    "lora_r": int(args.lora_r),
                    "freeze_vision": bool(args.freeze_vision),
                    "freeze_lm_head": bool(args.freeze_lm_head),
                    "plan_head_type": args.plan_head_type,
                    "plan_head_num_heads": int(args.plan_head_num_heads),
                    "sample_one_window_per_stem": bool(args.sample_one_window_per_stem),
                    "trainable_params": trainable,
                    "total_params": total,
                    "trainable_ratio": trainable / max(total, 1),
                }
            ),
            flush=True,
        )

    if args.ge_act_data_config is not None:
        ge_act_offsets = keyframe_offsets(
            args.sequence_length,
            args.num_keyframes,
            args.keyframe_scheme,
            args.keyframe_gamma,
        )
        dataset = load_ge_act_dual_camera_planner_dataset(
            args.ge_act_data_config,
            future_offsets=ge_act_offsets,
        )
        collator = DualCameraPlannerCollator(
            processor=processor,
            plan_sequence=plan_sequence,
        )
        if accelerator.is_main_process:
            print(
                json.dumps(
                    {
                        "online_plan_labels": True,
                        "dataset_source": "ge_act",
                        "stems": len(dataset),
                        "keyframe_offsets": list(ge_act_offsets),
                        "num_camera_views": 2,
                        "feature_type": args.sample_feature_type,
                    }
                ),
                flush=True,
            )
    elif args.online_plan_labels:
        if args.fastwam_data_config is not None:
            dataset = FastWAMOnlinePlannerDataset.from_config(
                args.fastwam_data_config,
                dataset_dirs=args.fastwam_dataset_dir,
                text_embedding_cache_dir=(
                    args.fastwam_text_embedding_cache_dir
                ),
                pretrained_norm_stats=args.fastwam_pretrained_norm_stats,
                max_samples=args.max_samples,
                offsets=keyframe_offsets(
                    args.sequence_length,
                    args.num_keyframes,
                    args.keyframe_scheme,
                    args.keyframe_gamma,
                ),
            )
        else:
            dataset = OnlineSemanticPlanDataset(
                dataset_root=args.dataset_root,
                frame_ranges_json=args.frame_ranges_json,
                num_keyframes=args.num_keyframes,
                sequence_length=args.sequence_length,
                keyframe_scheme=args.keyframe_scheme,
                keyframe_gamma=args.keyframe_gamma,
                max_samples=args.max_samples,
                seed=args.seed,
            )
        collator = Collator(
            processor=processor,
            plan_sequence=plan_sequence,
        )
        if accelerator.is_main_process:
            print(
                json.dumps(
                    {
                        "online_plan_labels": True,
                        "stems": len(dataset),
                        "keyframe_scheme": args.keyframe_scheme,
                        "keyframe_offsets": dataset.offsets,
                        "sequence_length": args.sequence_length,
                        "feature_type": args.sample_feature_type,
                    }
                ),
                flush=True,
            )
    else:
        dataset = SemanticPlanDataset(
            dataset_root=args.dataset_root,
            label_dir=args.plan_label_dir,
            num_keyframes=args.num_keyframes,
            grid_size=args.grid_size,
            max_samples=args.max_samples,
            sample_one_window_per_stem=args.sample_one_window_per_stem,
        )
        collator = Collator(
            processor=processor,
            plan_sequence=plan_sequence,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    optim = build_optimizer(wrapper, args)
    scheduler = build_scheduler(optim, args)
    wrapper, optim, loader = accelerator.prepare(wrapper, optim, loader)
    wandb_run = None
    if accelerator.is_main_process and os.environ.get("PLANNER_WANDB", "1") == "1":
        try:
            import wandb

            wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "qwen3vl_semantic_planner"),
                name=os.environ.get("WANDB_NAME", args.output_dir.name),
                dir=str(args.output_dir),
                mode=os.environ.get("WANDB_MODE", "offline"),
                config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            )
        except Exception as exc:  # missing wandb must not kill training; JSON stdout remains
            print(f"wandb disabled: {exc}", flush=True)
    step = 0
    micro_step = 0
    running_loss = 0.0
    deepspeed_enabled = is_deepspeed(accelerator)
    if not deepspeed_enabled:
        optim.zero_grad(set_to_none=True)
    pbar = tqdm(
        total=args.max_steps,
        disable=not accelerator.is_local_main_process,
        desc="qwen3vl planner",
    )
    while step < args.max_steps:
        # Advance the dataset epoch too, else sample_one_window_per_stem never resamples the
        # window per stem (the new epoch is picked up when the DataLoader respawns workers).
        dataset.set_epoch(step)
        for batch in loader:
            batch.pop("stems", None)
            keyframes = batch.pop("keyframe_images", None)
            current = batch.pop("current_image", None)
            current_camera_images = batch.pop("current_camera_images", None)
            future_camera_images = batch.pop("future_camera_images", None)
            if (current_camera_images is None) != (future_camera_images is None):
                raise RuntimeError(
                    "dual-camera batches must contain both current_camera_images "
                    "and future_camera_images"
                )
            future_video_effective_fps = batch.pop(
                "future_video_effective_fps",
                None,
            )
            module = accelerator.unwrap_model(wrapper)
            model_dtype = next(module.model.parameters()).dtype
            batch = move_qwen_inputs_to_device(batch, device, model_dtype=model_dtype)
            if current_camera_images is not None:
                if dino_encoder is None or depth_encoder is None:
                    raise RuntimeError(
                        "GE-Act dual-camera training requires both appearance and depth teachers"
                    )
                if future_camera_images.ndim == 6:
                    batch.update(
                        encode_dual_camera_future_targets(
                            current_camera_images,
                            future_camera_images,
                            appearance_encoder=dino_encoder,
                            depth_encoder=depth_encoder,
                        )
                    )
                else:
                    batch.update(
                        encode_dual_camera_teacher_targets(
                            current_camera_images,
                            future_camera_images,
                            appearance_encoder=dino_encoder,
                            depth_encoder=depth_encoder,
                        )
                    )
            elif keyframes is not None:
                with torch.no_grad():
                    if dino_encoder is not None:
                        # Online DINO-video targets: teacher over [current, current, keyframe_k] clips
                        # -> [B, K*256, 1024], matching the LingbotDinoPlanHead output.
                        cur = current.permute(0, 3, 1, 2).contiguous()  # (B,3,H,W)
                        kfs = [keyframes[:, j].permute(0, 3, 1, 2).contiguous() for j in range(keyframes.shape[1])]
                        if args.use_current_alignment:
                            current_dino, future_dino = dino_encoder.encode_current_and_future(
                                cur,
                                kfs[0],
                                effective_fps=(
                                    future_video_effective_fps[:, 0]
                                    if future_video_effective_fps is not None
                                    else None
                                ),
                            )
                            batch["current_dino_labels"] = current_dino.float()
                            batch["semantic_plan_labels"] = future_dino.float()
                            current_depth, future_depth = depth_encoder.encode_current_and_future(
                                cur, kfs[0]
                            )
                            batch["current_depth_labels"] = current_depth.float()
                            batch["depth_plan_labels"] = future_depth.float()
                        else:
                            batch["semantic_plan_labels"] = dino_encoder.encode_future_keyframes(
                                cur,
                                kfs,
                                effective_fps=future_video_effective_fps,
                            ).float()
                            if depth_encoder is not None:
                                # LingBot-Depth targets over the SAME future keyframes -> [B, K*256, 1024].
                                batch["depth_plan_labels"] = depth_encoder.encode_future_keyframes(kfs).float()
                    else:
                        # Online SigLIP2 targets (bit-consistent with the offline builder).
                        b, k = keyframes.shape[0], keyframes.shape[1]
                        imgs = [keyframes[i, j].detach().cpu().numpy() for i in range(b) for j in range(k)]
                        target = sig_encode(sig_model, sig_processor, imgs, args.online_grid_size, device, torch.float32)
                        batch["semantic_plan_labels"] = target.reshape(b, k * target.shape[1], target.shape[-1])
            with accumulation_context(accelerator, wrapper):
                out = wrapper(**batch)
                accelerator.backward(out["loss"])
                if not deepspeed_enabled:
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            (
                                parameter
                                for parameter in wrapper.parameters()
                                if parameter.requires_grad
                            ),
                            1.0,
                        )
                    optim.step()
                    optim.zero_grad(set_to_none=True)
            running_loss += float(out["loss"].detach())
            micro_step += 1
            if is_optimizer_update(accelerator, micro_step, args.grad_accum):
                if scheduler is not None:
                    scheduler.step()
                step += 1
                if accelerator.is_main_process:
                    pbar.update(1)
                    if step % args.log_steps == 0:
                        average_loss = running_loss / max(
                            args.log_steps * args.grad_accum,
                            1,
                        )
                        log_entry = {
                            "step": step,
                            "loss": average_loss,
                            "lr": (
                                scheduler.get_last_lr()[0]
                                if scheduler is not None
                                else args.lr
                            ),
                        }
                        log_entry.update(
                            {key: float(value) for key, value in out.items() if key != "loss"}
                        )
                        print(json.dumps(log_entry), flush=True)
                        if wandb_run is not None:
                            wandb_run.log(log_entry, step=step)
                if step % args.log_steps == 0:
                    running_loss = 0.0
                if should_save_periodic_checkpoint(
                    step=step,
                    max_steps=args.max_steps,
                    save_steps=args.save_steps,
                    save_start_step=args.save_start_step,
                ):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_checkpoint(
                            args.output_dir,
                            step,
                            checkpoint_module(accelerator, wrapper),
                            processor,
                            args,
                            rank=0,
                        )
                    accelerator.wait_for_everyone()
                if step >= args.max_steps:
                    break
        if len(loader) == 0:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(
            args.output_dir,
            step,
            checkpoint_module(accelerator, wrapper),
            processor,
            args,
            rank=0,
        )
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        pbar.close()
    if accelerator.is_main_process and wandb_run is not None:
        wandb_run.finish()
    accelerator.end_training()


if __name__ == "__main__":
    main()
