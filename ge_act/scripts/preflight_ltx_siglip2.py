#!/usr/bin/env python3
"""Preflight validation for eight-GPU LTX + SigLIP2 training."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml


REQUIRED_MODULES = (
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "av",
    "safetensors",
)


def _nearest_existing_parent(path: Path) -> Path:
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def collect_preflight_errors(
    config: dict[str, Any],
    *,
    world_size: int,
    check_paths: bool = True,
    ge_act_root: Path | None = None,
    minimum_free_gb: float = 100.0,
) -> list[str]:
    errors: list[str] = []
    ge_act_root = ge_act_root or Path(__file__).resolve().parents[1]

    semantic = config.get("semantic_plan", {})
    model_config = config.get("diffusion_model", {}).get("config", {})
    train_data = config.get("data", {}).get("train", {})
    val_data = config.get("data", {}).get("val", {})
    keyframes = semantic.get("keyframe_indices", [])
    semantic_source = semantic.get("source", "gt_siglip2")
    joint = config.get("joint_training", {})
    joint_enabled = isinstance(joint, dict) and bool(joint.get("enabled", False))
    hdf5_backend = config.get("train_data_class") == "LiberoFastWAMHDF5Dataset"
    if not semantic.get("enabled", False):
        errors.append("semantic_plan.enabled must be true")
    if semantic_source == "gt_siglip2":
        if keyframes != [0, 3, 5, 8]:
            errors.append("semantic keyframes must be [0, 3, 5, 8]")
        if semantic.get("validation_mode", "gt") != "gt":
            errors.append("GT SigLIP2 validation_mode must be gt")
        if not semantic.get("model_name_or_path"):
            errors.append("semantic_plan.model_name_or_path is required")
    elif semantic_source == "vlm_planner":
        if joint_enabled:
            if keyframes != [2, 4, 6, 8]:
                errors.append("joint VLM planner keyframe offsets must be [2, 4, 6, 8]")
        elif keyframes != [8]:
            errors.append("VLM planner semantic keyframes must be [8]")
        if semantic.get("validation_mode") != "planner":
            errors.append("VLM planner validation_mode must be planner")
        if not semantic.get("planner_checkpoint"):
            errors.append("semantic_plan.planner_checkpoint is required")
    else:
        errors.append(f"unknown semantic_plan.source: {semantic_source}")
    if semantic.get("tokens_per_frame") != 256:
        errors.append("SigLIP2 must provide 256 tokens per frame")
    if (
        semantic.get("feature_dim") != 1024
        or model_config.get("semantic_plan_in_dim") != 1024
    ):
        errors.append("SigLIP2 feature width must be 1024")
    if model_config.get("semantic_plan_cross_attention_blocks") != list(range(28)):
        errors.append("semantic cross-attention must be enabled in all 28 LTX blocks")
    if model_config.get("semantic_plan_num_keyframes") != len(keyframes):
        errors.append(
            "LTX semantic keyframe count must match semantic_plan.keyframe_indices"
        )
    if model_config.get("semantic_plan_num_views") != 2:
        errors.append("LTX semantic plan must preserve two camera views")
    if train_data.get("chunk") != 9 or train_data.get("n_previous") != 4:
        errors.append("FastWAM clip layout must use four memory and nine future frames")
    if train_data.get("source_fps") != 20:
        errors.append("LIBERO source_fps must be 20")
    train_cache_root = None
    if hdf5_backend:
        train_manifest = train_data.get("manifest_path")
        val_manifest = val_data.get("manifest_path")
        if not train_manifest:
            errors.append("training HDF5 manifest is missing")
        if train_manifest != val_manifest:
            errors.append("train and validation must use the same HDF5 manifest")
    else:
        if not train_data.get("require_predecoded", False):
            errors.append("training must require predecoded RGB caches")
        if not val_data.get("require_predecoded", False):
            errors.append("validation must require predecoded RGB caches")
        train_cache_root = train_data.get("predecoded_video_root")
        val_cache_root = val_data.get("predecoded_video_root")
        if not train_cache_root:
            errors.append("training predecoded RGB cache root is missing")
        if train_cache_root != val_cache_root:
            errors.append("train and validation must use the same predecoded RGB cache")
    if max(keyframes, default=-1) >= train_data.get("chunk", 0):
        errors.append("semantic keyframes exceed the future clip")

    global_batch = (
        int(config.get("batch_size", 0))
        * int(config.get("gradient_accumulation_steps", 0))
        * int(world_size)
    )
    if global_batch != 128:
        errors.append(f"global batch must be 128, got {global_batch}")
    if config.get("train_steps") != 30_000:
        errors.append("train_steps must be 30000")
    if not config.get("gradient_checkpointing", False):
        errors.append("gradient checkpointing must be enabled for the initial run")

    if joint_enabled:
        expected_offsets = [2, 4, 6, 8]
        if semantic_source != "vlm_planner":
            errors.append("joint training requires semantic_plan.source=vlm_planner")
        if joint.get("future_keyframe_offsets") != expected_offsets:
            errors.append("joint planner future_keyframe_offsets must be [2, 4, 6, 8]")
        if joint.get("num_keyframes") != 4:
            errors.append("joint training requires four semantic keyframes")
        if (
            model_config.get("semantic_plan_num_views") != 2
            or joint.get("num_camera_views") != 2
        ):
            errors.append("joint training requires two camera views")
        if (
            semantic.get("tokens_per_frame") != 256
            or joint.get("tokens_per_keyframe") != 256
        ):
            errors.append("joint semantic plan must use 256 tokens per keyframe")
        if (
            semantic.get("feature_dim") != 1024
            or model_config.get("semantic_plan_in_dim") != 1024
            or joint.get("semantic_feature_dim") != 1024
        ):
            errors.append("joint semantic feature width must be 1024")
        if joint.get("da3_align_strategy") != "wsa_multilayer":
            errors.append("joint DA3 teacher must use four-layer WSA")
        if joint.get("da3_teacher_layers") != [11, 15, 19, 23]:
            errors.append("joint DA3 teacher layers must be [11, 15, 19, 23]")
        if joint.get("da3_feature_dim") != 2048:
            errors.append("joint DA3 feature width must be 2048")
        if int(world_size) != 8:
            errors.append("joint formal training requires world size 8")
        if (
            config.get("batch_size") != 1
            or config.get("gradient_accumulation_steps") != 16
        ):
            errors.append("joint training requires per-GPU batch 1 and accumulation 16")
        if config.get("lr") != 2e-5:
            errors.append("joint LTX base lr must be 2e-5")
        if config.get("semantic_lr") != 1e-4:
            errors.append("joint LTX semantic lr must be 1e-4")
        if joint.get("qwen_lr") != 1e-6:
            errors.append("joint Qwen lr must be 1e-6")
        if joint.get("planner_head_lr") != 3e-5:
            errors.append("joint planner head lr must be 3e-5")
        if joint.get("planner_loss_weight") != 0.1:
            errors.append("joint planner_loss_weight must be 0.1")
        if float(joint.get("lm_plan_loss_weight", 0.0)) <= 0.0:
            errors.append("joint lm_plan_loss_weight must be positive")
        if bool(joint.get("bidirectional_plan_attn", True)):
            errors.append("joint planner checkpoint must use causal attention")
        if not config.get("gradient_checkpointing", False):
            errors.append("joint training requires LTX gradient checkpointing")
        if not joint.get("qwen_gradient_checkpointing", False):
            errors.append("joint training requires Qwen gradient checkpointing")
        if config.get("mixed_precision") != "bf16":
            errors.append("joint training mixed_precision must be bf16")
        deepspeed = config.get("deepspeed", {})
        if not config.get("use_deepspeed", False):
            errors.append("joint formal training requires DeepSpeed")
        if deepspeed.get("zero_optimization", {}).get("stage") != 2:
            errors.append("joint formal training requires DeepSpeed ZeRO-2")
        if not deepspeed.get("bf16", {}).get("enabled", False):
            errors.append("joint DeepSpeed bf16 must be enabled")
        if config.get("lr_warmup_steps") != 1_000:
            errors.append("joint lr_warmup_steps must be 1000")
        if config.get("save_steps") != [20_000, 25_000, 30_000]:
            errors.append("joint save_steps must be [20000, 25000, 30000]")
        if not joint.get("siglip2_model_dir"):
            errors.append("joint SigLIP2 teacher path is required")
        if not joint.get("da3_ckpt_dir"):
            errors.append("joint DA3 teacher checkpoint is required")
        if not joint.get("da3_code_root"):
            errors.append("joint DA3 code root is required")
        if not hdf5_backend:
            expected_cameras = [
                "observation.images.image",
                "observation.images.wrist_image",
            ]
            for split, data_config in (
                ("training", train_data),
                ("validation", val_data),
            ):
                if data_config.get("valid_cam") != expected_cameras:
                    errors.append(
                        f"joint {split} data must preserve ordered main/wrist cameras"
                    )
            if train_data.get("data_roots") != val_data.get("data_roots"):
                errors.append("joint train and validation data roots must match")
            if train_data.get("stat_file") != val_data.get("stat_file"):
                errors.append("joint train and validation statistics must match")

    if not check_paths:
        return errors

    for module_name in REQUIRED_MODULES:
        if importlib.util.find_spec(module_name) is None:
            errors.append(f"missing Python module: {module_name}")

    required_paths = {
        "LTX pretrained components": config.get("pretrained_model_name_or_path"),
        "base diffusion checkpoint": config.get("diffusion_model", {}).get(
            "model_path"
        ),
    }
    if semantic_source == "gt_siglip2":
        required_paths["SigLIP2 checkpoint"] = semantic.get("model_name_or_path")
    elif semantic_source == "vlm_planner":
        required_paths["dual-camera VLM planner"] = semantic.get("planner_checkpoint")
    if joint_enabled:
        required_paths.update(
            {
                "joint SigLIP2 teacher": joint.get("siglip2_model_dir"),
                "joint DA3 teacher": joint.get("da3_ckpt_dir"),
                "joint DA3 code root": joint.get("da3_code_root"),
            }
        )
    for label, raw_path in required_paths.items():
        if not raw_path or not Path(raw_path).exists():
            errors.append(f"missing {label}: {raw_path}")
    if semantic_source == "gt_siglip2":
        siglip_path = Path(semantic.get("model_name_or_path", ""))
        if siglip_path.is_dir() and not (
            list(siglip_path.glob("*.safetensors"))
            or (siglip_path / "pytorch_model.bin").is_file()
        ):
            errors.append(f"SigLIP2 directory has no model weights: {siglip_path}")
    ltx_path = Path(config.get("pretrained_model_name_or_path", ""))
    if ltx_path.is_dir():
        for component in ("tokenizer", "text_encoder", "vae"):
            if not (ltx_path / component).is_dir():
                errors.append(
                    f"LTX component directory is missing: {ltx_path / component}"
                )
    for data_root in sorted(set(train_data.get("data_roots", []))):
        if not Path(data_root).is_dir():
            errors.append(f"missing training data root: {data_root}")
    if joint_enabled:
        for data_root in sorted(set(val_data.get("data_roots", []))):
            if not Path(data_root).is_dir():
                errors.append(f"missing validation data root: {data_root}")
    if hdf5_backend:
        manifest_path = train_data.get("manifest_path")
        if manifest_path and not Path(manifest_path).is_file():
            errors.append(f"missing HDF5 manifest: {manifest_path}")
    elif train_cache_root and not Path(train_cache_root).is_dir():
        errors.append(f"missing predecoded RGB cache root: {train_cache_root}")

    stat_file = Path(train_data.get("stat_file", ""))
    if not stat_file.is_absolute():
        stat_file = ge_act_root / stat_file
    if not stat_file.is_file():
        errors.append(f"missing normalization statistics: {stat_file}")

    if joint_enabled:
        planner_path = Path(semantic.get("planner_checkpoint", ""))
        planner_meta_path = planner_path / "planner_meta.json"
        planner_meta: dict[str, Any] | None = None
        if not planner_meta_path.is_file():
            errors.append(f"missing planner metadata: {planner_meta_path}")
        else:
            try:
                loaded_meta = json.loads(planner_meta_path.read_text(encoding="utf-8"))
                if not isinstance(loaded_meta, dict):
                    raise TypeError("planner metadata root is not a mapping")
                planner_meta = loaded_meta
            except (OSError, json.JSONDecodeError, TypeError) as error:
                errors.append(f"invalid planner metadata: {planner_meta_path}: {error}")
        if planner_meta is not None:
            metadata_contract = (
                (
                    "future_keyframe_offsets",
                    [2, 4, 6, 8],
                    "planner metadata offsets must be [2, 4, 6, 8]",
                ),
                (
                    "num_camera_views",
                    2,
                    "planner metadata must describe two camera views",
                ),
                ("num_keyframes", 4, "planner metadata must describe K4"),
                (
                    "target_tokens_per_keyframe",
                    256,
                    "planner metadata must use 256 tokens per keyframe",
                ),
                (
                    "semantic_dim",
                    1024,
                    "planner metadata semantic width must be 1024",
                ),
                (
                    "da3_align_strategy",
                    "wsa_multilayer",
                    "planner metadata DA3 strategy must be wsa_multilayer",
                ),
                (
                    "da3_teacher_layers",
                    [11, 15, 19, 23],
                    "planner metadata DA3 teacher layers must be [11, 15, 19, 23]",
                ),
                (
                    "depth_feature_dim",
                    2048,
                    "planner metadata DA3 feature width must be 2048",
                ),
                (
                    "bidirectional_plan_attn",
                    False,
                    "planner metadata must set bidirectional_plan_attn=false",
                ),
            )
            for field, expected, message in metadata_contract:
                if planner_meta.get(field) != expected:
                    errors.append(message)

    output_path = _nearest_existing_parent(Path(config.get("output_dir", ".")))
    if not os.access(output_path, os.W_OK):
        errors.append(f"output path is not writable: {output_path}")
    else:
        free_gb = shutil.disk_usage(output_path).free / 1024**3
        if free_gb < minimum_free_gb:
            errors.append(
                f"output filesystem has only {free_gb:.1f} GiB free; require {minimum_free_gb:.1f} GiB"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--minimum-free-gb", type=float, default=100.0)
    args = parser.parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    errors = collect_preflight_errors(
        config,
        world_size=args.world_size,
        minimum_free_gb=args.minimum_free_gb,
    )
    if errors:
        print("GE-Act SigLIP2 preflight failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("GE-Act SigLIP2 preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
