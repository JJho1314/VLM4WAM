#!/usr/bin/env python3
"""Preflight validation for opt-in LIBERO FastWAM HDF5 training."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml


GE_ACT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODULES = (
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "deepspeed",
    "safetensors",
    "h5py",
    "cv2",
)
DATASET_CLASS_PATH = "data/libero_fastwam_hdf5_dataset.py"
DATASET_CLASS = "LiberoFastWAMHDF5Dataset"
CAMERAS = [
    "observation.images.image",
    "observation.images.wrist_image",
]
BATON_TEACHER_SOURCE = "qwen35_baton_teacher"
BATON_PREDICTION_SOURCE = "qwen35_baton_prediction"
BATON_SOURCES = {BATON_TEACHER_SOURCE, BATON_PREDICTION_SOURCE}
OLD_LOADER_KEYS = {
    "data_roots",
    "domains",
    "predecoded_video_root",
    "require_predecoded",
    "sample_size",
    "preprocess",
    "random_crop",
    "state_key",
    "action_key",
    "ignore_seek",
}
FIXED_DATA_FIELDS = {
    "source_fps": 20,
    "sample_n_frames": 500,
    "valid_cam": CAMERAS,
    "chunk": 9,
    "action_chunk": 36,
    "n_previous": 4,
    "previous_pick_mode": "random",
    "action_type": "absolute",
    "action_space": "eef",
}


def _nearest_existing_parent(path: Path) -> Path:
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def load_manifest(path: Path):
    """Lazily import H1 only after the h5py dependency check succeeds."""
    repository_root = str(GE_ACT_ROOT.parent)
    added_repository_root = repository_root not in sys.path
    if added_repository_root:
        sys.path.insert(0, repository_root)
    try:
        schema = importlib.import_module("ge_act.data.libero_fastwam_hdf5_schema")
        return schema.load_manifest(path)
    finally:
        if added_repository_root:
            sys.path.remove(repository_root)


def _mapping(value: Any) -> dict[str, Any]:
    return value if type(value) is dict else {}


def _matches_exact(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _matches_exact(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected)
        )
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(
            _matches_exact(actual[key], expected_value)
            for key, expected_value in expected.items()
        )
    return actual == expected


def _append_exact_error(
    errors: list[str], label: str, actual: Any, expected: Any
) -> None:
    if not _matches_exact(actual, expected):
        errors.append(f"{label} must be {expected!r}, got {actual!r}")


def _validate_dataset_config(
    config: dict[str, Any], errors: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    semantic = _mapping(config.get("semantic_plan"))
    grounded = semantic.get("source") == "qwen35_grounded"
    expected_classes = {
        "train_data_class_path": (
            "data/libero_hindsight_hdf5_dataset.py"
            if grounded
            else DATASET_CLASS_PATH
        ),
        "train_data_class": (
            "LiberoHindsightHDF5Dataset"
            if grounded
            else DATASET_CLASS
        ),
        "val_data_class_path": (
            "data/libero_hindsight_hdf5_dataset.py"
            if grounded
            else DATASET_CLASS_PATH
        ),
        "val_data_class": (
            "LiberoHindsightHDF5Dataset"
            if grounded
            else DATASET_CLASS
        ),
    }
    for field, expected in expected_classes.items():
        _append_exact_error(errors, field, config.get(field), expected)

    data = _mapping(config.get("data"))
    train_data = _mapping(data.get("train"))
    val_data = _mapping(data.get("val"))
    for split, split_data, is_training in (
        ("train", train_data, True),
        ("val", val_data, False),
    ):
        fixed_fields = dict(FIXED_DATA_FIELDS)
        if grounded:
            fixed_fields["previous_pick_mode"] = "uniform"
        for field, expected in fixed_fields.items():
            _append_exact_error(
                errors,
                f"{split} {field}",
                split_data.get(field),
                expected,
            )
        _append_exact_error(
            errors,
            f"{split} train_dataset",
            split_data.get("train_dataset"),
            is_training,
        )
        for forbidden in sorted(OLD_LOADER_KEYS & set(split_data)):
            errors.append(f"{split} data must not contain old-loader key {forbidden}")
        if grounded:
            cache = split_data.get("hindsight_cache")
            if type(cache) is not str or not cache:
                errors.append(
                    f"{split} hindsight_cache must be a non-empty path"
                )

    train_manifest = train_data.get("manifest_path")
    val_manifest = val_data.get("manifest_path")
    if type(train_manifest) is not str or not train_manifest:
        errors.append("train manifest_path must be a non-empty path")
    if type(val_manifest) is not str or not val_manifest:
        errors.append("val manifest_path must be a non-empty path")
    if train_manifest != val_manifest:
        errors.append("train and validation must use the same manifest")
    if grounded and train_data.get("hindsight_cache") != val_data.get(
        "hindsight_cache"
    ):
        errors.append("train and validation must use the same hindsight cache")
    for split, split_data in (("train", train_data), ("val", val_data)):
        stat_file = split_data.get("stat_file")
        if type(stat_file) is not str or not stat_file:
            errors.append(f"{split} stat_file must be a non-empty path")
    return train_data, val_data


def _validate_training_config(config: dict[str, Any], world_size: int) -> list[str]:
    errors: list[str] = []
    train_data, val_data = _validate_dataset_config(config, errors)

    semantic = _mapping(config.get("semantic_plan"))
    diffusion = _mapping(config.get("diffusion_model"))
    model_config = _mapping(diffusion.get("config"))
    semantic_source = semantic.get("source")
    if semantic_source == "qwen35_grounded":
        _validate_grounded_training_config(
            config,
            semantic=semantic,
            model_config=model_config,
            train_data=train_data,
            val_data=val_data,
            world_size=world_size,
            errors=errors,
        )
        return errors
    if semantic.get("source") in BATON_SOURCES:
        _validate_baton_training_config(
            config,
            semantic=semantic,
            model_config=model_config,
            train_data=train_data,
            val_data=val_data,
            world_size=world_size,
            errors=errors,
        )
        return errors
    _append_exact_error(errors, "semantic_plan.enabled", semantic.get("enabled"), True)
    _append_exact_error(
        errors,
        "semantic keyframes",
        semantic.get("keyframe_indices"),
        [0, 3, 5, 8],
    )
    if not _matches_exact(semantic.get("tokens_per_frame"), 256):
        errors.append(
            "SigLIP2 must provide 256 tokens per frame, "
            f"got {semantic.get('tokens_per_frame')!r}"
        )
    if not _matches_exact(semantic.get("feature_dim"), 1024) or not _matches_exact(
        model_config.get("semantic_plan_in_dim"), 1024
    ):
        errors.append("SigLIP2 feature width must be 1024")
    _append_exact_error(
        errors,
        "semantic cross-attention in all 28 LTX blocks",
        model_config.get("semantic_plan_cross_attention_blocks"),
        list(range(28)),
    )
    _append_exact_error(
        errors,
        "diffusion semantic_plan_context",
        model_config.get("semantic_plan_context"),
        True,
    )
    _append_exact_error(
        errors,
        "diffusion semantic_plan_num_keyframes",
        model_config.get("semantic_plan_num_keyframes"),
        4,
    )
    _append_exact_error(
        errors,
        "diffusion semantic_plan_num_views",
        model_config.get("semantic_plan_num_views"),
        2,
    )
    _append_exact_error(
        errors, "diffusion num_layers", model_config.get("num_layers"), 28
    )

    if type(world_size) is not int or type(world_size) is bool or world_size <= 0:
        errors.append("world_size must be a positive integer")
    batch_size = config.get("batch_size")
    accumulation = config.get("gradient_accumulation_steps")
    if all(
        type(value) is int and type(value) is not bool and value > 0
        for value in (batch_size, accumulation, world_size)
    ):
        global_batch = batch_size * accumulation * world_size
        if global_batch != 128:
            errors.append(f"global batch must be 128, got {global_batch}")
    else:
        errors.append(
            "batch_size and gradient_accumulation_steps must be positive integers"
        )
    _append_exact_error(errors, "train_steps", config.get("train_steps"), 30_000)
    _append_exact_error(
        errors,
        "save_steps",
        config.get("save_steps"),
        [20_000, 25_000, 30_000],
    )
    _append_exact_error(
        errors,
        "gradient checkpointing",
        config.get("gradient_checkpointing"),
        True,
    )

    safety_settings = {
        "model_name": "ltx_train",
        "is_i2v": True,
        "return_action": False,
        "return_video": True,
        "train_mode": "video_only",
        "mixed_precision": "bf16",
        "add_state": False,
        "load_weights": True,
        "use_deepspeed": True,
        "diffusion_model_class_path": "models/ltx_models/transformer_ltx_multiview.py",
        "diffusion_model_class": "LTXVideoTransformer3DModel",
    }
    for field, expected in safety_settings.items():
        _append_exact_error(errors, field, config.get(field), expected)
    deepspeed = _mapping(config.get("deepspeed"))
    zero = _mapping(deepspeed.get("zero_optimization"))
    _append_exact_error(errors, "DeepSpeed ZeRO stage", zero.get("stage"), 2)
    _append_exact_error(
        errors,
        "DeepSpeed bf16.enabled",
        _mapping(deepspeed.get("bf16")).get("enabled"),
        True,
    )
    _append_exact_error(
        errors,
        "DeepSpeed fp16.enabled",
        _mapping(deepspeed.get("fp16")).get("enabled"),
        False,
    )

    keyframes = semantic.get("keyframe_indices")
    chunk = train_data.get("chunk")
    if (
        type(keyframes) is list
        and keyframes
        and all(type(index) is int for index in keyframes)
        and type(chunk) is int
        and max(keyframes) >= chunk
    ):
        errors.append("semantic keyframes exceed the future clip")
    if train_data.get("stat_file") != val_data.get("stat_file"):
        errors.append("train and validation must use the same normalization statistics")
    return errors


def _validate_grounded_training_config(
    config: dict[str, Any],
    *,
    semantic: dict[str, Any],
    model_config: dict[str, Any],
    train_data: dict[str, Any],
    val_data: dict[str, Any],
    world_size: int,
    errors: list[str],
) -> None:
    """Validate the Task-12 joint contract without weakening old preflight."""

    expected_top_level = {
        "model_name": "ltx_train",
        "is_i2v": True,
        "return_action": True,
        "return_video": True,
        "train_mode": "all",
        "train_steps": 30_000,
        "steps_to_save": 5_000,
        "mixed_precision": "bf16",
        "allow_tf32": True,
        "gradient_checkpointing": True,
        "add_state": False,
        "noisy_video": False,
        "load_weights": True,
        "use_deepspeed": True,
        "lr": 2e-5,
        "action_lr": 1e-4,
        "semantic_lr": 5e-5,
        "qwen_top_lr": 1e-6,
        "qwen_vision_lr": 5e-7,
        "planner_aux_weight": 0.25,
        "qwen_ge_gradient_scale": 0.1,
    }
    for field, expected in expected_top_level.items():
        _append_exact_error(errors, field, config.get(field), expected)
    for field, expected in {
        "enabled": True,
        "source": "qwen35_grounded",
        "num_keyframes": 4,
        "tokens_per_frame": 96,
        "keyframe_indices": [0, 3, 5, 8],
        "dropout": 0.15,
        "validation_mode": "planner",
    }.items():
        _append_exact_error(
            errors,
            f"semantic_plan.{field}",
            semantic.get(field),
            expected,
        )
    planner_checkpoint = semantic.get("planner_checkpoint")
    if type(planner_checkpoint) is not str or not planner_checkpoint:
        errors.append("semantic_plan.planner_checkpoint must be a non-empty path")

    model_contract = {
        "action_expert": True,
        "action_in_channels": 7,
        "action_out_channels": 7,
        "semantic_plan_context": True,
        "semantic_plan_in_dim": 2048,
        "semantic_plan_num_keyframes": 4,
        "semantic_plan_num_views": 2,
        "semantic_plan_cross_attention_blocks": list(range(28)),
        "num_layers": 28,
    }
    for field, expected in model_contract.items():
        _append_exact_error(
            errors,
            f"diffusion {field}",
            model_config.get(field),
            expected,
        )

    if type(world_size) is not int or type(world_size) is bool or world_size != 8:
        errors.append(f"grounded training world_size must be 8, got {world_size!r}")
    batch_size = config.get("batch_size")
    accumulation = config.get("gradient_accumulation_steps")
    if (
        type(batch_size) is int
        and type(batch_size) is not bool
        and type(accumulation) is int
        and type(accumulation) is not bool
        and batch_size > 0
        and accumulation > 0
        and type(world_size) is int
    ):
        global_batch = batch_size * accumulation * world_size
        if global_batch != 128:
            errors.append(f"global batch must be 128, got {global_batch}")
    else:
        errors.append(
            "batch_size and gradient_accumulation_steps must be positive integers"
        )

    deepspeed = _mapping(config.get("deepspeed"))
    zero = _mapping(deepspeed.get("zero_optimization"))
    _append_exact_error(errors, "DeepSpeed ZeRO stage", zero.get("stage"), 2)
    _append_exact_error(
        errors,
        "DeepSpeed bf16.enabled",
        _mapping(deepspeed.get("bf16")).get("enabled"),
        True,
    )
    _append_exact_error(
        errors,
        "DeepSpeed fp16.enabled",
        _mapping(deepspeed.get("fp16")).get("enabled"),
        False,
    )
    if train_data.get("stat_file") != val_data.get("stat_file"):
        errors.append(
            "train and validation must use the same normalization statistics"
        )
def _validate_baton_training_config(
    config: dict[str, Any],
    *,
    semantic: dict[str, Any],
    model_config: dict[str, Any],
    train_data: dict[str, Any],
    val_data: dict[str, Any],
    world_size: int,
    errors: list[str],
) -> None:
    """Validate only the two action+video Baton HDF5 curricula."""

    source = semantic.get("source")
    expected_steps = (
        20_000 if source == BATON_TEACHER_SOURCE else 30_000
    )
    expected_mode = (
        "teacher" if source == BATON_TEACHER_SOURCE else "prediction"
    )
    for field, expected in {
        "model_name": "ltx_train",
        "is_i2v": True,
        "return_action": True,
        "return_video": True,
        "train_mode": "all",
        "train_steps": expected_steps,
        "steps_to_save": 5_000,
        "mixed_precision": "bf16",
        "allow_tf32": True,
        "gradient_checkpointing": True,
        "add_state": False,
        "noisy_video": False,
        "load_weights": True,
        "use_deepspeed": True,
        "lr": 2e-5,
        "action_lr": 1e-4,
        "semantic_lr": 5e-5,
        "diffusion_model_class_path": (
            "models/ltx_models/transformer_ltx_multiview.py"
        ),
        "diffusion_model_class": "LTXVideoTransformer3DModel",
    }.items():
        _append_exact_error(errors, field, config.get(field), expected)
    for field, expected in {
        "enabled": True,
        "source": source,
        "tokens_per_frame": 256,
        "feature_dim": 1024,
        "keyframe_indices": [0, 3, 5, 8],
        "dropout": 0.15,
        "validation_mode": expected_mode,
        "validation_modes": [expected_mode, "semantic_disabled"],
    }.items():
        _append_exact_error(
            errors,
            f"semantic_plan.{field}",
            semantic.get(field),
            expected,
        )
    for field, expected in {
        "action_expert": True,
        "action_in_channels": 7,
        "action_out_channels": 7,
        "semantic_plan_context": True,
        "semantic_plan_in_dim": 1024,
        "semantic_plan_num_keyframes": 4,
        "semantic_plan_num_views": 2,
        "semantic_plan_cross_attention_blocks": list(range(28)),
        "num_layers": 28,
    }.items():
        _append_exact_error(
            errors,
            f"diffusion {field}",
            model_config.get(field),
            expected,
        )
    required_artifacts = (
        (
            "siglip2_model_path",
            "siglip2_config_hash",
            "siglip2_artifact_hash",
            "teacher_preprocessing_hash",
        )
        if source == BATON_TEACHER_SOURCE
        else (
            "planner_checkpoint",
            "expected_planner_topology",
            "qwen_model_path",
            "qwen_tokenizer_path",
            "qwen_processor_path",
            "siglip2_model_path",
        )
    )
    for field in required_artifacts:
        if type(semantic.get(field)) is not str or not semantic[field]:
            errors.append(f"semantic_plan.{field} must be a non-empty string")
    if source == BATON_PREDICTION_SOURCE:
        for field in ("stage2_init_checkpoint", "stage2_init_topology_hash"):
            if type(config.get(field)) is not str or not config[field]:
                errors.append(f"{field} must be a non-empty string")
    if type(world_size) is not int or type(world_size) is bool or world_size <= 0:
        errors.append("Baton training world_size must be a positive integer")
    batch_size = config.get("batch_size")
    accumulation = config.get("gradient_accumulation_steps")
    if all(
        type(value) is int and type(value) is not bool and value > 0
        for value in (batch_size, accumulation, world_size)
    ):
        global_batch = batch_size * accumulation * world_size
        if global_batch != 128:
            errors.append(f"global batch must be 128, got {global_batch}")
    else:
        errors.append(
            "batch_size and gradient_accumulation_steps must be positive integers"
        )
    deepspeed = _mapping(config.get("deepspeed"))
    zero = _mapping(deepspeed.get("zero_optimization"))
    _append_exact_error(errors, "DeepSpeed ZeRO stage", zero.get("stage"), 2)
    _append_exact_error(
        errors,
        "DeepSpeed bf16.enabled",
        _mapping(deepspeed.get("bf16")).get("enabled"),
        True,
    )
    _append_exact_error(
        errors,
        "DeepSpeed fp16.enabled",
        _mapping(deepspeed.get("fp16")).get("enabled"),
        False,
    )
    if train_data.get("stat_file") != val_data.get("stat_file"):
        errors.append(
            "train and validation must use the same normalization statistics"
        )
    sampling_contract = {
        "baton_sampling_algorithm": (
            "libero_fastwam_hdf5_stateless_sha256"
        ),
        "baton_sampling_version": 1,
        "baton_sampling_seed": config.get("seed"),
    }
    for split, split_data in (("train", train_data), ("val", val_data)):
        for field, expected in sampling_contract.items():
            _append_exact_error(
                errors,
                f"data.{split}.{field}",
                split_data.get(field),
                expected,
            )


def _resolve_ge_act_path(raw_path: str, ge_act_root: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ge_act_root / path


def _collect_path_errors(
    config: dict[str, Any],
    *,
    ge_act_root: Path,
    minimum_free_gb: float,
) -> list[str]:
    errors: list[str] = []
    missing_modules: set[str] = set()
    for module_name in REQUIRED_MODULES:
        if importlib.util.find_spec(module_name) is None:
            missing_modules.add(module_name)
            errors.append(f"missing Python module: {module_name}")

    data = _mapping(config.get("data"))
    train_data = _mapping(data.get("train"))
    val_data = _mapping(data.get("val"))
    manifest_path = train_data.get("manifest_path")
    if "h5py" not in missing_modules and type(manifest_path) is str and manifest_path:
        try:
            load_manifest(Path(manifest_path))
        except json.JSONDecodeError as error:
            errors.append(f"manifest JSON error at {manifest_path}: {error.msg}")
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            errors.append(f"invalid HDF5 manifest {manifest_path}: {error}")

    checked_stats: set[Path] = set()
    for split, split_data in (("train", train_data), ("val", val_data)):
        raw_stat = split_data.get("stat_file")
        if type(raw_stat) is not str or not raw_stat:
            continue
        stat_path = _resolve_ge_act_path(raw_stat, ge_act_root)
        if stat_path in checked_stats:
            continue
        checked_stats.add(stat_path)
        if not stat_path.is_file():
            errors.append(f"missing normalization statistics for {split}: {stat_path}")

    raw_ltx = config.get("pretrained_model_name_or_path")
    ltx_path = Path(raw_ltx) if type(raw_ltx) is str and raw_ltx else None
    if ltx_path is None or not ltx_path.is_dir():
        errors.append(f"missing LTX pretrained components: {raw_ltx}")
    else:
        for component in ("tokenizer", "text_encoder", "vae"):
            component_path = ltx_path / component
            if not component_path.is_dir():
                errors.append(f"LTX component directory is missing: {component_path}")

    semantic = _mapping(config.get("semantic_plan"))
    semantic_source = semantic.get("source")
    diffusion = _mapping(config.get("diffusion_model"))
    raw_diffusion = diffusion.get("model_path")
    diffusion_path = (
        Path(raw_diffusion) if type(raw_diffusion) is str and raw_diffusion else None
    )
    valid_diffusion_path = diffusion_path is not None and (
        diffusion_path.is_file()
        or (
            semantic_source == BATON_PREDICTION_SOURCE
            and diffusion_path.is_dir()
        )
    )
    if not valid_diffusion_path:
        errors.append(f"missing base diffusion checkpoint: {raw_diffusion}")

    if semantic_source == "qwen35_grounded":
        raw_cache = train_data.get("hindsight_cache")
        raw_planner = semantic.get("planner_checkpoint")
        cache = None
        try:
            from qwen35_planx.hashing import sha256_file
            from qwen35_planx.hindsight_schema import HindsightCache

            cache = HindsightCache.open(Path(str(raw_cache)))
            if (
                type(manifest_path) is str
                and cache.metadata.hdf5_manifest_hash
                != sha256_file(Path(manifest_path))
            ):
                errors.append("hindsight cache differs from HDF5 manifest")
            from ge_act.models.ltx_models.vlm_semantic_planner import (
                _validate_grounded_checkpoint_contract,
                validate_qwen35_model_config_runtime,
            )

            _validate_grounded_checkpoint_contract(
                Path(str(raw_planner)),
                hindsight_cache_hash=cache.cache_hash,
            )
            validate_qwen35_model_config_runtime(
                Path(str(raw_planner)) / "model_config"
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            errors.append(f"invalid grounded planner/cache: {error}")
        finally:
            if cache is not None:
                cache.close()
    else:
        raw_siglip = (
            semantic.get("siglip2_model_path")
            if semantic_source in BATON_SOURCES
            else semantic.get("model_name_or_path")
        )
        siglip_path = Path(raw_siglip) if type(raw_siglip) is str and raw_siglip else None
        if siglip_path is None or not siglip_path.exists():
            errors.append(f"missing SigLIP2 checkpoint: {raw_siglip}")
        elif siglip_path.is_dir() and not (
            list(siglip_path.glob("*.safetensors"))
            or (siglip_path / "pytorch_model.bin").is_file()
        ):
            errors.append(f"SigLIP2 checkpoint directory has no weights: {siglip_path}")
        if semantic_source == BATON_PREDICTION_SOURCE:
            raw_stage2 = config.get("stage2_init_checkpoint")
            stage2_path = (
                Path(raw_stage2)
                if type(raw_stage2) is str and raw_stage2
                else None
            )
            if stage2_path is None or not stage2_path.is_dir():
                errors.append(
                    f"missing Stage-2 GE-Act checkpoint: {raw_stage2}"
                )

    raw_output = config.get("output_dir")
    if type(raw_output) is not str or not raw_output:
        errors.append("output_dir must be a non-empty path")
        return errors
    output_parent = _nearest_existing_parent(Path(raw_output).parent)
    if not os.access(output_parent, os.W_OK):
        errors.append(f"output path is not writable: {output_parent}")
    else:
        try:
            free_gb = shutil.disk_usage(output_parent).free / 1024**3
        except OSError as error:
            errors.append(f"cannot inspect output filesystem {output_parent}: {error}")
        else:
            if free_gb < minimum_free_gb:
                errors.append(
                    f"output filesystem has only {free_gb:.1f} GiB free; "
                    f"require {minimum_free_gb:.1f} GiB"
                )
    return errors


def collect_hdf5_preflight_errors(
    config: dict[str, Any],
    *,
    world_size: int,
    check_paths: bool = True,
    ge_act_root: Path | None = None,
    minimum_free_gb: float = 100.0,
) -> list[str]:
    """Return all actionable HDF5 training preflight failures."""
    if type(config) is not dict:
        return ["config must be a mapping"]
    errors = _validate_training_config(config, world_size)
    if not check_paths:
        return errors
    errors.extend(
        _collect_path_errors(
            config,
            ge_act_root=Path(ge_act_root) if ge_act_root else GE_ACT_ROOT,
            minimum_free_gb=minimum_free_gb,
        )
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--minimum-free-gb", type=float, default=100.0)
    args = parser.parse_args()
    try:
        with args.config.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        print(f"GE-Act HDF5 preflight failed: cannot load config: {error}")
        return 1
    errors = collect_hdf5_preflight_errors(
        config,
        world_size=args.world_size,
        minimum_free_gb=args.minimum_free_gb,
    )
    if errors:
        print("GE-Act HDF5 preflight failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("GE-Act HDF5 preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
