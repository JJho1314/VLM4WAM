#!/usr/bin/env python3
"""Preflight validation for eight-GPU LTX + SigLIP2 training."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any

import yaml


REQUIRED_MODULES = ("torch", "diffusers", "transformers", "accelerate", "av", "safetensors")
BATON_TEACHER_SOURCE = "qwen35_baton_teacher"
BATON_PREDICTION_SOURCE = "qwen35_baton_prediction"
BATON_SOURCES = {BATON_TEACHER_SOURCE, BATON_PREDICTION_SOURCE}


def _is_concrete_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and value != "0" * 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _nearest_existing_parent(path: Path) -> Path:
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def materialize_baton_config(
    template: dict[str, Any],
    environ: dict[str, str],
) -> dict[str, Any]:
    """Resolve one immutable Baton recipe from explicit deployment inputs."""

    if type(template) is not dict or type(environ) is not dict:
        raise TypeError("Baton template and environment must be mappings")
    resolved = deepcopy(template)
    semantic = resolved.get("semantic_plan")
    if type(semantic) is not dict:
        raise ValueError("Baton template semantic_plan must be a mapping")
    source = semantic.get("source")
    if source not in BATON_SOURCES:
        raise ValueError("materialization requires a Baton semantic source")

    def require(name: str) -> str:
        value = environ.get(name)
        if type(value) is not str or not value.strip():
            raise ValueError(f"required deployment variable is missing: {name}")
        return value.strip()

    resolved["pretrained_model_name_or_path"] = require(
        "BATON_LTX_PRETRAINED_PATH"
    )
    manifest = require("BATON_HDF5_MANIFEST_PATH")
    stat_file = require("BATON_STAT_FILE")
    resolved["output_dir"] = require("BATON_OUTPUT_DIR")
    for split in ("train", "val"):
        resolved["data"][split]["manifest_path"] = manifest
        resolved["data"][split]["stat_file"] = stat_file
    if "BATON_PER_DEVICE_BATCH" in environ:
        resolved["batch_size"] = int(require("BATON_PER_DEVICE_BATCH"))
    if "BATON_GRADIENT_ACCUMULATION_STEPS" in environ:
        resolved["gradient_accumulation_steps"] = int(
            require("BATON_GRADIENT_ACCUMULATION_STEPS")
        )
    resume_from = environ.get("BATON_RESUME_FROM_CHECKPOINT", "").strip()
    resolved["resume_from_checkpoint"] = resume_from or None

    semantic["siglip2_model_path"] = require("BATON_SIGLIP2_MODEL_PATH")
    if source == BATON_TEACHER_SOURCE:
        resolved["diffusion_model"]["model_path"] = require(
            "BATON_GE_BASE_CHECKPOINT"
        )
        semantic["siglip2_config_hash"] = require(
            "BATON_SIGLIP2_CONFIG_HASH"
        )
        semantic["siglip2_artifact_hash"] = require(
            "BATON_SIGLIP2_ARTIFACT_HASH"
        )
        semantic["teacher_preprocessing_hash"] = require(
            "BATON_TEACHER_PREPROCESSING_HASH"
        )
    else:
        stage2_checkpoint = require("BATON_STAGE2_INIT_CHECKPOINT")
        resolved["stage2_init_checkpoint"] = stage2_checkpoint
        resolved["stage2_init_topology_hash"] = require(
            "BATON_STAGE2_INIT_TOPOLOGY_HASH"
        )
        resolved["diffusion_model"]["model_path"] = str(
            Path(stage2_checkpoint) / "diffusion_model"
        )
        semantic["planner_checkpoint"] = require(
            "BATON_PLANNER_CHECKPOINT"
        )
        semantic["expected_planner_topology"] = require(
            "BATON_PLANNER_TOPOLOGY"
        )
        semantic["qwen_model_path"] = require("BATON_QWEN_MODEL_PATH")
        semantic["qwen_tokenizer_path"] = require(
            "BATON_QWEN_TOKENIZER_PATH"
        )
        semantic["qwen_processor_path"] = require(
            "BATON_QWEN_PROCESSOR_PATH"
        )
    return resolved


def collect_preflight_errors(
    config: dict[str, Any],
    *,
    world_size: int,
    check_paths: bool = True,
    ge_act_root: Path | None = None,
    minimum_free_gb: float = 100.0,
    per_device_batch: int | None = None,
    gradient_accumulation_steps: int | None = None,
) -> list[str]:
    errors: list[str] = []
    ge_act_root = ge_act_root or Path(__file__).resolve().parents[1]

    semantic = config.get("semantic_plan", {})
    model_config = config.get("diffusion_model", {}).get("config", {})
    train_data = config.get("data", {}).get("train", {})
    val_data = config.get("data", {}).get("val", {})
    keyframes = semantic.get("keyframe_indices", [])
    semantic_source = semantic.get("source", "gt_siglip2")
    hdf5_backend = config.get("train_data_class") == "LiberoFastWAMHDF5Dataset"
    if not semantic.get("enabled", False):
        errors.append("semantic_plan.enabled must be true")
    if semantic_source in BATON_SOURCES:
        expected_mode = (
            "teacher"
            if semantic_source == BATON_TEACHER_SOURCE
            else "prediction"
        )
        if keyframes != [0, 3, 5, 8]:
            errors.append("Baton semantic keyframes must be [0, 3, 5, 8]")
        if semantic.get("validation_mode") != expected_mode:
            errors.append(
                f"{semantic_source} validation_mode must be {expected_mode}"
            )
        if semantic.get("validation_modes") != [
            expected_mode,
            "semantic_disabled",
        ]:
            errors.append(
                f"{semantic_source} validation_modes must include "
                f"{expected_mode} and semantic_disabled"
            )
        if semantic.get("dropout") != 0.15:
            errors.append("Baton semantic dropout must be 0.15")
        forbidden = {
            "hindsight_cache",
            "planner_aux_loss",
            "planner_aux_weight",
            "qwen_ge_gradient_scale",
            "relevance",
            "semantic_plan_relevance",
            "mask",
            "semantic_plan_mask",
        }
        present = sorted(
            field
            for field in forbidden
            if config.get(field) is not None or semantic.get(field) is not None
        )
        for split in ("train", "val"):
            split_data = config.get("data", {}).get(split, {})
            present.extend(
                f"data.{split}.{field}"
                for field in forbidden
                if split_data.get(field) is not None
            )
        if present:
            errors.append(
                "Baton configs reject cache/auxiliary/relevance/mask fields: "
                + ", ".join(present)
            )
        if semantic_source == BATON_TEACHER_SOURCE:
            for field in (
                "planner_checkpoint",
                "expected_planner_topology",
                "qwen_model_path",
                "qwen_tokenizer_path",
                "qwen_processor_path",
            ):
                if semantic.get(field) is not None:
                    errors.append(
                        f"Baton teacher source rejects planner field {field}"
                    )
            for field in (
                "siglip2_model_path",
                "siglip2_config_hash",
                "siglip2_artifact_hash",
                "teacher_preprocessing_hash",
            ):
                if not semantic.get(field):
                    errors.append(f"semantic_plan.{field} is required")
            for field in (
                "siglip2_config_hash",
                "siglip2_artifact_hash",
                "teacher_preprocessing_hash",
            ):
                if semantic.get(field) and not _is_concrete_sha256(
                    semantic[field]
                ):
                    errors.append(
                        f"semantic_plan.{field} must be a concrete nonzero "
                        "lowercase SHA-256, not a placeholder"
                    )
        else:
            if semantic.get("frame_microbatch_size") is not None:
                errors.append(
                    "Baton prediction source rejects teacher field "
                    "frame_microbatch_size"
                )
            for field in (
                "planner_checkpoint",
                "expected_planner_topology",
                "qwen_model_path",
                "qwen_tokenizer_path",
                "qwen_processor_path",
                "siglip2_model_path",
            ):
                if not semantic.get(field):
                    errors.append(f"semantic_plan.{field} is required")
    elif semantic_source == "gt_siglip2":
        if keyframes != [0, 3, 5, 8]:
            errors.append("semantic keyframes must be [0, 3, 5, 8]")
        if semantic.get("validation_mode", "gt") != "gt":
            errors.append("GT SigLIP2 validation_mode must be gt")
        if not semantic.get("model_name_or_path"):
            errors.append("semantic_plan.model_name_or_path is required")
    elif semantic_source == "vlm_planner":
        if keyframes != [8]:
            errors.append("VLM planner semantic keyframes must be [8]")
        if semantic.get("validation_mode") != "planner":
            errors.append("VLM planner validation_mode must be planner")
        if not semantic.get("planner_checkpoint"):
            errors.append("semantic_plan.planner_checkpoint is required")
    else:
        errors.append(f"unknown semantic_plan.source: {semantic_source}")
    if semantic.get("tokens_per_frame") != 256:
        errors.append("SigLIP2 must provide 256 tokens per frame")
    if semantic.get("feature_dim") != 1024 or model_config.get("semantic_plan_in_dim") != 1024:
        errors.append("SigLIP2 feature width must be 1024")
    if model_config.get("semantic_plan_cross_attention_blocks") != list(range(28)):
        errors.append("semantic cross-attention must be enabled in all 28 LTX blocks")
    if model_config.get("semantic_plan_num_keyframes") != len(keyframes):
        errors.append("LTX semantic keyframe count must match semantic_plan.keyframe_indices")
    if model_config.get("semantic_plan_num_views") != 2:
        errors.append("LTX semantic plan must preserve two camera views")
    if semantic_source in BATON_SOURCES:
        if config.get("return_video") is not True:
            errors.append("Baton curricula must train video")
        if config.get("return_action") is not True:
            errors.append("Baton curricula must train action")
        if config.get("train_mode") != "all":
            errors.append("Baton curricula train_mode must be all")
        if model_config.get("action_expert") is not True:
            errors.append("Baton curricula require the action expert")
        expected_rates = {
            "lr": 2e-5,
            "action_lr": 1e-4,
            "semantic_lr": 5e-5,
        }
        for field, expected in expected_rates.items():
            if config.get(field) != expected:
                errors.append(f"{field} must be {expected}")
        if config.get("steps_to_save") != 5_000:
            errors.append("steps_to_save must be 5000")
        if semantic_source == BATON_PREDICTION_SOURCE:
            stage2_checkpoint = config.get("stage2_init_checkpoint")
            stage2_topology = config.get("stage2_init_topology_hash")
            if type(stage2_checkpoint) is not str or not stage2_checkpoint:
                errors.append("stage2_init_checkpoint is required")
            if not _is_concrete_sha256(stage2_topology):
                errors.append(
                    "stage2_init_topology_hash must be a concrete nonzero "
                    "lowercase SHA-256, not a placeholder"
                )
            expected_model_path = (
                str(Path(stage2_checkpoint) / "diffusion_model")
                if type(stage2_checkpoint) is str and stage2_checkpoint
                else None
            )
            if (
                config.get("diffusion_model", {}).get("model_path")
                != expected_model_path
            ):
                errors.append(
                    "Stage 3 diffusion model_path must be the validated "
                    "Stage-2 checkpoint diffusion_model directory"
                )
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
        if semantic_source in BATON_SOURCES:
            expected_sampling = {
                "baton_sampling_algorithm": (
                    "libero_fastwam_hdf5_stateless_sha256"
                ),
                "baton_sampling_version": 1,
                "baton_sampling_seed": config.get("seed"),
            }
            for split_name, split_data in (
                ("train", train_data),
                ("val", val_data),
            ):
                for field, expected in expected_sampling.items():
                    if split_data.get(field) != expected:
                        errors.append(
                            f"data.{split_name}.{field} must be {expected!r}"
                        )
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
    expected_steps = (
        20_000
        if semantic_source == BATON_TEACHER_SOURCE
        else 30_000
    )
    if config.get("train_steps") != expected_steps:
        errors.append(f"train_steps must be {expected_steps}")
    if not config.get("gradient_checkpointing", False):
        errors.append("gradient checkpointing must be enabled for the initial run")
    if (
        per_device_batch is not None
        and config.get("batch_size") != per_device_batch
    ):
        errors.append(
            "launcher per-device batch differs from config: "
            f"{per_device_batch} != {config.get('batch_size')}"
        )
    if (
        gradient_accumulation_steps is not None
        and config.get("gradient_accumulation_steps")
        != gradient_accumulation_steps
    ):
        errors.append(
            "launcher gradient accumulation differs from config: "
            f"{gradient_accumulation_steps} != "
            f"{config.get('gradient_accumulation_steps')}"
        )

    if not check_paths:
        return errors

    for module_name in REQUIRED_MODULES:
        if importlib.util.find_spec(module_name) is None:
            errors.append(f"missing Python module: {module_name}")

    required_paths = {
        "LTX pretrained components": config.get("pretrained_model_name_or_path"),
        "base diffusion checkpoint": config.get("diffusion_model", {}).get("model_path"),
    }
    if semantic_source == BATON_TEACHER_SOURCE:
        required_paths["SigLIP2 checkpoint"] = semantic.get(
            "siglip2_model_path"
        )
    elif semantic_source == BATON_PREDICTION_SOURCE:
        required_paths.update(
            {
                "Baton planner checkpoint": semantic.get(
                    "planner_checkpoint"
                ),
                "trusted planner topology": semantic.get(
                    "expected_planner_topology"
                ),
                "Qwen model": semantic.get("qwen_model_path"),
                "Qwen tokenizer": semantic.get("qwen_tokenizer_path"),
                "Qwen processor": semantic.get("qwen_processor_path"),
                "SigLIP2 checkpoint": semantic.get("siglip2_model_path"),
                "Stage-2 GE-Act checkpoint": config.get(
                    "stage2_init_checkpoint"
                ),
            }
        )
    elif semantic_source == "gt_siglip2":
        required_paths["SigLIP2 checkpoint"] = semantic.get("model_name_or_path")
    elif semantic_source == "vlm_planner":
        required_paths["dual-camera VLM planner"] = semantic.get("planner_checkpoint")
    for label, raw_path in required_paths.items():
        if not raw_path or not Path(raw_path).exists():
            errors.append(f"missing {label}: {raw_path}")
    if semantic_source in (BATON_TEACHER_SOURCE, BATON_PREDICTION_SOURCE):
        siglip_path = Path(semantic.get("siglip2_model_path", ""))
    else:
        siglip_path = Path(semantic.get("model_name_or_path", ""))
    if semantic_source in (
        "gt_siglip2",
        BATON_TEACHER_SOURCE,
        BATON_PREDICTION_SOURCE,
    ):
        if siglip_path.is_dir() and not (
            list(siglip_path.glob("*.safetensors"))
            or (siglip_path / "pytorch_model.bin").is_file()
        ):
            errors.append(f"SigLIP2 directory has no model weights: {siglip_path}")
    if semantic_source == BATON_TEACHER_SOURCE and siglip_path.is_dir():
        try:
            from qwen35_baton.cli.preflight import _siglip_geometry
            from qwen35_baton.hashing import sha256_artifact, sha256_file

            _siglip_geometry(siglip_path)
            config_hash = sha256_file(siglip_path / "config.json")
            artifact_hash = sha256_artifact(siglip_path)
            if config_hash != semantic.get("siglip2_config_hash"):
                errors.append("SigLIP2 config hash mismatch")
            if artifact_hash != semantic.get("siglip2_artifact_hash"):
                errors.append("SigLIP2 artifact hash mismatch")
            if artifact_hash != semantic.get("teacher_preprocessing_hash"):
                errors.append("SigLIP2 preprocessing hash mismatch")
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            errors.append(f"invalid local SigLIP2 artifact: {error}")
    if semantic_source == BATON_PREDICTION_SOURCE:
        try:
            from qwen35_baton.hashing import sha256_file
            from qwen35_baton.provider import (
                _validate_checkpoint_envelope,
                _validate_local_artifact_contract,
                _validate_siglip2_artifact_contract,
                _validate_trusted_planner_topology,
            )

            checkpoint = Path(semantic["planner_checkpoint"]).resolve()
            metadata = _validate_checkpoint_envelope(checkpoint)
            _validate_trusted_planner_topology(
                metadata,
                checkpoint=checkpoint,
                expected=semantic["expected_planner_topology"],
            )
            _validate_local_artifact_contract(
                metadata,
                qwen_model_path=Path(semantic["qwen_model_path"]).resolve(),
                qwen_tokenizer_path=Path(
                    semantic["qwen_tokenizer_path"]
                ).resolve(),
                qwen_processor_path=Path(
                    semantic["qwen_processor_path"]
                ).resolve(),
            )
            _validate_siglip2_artifact_contract(
                metadata,
                siglip2_model_path=Path(
                    semantic["siglip2_model_path"]
                ).resolve(),
            )
            manifest_path = train_data.get("manifest_path")
            if (
                isinstance(manifest_path, str)
                and Path(manifest_path).is_file()
                and metadata.hdf5_manifest_hash
                != sha256_file(Path(manifest_path))
            ):
                errors.append("Baton planner checkpoint differs from HDF5 manifest")
            from runner.ge_trainer import (
                validate_baton_stage3_artifact_chain,
            )

            artifact_chain = validate_baton_stage3_artifact_chain(
                stage2_checkpoint=config["stage2_init_checkpoint"],
                expected_stage2_topology_hash=config[
                    "stage2_init_topology_hash"
                ],
                planner_checkpoint=semantic["planner_checkpoint"],
                expected_planner_topology=semantic[
                    "expected_planner_topology"
                ],
                expected_hdf5_manifest_hash=(
                    sha256_file(Path(manifest_path))
                    if isinstance(manifest_path, str)
                    and Path(manifest_path).is_file()
                    else ""
                ),
            )
            if (
                Path(config["diffusion_model"]["model_path"])
                != artifact_chain.diffusion_model_dir
            ):
                errors.append(
                    "Stage-3 diffusion model path differs from validated "
                    "Stage-2 checkpoint"
                )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            errors.append(
                f"invalid Baton planner/Stage-2 provenance/topology: {error}"
            )
    ltx_path = Path(config.get("pretrained_model_name_or_path", ""))
    if ltx_path.is_dir():
        for component in ("tokenizer", "text_encoder", "vae"):
            if not (ltx_path / component).is_dir():
                errors.append(f"LTX component directory is missing: {ltx_path / component}")
    for data_root in sorted(set(train_data.get("data_roots", []))):
        if not Path(data_root).is_dir():
            errors.append(f"missing training data root: {data_root}")
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
    parser.add_argument("--per-device-batch", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--materialize-output", type=Path)
    args = parser.parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    if args.materialize_output is not None:
        try:
            resolved = materialize_baton_config(config, dict(os.environ))
            args.materialize_output.write_text(
                yaml.safe_dump(resolved, sort_keys=False),
                encoding="utf-8",
            )
            args.materialize_output.chmod(0o600)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
            print(f"GE-Act Baton config materialization failed: {error}")
            return 1
        print(f"GE-Act Baton config materialized: {args.materialize_output}")
        return 0
    errors = collect_preflight_errors(
        config,
        world_size=args.world_size,
        minimum_free_gb=args.minimum_free_gb,
        per_device_batch=args.per_device_batch,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
