"""Preflight checks for immutable grounded Plan-X artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Sequence

import torch
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
)

from qwen35_planx.official_ta_tok import inspect_released_checkpoint
from qwen35_planx.hashing import sha256_file, sha256_json


_MINIMUM_CODEBOOK_EXPORT_BYTES = 1024 * 1024 * 1024
_MINIMUM_HINDSIGHT_CACHE_BYTES = 1024 * 1024 * 1024
_SIGLIP_CONFIG_VALUES = {
    ("model_type",): "siglip",
    ("vision_config", "hidden_size"): 1152,
    ("vision_config", "intermediate_size"): 4304,
    ("vision_config", "num_attention_heads"): 16,
    ("vision_config", "num_hidden_layers"): 27,
    ("vision_config", "patch_size"): 14,
    ("vision_config", "image_size"): 384,
    ("text_config", "hidden_size"): 1152,
    ("text_config", "intermediate_size"): 4304,
    ("text_config", "num_attention_heads"): 16,
    ("text_config", "num_hidden_layers"): 27,
    ("text_config", "projection_size"): 1152,
    ("text_config", "vocab_size"): 256_000,
}
_WEIGHT_INDEX_NAMES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
_WEIGHT_FILE_NAMES = (
    "model.safetensors",
    "pytorch_model.bin",
)


def _nested_config_value(
    payload: Mapping[str, object], path: tuple[str, ...]
) -> object:
    value: object = payload
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"missing config field {'.'.join(path)}")
        value = value[part]
    return value


def _load_exact_siglip_config(model_path: Path) -> tuple[object, list[str]]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return None, [f"local SigLIP2 config.json is missing from: {model_path}"]
    if config_path.stat().st_size == 0:
        return None, [f"local SigLIP2 config.json is zero-byte: {config_path}"]
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, [f"local SigLIP2 config.json is invalid: {error}"]
    if not isinstance(payload, Mapping):
        return None, ["local SigLIP2 config.json must contain an object"]
    errors: list[str] = []
    for path, expected in _SIGLIP_CONFIG_VALUES.items():
        try:
            actual = _nested_config_value(payload, path)
        except ValueError as error:
            errors.append(f"local SigLIP2 {error}")
            continue
        if actual != expected:
            errors.append(
                f"local SigLIP2 {'.'.join(path)} must be {expected!r}, got {actual!r}"
            )
    if errors:
        return None, errors
    try:
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    except Exception as error:
        return None, [f"local SigLIP2 config cannot be parsed by transformers: {error}"]
    return config, []


def _expected_weight_shapes(config: object) -> dict[str, tuple[int, ...]]:
    with torch.device("meta"):
        model = AutoModel.from_config(config)
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


def _inventory_safetensors(path: Path) -> dict[str, tuple[int, ...]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {
            name: tuple(handle.get_slice(name).get_shape()) for name in handle.keys()
        }


def _inventory_torch_weights(path: Path) -> dict[str, tuple[int, ...]]:
    payload = torch.load(
        path,
        weights_only=True,
        map_location="cpu",
        mmap=True,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("PyTorch weight file must contain a state mapping")
    inventory: dict[str, tuple[int, ...]] = {}
    for name, value in payload.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("PyTorch weight state must map names directly to tensors")
        inventory[name] = tuple(value.shape)
    return inventory


def _inventory_weight_file(path: Path) -> dict[str, tuple[int, ...]]:
    if not path.is_file():
        raise FileNotFoundError(f"referenced weight shard is missing: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"weight file is zero-byte: {path}")
    if path.name.endswith(".safetensors"):
        return _inventory_safetensors(path)
    if path.name.endswith(".bin"):
        return _inventory_torch_weights(path)
    raise ValueError(f"unsupported weight shard type: {path}")


def _inventory_indexed_weights(
    model_path: Path,
    index_path: Path,
) -> dict[str, tuple[int, ...]]:
    if index_path.stat().st_size == 0:
        raise ValueError(f"weight index is zero-byte: {index_path}")
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"weight index is invalid: {error}") from error
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("weight_map"), Mapping
    ):
        raise ValueError("weight index is invalid: missing weight_map object")
    weight_map = payload["weight_map"]
    if not weight_map:
        raise ValueError("weight index is incomplete: weight_map is empty")
    grouped: dict[str, set[str]] = {}
    for name, filename in weight_map.items():
        if not isinstance(name, str) or not isinstance(filename, str):
            raise ValueError(
                "weight index is invalid: weight_map entries must be strings"
            )
        shard_path = model_path / filename
        if (
            Path(filename).is_absolute()
            or shard_path.resolve().parent != model_path.resolve()
        ):
            raise ValueError(f"weight index has unsafe shard path: {filename}")
        grouped.setdefault(filename, set()).add(name)
    inventory: dict[str, tuple[int, ...]] = {}
    for filename, declared_names in grouped.items():
        shard_path = model_path / filename
        shard_inventory = _inventory_weight_file(shard_path)
        missing_from_shard = sorted(declared_names - set(shard_inventory))
        if missing_from_shard:
            raise ValueError(
                f"weight index is invalid for {filename}; declared keys are missing: "
                + ", ".join(missing_from_shard[:5])
            )
        unexpected_in_shard = sorted(set(shard_inventory) - declared_names)
        if unexpected_in_shard:
            raise ValueError(
                f"weight index is invalid for {filename}; undeclared keys are present: "
                + ", ".join(unexpected_in_shard[:5])
            )
        inventory.update(shard_inventory)
    return inventory


def _validate_weight_inventory(
    inventory: Mapping[str, tuple[int, ...]],
    expected: Mapping[str, tuple[int, ...]],
) -> list[str]:
    missing = sorted(set(expected) - set(inventory))
    if missing:
        return [
            "local SigLIP2 weights are incomplete; missing keys: "
            + ", ".join(missing[:5])
        ]
    unexpected = sorted(set(inventory) - set(expected))
    if unexpected:
        return [
            "local SigLIP2 weights have unexpected keys: " + ", ".join(unexpected[:5])
        ]
    wrong_shapes = sorted(
        name for name in expected if inventory[name] != expected[name]
    )
    if wrong_shapes:
        name = wrong_shapes[0]
        return [
            f"local SigLIP2 weight {name} has shape {inventory[name]}, "
            f"expected {expected[name]}"
        ]
    return []


def _validate_local_siglip_model(model_path: Path) -> list[str]:
    if not model_path.exists():
        return [f"local SigLIP2 model path does not exist: {model_path}"]
    if not model_path.is_dir():
        return [f"local SigLIP2 model path must be a directory: {model_path}"]
    config, config_errors = _load_exact_siglip_config(model_path)
    if config_errors:
        return config_errors
    expected = _expected_weight_shapes(config)
    index_paths = [
        model_path / name
        for name in _WEIGHT_INDEX_NAMES
        if (model_path / name).is_file()
    ]
    weight_paths = [
        model_path / name
        for name in _WEIGHT_FILE_NAMES
        if (model_path / name).is_file()
    ]
    if len(index_paths) + len(weight_paths) == 0:
        if list(model_path.glob("model-*.safetensors")) or list(
            model_path.glob("pytorch_model-*.bin")
        ):
            return ["local SigLIP2 sharded weights are incomplete without an index"]
        return [f"local SigLIP2 model weights are missing from: {model_path}"]
    if len(index_paths) + len(weight_paths) > 1:
        return ["local SigLIP2 model has ambiguous duplicate weight artifacts"]
    try:
        inventory = (
            _inventory_indexed_weights(model_path, index_paths[0])
            if index_paths
            else _inventory_weight_file(weight_paths[0])
        )
    except Exception as error:
        return [f"local SigLIP2 weights are invalid: {error}"]
    return _validate_weight_inventory(inventory, expected)


def _validate_processor_artifacts(
    model_path: Path,
    *,
    label: str,
    requires_text: bool,
) -> list[str]:
    preprocessor = model_path / "preprocessor_config.json"
    if not preprocessor.is_file() or preprocessor.stat().st_size == 0:
        return [f"local {label} preprocessor_config.json is missing or empty"]
    try:
        payload = json.loads(preprocessor.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return [f"local {label} preprocessor_config.json is invalid: {error}"]
    if not isinstance(payload, Mapping):
        return [f"local {label} preprocessor_config.json must contain an object"]
    if not requires_text:
        loader = AutoImageProcessor
    else:
        tokenizer_config = model_path / "tokenizer_config.json"
        tokenizer_files = (
            model_path / "tokenizer.json",
            model_path / "tokenizer.model",
            model_path / "sentencepiece.bpe.model",
        )
        if (
            not tokenizer_config.is_file()
            or tokenizer_config.stat().st_size == 0
            or not any(
                path.is_file() and path.stat().st_size > 0
                for path in tokenizer_files
            )
        ):
            return [f"local {label} tokenizer artifacts are missing or empty"]
        loader = AutoProcessor
    try:
        loader.from_pretrained(model_path, local_files_only=True)
    except Exception as error:
        return [f"local {label} processor artifacts are invalid: {error}"]
    return []


def _validate_local_dinov3_model(model_path: Path) -> list[str]:
    if not model_path.exists():
        return [f"local DINOv3 model path does not exist: {model_path}"]
    if not model_path.is_dir():
        return [f"local DINOv3 model path must be a directory: {model_path}"]
    config_path = model_path / "config.json"
    if not config_path.is_file() or config_path.stat().st_size == 0:
        return [f"local DINOv3 config.json is missing or empty: {config_path}"]
    try:
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    except Exception as error:
        return [f"local DINOv3 config cannot be parsed by transformers: {error}"]
    if not str(getattr(config, "model_type", "")).startswith("dinov3"):
        return [
            "local DINOv3 config model_type must identify DINOv3, got "
            f"{getattr(config, 'model_type', None)!r}"
        ]
    artifacts = [
        model_path / name
        for name in (*_WEIGHT_INDEX_NAMES, *_WEIGHT_FILE_NAMES)
        if (model_path / name).is_file()
    ]
    if not artifacts:
        if list(model_path.glob("model-*.safetensors")) or list(
            model_path.glob("pytorch_model-*.bin")
        ):
            return ["local DINOv3 sharded weights are incomplete without an index"]
        return [f"local DINOv3 model weights are missing from: {model_path}"]
    if any(path.stat().st_size == 0 for path in artifacts):
        return ["local DINOv3 model contains a zero-byte weight artifact"]
    index_paths = [
        path for path in artifacts if path.name in _WEIGHT_INDEX_NAMES
    ]
    if len(artifacts) != 1:
        return ["local DINOv3 model has ambiguous duplicate weight artifacts"]
    try:
        inventory = (
            _inventory_indexed_weights(model_path, index_paths[0])
            if index_paths
            else _inventory_weight_file(artifacts[0])
        )
        expected = _expected_weight_shapes(config)
    except Exception as error:
        return [f"local DINOv3 weights are invalid: {error}"]
    errors = _validate_weight_inventory(inventory, expected)
    errors = [
        error.replace("local SigLIP2", "local DINOv3")
        for error in errors
    ]
    if errors:
        return errors
    return _validate_processor_artifacts(
        model_path,
        label="DINOv3",
        requires_text=False,
    )


def _validate_local_qwen_model(model_path: Path) -> list[str]:
    try:
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        with torch.device("meta"):
            model = AutoModelForImageTextToText.from_config(config)
        expected = {
            name: tuple(tensor.shape)
            for name, tensor in model.state_dict().items()
        }
        index_paths = [
            model_path / name
            for name in _WEIGHT_INDEX_NAMES
            if (model_path / name).is_file()
        ]
        weight_paths = [
            model_path / name
            for name in _WEIGHT_FILE_NAMES
            if (model_path / name).is_file()
        ]
        if len(index_paths) + len(weight_paths) != 1:
            return ["local Qwen weights must contain exactly one indexed or single artifact"]
        inventory = (
            _inventory_indexed_weights(model_path, index_paths[0])
            if index_paths
            else _inventory_weight_file(weight_paths[0])
        )
    except Exception as error:
        return [f"local Qwen weights are invalid: {error}"]
    errors = _validate_weight_inventory(inventory, expected)
    return [error.replace("local SigLIP2", "local Qwen") for error in errors]


def _existing_ancestor(path: Path) -> Path | None:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate if candidate.exists() else None


def collect_released_ta_preflight_errors(
    ta_checkpoint: Path | str,
    siglip_model: Path | str,
    output_dir: Path | str = ".",
    *,
    minimum_free_bytes: int = _MINIMUM_CODEBOOK_EXPORT_BYTES,
) -> list[str]:
    """Return all actionable released-TA-Tok preflight errors."""

    errors: list[str] = []
    checkpoint_path = Path(ta_checkpoint)
    model_path = Path(siglip_model)
    output_path = Path(output_dir)

    if not checkpoint_path.is_file():
        errors.append(f"released TA-Tok checkpoint does not exist: {checkpoint_path}")
    else:
        try:
            inspect_released_checkpoint(
                checkpoint_path,
                compute_state_hash=False,
            )
        except Exception as error:
            errors.append(f"released TA-Tok checkpoint failed safe validation: {error}")

    errors.extend(_validate_local_siglip_model(model_path))

    existing_output = _existing_ancestor(output_path)
    if existing_output is None:
        errors.append(f"no existing parent for output directory: {output_path}")
    else:
        if not os.access(existing_output, os.W_OK):
            errors.append(f"output directory is not writable: {existing_output}")
        free_bytes = shutil.disk_usage(existing_output).free
        if free_bytes < minimum_free_bytes:
            errors.append(
                "insufficient free output space: "
                f"need at least {minimum_free_bytes} bytes, found {free_bytes}"
            )
    return errors


def collect_hindsight_cache_preflight_errors(
    *,
    hdf5_manifest: Path | str,
    window_manifest: Path | str,
    ta_checkpoint: Path | str,
    siglip_model: Path | str,
    dinov3_model: Path | str,
    output_dir: Path | str,
    minimum_free_bytes: int = _MINIMUM_HINDSIGHT_CACHE_BYTES,
    require_new_output: bool = True,
) -> list[str]:
    """Validate every local cache-build input without allocating teachers."""

    errors: list[str] = []
    hdf5_path = Path(hdf5_manifest)
    windows_path = Path(window_manifest)
    checkpoint_path = Path(ta_checkpoint)
    output_path = Path(output_dir)
    episode_lookup: dict[str, object] = {}

    if not hdf5_path.is_file():
        errors.append(f"HDF5 manifest does not exist: {hdf5_path}")
    else:
        try:
            from ge_act.data.libero_fastwam_hdf5_schema import load_manifest

            _, episodes = load_manifest(hdf5_path)
            episode_lookup = {episode.key: episode for episode in episodes}
        except Exception as error:
            errors.append(f"HDF5 manifest failed safe validation: {error}")
            episode_lookup = {}

    if not windows_path.is_file():
        errors.append(f"window manifest does not exist: {windows_path}")
    else:
        try:
            from qwen35_planx.cli.build_hindsight_cache import load_window_records

            windows = load_window_records(
                windows_path,
                expected_hdf5_manifest=hdf5_path,
            )
            if episode_lookup:
                for window in windows:
                    episode = episode_lookup.get(window.episode_key)
                    if episode is None:
                        errors.append(
                            "window manifest references unknown HDF5 episode: "
                            f"{window.episode_key}"
                        )
                        continue
                    if window.caption != episode.caption:
                        errors.append(
                            "window caption does not match HDF5 episode: "
                            f"{window.episode_key}"
                        )
                    indices = (
                        window.current_index,
                        *window.future_indices,
                        *window.frame_indices,
                        *window.action_indices,
                    )
                    if any(index < 0 or index >= episode.length for index in indices):
                        errors.append(
                            "window indices exceed complete trajectory bounds "
                            f"for {window.sample_id}: [0, {episode.length})"
                        )
                unknown = sorted(
                    window.episode_key
                    for window in windows
                    if window.episode_key not in episode_lookup
                )
                if unknown and not any(
                    "unknown HDF5 episode" in error for error in errors
                ):
                    errors.append(
                        "window manifest references unknown HDF5 episodes: "
                        + ", ".join(unknown[:5])
                    )
        except Exception as error:
            errors.append(f"window manifest failed safe validation: {error}")

    if episode_lookup:
        from qwen35_planx.hindsight_data import read_full_trajectory

        for episode_key in sorted(episode_lookup):
            try:
                read_full_trajectory(episode_lookup[episode_key])
            except Exception as error:
                errors.append(
                    f"HDF5 episode {episode_key} failed complete validation: {error}"
                )

    if not checkpoint_path.is_file():
        errors.append(f"released TA-Tok checkpoint does not exist: {checkpoint_path}")
    else:
        try:
            inspect_released_checkpoint(
                checkpoint_path,
                compute_state_hash=False,
            )
        except Exception as error:
            errors.append(f"released TA-Tok checkpoint failed safe validation: {error}")

    siglip_path = Path(siglip_model)
    errors.extend(_validate_local_siglip_model(siglip_path))
    if siglip_path.is_dir():
        errors.extend(
            _validate_processor_artifacts(
                siglip_path,
                label="SigLIP2",
                requires_text=True,
            )
        )
    errors.extend(_validate_local_dinov3_model(Path(dinov3_model)))

    if require_new_output and output_path.exists():
        errors.append(f"output directory already exists: {output_path}")
    existing_output = _existing_ancestor(output_path)
    if existing_output is None:
        errors.append(f"no existing parent for output directory: {output_path}")
    else:
        if not os.access(existing_output, os.W_OK):
            errors.append(f"output directory is not writable: {existing_output}")
        free_bytes = shutil.disk_usage(existing_output).free
        if free_bytes < minimum_free_bytes:
            errors.append(
                "insufficient free hindsight-cache space: "
                f"need at least {minimum_free_bytes} bytes, found {free_bytes}"
            )
    return errors


def _artifact_directory_hash(path: Path) -> str:
    entries = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        entries.append(
            (
                child.relative_to(path).as_posix(),
                child.stat().st_size,
                sha256_file(child),
            )
        )
    if not entries:
        raise ValueError(f"artifact directory is empty: {path}")
    return sha256_json(entries)


def _load_codebook_export_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"TA codebook metadata does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"TA codebook metadata is invalid: {error}") from error
    required = {
        "format_version",
        "checkpoint_sha256",
        "state_sha256",
        "artifact_sha256",
        "geometry",
        "teacher",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("TA codebook metadata fields are invalid")
    geometry = payload["geometry"]
    teacher = payload["teacher"]
    if (
        payload["format_version"] != 1
        or not isinstance(geometry, Mapping)
        or geometry.get("visual_vocab_size") != 65_536
        or geometry.get("ta_code_dim") != 1_536
        or geometry.get("tokens_per_frame") != 729
        or not isinstance(teacher, Mapping)
        or teacher.get("checkpoint_hash") != payload["checkpoint_sha256"]
        or teacher.get("image_size") != 384
        or teacher.get("grid_size") != 27
    ):
        raise ValueError("TA codebook metadata geometry/hash contract is invalid")
    for name in ("checkpoint_sha256", "state_sha256", "artifact_sha256"):
        if not isinstance(payload[name], str) or not payload[name]:
            raise ValueError(f"TA codebook metadata {name} must be nonempty")
    return payload


def _validate_codebook_export(
    tensor_path: Path,
    metadata_path: Path,
) -> dict[str, object]:
    metadata = _load_codebook_export_metadata(metadata_path)
    if not tensor_path.is_file():
        raise FileNotFoundError(f"TA codebook safetensors does not exist: {tensor_path}")
    if tensor_path.stat().st_size == 0:
        raise ValueError("TA codebook safetensors is zero-byte")
    if sha256_file(tensor_path) != metadata["artifact_sha256"]:
        raise ValueError("TA codebook artifact SHA-256 mismatch")
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        if list(handle.keys()) != ["codebook"]:
            raise ValueError("TA codebook safetensors must contain only codebook")
        codebook = handle.get_tensor("codebook")
        shape = tuple(codebook.shape)
    if not bool(torch.isfinite(codebook).all()):
        raise ValueError("TA codebook contains non-finite values")
    if shape != (65_536, 1_536):
        raise ValueError(
            f"TA codebook shape must be (65536, 1536), got {shape}"
        )
    return {
        "checkpoint_hash": metadata["checkpoint_sha256"],
        "state_hash": metadata["state_sha256"],
        "artifact_hash": sha256_file(tensor_path),
        "shape": shape,
    }


def _estimate_sample_bytes(base_config: Mapping[str, object]) -> int:
    text_config = base_config.get("text_config")
    values = text_config if isinstance(text_config, Mapping) else base_config
    hidden = int(values.get("hidden_size", 2_048))
    layers = int(values.get("num_hidden_layers", 32))
    sequence = 3_200
    bytes_per_value = 2
    recompute_factor = 12
    camera_views = 2
    return (
        sequence
        * hidden
        * layers
        * bytes_per_value
        * recompute_factor
        * camera_views
    )


def planner_training_preflight_report(
    config: object,
    *,
    num_processes: int | None = None,
) -> dict[str, object]:
    """Load one real sample and validate all stage-one artifact identities."""

    from qwen35_planx.cli.train_semantic_planner import (
        PlannerTrainingConfig,
        estimate_per_gpu_batch_candidates,
        validate_planner_checkpoint,
    )

    if isinstance(config, Mapping):
        config = PlannerTrainingConfig.from_mapping(config)
    if not isinstance(config, PlannerTrainingConfig):
        raise TypeError("config must be a PlannerTrainingConfig or mapping")
    if config.resume_from is not None:
        validate_planner_checkpoint(
            config.resume_from,
            allow_test_artifacts=config.tiny_smoke,
        )
    if config.tiny_smoke:
        return {
            "tiny_smoke": True,
            "sample_shapes": None,
            "batch_candidates": (
                (config.per_device_batch, config.gradient_accumulation_steps),
            ),
        }

    from qwen35_planx.hindsight_schema import HindsightCache
    from qwen35_planx.planner_dataset import (
        CachedPlannerTargets,
        GroundedPlannerBatch,
        GroundedPlannerCollator,
        HindsightPlannerDataset,
    )
    from qwen35_planx.vocabulary import (
        STRUCTURE_TOKENS,
        VisualVocabularyLayout,
    )
    from transformers import AutoProcessor, AutoTokenizer

    base_model = Path(str(config.base_model)).resolve()
    if not base_model.is_dir():
        raise FileNotFoundError(f"base Qwen directory does not exist: {base_model}")
    base_config_path = base_model / "config.json"
    if not base_config_path.is_file():
        raise FileNotFoundError(f"base Qwen config.json is missing: {base_config_path}")
    try:
        base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"base Qwen config.json is invalid: {error}") from error
    if (
        not isinstance(base_config, Mapping)
        or base_config.get("model_type") != "qwen3_5"
    ):
        raise ValueError("base Qwen config model_type must be 'qwen3_5'")
    qwen_errors = _validate_local_qwen_model(base_model)
    if qwen_errors:
        raise ValueError("; ".join(qwen_errors))
    processor_errors = _validate_processor_artifacts(
        base_model,
        label="Qwen3.5",
        requires_text=True,
    )
    if processor_errors:
        raise ValueError("; ".join(processor_errors))
    base_model_hash = _artifact_directory_hash(base_model)

    output_dir = Path(config.output_dir).resolve()
    codebook_path = Path(str(config.ta_codebook)).resolve()
    for label, protected in (
        ("base Qwen", base_model),
        ("released TA-Tok/codebook", codebook_path.parent),
    ):
        if output_dir == protected or protected in output_dir.parents:
            raise ValueError(
                f"checkpoint output must not be inside the {label} directory"
            )
    codebook_report = _validate_codebook_export(
        codebook_path,
        Path(str(config.ta_codebook_metadata)).resolve(),
    )
    cache_dir = Path(str(config.hindsight_cache)).resolve()
    hdf5_manifest = Path(str(config.hdf5_manifest)).resolve()
    with HindsightCache.open(cache_dir) as cache:
        dataset = HindsightPlannerDataset(cache, hdf5_manifest)
        sample = dataset[0]
        targets = CachedPlannerTargets(
            codes=sample["codes"].unsqueeze(0).long(),
            relevance=sample["relevance"].unsqueeze(0),
            relevance_confidence=sample["relevance_confidence"].unsqueeze(0),
            flow=sample["flow"].unsqueeze(0),
            phrase_embeddings=sample["phrase_embeddings"].unsqueeze(0),
        )
        if tuple(sample["current_images"].shape) != (2, 3, 256, 256):
            raise ValueError("HDF5 current images must have shape [2,3,256,256]")
        splits = {record.split for record in cache.records}
        if not {"train", "val"}.issubset(splits):
            raise ValueError("hindsight cache must contain train and val records")
        sample_shapes = {
            "current_images": tuple(sample["current_images"].shape),
            "codes": tuple(targets.codes.shape),
            "relevance": tuple(targets.relevance.shape),
            "flow": tuple(targets.flow.shape),
            "phrase_embeddings": tuple(targets.phrase_embeddings.shape),
        }
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, local_files_only=True
        )
        processor = AutoProcessor.from_pretrained(
            base_model, local_files_only=True
        )
        original_vocab_size = len(tokenizer)
        visual_tokens = [
            f"<|ta_{index:05d}|>" for index in range(65_536)
        ]
        added = tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    *STRUCTURE_TOKENS,
                    *visual_tokens,
                ]
            }
        )
        if added != len(STRUCTURE_TOKENS) + 65_536:
            raise ValueError("Qwen tokenizer visual expansion is not exact")
        structure_ids = tuple(
            (token, int(tokenizer.convert_tokens_to_ids(token)))
            for token in STRUCTURE_TOKENS
        )
        visual_start = int(tokenizer.convert_tokens_to_ids(visual_tokens[0]))
        layout = VisualVocabularyLayout(
            original_vocab_size=original_vocab_size,
            visual_start_id=visual_start,
            visual_end_id=visual_start + 65_536,
            structure_token_ids=structure_ids,
            tokenizer_hash=sha256_json(
                sorted(
                    (str(token), int(token_id))
                    for token, token_id in tokenizer.get_vocab().items()
                )
            ),
            base_embedding_hash="preflight-shape-only",
            expanded_embedding_hash="preflight-shape-only",
        )
        processor.tokenizer = tokenizer
        collator = GroundedPlannerCollator(
            processor,
            layout,
            cache_dir=cache_dir,
            dataset=dataset,
        )
        cpu_batch = collator([sample])
        if not isinstance(cpu_batch, GroundedPlannerBatch) or any(
            tensor.device.type != "cpu"
            for tensor in cpu_batch.qwen_inputs.values()
        ):
            raise ValueError("preflight collator did not produce a CPU planner batch")
        sample_shapes["qwen_input_ids"] = tuple(
            cpu_batch.qwen_inputs["input_ids"].shape
        )
        cache_hash = cache.cache_hash
        cache_ta_hash = cache.metadata.ta_tok_hash
    if cache_ta_hash != codebook_report["checkpoint_hash"]:
        raise ValueError("hindsight cache TA-Tok hash differs from codebook export")

    if config.resume_from is not None:
        resume_metadata, _ = validate_planner_checkpoint(config.resume_from)
        expected_hashes = {
            "base_model_hash": base_model_hash,
            "hindsight_cache_hash": cache_hash,
            "ta_tok_hash": codebook_report["checkpoint_hash"],
        }
        for name, expected in expected_hashes.items():
            if resume_metadata[name] != expected:
                raise ValueError(
                    f"resume checkpoint {name} mismatch: "
                    f"expected {expected!r}, got {resume_metadata[name]!r}"
                )

    if num_processes is None:
        num_processes = int(
            os.environ.get(
                "WORLD_SIZE",
                max(1, torch.cuda.device_count()),
            )
        )
    if torch.cuda.is_available():
        free_bytes = min(
            torch.cuda.mem_get_info(index)[0]
            for index in range(torch.cuda.device_count())
        )
        batch_candidates = estimate_per_gpu_batch_candidates(
            num_processes=num_processes,
            available_bytes=int(free_bytes),
            estimated_bytes_per_sample=_estimate_sample_bytes(base_config),
        )
    else:
        batch_candidates = ()
    return {
        "tiny_smoke": False,
        "base_model_hash": base_model_hash,
        "hindsight_cache_hash": cache_hash,
        "ta_codebook": codebook_report,
        "sample_shapes": sample_shapes,
        "batch_candidates": batch_candidates,
    }


def collect_planner_training_preflight_errors(
    config: object,
    *,
    num_processes: int | None = None,
) -> list[str]:
    """Return a single actionable fail-closed stage-one preflight error."""

    try:
        planner_training_preflight_report(
            config,
            num_processes=num_processes,
        )
    except Exception as error:
        return [f"planner training preflight failed: {error}"]
    return []


def _released_ta_command(arguments: argparse.Namespace) -> int:
    errors = collect_released_ta_preflight_errors(
        arguments.ta_checkpoint,
        arguments.siglip_model,
        arguments.output_dir,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    inspection = inspect_released_checkpoint(
        arguments.ta_checkpoint,
        compute_state_hash=False,
    )
    print(
        "released TA-Tok preflight OK: "
        f"shape={inspection.shape} "
        f"checkpoint_sha256={inspection.checkpoint_hash}"
    )
    return 0


def _hindsight_cache_command(arguments: argparse.Namespace) -> int:
    errors = collect_hindsight_cache_preflight_errors(
        hdf5_manifest=arguments.hdf5_manifest,
        window_manifest=arguments.window_manifest,
        ta_checkpoint=arguments.ta_checkpoint,
        siglip_model=arguments.siglip_model,
        dinov3_model=arguments.dinov3_model,
        output_dir=arguments.output_dir,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print("hindsight-cache preflight OK: all artifacts are local and validated")
    return 0


def _planner_training_command(arguments: argparse.Namespace) -> int:
    try:
        payload = json.loads(arguments.config.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("config must contain an object")
        from qwen35_planx.cli.train_semantic_planner import PlannerTrainingConfig

        config = PlannerTrainingConfig.from_mapping(payload)
        if arguments.resume_from is not None:
            from dataclasses import replace

            config = replace(config, resume_from=str(arguments.resume_from))
        report = planner_training_preflight_report(
            config,
            num_processes=arguments.num_processes,
        )
    except Exception as error:
        print(f"ERROR: planner training preflight failed: {error}", file=sys.stderr)
        return 2
    print(
        "planner-training preflight OK: "
        f"sample_shapes={report['sample_shapes']} "
        f"batch_candidates={report['batch_candidates']}"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    released = subparsers.add_parser(
        "released-ta",
        help="validate the released 384px TA-Tok and local SigLIP2 artifacts",
    )
    released.add_argument("--ta-checkpoint", type=Path, required=True)
    released.add_argument("--siglip-model", type=Path, required=True)
    released.add_argument("--output-dir", type=Path, default=Path("."))
    released.set_defaults(handler=_released_ta_command)
    hindsight = subparsers.add_parser(
        "hindsight-cache",
        help="validate local teachers, HDF5 windows, and cache output capacity",
    )
    hindsight.add_argument("--hdf5-manifest", type=Path, required=True)
    hindsight.add_argument("--window-manifest", type=Path, required=True)
    hindsight.add_argument("--ta-checkpoint", type=Path, required=True)
    hindsight.add_argument("--siglip-model", type=Path, required=True)
    hindsight.add_argument("--dinov3-model", type=Path, required=True)
    hindsight.add_argument("--output-dir", type=Path, required=True)
    hindsight.set_defaults(handler=_hindsight_cache_command)
    training = subparsers.add_parser(
        "planner-training",
        help="validate real HDF5/cache/Qwen/codebook inputs and resume state",
    )
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--resume-from", type=Path)
    training.add_argument("--num-processes", type=int)
    training.set_defaults(handler=_planner_training_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
