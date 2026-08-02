"""CPU-only, fail-closed preflight for local Baton Stage-1 artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
import importlib
import json
from pathlib import Path
from typing import Any

from qwen35_baton.hashing import sha256_artifact, sha256_file
from qwen35_baton.sequence import ADDED_TOKENS


_WORLD_ARENA_SOURCE_REPOSITORY = "worldarena2026-robotwin-data"
_CAMERA_NAMES = {
    "libero_hdf5": ("main", "wrist"),
    "worldarena_hdf5": ("head",),
}


def require_qwen35_fast_path() -> dict[str, str]:
    """Reject the released Qwen3.5 slow fallback before loading model weights."""

    modules: dict[str, Any] = {}
    missing: list[str] = []
    for import_name, package_name in (
        ("fla", "flash-linear-attention"),
        ("causal_conv1d", "causal-conv1d"),
    ):
        try:
            modules[import_name] = importlib.import_module(import_name)
        except (ImportError, OSError):
            missing.append(package_name)
    causal_function = getattr(modules.get("causal_conv1d"), "causal_conv1d_fn", None)
    if "causal-conv1d" not in missing and not callable(causal_function):
        missing.append("causal-conv1d")
    if missing:
        raise RuntimeError(
            "Qwen3.5 fused fast path is unavailable. Install both "
            "flash-linear-attention[cuda] and causal-conv1d in the training "
            f"environment; failed: {', '.join(sorted(set(missing)))}"
        )
    return {
        name: str(getattr(module, "__version__", "installed"))
        for name, module in modules.items()
    }


def _load_json(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _require_directory(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {resolved}")
    return resolved


def _manifest_dataset_type(payload: Mapping[str, Any]) -> str:
    worldarena = (
        payload.get("version") == 1
        and payload.get("source_repository") == _WORLD_ARENA_SOURCE_REPOSITORY
        and isinstance(payload.get("records"), list)
    )
    libero = (
        payload.get("schema_version") == 1
        and payload.get("camera_names") == ["main", "wrist"]
        and isinstance(payload.get("episodes"), list)
    )
    matches = [
        dataset_type
        for dataset_type, matched in (
            ("libero_hdf5", libero),
            ("worldarena_hdf5", worldarena),
        )
        if matched
    ]
    if len(matches) != 1:
        raise ValueError(
            "HDF5 manifest must declare exactly one supported dataset_type"
        )
    return matches[0]


def _validate_dataset_manifest(
    *,
    dataset_type: str,
    manifest: Path,
) -> dict[str, int] | None:
    if dataset_type == "libero_hdf5":
        from ge_act.data.libero_fastwam_hdf5_schema import load_manifest

        load_manifest(manifest)
        return None
    elif dataset_type == "worldarena_hdf5":
        from qwen35_baton.worldarena_data import audit_worldarena_hdf5_cache

        audit = audit_worldarena_hdf5_cache(manifest)
        if audit.train_count <= 0:
            raise ValueError("WorldArena cache must contain at least one train record")
        return audit.to_dict()
    else:
        raise AssertionError("validated dataset_type is unreachable")


def _validate_qwen_config(path: Path) -> None:
    payload = _load_json(path / "config.json", label="local Qwen config")
    text_config = payload.get("text_config")
    architecture = payload.get("architectures")
    dense = (
        payload.get("model_type") == "qwen3_5"
        and isinstance(text_config, Mapping)
        and text_config.get("model_type") == "qwen3_5_text"
        and text_config.get("num_hidden_layers") == 24
        and text_config.get("hidden_size") == 2048
        and text_config.get("intermediate_size") == 6144
        and isinstance(payload.get("vision_config"), Mapping)
        and payload["vision_config"].get("depth") == 24
        and payload["vision_config"].get("hidden_size") == 1024
        and payload["vision_config"].get("out_hidden_size") == 2048
        and isinstance(architecture, list)
        and architecture == ["Qwen3_5ForConditionalGeneration"]
        and "moe" not in json.dumps(payload).lower()
        and not payload.get("num_experts")
    )
    if not dense:
        raise ValueError(
            "local Qwen config must identify the dense Qwen3.5-2B "
            "conditional-generation model"
        )


def _validate_processor(path: Path) -> None:
    candidates = (
        "processor_config.json",
        "preprocessor_config.json",
        "processor.json",
    )
    if not any((path / name).is_file() for name in candidates):
        raise FileNotFoundError(
            f"persisted Qwen processor configuration is missing from: {path}"
        )


def _added_token_ids(path: Path) -> tuple[int, ...]:
    payload = _load_json(path / "tokenizer.json", label="local Qwen tokenizer")
    entries = payload.get("added_tokens")
    if not isinstance(entries, list):
        raise ValueError("local Qwen tokenizer added_tokens must be a list")
    by_content: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("local Qwen tokenizer added_tokens entries are invalid")
        content = entry.get("content")
        token_id = entry.get("id")
        if isinstance(content, str) and type(token_id) is int:
            if content in by_content:
                raise ValueError(
                    f"local Qwen tokenizer repeats added token {content!r}"
                )
            by_content[content] = token_id
    try:
        identifiers = tuple(by_content[token] for token in ADDED_TOKENS)
    except KeyError as error:
        raise ValueError(
            f"local Qwen tokenizer is missing Baton token {error.args[0]!r}"
        ) from error
    if len(set(identifiers)) != len(ADDED_TOKENS) or any(
        identifier < 0 for identifier in identifiers
    ):
        raise ValueError("all seven Baton added tokens must map to unique IDs")
    return identifiers


def _siglip_geometry(path: Path) -> dict[str, int]:
    try:
        from transformers import AutoConfig

        resolved = AutoConfig.from_pretrained(path, local_files_only=True)
    except Exception as error:
        raise ValueError(f"local SigLIP2 config is invalid: {path}") from error
    if getattr(resolved, "model_type", None) not in {"siglip", "siglip2"}:
        raise ValueError(
            "local SigLIP2 config must use the released 'siglip' or "
            "'siglip2' configuration class"
        )
    vision = getattr(resolved, "vision_config", None)
    if vision is None:
        raise ValueError("local SigLIP2 config is missing vision_config")
    expected = {"image_size": 256, "patch_size": 16, "hidden_size": 1024}
    for name, value in expected.items():
        actual = getattr(vision, name, None)
        if actual != value:
            raise ValueError(f"local SigLIP2 {name} must be {value}, got {actual!r}")
    return expected


def _reject_output_ancestor(output: Path, protected: Sequence[Path]) -> None:
    output = output.expanduser().resolve()
    for path in protected:
        resolved = path.expanduser().resolve()
        if output == resolved or output in resolved.parents:
            raise ValueError(
                f"checkpoint output {output} must not be an ancestor of "
                f"model or dataset path {resolved}"
            )


def preflight_stage1(
    config: str | Path | Mapping[str, Any] | Any,
    *,
    world_size: int,
) -> dict[str, Any]:
    """Validate all local provenance and geometry without loading tensor weights."""

    from qwen35_baton.cli.train_semantic_planner import Stage1TrainingConfig

    if isinstance(config, (str, Path)):
        config = Stage1TrainingConfig.from_json(config)
    elif isinstance(config, Mapping):
        config = Stage1TrainingConfig.from_mapping(config)
    if not isinstance(config, Stage1TrainingConfig):
        raise TypeError("config must be a Stage1TrainingConfig, mapping, or JSON path")
    if config.tiny_test:
        from qwen35_baton.cli.train_semantic_planner import validate_global_batch

        return {
            "tiny_test": True,
            "global_batch": validate_global_batch(
                per_device_batch=config.per_device_batch,
                world_size=world_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
            ),
            "dataset_type": config.dataset_type,
            "camera_names": list(_CAMERA_NAMES[config.dataset_type]),
        }

    qwen = _require_directory(Path(config.qwen_model_path), label="Qwen model")
    processor = _require_directory(
        Path(config.qwen_processor_path), label="Qwen processor"
    )
    tokenizer = _require_directory(
        Path(config.qwen_tokenizer_path), label="Qwen tokenizer"
    )
    siglip = _require_directory(Path(config.siglip2_model_path), label="SigLIP2 model")
    manifest = Path(config.hdf5_manifest_path).expanduser().resolve()
    statistics = Path(config.dataset_statistics_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"HDF5 manifest does not exist: {manifest}")
    if not statistics.is_file():
        raise FileNotFoundError(f"dataset statistics file does not exist: {statistics}")
    manifest_payload = _load_json(manifest, label="HDF5 manifest")
    manifest_dataset_type = _manifest_dataset_type(manifest_payload)
    if manifest_dataset_type != config.dataset_type:
        raise ValueError(
            "HDF5 manifest dataset_type mismatch: "
            f"config declares {config.dataset_type!r}, manifest declares "
            f"{manifest_dataset_type!r}"
        )
    if config.dataset_type == "worldarena_hdf5" and statistics != manifest.with_name(
        "stats.json"
    ):
        raise ValueError(
            "WorldArena dataset_statistics_path must be the cache stats.json"
        )
    actual_manifest_hash = sha256_file(manifest)
    if actual_manifest_hash != config.hdf5_manifest_hash:
        raise ValueError(
            "HDF5 manifest hash mismatch: "
            f"expected {config.hdf5_manifest_hash}, got {actual_manifest_hash}"
        )
    if config.dataset_type == "worldarena_hdf5":
        stats_payload = _load_json(statistics, label="WorldArena cache stats.json")
        if (
            stats_payload.get("source_repository") != _WORLD_ARENA_SOURCE_REPOSITORY
            or stats_payload.get("manifest_sha256") != actual_manifest_hash
        ):
            raise ValueError(
                "WorldArena cache stats.json provenance differs from its manifest"
            )
    worldarena_cache_audit = _validate_dataset_manifest(
        dataset_type=config.dataset_type,
        manifest=manifest,
    )
    _validate_qwen_config(qwen)
    _validate_processor(processor)
    token_ids = _added_token_ids(tokenizer)
    geometry = _siglip_geometry(siglip)
    actual_siglip_config_hash = sha256_file(siglip / "config.json")
    if actual_siglip_config_hash != config.siglip2_config_hash:
        raise ValueError(
            "SigLIP2 config hash mismatch: "
            f"expected {config.siglip2_config_hash}, got {actual_siglip_config_hash}"
        )
    actual_siglip_artifact_hash = sha256_artifact(siglip)
    if actual_siglip_artifact_hash != config.siglip2_artifact_hash:
        raise ValueError(
            "SigLIP2 artifact hash mismatch: "
            f"expected {config.siglip2_artifact_hash}, got {actual_siglip_artifact_hash}"
        )
    _reject_output_ancestor(
        Path(config.output_dir),
        (qwen, processor, tokenizer, siglip, manifest, statistics),
    )
    fast_path = require_qwen35_fast_path()
    from qwen35_baton.cli.train_semantic_planner import (
        require_stage1_global_batch,
        resolve_deepspeed_runtime_config,
    )

    global_batch = require_stage1_global_batch(
        per_device_batch=config.per_device_batch,
        world_size=world_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    if config.distributed_strategy == "zero2":
        resolve_deepspeed_runtime_config(config, world_size=world_size)
    report = {
        "tiny_test": False,
        "global_batch": global_batch,
        "dataset_type": config.dataset_type,
        "camera_names": list(_CAMERA_NAMES[config.dataset_type]),
        "qwen_backbone": "dense Qwen3.5-2B",
        "added_token_ids": token_ids,
        "siglip2_geometry": geometry,
        "siglip2_config_hash": actual_siglip_config_hash,
        "siglip2_artifact_hash": actual_siglip_artifact_hash,
        "hdf5_manifest_hash": actual_manifest_hash,
        "qwen35_fast_path": fast_path,
        "distributed_strategy": config.distributed_strategy,
    }
    if worldarena_cache_audit is not None:
        report["worldarena_cache_audit"] = worldarena_cache_audit
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--per-device-batch", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--deepspeed-config-path", type=str)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from qwen35_baton.cli.train_semantic_planner import Stage1TrainingConfig

    config = Stage1TrainingConfig.from_json(args.config)
    overrides = {}
    if args.per_device_batch is not None:
        overrides["per_device_batch"] = args.per_device_batch
    if args.gradient_accumulation_steps is not None:
        overrides["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.deepspeed_config_path is not None:
        overrides["deepspeed_config_path"] = args.deepspeed_config_path
    if overrides:
        config = replace(config, **overrides)
    report = preflight_stage1(config, world_size=args.world_size)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
