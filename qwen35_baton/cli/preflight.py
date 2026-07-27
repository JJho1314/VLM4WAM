"""CPU-only, fail-closed preflight for local Baton Stage-1 artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from qwen35_baton.hashing import sha256_artifact, sha256_file
from qwen35_baton.sequence import ADDED_TOKENS


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
    if getattr(resolved, "model_type", None) != "siglip2":
        raise ValueError("local SigLIP2 config must have model_type 'siglip2'")
    vision = getattr(resolved, "vision_config", None)
    if vision is None:
        raise ValueError("local SigLIP2 config is missing vision_config")
    expected = {"image_size": 256, "patch_size": 16, "hidden_size": 1024}
    for name, value in expected.items():
        actual = getattr(vision, name, None)
        if actual != value:
            raise ValueError(
                f"local SigLIP2 {name} must be {value}, got {actual!r}"
            )
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
        }

    qwen = _require_directory(Path(config.qwen_model_path), label="Qwen model")
    processor = _require_directory(
        Path(config.qwen_processor_path), label="Qwen processor"
    )
    tokenizer = _require_directory(
        Path(config.qwen_tokenizer_path), label="Qwen tokenizer"
    )
    siglip = _require_directory(
        Path(config.siglip2_model_path), label="SigLIP2 model"
    )
    manifest = Path(config.hdf5_manifest_path).expanduser().resolve()
    statistics = Path(config.dataset_statistics_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"HDF5 manifest does not exist: {manifest}")
    if not statistics.is_file():
        raise FileNotFoundError(
            f"dataset statistics file does not exist: {statistics}"
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
    actual_manifest_hash = sha256_file(manifest)
    if actual_manifest_hash != config.hdf5_manifest_hash:
        raise ValueError(
            "HDF5 manifest hash mismatch: "
            f"expected {config.hdf5_manifest_hash}, got {actual_manifest_hash}"
        )
    _reject_output_ancestor(
        Path(config.output_dir),
        (qwen, processor, tokenizer, siglip, manifest, statistics),
    )
    from qwen35_baton.cli.train_semantic_planner import validate_global_batch

    global_batch = validate_global_batch(
        per_device_batch=config.per_device_batch,
        world_size=world_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    return {
        "tiny_test": False,
        "global_batch": global_batch,
        "qwen_backbone": "dense Qwen3.5-2B",
        "added_token_ids": token_ids,
        "siglip2_geometry": geometry,
        "siglip2_config_hash": actual_siglip_config_hash,
        "siglip2_artifact_hash": actual_siglip_artifact_hash,
        "hdf5_manifest_hash": actual_manifest_hash,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--per-device-batch", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from qwen35_baton.cli.train_semantic_planner import Stage1TrainingConfig

    config = Stage1TrainingConfig.from_json(args.config)
    overrides = {}
    if args.per_device_batch is not None:
        overrides["per_device_batch"] = args.per_device_batch
    if args.gradient_accumulation_steps is not None:
        overrides["gradient_accumulation_steps"] = (
            args.gradient_accumulation_steps
        )
    if overrides:
        config = replace(config, **overrides)
    report = preflight_stage1(config, world_size=args.world_size)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
