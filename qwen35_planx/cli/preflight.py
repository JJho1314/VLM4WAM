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
from transformers import AutoConfig, AutoModel

from qwen35_planx.official_ta_tok import inspect_released_checkpoint


_MINIMUM_CODEBOOK_EXPORT_BYTES = 1024 * 1024 * 1024
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
