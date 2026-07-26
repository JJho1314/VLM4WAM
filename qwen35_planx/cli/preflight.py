"""Preflight checks for immutable grounded Plan-X artifacts."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

from qwen35_planx.official_ta_tok import inspect_released_checkpoint


_MINIMUM_CODEBOOK_EXPORT_BYTES = 1024 * 1024 * 1024
_SIGLIP_WEIGHT_PATTERNS = (
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "pytorch_model-*.bin",
)


def _has_local_model_weights(model_path: Path) -> bool:
    if model_path.is_file():
        return model_path.suffix in {".bin", ".safetensors"}
    return any(
        candidate.is_file()
        for pattern in _SIGLIP_WEIGHT_PATTERNS
        for candidate in model_path.glob(pattern)
    )


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

    if not model_path.exists():
        errors.append(f"local SigLIP2 model path does not exist: {model_path}")
    elif not _has_local_model_weights(model_path):
        errors.append(f"local SigLIP2 model weights are missing from: {model_path}")
    elif model_path.is_dir() and not (model_path / "config.json").is_file():
        errors.append(f"local SigLIP2 config.json is missing from: {model_path}")

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
