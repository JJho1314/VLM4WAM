#!/usr/bin/env python3
"""Strict preflight checks for Qwen3.5 Plan-X artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from qwen35_planx.config import PlanGeometry, TATokMetadata
from qwen35_planx.hashing import sha256_file
from qwen35_planx.ta_tok_trainer import validate_resume_anchors


def _finite_metrics(value: object, *, path: str = "metrics") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _finite_metrics(nested, path=f"{path}.{key}")
    elif isinstance(value, (int, float)) and not math.isfinite(float(value)):
        raise ValueError(f"non-finite metric: {path}={value}")


def preflight_ta_tok_checkpoint(
    checkpoint_dir: Path | str,
    *,
    min_coverage: float,
    max_dead_code_ratio: float,
) -> dict[str, object]:
    checkpoint_dir = Path(checkpoint_dir)
    required = (
        "ta_tok.safetensors",
        "metadata.json",
        "anchor_ids.npy",
        "metrics.json",
    )
    missing = [name for name in required if not (checkpoint_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "TA-Tok checkpoint is missing: " + ", ".join(missing)
        )

    metadata = TATokMetadata.from_dict(
        json.loads((checkpoint_dir / "metadata.json").read_text())
    )
    geometry = PlanGeometry()
    if metadata.visual_vocab_size != geometry.visual_vocab_size:
        raise ValueError("TA-Tok codebook size must be 65,536")
    if metadata.siglip_model != "google/siglip2-large-patch16-256":
        raise ValueError("teacher must be project SigLIP2-Large/Patch16/256")
    if metadata.selected_layer != -2:
        raise ValueError("selected_layer must be -2")
    if metadata.grid_size * metadata.grid_size != geometry.tokens_per_frame:
        raise ValueError("TA-Tok output must contain exactly 256 codes")

    anchors = np.load(
        checkpoint_dir / "anchor_ids.npy", allow_pickle=False
    ).astype(np.int64, copy=False)
    if anchors.shape != (geometry.visual_vocab_size,):
        raise ValueError("anchor count must equal 65,536")
    validate_resume_anchors(metadata.anchor_token_ids, anchors)

    actual_state_hash = sha256_file(checkpoint_dir / "ta_tok.safetensors")
    if actual_state_hash != metadata.state_hash:
        raise ValueError("TA-Tok checkpoint state hash mismatch")
    metrics = json.loads((checkpoint_dir / "metrics.json").read_text())
    _finite_metrics(metrics)
    overall = metrics.get("overall")
    if not isinstance(overall, Mapping):
        raise ValueError("metrics must contain an overall validation report")
    coverage = float(overall.get("coverage", -1.0))
    dead_code_ratio = float(overall.get("dead_code_ratio", 2.0))
    if coverage < min_coverage:
        raise ValueError(
            f"validation coverage {coverage} is below {min_coverage}"
        )
    if dead_code_ratio > max_dead_code_ratio:
        raise ValueError(
            f"dead-code ratio {dead_code_ratio} exceeds "
            f"{max_dead_code_ratio}"
        )
    for camera in ("main", "wrist"):
        if camera not in metrics:
            raise ValueError(f"validation metrics are missing {camera}")
    return {
        "checkpoint": str(checkpoint_dir.resolve()),
        "state_hash": actual_state_hash,
        "coverage": coverage,
        "dead_code_ratio": dead_code_ratio,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="artifact", required=True)
    ta_tok = subparsers.add_parser("ta-tok")
    ta_tok.add_argument("--checkpoint-dir", type=Path, required=True)
    ta_tok.add_argument("--min-coverage", type=float, default=0.1)
    ta_tok.add_argument("--max-dead-code-ratio", type=float, default=0.9)
    args = parser.parse_args()
    report = preflight_ta_tok_checkpoint(
        args.checkpoint_dir,
        min_coverage=args.min_coverage,
        max_dead_code_ratio=args.max_dead_code_ratio,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
