#!/usr/bin/env python3
"""Extract one InstructSAM target feature for mask-free Cosmos inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra_path in (REPO_ROOT, REPO_ROOT / "packages" / "cosmos-oss"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import torch

from cosmos_predict2._src.predict2.target_aware.instructsam_mask import InstructSAMTargetMaskGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--target-query", required=True)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--feature-mode", default="decoder_dense", choices=["mask_query", "raw_seg", "decoder_dense"])
    parser.add_argument(
        "--include-raw-seg",
        action="store_true",
        help="When extracting decoder_dense, also save raw_seg as target_feature and decoder_dense as target_dense_feature.",
    )
    parser.add_argument("--combine-mode", default="best", choices=["best", "union"])
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    generator = InstructSAMTargetMaskGenerator(
        model_path=args.model_path,
        source_root=args.source_root,
    )
    result = generator.predict_from_input(
        args.input_path,
        args.target_query,
        combine_mode=args.combine_mode,
        mask_threshold=args.mask_threshold,
        feature_mode=args.feature_mode,
    )
    if result.feature_B_L_D is None:
        raise RuntimeError(f"InstructSAM did not expose target feature for query: {args.target_query}")
    feature = result.feature_B_L_D.squeeze(0).detach().cpu().float().contiguous()
    target_feature = feature
    target_dense_feature = None
    feature_mode = args.feature_mode
    if args.include_raw_seg:
        if args.feature_mode != "decoder_dense":
            raise ValueError("--include-raw-seg is only supported with --feature-mode decoder_dense")
        raw = generator._extract_target_feature(feature_mode="raw_seg")
        if raw is None:
            raise RuntimeError(f"InstructSAM did not expose raw_seg feature for query: {args.target_query}")
        target_feature = raw.squeeze(0).detach().cpu().float().contiguous()
        target_dense_feature = feature
        feature_mode = "raw_seg+decoder_dense"
    payload = {
        "target_feature": target_feature,
        "query": args.target_query,
        "instructsam_text": result.text,
        "score": result.score,
        "feature_mode": feature_mode,
        "feature_shape": list(target_feature.shape),
        "source": "online_instructsam_feature_once",
        "input_path": str(args.input_path),
        "model_path": str(args.model_path),
    }
    if target_dense_feature is not None:
        payload["target_dense_feature"] = target_dense_feature
        payload["target_dense_feature_shape"] = list(target_dense_feature.shape)
    torch.save(payload, args.output_path)
    print(json.dumps({k: v for k, v in payload.items() if k != "target_feature"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
