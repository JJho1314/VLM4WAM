#!/usr/bin/env python3
"""Run one-piece mask-free InstructSAM -> Cosmos video2world inference.

This script keeps InstructSAM and Cosmos in the same inference process:
1. read the conditioning image/video first frame,
2. run InstructSAM with the target text query,
3. export decoder-dense target features,
4. pass raw_seg as target_feature and decoder_dense as target_dense_feature,
5. generate the video.

The predicted InstructSAM mask is allowed to exist inside InstructSAM, but is
discarded before Cosmos inference.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra_path in (REPO_ROOT, REPO_ROOT / "packages" / "cosmos-oss", REPO_ROOT / "packages" / "cosmos-cuda"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

from cosmos_predict2.config import InferenceArguments, SetupArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Cosmos checkpoint directory or model checkpoint path.")
    parser.add_argument(
        "--experiment",
        default="predict2_video2world_training_2b_droid_success_v21_what_where_context",
        help="Cosmos experiment/config registry name.",
    )
    parser.add_argument(
        "--config-file",
        default="cosmos_predict2/_src/predict2/configs/video2world/config.py",
        help="Cosmos config file.",
    )
    parser.add_argument("--input-path", required=True, type=Path, help="Conditioning image or video path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write generated outputs.")
    parser.add_argument("--name", default="maskfree_instructsam_cosmos", help="Output sample name.")
    parser.add_argument("--prompt", required=True, help="Cosmos video generation prompt.")
    parser.add_argument(
        "--target-query",
        required=True,
        help="Target text query for InstructSAM, e.g. 'Please segment the yellow carrot with green leaves.'",
    )
    parser.add_argument("--instructsam-model-path", required=True, type=Path, help="InstructSAM checkpoint path.")
    parser.add_argument(
        "--instructsam-source-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "InstructSAM",
        help="InstructSAM source checkout used for imports.",
    )
    parser.add_argument("--resolution", default="480,864", help="Cosmos inference resolution as H,W.")
    parser.add_argument("--num-output-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=8, help="FPS used when saving the generated video.")
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--guidance", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--disable-guardrails", action="store_true", default=True)
    parser.add_argument("--offload-diffusion-model", action="store_true")
    parser.add_argument("--offload-text-encoder", action="store_true")
    parser.add_argument("--offload-tokenizer", action="store_true")
    parser.add_argument(
        "--pass-mask-to-cosmos",
        action="store_true",
        help="Debug only. If set, also pass InstructSAM mask to Cosmos. Default is mask-free.",
    )
    parser.add_argument(
        "--instructsam-python",
        type=Path,
        default=None,
        help=(
            "Optional Python executable for InstructSAM feature extraction. "
            "Use this when InstructSAM requires a different transformers stack than Cosmos. "
            "The script remains a one-command pipeline, but feature extraction runs in a subprocess."
        ),
    )
    return parser.parse_args()


def infer_type(input_path: Path) -> str:
    suffix = input_path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image2world"
    if suffix == ".mp4":
        return "video2world"
    raise ValueError(f"Unsupported input extension for Cosmos inference: {suffix}")


def main() -> None:
    args = parse_args()
    init_environment()
    try:
        target_feature_path = None
        target_query = args.target_query
        instructsam_model_path = args.instructsam_model_path
        if args.instructsam_python is not None:
            target_feature_path = args.output_dir / "_online_instructsam_features" / f"{args.name}.pt"
            extractor_env = os.environ.copy()
            extractor_env["PYTHONPATH"] = ":".join(
                [
                    str(REPO_ROOT),
                    str(REPO_ROOT / "packages" / "cosmos-oss"),
                    str(args.instructsam_source_root),
                    extractor_env.get("PYTHONPATH", ""),
                ]
            )
            extractor_cmd = [
                str(args.instructsam_python),
                str(REPO_ROOT / "scripts" / "extract_instructsam_feature_once.py"),
                "--input-path",
                str(args.input_path),
                "--target-query",
                args.target_query,
                "--model-path",
                str(args.instructsam_model_path),
                "--source-root",
                str(args.instructsam_source_root),
                "--output-path",
                str(target_feature_path),
                "--feature-mode",
                "decoder_dense",
                "--include-raw-seg",
            ]
            print("Running InstructSAM feature extraction subprocess:")
            print(" ".join(extractor_cmd))
            subprocess.run(extractor_cmd, check=True, env=extractor_env)
            target_query = None
            instructsam_model_path = None

        setup = SetupArguments(
            output_dir=args.output_dir,
            model="2B/post-trained",
            checkpoint_path=str(args.checkpoint),
            experiment=args.experiment,
            config_file=args.config_file,
            context_parallel_size=args.context_parallel_size,
            offload_diffusion_model=args.offload_diffusion_model,
            offload_text_encoder=args.offload_text_encoder,
            offload_tokenizer=args.offload_tokenizer,
            disable_guardrails=args.disable_guardrails,
        )
        sample = InferenceArguments(
            name=args.name,
            inference_type=infer_type(args.input_path),
            input_path=args.input_path,
            prompt=args.prompt,
            target_query=target_query,
            instructsam_model_path=instructsam_model_path,
            instructsam_source_root=args.instructsam_source_root,
            instructsam_feature_mode="decoder_dense",
            route_decoder_dense_to_target_dense_feature=True,
            pass_instructsam_mask_to_cosmos=args.pass_mask_to_cosmos,
            target_feature_path=target_feature_path,
            resolution=args.resolution,
            num_output_frames=args.num_output_frames,
            save_fps=args.fps,
            num_steps=args.num_steps,
            guidance=args.guidance,
            seed=args.seed,
        )

        init_output_dir(setup.output_dir, profile=setup.profile)
        from cosmos_predict2.inference import Inference

        inference = Inference(setup)
        outputs = inference.generate([sample], output_dir=setup.output_dir)
        if outputs:
            print("\n".join(outputs))
    finally:
        cleanup_environment()


if __name__ == "__main__":
    main()
