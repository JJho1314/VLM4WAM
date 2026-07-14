#!/usr/bin/env python3
"""Export an oracle semantic plan from a video window using the online SigLIP2 encoder.

Produces a .pt payload compatible with inference `--semantic-plan-path`: the tensor plus
frame indices, so keyframe times are auto-extracted at load time. Uses the exact same
OnlineSemanticPlanEncoder as training, so features match the training conditioning space.

Example (HPC3, from the cosmos-predict2.5 repo root):
    python scripts/export_semantic_plan_from_video.py \
        --video $DATASET_ROOT/videos/episode_000123_left_external.mp4 \
        --start 10 --num-frames 49 --frame-stride 1 \
        --num-keyframes 5 --grid-size 0 \
        --encoder-path /data/user/jhe724/workspace/VLM4WAM/third_party/siglip2-so400m-patch14-384 \
        --output /tmp/episode_000123_oracle_plan.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cosmos_predict2._src.predict2.networks.semantic_plan_conditioning import OnlineSemanticPlanEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0, help="First frame index of the window")
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--num-keyframes", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=0, help="<=0 keeps the native 27x27 grid")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import decord

    reader = decord.VideoReader(str(args.video), ctx=decord.cpu(0))
    video_len = len(reader)
    frame_ids = [args.start + args.frame_stride * i for i in range(args.num_frames)]
    if frame_ids[-1] >= video_len:
        raise ValueError(
            f"Window [{frame_ids[0]}, {frame_ids[-1]}] exceeds video length {video_len} for {args.video}"
        )
    frames = torch.from_numpy(reader.get_batch(frame_ids).asnumpy())  # [T, H, W, C] uint8
    video = frames.permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1.0  # [1, C, T, H, W] in [-1,1]
    video = video.to(args.device)

    encoder = OnlineSemanticPlanEncoder(
        str(args.encoder_path),
        num_keyframes=args.num_keyframes,
        grid_size=args.grid_size,
        image_size=args.image_size,
    )
    semantic_plan, _times = encoder(video)
    keyframe_positions = encoder.keyframe_indices(args.num_frames).tolist()
    future_frame_indices = [frame_ids[pos] for pos in keyframe_positions]

    per_frame_tokens = semantic_plan.shape[1] // len(keyframe_positions)
    payload = {
        "semantic_plan": semantic_plan[0].to(dtype=torch.bfloat16).cpu(),
        "video_path": str(args.video),
        "video_frame_indices": frame_ids,
        "future_frame_indices": future_frame_indices,
        "first_frame_index": frame_ids[0],
        "frame_stride": args.frame_stride,
        "sequence_length": args.num_frames,
        "num_keyframes": len(keyframe_positions),
        "spatial_tokens_per_keyframe": per_frame_tokens,
        "grid_size": args.grid_size if args.grid_size > 0 else int(per_frame_tokens**0.5),
        "semantic_dim": int(semantic_plan.shape[-1]),
        "feature_type": (
            "siglip2_penultimate_spatial_native" if args.grid_size <= 0 else "siglip2_penultimate_spatial_pooled"
        ),
        "encoder_path": str(args.encoder_path),
        "online_exported": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "semantic_plan_shape": list(payload["semantic_plan"].shape),
                "future_frame_indices": future_frame_indices,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
