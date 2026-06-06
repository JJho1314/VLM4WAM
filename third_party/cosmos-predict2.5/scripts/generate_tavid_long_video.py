#!/usr/bin/env python3
"""Long-video TAViD evaluation: autoregressive chunked rollout with optional mask.

For each sample drawn from the experiment's dataloader, generates three
variants and writes them side-by-side as a strip mp4:
  * no_mask      : target_mask=None  (recovers base behaviour)
  * true_mask    : real SAM target mask on the first chunk
  * shifted_mask : mask spatially rolled 1/4 — sanity check that guidance bites

Usage:
  torchrun --standalone --nproc_per_node=1 -m scripts.generate_tavid_long_video \
    --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --experiment predict2_video2world_training_2b_robointer_droid_tavid_v2 \
    --checkpoint outputs/.../iter_005000 \
    --output-dir outputs/long_eval/
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from cosmos_oss.init import init_environment
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils import distributed, misc
from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override
from cosmos_predict2._src.predict2.inference.utils import write_video
from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="Hydra config file (cosmos_predict2/_src/predict2/configs/video2world/config.py)")
    p.add_argument("--experiment", required=True,
                   help="Experiment name e.g. predict2_video2world_training_2b_robointer_droid_tavid_v2")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--num-output-frames", type=int, default=121,
                   help="Total pixel frames in the long video (121 ~= 15s @ 8fps).")
    p.add_argument("--chunk-overlap-pixel", type=int, default=4)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--num-steps", type=int, default=35)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=200)
    p.add_argument("--resolution", default="176,320")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("opts", nargs=argparse.REMAINDER,
                   help="Extra hydra-style overrides forwarded as experiment_opts")
    return p.parse_args()


def gen_to_01(video_m11: torch.Tensor) -> torch.Tensor:
    return ((video_m11.detach().float().cpu().clamp(-1, 1) + 1.0) / 2.0)[0]


def to_uint8(video01_chw: torch.Tensor) -> np.ndarray:
    return (video01_chw.clamp(0, 1).float().cpu().permute(1, 2, 3, 0).numpy() * 255.0).round().astype(np.uint8)


def shift_mask(mask: torch.Tensor) -> torch.Tensor:
    sh = max(1, mask.shape[-2] // 4)
    sw = max(1, mask.shape[-1] // 4)
    return torch.roll(mask, shifts=(sh, sw), dims=(-2, -1))


def main() -> None:
    args = parse_args()
    init_environment()

    user_opts = list(args.opts)
    if user_opts and user_opts[0] == "--":
        user_opts = user_opts[1:]
    experiment_opts = [
        f"experiment={args.experiment}",
        "checkpoint.save_to_object_store.enabled=False",
        "checkpoint.load_from_object_store.enabled=False",
        "checkpoint.load_training_state=False",
        "trainer.run_validation=False",
        # eval dataloader: always serve real masks, no dropout
        "dataloader_train.dataset.target_mask_dropout_prob=0.0",
        *user_opts,
    ]

    inference = Video2WorldInference(
        experiment_name=args.experiment,
        ckpt_path=args.checkpoint,
        s3_credential_path="",
        context_parallel_size=1,
        config_file=args.config,
        experiment_opts=experiment_opts,
        offload_diffusion_model=False,
        offload_text_encoder=False,
        offload_tokenizer=False,
    )
    model = inference.model
    model.eval()

    dataloader = instantiate(inference.config.dataloader_train)
    output_dir = Path(args.output_dir)
    if distributed.is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)

    chunk_size_pixel = model.tokenizer.get_pixel_num_frames(model.config.state_t)
    summary = {
        "checkpoint": args.checkpoint,
        "experiment": args.experiment,
        "chunk_size_pixel": int(chunk_size_pixel),
        "chunk_overlap_pixel": int(args.chunk_overlap_pixel),
        "num_output_frames": int(args.num_output_frames),
        "guidance": float(args.guidance),
        "samples": [],
    }

    saved = 0
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(dataloader):
            if batch_idx >= args.max_batches or saved >= args.num_samples:
                break
            if "target_mask" not in data_batch or float(data_batch["target_mask"].sum()) <= 0:
                continue

            data_batch = misc.to(data_batch, device="cuda")
            target_mask = data_batch["target_mask"][:1]            # [1, 1, T, H, W]
            video = data_batch["video"][:1]                        # [1, C, T, H, W] uint8
            caption = data_batch.get("ai_caption", [""])[0]

            # Long rollout seeded from the clip's first frame
            first_frame = video[:, :, :1].to(torch.uint8)

            modes = {
                "no_mask": None,
                "true_mask": target_mask,
                "shifted_mask": shift_mask(target_mask),
            }
            mode_videos = {}
            for name, mask in modes.items():
                long_video = inference.generate_autoregressive_from_batch(
                    prompt=caption,
                    input_path=first_frame,
                    num_output_frames=args.num_output_frames,
                    chunk_size=chunk_size_pixel,
                    chunk_overlap=args.chunk_overlap_pixel,
                    guidance=args.guidance,
                    num_latent_conditional_frames=1,
                    resolution=args.resolution,
                    seed=args.seed + saved,
                    negative_prompt=args.negative_prompt,
                    num_steps=args.num_steps,
                    target_mask=mask,
                )
                mode_videos[name] = gen_to_01(long_video)          # [C, T, H, W]

            if distributed.is_rank0():
                stem = f"sample_{saved:03d}"
                for name, vid01 in mode_videos.items():
                    write_video(str(output_dir / f"{stem}_{name}.mp4"),
                                to_uint8(vid01), fps=args.fps, lossless=False)
                # Side-by-side strip: no_mask | true_mask | shifted_mask
                strip = torch.cat([mode_videos[n] for n in ("no_mask", "true_mask", "shifted_mask")], dim=3)
                write_video(str(output_dir / f"{stem}_strip.mp4"),
                            to_uint8(strip), fps=args.fps, lossless=False)
                (output_dir / f"{stem}_caption.txt").write_text(caption + "\n")
                summary["samples"].append({"index": saved, "caption": caption})
                print(json.dumps({"sample": saved, "caption": caption}, ensure_ascii=False), flush=True)
            saved += 1

    if distributed.is_rank0():
        (output_dir / "long_video_eval_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )


if __name__ == "__main__":
    main()
