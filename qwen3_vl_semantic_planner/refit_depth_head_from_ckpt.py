#!/usr/bin/env python3
"""Refit the auxiliary future-DEPTH head against a FROZEN trained planner checkpoint.

Why: earlier checkpoints only saved the video plan head (the depth head was a train-only
auxiliary), so Depth-Pred cannot be visualized from them. The depth head only reads the frozen
VLM's hiddens (image tokens + plan latents), so retraining JUST the head — warm-started from the
lingbot future_depth_align_head like in training — recovers a faithful stand-in quickly.

Everything except the depth head is frozen; the VLM forward runs under no_grad, so steps are
cheap. Saves `depth_head_refit.pt` into the checkpoint dir (load_planner picks it up).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_semantic_planner as T  # noqa: E402
from evaluate_qwen3vl4b_lingbot_dino_planner import load_planner  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--frame-ranges-json", type=Path, default=None)
    p.add_argument("--head-warmstart-ckpt", type=Path, required=True,
                   help="dir with model.safetensors.index.json holding future_depth_align_* tensors")
    p.add_argument("--depth-moge-path", type=Path, required=True)
    p.add_argument("--depth-morgbd-path", type=Path, required=True)
    p.add_argument("--depth-input-size", type=int, default=256)
    p.add_argument("--num-keyframes", type=int, default=5)
    p.add_argument("--grid-size", type=int, default=16)
    p.add_argument("--num-latent-per-keyframe", type=int, default=8)
    p.add_argument("--semantic-dim", type=int, default=1024)
    p.add_argument("--sequence-length", type=int, default=49)
    p.add_argument("--keyframe-scheme", default="uniform")
    p.add_argument("--keyframe-gamma", type=float, default=0.6)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--log-steps", type=int, default=50)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.frame_ranges_json is None:
        args.frame_ranges_json = args.dataset_root / "frame_ranges.json"

    # frozen planner (VLM + video head) + a fresh depth head (uninitialized -> warm-start below)
    wrapper, processor, meta = load_planner(args.checkpoint_dir, args, device, use_depth=True)
    head_state = T._load_lingbot_head_state(args.head_warmstart_ckpt)
    report = wrapper.depth_head.load_lingbot_warmstart(head_state, head_name="future_depth_align_head")
    print(json.dumps({"depth_head_warmstart": report}), flush=True)

    for p_ in wrapper.parameters():
        p_.requires_grad_(False)
    for p_ in wrapper.depth_head.parameters():
        p_.requires_grad_(True)
    wrapper.depth_head.train()

    depth_encoder = T.DepthTargetEncoder(
        moge_path=args.depth_moge_path, morgbd_path=args.depth_morgbd_path,
        input_size=args.depth_input_size, num_tokens=args.grid_size * args.grid_size, device=device,
    )

    latent_len = wrapper.latent_len  # from ckpt meta (handles grouped shared+own latents)
    plan_sequence = [f"<|sem_plan_{i}|>" for i in range(latent_len)]
    dataset = T.OnlineSemanticPlanDataset(
        dataset_root=args.dataset_root, frame_ranges_json=args.frame_ranges_json,
        num_keyframes=args.num_keyframes, sequence_length=args.sequence_length,
        keyframe_scheme=args.keyframe_scheme, keyframe_gamma=args.keyframe_gamma,
        max_samples=0, seed=args.seed,  # 0 = all stems (the dataset's "no cap" convention)
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=T.Collator(processor=processor, plan_sequence=plan_sequence), pin_memory=True,
    )

    optim = torch.optim.AdamW(wrapper.depth_head.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        prog = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    model_dtype = next(wrapper.model.parameters()).dtype
    head_dtype = next(wrapper.depth_head.parameters()).dtype

    step = 0
    while step < args.max_steps:
        dataset.set_epoch(step)
        for batch in loader:
            batch.pop("stems", None)
            keyframes = batch.pop("keyframe_images")
            current = batch.pop("current_image")  # noqa: F841 (depth teacher is per-frame monocular)
            inputs = T.move_qwen_inputs_to_device(batch, device, model_dtype=model_dtype)
            with torch.no_grad():
                image_hidden, plan_hidden = wrapper._forward_hiddens(**inputs)
                _, depth_lat = wrapper._split_latents(plan_hidden)
                kfs = [keyframes[:, j].permute(0, 3, 1, 2).contiguous() for j in range(keyframes.shape[1])]
                target = depth_encoder.encode_future_keyframes(kfs).float()
            pred = wrapper.depth_head(image_hidden.to(head_dtype), depth_lat.to(head_dtype)).float()
            loss = F.smooth_l1_loss(pred, target.to(pred.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wrapper.depth_head.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_steps == 0:
                nr = (pred.norm(dim=-1).mean() / target.norm(dim=-1).mean().clamp_min(1e-6)).item()
                print(json.dumps({"step": step, "depth_smooth_l1": float(loss),
                                  "depth_norm_ratio": nr, "lr": sched.get_last_lr()[0]}), flush=True)
            if step >= args.max_steps:
                break

    out = args.checkpoint_dir / "depth_head_refit.pt"
    torch.save(wrapper.depth_head.state_dict(), out)
    print(json.dumps({"saved": str(out)}), flush=True)


if __name__ == "__main__":
    main()
