#!/usr/bin/env python3
"""Latent-ablation diagnostic for the 4B lingbot-DINO planner.

Question: is the 8-latents/keyframe bottleneck what limits prediction quality?
The head reads [image_hidden ⊕ plan latents]. We re-run the SAME head on ablated inputs:

  normal        head(image, latents)            — reference
  latents_zero  head(image, 0)                  — no plan information at all
  latents_swap  head(image, latents_of_OTHER)   — plan info from a different episode
  image_zero    head(0, latents)                — plan latents only, no image context

If normal ≈ latents_zero/swap  -> the head barely uses the latents; widening 8→16 cannot help
                                  (the smoothing comes from the MSE objective, not capacity).
If normal >> latents_zero/swap -> the latents carry real signal; a capacity experiment is
                                  justified.
image_zero shows how much comes from the plan latents alone.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_qwen3vl4b_lingbot_dino_planner as T  # noqa: E402
from evaluate_qwen3vl4b_lingbot_dino_planner import load_planner  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--frame-ranges-json", type=Path, default=None)
    p.add_argument("--dino-teacher-ckpt", type=Path, required=True)
    p.add_argument("--dino-teacher-config", type=Path, required=True)
    p.add_argument("--dino-input-size", type=int, default=256)
    p.add_argument("--num-keyframes", type=int, default=5)
    p.add_argument("--grid-size", type=int, default=16)
    p.add_argument("--num-latent-per-keyframe", type=int, default=8)
    p.add_argument("--semantic-dim", type=int, default=1024)
    p.add_argument("--sequence-length", type=int, default=49)
    p.add_argument("--keyframe-scheme", default="uniform")
    p.add_argument("--keyframe-gamma", type=float, default=0.6)
    p.add_argument("--max-samples", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=4321)
    p.add_argument("--output-json", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.frame_ranges_json is None:
        args.frame_ranges_json = args.dataset_root / "frame_ranges.json"

    wrapper, processor, meta = load_planner(args.checkpoint_dir, args, device)
    dino_encoder = T.DinoVideoTargetEncoder(
        ckpt_path=args.dino_teacher_ckpt, config_path=args.dino_teacher_config,
        input_size=args.dino_input_size, device=device,
    )

    latent_len = wrapper.latent_len  # from ckpt meta (handles grouped shared+own latents)
    plan_sequence = [f"<|sem_plan_{i}|>" for i in range(latent_len)]
    dataset = T.OnlineSemanticPlanDataset(
        dataset_root=args.dataset_root, frame_ranges_json=args.frame_ranges_json,
        num_keyframes=args.num_keyframes, sequence_length=args.sequence_length,
        keyframe_scheme=args.keyframe_scheme, keyframe_gamma=args.keyframe_gamma,
        max_samples=args.max_samples, seed=args.seed,
    )
    # batch_size=2 so latents_swap can exchange the two samples' plan latents
    loader = DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=args.num_workers, drop_last=True,
        collate_fn=T.Collator(processor=processor, plan_sequence=plan_sequence), pin_memory=True,
    )

    K, G = args.num_keyframes, args.grid_size
    model_dtype = next(wrapper.model.parameters()).dtype
    head_dtype = next(wrapper.plan_head.parameters()).dtype
    conds = ["normal", "latents_zero", "latents_swap", "image_zero"]
    agg = {c: {"mse": 0.0, "cos": 0.0} for c in conds}
    per_kf_cos = {c: [0.0] * K for c in conds}
    n = 0

    with torch.no_grad():
        for batch in loader:
            batch.pop("stems", None)
            keyframes = batch.pop("keyframe_images", None)
            current = batch.pop("current_image", None)
            if keyframes is None:
                continue
            inputs = T.move_qwen_inputs_to_device(batch, device, model_dtype=model_dtype)
            ih, ph_all = wrapper._forward_hiddens(**inputs)
            ph, _ = wrapper._split_latents(ph_all)  # the VIDEO head's latent view
            ih, ph = ih.to(head_dtype), ph.to(head_dtype)

            cur = current.permute(0, 3, 1, 2).contiguous()
            kfs = [keyframes[:, j].permute(0, 3, 1, 2).contiguous() for j in range(keyframes.shape[1])]
            target = dino_encoder.encode_future_keyframes(cur, kfs).float().to(device)

            preds = {
                "normal": wrapper.plan_head(ih, ph),
                "latents_zero": wrapper.plan_head(ih, torch.zeros_like(ph)),
                "latents_swap": wrapper.plan_head(ih, ph.flip(0)),  # bs=2 -> exchange samples
                "image_zero": wrapper.plan_head(torch.zeros_like(ih), ph),
            }
            bs = target.shape[0]
            for c, pred in preds.items():
                pred = pred.float()
                agg[c]["mse"] += float(F.mse_loss(pred, target)) * bs
                agg[c]["cos"] += float(
                    F.cosine_similarity(pred.flatten(0, 1), target.flatten(0, 1), dim=-1).mean()) * bs
                tok = pred.shape[1] // K
                pk = pred.reshape(bs, K, tok, -1)
                tk = target.reshape(bs, K, tok, -1)
                for i in range(K):
                    per_kf_cos[c][i] += float(
                        F.cosine_similarity(pk[:, i].flatten(0, 1), tk[:, i].flatten(0, 1), dim=-1).mean()) * bs
            n += bs
            print(f"[ablate] {n}/{args.max_samples}", flush=True)

    denom = max(n, 1)
    result = {
        "checkpoint": str(args.checkpoint_dir),
        "num_samples": n,
        "conditions": {
            c: {
                "mse": agg[c]["mse"] / denom,
                "cosine_sim": agg[c]["cos"] / denom,
                "per_kf_cosine": [v / denom for v in per_kf_cos[c]],
            }
            for c in conds
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
