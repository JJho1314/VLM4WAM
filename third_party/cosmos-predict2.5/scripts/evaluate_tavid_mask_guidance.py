#!/usr/bin/env python3
"""Evaluate whether TAViD target masks affect Cosmos video generation.

This script loads a trained Cosmos checkpoint, draws a few data batches, and
generates paired samples with the original target mask, a zero mask, and a
spatially shifted mask.  It saves contact sheets plus a JSON summary of how much
the generated video changes inside versus outside the target-mask region.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from copy import copy
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

from cosmos_oss.init import init_environment
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils import distributed, misc
from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-conditional-frames", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def clone_batch(batch: dict) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.clone() if torch.is_tensor(value) else copy(value)
    return out


def add_online_text_embeddings(model, data_batch: dict) -> None:
    text_encoder_config = getattr(model.config, "text_encoder_config", None)
    if text_encoder_config is not None and text_encoder_config.compute_online:
        text_embeddings = model.text_encoder.compute_text_embeddings_online(data_batch, model.input_caption_key)
        data_batch["t5_text_embeddings"] = text_embeddings
        data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")


def make_shifted_mask(mask: torch.Tensor) -> torch.Tensor:
    # Shift by about one quarter of the spatial extent. This keeps mask size but
    # moves the requested target region enough for a visible control test.
    shift_h = max(1, mask.shape[-2] // 4)
    shift_w = max(1, mask.shape[-1] // 4)
    return torch.roll(mask, shifts=(shift_h, shift_w), dims=(-2, -1))


def to_01(video: torch.Tensor) -> torch.Tensor:
    return ((video.detach().float().cpu().clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)


def mask_to_rgb(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    # mask: [1, T, H, W] or [T, H, W]
    if mask.ndim == 4:
        mask = mask[0]
    mask = F.interpolate(mask[None, None].float(), size=(mask.shape[0], *size), mode="nearest")[0, 0]
    return mask.unsqueeze(0).repeat(3, 1, 1, 1).cpu()


def save_contact_sheet(
    path: Path,
    raw: torch.Tensor,
    mask: torch.Tensor,
    zero: torch.Tensor,
    true: torch.Tensor,
    shifted: torch.Tensor,
) -> None:
    raw = to_01(raw[0])
    zero = to_01(zero[0])
    true = to_01(true[0])
    shifted = to_01(shifted[0])
    mask_rgb = mask_to_rgb(mask[0].cpu(), size=raw.shape[-2:])
    overlay = (raw * 0.55 + torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1, 1) * mask_rgb * 0.45).clamp(0, 1)
    diff = (true - zero).abs().mul(3.0).clamp(0, 1)

    frames = sorted(set([0, raw.shape[1] // 4, raw.shape[1] // 2, raw.shape[1] * 3 // 4, raw.shape[1] - 1]))
    rows = [raw, overlay, zero, true, shifted, diff]
    tiles = []
    for row in rows:
        tiles.extend([row[:, idx] for idx in frames])
    grid = torchvision.utils.make_grid(tiles, nrow=len(frames), padding=2)
    grid = TF.resize(grid, [grid.shape[-2] * 2, grid.shape[-1] * 2])
    torchvision.utils.save_image(grid, path)


def diff_metrics(mask: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    # Compare generated videos a/b in pixel space, reporting whether changes are
    # concentrated in the requested mask region.
    a01 = to_01(a)
    b01 = to_01(b)
    diff = (a01 - b01).abs().mean(dim=1, keepdim=True)
    mask = F.interpolate(mask.detach().float().cpu(), size=diff.shape[-3:], mode="nearest").clamp(0, 1)
    inside = (diff * mask).sum() / (mask.sum() + 1e-6)
    outside_mask = 1.0 - mask
    outside = (diff * outside_mask).sum() / (outside_mask.sum() + 1e-6)
    return {
        "mean_abs_diff": float(diff.mean().item()),
        "inside_mask_abs_diff": float(inside.item()),
        "outside_mask_abs_diff": float(outside.item()),
        "inside_outside_ratio": float((inside / (outside + 1e-6)).item()),
        "mask_occupancy": float(mask.mean().item()),
    }


def main() -> None:
    args = parse_args()
    init_environment()

    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    user_opts = list(args.opts)
    if user_opts and user_opts[0] == "--":
        user_opts = user_opts[1:]
    opts = [
        "--",
        *user_opts,
        f"checkpoint.load_path={args.checkpoint}",
        "checkpoint.save_to_object_store.enabled=False",
        "checkpoint.load_from_object_store.enabled=False",
        "trainer.run_validation=False",
    ]
    config = override(config, opts)
    config.validate()
    config.freeze()

    trainer = config.trainer.type(config)
    model = instantiate(config.model)
    model = model.to("cuda", memory_format=config.trainer.memory_format)
    model.on_train_start(config.trainer.memory_format)
    optimizer, scheduler = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    grad_scaler = torch.amp.GradScaler("cuda", **config.trainer.grad_scaler_args)
    loaded_iter = trainer.checkpointer.load(model, optimizer, scheduler, grad_scaler)
    model.eval()

    dataloader = instantiate(config.dataloader_train)
    output_dir = Path(args.output_dir)
    if distributed.is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    saved = 0
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(dataloader):
            if batch_idx >= args.max_batches or saved >= args.num_samples:
                break
            if "target_mask" not in data_batch or float(data_batch["target_mask"].sum()) <= 0:
                continue
            data_batch = misc.to(data_batch, device="cuda")
            data_batch["num_conditional_frames"] = torch.full(
                (data_batch[model.input_data_key].shape[0],),
                args.num_conditional_frames,
                dtype=torch.long,
                device="cuda",
            )
            add_online_text_embeddings(model, data_batch)

            raw, x0, _ = model.get_data_and_condition(data_batch)
            state_shape = x0.shape[1:]

            zero_batch = clone_batch(data_batch)
            zero_batch["target_mask"] = torch.zeros_like(data_batch["target_mask"])
            shifted_batch = clone_batch(data_batch)
            shifted_batch["target_mask"] = make_shifted_mask(data_batch["target_mask"])

            with torch.autocast("cuda", dtype=torch.bfloat16):
                true_latent = model.generate_samples_from_batch(
                    data_batch,
                    guidance=args.guidance,
                    seed=args.seed,
                    state_shape=state_shape,
                    n_sample=x0.shape[0],
                    num_steps=args.num_steps,
                )
                zero_latent = model.generate_samples_from_batch(
                    zero_batch,
                    guidance=args.guidance,
                    seed=args.seed,
                    state_shape=state_shape,
                    n_sample=x0.shape[0],
                    num_steps=args.num_steps,
                )
                shifted_latent = model.generate_samples_from_batch(
                    shifted_batch,
                    guidance=args.guidance,
                    seed=args.seed,
                    state_shape=state_shape,
                    n_sample=x0.shape[0],
                    num_steps=args.num_steps,
                )
                true_video = model.decode(true_latent)
                zero_video = model.decode(zero_latent)
                shifted_video = model.decode(shifted_latent)

            sample_metrics = {
                "sample_index": saved,
                "loaded_iter": int(loaded_iter),
                "caption": data_batch.get(model.input_caption_key, [""])[0],
                "true_vs_zero": diff_metrics(data_batch["target_mask"], true_video, zero_video),
                "true_vs_shifted": diff_metrics(data_batch["target_mask"], true_video, shifted_video),
            }
            if distributed.is_rank0():
                sheet_path = output_dir / f"mask_guidance_sample_{saved:03d}.jpg"
                save_contact_sheet(
                    sheet_path,
                    raw,
                    data_batch["target_mask"],
                    zero_video,
                    true_video,
                    shifted_video,
                )
                sample_metrics["contact_sheet"] = str(sheet_path)
                print(json.dumps(sample_metrics, ensure_ascii=False), flush=True)
            results.append(sample_metrics)
            saved += 1

    if distributed.is_rank0():
        summary = {
            "checkpoint": args.checkpoint,
            "loaded_iter": int(loaded_iter),
            "num_samples": len(results),
            "num_steps": args.num_steps,
            "guidance": args.guidance,
            "seed": args.seed,
            "samples": results,
        }
        if results:
            for pair_name in ("true_vs_zero", "true_vs_shifted"):
                for metric_name in ("mean_abs_diff", "inside_mask_abs_diff", "outside_mask_abs_diff", "inside_outside_ratio"):
                    values = [sample[pair_name][metric_name] for sample in results]
                    summary[f"{pair_name}_{metric_name}_mean"] = float(sum(values) / len(values))
        with open(output_dir / "mask_guidance_eval_summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
