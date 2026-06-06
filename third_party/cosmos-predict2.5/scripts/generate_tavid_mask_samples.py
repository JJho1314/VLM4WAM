#!/usr/bin/env python3
"""Generate TAViD-style qualitative samples with target-mask conditioning."""

from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import tqdm
from megatron.core import parallel_state

from cosmos_oss.init import init_environment
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils import distributed, misc
from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.inference.utils import write_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--skip-samples", type=int, default=0)
    parser.add_argument("--sample-index-offset", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--num-conditional-frames", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=400)
    parser.add_argument("--standalone-only", action="store_true", help="Only save generated/GT videos and captions.")
    parser.add_argument(
        "--reuse-encoded-latent",
        action="store_true",
        help="Reuse the initial VAE latent when building the inference velocity function to reduce peak memory.",
    )
    parser.add_argument(
        "--offload-denoiser-before-decode",
        action="store_true",
        help="Move the denoising network to CPU before VAE decode. Intended for single-sample memory-constrained evals.",
    )
    parser.add_argument(
        "--offload-denoiser-during-vae",
        action="store_true",
        help="Move the denoising network to CPU while VAE-encoding the conditioning video, then move it back for sampling.",
    )
    parser.add_argument(
        "--target-feature-mode",
        choices=["keep", "zero", "drop", "path"],
        default="keep",
        help="Evaluation-only override for target_feature ablations.",
    )
    parser.add_argument(
        "--target-feature-path",
        type=Path,
        default=None,
        help="Feature .pt to use when --target-feature-mode=path.",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def add_online_text_embeddings(model, data_batch: dict) -> None:
    text_encoder_config = getattr(model.config, "text_encoder_config", None)
    if text_encoder_config is not None and text_encoder_config.compute_online:
        text_embeddings = model.text_encoder.compute_text_embeddings_online(data_batch, model.input_caption_key)
        data_batch["t5_text_embeddings"] = text_embeddings
        data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")


def checkpoint_iter(path: str) -> int | None:
    match = re.search(r"iter_(\d+)", path)
    return int(match.group(1)) if match else None


def to_01(video: torch.Tensor) -> torch.Tensor:
    return ((video.detach().float().cpu().clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)


def mask_to_rgb(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    # mask: [1, T, H, W] or [T, H, W]
    if mask.ndim == 4:
        mask = mask[0]
    mask = F.interpolate(mask[None, None].float(), size=(mask.shape[0], *size), mode="nearest")[0, 0]
    return mask.unsqueeze(0).repeat(3, 1, 1, 1).cpu()


def make_overlay(raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raw01 = to_01(raw)
    mask_rgb = mask_to_rgb(mask.cpu(), size=raw01.shape[-2:])
    red = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1, 1)
    return (raw01 * 0.55 + red * mask_rgb * 0.45).clamp(0, 1)


def video_to_uint8(video: torch.Tensor) -> np.ndarray:
    # input: [3, T, H, W] in [0, 1]
    video = video.detach().float().cpu().clamp(0, 1)
    return (video.permute(1, 2, 3, 0).numpy() * 255.0).round().astype(np.uint8)


def save_contact_sheet(path: Path, raw: torch.Tensor, overlay: torch.Tensor, generated: torch.Tensor) -> None:
    raw = to_01(raw)
    generated = to_01(generated)
    frames = sorted(set([0, raw.shape[1] // 4, raw.shape[1] // 2, raw.shape[1] * 3 // 4, raw.shape[1] - 1]))
    rows = [raw, overlay, generated]
    tiles = []
    for row in rows:
        tiles.extend([row[:, idx] for idx in frames])
    grid = torchvision.utils.make_grid(tiles, nrow=len(frames), padding=2)
    grid = TF.resize(grid, [grid.shape[-2] * 2, grid.shape[-1] * 2])
    torchvision.utils.save_image(grid, path)


def save_sample_outputs(
    output_dir: Path,
    sample_index: int,
    raw: torch.Tensor,
    mask: torch.Tensor,
    generated: torch.Tensor,
    caption: str,
    fps: int,
    standalone_only: bool = False,
) -> dict:
    stem = f"sample_{sample_index:03d}"
    raw01 = to_01(raw)
    gen01 = to_01(generated)

    generated_np = video_to_uint8(gen01)
    gt_np = video_to_uint8(raw01)

    generated_path = output_dir / f"{stem}_generated.mp4"
    gt_path = output_dir / f"{stem}_gt.mp4"
    caption_path = output_dir / f"{stem}_caption.txt"

    write_video(str(generated_path), generated_np, fps=fps, lossless=False)
    write_video(str(gt_path), gt_np, fps=fps, lossless=False)
    caption_path.write_text(caption + "\n")

    record = {
        "sample_index": sample_index,
        "caption": caption,
        "generated": str(generated_path),
        "gt": str(gt_path),
        "caption_file": str(caption_path),
    }
    if standalone_only:
        return record

    overlay = make_overlay(raw, mask)
    overlay_np = video_to_uint8(overlay)
    grid_np = np.concatenate([overlay_np, generated_np, gt_np], axis=2)
    overlay_path = output_dir / f"{stem}_mask_overlay.mp4"
    grid_path = output_dir / f"{stem}_overlay_generated_gt.mp4"
    sheet_path = output_dir / f"{stem}_contact.jpg"
    write_video(str(overlay_path), overlay_np, fps=fps, lossless=False)
    write_video(str(grid_path), grid_np, fps=fps, lossless=False)
    save_contact_sheet(sheet_path, raw, overlay, generated)
    record.update(
        {
            "mask_overlay": str(overlay_path),
            "overlay_generated_gt": str(grid_path),
            "contact_sheet": str(sheet_path),
        }
    )
    return record


def _extract_feature_from_payload(payload) -> torch.Tensor:
    if isinstance(payload, dict):
        for key in ("target_feature", "features", "feature_B_L_D", "feature", "seg_output_embeddings"):
            if key in payload:
                payload = payload[key]
                break
    feature = torch.as_tensor(payload, dtype=torch.float32)
    if feature.ndim == 1:
        feature = feature.view(1, -1)
    if feature.ndim == 3 and feature.shape[0] == 1:
        feature = feature[0]
    if feature.ndim != 2:
        raise ValueError(f"Expected override target feature [L,D] or [1,L,D], got {tuple(feature.shape)}")
    return torch.nan_to_num(feature).contiguous()


def _match_feature_shape(feature: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if reference.ndim == 2:
        max_tokens, feature_dim = reference.shape
        batch = None
    elif reference.ndim == 3:
        batch, max_tokens, feature_dim = reference.shape
    else:
        raise ValueError(f"Unsupported reference target_feature shape: {tuple(reference.shape)}")
    if feature.shape[-1] != feature_dim:
        raise ValueError(f"Override feature dim {feature.shape[-1]} does not match batch dim {feature_dim}")
    if feature.shape[0] > max_tokens:
        feature = feature[:max_tokens]
    elif feature.shape[0] < max_tokens:
        pad = torch.zeros(max_tokens - feature.shape[0], feature_dim, dtype=feature.dtype)
        feature = torch.cat([feature, pad], dim=0)
    if batch is not None:
        feature = feature.unsqueeze(0).expand(batch, -1, -1).contiguous()
    return feature.to(device=reference.device, dtype=reference.dtype)


def apply_target_feature_override(args: argparse.Namespace, data_batch: dict) -> str:
    mode = args.target_feature_mode
    if mode == "keep":
        return "keep"
    if mode == "drop":
        data_batch.pop("target_feature", None)
        return "drop"
    reference = data_batch.get("target_feature", None)
    if reference is None:
        raise ValueError(f"--target-feature-mode={mode} requires a target_feature in the batch")
    if mode == "zero":
        data_batch["target_feature"] = torch.zeros_like(reference)
        return "zero"
    if mode == "path":
        if args.target_feature_path is None:
            raise ValueError("--target-feature-mode=path requires --target-feature-path")
        payload = torch.load(args.target_feature_path, map_location="cpu", weights_only=False)
        feature = _extract_feature_from_payload(payload)
        data_batch["target_feature"] = _match_feature_shape(feature, reference)
        return f"path:{args.target_feature_path}"
    raise ValueError(mode)


def get_velocity_fn_from_preencoded_batch(
    model,
    data_batch: dict,
    x0: torch.Tensor,
    guidance: float,
    is_negative_prompt: bool = False,
):
    num_conditional_frames = data_batch.get("num_conditional_frames", 1)

    if is_negative_prompt:
        condition, uncondition = model.conditioner.get_condition_with_negative_prompt(data_batch)
    else:
        condition, uncondition = model.conditioner.get_condition_uncondition(data_batch)

    is_image_batch = model.is_image_batch(data_batch)
    condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
    uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

    condition = condition.set_video_condition(
        gt_frames=x0,
        random_min_num_conditional_frames=model.config.min_num_conditional_frames,
        random_max_num_conditional_frames=model.config.max_num_conditional_frames,
        num_conditional_frames=num_conditional_frames,
        conditional_frames_probs=model.config.conditional_frames_probs,
    )
    uncondition = uncondition.set_video_condition(
        gt_frames=x0,
        random_min_num_conditional_frames=model.config.min_num_conditional_frames,
        random_max_num_conditional_frames=model.config.max_num_conditional_frames,
        num_conditional_frames=num_conditional_frames,
        conditional_frames_probs=model.config.conditional_frames_probs,
    )
    condition = condition.edit_for_inference(is_cfg_conditional=True, num_conditional_frames=num_conditional_frames)
    uncondition = uncondition.edit_for_inference(
        is_cfg_conditional=False, num_conditional_frames=num_conditional_frames
    )

    target_mask = data_batch.get("target_mask", None)
    if target_mask is not None:
        target_mask = target_mask.to(device=x0.device, dtype=x0.dtype)
        target_mask = F.interpolate(target_mask, size=x0.shape[2:], mode="nearest")
        if model.config.target_mask_condition_frames_only:
            target_mask = target_mask * condition.condition_video_input_mask_B_C_T_H_W.type_as(target_mask)
        condition = condition.set_target_mask(target_mask)

    target_feature = data_batch.get("target_feature", None)
    if target_feature is not None:
        target_feature = target_feature.to(device=x0.device, dtype=x0.dtype)
        condition = condition.set_target_feature(target_feature)

    tgt_token_indices = data_batch.get("tgt_token_indices", None)
    if tgt_token_indices is not None:
        condition = condition.set_tgt_token_indices(tgt_token_indices.to(device=x0.device, dtype=torch.long))

    _, condition, _, _ = model.broadcast_split_for_model_parallelsim(x0, condition, None, None)
    _, uncondition, _, _ = model.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

    if not parallel_state.is_initialized():
        assert not model.net.is_context_parallel_enabled, (
            "parallel_state is not initialized, context parallel should be turned off."
        )

    def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        cond_v = model.denoise(noise, noise_x, timestep, condition)
        uncond_v = model.denoise(noise, noise_x, timestep, uncondition)
        return cond_v + guidance * (cond_v - uncond_v)

    return velocity_fn


def generate_samples_from_preencoded_batch(
    model,
    data_batch: dict,
    x0: torch.Tensor,
    guidance: float,
    seed: int,
    state_shape: tuple[int, ...],
    n_sample: int,
    num_steps: int,
    shift: float = 5.0,
) -> torch.Tensor:
    noise = misc.arch_invariant_rand(
        (n_sample,) + tuple(state_shape),
        torch.float32,
        model.tensor_kwargs["device"],
        seed,
    )
    seed_g = torch.Generator(device=model.tensor_kwargs["device"])
    seed_g.manual_seed(seed)

    model.sample_scheduler.set_timesteps(
        num_steps,
        device=model.tensor_kwargs["device"],
        shift=shift,
        use_kerras_sigma=model.config.use_kerras_sigma_at_inference,
    )

    velocity_fn = get_velocity_fn_from_preencoded_batch(data_batch=data_batch, model=model, x0=x0, guidance=guidance)
    latents = noise
    for t in tqdm.tqdm(model.sample_scheduler.timesteps, desc="Generating samples", total=num_steps):
        timestep = torch.stack([t])
        velocity_pred = velocity_fn(noise, latents, timestep.unsqueeze(0))
        temp_x0 = model.sample_scheduler.step(
            velocity_pred.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False, generator=seed_g
        )[0]
        latents = temp_x0.squeeze(0)

    return latents


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
        "checkpoint.load_training_state=False",
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

    records = []
    saved = 0
    skipped = 0
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(dataloader):
            if batch_idx >= args.max_batches or saved >= args.num_samples:
                break
            if "target_mask" not in data_batch or float(data_batch["target_mask"].sum()) <= 0:
                continue
            if skipped < args.skip_samples:
                skipped += 1
                continue

            data_batch = misc.to(data_batch, device="cuda")
            feature_override = apply_target_feature_override(args, data_batch)
            data_batch["num_conditional_frames"] = torch.full(
                (data_batch[model.input_data_key].shape[0],),
                args.num_conditional_frames,
                dtype=torch.long,
                device="cuda",
            )
            add_online_text_embeddings(model, data_batch)

            if args.offload_denoiser_during_vae:
                model.net.to("cpu")
                torch.cuda.empty_cache()
            raw, x0, _ = model.get_data_and_condition(data_batch)
            if args.offload_denoiser_during_vae:
                model.net.to("cuda")
                torch.cuda.empty_cache()
            state_shape = x0.shape[1:]
            n_sample = x0.shape[0]
            caption = data_batch.get(model.input_caption_key, [""])[0]
            raw_for_save = raw[0].detach().cpu()
            mask_for_save = data_batch["target_mask"][0].detach().cpu()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.reuse_encoded_latent:
                    latent = generate_samples_from_preencoded_batch(
                        model,
                        data_batch=data_batch,
                        x0=x0,
                        guidance=args.guidance,
                        seed=args.seed + saved,
                        state_shape=state_shape,
                        n_sample=n_sample,
                        num_steps=args.num_steps,
                    )
                else:
                    latent = model.generate_samples_from_batch(
                        data_batch,
                        guidance=args.guidance,
                        seed=args.seed + saved,
                        state_shape=state_shape,
                        n_sample=n_sample,
                        num_steps=args.num_steps,
                    )
                del raw, x0, data_batch
                if args.offload_denoiser_before_decode:
                    model.net.to("cpu")
                torch.cuda.empty_cache()
                generated = model.decode(latent)

            if distributed.is_rank0():
                record = save_sample_outputs(
                    output_dir=output_dir,
                    sample_index=args.sample_index_offset + saved,
                    raw=raw_for_save,
                    mask=mask_for_save,
                    generated=generated[0],
                    caption=caption,
                    fps=args.fps,
                    standalone_only=args.standalone_only,
                )
                record["target_feature_override"] = feature_override
                print(json.dumps(record, ensure_ascii=False), flush=True)
                records.append(record)
            saved += 1

    if distributed.is_rank0():
        summary = {
            "checkpoint": args.checkpoint,
            "checkpoint_iter_from_path": checkpoint_iter(args.checkpoint),
            "loaded_iter_returned_by_checkpointer": int(loaded_iter),
            "num_samples": len(records),
            "skip_samples": args.skip_samples,
            "sample_index_offset": args.sample_index_offset,
            "num_steps": args.num_steps,
            "guidance": args.guidance,
            "seed": args.seed,
            "fps": args.fps,
            "layout": "standalone generated/GT videos" if args.standalone_only else "overlay_generated_gt mp4 columns are: mask overlay, generated, ground truth",
            "target_feature_mode": args.target_feature_mode,
            "target_feature_path": str(args.target_feature_path) if args.target_feature_path is not None else None,
            "samples": records,
        }
        with open(output_dir / "tavid_generation_summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
