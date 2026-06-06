#!/usr/bin/env python3
"""Visualize TAViD-style target-token cross-attention maps in Cosmos.

Outputs two paper-style qualitative views:

1. ``effect_cross_attention_loss``:
   raw frames + target mask + target-token cross-attention.  If a baseline
   checkpoint is provided, it is shown side-by-side against the trained
   checkpoint to visualize the effect of the cross-attention alignment loss.

2. ``selective_cross_attention_loss``:
   block-wise target-token cross-attention maps for selected DiT blocks,
   matching the selective-block loss idea used by our TAViD-style config.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "packages" / "cosmos-oss"))

from cosmos_oss.init import init_environment
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils import distributed, misc
from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override


@dataclass
class BlockMetric:
    block: int
    mask_mass: float
    attn_mass_inside_mask: float
    attn_inside_mean: float
    attn_outside_mean: float
    inside_outside_ratio: float


@dataclass
class SampleRecord:
    sample_index: int
    caption: str
    tgt_token_index: int
    initial_attention_figure: str
    effect_figure: str
    selective_figure: str
    block_metrics: list[BlockMetric]


def parse_int_list(text: str) -> list[int]:
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint to visualize. Use 'latest' to load latest_checkpoint.txt from job.path_local.",
    )
    parser.add_argument("--baseline-checkpoint", default="", help="Optional no-attention-loss/baseline checkpoint for side-by-side comparison.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val", help="Dataset split used for attention visualization.")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--num-conditional-frames", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--blocks", default="8,12,16,20", help="Comma-separated DiT block ids to capture.")
    parser.add_argument("--selected-blocks", default="8,12,16,20", help="Blocks treated as selected by the loss.")
    parser.add_argument(
        "--model-only-load",
        action="store_true",
        help="Load only model weights. Use this for local single-GPU visualization with a consolidated .pt checkpoint.",
    )
    parser.add_argument(
        "--skip-init-environment",
        action="store_true",
        help="Skip Cosmos distributed initialization for local single-process visualization.",
    )
    parser.add_argument(
        "--token-source",
        choices=("config", "text", "feature", "text_feature"),
        default="config",
        help="Which conditioning tokens to visualize. 'config' keeps the checkpoint config setting.",
    )
    parser.add_argument(
        "--dummy-text-embeddings",
        action="store_true",
        help="Use zero text embeddings when offline embeddings are absent. Useful for feature-token attention debugging.",
    )
    parser.add_argument("--dummy-text-tokens", type=int, default=512)
    parser.add_argument(
        "--offload-denoiser-during-vae",
        action="store_true",
        help="Move the denoising network to CPU while VAE-encoding the visualization batch, then move it back.",
    )
    parser.add_argument("--sample-label", default="with_cross_attention_loss")
    parser.add_argument("--baseline-label", default="without_cross_attention_loss")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def add_online_text_embeddings(model, data_batch: dict) -> None:
    text_encoder_config = getattr(model.config, "text_encoder_config", None)
    if text_encoder_config is not None and text_encoder_config.compute_online:
        text_embeddings = model.text_encoder.compute_text_embeddings_online(data_batch, model.input_caption_key)
        data_batch["t5_text_embeddings"] = text_embeddings
        data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")


def add_dummy_text_embeddings(args: argparse.Namespace, model, data_batch: dict) -> None:
    if not args.dummy_text_embeddings or "t5_text_embeddings" in data_batch:
        return
    crossattn_proj = getattr(model.net, "crossattn_proj", None)
    if crossattn_proj is not None and len(crossattn_proj) > 0 and hasattr(crossattn_proj[0], "in_features"):
        dim = int(crossattn_proj[0].in_features)
    else:
        dim = int(getattr(model.net, "crossattn_emb_channels", 1024))
    batch_size = data_batch[model.input_data_key].shape[0]
    data_batch["t5_text_embeddings"] = torch.zeros(
        batch_size,
        args.dummy_text_tokens,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    data_batch["t5_text_mask"] = torch.zeros(
        batch_size,
        args.dummy_text_tokens,
        device="cuda",
        dtype=torch.long,
    )


def forward_with_optional_vae_offload(args: argparse.Namespace, model, data_batch: dict):
    if not args.offload_denoiser_during_vae:
        return model(data_batch)

    if model.config.text_encoder_config is not None and model.config.text_encoder_config.compute_online:
        text_embeddings = model.text_encoder.compute_text_embeddings_online(data_batch, model.input_caption_key)
        data_batch["t5_text_embeddings"] = text_embeddings
        data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

    model.net.to("cpu")
    torch.cuda.empty_cache()
    _, x0_B_C_T_H_W, condition = model.get_data_and_condition(data_batch)
    model.net.to("cuda")
    torch.cuda.empty_cache()

    epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **model.tensor_kwargs_fp32)
    batch_size = x0_B_C_T_H_W.size()[0]
    t_B = model.rectified_flow.sample_train_time(batch_size).to(**model.tensor_kwargs_fp32)
    t_B = rearrange(t_B, "b -> b 1")

    x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B = model.broadcast_split_for_model_parallelsim(
        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B
    )
    timesteps = model.rectified_flow.get_discrete_timestamp(t_B, model.tensor_kwargs_fp32)

    if model.config.use_high_sigma_strategy:
        raise NotImplementedError("High sigma strategy is not supported in visualization offload mode.")

    sigmas = model.rectified_flow.get_sigmas(timesteps, model.tensor_kwargs_fp32)
    timesteps = rearrange(timesteps, "b -> b 1")
    sigmas = rearrange(sigmas, "b -> b 1")
    xt_B_C_T_H_W, vt_B_C_T_H_W = model.rectified_flow.get_interpolation(
        epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas
    )

    vt_pred_B_C_T_H_W = model.denoise(
        noise=epsilon_B_C_T_H_W,
        xt_B_C_T_H_W=xt_B_C_T_H_W.to(**model.tensor_kwargs),
        timesteps_B_T=timesteps,
        condition=condition,
    )

    time_weights_B = model.rectified_flow.train_time_weight(timesteps, model.tensor_kwargs_fp32)
    per_instance_loss = torch.mean(
        (vt_pred_B_C_T_H_W - vt_B_C_T_H_W) ** 2,
        dim=list(range(1, vt_pred_B_C_T_H_W.dim())),
    )
    loss = torch.mean(time_weights_B * per_instance_loss)
    output_batch = {
        "x0": x0_B_C_T_H_W,
        "xt": xt_B_C_T_H_W,
        "sigma": sigmas,
        "condition": condition,
        "model_pred": vt_pred_B_C_T_H_W,
        "edm_loss": loss,
        "timesteps": timesteps,
        "per_instance_loss": per_instance_loss,
        "n_cond_frames": condition.num_conditional_frames_B,
    }
    if hasattr(model, "compute_extra_training_loss"):
        extra_output, extra_loss = model.compute_extra_training_loss(condition)
        output_batch.update(extra_output)
        loss = loss + extra_loss

    return output_batch, loss


def make_config(args: argparse.Namespace, checkpoint: str):
    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    user_opts = list(args.opts)
    if user_opts and user_opts[0] == "--":
        user_opts = user_opts[1:]
    opts = [
        "--",
        *user_opts,
        "checkpoint.save_to_object_store.enabled=False",
        "checkpoint.load_from_object_store.enabled=False",
        "checkpoint.load_training_state=False",
        "trainer.run_validation=False",
    ]
    if checkpoint.lower() not in {"latest", ""}:
        opts.append(f"checkpoint.load_path={checkpoint}")
        digest = hashlib.sha1(checkpoint.encode("utf-8")).hexdigest()[:10]
        opts.append(f"job.name=target_attention_viz_load_{digest}")
    config = override(config, opts)
    config.validate()
    config.freeze()
    if checkpoint.lower() == "latest":
        latest_file = Path(config.job.path_local) / "checkpoints" / "latest_checkpoint.txt"
        if not latest_file.exists():
            raise FileNotFoundError(
                f"--checkpoint latest requested, but {latest_file} does not exist. "
                "Pass a checkpoint path or set job.name/IMAGINAIRE_OUTPUT_ROOT to the training run."
            )
    return config


def load_model_and_dataloader(args: argparse.Namespace, checkpoint: str, blocks: list[int]):
    config = make_config(args, checkpoint)
    direct_model_load = args.model_only_load and checkpoint.lower().endswith(".pt")
    trainer = None if direct_model_load else config.trainer.type(config)
    model = instantiate(config.model)
    model = model.to("cuda", memory_format=config.trainer.memory_format)
    model.on_train_start(config.trainer.memory_format)
    if direct_model_load:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        load_info = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model-only checkpoint from {checkpoint}: {load_info}", flush=True)
        match = re.search(r"iter[_-]?0*(\d+)", checkpoint)
        loaded_iter = int(match.group(1)) if match else 0
    else:
        optimizer = scheduler = grad_scaler = None
        if not args.model_only_load:
            optimizer, scheduler = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
            grad_scaler = torch.amp.GradScaler("cuda", **config.trainer.grad_scaler_args)
        loaded_iter = trainer.checkpointer.load(model, optimizer, scheduler, grad_scaler)
    model.net.tavid_attn_alignment_blocks = set(blocks)
    if args.token_source != "config":
        model.net.tavid_attn_alignment_token_source = args.token_source
    model.eval()
    dataloader_cfg = config.dataloader_train if args.split == "train" else config.dataloader_val
    if args.skip_init_environment:
        dataset_cfg = getattr(dataloader_cfg, "dataset", None) or dataloader_cfg["dataset"]
        dataset = instantiate(dataset_cfg)
        batch_size = int(getattr(dataloader_cfg, "batch_size", 1))
        drop_last = bool(getattr(dataloader_cfg, "drop_last", False))
        collate_fn = getattr(dataloader_cfg, "collate_fn", None)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=None,
            num_workers=0,
            pin_memory=False,
            drop_last=drop_last,
            collate_fn=instantiate(collate_fn) if collate_fn is not None else None,
        )
    else:
        dataloader = instantiate(dataloader_cfg)
    return model, dataloader, int(loaded_iter)


def to_uint8_frame(frame: torch.Tensor) -> np.ndarray:
    """Convert [C,H,W] uint8 or [-1,1]/[0,1] tensor to RGB uint8."""
    frame = frame.detach().float().cpu()
    if frame.max() <= 2.0:
        if frame.min() < 0:
            frame = (frame.clamp(-1, 1) + 1.0) * 127.5
        else:
            frame = frame.clamp(0, 1) * 255.0
    return frame.clamp(0, 255).byte().permute(1, 2, 0).numpy()


def normalize_heatmap(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().cpu()
    flat = x.flatten()
    if flat.numel() > 1_000_000:
        step = int(np.ceil(flat.numel() / 1_000_000))
        flat = flat[::step]
    lo = torch.quantile(flat, 0.01)
    hi = torch.quantile(flat, 0.99)
    return ((x - lo) / (hi - lo + 1e-6)).clamp(0, 1)


def upsample_volume(volume_T_H_W: torch.Tensor, size: tuple[int, int, int], mode: str = "trilinear") -> torch.Tensor:
    return F.interpolate(volume_T_H_W[None, None].float(), size=size, mode=mode, align_corners=False if mode != "nearest" else None)[0, 0]


def heat_color(heat_H_W: torch.Tensor) -> np.ndarray:
    heat = heat_H_W.detach().float().cpu().clamp(0, 1).numpy()
    rgb = np.zeros((*heat.shape, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(1.8 * heat, 0, 1)
    rgb[..., 1] = np.clip(1.8 * (1.0 - np.abs(heat - 0.55) / 0.55), 0, 1)
    rgb[..., 2] = np.clip(1.4 * (1.0 - heat), 0, 1) * (heat > 0.05)
    return (rgb * 255).astype(np.uint8)


def overlay_heat(frame_rgb: np.ndarray, heat_H_W: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    heat_rgb = heat_color(heat_H_W)
    heat_mask = heat_H_W.detach().float().cpu().clamp(0, 1).numpy()[..., None]
    out = frame_rgb.astype(np.float32) * (1.0 - alpha * heat_mask) + heat_rgb.astype(np.float32) * (alpha * heat_mask)
    return out.clip(0, 255).astype(np.uint8)


def overlay_mask(frame_rgb: np.ndarray, mask_H_W: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    mask = mask_H_W.detach().float().cpu().clamp(0, 1).numpy()[..., None]
    red = np.zeros_like(frame_rgb, dtype=np.float32)
    red[..., 0] = 255
    out = frame_rgb.astype(np.float32) * (1.0 - alpha * mask) + red * (alpha * mask)
    return out.clip(0, 255).astype(np.uint8)


def labeled_tile(image: np.ndarray, label: str, scale: int = 1) -> Image.Image:
    pil = Image.fromarray(image).resize((image.shape[1] * scale, image.shape[0] * scale))
    draw = ImageDraw.Draw(pil)
    draw.rectangle((0, 0, pil.width, 22), fill=(0, 0, 0))
    draw.text((5, 5), label, fill=(255, 255, 255))
    return pil


def save_horizontal_tiles(path: Path, tiles: list[tuple[str, np.ndarray]]) -> None:
    if not tiles:
        return
    tile_w = tiles[0][1].shape[1]
    tile_h = tiles[0][1].shape[0]
    header_h = 24
    canvas = Image.new("RGB", (tile_w * len(tiles), header_h + tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(tiles):
        x = idx * tile_w
        draw.rectangle((x, 0, x + tile_w, header_h), fill=(0, 0, 0))
        draw.text((x + 6, 6), label, fill=(255, 255, 255))
        canvas.paste(Image.fromarray(image), (x, header_h))
    canvas.save(path, quality=95)


def save_grid(path: Path, rows: list[tuple[str, list[np.ndarray]]], frame_labels: list[str]) -> None:
    if not rows:
        return
    tile_w = rows[0][1][0].shape[1]
    tile_h = rows[0][1][0].shape[0]
    label_w = 360
    header_h = 48
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    except Exception:
        font = None
        small_font = None
    canvas = Image.new("RGB", (label_w + tile_w * len(frame_labels), header_h + tile_h * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(frame_labels):
        draw.text((label_w + col * tile_w + 12, 8), label, fill=(0, 0, 0), font=small_font)
    for row_idx, (row_label, images) in enumerate(rows):
        y = header_h + row_idx * tile_h
        draw.rectangle((0, y, label_w, y + tile_h), fill=(245, 245, 245))
        draw.text((14, y + 14), row_label[:32], fill=(0, 0, 0), font=font)
        for col_idx, image in enumerate(images):
            canvas.paste(Image.fromarray(image), (label_w + col_idx * tile_w, y))
    canvas.save(path, quality=95)


def compute_metrics(attn_T_H_W: torch.Tensor, mask_T_H_W: torch.Tensor, block: int) -> BlockMetric:
    attn = attn_T_H_W.detach().float().cpu().clamp(min=0)
    mask = mask_T_H_W.detach().float().cpu().clamp(0, 1)
    if mask.shape != attn.shape:
        mask = upsample_volume(mask, tuple(attn.shape), mode="nearest")
    attn_sum = attn.sum() + 1e-6
    mask_sum = mask.sum() + 1e-6
    inv = 1.0 - mask
    inv_sum = inv.sum() + 1e-6
    inside_mean = (attn * mask).sum() / mask_sum
    outside_mean = (attn * inv).sum() / inv_sum
    return BlockMetric(
        block=block,
        mask_mass=float(mask.mean().item()),
        attn_mass_inside_mask=float(((attn * mask).sum() / attn_sum).item()),
        attn_inside_mean=float(inside_mean.item()),
        attn_outside_mean=float(outside_mean.item()),
        inside_outside_ratio=float((inside_mean / (outside_mean + 1e-6)).item()),
    )


def collect_maps(
    args: argparse.Namespace,
    checkpoint: str,
    label: str,
    blocks: list[int],
    output_dir: Path,
) -> tuple[list[dict], int]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, dataloader, loaded_iter = load_model_and_dataloader(args, checkpoint, blocks)
    block_set = set(blocks)
    block_ids = [idx for idx in range(len(model.net.blocks)) if idx in block_set]
    records = []
    saved = 0
    with torch.no_grad():
        for batch_idx, cpu_batch in enumerate(dataloader):
            if batch_idx >= args.max_batches or saved >= args.num_samples:
                break
            if "target_mask" not in cpu_batch or float(cpu_batch["target_mask"].sum()) <= 0:
                continue
            raw_cpu = cpu_batch[model.input_data_key][0].detach().cpu()
            mask_cpu = cpu_batch["target_mask"][0].detach().cpu()
            caption = cpu_batch.get(model.input_caption_key, [""])[0]

            data_batch = misc.to(cpu_batch, device="cuda")
            data_batch["num_conditional_frames"] = torch.full(
                (data_batch[model.input_data_key].shape[0],),
                args.num_conditional_frames,
                dtype=torch.long,
                device="cuda",
            )
            add_online_text_embeddings(model, data_batch)
            add_dummy_text_embeddings(args, model, data_batch)
            batch_size = data_batch[model.input_data_key].shape[0]
            tgt_indices = data_batch.get("tgt_token_indices", torch.full((batch_size,), -1, device="cuda"))
            if tgt_indices.ndim > 1:
                tgt_for_skip = tgt_indices[:, 0]
            else:
                tgt_for_skip = tgt_indices
            token_source = getattr(model.net, "tavid_attn_alignment_token_source", "text")
            has_text_target = bool((tgt_for_skip >= 0).any().item())
            has_feature_target = "target_feature" in data_batch and bool(
                (data_batch["target_feature"].detach().float().abs().sum(dim=-1) > 0).any().item()
            )
            if token_source == "text" and not has_text_target:
                continue
            if token_source == "feature" and not has_feature_target:
                continue
            if token_source == "text_feature" and not (has_text_target or has_feature_target):
                continue
            sample_tgt_index = int(tgt_for_skip[0].item()) if tgt_for_skip.numel() else -1

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_batch, loss = forward_with_optional_vae_offload(args, model, data_batch)

            attn_maps = [item[0].detach().float().cpu() for item in getattr(model.net, "tavid_target_attn_maps", [])]
            latent_mask = getattr(model.net, "tavid_target_mask_B_T_H_W", None)
            latent_mask_cpu = latent_mask[0].detach().float().cpu() if latent_mask is not None else None
            if latent_mask_cpu is None or float(latent_mask_cpu.sum()) <= 0:
                continue
            metrics = [
                compute_metrics(attn_map, latent_mask_cpu, block)
                for block, attn_map in zip(block_ids, attn_maps)
            ]
            records.append(
                {
                    "label": label,
                    "sample_index": saved,
                    "caption": caption,
                    "raw": raw_cpu,
                    "mask": mask_cpu,
                    "latent_mask": latent_mask_cpu,
                    "block_ids": block_ids[: len(attn_maps)],
                    "attn_maps": attn_maps,
                    "tgt_token_index": sample_tgt_index,
                    "loaded_iter": loaded_iter,
                    "loss": float(loss.detach().float().item()),
                    "extra_losses": {
                        key: float(value.detach().float().item())
                        for key, value in output_batch.items()
                        if key.startswith("target_attention_") and torch.is_tensor(value)
                    },
                    "metrics": metrics,
                }
            )
            print(json.dumps({
                "label": label,
                "sample_index": saved,
                "caption": caption,
                "tgt_token_index": sample_tgt_index,
                "metrics": [asdict(item) for item in metrics],
            }, ensure_ascii=False), flush=True)
            saved += 1

    del model, dataloader
    gc.collect()
    torch.cuda.empty_cache()
    return records, loaded_iter


def selected_mean_attention(record: dict, selected_blocks: set[int]) -> torch.Tensor:
    selected = [
        attn for block, attn in zip(record["block_ids"], record["attn_maps"])
        if block in selected_blocks
    ]
    if not selected:
        selected = record["attn_maps"]
    return torch.stack(selected).mean(dim=0)


def frame_indices(num_frames: int) -> list[int]:
    return sorted(set([0, num_frames // 4, num_frames // 2, 3 * num_frames // 4, num_frames - 1]))


def make_effect_figure(path: Path, main: dict, baseline: dict | None, selected_blocks: set[int]) -> None:
    raw = main["raw"]
    mask = main["mask"][0]
    T_raw = raw.shape[1]
    H_raw, W_raw = raw.shape[-2:]
    frames = frame_indices(T_raw)
    frame_labels = [f"f{idx}" for idx in frames]

    raw_frames = [to_uint8_frame(raw[:, idx]) for idx in frames]
    mask_up = upsample_volume(mask, (T_raw, H_raw, W_raw), mode="nearest")
    rows: list[tuple[str, list[np.ndarray]]] = [
        ("RGB", raw_frames),
        ("target mask", [overlay_mask(raw_frames[i], mask_up[idx]) for i, idx in enumerate(frames)]),
    ]

    if baseline is not None:
        base_attn = normalize_heatmap(upsample_volume(selected_mean_attention(baseline, selected_blocks), (T_raw, H_raw, W_raw)))
        rows.append((
            baseline["label"],
            [overlay_heat(raw_frames[i], base_attn[idx]) for i, idx in enumerate(frames)],
        ))

    main_attn = normalize_heatmap(upsample_volume(selected_mean_attention(main, selected_blocks), (T_raw, H_raw, W_raw)))
    rows.append((main["label"], [overlay_heat(raw_frames[i], main_attn[idx]) for i, idx in enumerate(frames)]))

    if baseline is not None:
        delta = (main_attn - base_attn).clamp(min=0)
        rows.append(("attention gain", [overlay_heat(raw_frames[i], normalize_heatmap(delta)[idx]) for i, idx in enumerate(frames)]))

    save_grid(path, rows, frame_labels)


def make_initial_frame_figure(path: Path, record: dict, selected_blocks: set[int], frame_idx: int = 0) -> None:
    raw = record["raw"]
    mask = record["mask"][0]
    T_raw = raw.shape[1]
    H_raw, W_raw = raw.shape[-2:]
    frame_idx = max(0, min(frame_idx, T_raw - 1))
    frame_rgb = to_uint8_frame(raw[:, frame_idx])
    mask_up = upsample_volume(mask, (T_raw, H_raw, W_raw), mode="nearest")
    attn_up = normalize_heatmap(
        upsample_volume(selected_mean_attention(record, selected_blocks), (T_raw, H_raw, W_raw))
    )
    heat_rgb = heat_color(attn_up[frame_idx])
    save_horizontal_tiles(
        path,
        [
            ("initial frame", frame_rgb),
            ("target mask", overlay_mask(frame_rgb, mask_up[frame_idx])),
            ("cross-attention", heat_rgb),
            ("attention overlay", overlay_heat(frame_rgb, attn_up[frame_idx])),
        ],
    )


def make_selective_figure(path: Path, record: dict, selected_blocks: set[int]) -> None:
    raw = record["raw"]
    mask = record["mask"][0]
    T_raw = raw.shape[1]
    H_raw, W_raw = raw.shape[-2:]
    frames = frame_indices(T_raw)
    frame_labels = [f"f{idx}" for idx in frames]
    raw_frames = [to_uint8_frame(raw[:, idx]) for idx in frames]
    mask_up = upsample_volume(mask, (T_raw, H_raw, W_raw), mode="nearest")

    rows: list[tuple[str, list[np.ndarray]]] = [
        ("RGB", raw_frames),
        ("target mask", [overlay_mask(raw_frames[i], mask_up[idx]) for i, idx in enumerate(frames)]),
    ]
    for block, attn in zip(record["block_ids"], record["attn_maps"]):
        heat = normalize_heatmap(upsample_volume(attn, (T_raw, H_raw, W_raw)))
        prefix = "selected" if block in selected_blocks else "not selected"
        rows.append((f"{prefix} block {block}", [overlay_heat(raw_frames[i], heat[idx]) for i, idx in enumerate(frames)]))

    mean_heat = normalize_heatmap(upsample_volume(selected_mean_attention(record, selected_blocks), (T_raw, H_raw, W_raw)))
    rows.append(("selected mean", [overlay_heat(raw_frames[i], mean_heat[idx]) for i, idx in enumerate(frames)]))
    save_grid(path, rows, frame_labels)


def main() -> None:
    args = parse_args()
    if not args.skip_init_environment:
        init_environment()
    output_dir = Path(args.output_dir)
    if distributed.is_rank0():
        output_dir.mkdir(parents=True, exist_ok=True)

    blocks = parse_int_list(args.blocks)
    selected_blocks = set(parse_int_list(args.selected_blocks))
    if not blocks:
        raise ValueError("--blocks cannot be empty")

    baseline_records = None
    baseline_loaded_iter = None
    if args.baseline_checkpoint:
        baseline_records, baseline_loaded_iter = collect_maps(
            args, args.baseline_checkpoint, args.baseline_label, blocks, output_dir
        )

    main_records, loaded_iter = collect_maps(args, args.checkpoint, args.sample_label, blocks, output_dir)
    if not distributed.is_rank0():
        return

    summaries = []
    for idx, main_record in enumerate(main_records):
        baseline_record = baseline_records[idx] if baseline_records and idx < len(baseline_records) else None
        initial_path = output_dir / f"sample_{idx:03d}_initial_frame_cross_attention.jpg"
        effect_path = output_dir / f"sample_{idx:03d}_effect_cross_attention_loss.jpg"
        selective_path = output_dir / f"sample_{idx:03d}_selective_cross_attention_loss.jpg"
        make_initial_frame_figure(initial_path, main_record, selected_blocks)
        make_effect_figure(effect_path, main_record, baseline_record, selected_blocks)
        make_selective_figure(selective_path, main_record, selected_blocks)
        summaries.append(
            SampleRecord(
                sample_index=idx,
                caption=main_record["caption"],
                tgt_token_index=main_record["tgt_token_index"],
                initial_attention_figure=str(initial_path),
                effect_figure=str(effect_path),
                selective_figure=str(selective_path),
                block_metrics=main_record["metrics"],
            )
        )

    summary = {
        "checkpoint": args.checkpoint,
        "baseline_checkpoint": args.baseline_checkpoint or None,
        "loaded_iter": loaded_iter,
        "baseline_loaded_iter": baseline_loaded_iter,
        "split": args.split,
        "token_source": args.token_source,
        "blocks": blocks,
        "selected_blocks": sorted(selected_blocks),
        "num_samples": len(summaries),
        "samples": [
            {
                **asdict(item),
                "block_metrics": [asdict(metric) for metric in item.block_metrics],
            }
            for item in summaries
        ],
    }
    (output_dir / "cross_attention_visualization_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
