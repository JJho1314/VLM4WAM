#!/usr/bin/env python3
"""Intermediate diagnostics for VLM target-feature guidance.

This is an analysis tool, not an inference path.  It can keep the GT target
mask only to measure whether attention/logit/delta maps concentrate on the
target region; generation ablations should use generate_tavid_mask_samples.py
with --remove-target-mask.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "packages" / "cosmos-oss"))

import visualize_tavid_cross_attention as attnviz
from generate_tavid_mask_samples import apply_target_feature_override


def parse_int_list(text: str) -> list[int]:
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--num-conditional-frames", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--blocks", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27")
    parser.add_argument("--selected-blocks", default="")
    parser.add_argument("--variants", nargs="+", default=["keep", "zero", "drop", "wrong"])
    parser.add_argument("--wrong-target-feature-path", type=Path, default=None)
    parser.add_argument("--wrong-target-dense-feature-path", type=Path, default=None)
    parser.add_argument("--model-only-load", action="store_true")
    parser.add_argument("--skip-init-environment", action="store_true")
    parser.add_argument("--token-source", choices=("config", "text", "feature", "text_feature"), default="config")
    parser.add_argument("--dummy-text-embeddings", action="store_true")
    parser.add_argument("--dummy-text-tokens", type=int, default=512)
    parser.add_argument("--offload-denoiser-during-vae", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def clone_cpu_batch(batch: dict) -> dict:
    cloned = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().cpu().clone()
        elif isinstance(value, list):
            cloned[key] = list(value)
        elif isinstance(value, tuple):
            cloned[key] = tuple(value)
        else:
            cloned[key] = value
    return cloned


def override_args_for_variant(args: argparse.Namespace, variant: str) -> SimpleNamespace:
    if variant == "keep":
        return SimpleNamespace(target_feature_mode="keep", target_feature_path=None, target_dense_feature_path=None)
    if variant == "zero":
        return SimpleNamespace(target_feature_mode="zero", target_feature_path=None, target_dense_feature_path=None)
    if variant == "drop":
        return SimpleNamespace(target_feature_mode="drop", target_feature_path=None, target_dense_feature_path=None)
    if variant == "wrong":
        return SimpleNamespace(
            target_feature_mode="path",
            target_feature_path=args.wrong_target_feature_path,
            target_dense_feature_path=args.wrong_target_dense_feature_path,
        )
    if variant.startswith("path:"):
        return SimpleNamespace(
            target_feature_mode="path",
            target_feature_path=None,
            target_dense_feature_path=Path(variant.split(":", 1)[1]),
        )
    raise ValueError(f"Unsupported variant {variant!r}")


def inside_outside_stats(volume_T_H_W: torch.Tensor, mask_T_H_W: torch.Tensor, prefix: str) -> dict[str, float]:
    volume = volume_T_H_W.detach().float().cpu()
    mask = mask_T_H_W.detach().float().cpu().clamp(0, 1)
    if mask.shape != volume.shape:
        mask = attnviz.upsample_volume(mask, tuple(volume.shape), mode="nearest")
    binary = mask > 0.5
    inv = ~binary
    out = {
        f"{prefix}_mean": float(volume.mean().item()),
        f"{prefix}_max": float(volume.max().item()),
        f"{prefix}_mask_area": float(binary.float().mean().item()),
    }
    if binary.any():
        out[f"{prefix}_inside_mean"] = float(volume[binary].mean().item())
    if inv.any():
        out[f"{prefix}_outside_mean"] = float(volume[inv].mean().item())
    if binary.any() and inv.any():
        inside = volume[binary].mean()
        outside = volume[inv].mean()
        out[f"{prefix}_inside_outside_ratio"] = float((inside / (outside + 1e-6)).item())
        out[f"{prefix}_inside_minus_outside"] = float((inside - outside).item())
    return out


def selected_mean_attention(record: dict, selected_blocks: set[int]) -> torch.Tensor | None:
    if not record["attn_maps"]:
        return None
    selected = [
        attn
        for block, attn in zip(record["block_ids"], record["attn_maps"])
        if not selected_blocks or block in selected_blocks
    ]
    if not selected:
        selected = record["attn_maps"]
    return torch.stack(selected).mean(dim=0)


def collect_one_variant(
    args: argparse.Namespace,
    model,
    cpu_batch: dict,
    variant: str,
    blocks: list[int],
) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_batch = attnviz.misc.to(clone_cpu_batch(cpu_batch), device="cuda")
    feature_override = apply_target_feature_override(override_args_for_variant(args, variant), data_batch)
    data_batch["num_conditional_frames"] = torch.full(
        (data_batch[model.input_data_key].shape[0],),
        args.num_conditional_frames,
        dtype=torch.long,
        device="cuda",
    )
    attnviz.add_online_text_embeddings(model, data_batch)
    attnviz.add_dummy_text_embeddings(args, model, data_batch)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output_batch, loss = attnviz.forward_with_optional_vae_offload(args, model, data_batch)

    block_set = set(blocks)
    block_ids = [idx for idx in range(len(model.net.blocks)) if idx in block_set]
    attn_maps = [item[0].detach().float().cpu() for item in getattr(model.net, "tavid_target_attn_maps", [])]
    latent_mask = getattr(model.net, "tavid_target_mask_B_T_H_W", None)
    latent_mask_cpu = latent_mask[0].detach().float().cpu() if latent_mask is not None else None
    matching = getattr(model.net, "target_matching_logits_B_T_H_W", None)
    matching_cpu = matching[0].detach().float().cpu() if matching is not None else None
    pred_cpu = output_batch["model_pred"][0].detach().float().cpu()
    context_tokens = getattr(model.net, "target_feature_context_tokens_B_L_D", None)
    valid_tokens = getattr(model.net, "target_feature_context_valid_B_L", None)

    metrics = {}
    if latent_mask_cpu is not None and attn_maps:
        block_metrics = [
            attnviz.compute_metrics(attn_map, latent_mask_cpu, block)
            for block, attn_map in zip(block_ids, attn_maps)
        ]
        metrics["attention_blocks"] = [asdict(item) for item in block_metrics]
        selected_attn = selected_mean_attention({"attn_maps": attn_maps, "block_ids": block_ids}, set(parse_int_list(args.selected_blocks)))
        if selected_attn is not None:
            metrics.update(inside_outside_stats(selected_attn, latent_mask_cpu, "selected_attention"))
    if latent_mask_cpu is not None and matching_cpu is not None:
        metrics.update(inside_outside_stats(torch.sigmoid(matching_cpu), latent_mask_cpu, "matching_prob"))
    metrics["loss"] = float(loss.detach().float().item())
    metrics["model_pred_abs_mean"] = float(pred_cpu.abs().mean().item())
    if context_tokens is not None:
        context_tokens = context_tokens.detach().float().cpu()
        metrics["target_context_token_abs_mean"] = float(context_tokens.abs().mean().item())
        metrics["target_context_token_norm_mean"] = float(context_tokens.norm(dim=-1).mean().item())
    if valid_tokens is not None:
        metrics["target_context_valid_fraction"] = float(valid_tokens.detach().float().mean().item())

    return {
        "variant": variant,
        "feature_override": feature_override,
        "loss": metrics["loss"],
        "metrics": metrics,
        "attn_maps": attn_maps,
        "block_ids": block_ids[: len(attn_maps)],
        "latent_mask": latent_mask_cpu,
        "matching": matching_cpu,
        "model_pred": pred_cpu,
    }


def save_diagnostic_grid(path: Path, raw: torch.Tensor, mask: torch.Tensor, records: list[dict], selected_blocks: set[int]) -> None:
    T_raw = raw.shape[1]
    H_raw, W_raw = raw.shape[-2:]
    frames = attnviz.frame_indices(T_raw)
    frame_labels = [f"f{idx}" for idx in frames]
    raw_frames = [attnviz.to_uint8_frame(raw[:, idx]) for idx in frames]
    mask_T_H_W = mask[0]
    mask_up = attnviz.upsample_volume(mask_T_H_W, (T_raw, H_raw, W_raw), mode="nearest")

    rows: list[tuple[str, list[np.ndarray]]] = [
        ("RGB", raw_frames),
        ("GT mask for metrics", [attnviz.overlay_mask(raw_frames[i], mask_up[idx]) for i, idx in enumerate(frames)]),
    ]
    for record in records:
        selected_attn = selected_mean_attention(record, selected_blocks)
        if selected_attn is not None:
            attn_up = attnviz.normalize_heatmap(attnviz.upsample_volume(selected_attn, (T_raw, H_raw, W_raw)))
            rows.append((
                f"{record['variant']} attn",
                [attnviz.overlay_heat(raw_frames[i], attn_up[idx]) for i, idx in enumerate(frames)],
            ))
        if record["matching"] is not None:
            match_up = attnviz.normalize_heatmap(
                attnviz.upsample_volume(torch.sigmoid(record["matching"]), (T_raw, H_raw, W_raw))
            )
            rows.append((
                f"{record['variant']} match",
                [attnviz.overlay_heat(raw_frames[i], match_up[idx]) for i, idx in enumerate(frames)],
            ))

    keep = next((record for record in records if record["variant"] == "keep"), None)
    if keep is not None:
        for record in records:
            if record is keep:
                continue
            delta = (record["model_pred"] - keep["model_pred"]).abs().mean(dim=0)
            delta_up = attnviz.normalize_heatmap(attnviz.upsample_volume(delta, (T_raw, H_raw, W_raw)))
            rows.append((
                f"{record['variant']} velocity delta",
                [attnviz.overlay_heat(raw_frames[i], delta_up[idx]) for i, idx in enumerate(frames)],
            ))
    attnviz.save_grid(path, rows, frame_labels)


def main() -> None:
    args = parse_args()
    # This script compares velocity predictions across feature interventions,
    # so it needs the explicit `model_pred` tensor returned by the offload path.
    args.offload_denoiser_during_vae = True
    if "wrong" in args.variants and args.wrong_target_feature_path is None and args.wrong_target_dense_feature_path is None:
        raise ValueError("--variants includes wrong, but no wrong feature path was provided")

    if not args.skip_init_environment:
        attnviz.init_environment()

    blocks = parse_int_list(args.blocks)
    selected_blocks = set(parse_int_list(args.selected_blocks) or blocks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, dataloader, loaded_iter = attnviz.load_model_and_dataloader(args, args.checkpoint, blocks)

    sample_records = []
    saved = 0
    for batch_idx, cpu_batch in enumerate(dataloader):
        if batch_idx >= args.max_batches or saved >= args.num_samples:
            break
        if "target_mask" not in cpu_batch or float(cpu_batch["target_mask"].sum()) <= 0:
            continue
        raw_cpu = cpu_batch[model.input_data_key][0].detach().cpu()
        mask_cpu = cpu_batch["target_mask"][0].detach().cpu()
        caption = cpu_batch.get(model.input_caption_key, [""])[0]
        records = [collect_one_variant(args, model, cpu_batch, variant, blocks) for variant in args.variants]

        keep = next((record for record in records if record["variant"] == "keep"), None)
        if keep is not None and keep["latent_mask"] is not None:
            for record in records:
                if record is keep:
                    continue
                delta = (record["model_pred"] - keep["model_pred"]).abs().mean(dim=0)
                record["metrics"].update(inside_outside_stats(delta, keep["latent_mask"], "velocity_delta_vs_keep"))

        figure_path = output_dir / f"sample_{saved:03d}_vlm_guidance_intermediate.jpg"
        save_diagnostic_grid(figure_path, raw_cpu, mask_cpu, records, selected_blocks)

        sample_summary = {
            "sample_index": saved,
            "batch_index": batch_idx,
            "caption": caption,
            "figure": str(figure_path),
            "variants": [
                {
                    "variant": record["variant"],
                    "feature_override": record["feature_override"],
                    "metrics": record["metrics"],
                }
                for record in records
            ],
        }
        print(json.dumps(sample_summary, ensure_ascii=False), flush=True)
        sample_records.append(sample_summary)
        saved += 1

    summary = {
        "checkpoint": args.checkpoint,
        "loaded_iter": loaded_iter,
        "blocks": blocks,
        "selected_blocks": sorted(selected_blocks),
        "variants": args.variants,
        "note": "GT target_mask is used only for diagnostics/metrics in this script.",
        "samples": sample_records,
    }
    summary_path = output_dir / "vlm_feature_guidance_intermediate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path)}, ensure_ascii=False), flush=True)

    del model, dataloader
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
