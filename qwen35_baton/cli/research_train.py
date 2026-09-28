"""Research Stage-1 trainer for Baton planner variants (E4).

Warm-starts from a strict v3 Baton checkpoint and trains one variant:
``--residual`` predicts ``SigLIP2(current) + delta`` with a zero-initialized
delta head, and ``--spatial-weight > 0`` adds the DA3 WSA auxiliary head.
Checkpoints are plain safetensors plus ``variant.json``; the strict v3
production contract is not used until a variant is promoted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.nn as nn


_NEW_MODULE_PREFIXES = ("current_projection.", "spa_query_tower.", "spa_mlp.")
_HEAD_PREFIXES = ("query_tower.", "sem_mlp.", *_NEW_MODULE_PREFIXES)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--qwen-path", required=True)
    parser.add_argument("--siglip2-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stat-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--spatial-weight", type=float, default=0.0)
    parser.add_argument("--da3-ckpt")
    parser.add_argument("--da3-code-root")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--per-device-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-heads", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser


def _build_planner(args: argparse.Namespace) -> tuple[object, nn.Module]:
    from safetensors.torch import load_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    from qwen35_baton.model import BatonQwen35Planner
    from qwen35_baton.sequence import ADDED_TOKENS

    metadata = json.loads(
        (Path(args.init_checkpoint) / "metadata.json").read_text(encoding="utf-8")
    )
    added_token_ids = tuple(int(value) for value in metadata["added_token_ids"])
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.qwen_path, local_files_only=True)
    processor.tokenizer = tokenizer
    actual = tuple(int(tokenizer.convert_tokens_to_ids(token)) for token in ADDED_TOKENS)
    if actual != added_token_ids:
        raise ValueError("tokenizer added-token IDs differ from the init checkpoint")
    qwen = AutoModelForImageTextToText.from_pretrained(
        args.qwen_path,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    planner = BatonQwen35Planner(
        qwen,
        added_token_ids=added_token_ids,
        residual=args.residual,
        spatial=args.spatial_weight > 0,
    )
    missing, unexpected = load_model(
        planner,
        str(Path(args.init_checkpoint) / "planner.safetensors"),
        strict=False,
    )
    if unexpected:
        raise ValueError(f"init checkpoint has unexpected tensors: {unexpected[:5]}")
    unexplained = [
        name
        for name in missing
        if not name.startswith(_NEW_MODULE_PREFIXES)
        and name != "frozen_base_embedding.weight"
    ]
    if unexplained:
        raise ValueError(f"init checkpoint is missing trained tensors: {unexplained[:5]}")
    if args.residual:
        # The loaded absolute head would add a full feature map on top of the
        # current grid; restart the delta at zero (= copy-current baseline).
        planner.zero_init_residual_head()
    planner.requires_grad_(True)
    return processor, planner


def _optimizer(planner: nn.Module, args: argparse.Namespace) -> torch.optim.AdamW:
    heads, backbone = [], []
    for name, parameter in planner.named_parameters():
        (heads if name.startswith(_HEAD_PREFIXES) else backbone).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": args.lr_backbone},
            {"params": heads, "lr": args.lr_heads},
        ],
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )


def _lr_lambda(step: int, *, warmup: int, total: int) -> float:
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _per_frame_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """[B,2,4,256,D] -> mean squared error per (camera, keyframe), [2,4]."""

    return (prediction.float() - target.float()).square().mean(dim=(0, 3, 4))


def main() -> int:
    args = _parser().parse_args()
    if args.spatial_weight > 0 and not (args.da3_ckpt and args.da3_code_root):
        raise ValueError("--spatial-weight > 0 requires --da3-ckpt and --da3-code-root")
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed
    from safetensors.torch import save_model

    from ge_act.data.libero_fastwam_hdf5_dataset import LiberoFastWAMHDF5Dataset
    from qwen35_baton.cli.train_semantic_planner import _move_batch
    from qwen35_baton.data import BatonLiberoDataset, BatonPlannerCollator
    from qwen35_baton.losses import compute_spatial_wsa_loss
    from qwen35_baton.teacher import FrozenDA3Teacher, FrozenSiglip2Teacher

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.grad_accum,
        device_placement=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    set_seed(args.seed, device_specific=True)
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "variant.json").write_text(
            json.dumps(vars(args), indent=2), encoding="utf-8"
        )

    processor, planner = _build_planner(args)
    if args.gradient_checkpointing:
        planner.backbone.gradient_checkpointing_enable()
    optimizer = _optimizer(planner, args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_lambda(step, warmup=args.warmup_steps, total=args.max_steps),
    )
    dataset = BatonLiberoDataset(
        LiberoFastWAMHDF5Dataset(args.manifest, args.stat_file, train_dataset=True),
        seed=args.seed,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.per_device_batch,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=BatonPlannerCollator(processor),
        persistent_workers=args.num_workers > 0,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
    )
    planner.to(accelerator.device)
    planner, optimizer, loader = accelerator.prepare(planner, optimizer, loader)
    siglip = FrozenSiglip2Teacher(args.siglip2_path, device=accelerator.device)
    da3 = None
    if args.spatial_weight > 0:
        da3 = FrozenDA3Teacher(
            args.da3_ckpt, args.da3_code_root, device=accelerator.device
        )

    metrics_path = output_dir / "metrics.jsonl"
    window: dict[str, torch.Tensor] = {}
    window_count = 0
    step = 0
    epoch = 0
    started = time.time()
    planner.train()
    while step < args.max_steps:
        set_epoch = getattr(dataset, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
        for batch in loader:
            batch = _move_batch(batch, accelerator.device)
            with accelerator.accumulate(planner):
                with torch.no_grad():
                    current = siglip.encode_current(batch.current_images)
                    future = siglip.encode_future(batch.future_images)
                output = planner(
                    batch, current_features=current if args.residual else None
                )
                semantic_mse = (
                    output.positive.float() - future.float()
                ).square().mean()
                total = semantic_mse
                record = {
                    "semantic_mse": _per_frame_mse(output.positive, future).detach(),
                    "copy_mse": _per_frame_mse(
                        current.unsqueeze(2).expand_as(future), future
                    ).detach(),
                }
                if da3 is not None:
                    spatial_target = da3.encode_future(batch.future_images)
                    spatial = compute_spatial_wsa_loss(
                        output.spatial, spatial_target, da3.layer_weights
                    )
                    total = total + args.spatial_weight * spatial.total
                    record["spatial_wsa"] = spatial.total.detach().reshape(1)
                    record["spatial_cos"] = spatial.cosine.detach().reshape(1)
                    record["spatial_lnmse"] = spatial.layernorm_mse.detach().reshape(1)
                if not torch.isfinite(total):
                    raise FloatingPointError(f"nonfinite loss at step {step}")
                accelerator.backward(total)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(planner.parameters(), 1.0)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            for name, value in record.items():
                window[name] = window.get(name, 0) + value
            window_count += 1
            if not accelerator.sync_gradients:
                continue
            step += 1
            if step % args.log_every == 0 or step == args.max_steps:
                reduced = {
                    name: accelerator.reduce(value / window_count, reduction="mean")
                    for name, value in window.items()
                }
                if accelerator.is_main_process:
                    entry = {
                        "step": step,
                        "elapsed_s": round(time.time() - started, 1),
                        "lr_backbone": scheduler.get_last_lr()[0],
                        "semantic_mse": float(reduced["semantic_mse"].mean()),
                        "copy_mse": float(reduced["copy_mse"].mean()),
                    }
                    for camera_index, camera in enumerate(("main", "wrist")):
                        for frame in range(4):
                            entry[f"mse/{camera}/k{frame}"] = float(
                                reduced["semantic_mse"][camera_index, frame]
                            )
                            entry[f"copy/{camera}/k{frame}"] = float(
                                reduced["copy_mse"][camera_index, frame]
                            )
                    for name in ("spatial_wsa", "spatial_cos", "spatial_lnmse"):
                        if name in reduced:
                            entry[name] = float(reduced[name])
                    with metrics_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(entry) + "\n")
                    print(json.dumps(entry), flush=True)
                window, window_count = {}, 0
            if step % args.save_every == 0 or step == args.max_steps:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    destination = output_dir / f"step_{step:06d}"
                    destination.mkdir(parents=True, exist_ok=True)
                    save_model(
                        accelerator.unwrap_model(planner),
                        str(destination / "planner.safetensors"),
                    )
                accelerator.wait_for_everyone()
            if step >= args.max_steps:
                break
        epoch += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
