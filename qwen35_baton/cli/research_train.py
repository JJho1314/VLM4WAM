"""Research Stage-1 trainer for Baton planner variants (E4).

Warm-starts from a strict v3 Baton checkpoint and trains one variant:
``--residual`` predicts ``SigLIP2(current) + delta`` with a zero-initialized
delta head, and ``--spatial-weight > 0`` adds the DA3 WSA auxiliary head.
Checkpoints are plain safetensors plus ``variant.json``; the strict v3
production contract is not used until a variant is promoted.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.nn as nn


_NEW_MODULE_PREFIXES = ("current_projection.", "spa_query_tower.", "spa_mlp.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--qwen-path", required=True)
    parser.add_argument("--siglip2-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stat-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--current-mode",
        choices=("none", "context", "residual"),
        default="none",
        help="how the current-frame SigLIP2 grid enters the planner",
    )
    parser.add_argument("--spatial-weight", type=float, default=0.0)
    parser.add_argument("--da3-ckpt")
    parser.add_argument("--da3-code-root")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--per-device-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    # The warm-start source stopped at 22.6k/30k of a 1e-5 cosine (~1.5e-6), so
    # already-trained weights continue near that rate; only new modules start hot.
    parser.add_argument("--lr-pretrained", type=float, default=2e-6)
    parser.add_argument("--lr-new", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--worker-malloc-trim",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="return freed heap to the OS in DataLoader workers after every batch",
    )
    parser.add_argument("--max-open-shards", type=int, default=8)
    parser.add_argument(
        "--gpu-teacher-preprocess",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="rescale/normalize SigLIP2 inputs on GPU instead of the PIL processor",
    )
    # On by default: under plain DDP the fp32 weights, grads and AdamW states
    # alone take ~37 GB/GPU, and full activations push 2 samples/GPU past 80 GB.
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _build_planner(args: argparse.Namespace) -> tuple[object, nn.Module]:
    from safetensors.torch import load_model

    from qwen35_baton.research_provider import (
        build_research_planner,
        init_checkpoint_token_ids,
    )

    processor, planner = build_research_planner(
        qwen_path=args.qwen_path,
        added_token_ids=init_checkpoint_token_ids(args.init_checkpoint),
        current_mode=args.current_mode,
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
    if args.current_mode == "residual":
        # The loaded absolute head would add a full feature map on top of the
        # current grid; restart the delta at zero (= copy-current baseline).
        planner.zero_init_residual_head()
    planner.requires_grad_(True)
    return processor, planner


def _optimizer(planner: nn.Module, args: argparse.Namespace) -> torch.optim.AdamW:
    new, pretrained = [], []
    for name, parameter in planner.named_parameters():
        (new if name.startswith(_NEW_MODULE_PREFIXES) else pretrained).append(parameter)
    groups = [{"params": pretrained, "lr": args.lr_pretrained}]
    if new:
        groups.append({"params": new, "lr": args.lr_new})
    return torch.optim.AdamW(
        groups,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )


def _warm_start_adam_state(
    optimizer: torch.optim.AdamW,
    planner: nn.Module,
    init_checkpoint: str | Path,
) -> tuple[int, int]:
    """Copy the source run's AdamW moments onto same-named, same-shaped params.

    A fresh AdamW takes ~lr-sized sign steps on every weight, which measurably
    degraded the warm-started planner; restoring moments makes the run a true
    continuation. New modules keep an empty state. Returns (restored, total).
    """

    saved = torch.load(
        Path(init_checkpoint) / "optimizer.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    by_name: dict[str, dict[str, torch.Tensor]] = {}
    for group in saved["param_groups"]:
        for name, index in zip(group["parameter_names"], group["params"]):
            if index in saved["state"]:
                by_name[name] = saved["state"][index]
    name_of = {id(parameter): name for name, parameter in planner.named_parameters()}
    restored = total = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            total += 1
            state = by_name.get(name_of.get(id(parameter), ""))
            if state is None or tuple(state["exp_avg"].shape) != tuple(parameter.shape):
                continue
            optimizer.state[parameter] = {
                "step": torch.as_tensor(state["step"], dtype=torch.float32).clone(),
                "exp_avg": state["exp_avg"].to(parameter.device, parameter.dtype),
                "exp_avg_sq": state["exp_avg_sq"].to(parameter.device, parameter.dtype),
            }
            restored += 1
    del saved, by_name
    gc.collect()
    return restored, total


def _lr_lambda(step: int, *, warmup: int, total: int) -> float:
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class _TrimmingCollate:
    """Collate, then malloc_trim: worker RSS grew ~9 MiB per sample otherwise."""

    def __init__(self, collate_fn) -> None:
        self.collate_fn = collate_fn

    def __call__(self, samples):
        import ctypes

        batch = self.collate_fn(samples)
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        return batch


def _host_rss_gib() -> tuple[float, float]:
    """Resident memory (GiB) of this rank and of its DataLoader workers."""

    import psutil

    process = psutil.Process()
    children = 0
    for child in process.children(recursive=True):
        try:
            children += child.memory_info().rss
        except psutil.Error:
            pass
    return process.memory_info().rss / 2**30, children / 2**30


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
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=True,
                # Alias grads to the DDP buckets instead of keeping a second copy.
                gradient_as_bucket_view=True,
            )
        ],
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
        LiberoFastWAMHDF5Dataset(
            args.manifest,
            args.stat_file,
            train_dataset=True,
            max_open_shards=args.max_open_shards,
        ),
        seed=args.seed,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.per_device_batch,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=(
            _TrimmingCollate(BatonPlannerCollator(processor))
            if args.worker_malloc_trim
            else BatonPlannerCollator(processor)
        ),
        persistent_workers=args.num_workers > 0,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
    )
    planner.to(accelerator.device)
    restored, total = _warm_start_adam_state(optimizer, planner, args.init_checkpoint)
    if accelerator.is_main_process:
        print(json.dumps({"adam_state_restored": restored, "params": total}), flush=True)
    planner, optimizer, loader = accelerator.prepare(planner, optimizer, loader)
    siglip = FrozenSiglip2Teacher(args.siglip2_path, device=accelerator.device)
    siglip.gpu_preprocess = args.gpu_teacher_preprocess
    da3 = None
    if args.spatial_weight > 0:
        da3 = FrozenDA3Teacher(
            args.da3_ckpt, args.da3_code_root, device=accelerator.device
        )

    # Diagnostic: RESEARCH_MEMPROFILE=1 diffs Python allocations on rank 0.
    profile_memory = os.environ.get("RESEARCH_MEMPROFILE") == "1"
    snapshots = []
    if profile_memory and accelerator.is_main_process:
        import tracemalloc

        tracemalloc.start(8)
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
                    batch,
                    current_features=None if args.current_mode == "none" else current,
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
            if profile_memory and accelerator.is_main_process and step in (20, 120):
                snapshots.append(tracemalloc.take_snapshot())
                if len(snapshots) == 2:
                    top = snapshots[1].compare_to(snapshots[0], "lineno")[:15]
                    print("TRACEMALLOC_TOP", flush=True)
                    for stat in top:
                        print(f"  {stat}", flush=True)
            # Batches leave reference cycles that only full collections free;
            # without this rank RSS grew ~0.15 GiB/step until the host OOMed.
            gc.collect()
            if step % args.log_every == 0 or step == args.max_steps:
                reduced = {
                    name: accelerator.reduce(value / window_count, reduction="mean")
                    for name, value in window.items()
                }
                if accelerator.is_main_process:
                    entry = {
                        "step": step,
                        "elapsed_s": round(time.time() - started, 1),
                        "lr_pretrained": scheduler.get_last_lr()[0],
                        "semantic_mse": float(reduced["semantic_mse"].mean()),
                        "copy_mse": float(reduced["copy_mse"].mean()),
                        "peak_mem_gib": round(
                            torch.cuda.max_memory_allocated(accelerator.device) / 2**30, 2
                        ),
                        "rank0_host_rss_gib": round(sum(_host_rss_gib()), 2),
                        "rank0_main_rss_gib": round(_host_rss_gib()[0], 2),
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
