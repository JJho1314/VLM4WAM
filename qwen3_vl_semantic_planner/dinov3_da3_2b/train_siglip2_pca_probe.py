#!/usr/bin/env python3
"""Train the fixed-PCA SigLIP2 spatial-feature upsampling probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from qwen3_vl_semantic_planner.dinov3_da3_2b.siglip2_pca_probe import (
    SiglipPCAUpsampler,
    fit_fixed_pca,
    multiscale_gradient_loss,
    pca_target_images,
    validation_gate_passed,
    validation_metrics,
)
from qwen3_vl_semantic_planner.dinov3_da3_2b.siglip2_target import (
    Siglip2TargetEncoder,
)


TARGET_SUITE = "libero_10_no_noops_lerobot"
TARGET_EPISODE = 288
EPISODE_PATTERN = re.compile(r"episode_(\d{6})\.npy$")
VALIDATION_MODULUS = 10
EXPECTED_MODEL_NAME = "siglip2-large-patch16-256"
EXPECTED_FEATURE_DIM = 1024


def discover_episode_files(cache_root: Path) -> list[Path]:
    """Discover all cached camera arrays in deterministic path order."""
    files = sorted(cache_root.glob("**/observation.images.*/episode_*.npy"))
    if not files:
        raise FileNotFoundError(
            f"no LIBERO episode frame caches under {cache_root}"
        )
    return files


def _episode_identity(path: Path, cache_root: Path) -> tuple[str, int]:
    relative = path.relative_to(cache_root)
    match = EPISODE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"invalid episode cache name: {path}")
    return relative.parts[0], int(match.group(1))


def split_episode_files(
    files: Sequence[Path],
    *,
    cache_root: Path,
    validation_modulus: int = VALIDATION_MODULUS,
) -> tuple[list[Path], list[Path]]:
    """Split by suite/episode identity so all cameras stay together."""
    if validation_modulus <= 1:
        raise ValueError("validation_modulus must exceed one")
    train: list[Path] = []
    validation: list[Path] = []
    for path in sorted(files):
        suite, episode = _episode_identity(path, cache_root)
        if suite == TARGET_SUITE and episode == TARGET_EPISODE:
            continue
        key = f"{suite}/episode_{episode:06d}".encode()
        bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        destination = (
            validation
            if bucket % validation_modulus == 0
            else train
        )
        destination.append(path)
    if not train or not validation:
        raise RuntimeError("episode split produced an empty partition")
    return train, validation


class CachedFrameDataset(torch.utils.data.Dataset):
    """Deterministically sample frames from memory-mapped episode arrays."""

    def __init__(
        self,
        files: Sequence[Path],
        *,
        virtual_length: int,
        seed: int,
    ) -> None:
        if not files:
            raise ValueError("CachedFrameDataset requires episode files")
        self.files = tuple(files)
        self.virtual_length = virtual_length
        self.seed = seed
        self._memmaps: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return self.virtual_length

    def __getitem__(self, index: int) -> torch.Tensor:
        generator = random.Random(self.seed + index * 2_654_435_761)
        path = self.files[generator.randrange(len(self.files))]
        array = self._memmaps.get(path)
        if array is None:
            array = np.load(path, mmap_mode="r")
            self._memmaps[path] = array
        frame = np.ascontiguousarray(
            array[generator.randrange(array.shape[0])]
        )
        if frame.ndim != 3:
            raise ValueError(f"expected an image frame, got {frame.shape}")
        channels_first = frame.shape[0] == 3
        channels_last = frame.shape[-1] == 3
        if channels_first == channels_last:
            raise ValueError(f"cannot infer cache layout from {frame.shape}")
        if channels_last:
            frame = np.moveaxis(frame, -1, 0)
        frame = np.array(frame, copy=True, order="C")
        return torch.from_numpy(frame).float().div_(255.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-cache-dir", type=Path, required=True)
    parser.add_argument("--siglip2-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--pca-batches", type=int, default=25)
    parser.add_argument("--pca-max-tokens", type=int, default=50_000)
    parser.add_argument("--validation-batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser


def make_teachers(
    model_dir: Path,
    device: torch.device,
) -> tuple[Siglip2TargetEncoder, Siglip2TargetEncoder]:
    """Create frozen native-grid and interpolated high-resolution teachers."""
    if model_dir.name != EXPECTED_MODEL_NAME:
        raise ValueError(
            "SigLIP2 teacher model identity must be "
            f"{EXPECTED_MODEL_NAME}, got {model_dir.name}"
        )
    low = Siglip2TargetEncoder(
        model_dir=model_dir,
        input_size=256,
        grid_size=16,
        device=device,
    )
    high = Siglip2TargetEncoder(
        model_dir=model_dir,
        input_size=512,
        grid_size=0,
        device=device,
    )
    teacher_contract = (
        low.feature_dim,
        high.feature_dim,
        low.input_size,
        high.input_size,
        low.grid_size,
        high.grid_size,
        low.native_size,
        high.native_size,
        getattr(low.model.config, "patch_size", None),
        getattr(high.model.config, "patch_size", None),
    )
    expected_contract = (
        EXPECTED_FEATURE_DIM,
        EXPECTED_FEATURE_DIM,
        256,
        512,
        16,
        0,
        256,
        256,
        16,
        16,
    )
    if teacher_contract != expected_contract:
        raise ValueError(
            "SigLIP2 teacher feature contract is incompatible: "
            f"expected {expected_contract}, got {teacher_contract}"
        )
    return low, high


def _make_loader(
    files: Sequence[Path],
    *,
    batches: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> torch.utils.data.DataLoader:
    dataset = CachedFrameDataset(
        files,
        virtual_length=batches * batch_size,
        seed=seed,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def _fit_training_pca(
    loader: torch.utils.data.DataLoader,
    *,
    high_teacher: Siglip2TargetEncoder,
    device: torch.device,
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    token_batches: list[torch.Tensor] = []
    token_count = 0
    with torch.inference_mode():
        for frames in loader:
            frames = frames.to(device, non_blocking=True)
            tokens = high_teacher._patch_tokens(
                high_teacher._prep(frames)
            ).float()
            flat = tokens.reshape(-1, tokens.shape[-1])
            remaining = max_tokens - token_count
            if remaining <= 0:
                break
            selected = flat[:remaining]
            token_batches.append(selected)
            token_count += selected.shape[0]
            if token_count == max_tokens:
                break
    if not token_batches:
        raise RuntimeError("PCA fitting produced no teacher tokens")
    return fit_fixed_pca(
        torch.cat(token_batches, dim=0),
        max_tokens=max_tokens,
        seed=seed,
    )


def _train_probe(
    probe: SiglipPCAUpsampler,
    loader: torch.utils.data.DataLoader,
    *,
    low_teacher: Siglip2TargetEncoder,
    high_teacher: Siglip2TargetEncoder,
    pca_state: Mapping[str, Any],
    device: torch.device,
    steps: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=steps,
    )
    probe.train()
    step_count = 0
    for frames in loader:
        frames = frames.to(device, non_blocking=True)
        with torch.inference_mode():
            low_tokens = low_teacher._patch_tokens(
                low_teacher._prep(frames)
            ).float()
            high_tokens = high_teacher._patch_tokens(
                high_teacher._prep(frames)
            ).float()
            target = pca_target_images(
                high_tokens,
                pca_state,
                grid_size=32,
                output_size=256,
            )

        prediction = probe(low_tokens)
        pixel_loss = F.l1_loss(prediction, target)
        edge_loss = multiscale_gradient_loss(prediction, target)
        loss = pixel_loss + 0.25 * edge_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        step_count += 1
    if step_count != steps:
        raise RuntimeError(
            f"training loader yielded {step_count} batches, expected {steps}"
        )


def _validate_probe(
    probe: SiglipPCAUpsampler,
    loader: torch.utils.data.DataLoader,
    *,
    low_teacher: Siglip2TargetEncoder,
    high_teacher: Siglip2TargetEncoder,
    pca_state: Mapping[str, Any],
    device: torch.device,
    validation_batches: int,
) -> dict[str, float]:
    totals = {
        "probe_l1": 0.0,
        "baseline_l1": 0.0,
        "probe_gradient": 0.0,
        "baseline_gradient": 0.0,
    }
    probe.eval()
    batch_count = 0
    with torch.inference_mode():
        for frames in loader:
            frames = frames.to(device, non_blocking=True)
            low_tokens = low_teacher._patch_tokens(
                low_teacher._prep(frames)
            ).float()
            high_tokens = high_teacher._patch_tokens(
                high_teacher._prep(frames)
            ).float()
            target = pca_target_images(
                high_tokens,
                pca_state,
                grid_size=32,
                output_size=256,
            )
            baseline = pca_target_images(
                low_tokens,
                pca_state,
                grid_size=16,
                output_size=256,
            )
            batch_metrics = validation_metrics(
                prediction=probe(low_tokens),
                baseline=baseline,
                target=target,
            )
            for name, value in batch_metrics.items():
                totals[name] += value
            batch_count += 1
    if batch_count != validation_batches:
        raise RuntimeError(
            "validation loader yielded "
            f"{batch_count} batches, expected {validation_batches}"
        )
    return {
        name: value / batch_count
        for name, value in totals.items()
    }


def _cpu_pca_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value.detach().cpu()
        if isinstance(value, torch.Tensor)
        else value
        for name, value in state.items()
    }


def run(args: argparse.Namespace) -> Path:
    """Run all three deterministic stages and save the gated checkpoint."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    files = discover_episode_files(args.frame_cache_dir)
    train_files, validation_files = split_episode_files(
        files,
        cache_root=args.frame_cache_dir,
        validation_modulus=VALIDATION_MODULUS,
    )
    pin_memory = device.type == "cuda"
    pca_loader = _make_loader(
        train_files,
        batches=args.pca_batches,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
    )
    training_loader = _make_loader(
        train_files,
        batches=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed + 1,
        pin_memory=pin_memory,
    )
    validation_loader = _make_loader(
        validation_files,
        batches=args.validation_batches,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed + 2,
        pin_memory=pin_memory,
    )

    low_teacher, high_teacher = make_teachers(
        args.siglip2_model_dir,
        device,
    )
    pca_state = _fit_training_pca(
        pca_loader,
        high_teacher=high_teacher,
        device=device,
        max_tokens=args.pca_max_tokens,
        seed=args.seed,
    )
    probe = SiglipPCAUpsampler(in_dim=low_teacher.feature_dim).to(device)
    _train_probe(
        probe,
        training_loader,
        low_teacher=low_teacher,
        high_teacher=high_teacher,
        pca_state=pca_state,
        device=device,
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    aggregate_metrics = _validate_probe(
        probe,
        validation_loader,
        low_teacher=low_teacher,
        high_teacher=high_teacher,
        pca_state=pca_state,
        device=device,
        validation_batches=args.validation_batches,
    )
    accepted = validation_gate_passed(aggregate_metrics)
    print(
        json.dumps(
            {
                "validation_metrics": aggregate_metrics,
                "accepted": accepted,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    payload = {
        "accepted": accepted,
        "model_name": args.siglip2_model_dir.name,
        "feature_layer": "penultimate_spatial",
        "low_input_size": 256,
        "high_input_size": 512,
        "high_grid_size": 32,
        "state_dict": probe.state_dict(),
        "config": probe.config(),
        "pca_state": _cpu_pca_state(pca_state),
        "validation_metrics": aggregate_metrics,
        "split": {
            "target_exclusion": (
                "libero_10_no_noops_lerobot/episode_000288"
            ),
            "train_files": len(train_files),
            "validation_files": len(validation_files),
            "train_relative_paths": [
                str(path.relative_to(args.frame_cache_dir))
                for path in train_files
            ],
            "validation_relative_paths": [
                str(path.relative_to(args.frame_cache_dir))
                for path in validation_files
            ],
            "validation_modulus": VALIDATION_MODULUS,
        },
        "training": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        "siglip2_pca_upsample_probe.pt"
        if accepted
        else "siglip2_pca_upsample_probe_rejected.pt"
    )
    checkpoint_path = args.output_dir / filename
    torch.save(payload, checkpoint_path)
    if not accepted:
        raise SystemExit(2)
    return checkpoint_path


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
