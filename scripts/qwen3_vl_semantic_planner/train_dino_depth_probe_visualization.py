#!/usr/bin/env python3
"""Fit lightweight DINO/Depth probes and render separate 224px outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


TOKEN_GRID_SIZE = 16
TOKEN_COUNT = TOKEN_GRID_SIZE * TOKEN_GRID_SIZE
OUTPUT_SIZE = 224
DINO_OUTPUT_NAMES = (
    "dino_teacher_current_224",
    "dino_planner_current_224",
    "dino_teacher_future_224",
    "dino_planner_future_224",
)
DEPTH_OUTPUT_NAMES = (
    "depth_target_current_224",
    "depth_planner_current_224",
    "depth_target_future_224",
    "depth_planner_future_224",
)
EXPECTED_SAMPLE_FILES = (
    "observation_current.png",
    "observation_future.png",
    "instruction.txt",
    *(f"{name}.png" for name in DINO_OUTPUT_NAMES),
    *(f"{name}.png" for name in DEPTH_OUTPUT_NAMES),
)


class ProbeTrainingCache:
    def __init__(
        self,
        *,
        dino: torch.Tensor,
        depth: torch.Tensor,
        relative_depth: torch.Tensor,
    ) -> None:
        self.dino = dino
        self.depth = depth
        self.relative_depth = relative_depth

    def validate(self) -> None:
        validate_token_features(
            self.dino,
            feature_dim=1024,
            name="cached DINO",
        )
        validate_token_features(
            self.depth,
            feature_dim=1024,
            name="cached Depth",
        )
        if self.dino.shape != self.depth.shape:
            raise ValueError(
                "cached DINO and Depth feature shapes differ: "
                f"{tuple(self.dino.shape)} != {tuple(self.depth.shape)}"
            )
        expected_depth_shape = (self.dino.shape[0], TOKEN_GRID_SIZE, TOKEN_GRID_SIZE)
        if tuple(self.relative_depth.shape) != expected_depth_shape:
            raise ValueError(
                "cached relative depth must have shape "
                f"{expected_depth_shape}, got {tuple(self.relative_depth.shape)}"
            )
        if not bool(torch.isfinite(self.relative_depth).all()):
            raise ValueError("cached relative depth contains non-finite values")


def compute_dino_map_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float | int]:
    expected_tail = (3, OUTPUT_SIZE, OUTPUT_SIZE)
    if prediction.ndim != 4 or tuple(prediction.shape[1:]) != expected_tail:
        raise ValueError(
            f"DINO prediction must be [B,3,{OUTPUT_SIZE},{OUTPUT_SIZE}], "
            f"got {tuple(prediction.shape)}"
        )
    if prediction.shape != target.shape:
        raise ValueError(
            f"DINO prediction and target shapes differ: "
            f"{tuple(prediction.shape)} != {tuple(target.shape)}"
        )
    prediction = prediction.detach().to(device="cpu", dtype=torch.float32)
    target = target.detach().to(device="cpu", dtype=torch.float32)
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("DINO projected maps contain non-finite values")
    cosine = F.cosine_similarity(prediction, target, dim=1, eps=1e-8)
    pixels = prediction.shape[0] * OUTPUT_SIZE * OUTPUT_SIZE
    return {
        "num_pixels": int(pixels),
        "num_values": int(prediction.numel()),
        "mse": float(F.mse_loss(prediction, target)),
        "mean_cosine": float(cosine.mean()),
    }


def validate_token_features(
    features: torch.Tensor,
    *,
    feature_dim: int | None = None,
    name: str = "features",
) -> None:
    if features.ndim != 3 or features.shape[1] != TOKEN_COUNT:
        raise ValueError(
            f"{name} must be [B,{TOKEN_COUNT},D], got {tuple(features.shape)}"
        )
    if feature_dim is not None and features.shape[2] != int(feature_dim):
        raise ValueError(
            f"{name} feature dimension must be {feature_dim}, got {features.shape[2]}"
        )
    if features.shape[2] <= 0:
        raise ValueError(f"{name} feature dimension must be positive")
    if not bool(torch.isfinite(features).all()):
        raise ValueError(f"{name} contains non-finite values")


class DinoPCAProbe(nn.Module):
    def __init__(
        self,
        mean: torch.Tensor,
        basis: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
        *,
        output_size: int = OUTPUT_SIZE,
    ) -> None:
        super().__init__()
        if int(output_size) <= 0:
            raise ValueError("output_size must be positive")
        self.register_buffer("mean", mean.detach().to(torch.float32))
        self.register_buffer("basis", basis.detach().to(torch.float32))
        self.register_buffer("low", low.detach().to(torch.float32))
        self.register_buffer("high", high.detach().to(torch.float32))
        self.output_size = int(output_size)

    @classmethod
    def fit(
        cls,
        features: torch.Tensor,
        *,
        seed: int = 0,
        output_size: int = OUTPUT_SIZE,
    ) -> "DinoPCAProbe":
        validate_token_features(features, name="DINO PCA training features")
        flat = features.detach().to(device="cpu", dtype=torch.float32).flatten(0, 1)
        mean = flat.mean(dim=0)
        centered = flat - mean
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            _u, _s, basis = torch.pca_lowrank(
                centered,
                q=3,
                center=False,
                niter=4,
            )
        basis = basis[:, :3]
        # Remove the arbitrary PCA sign so serialized probes are reproducible.
        pivots = basis.abs().argmax(dim=0)
        signs = basis[pivots, torch.arange(3)].sign()
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        basis = basis * signs
        projected = centered @ basis
        low = torch.quantile(projected, 0.01, dim=0)
        high = torch.quantile(projected, 0.99, dim=0)
        if not all(
            bool(torch.isfinite(value).all())
            for value in (mean, basis, low, high)
        ):
            raise ValueError("DINO PCA probe contains non-finite statistics")
        return cls(
            mean,
            basis,
            low,
            high,
            output_size=output_size,
        )

    def project_224(self, features: torch.Tensor) -> torch.Tensor:
        validate_token_features(
            features,
            feature_dim=self.mean.numel(),
            name="DINO features",
        )
        projected = (features.to(torch.float32) - self.mean) @ self.basis
        projected = (projected - self.low) / (self.high - self.low).clamp_min(1e-6)
        grid = (
            projected.clamp(0.0, 1.0)
            .reshape(-1, TOKEN_GRID_SIZE, TOKEN_GRID_SIZE, 3)
            .permute(0, 3, 1, 2)
        )
        return F.interpolate(
            grid,
            size=(self.output_size, self.output_size),
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)


def _as_bhw(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    elif value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(f"{name} must be [B,H,W] or [B,1,H,W], got {tuple(value.shape)}")
    return value


def resize_depth_target_224(target_depth: torch.Tensor) -> torch.Tensor:
    target = _as_bhw(target_depth, name="target_depth").to(torch.float32)
    target = torch.nan_to_num(
        target,
        nan=1e-6,
        posinf=1e6,
        neginf=1e-6,
    ).clamp_min(1e-6)
    return F.interpolate(
        target.unsqueeze(1),
        size=(OUTPUT_SIZE, OUTPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    )[:, 0]


def decode_depth_224(
    relative_log_prediction: torch.Tensor,
    target_depth: torch.Tensor,
) -> torch.Tensor:
    prediction = _as_bhw(
        relative_log_prediction,
        name="relative_log_prediction",
    ).to(torch.float32)
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("relative_log_prediction contains non-finite values")
    prediction = F.interpolate(
        prediction.unsqueeze(1),
        size=(OUTPUT_SIZE, OUTPUT_SIZE),
        mode="bicubic",
        align_corners=False,
    )[:, 0]
    prediction = prediction - prediction.mean(dim=(-2, -1), keepdim=True)
    target = resize_depth_target_224(target_depth).to(prediction.device)
    target_log = target.log()
    shift = (target_log - prediction).flatten(1).median(dim=1).values.view(-1, 1, 1)
    decoded = (prediction + shift).exp()
    if not bool(torch.isfinite(decoded).all()):
        raise ValueError("decoded depth contains non-finite values")
    return decoded


def _rgb_image_224(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
    else:
        tensor = torch.as_tensor(value).detach().cpu()
        if tensor.ndim != 3:
            raise ValueError(f"RGB observation must be 3-D, got {tuple(tensor.shape)}")
        if tensor.shape[0] == 3 and tensor.shape[-1] != 3:
            tensor = tensor.permute(1, 2, 0)
        if tensor.shape[-1] != 3:
            raise ValueError(f"RGB observation must end in 3 channels, got {tuple(tensor.shape)}")
        if tensor.dtype != torch.uint8:
            maximum = float(tensor.max()) if tensor.numel() else 0.0
            tensor = tensor.float()
            if maximum <= 1.5:
                tensor = tensor * 255.0
            tensor = tensor.round().clamp(0, 255).to(torch.uint8)
        image = Image.fromarray(tensor.numpy())
    if image.height < OUTPUT_SIZE or image.width < OUTPUT_SIZE:
        raise ValueError(
            f"RGB observation must be at least {OUTPUT_SIZE}x{OUTPUT_SIZE}, "
            f"got {image.width}x{image.height}"
        )
    # FastWAM composes the external view on the left and wrist view on the right.
    image = image.crop((0, 0, OUTPUT_SIZE, OUTPUT_SIZE))
    if image.size != (OUTPUT_SIZE, OUTPUT_SIZE):
        raise RuntimeError(f"failed to produce a {OUTPUT_SIZE}x{OUTPUT_SIZE} observation")
    return image


def _dino_image_224(value: torch.Tensor) -> Image.Image:
    tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32)
    if tensor.shape != (3, OUTPUT_SIZE, OUTPUT_SIZE):
        raise ValueError(
            f"DINO map must be [3,{OUTPUT_SIZE},{OUTPUT_SIZE}], got {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("DINO map contains non-finite values")
    array = (
        tensor.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array)


def _depth_image_224(
    value: torch.Tensor,
    *,
    low: float,
    high: float,
) -> Image.Image:
    from matplotlib import colormaps

    tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32)
    if tensor.shape != (OUTPUT_SIZE, OUTPUT_SIZE):
        raise ValueError(
            f"depth map must be [{OUTPUT_SIZE},{OUTPUT_SIZE}], got {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("depth map contains non-finite values")
    normalized = ((tensor - float(low)) / max(float(high) - float(low), 1e-6)).clamp(0, 1)
    rgb = colormaps["viridis"](normalized.numpy(), bytes=True)[..., :3]
    return Image.fromarray(rgb)


def save_sample_outputs(
    *,
    output_dir: Path,
    current_rgb: Any,
    future_rgb: Any,
    instruction: str,
    dino_maps: dict[str, torch.Tensor],
    depth_maps: dict[str, torch.Tensor],
) -> list[Path]:
    output_dir = Path(output_dir)
    if set(dino_maps) != set(DINO_OUTPUT_NAMES):
        raise ValueError(f"DINO output names must be {DINO_OUTPUT_NAMES}")
    if set(depth_maps) != set(DEPTH_OUTPUT_NAMES):
        raise ValueError(f"Depth output names must be {DEPTH_OUTPUT_NAMES}")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for filename, value in (
        ("observation_current.png", current_rgb),
        ("observation_future.png", future_rgb),
    ):
        path = output_dir / filename
        _rgb_image_224(value).save(path)
        paths.append(path)

    instruction_path = output_dir / "instruction.txt"
    instruction_path.write_text(instruction.strip() + "\n", encoding="utf-8")
    paths.append(instruction_path)

    for name in DINO_OUTPUT_NAMES:
        path = output_dir / f"{name}.png"
        _dino_image_224(dino_maps[name]).save(path)
        paths.append(path)

    bounds = {}
    for time_name in ("current", "future"):
        target = torch.as_tensor(
            depth_maps[f"depth_target_{time_name}_224"]
        ).detach().to(device="cpu", dtype=torch.float32)
        quantiles = torch.quantile(target, torch.tensor([0.02, 0.98]))
        bounds[time_name] = (float(quantiles[0]), float(quantiles[1]))
    for name in DEPTH_OUTPUT_NAMES:
        time_name = "current" if "current" in name else "future"
        path = output_dir / f"{name}.png"
        low, high = bounds[time_name]
        _depth_image_224(depth_maps[name], low=low, high=high).save(path)
        paths.append(path)

    if {path.name for path in paths} != set(EXPECTED_SAMPLE_FILES):
        raise RuntimeError("sample output layout is incomplete")
    return paths


def _dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a global DINO PCA probe and a shared linear Depth probe, then "
            "render planner/teacher outputs as separate 224x224 files."
        )
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--fastwam-data-config", type=Path, required=True)
    parser.add_argument("--fastwam-dataset-dir", action="append", required=True)
    parser.add_argument("--fastwam-text-embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--fastwam-pretrained-norm-stats", type=Path, required=True)
    parser.add_argument("--dino-teacher-ckpt", type=Path, required=True)
    parser.add_argument("--dino-teacher-config", type=Path, required=True)
    parser.add_argument("--depth-moge-path", type=Path, required=True)
    parser.add_argument("--depth-morgbd-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-windows-per-suite", type=int, default=64)
    parser.add_argument("--eval-windows-per-suite", type=int, default=16)
    parser.add_argument("--teacher-batch-size", type=int, default=8)
    parser.add_argument("--planner-batch-size", type=int, default=8)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--probe-lr", type=float, default=3e-3)
    parser.add_argument("--gradient-loss-weight", type=float, default=0.2)
    parser.add_argument("--visualizations-per-suite", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def _import_pipeline_helpers():
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from evaluate_lingbot_current_future_planner import (
        _build_suite_dataset,
        _load_runtime,
    )
    from train_depth_probe_visualization import (
        BestProbeStateTracker,
        LinearDepthProbe,
        _build_depth_teacher,
        _empty_depth_sums,
        _finalize_depth_sums,
        _update_depth_sums,
        depth_gradient_loss,
        extract_depth_teacher_outputs,
        relative_log_depth,
        select_disjoint_indices,
    )
    from train_qwen3vl4b_lingbot_dino_planner import (
        build_planner_inputs,
        move_qwen_inputs_to_device,
    )

    return {
        "build_suite_dataset": _build_suite_dataset,
        "load_runtime": _load_runtime,
        "BestProbeStateTracker": BestProbeStateTracker,
        "LinearDepthProbe": LinearDepthProbe,
        "build_depth_teacher": _build_depth_teacher,
        "empty_depth_sums": _empty_depth_sums,
        "finalize_depth_sums": _finalize_depth_sums,
        "update_depth_sums": _update_depth_sums,
        "depth_gradient_loss": depth_gradient_loss,
        "extract_depth_teacher_outputs": extract_depth_teacher_outputs,
        "relative_log_depth": relative_log_depth,
        "select_disjoint_indices": select_disjoint_indices,
        "build_planner_inputs": build_planner_inputs,
        "move_qwen_inputs_to_device": move_qwen_inputs_to_device,
    }


def _preflight(args: argparse.Namespace) -> None:
    positive = (
        "train_windows_per_suite",
        "eval_windows_per_suite",
        "teacher_batch_size",
        "planner_batch_size",
        "probe_batch_size",
        "probe_epochs",
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.probe_lr <= 0:
        raise ValueError("--probe-lr must be positive")
    if args.gradient_loss_weight < 0:
        raise ValueError("--gradient-loss-weight must be non-negative")
    if args.visualizations_per_suite < 0:
        raise ValueError("--visualizations-per-suite must be non-negative")
    paths = (
        args.checkpoint_dir,
        args.fastwam_data_config,
        args.fastwam_text_embedding_cache_dir,
        args.fastwam_pretrained_norm_stats,
        args.dino_teacher_ckpt,
        args.dino_teacher_config,
        args.depth_moge_path,
        args.depth_morgbd_path,
        *(Path(item) for item in args.fastwam_dataset_dir),
    )
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("missing required inputs:\n" + "\n".join(missing))
    for variable in ("LINGBOT_SRC_ROOT", "UTILS3D_MOGE_PATH"):
        value = os.environ.get(variable)
        if not value or not Path(value).exists():
            raise FileNotFoundError(f"{variable} must point to an existing path")
    frame_cache = os.environ.get("FASTWAM_FRAME_CACHE_DIR")
    if not frame_cache or not Path(frame_cache).exists():
        raise FileNotFoundError(
            "FASTWAM_FRAME_CACHE_DIR must point to the predecoded 224px cache"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)


def _build_dino_teacher(args: argparse.Namespace, device: torch.device):
    module_dir = Path(__file__).resolve().parent / "lingbot_dino_4b"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    from dino_video_target import DinoVideoTargetEncoder

    return DinoVideoTargetEncoder(
        ckpt_path=args.dino_teacher_ckpt,
        config_path=args.dino_teacher_config,
        input_size=256,
        device=device,
        lingbot_root=os.environ["LINGBOT_SRC_ROOT"],
    ).eval()


def _frames_from_items(
    items: list[dict[str, Any]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    current = torch.stack([item["current_image"] for item in items]).permute(0, 3, 1, 2)
    future = torch.stack([item["keyframe_images"][0] for item in items]).permute(0, 3, 1, 2)
    return current.to(device, non_blocking=True), future.to(device, non_blocking=True)


def _effective_fps_from_items(
    items: list[dict[str, Any]],
    *,
    device: torch.device,
) -> torch.Tensor:
    if not all("future_video_effective_fps" in item for item in items):
        raise KeyError("every sample must provide future_video_effective_fps")
    fps = torch.stack([item["future_video_effective_fps"] for item in items])
    if fps.ndim == 2 and fps.shape[1] == 1:
        fps = fps[:, 0]
    if tuple(fps.shape) != (len(items),):
        raise ValueError(
            "future_video_effective_fps must stack to [B] or [B,1], got "
            f"{tuple(fps.shape)}"
        )
    return fps.to(device=device, dtype=torch.float32)


@torch.inference_mode()
def _teacher_outputs_for_items(
    *,
    items: list[dict[str, Any]],
    dino_teacher,
    depth_teacher,
    device: torch.device,
    extract_depth_teacher_outputs,
    relative_log_depth,
) -> dict[str, torch.Tensor]:
    current, future = _frames_from_items(items, device=device)
    current_dino, future_dino = dino_teacher.encode_current_and_future(
        current,
        future,
        effective_fps=_effective_fps_from_items(items, device=device),
    )
    batch = len(items)
    depth_features, dense_depth = extract_depth_teacher_outputs(
        depth_teacher,
        torch.cat([current, future], dim=0),
    )
    current_depth_features = depth_features[:batch]
    future_depth_features = depth_features[batch:]
    current_dense_depth = dense_depth[:batch]
    future_dense_depth = dense_depth[batch:]
    for name, value in (
        ("current DINO teacher", current_dino),
        ("future DINO teacher", future_dino),
        ("current Depth teacher", current_depth_features),
        ("future Depth teacher", future_depth_features),
    ):
        validate_token_features(value, feature_dim=1024, name=name)
    return {
        "current_dino": current_dino,
        "future_dino": future_dino,
        "current_depth": current_depth_features,
        "future_depth": future_depth_features,
        "current_dense_depth": current_dense_depth,
        "future_dense_depth": future_dense_depth,
        "current_relative_depth": relative_log_depth(
            current_dense_depth,
            grid_size=TOKEN_GRID_SIZE,
        ),
        "future_relative_depth": relative_log_depth(
            future_dense_depth,
            grid_size=TOKEN_GRID_SIZE,
        ),
    }


def _extract_training_cache(
    *,
    args: argparse.Namespace,
    datasets: dict[str, Any],
    train_indices: dict[str, list[int]],
    dino_teacher,
    depth_teacher,
    device: torch.device,
    helpers: dict[str, Any],
) -> ProbeTrainingCache:
    dino_chunks: list[torch.Tensor] = []
    depth_chunks: list[torch.Tensor] = []
    relative_depth_chunks: list[torch.Tensor] = []
    extracted = 0
    total = sum(len(indices) for indices in train_indices.values())
    for suite, dataset in datasets.items():
        indices = train_indices[suite]
        for start in range(0, len(indices), args.teacher_batch_size):
            selected = indices[start : start + args.teacher_batch_size]
            items = [dataset[index] for index in selected]
            outputs = _teacher_outputs_for_items(
                items=items,
                dino_teacher=dino_teacher,
                depth_teacher=depth_teacher,
                device=device,
                extract_depth_teacher_outputs=helpers["extract_depth_teacher_outputs"],
                relative_log_depth=helpers["relative_log_depth"],
            )
            dino_chunks.append(
                torch.cat([outputs["current_dino"], outputs["future_dino"]])
                .to(device="cpu", dtype=torch.bfloat16)
            )
            depth_chunks.append(
                torch.cat([outputs["current_depth"], outputs["future_depth"]])
                .to(device="cpu", dtype=torch.bfloat16)
            )
            relative_depth_chunks.append(
                torch.cat(
                    [
                        outputs["current_relative_depth"],
                        outputs["future_relative_depth"],
                    ]
                ).to(device="cpu", dtype=torch.float16)
            )
            extracted += len(items)
            print(
                json.dumps(
                    {
                        "phase": "teacher_cache",
                        "suite": suite,
                        "windows": extracted,
                        "total_windows": total,
                        "cached_frames": 2 * extracted,
                    }
                ),
                flush=True,
            )
    cache = ProbeTrainingCache(
        dino=torch.cat(dino_chunks),
        depth=torch.cat(depth_chunks),
        relative_depth=torch.cat(relative_depth_chunks),
    )
    cache.validate()
    return cache


def _train_depth_probe(
    *,
    args: argparse.Namespace,
    cache: ProbeTrainingCache,
    device: torch.device,
    helpers: dict[str, Any],
) -> tuple[nn.Module, list[dict[str, float | int]], dict[str, float | int]]:
    probe = helpers["LinearDepthProbe"](
        feature_dim=cache.depth.shape[-1],
        grid_size=TOKEN_GRID_SIZE,
    ).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=1e-4)
    features = cache.depth.to(device=device, dtype=torch.float32)
    targets = cache.relative_depth.to(device=device, dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    tracker = helpers["BestProbeStateTracker"]()
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.probe_epochs + 1):
        permutation = torch.randperm(features.shape[0], generator=generator).to(device)
        loss_sum = 0.0
        regression_sum = 0.0
        gradient_sum = 0.0
        batches = 0
        probe.train()
        for start in range(0, features.shape[0], args.probe_batch_size):
            indices = permutation[start : start + args.probe_batch_size]
            prediction = probe(features[indices])
            prediction = prediction - prediction.mean(dim=(-2, -1), keepdim=True)
            target = targets[indices]
            regression = F.smooth_l1_loss(prediction, target)
            gradient = helpers["depth_gradient_loss"](prediction, target)
            loss = regression + args.gradient_loss_weight * gradient
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            regression_sum += float(regression.detach())
            gradient_sum += float(gradient.detach())
            batches += 1
        record: dict[str, float | int] = {
            "epoch": epoch,
            "loss": loss_sum / batches,
            "regression_loss": regression_sum / batches,
            "gradient_loss": gradient_sum / batches,
        }
        history.append(record)
        tracker.consider(epoch=epoch, loss=float(record["loss"]), probe=probe)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.probe_epochs:
            print(json.dumps({"phase": "depth_probe_train", **record}), flush=True)
    tracker.restore(probe)
    probe.eval()
    return probe, history, {
        "best_epoch": tracker.best_epoch,
        "best_loss": tracker.best_loss,
    }


@torch.inference_mode()
def _planner_predictions(
    *,
    items: list[dict[str, Any]],
    wrapper,
    processor,
    metadata: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    helpers: dict[str, Any],
) -> dict[str, torch.Tensor]:
    inputs = helpers["build_planner_inputs"](
        processor,
        [item["image"] for item in items],
        [item["prompt"] for item in items],
        list(metadata["plan_token_strings"]),
    )
    inputs = helpers["move_qwen_inputs_to_device"](
        inputs,
        device,
        model_dtype=dtype,
    )
    predictions = wrapper.predict_current_future_plans(**inputs)
    result = {
        name: predictions[name].detach()
        for name in (
            "current_dino",
            "future_dino",
            "current_depth",
            "future_depth",
        )
    }
    for name, value in result.items():
        validate_token_features(value, feature_dim=1024, name=f"planner {name}")
    return result


def _decode_depth_features_224(
    *,
    probe: nn.Module,
    features: torch.Tensor,
    target_depth: torch.Tensor,
) -> torch.Tensor:
    device = next(probe.parameters()).device
    with torch.inference_mode():
        relative = probe(features.to(device=device, dtype=torch.float32))
    return decode_depth_224(relative, target_depth.to(device))


def _empty_dino_sums() -> dict[str, float | int]:
    return {
        "num_pixels": 0,
        "num_values": 0,
        "squared_error_sum": 0.0,
        "cosine_sum": 0.0,
    }


def _update_dino_sums(
    sums: dict[str, float | int],
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    metrics = compute_dino_map_metrics(prediction, target)
    sums["num_pixels"] += int(metrics["num_pixels"])
    sums["num_values"] += int(metrics["num_values"])
    sums["squared_error_sum"] += float(metrics["mse"]) * int(metrics["num_values"])
    sums["cosine_sum"] += float(metrics["mean_cosine"]) * int(metrics["num_pixels"])


def _finalize_dino_sums(sums: dict[str, float | int]) -> dict[str, float | int]:
    pixels = max(int(sums["num_pixels"]), 1)
    values = max(int(sums["num_values"]), 1)
    return {
        "num_pixels": int(sums["num_pixels"]),
        "mse": float(sums["squared_error_sum"]) / values,
        "mean_cosine": float(sums["cosine_sum"]) / pixels,
    }


def _save_training_curve_224(
    history: list[dict[str, float | int]],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(2.24, 2.24), dpi=100, constrained_layout=True)
    axis.plot(
        [int(item["epoch"]) for item in history],
        [float(item["loss"]) for item in history],
        linewidth=1.2,
    )
    axis.set_xlabel("Epoch", fontsize=7)
    axis.set_ylabel("Loss", fontsize=7)
    axis.set_title("Depth probe", fontsize=8)
    axis.tick_params(labelsize=6)
    axis.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    with Image.open(path) as image:
        if image.size != (OUTPUT_SIZE, OUTPUT_SIZE):
            raise RuntimeError(f"training curve must be 224x224, got {image.size}")


def _save_summary_csv(summary: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for scope, metrics in (("overall", summary["overall"]), *summary["suites"].items()):
        for modality, cases in metrics.items():
            for case, values in cases.items():
                rows.append(
                    {
                        "scope": scope,
                        "modality": modality,
                        "case": case,
                        **values,
                    }
                )
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _suite_name(dataset_dir: str) -> str:
    return Path(dataset_dir).name.replace("_no_noops_lerobot", "")


def main() -> None:
    args = parse_args()
    _preflight(args)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    helpers = _import_pipeline_helpers()

    datasets: dict[str, Any] = {}
    train_indices: dict[str, list[int]] = {}
    eval_indices: dict[str, list[int]] = {}
    for dataset_dir in args.fastwam_dataset_dir:
        suite = _suite_name(dataset_dir)
        if suite in datasets:
            raise ValueError(f"duplicate suite name: {suite}")
        dataset = helpers["build_suite_dataset"](args, dataset_dir)
        training, evaluation = helpers["select_disjoint_indices"](
            len(dataset),
            args.train_windows_per_suite,
            args.eval_windows_per_suite,
        )
        datasets[suite] = dataset
        train_indices[suite] = training
        eval_indices[suite] = evaluation

    dino_teacher = _build_dino_teacher(args, device)
    depth_teacher = helpers["build_depth_teacher"](args, device)
    cache = _extract_training_cache(
        args=args,
        datasets=datasets,
        train_indices=train_indices,
        dino_teacher=dino_teacher,
        depth_teacher=depth_teacher,
        device=device,
        helpers=helpers,
    )

    dino_probe = DinoPCAProbe.fit(cache.dino, seed=args.seed)
    torch.save(
        {
            "state_dict": dino_probe.state_dict(),
            "type": "global_training_set_pca_1024_to_3",
            "input_tokens": TOKEN_COUNT,
            "output_size": OUTPUT_SIZE,
            "quantile_range": [0.01, 0.99],
            "training_frames": int(cache.dino.shape[0]),
            "seed": args.seed,
        },
        args.output_dir / "dino_pca_probe.pt",
    )
    depth_probe, history, best_depth_probe = _train_depth_probe(
        args=args,
        cache=cache,
        device=device,
        helpers=helpers,
    )
    torch.save(
        {
            "state_dict": {
                name: value.detach().cpu()
                for name, value in depth_probe.state_dict().items()
            },
            "type": "shared_per_token_linear_1024_to_1",
            "feature_dim": 1024,
            "grid_size": TOKEN_GRID_SIZE,
            "output_size": OUTPUT_SIZE,
            "training_frames": int(cache.depth.shape[0]),
            "probe_epochs": args.probe_epochs,
            "probe_lr": args.probe_lr,
            "gradient_loss_weight": args.gradient_loss_weight,
            **best_depth_probe,
        },
        args.output_dir / "depth_linear_probe.pt",
    )
    (args.output_dir / "probe_training_history.json").write_text(
        json.dumps(history, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _save_training_curve_224(history, args.output_dir / "probe_training_curve.png")
    del cache
    torch.cuda.empty_cache()

    wrapper, processor, metadata, runtime_device, runtime_dtype = helpers["load_runtime"](args)
    dino_cases = ("planner_current", "planner_future")
    depth_cases = (
        "oracle_current",
        "planner_current",
        "oracle_future",
        "planner_future",
        "persistence_future",
    )
    overall_dino = {case: _empty_dino_sums() for case in dino_cases}
    overall_depth = {case: helpers["empty_depth_sums"]() for case in depth_cases}
    suite_results: dict[str, Any] = {}
    sample_dirs: list[str] = []
    total_evaluated = 0
    dino_probe = dino_probe.to("cpu")

    for suite, dataset in datasets.items():
        indices = eval_indices[suite]
        suite_dino = {case: _empty_dino_sums() for case in dino_cases}
        suite_depth = {case: helpers["empty_depth_sums"]() for case in depth_cases}
        visualized = 0
        for start in range(0, len(indices), args.planner_batch_size):
            selected = indices[start : start + args.planner_batch_size]
            items = [dataset[index] for index in selected]
            teacher = _teacher_outputs_for_items(
                items=items,
                dino_teacher=dino_teacher,
                depth_teacher=depth_teacher,
                device=device,
                extract_depth_teacher_outputs=helpers["extract_depth_teacher_outputs"],
                relative_log_depth=helpers["relative_log_depth"],
            )
            planner = _planner_predictions(
                items=items,
                wrapper=wrapper,
                processor=processor,
                metadata=metadata,
                device=runtime_device,
                dtype=runtime_dtype,
                helpers=helpers,
            )
            dino_maps = {
                "teacher_current": dino_probe.project_224(teacher["current_dino"].cpu()),
                "planner_current": dino_probe.project_224(planner["current_dino"].cpu()),
                "teacher_future": dino_probe.project_224(teacher["future_dino"].cpu()),
                "planner_future": dino_probe.project_224(planner["future_dino"].cpu()),
            }
            for case, prediction, target in (
                ("planner_current", dino_maps["planner_current"], dino_maps["teacher_current"]),
                ("planner_future", dino_maps["planner_future"], dino_maps["teacher_future"]),
            ):
                _update_dino_sums(suite_dino[case], prediction, target)
                _update_dino_sums(overall_dino[case], prediction, target)

            current_dense = teacher["current_dense_depth"]
            future_dense = teacher["future_dense_depth"]
            decoded_depth = {
                "oracle_current": _decode_depth_features_224(
                    probe=depth_probe,
                    features=teacher["current_depth"],
                    target_depth=current_dense,
                ),
                "planner_current": _decode_depth_features_224(
                    probe=depth_probe,
                    features=planner["current_depth"],
                    target_depth=current_dense,
                ),
                "oracle_future": _decode_depth_features_224(
                    probe=depth_probe,
                    features=teacher["future_depth"],
                    target_depth=future_dense,
                ),
                "planner_future": _decode_depth_features_224(
                    probe=depth_probe,
                    features=planner["future_depth"],
                    target_depth=future_dense,
                ),
                "persistence_future": _decode_depth_features_224(
                    probe=depth_probe,
                    features=teacher["current_depth"],
                    target_depth=future_dense,
                ),
            }
            resized_targets = {
                "current": resize_depth_target_224(current_dense),
                "future": resize_depth_target_224(future_dense),
            }
            for case in depth_cases:
                target = (
                    resized_targets["current"]
                    if case.endswith("current")
                    else resized_targets["future"]
                )
                helpers["update_depth_sums"](suite_depth[case], decoded_depth[case], target)
                helpers["update_depth_sums"](overall_depth[case], decoded_depth[case], target)

            for local_index, (dataset_index, item) in enumerate(
                zip(selected, items, strict=True)
            ):
                if visualized >= args.visualizations_per_suite:
                    break
                sample_dir = (
                    args.output_dir
                    / "samples"
                    / suite
                    / f"index{dataset_index:09d}"
                )
                save_sample_outputs(
                    output_dir=sample_dir,
                    current_rgb=item["current_image"],
                    future_rgb=item["keyframe_images"][0],
                    instruction=str(item["prompt"]),
                    dino_maps={
                        "dino_teacher_current_224": dino_maps["teacher_current"][local_index],
                        "dino_planner_current_224": dino_maps["planner_current"][local_index],
                        "dino_teacher_future_224": dino_maps["teacher_future"][local_index],
                        "dino_planner_future_224": dino_maps["planner_future"][local_index],
                    },
                    depth_maps={
                        "depth_target_current_224": resized_targets["current"][local_index],
                        "depth_planner_current_224": decoded_depth["planner_current"][local_index],
                        "depth_target_future_224": resized_targets["future"][local_index],
                        "depth_planner_future_224": decoded_depth["planner_future"][local_index],
                    },
                )
                sample_dirs.append(str(sample_dir.relative_to(args.output_dir)))
                visualized += 1
            total_evaluated += len(items)
            print(
                json.dumps(
                    {
                        "phase": "planner_eval",
                        "suite": suite,
                        "evaluated": start + len(items),
                        "suite_total": len(indices),
                        "total_evaluated": total_evaluated,
                    }
                ),
                flush=True,
            )
        suite_results[suite] = {
            "dino": {
                case: _finalize_dino_sums(suite_dino[case])
                for case in dino_cases
            },
            "depth": {
                case: helpers["finalize_depth_sums"](suite_depth[case])
                for case in depth_cases
            },
        }

    summary = {
        "protocol": {
            "suites": list(datasets),
            "train_windows_per_suite": args.train_windows_per_suite,
            "eval_windows_per_suite": args.eval_windows_per_suite,
            "training_frames": 2 * sum(len(item) for item in train_indices.values()),
            "evaluation_frames": 2 * sum(len(item) for item in eval_indices.values()),
            "future_offset_frames": 8,
            "output_size": [OUTPUT_SIZE, OUTPUT_SIZE],
            "dino_probe": "global training-set PCA 1024->3 with fixed 1/99 percentiles",
            "depth_probe": "shared per-token linear 1024->1 relative-log-depth",
            "depth_loss": "SmoothL1 + gradient_loss_weight * gradient_SmoothL1",
            "depth_gradient_loss_weight": args.gradient_loss_weight,
            "depth_probe_best": best_depth_probe,
        },
        "evaluation_warning": (
            "Probe train/eval window indices are disjoint, but the planner was trained on "
            "all LIBERO episodes; this is not a held-out planner benchmark."
        ),
        "overall": {
            "dino": {
                case: _finalize_dino_sums(overall_dino[case])
                for case in dino_cases
            },
            "depth": {
                case: helpers["finalize_depth_sums"](overall_depth[case])
                for case in depth_cases
            },
        },
        "suites": suite_results,
        "sample_dirs": sample_dirs,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _save_summary_csv(summary, args.output_dir / "summary.csv")
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
