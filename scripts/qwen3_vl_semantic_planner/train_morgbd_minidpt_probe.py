#!/usr/bin/env python3
"""Train a dense MiniDPT visualization probe for frozen MoRGBD features."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from morgbd_minidpt_probe import (
    MiniDPTDepthProbe,
    dense_log_depth_target,
    multiscale_gradient_loss,
    silog_loss,
)


TOKEN_GRID = 16
TOKEN_COUNT = TOKEN_GRID * TOKEN_GRID
OUTPUT_SIZE = 224
TIMEPOINTS = ("current", "future")


@dataclass
class DenseProbeCache:
    features: torch.Tensor
    log_depth: torch.Tensor
    records: list[dict[str, object]]
    grid: int = TOKEN_GRID
    feature_dim: int = 1024
    output_size: int = OUTPUT_SIZE

    def validate(self) -> None:
        count = len(self.records)
        expected_features = (count, self.grid * self.grid, self.feature_dim)
        expected_depth = (count, 1, self.output_size, self.output_size)
        if tuple(self.features.shape) != expected_features:
            raise ValueError(
                f"features must be {expected_features}, got {tuple(self.features.shape)}"
            )
        if tuple(self.log_depth.shape) != expected_depth:
            raise ValueError(
                f"log_depth must be {expected_depth}, got {tuple(self.log_depth.shape)}"
            )
        if not bool(torch.isfinite(self.features).all()):
            raise ValueError("features contain non-finite values")
        if not bool(torch.isfinite(self.log_depth).all()):
            raise ValueError("log_depth contains non-finite values")
        for index, record in enumerate(self.records):
            if set(record) != {"suite", "dataset_index", "time"}:
                raise ValueError(f"record {index} has invalid keys: {sorted(record)}")
            if not isinstance(record["suite"], str) or not record["suite"]:
                raise ValueError(f"record {index} has invalid suite")
            if not isinstance(record["dataset_index"], int):
                raise ValueError(f"record {index} has invalid dataset_index")
            if record["time"] not in TIMEPOINTS:
                raise ValueError(f"record {index} has invalid time")

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "features": self.features,
            "log_depth": self.log_depth,
            "records": self.records,
            "grid": self.grid,
            "feature_dim": self.feature_dim,
            "output_size": self.output_size,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "DenseProbeCache":
        cache = cls(
            features=payload["features"],
            log_depth=payload["log_depth"],
            records=list(payload["records"]),
            grid=int(payload["grid"]),
            feature_dim=int(payload["feature_dim"]),
            output_size=int(payload["output_size"]),
        )
        cache.validate()
        return cache


def _window_keys(cache: DenseProbeCache) -> set[tuple[str, int]]:
    return {
        (str(record["suite"]), int(record["dataset_index"]))
        for record in cache.records
    }


def validate_disjoint_caches(
    training: DenseProbeCache,
    evaluation: DenseProbeCache,
) -> None:
    training.validate()
    evaluation.validate()
    overlap = _window_keys(training) & _window_keys(evaluation)
    if overlap:
        preview = sorted(overlap)[:5]
        raise ValueError(f"training/evaluation window overlap: {preview}")


class BestValidationState:
    def __init__(self) -> None:
        self.best_step = 0
        self.best_loss = math.inf
        self._state: dict[str, torch.Tensor] | None = None

    def consider(
        self,
        *,
        step: int,
        loss: float,
        model: nn.Module,
    ) -> bool:
        if not math.isfinite(float(loss)):
            raise ValueError(f"validation loss must be finite, got {loss}")
        if float(loss) >= self.best_loss:
            return False
        self.best_step = int(step)
        self.best_loss = float(loss)
        self._state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        return True

    def restore(self, model: nn.Module) -> None:
        if self._state is None:
            raise RuntimeError("no validation state has been recorded")
        model.load_state_dict(self._state, strict=True)


def align_log_depth(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must share [B,1,H,W]")
    residual = (target.float() - prediction.float()).flatten(1)
    shift = residual.median(dim=1).values[:, None, None, None]
    return prediction.float() + shift


def compute_log_depth_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float | int]:
    aligned = align_log_depth(prediction, target)
    truth_log = target.float()
    prediction_depth = aligned.exp().clamp_min(1e-6)
    truth_depth = truth_log.exp().clamp_min(1e-6)
    ratio = torch.maximum(
        prediction_depth / truth_depth,
        truth_depth / prediction_depth,
    )
    pred_flat = aligned.flatten(1)
    truth_flat = truth_log.flatten(1)
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    truth_centered = truth_flat - truth_flat.mean(dim=1, keepdim=True)
    pearson = (
        (pred_centered * truth_centered).sum(dim=1)
        / (
            pred_centered.square().sum(dim=1).sqrt()
            * truth_centered.square().sum(dim=1).sqrt()
        ).clamp_min(1e-8)
    )
    return {
        "num_frames": int(prediction.shape[0]),
        "num_pixels": int(prediction.numel()),
        "abs_rel": float(
            ((prediction_depth - truth_depth).abs() / truth_depth).mean()
        ),
        "rmse": float(
            (prediction_depth - truth_depth).square().mean().sqrt()
        ),
        "delta1": float((ratio < 1.25).float().mean()),
        "pearson": float(pearson.mean()),
        "gradient_error": float(
            multiscale_gradient_loss(aligned, truth_log)
        ),
    }


def probe_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    gradient_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = silog_loss(prediction.float(), target.float())
    gradient = multiscale_gradient_loss(prediction.float(), target.float())
    return scale + float(gradient_weight) * gradient, scale, gradient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MiniDPT on frozen MoRGBD tokens and dense MoGe depth."
    )
    parser.add_argument("--fastwam-data-config", type=Path, required=True)
    parser.add_argument("--fastwam-dataset-dir", action="append", required=True)
    parser.add_argument("--fastwam-text-embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--fastwam-pretrained-norm-stats", type=Path, required=True)
    parser.add_argument("--depth-moge-path", type=Path, required=True)
    parser.add_argument("--depth-morgbd-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--linear-probe", type=Path)
    parser.add_argument("--train-windows-per-suite", type=int, default=256)
    parser.add_argument("--eval-windows-per-suite", type=int, default=64)
    parser.add_argument("--teacher-batch-size", type=int, default=8)
    parser.add_argument("--probe-batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--probe-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-loss-weight", type=float, default=0.5)
    parser.add_argument("--feature-channels", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--reuse-cache", action="store_true")
    return parser.parse_args()


def _preflight(args: argparse.Namespace) -> None:
    positive = (
        "train_windows_per_suite",
        "eval_windows_per_suite",
        "teacher_batch_size",
        "probe_batch_size",
        "eval_batch_size",
        "steps",
        "validate_every",
        "log_every",
        "feature_channels",
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("probe_lr", "weight_decay", "gradient_loss_weight"):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.probe_lr == 0:
        raise ValueError("--probe-lr must be positive")
    required = (
        args.fastwam_data_config,
        args.fastwam_text_embedding_cache_dir,
        args.fastwam_pretrained_norm_stats,
        args.depth_moge_path,
        args.depth_morgbd_path,
        *(Path(path) for path in args.fastwam_dataset_dir),
    )
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("missing required inputs:\n" + "\n".join(missing))
    for variable in ("LINGBOT_SRC_ROOT", "UTILS3D_MOGE_PATH"):
        value = os.environ.get(variable)
        if not value or not Path(value).exists():
            raise FileNotFoundError(f"{variable} must point to an existing path")
    args.output_dir.mkdir(parents=True, exist_ok=True)


def _suite_name(path: str | Path) -> str:
    name = Path(path).name.lower()
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        if suite in name:
            return suite
    return Path(path).name


def _pipeline_helpers():
    from evaluate_lingbot_current_future_planner import _build_suite_dataset
    from train_depth_probe_visualization import (
        _build_depth_teacher,
        extract_depth_teacher_outputs,
        select_disjoint_indices,
    )

    return _build_suite_dataset, _build_depth_teacher, extract_depth_teacher_outputs, select_disjoint_indices


@torch.inference_mode()
def _extract_cache(
    *,
    datasets: dict[str, Any],
    indices: dict[str, list[int]],
    teacher,
    teacher_batch_size: int,
    device: torch.device,
    extract_depth_teacher_outputs,
    phase: str,
) -> DenseProbeCache:
    feature_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    records: list[dict[str, object]] = []
    completed = 0
    total = sum(len(value) for value in indices.values())
    for suite, dataset in datasets.items():
        suite_indices = indices[suite]
        for start in range(0, len(suite_indices), teacher_batch_size):
            selected = suite_indices[start : start + teacher_batch_size]
            items = [dataset[index] for index in selected]
            current = torch.stack([item["current_image"] for item in items]).permute(0, 3, 1, 2)
            future = torch.stack([item["keyframe_images"][0] for item in items]).permute(0, 3, 1, 2)
            frames = torch.cat([current, future]).to(device, non_blocking=True)
            features, dense_depth = extract_depth_teacher_outputs(teacher, frames)
            feature_chunks.append(features.to(device="cpu", dtype=torch.bfloat16))
            target_chunks.append(
                dense_log_depth_target(dense_depth, output_size=OUTPUT_SIZE).to(
                    device="cpu",
                    dtype=torch.float16,
                )
            )
            records.extend(
                {"suite": suite, "dataset_index": int(index), "time": "current"}
                for index in selected
            )
            records.extend(
                {"suite": suite, "dataset_index": int(index), "time": "future"}
                for index in selected
            )
            completed += len(selected)
            print(
                json.dumps(
                    {
                        "phase": f"{phase}_teacher_cache",
                        "suite": suite,
                        "windows": completed,
                        "total_windows": total,
                        "frames": 2 * completed,
                    }
                ),
                flush=True,
            )
    cache = DenseProbeCache(
        features=torch.cat(feature_chunks),
        log_depth=torch.cat(target_chunks),
        records=records,
    )
    cache.validate()
    return cache


def _load_or_extract_caches(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> tuple[DenseProbeCache, DenseProbeCache, Path]:
    cache_path = args.cache_path or (args.output_dir / "teacher_cache.pt")
    if args.reuse_cache and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        training = DenseProbeCache.from_payload(payload["training"])
        evaluation = DenseProbeCache.from_payload(payload["evaluation"])
        validate_disjoint_caches(training, evaluation)
        return training, evaluation, cache_path

    build_dataset, build_teacher, extract_outputs, select_indices = _pipeline_helpers()
    datasets: dict[str, Any] = {}
    train_indices: dict[str, list[int]] = {}
    eval_indices: dict[str, list[int]] = {}
    for dataset_dir in args.fastwam_dataset_dir:
        suite = _suite_name(dataset_dir)
        if suite in datasets:
            raise ValueError(f"duplicate suite name: {suite}")
        dataset = build_dataset(args, dataset_dir)
        training, evaluation = select_indices(
            len(dataset),
            args.train_windows_per_suite,
            args.eval_windows_per_suite,
        )
        datasets[suite] = dataset
        train_indices[suite] = training
        eval_indices[suite] = evaluation
    teacher = build_teacher(args, device)
    training_cache = _extract_cache(
        datasets=datasets,
        indices=train_indices,
        teacher=teacher,
        teacher_batch_size=args.teacher_batch_size,
        device=device,
        extract_depth_teacher_outputs=extract_outputs,
        phase="train",
    )
    evaluation_cache = _extract_cache(
        datasets=datasets,
        indices=eval_indices,
        teacher=teacher,
        teacher_batch_size=args.teacher_batch_size,
        device=device,
        extract_depth_teacher_outputs=extract_outputs,
        phase="eval",
    )
    validate_disjoint_caches(training_cache, evaluation_cache)
    torch.save(
        {
            "training": training_cache.to_payload(),
            "evaluation": evaluation_cache.to_payload(),
            "protocol": {
                "train_windows_per_suite": args.train_windows_per_suite,
                "eval_windows_per_suite": args.eval_windows_per_suite,
                "seed": args.seed,
                "teacher_input": "224x448 composite resized to 256x256",
            },
        },
        cache_path,
    )
    del teacher
    torch.cuda.empty_cache()
    return training_cache, evaluation_cache, cache_path


def _amp_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


@torch.inference_mode()
def _validation_loss(
    probe: MiniDPTDepthProbe,
    cache: DenseProbeCache,
    *,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    gradient_weight: float,
) -> float:
    probe.eval()
    total = 0.0
    frames = 0
    for start in range(0, cache.features.shape[0], batch_size):
        features = cache.features[start : start + batch_size].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        target = cache.log_depth[start : start + batch_size].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda" and amp_dtype != torch.float32,
        ):
            prediction = probe(features)
        loss, _scale, _gradient = probe_objective(
            prediction,
            target,
            gradient_weight=gradient_weight,
        )
        count = features.shape[0]
        total += float(loss) * count
        frames += count
    return total / frames


def _train_probe(
    args: argparse.Namespace,
    *,
    training: DenseProbeCache,
    evaluation: DenseProbeCache,
    device: torch.device,
) -> tuple[MiniDPTDepthProbe, list[dict[str, float | int]], BestValidationState]:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    probe = MiniDPTDepthProbe(
        in_dim=training.feature_dim,
        feat=args.feature_channels,
        grid=training.grid,
        output_size=training.output_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=args.probe_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    amp_dtype = _amp_dtype(args.dtype)
    tracker = BestValidationState()
    history: list[dict[str, float | int]] = []
    recent: list[float] = []
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            training.features.shape[0],
            (args.probe_batch_size,),
            generator=generator,
        )
        features = training.features[indices].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        target = training.log_depth[indices].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        probe.train()
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda" and amp_dtype != torch.float32,
        ):
            prediction = probe(features)
        loss, scale_loss, gradient_loss = probe_objective(
            prediction,
            target,
            gradient_weight=args.gradient_loss_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        recent.append(float(loss.detach()))
        record: dict[str, float | int] = {
            "step": step,
            "train_loss": float(loss.detach()),
            "silog_loss": float(scale_loss.detach()),
            "gradient_loss": float(gradient_loss.detach()),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        should_validate = step == 1 or step % args.validate_every == 0 or step == args.steps
        if should_validate:
            validation_loss = _validation_loss(
                probe,
                evaluation,
                batch_size=args.eval_batch_size,
                device=device,
                amp_dtype=amp_dtype,
                gradient_weight=args.gradient_loss_weight,
            )
            record["validation_loss"] = validation_loss
            record["new_best"] = int(
                tracker.consider(
                    step=step,
                    loss=validation_loss,
                    model=probe,
                )
            )
        history.append(record)
        if step == 1 or step % args.log_every == 0 or should_validate:
            log = {
                "phase": "probe_train",
                **record,
                "recent_train_loss": sum(recent[-args.log_every :])
                / min(len(recent), args.log_every),
            }
            print(json.dumps(log), flush=True)
    tracker.restore(probe)
    probe.eval()
    return probe, history, tracker


@torch.inference_mode()
def _evaluate_decoder(
    decoder,
    cache: DenseProbeCache,
    *,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[str, float | int]:
    sums: dict[str, float] = {}
    frames = 0
    for start in range(0, cache.features.shape[0], batch_size):
        features = cache.features[start : start + batch_size].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        target = cache.log_depth[start : start + batch_size].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda" and amp_dtype != torch.float32,
        ):
            prediction = decoder(features)
        if prediction.ndim == 3:
            prediction = prediction.unsqueeze(1)
        if prediction.shape[-2:] != target.shape[-2:]:
            prediction = F.interpolate(
                prediction.float(),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        metrics = compute_log_depth_metrics(prediction.float(), target)
        count = int(metrics["num_frames"])
        for name, value in metrics.items():
            if name not in {"num_frames", "num_pixels"}:
                sums[name] = sums.get(name, 0.0) + float(value) * count
        frames += count
    return {
        "num_frames": frames,
        **{name: value / frames for name, value in sums.items()},
    }


def _load_linear_probe(path: Path, device: torch.device):
    from train_depth_probe_visualization import LinearDepthProbe

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    probe = LinearDepthProbe(
        feature_dim=int(config.get("feature_dim", 1024)),
        grid_size=int(config.get("grid_size", TOKEN_GRID)),
    )
    probe.load_state_dict(checkpoint["state_dict"], strict=True)
    return probe.to(device).eval()


def main() -> None:
    args = _parse_args()
    _preflight(args)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    training, evaluation, cache_path = _load_or_extract_caches(args, device=device)
    probe, history, tracker = _train_probe(
        args,
        training=training,
        evaluation=evaluation,
        device=device,
    )
    checkpoint_path = args.output_dir / "minidpt_depth_probe.pt"
    checkpoint = {
        "state_dict": {
            name: value.detach().cpu()
            for name, value in probe.state_dict().items()
        },
        "config": probe.config(),
        "training": {
            "steps": args.steps,
            "batch_size": args.probe_batch_size,
            "lr": args.probe_lr,
            "weight_decay": args.weight_decay,
            "gradient_loss_weight": args.gradient_loss_weight,
            "best_step": tracker.best_step,
            "best_validation_loss": tracker.best_loss,
            "seed": args.seed,
        },
        "teacher": "MoGe dense depth -> MoRGBD 256x1024 features",
        "cache_path": str(cache_path),
    }
    torch.save(checkpoint, checkpoint_path)
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    amp_dtype = _amp_dtype(args.dtype)
    metrics: dict[str, object] = {
        "protocol": {
            "train_frames": len(training.records),
            "eval_frames": len(evaluation.records),
            "disjoint_windows": True,
            "output_size": [OUTPUT_SIZE, OUTPUT_SIZE],
            "probe_input": [TOKEN_COUNT, training.feature_dim],
        },
        "minidpt_teacher": _evaluate_decoder(
            probe,
            evaluation,
            batch_size=args.eval_batch_size,
            device=device,
            amp_dtype=amp_dtype,
        ),
    }
    if args.linear_probe is not None:
        linear = _load_linear_probe(args.linear_probe, device)
        metrics["linear_teacher"] = _evaluate_decoder(
            linear,
            evaluation,
            batch_size=args.eval_batch_size,
            device=device,
            amp_dtype=amp_dtype,
        )
        mini = metrics["minidpt_teacher"]
        baseline = metrics["linear_teacher"]
        metrics["acceptance"] = {
            "pearson_improved": mini["pearson"] > baseline["pearson"],
            "gradient_error_improved": mini["gradient_error"]
            < baseline["gradient_error"],
        }
    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "phase": "complete",
                "checkpoint": str(checkpoint_path),
                "cache": str(cache_path),
                "metrics": metrics,
            },
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
