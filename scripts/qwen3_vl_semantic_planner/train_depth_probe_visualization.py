#!/usr/bin/env python3
"""Train a lightweight depth probe and visualize planner depth tokens.

The probe is deliberately limited to one shared 1024->1 linear projection per
16x16 token.  It predicts scale-invariant relative log-depth at token resolution;
bilinear upsampling is used only for visualization and pixel metrics.  A powerful
decoder is intentionally avoided so the probe measures information already present
in the depth tokens rather than learning to hallucinate scene geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def select_disjoint_indices(
    length: int,
    train_count: int,
    eval_count: int,
) -> tuple[list[int], list[int]]:
    if min(length, train_count, eval_count) < 0:
        raise ValueError("length and split counts must be non-negative")
    total = train_count + eval_count
    if total > length:
        raise ValueError(
            f"requested {total} disjoint samples from dataset of length {length}"
        )
    if total == 0:
        return [], []
    candidates = (
        torch.linspace(0, length - 1, steps=total)
        .round()
        .to(torch.long)
        .unique(sorted=True)
        .tolist()
    )
    if len(candidates) != total:
        raise RuntimeError("failed to construct unique deterministic split indices")
    if eval_count == 0:
        return candidates, []
    eval_positions = (
        torch.linspace(0, total - 1, steps=eval_count)
        .round()
        .to(torch.long)
        .unique(sorted=True)
        .tolist()
    )
    evaluation = [candidates[position] for position in eval_positions]
    evaluation_set = set(evaluation)
    training = [index for index in candidates if index not in evaluation_set]
    if len(training) != train_count or len(evaluation) != eval_count:
        raise RuntimeError("deterministic split produced incorrect sample counts")
    return training, evaluation


def _as_bhw(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    elif depth.ndim == 2:
        depth = depth.unsqueeze(0)
    if depth.ndim != 3:
        raise ValueError(f"depth must be [B,H,W] or [B,1,H,W], got {tuple(depth.shape)}")
    return depth


def relative_log_depth(depth: torch.Tensor, *, grid_size: int) -> torch.Tensor:
    depth = _as_bhw(depth).to(torch.float32)
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    depth = torch.nan_to_num(depth, nan=1e-6, posinf=1e6, neginf=1e-6).clamp_min(1e-6)
    downsampled = F.interpolate(
        depth.unsqueeze(1),
        size=(grid_size, grid_size),
        mode="area",
    )[:, 0]
    log_depth = downsampled.clamp_min(1e-6).log()
    return log_depth - log_depth.mean(dim=(-2, -1), keepdim=True)


class LinearDepthProbe(nn.Module):
    def __init__(self, *, feature_dim: int = 1024, grid_size: int = 16) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.grid_size = int(grid_size)
        self.projection = nn.Linear(self.feature_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        expected_tokens = self.grid_size * self.grid_size
        if features.ndim != 3 or features.shape[1:] != (
            expected_tokens,
            self.feature_dim,
        ):
            raise ValueError(
                f"features must be [B,{expected_tokens},{self.feature_dim}], "
                f"got {tuple(features.shape)}"
            )
        output = self.projection(features).squeeze(-1)
        return output.reshape(features.shape[0], self.grid_size, self.grid_size)


class BestProbeStateTracker:
    def __init__(self) -> None:
        self.best_loss = math.inf
        self.best_epoch = 0
        self._state: dict[str, torch.Tensor] | None = None

    def consider(self, *, epoch: int, loss: float, probe: nn.Module) -> None:
        if not math.isfinite(float(loss)):
            raise ValueError(f"probe loss must be finite, got {loss}")
        if float(loss) >= self.best_loss:
            return
        self.best_loss = float(loss)
        self.best_epoch = int(epoch)
        self._state = {
            name: value.detach().to(device="cpu").clone()
            for name, value in probe.state_dict().items()
        }

    def restore(self, probe: nn.Module) -> None:
        if self._state is None:
            raise RuntimeError("no probe state has been tracked")
        probe.load_state_dict(self._state, strict=True)


def depth_gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share [B,H,W]")
    pred_dx = prediction[:, :, 1:] - prediction[:, :, :-1]
    pred_dy = prediction[:, 1:, :] - prediction[:, :-1, :]
    target_dx = target[:, :, 1:] - target[:, :, :-1]
    target_dy = target[:, 1:, :] - target[:, :-1, :]
    return F.smooth_l1_loss(pred_dx, target_dx) + F.smooth_l1_loss(pred_dy, target_dy)


def decode_relative_log_depth(
    relative_log_prediction: torch.Tensor,
    target_depth: torch.Tensor,
) -> torch.Tensor:
    prediction = _as_bhw(relative_log_prediction).to(torch.float32)
    target = _as_bhw(target_depth).to(torch.float32)
    prediction = F.interpolate(
        prediction.unsqueeze(1),
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    prediction = prediction - prediction.mean(dim=(-2, -1), keepdim=True)
    target_log = target.clamp_min(1e-6).log()
    shift = (target_log - prediction).flatten(1).median(dim=1).values.view(-1, 1, 1)
    return (prediction + shift).exp()


def compute_depth_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float | int]:
    prediction = _as_bhw(prediction).to(torch.float32)
    target = _as_bhw(target).to(torch.float32)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target depth shapes differ")
    valid = torch.isfinite(prediction) & torch.isfinite(target) & (target > 1e-6)
    if not valid.any():
        raise ValueError("depth metric has no valid pixels")
    pred = prediction[valid].clamp_min(1e-6)
    truth = target[valid].clamp_min(1e-6)
    ratio = torch.maximum(pred / truth, truth / pred)
    return {
        "num_valid_pixels": int(valid.sum()),
        "abs_rel": float(((pred - truth).abs() / truth).mean()),
        "rmse": float((pred - truth).square().mean().sqrt()),
        "delta1": float((ratio < 1.25).float().mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--fastwam-data-config", type=Path, required=True)
    parser.add_argument("--fastwam-dataset-dir", action="append", required=True)
    parser.add_argument("--fastwam-text-embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--fastwam-pretrained-norm-stats", type=Path, required=True)
    parser.add_argument("--depth-moge-path", type=Path, required=True)
    parser.add_argument("--depth-morgbd-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-windows-per-suite", type=int, default=64)
    parser.add_argument("--eval-windows-per-suite", type=int, default=16)
    parser.add_argument("--teacher-batch-size", type=int, default=4)
    parser.add_argument("--planner-batch-size", type=int, default=2)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--probe-lr", type=float, default=3e-3)
    parser.add_argument("--gradient-loss-weight", type=float, default=0.2)
    parser.add_argument("--visualizations-per-suite", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def _import_eval_helpers():
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from evaluate_lingbot_current_future_planner import (
        _build_suite_dataset,
        _load_runtime,
    )

    return _build_suite_dataset, _load_runtime


def _build_depth_teacher(args: argparse.Namespace, device: torch.device):
    module_dir = Path(__file__).resolve().parent / "lingbot_dino_4b"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    from depth_target import DepthTargetEncoder

    return DepthTargetEncoder(
        moge_path=args.depth_moge_path,
        morgbd_path=args.depth_morgbd_path,
        input_size=256,
        num_tokens=256,
        device=device,
        lingbot_root=os.environ["LINGBOT_SRC_ROOT"],
        utils3d_path=os.environ["UTILS3D_MOGE_PATH"],
    ).eval()


@torch.inference_mode()
def extract_depth_teacher_outputs(
    teacher,
    frames_b3hw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = teacher._prep(frames_b3hw)
    unit_images = images / 255.0
    output = teacher.moge.infer(
        unit_images,
        resolution_level=teacher.resolution_level,
        num_tokens=teacher.num_tokens,
        apply_mask=False,
    )
    depth = output["depth"].detach()
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth[:, 0]
    elif depth.ndim == 2:
        depth = depth.unsqueeze(0)
    depth = torch.nan_to_num(depth, nan=1e-6, posinf=1e6, neginf=1e-6).clamp_min(1e-6)
    features, _cls = teacher.morgbd.infer_feat(
        unit_images,
        depth,
        depth_down_scale=1,
        resolution_level=teacher.resolution_level,
        num_tokens=teacher.num_tokens,
        enable_depth_mask=False,
    )
    features = features.permute(0, 2, 3, 1).reshape(features.shape[0], -1, features.shape[1])
    return features.detach(), depth.float()


def _frames_from_items(items: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    current = torch.stack([item["current_image"] for item in items]).permute(0, 3, 1, 2)
    future = torch.stack([item["keyframe_images"][0] for item in items]).permute(0, 3, 1, 2)
    return current, future


def _extract_probe_training_cache(
    *,
    args: argparse.Namespace,
    datasets: dict[str, Any],
    train_indices: dict[str, list[int]],
    teacher,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_chunks = []
    target_chunks = []
    extracted = 0
    total = sum(len(indices) for indices in train_indices.values())
    for suite, dataset in datasets.items():
        indices = train_indices[suite]
        for start in range(0, len(indices), args.teacher_batch_size):
            selected = indices[start : start + args.teacher_batch_size]
            items = [dataset[index] for index in selected]
            current, future = _frames_from_items(items)
            features, depth = extract_depth_teacher_outputs(
                teacher,
                torch.cat([current, future], dim=0),
            )
            feature_chunks.append(features.to(device="cpu", dtype=torch.bfloat16))
            target_chunks.append(
                relative_log_depth(depth, grid_size=16).to(device="cpu", dtype=torch.float16)
            )
            extracted += len(items)
            print(
                json.dumps(
                    {"phase": "probe_cache", "suite": suite, "windows": extracted, "total": total}
                ),
                flush=True,
            )
    return torch.cat(feature_chunks), torch.cat(target_chunks)


def _train_probe(
    *,
    args: argparse.Namespace,
    features: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> tuple[LinearDepthProbe, list[dict[str, float | int]], dict[str, float | int]]:
    probe = LinearDepthProbe(feature_dim=features.shape[-1], grid_size=16).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=1e-4)
    features = features.to(device=device, dtype=torch.float32)
    targets = targets.to(device=device, dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    history = []
    tracker = BestProbeStateTracker()
    for epoch in range(1, args.probe_epochs + 1):
        permutation = torch.randperm(features.shape[0], generator=generator).to(device)
        total_loss = 0.0
        batches = 0
        probe.train()
        for start in range(0, features.shape[0], args.probe_batch_size):
            indices = permutation[start : start + args.probe_batch_size]
            prediction = probe(features[indices])
            prediction = prediction - prediction.mean(dim=(-2, -1), keepdim=True)
            target = targets[indices]
            regression = F.smooth_l1_loss(prediction, target)
            gradient = depth_gradient_loss(prediction, target)
            loss = regression + args.gradient_loss_weight * gradient
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        record = {"epoch": epoch, "loss": total_loss / max(batches, 1)}
        history.append(record)
        tracker.consider(epoch=epoch, loss=float(record["loss"]), probe=probe)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.probe_epochs:
            print(json.dumps({"phase": "probe_train", **record}), flush=True)
    tracker.restore(probe)
    probe.eval()
    return probe, history, {
        "best_epoch": tracker.best_epoch,
        "best_loss": tracker.best_loss,
    }


def _planner_depth_predictions(
    *,
    items: list[dict[str, Any]],
    wrapper,
    processor,
    metadata: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from train_qwen3vl4b_lingbot_dino_planner import (
        build_planner_inputs,
        move_qwen_inputs_to_device,
    )

    inputs = build_planner_inputs(
        processor,
        [item["image"] for item in items],
        [item["prompt"] for item in items],
        list(metadata["plan_token_strings"]),
    )
    inputs = move_qwen_inputs_to_device(inputs, device, model_dtype=dtype)
    with torch.inference_mode():
        predictions = wrapper.predict_current_future_plans(**inputs)
    return {
        "current": predictions["current_depth"],
        "future": predictions["future_depth"],
    }


def _empty_depth_sums() -> dict[str, float | int]:
    return {"pixels": 0, "abs_rel_sum": 0.0, "squared_error_sum": 0.0, "delta1_hits": 0}


def _update_depth_sums(
    sums: dict[str, float | int],
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    prediction = _as_bhw(prediction).detach().float().cpu()
    target = _as_bhw(target).detach().float().cpu()
    valid = torch.isfinite(prediction) & torch.isfinite(target) & (target > 1e-6)
    pred = prediction[valid].clamp_min(1e-6)
    truth = target[valid].clamp_min(1e-6)
    ratio = torch.maximum(pred / truth, truth / pred)
    sums["pixels"] += int(valid.sum())
    sums["abs_rel_sum"] += float(((pred - truth).abs() / truth).sum())
    sums["squared_error_sum"] += float((pred - truth).square().sum())
    sums["delta1_hits"] += int((ratio < 1.25).sum())


def _finalize_depth_sums(sums: dict[str, float | int]) -> dict[str, float | int]:
    pixels = max(int(sums["pixels"]), 1)
    return {
        "num_valid_pixels": int(sums["pixels"]),
        "abs_rel": float(sums["abs_rel_sum"]) / pixels,
        "rmse": math.sqrt(float(sums["squared_error_sum"]) / pixels),
        "delta1": int(sums["delta1_hits"]) / pixels,
    }


def _decode_probe(
    probe: LinearDepthProbe,
    features: torch.Tensor,
    target_depth: torch.Tensor,
) -> torch.Tensor:
    with torch.inference_mode():
        relative = probe(features.to(dtype=next(probe.parameters()).dtype))
    return decode_relative_log_depth(relative, target_depth)


def _save_depth_visualization(
    *,
    path: Path,
    item: dict[str, Any],
    current_target: torch.Tensor,
    future_target: torch.Tensor,
    current_oracle: torch.Tensor,
    future_oracle: torch.Tensor,
    current_planner: torch.Tensor,
    future_planner: torch.Tensor,
    current_metrics: dict[str, float | int],
    future_metrics: dict[str, float | int],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
    axes[0, 0].imshow(item["image"])
    axes[0, 0].set_title("Current RGB")
    axes[0, 1].imshow(Image.fromarray(item["keyframe_images"][0].numpy()))
    axes[0, 1].set_title("Future RGB (offset 8)")
    axes[0, 2].text(
        0,
        1,
        textwrap.fill(str(item["prompt"]), width=42),
        va="top",
        fontsize=10,
    )
    axes[0, 2].set_title("Instruction")
    axes[0, 3].text(
        0,
        1,
        "Current planner\n"
        f"AbsRel={current_metrics['abs_rel']:.3f}\n"
        f"RMSE={current_metrics['rmse']:.3f}\n"
        f"delta1={current_metrics['delta1']:.3f}\n\n"
        "Future planner\n"
        f"AbsRel={future_metrics['abs_rel']:.3f}\n"
        f"RMSE={future_metrics['rmse']:.3f}\n"
        f"delta1={future_metrics['delta1']:.3f}",
        va="top",
        family="monospace",
    )
    axes[0, 3].set_title("Probe metrics")
    rows = (
        (current_target, current_oracle, current_planner, "Current"),
        (future_target, future_oracle, future_planner, "Future"),
    )
    for row_index, (target, oracle, planner, label) in enumerate(rows, start=1):
        target_np = target.detach().float().cpu().numpy()
        oracle_np = oracle.detach().float().cpu().numpy()
        planner_np = planner.detach().float().cpu().numpy()
        low, high = torch.quantile(target.detach().float().cpu(), torch.tensor([0.02, 0.98])).tolist()
        axes[row_index, 0].imshow(target_np, cmap="viridis", vmin=low, vmax=high)
        axes[row_index, 0].set_title(f"{label} MoGe target")
        axes[row_index, 1].imshow(oracle_np, cmap="viridis", vmin=low, vmax=high)
        axes[row_index, 1].set_title(f"{label} GT-token probe")
        axes[row_index, 2].imshow(planner_np, cmap="viridis", vmin=low, vmax=high)
        axes[row_index, 2].set_title(f"{label} planner-token probe")
        error = abs(planner_np - target_np)
        axes[row_index, 3].imshow(error, cmap="magma")
        axes[row_index, 3].set_title(f"{label} absolute error")
    for axis in axes.flat:
        axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_training_curve(history: list[dict[str, float | int]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot([item["epoch"] for item in history], [item["loss"] for item in history])
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Probe loss")
    axis.set_title("Linear depth probe training")
    axis.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_metric_overview(summary: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = summary["overall"]
    cases = ["oracle_current", "planner_current", "oracle_future", "planner_future", "persistence_future"]
    labels = ["GT current", "Planner current", "GT future", "Planner future", "Persistence future"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "abs_rel", "AbsRel (lower is better)"),
        (axes[1], "rmse", "RMSE (lower is better)"),
        (axes[2], "delta1", "delta1 (higher is better)"),
    ):
        axis.bar(labels, [metrics[case][metric] for case in cases])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=30)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    build_suite_dataset, load_runtime = _import_eval_helpers()
    datasets = {}
    train_indices = {}
    eval_indices = {}
    for dataset_dir in args.fastwam_dataset_dir:
        suite = Path(dataset_dir).name.replace("_no_noops_lerobot", "")
        dataset = build_suite_dataset(args, dataset_dir)
        training, evaluation = select_disjoint_indices(
            len(dataset),
            args.train_windows_per_suite,
            args.eval_windows_per_suite,
        )
        datasets[suite] = dataset
        train_indices[suite] = training
        eval_indices[suite] = evaluation

    teacher = _build_depth_teacher(args, device)
    features, targets = _extract_probe_training_cache(
        args=args,
        datasets=datasets,
        train_indices=train_indices,
        teacher=teacher,
    )
    probe, history, best_probe = _train_probe(
        args=args,
        features=features,
        targets=targets,
        device=device,
    )
    torch.save(
        {
            "state_dict": probe.state_dict(),
            "feature_dim": probe.feature_dim,
            "grid_size": probe.grid_size,
            "train_windows_per_suite": args.train_windows_per_suite,
            "probe_epochs": args.probe_epochs,
            **best_probe,
        },
        args.output_dir / "depth_linear_probe.pt",
    )
    (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    _save_training_curve(history, args.output_dir / "probe_training_curve.png")
    del features, targets
    torch.cuda.empty_cache()

    wrapper, processor, metadata, runtime_device, runtime_dtype = load_runtime(args)
    cases = (
        "oracle_current",
        "planner_current",
        "oracle_future",
        "planner_future",
        "persistence_future",
    )
    overall_sums = {case: _empty_depth_sums() for case in cases}
    suite_results = {}
    visualization_paths = []
    total = 0
    for suite, dataset in datasets.items():
        indices = eval_indices[suite]
        suite_sums = {case: _empty_depth_sums() for case in cases}
        visualized = 0
        for start in range(0, len(indices), args.planner_batch_size):
            selected = indices[start : start + args.planner_batch_size]
            items = [dataset[index] for index in selected]
            current_frames, future_frames = _frames_from_items(items)
            gt_features, gt_depth = extract_depth_teacher_outputs(
                teacher,
                torch.cat([current_frames, future_frames], dim=0),
            )
            batch_size = len(items)
            current_features, future_features = gt_features[:batch_size], gt_features[batch_size:]
            current_depth, future_depth = gt_depth[:batch_size], gt_depth[batch_size:]
            planner_features = _planner_depth_predictions(
                items=items,
                wrapper=wrapper,
                processor=processor,
                metadata=metadata,
                device=runtime_device,
                dtype=runtime_dtype,
            )
            decoded = {
                "oracle_current": _decode_probe(probe, current_features, current_depth),
                "planner_current": _decode_probe(probe, planner_features["current"], current_depth),
                "oracle_future": _decode_probe(probe, future_features, future_depth),
                "planner_future": _decode_probe(probe, planner_features["future"], future_depth),
                "persistence_future": _decode_probe(probe, current_features, future_depth),
            }
            case_targets = {
                "oracle_current": current_depth,
                "planner_current": current_depth,
                "oracle_future": future_depth,
                "planner_future": future_depth,
                "persistence_future": future_depth,
            }
            for case in cases:
                _update_depth_sums(suite_sums[case], decoded[case], case_targets[case])
                _update_depth_sums(overall_sums[case], decoded[case], case_targets[case])
            for local_index, (dataset_index, item) in enumerate(zip(selected, items, strict=True)):
                if visualized >= args.visualizations_per_suite:
                    break
                current_metric = compute_depth_metrics(
                    decoded["planner_current"][local_index : local_index + 1],
                    current_depth[local_index : local_index + 1],
                )
                future_metric = compute_depth_metrics(
                    decoded["planner_future"][local_index : local_index + 1],
                    future_depth[local_index : local_index + 1],
                )
                figure_path = (
                    args.output_dir
                    / "visualizations"
                    / f"{suite}_index{dataset_index:09d}.png"
                )
                _save_depth_visualization(
                    path=figure_path,
                    item=item,
                    current_target=current_depth[local_index],
                    future_target=future_depth[local_index],
                    current_oracle=decoded["oracle_current"][local_index],
                    future_oracle=decoded["oracle_future"][local_index],
                    current_planner=decoded["planner_current"][local_index],
                    future_planner=decoded["planner_future"][local_index],
                    current_metrics=current_metric,
                    future_metrics=future_metric,
                )
                visualization_paths.append(str(figure_path))
                visualized += 1
            total += len(items)
            print(
                json.dumps(
                    {
                        "phase": "probe_eval",
                        "suite": suite,
                        "evaluated": start + len(items),
                        "suite_total": len(indices),
                        "total_evaluated": total,
                    }
                ),
                flush=True,
            )
        suite_results[suite] = {
            case: _finalize_depth_sums(suite_sums[case]) for case in cases
        }

    summary = {
        "probe": {
            "type": "shared_per_token_linear_1024_to_1",
            "grid_size": 16,
            "output_visualization_size": 256,
            "target": "scale_invariant_relative_log_moge_depth",
            "train_windows_per_suite": args.train_windows_per_suite,
            "eval_windows_per_suite": args.eval_windows_per_suite,
            "probe_epochs": args.probe_epochs,
            "final_epoch_loss": history[-1]["loss"],
            **best_probe,
        },
        "evaluation_warning": (
            "Probe train/eval indices are disjoint, but the planner itself was trained on all "
            "LIBERO episodes; this is not a held-out planner benchmark."
        ),
        "overall": {case: _finalize_depth_sums(overall_sums[case]) for case in cases},
        "suites": suite_results,
        "visualizations": visualization_paths,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _save_metric_overview(summary, args.output_dir / "depth_probe_metrics.png")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
