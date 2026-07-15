#!/usr/bin/env python3
"""Evaluate current/future LingBot DINO+Depth planner checkpoints.

The FastWAM training configuration uses every LIBERO episode (val proportion 0),
so this script reports a deterministic in-distribution evaluation subset rather
than claiming a held-out episode split.  It disables video augmentation, samples
indices evenly from every requested suite, compares future predictions against
current-feature persistence and spatial-collapse baselines, and writes PCA maps.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from PIL import Image


BRANCHES = ("current_dino", "current_depth", "future_dino", "future_depth")


def select_eval_indices(*, length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    count = min(int(count), int(length))
    if count == 1:
        return [0]
    return (
        torch.linspace(0, length - 1, steps=count)
        .round()
        .to(torch.long)
        .unique(sorted=True)
        .tolist()
    )


def _empty_metric_sums() -> dict[str, float | int]:
    return {
        "num_samples": 0,
        "num_tokens": 0,
        "num_values": 0,
        "squared_error_sum": 0.0,
        "smooth_l1_sum": 0.0,
        "cosine_sum": 0.0,
        "retrieval_hits": 0,
        "pred_norm_sum": 0.0,
        "target_norm_sum": 0.0,
        "pred_dispersion_sum": 0.0,
        "target_dispersion_sum": 0.0,
    }


def _update_metric_sums(
    sums: dict[str, float | int],
    pred: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError(
            "pred and target must share [B, N, D], got "
            f"{tuple(pred.shape)} and {tuple(target.shape)}"
        )
    pred = pred.detach().to(device="cpu", dtype=torch.float32)
    target = target.detach().to(device="cpu", dtype=torch.float32)
    batch, tokens, dim = pred.shape
    pred_flat = pred.flatten(0, 1)
    target_flat = target.flatten(0, 1)
    cosine = F.cosine_similarity(pred_flat, target_flat, dim=-1)
    pred_normalized = F.normalize(pred, dim=-1)
    target_normalized = F.normalize(target, dim=-1)
    nearest = torch.bmm(pred_normalized, target_normalized.transpose(1, 2)).argmax(dim=-1)
    token_ids = torch.arange(tokens).view(1, -1)
    pred_dispersion = (pred - pred.mean(dim=1, keepdim=True)).norm(dim=-1).mean(dim=1)
    target_dispersion = (target - target.mean(dim=1, keepdim=True)).norm(dim=-1).mean(dim=1)

    sums["num_samples"] += batch
    sums["num_tokens"] += batch * tokens
    sums["num_values"] += batch * tokens * dim
    sums["squared_error_sum"] += float((pred - target).square().sum())
    sums["smooth_l1_sum"] += float(F.smooth_l1_loss(pred, target, reduction="sum"))
    sums["cosine_sum"] += float(cosine.sum())
    sums["retrieval_hits"] += int((nearest == token_ids).sum())
    sums["pred_norm_sum"] += float(pred_flat.norm(dim=-1).sum())
    sums["target_norm_sum"] += float(target_flat.norm(dim=-1).sum())
    sums["pred_dispersion_sum"] += float(pred_dispersion.sum())
    sums["target_dispersion_sum"] += float(target_dispersion.sum())


def _finalize_metric_sums(sums: dict[str, float | int]) -> dict[str, float | int]:
    samples = max(int(sums["num_samples"]), 1)
    tokens = max(int(sums["num_tokens"]), 1)
    values = max(int(sums["num_values"]), 1)
    pred_norm = float(sums["pred_norm_sum"]) / tokens
    target_norm = float(sums["target_norm_sum"]) / tokens
    pred_dispersion = float(sums["pred_dispersion_sum"]) / samples
    target_dispersion = float(sums["target_dispersion_sum"]) / samples
    return {
        "num_samples": int(sums["num_samples"]),
        "num_tokens": int(sums["num_tokens"]),
        "mse_per_value": float(sums["squared_error_sum"]) / values,
        "smooth_l1_per_value": float(sums["smooth_l1_sum"]) / values,
        "mean_cosine": float(sums["cosine_sum"]) / tokens,
        "token_retrieval_top1": int(sums["retrieval_hits"]) / tokens,
        "pred_norm": pred_norm,
        "target_norm": target_norm,
        "norm_ratio": pred_norm / max(target_norm, 1e-12),
        "pred_token_dispersion": pred_dispersion,
        "target_token_dispersion": target_dispersion,
        "dispersion_ratio": pred_dispersion / max(target_dispersion, 1e-12),
    }


def compute_branch_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float | int]:
    sums = _empty_metric_sums()
    _update_metric_sums(sums, pred, target)
    return _finalize_metric_sums(sums)


def compute_future_baselines(
    current_target: torch.Tensor,
    future_target: torch.Tensor,
) -> dict[str, float]:
    persistence = compute_branch_metrics(current_target, future_target)
    collapsed = future_target.mean(dim=1, keepdim=True).expand_as(future_target)
    collapsed_metrics = compute_branch_metrics(collapsed, future_target)
    return {
        "persistence_mse_per_value": float(persistence["mse_per_value"]),
        "persistence_smooth_l1_per_value": float(persistence["smooth_l1_per_value"]),
        "persistence_mean_cosine": float(persistence["mean_cosine"]),
        "collapsed_mean_mse_per_value": float(collapsed_metrics["mse_per_value"]),
        "collapsed_mean_smooth_l1_per_value": float(
            collapsed_metrics["smooth_l1_per_value"]
        ),
        "collapsed_mean_cosine": float(collapsed_metrics["mean_cosine"]),
    }


def joint_pca_maps(
    features: Sequence[torch.Tensor],
    *,
    grid_size: int,
) -> list[torch.Tensor]:
    if not features:
        return []
    tensors = [item.detach().to(device="cpu", dtype=torch.float32) for item in features]
    expected_tokens = grid_size * grid_size
    dim = tensors[0].shape[-1]
    if any(item.ndim != 2 or item.shape != (expected_tokens, dim) for item in tensors):
        raise ValueError(
            f"every feature map must be [{expected_tokens}, {dim}]"
        )
    combined = torch.cat(tensors, dim=0)
    centered = combined - combined.mean(dim=0, keepdim=True)
    q = min(3, centered.shape[0], centered.shape[1])
    with torch.random.fork_rng():
        torch.manual_seed(0)
        _u, _s, vectors = torch.pca_lowrank(centered, q=q, center=False, niter=4)
    projected = centered @ vectors[:, :q]
    if q < 3:
        projected = F.pad(projected, (0, 3 - q))
    low = torch.quantile(projected, 0.01, dim=0)
    high = torch.quantile(projected, 0.99, dim=0)
    projected = ((projected - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)
    chunks = projected.split(expected_tokens)
    return [chunk.reshape(grid_size, grid_size, 3) for chunk in chunks]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--samples-per-suite", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--visualizations-per-suite", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _load_runtime(args: argparse.Namespace):
    trainer_dir = Path(__file__).resolve().parent
    if str(trainer_dir) not in sys.path:
        sys.path.insert(0, str(trainer_dir))
    from lingbot_dino_4b.dino_depth_plan_provider import (
        validate_checkpoint_files,
        validate_planner_metadata,
    )
    from train_qwen3vl4b_lingbot_dino_planner import PlannerWrapper
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    checkpoint_dir = validate_checkpoint_files(args.checkpoint_dir)
    metadata = json.loads((checkpoint_dir / "planner_meta.json").read_text())
    validate_planner_metadata(metadata)
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    processor = AutoProcessor.from_pretrained(
        checkpoint_dir / "processor",
        local_files_only=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint_dir / "qwen3vl_lora_or_model",
        torch_dtype=dtype,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to(device)
    wrapper = PlannerWrapper.from_exported_checkpoint(
        model=model,
        checkpoint_dir=checkpoint_dir,
        metadata=metadata,
    ).to(device).eval()
    return wrapper, processor, metadata, device, dtype


def _build_suite_dataset(args: argparse.Namespace, dataset_dir: str):
    trainer_dir = Path(__file__).resolve().parent
    if str(trainer_dir) not in sys.path:
        sys.path.insert(0, str(trainer_dir))
    from hydra.utils import instantiate
    from train_qwen3vl4b_lingbot_dino_planner import (
        FastWAMOnlinePlannerDataset,
        prepare_fastwam_data_config,
    )

    root_config = prepare_fastwam_data_config(
        args.fastwam_data_config,
        dataset_dirs=[dataset_dir],
        text_embedding_cache_dir=args.fastwam_text_embedding_cache_dir,
        pretrained_norm_stats=args.fastwam_pretrained_norm_stats,
    )
    config = root_config.data.train
    config.is_training_set = False
    config.val_set_proportion = 0.0
    config.video_augmentation = None
    base_dataset = instantiate(config)
    return FastWAMOnlinePlannerDataset.from_dataset(base_dataset, offsets=[8])


def _build_teachers(args: argparse.Namespace, device: torch.device):
    trainer_dir = Path(__file__).resolve().parent
    lingbot_dir = trainer_dir / "lingbot_dino_4b"
    if str(lingbot_dir) not in sys.path:
        sys.path.insert(0, str(lingbot_dir))
    from depth_target import DepthTargetEncoder
    from dino_video_target import DinoVideoTargetEncoder

    dino = DinoVideoTargetEncoder(
        ckpt_path=args.dino_teacher_ckpt,
        config_path=args.dino_teacher_config,
        input_size=256,
        device=device,
        lingbot_root=os.environ.get("LINGBOT_SRC_ROOT", ""),
    ).eval()
    depth = DepthTargetEncoder(
        moge_path=args.depth_moge_path,
        morgbd_path=args.depth_morgbd_path,
        input_size=256,
        num_tokens=256,
        device=device,
        lingbot_root=os.environ.get("LINGBOT_SRC_ROOT", ""),
        utils3d_path=os.environ.get("UTILS3D_MOGE_PATH", ""),
    ).eval()
    return dino, depth


def _predict_and_target(
    *,
    items: list[dict[str, Any]],
    wrapper,
    processor,
    metadata: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    dino_teacher,
    depth_teacher,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    trainer_dir = Path(__file__).resolve().parent
    if str(trainer_dir) not in sys.path:
        sys.path.insert(0, str(trainer_dir))
    from train_qwen3vl4b_lingbot_dino_planner import (
        build_planner_inputs,
        move_qwen_inputs_to_device,
    )

    model_inputs = build_planner_inputs(
        processor,
        [item["image"] for item in items],
        [item["prompt"] for item in items],
        list(metadata["plan_token_strings"]),
    )
    model_inputs = move_qwen_inputs_to_device(
        model_inputs,
        device,
        model_dtype=dtype,
    )
    with torch.inference_mode():
        predictions = wrapper.predict_current_future_plans(**model_inputs)
        current = torch.stack([item["current_image"] for item in items]).permute(0, 3, 1, 2)
        future = torch.stack([item["keyframe_images"][0] for item in items]).permute(0, 3, 1, 2)
        current_dino, future_dino = dino_teacher.encode_current_and_future(current, future)
        current_depth, future_depth = depth_teacher.encode_current_and_future(current, future)
    targets = {
        "current_dino": current_dino.float(),
        "future_dino": future_dino.float(),
        "current_depth": current_depth.float(),
        "future_depth": future_depth.float(),
    }
    return predictions, targets


def _save_sample_visualization(
    *,
    path: Path,
    item: dict[str, Any],
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    sample_metrics: dict[str, dict[str, float | int]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dino_maps = joint_pca_maps(
        [
            targets["current_dino"],
            predictions["current_dino"],
            targets["future_dino"],
            predictions["future_dino"],
        ],
        grid_size=16,
    )
    depth_maps = joint_pca_maps(
        [
            targets["current_depth"],
            predictions["current_depth"],
            targets["future_depth"],
            predictions["future_depth"],
        ],
        grid_size=16,
    )
    future_image = Image.fromarray(item["keyframe_images"][0].numpy())
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
    axes[0, 0].imshow(item["image"])
    axes[0, 0].set_title("Current RGB (planner input)")
    axes[0, 1].imshow(future_image)
    axes[0, 1].set_title("Future RGB (offset 8)")
    axes[0, 2].text(
        0.0,
        1.0,
        textwrap.fill(str(item["prompt"]), width=42),
        va="top",
        fontsize=10,
    )
    axes[0, 2].set_title("Instruction")
    score_lines = []
    for branch in BRANCHES:
        metric = sample_metrics[branch]
        score_lines.append(
            f"{branch}:\n  cos={metric['mean_cosine']:.3f} "
            f"retr={metric['token_retrieval_top1']:.3f}"
        )
    axes[0, 3].text(0.0, 1.0, "\n".join(score_lines), va="top", family="monospace")
    axes[0, 3].set_title("Per-sample metrics")
    titles = (
        "Current DINO GT",
        "Current DINO Pred",
        "Future DINO GT",
        "Future DINO Pred",
    )
    for axis, feature_map, title in zip(axes[1], dino_maps, titles, strict=True):
        axis.imshow(feature_map.numpy(), interpolation="nearest")
        axis.set_title(title)
    titles = (
        "Current Depth GT",
        "Current Depth Pred",
        "Future Depth GT",
        "Future Depth Pred",
    )
    for axis, feature_map, title in zip(axes[2], depth_maps, titles, strict=True):
        axis.imshow(feature_map.numpy(), interpolation="nearest")
        axis.set_title(title)
    for axis in axes.flat:
        axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _new_scope_sums() -> dict[str, dict[str, float | int]]:
    return {
        **{branch: _empty_metric_sums() for branch in BRANCHES},
        "future_dino_persistence": _empty_metric_sums(),
        "future_dino_collapsed": _empty_metric_sums(),
        "future_depth_persistence": _empty_metric_sums(),
        "future_depth_collapsed": _empty_metric_sums(),
    }


def _update_scope_sums(
    sums: dict[str, dict[str, float | int]],
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> None:
    for branch in BRANCHES:
        _update_metric_sums(sums[branch], predictions[branch], targets[branch])
    for modality in ("dino", "depth"):
        future = targets[f"future_{modality}"]
        current = targets[f"current_{modality}"]
        collapsed = future.mean(dim=1, keepdim=True).expand_as(future)
        _update_metric_sums(sums[f"future_{modality}_persistence"], current, future)
        _update_metric_sums(sums[f"future_{modality}_collapsed"], collapsed, future)


def _finalize_scope(sums: dict[str, dict[str, float | int]]) -> dict[str, Any]:
    branches = {branch: _finalize_metric_sums(sums[branch]) for branch in BRANCHES}
    baselines = {}
    for modality in ("dino", "depth"):
        persistence = _finalize_metric_sums(sums[f"future_{modality}_persistence"])
        collapsed = _finalize_metric_sums(sums[f"future_{modality}_collapsed"])
        model = branches[f"future_{modality}"]
        primary = "mse_per_value" if modality == "dino" else "smooth_l1_per_value"
        baselines[modality] = {
            "persistence": persistence,
            "collapsed_spatial_mean": collapsed,
            "model_relative_improvement_vs_persistence": 1.0
            - float(model[primary]) / max(float(persistence[primary]), 1e-12),
            "model_relative_improvement_vs_collapsed_mean": 1.0
            - float(model[primary]) / max(float(collapsed[primary]), 1e-12),
            "model_cosine_delta_vs_persistence": float(model["mean_cosine"])
            - float(persistence["mean_cosine"]),
        }
    return {"branches": branches, "future_baselines": baselines}


def _save_metrics_overview(summary: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overall = summary["overall"]
    branch_metrics = overall["branches"]
    labels = list(BRANCHES)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].bar(labels, [branch_metrics[x]["mean_cosine"] for x in labels])
    axes[0, 0].set_title("Token cosine similarity (higher is better)")
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].tick_params(axis="x", rotation=25)
    axes[0, 1].bar(labels, [branch_metrics[x]["token_retrieval_top1"] for x in labels])
    axes[0, 1].set_title("Within-sample token retrieval top-1")
    axes[0, 1].tick_params(axis="x", rotation=25)
    for axis, modality, metric in (
        (axes[1, 0], "dino", "mse_per_value"),
        (axes[1, 1], "depth", "smooth_l1_per_value"),
    ):
        model = branch_metrics[f"future_{modality}"][metric]
        baseline = overall["future_baselines"][modality]
        persistence = baseline["persistence"][metric]
        collapsed = baseline["collapsed_spatial_mean"][metric]
        axis.bar(["model", "persistence", "spatial mean"], [model, persistence, collapsed])
        axis.set_title(f"Future {modality.upper()} {metric} (lower is better)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    rows = []
    for scope, result in [("overall", summary["overall"]), *summary["suites"].items()]:
        for branch, metrics in result["branches"].items():
            rows.append({"scope": scope, "branch": branch, **metrics})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.samples_per_suite <= 0 or args.batch_size <= 0:
        raise ValueError("samples-per-suite and batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    wrapper, processor, metadata, device, dtype = _load_runtime(args)
    dino_teacher, depth_teacher = _build_teachers(args, device)

    overall_sums = _new_scope_sums()
    suite_results: dict[str, Any] = {}
    per_sample_path = args.output_dir / "per_sample.jsonl"
    visualization_paths: list[str] = []
    total_samples = 0
    with per_sample_path.open("w", encoding="utf-8") as per_sample_file:
        for dataset_dir in args.fastwam_dataset_dir:
            suite = Path(dataset_dir).name.replace("_no_noops_lerobot", "")
            dataset = _build_suite_dataset(args, dataset_dir)
            indices = select_eval_indices(
                length=len(dataset),
                count=args.samples_per_suite,
            )
            suite_sums = _new_scope_sums()
            visualized = 0
            for start in range(0, len(indices), args.batch_size):
                batch_indices = indices[start : start + args.batch_size]
                items = [dataset[index] for index in batch_indices]
                predictions, targets = _predict_and_target(
                    items=items,
                    wrapper=wrapper,
                    processor=processor,
                    metadata=metadata,
                    device=device,
                    dtype=dtype,
                    dino_teacher=dino_teacher,
                    depth_teacher=depth_teacher,
                )
                _update_scope_sums(suite_sums, predictions, targets)
                _update_scope_sums(overall_sums, predictions, targets)
                for local_index, (dataset_index, item) in enumerate(
                    zip(batch_indices, items, strict=True)
                ):
                    sample_metrics = {
                        branch: compute_branch_metrics(
                            predictions[branch][local_index : local_index + 1],
                            targets[branch][local_index : local_index + 1],
                        )
                        for branch in BRANCHES
                    }
                    record = {
                        "suite": suite,
                        "dataset_index": dataset_index,
                        "sample_id": item["sample_id"],
                        "instruction": item["prompt"],
                        "branches": sample_metrics,
                        "future_dino_baselines": compute_future_baselines(
                            targets["current_dino"][local_index : local_index + 1],
                            targets["future_dino"][local_index : local_index + 1],
                        ),
                        "future_depth_baselines": compute_future_baselines(
                            targets["current_depth"][local_index : local_index + 1],
                            targets["future_depth"][local_index : local_index + 1],
                        ),
                    }
                    per_sample_file.write(json.dumps(record) + "\n")
                    if visualized < args.visualizations_per_suite:
                        figure_path = (
                            args.output_dir
                            / "visualizations"
                            / f"{suite}_index{dataset_index:09d}.png"
                        )
                        _save_sample_visualization(
                            path=figure_path,
                            item=item,
                            predictions={
                                branch: predictions[branch][local_index]
                                for branch in BRANCHES
                            },
                            targets={
                                branch: targets[branch][local_index]
                                for branch in BRANCHES
                            },
                            sample_metrics=sample_metrics,
                        )
                        visualization_paths.append(str(figure_path))
                        visualized += 1
                total_samples += len(items)
                print(
                    json.dumps(
                        {
                            "suite": suite,
                            "evaluated": start + len(items),
                            "suite_total": len(indices),
                            "total_evaluated": total_samples,
                        }
                    ),
                    flush=True,
                )
            suite_results[suite] = _finalize_scope(suite_sums)

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "evaluation_protocol": {
            "split": "deterministic_fixed_in_distribution_subset",
            "warning": (
                "Training used all LIBERO episodes (val_set_proportion=0); "
                "these are not unseen held-out episodes."
            ),
            "samples_per_suite": args.samples_per_suite,
            "num_suites": len(args.fastwam_dataset_dir),
            "num_samples": total_samples,
            "augmentation": "disabled",
            "future_offset": 8,
        },
        "overall": _finalize_scope(overall_sums),
        "suites": suite_results,
        "visualizations": visualization_paths,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_csv(summary, args.output_dir / "summary.csv")
    _save_metrics_overview(summary, args.output_dir / "metrics_overview.png")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
