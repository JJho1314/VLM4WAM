#!/usr/bin/env python3
"""Reference-style dense MiniDPT visualization for the 4B MoRGBD planner."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from morgbd_minidpt_probe import MiniDPTDepthProbe, dense_log_depth_target
from train_morgbd_minidpt_probe import compute_log_depth_metrics
from train_dino_depth_probe_visualization import (
    DinoPCAProbe,
    _build_dino_teacher,
    _import_pipeline_helpers,
    _planner_predictions,
    _suite_name,
    _teacher_outputs_for_items,
)
from visualize_dual_camera_probes import (
    project_dino_cameras_224,
    split_rgb_cameras_224,
)


CAMERAS = ("main", "wrist")
TIMEPOINTS = ("current", "future")
DINO_SOURCES = ("teacher", "planner")
DEPTH_SOURCES = ("teacher", "planner", "moge")
OUTPUT_SIZE = 224
EXPECTED_PANEL_NAMES = tuple(
    [
        f"observation_{camera}_{time}"
        for camera in CAMERAS
        for time in TIMEPOINTS
    ]
    + [
        f"dino_{source}_{camera}_{time}"
        for source in DINO_SOURCES
        for camera in CAMERAS
        for time in TIMEPOINTS
    ]
    + [
        f"depth_{source}_{camera}_{time}"
        for source in DEPTH_SOURCES
        for camera in CAMERAS
        for time in TIMEPOINTS
    ]
)


def unsquish_and_split(value: torch.Tensor) -> dict[str, torch.Tensor]:
    """Restore a square composite to native 2:1 aspect, then split main/wrist."""
    if value.ndim != 4:
        raise ValueError(f"value must be [B,C,H,W], got {tuple(value.shape)}")
    restored = F.interpolate(
        value.float(),
        size=(OUTPUT_SIZE, 2 * OUTPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    return {
        "main": restored[..., :OUTPUT_SIZE],
        "wrist": restored[..., OUTPUT_SIZE:],
    }


def normalize_log_depth_pair(
    teacher_log_depth: torch.Tensor,
    planner_log_depth: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Jointly normalize teacher/planner disparity per sample for honest colors."""
    if teacher_log_depth.shape != planner_log_depth.shape:
        raise ValueError("teacher and planner log-depth shapes differ")
    if teacher_log_depth.ndim != 4 or teacher_log_depth.shape[1] != 1:
        raise ValueError("log depth must be [B,1,H,W]")
    teacher = (-teacher_log_depth.float()).exp()
    planner = (-planner_log_depth.float()).exp()
    both = torch.stack([teacher, planner], dim=1)
    flat = both.flatten(1)
    low = flat.min(dim=1).values[:, None, None, None]
    high = flat.max(dim=1).values[:, None, None, None]
    scale = (high - low).clamp_min(1e-6)
    return {
        "teacher": ((teacher - low) / scale).clamp(0.0, 1.0),
        "planner": ((planner - low) / scale).clamp(0.0, 1.0),
    }


def normalize_log_depth_reference(log_depth: torch.Tensor) -> torch.Tensor:
    if log_depth.ndim != 4 or log_depth.shape[1] != 1:
        raise ValueError("log depth must be [B,1,H,W]")
    disparity = (-log_depth.float()).exp()
    flat = disparity.flatten(1)
    low = flat.min(dim=1).values[:, None, None, None]
    high = flat.max(dim=1).values[:, None, None, None]
    return ((disparity - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)


def _as_pil_rgb(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
    else:
        tensor = torch.as_tensor(value).detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        if tensor.ndim != 3 or tensor.shape[-1] not in (1, 3):
            raise ValueError(f"panel must be HWC/CHW RGB, got {tuple(tensor.shape)}")
        if tensor.shape[-1] == 1:
            tensor = tensor.expand(-1, -1, 3)
        tensor = tensor.float()
        if float(tensor.max()) <= 1.5:
            tensor = tensor * 255.0
        array = tensor.round().clamp(0, 255).to(torch.uint8).numpy()
        image = Image.fromarray(array, mode="RGB")
    if image.size != (OUTPUT_SIZE, OUTPUT_SIZE):
        image = image.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.BILINEAR)
    return image


def _turbo_image(value: torch.Tensor) -> Image.Image:
    import matplotlib

    normalized = torch.as_tensor(value).detach().cpu().float().squeeze()
    if tuple(normalized.shape) != (OUTPUT_SIZE, OUTPUT_SIZE):
        raise ValueError(
            f"depth panel must be [{OUTPUT_SIZE},{OUTPUT_SIZE}], "
            f"got {tuple(normalized.shape)}"
        )
    rgba = matplotlib.colormaps["turbo"](
        normalized.clamp(0.0, 1.0).numpy()
    )
    return Image.fromarray(
        (rgba[..., :3] * 255.0).round().astype(np.uint8),
        mode="RGB",
    )


def _render_grid(
    *,
    path: Path,
    camera: str,
    instruction: str,
    panels: dict[str, Image.Image],
    metrics: dict[str, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        [
            f"observation_{camera}_current",
            f"observation_{camera}_future",
            f"dino_teacher_{camera}_current",
            f"dino_planner_{camera}_current",
            f"dino_teacher_{camera}_future",
            f"dino_planner_{camera}_future",
        ],
        [
            f"observation_{camera}_current",
            f"observation_{camera}_future",
            f"depth_teacher_{camera}_current",
            f"depth_planner_{camera}_current",
            f"depth_teacher_{camera}_future",
            f"depth_planner_{camera}_future",
        ],
        [
            None,
            None,
            f"depth_moge_{camera}_current",
            None,
            f"depth_moge_{camera}_future",
            None,
        ],
    ]
    titles = [
        [
            "Current RGB",
            "Future RGB",
            "DINO cur TARGET",
            "DINO cur PRED",
            "DINO fut TARGET",
            "DINO fut PRED",
        ],
        [
            "Current RGB",
            "Future RGB",
            "Depth cur TARGET(v2)",
            "Depth cur PRED(v2)",
            "Depth fut TARGET(v2)",
            "Depth fut PRED(v2)",
        ],
        ["", "", "MoGe-full cur GT", "", "MoGe-full fut GT", ""],
    ]
    blank = Image.new("RGB", (OUTPUT_SIZE, OUTPUT_SIZE), color="white")
    figure, axes = plt.subplots(3, 6, figsize=(20, 10.5))
    for row in range(3):
        for column in range(6):
            name = rows[row][column]
            axes[row, column].imshow(blank if name is None else panels[name])
            axes[row, column].set_box_aspect(1.0)
            if titles[row][column]:
                axes[row, column].set_title(titles[row][column], fontsize=9)
            axes[row, column].axis("off")
    figure.suptitle(
        f"[{camera} cam] {instruction[:100]}\n"
        f"dino_mse cur={metrics['dino_current_mse']:.4f} "
        f"fut={metrics['dino_future_mse']:.4f} | "
        f"depth_absrel cur={metrics['depth_current_abs_rel']:.3f} "
        f"fut={metrics['depth_future_abs_rel']:.3f}",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)


def save_reference_style_sample(
    *,
    output_dir: Path,
    sample_index: int,
    instruction: str,
    panels: dict[str, Any],
    metrics: dict[str, float],
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if set(panels) != set(EXPECTED_PANEL_NAMES):
        missing = sorted(set(EXPECTED_PANEL_NAMES) - set(panels))
        extra = sorted(set(panels) - set(EXPECTED_PANEL_NAMES))
        raise ValueError(f"panel names differ; missing={missing}, extra={extra}")
    required_metrics = {
        "dino_current_mse",
        "dino_future_mse",
        "depth_current_abs_rel",
        "depth_future_abs_rel",
    }
    if set(metrics) != required_metrics:
        raise ValueError(f"metrics must contain exactly {sorted(required_metrics)}")
    converted = {name: _as_pil_rgb(value) for name, value in panels.items()}
    written: list[Path] = []
    for name, image in converted.items():
        path = output_dir / f"{name}.png"
        image.save(path)
        written.append(path)
    instruction_path = output_dir / "instruction.txt"
    instruction_path.write_text(instruction.rstrip() + "\n", encoding="utf-8")
    written.append(instruction_path)
    for camera in CAMERAS:
        path = output_dir / f"sample_{sample_index:02d}_{camera}.png"
        _render_grid(
            path=path,
            camera=camera,
            instruction=instruction,
            panels=converted,
            metrics=metrics,
        )
        written.append(path)
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render reference-style MiniDPT target/prediction depth by camera."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--minidpt-probe", type=Path, required=True)
    parser.add_argument("--dino-probe", type=Path, required=True)
    parser.add_argument("--fastwam-data-config", type=Path, required=True)
    parser.add_argument("--fastwam-dataset-dir", action="append", required=True)
    parser.add_argument("--fastwam-text-embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--fastwam-pretrained-norm-stats", type=Path, required=True)
    parser.add_argument("--dino-teacher-ckpt", type=Path, required=True)
    parser.add_argument("--dino-teacher-config", type=Path, required=True)
    parser.add_argument("--depth-moge-path", type=Path, required=True)
    parser.add_argument("--depth-morgbd-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-windows-per-suite", type=int, default=256)
    parser.add_argument("--eval-windows-per-suite", type=int, default=16)
    parser.add_argument("--planner-batch-size", type=int, default=8)
    parser.add_argument("--visualizations-per-suite", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def _preflight(args: argparse.Namespace) -> None:
    for name in (
        "train_windows_per_suite",
        "eval_windows_per_suite",
        "planner_batch_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.visualizations_per_suite < 0:
        raise ValueError("--visualizations-per-suite must be non-negative")
    required = (
        args.checkpoint_dir,
        args.minidpt_probe,
        args.dino_probe,
        args.fastwam_data_config,
        args.fastwam_text_embedding_cache_dir,
        args.fastwam_pretrained_norm_stats,
        args.dino_teacher_ckpt,
        args.dino_teacher_config,
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
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)


def _load_dino_probe(path: Path) -> DinoPCAProbe:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    return DinoPCAProbe(
        state["mean"],
        state["basis"],
        state["low"],
        state["high"],
        output_size=OUTPUT_SIZE,
    ).eval()


def _load_depth_probe(path: Path, device: torch.device) -> MiniDPTDepthProbe:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    probe = MiniDPTDepthProbe(**checkpoint["config"])
    probe.load_state_dict(checkpoint["state_dict"], strict=True)
    return probe.to(device).eval()


def _add_metrics(
    sums: dict[str, dict[str, float]],
    name: str,
    values: dict[str, float | int],
) -> None:
    count = int(values["num_frames"])
    bucket = sums.setdefault(name, {"num_frames": 0.0})
    bucket["num_frames"] += count
    for key, value in values.items():
        if key not in {"num_frames", "num_pixels"}:
            bucket[key] = bucket.get(key, 0.0) + float(value) * count


def _finalize_metrics(
    sums: dict[str, dict[str, float]],
) -> dict[str, dict[str, float | int]]:
    result = {}
    for name, bucket in sums.items():
        frames = int(bucket["num_frames"])
        result[name] = {
            "num_frames": frames,
            **{
                key: value / frames
                for key, value in bucket.items()
                if key != "num_frames"
            },
        }
    return result


def _dino_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float | int]:
    prediction = prediction.float()
    target = target.float()
    cosine = F.cosine_similarity(prediction, target, dim=1, eps=1e-8)
    return {
        "num_frames": prediction.shape[0],
        "mse": float(F.mse_loss(prediction, target)),
        "cosine": float(cosine.mean()),
    }


def _sample_metrics(
    *,
    teacher_dino_current: torch.Tensor,
    planner_dino_current: torch.Tensor,
    teacher_dino_future: torch.Tensor,
    planner_dino_future: torch.Tensor,
    planner_log_current: torch.Tensor,
    planner_log_future: torch.Tensor,
    target_log_current: torch.Tensor,
    target_log_future: torch.Tensor,
    sample_index: int,
) -> dict[str, float]:
    index = slice(sample_index, sample_index + 1)
    current_depth = compute_log_depth_metrics(
        planner_log_current[index],
        target_log_current[index],
    )
    future_depth = compute_log_depth_metrics(
        planner_log_future[index],
        target_log_future[index],
    )
    return {
        "dino_current_mse": float(
            F.mse_loss(
                planner_dino_current[index].float(),
                teacher_dino_current[index].float(),
            )
        ),
        "dino_future_mse": float(
            F.mse_loss(
                planner_dino_future[index].float(),
                teacher_dino_future[index].float(),
            )
        ),
        "depth_current_abs_rel": float(current_depth["abs_rel"]),
        "depth_future_abs_rel": float(future_depth["abs_rel"]),
    }


def _build_panels(
    *,
    item: dict[str, Any],
    sample_index: int,
    dino_maps: dict[str, dict[str, torch.Tensor]],
    depth_display: dict[str, dict[str, torch.Tensor]],
    moge_display: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Image.Image]:
    panels: dict[str, Image.Image] = {}
    observations = {
        "current": split_rgb_cameras_224(item["current_image"]),
        "future": split_rgb_cameras_224(item["keyframe_images"][0]),
    }
    for camera in CAMERAS:
        for time in TIMEPOINTS:
            panels[f"observation_{camera}_{time}"] = observations[time][camera]
            for source in DINO_SOURCES:
                panels[f"dino_{source}_{camera}_{time}"] = _as_pil_rgb(
                    dino_maps[f"{source}_{time}"][camera][sample_index]
                )
            for source in ("teacher", "planner"):
                panels[f"depth_{source}_{camera}_{time}"] = _turbo_image(
                    depth_display[f"{source}_{time}"][camera][sample_index]
                )
            panels[f"depth_moge_{camera}_{time}"] = _turbo_image(
                moge_display[time][camera][sample_index]
            )
    return panels


def _write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    rows = []
    for suite, metrics in (("overall", summary["overall"]), *summary["suites"].items()):
        for name, values in metrics.items():
            rows.append({"suite": suite, "case": name, **values})
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    _preflight(args)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    helpers = _import_pipeline_helpers()
    datasets: dict[str, Any] = {}
    eval_indices: dict[str, list[int]] = {}
    for dataset_dir in args.fastwam_dataset_dir:
        suite = _suite_name(dataset_dir)
        if suite in datasets:
            raise ValueError(f"duplicate suite name: {suite}")
        dataset = helpers["build_suite_dataset"](args, dataset_dir)
        _training, evaluation = helpers["select_disjoint_indices"](
            len(dataset),
            args.train_windows_per_suite,
            args.eval_windows_per_suite,
        )
        datasets[suite] = dataset
        eval_indices[suite] = evaluation

    dino_probe = _load_dino_probe(args.dino_probe)
    depth_probe = _load_depth_probe(args.minidpt_probe, device)
    dino_teacher = _build_dino_teacher(args, device)
    depth_teacher = helpers["build_depth_teacher"](args, device)
    wrapper, processor, metadata, runtime_device, runtime_dtype = helpers["load_runtime"](args)

    overall_sums: dict[str, dict[str, float]] = {}
    suite_results: dict[str, Any] = {}
    sample_dirs: list[str] = []
    visual_index = 0
    total_evaluated = 0
    for suite, dataset in datasets.items():
        suite_sums: dict[str, dict[str, float]] = {}
        visualized = 0
        indices = eval_indices[suite]
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
                f"{source}_{time}": project_dino_cameras_224(
                    dino_probe,
                    (teacher if source == "teacher" else planner)[f"{time}_dino"].cpu(),
                )
                for source in DINO_SOURCES
                for time in TIMEPOINTS
            }
            log_depth: dict[str, torch.Tensor] = {}
            target_log: dict[str, torch.Tensor] = {}
            depth_display: dict[str, dict[str, torch.Tensor]] = {}
            moge_display: dict[str, dict[str, torch.Tensor]] = {}
            for time in TIMEPOINTS:
                teacher_log = depth_probe(
                    teacher[f"{time}_depth"].to(device=device, dtype=torch.float32)
                ).float()
                planner_log = depth_probe(
                    planner[f"{time}_depth"].to(device=device, dtype=torch.float32)
                ).float()
                log_depth[f"teacher_{time}"] = teacher_log
                log_depth[f"planner_{time}"] = planner_log
                target_log[time] = dense_log_depth_target(
                    teacher[f"{time}_dense_depth"],
                    output_size=OUTPUT_SIZE,
                ).to(device)
                normalized = normalize_log_depth_pair(teacher_log, planner_log)
                for source in ("teacher", "planner"):
                    depth_display[f"{source}_{time}"] = unsquish_and_split(
                        normalized[source]
                    )
                moge_display[time] = unsquish_and_split(
                    normalize_log_depth_reference(target_log[time])
                )

            log_cameras = {
                name: unsquish_and_split(value)
                for name, value in log_depth.items()
            }
            target_cameras = {
                time: unsquish_and_split(value)
                for time, value in target_log.items()
            }
            for camera in CAMERAS:
                for time in TIMEPOINTS:
                    dino_values = _dino_metrics(
                        dino_maps[f"planner_{time}"][camera],
                        dino_maps[f"teacher_{time}"][camera],
                    )
                    _add_metrics(suite_sums, f"dino_planner_{camera}_{time}", dino_values)
                    _add_metrics(overall_sums, f"dino_planner_{camera}_{time}", dino_values)
                    for source in ("teacher", "planner"):
                        depth_values = compute_log_depth_metrics(
                            log_cameras[f"{source}_{time}"][camera],
                            target_cameras[time][camera],
                        )
                        name = f"depth_{source}_{camera}_{time}"
                        _add_metrics(suite_sums, name, depth_values)
                        _add_metrics(overall_sums, name, depth_values)

            for local_index, (dataset_index, item) in enumerate(
                zip(selected, items, strict=True)
            ):
                if visualized >= args.visualizations_per_suite:
                    break
                sample_dir = args.output_dir / "samples" / suite / f"index{dataset_index:09d}"
                sample_metrics = _sample_metrics(
                    teacher_dino_current=teacher["current_dino"],
                    planner_dino_current=planner["current_dino"],
                    teacher_dino_future=teacher["future_dino"],
                    planner_dino_future=planner["future_dino"],
                    planner_log_current=log_depth["planner_current"],
                    planner_log_future=log_depth["planner_future"],
                    target_log_current=target_log["current"],
                    target_log_future=target_log["future"],
                    sample_index=local_index,
                )
                panels = _build_panels(
                    item=item,
                    sample_index=local_index,
                    dino_maps=dino_maps,
                    depth_display=depth_display,
                    moge_display=moge_display,
                )
                save_reference_style_sample(
                    output_dir=sample_dir,
                    sample_index=visual_index,
                    instruction=str(item["prompt"]),
                    panels=panels,
                    metrics=sample_metrics,
                )
                sample_dirs.append(str(sample_dir.relative_to(args.output_dir)))
                visualized += 1
                visual_index += 1
            total_evaluated += len(items)
            print(
                json.dumps(
                    {
                        "phase": "minidpt_eval",
                        "suite": suite,
                        "evaluated": start + len(items),
                        "suite_total": len(indices),
                        "total_evaluated": total_evaluated,
                    }
                ),
                flush=True,
            )
        suite_results[suite] = _finalize_metrics(suite_sums)

    summary = {
        "protocol": {
            "suites": list(datasets),
            "evaluated_windows": sum(len(value) for value in eval_indices.values()),
            "future_offset_frames": 8,
            "planner_checkpoint": str(args.checkpoint_dir),
            "minidpt_probe": str(args.minidpt_probe),
            "depth_decode": "feature-only MiniDPT 256x1024 -> dense 224x224 log depth",
            "camera_geometry": "decode square composite, restore 224x448, then split main|wrist",
            "depth_colors": "teacher/planner joint per-sample per-time disparity range",
            "rgb_guidance": False,
        },
        "overall": _finalize_metrics(overall_sums),
        "suites": suite_results,
        "sample_dirs": sample_dirs,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_summary_csv(summary, args.output_dir / "summary.csv")
    print(json.dumps({"phase": "complete", **summary["protocol"]}), flush=True)


if __name__ == "__main__":
    main()
