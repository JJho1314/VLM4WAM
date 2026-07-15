#!/usr/bin/env python3
"""Render saved planner probes as aligned main/wrist 224px maps."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_dino_depth_probe_visualization import (  # noqa: E402
    DinoPCAProbe,
    OUTPUT_SIZE,
    TOKEN_GRID_SIZE,
    _build_dino_teacher,
    _depth_image_224,
    _dino_image_224,
    _empty_dino_sums,
    _finalize_dino_sums,
    _import_pipeline_helpers,
    _planner_predictions,
    _suite_name,
    _teacher_outputs_for_items,
    _update_dino_sums,
    validate_token_features,
)
from train_depth_probe_visualization import LinearDepthProbe  # noqa: E402


CAMERAS = ("main", "wrist")
TIMEPOINTS = ("current", "future")
DINO_SOURCES = ("teacher", "planner")
DEPTH_SOURCES = ("moge", "teacher_probe", "planner_probe")
OBSERVATION_NAMES = tuple(
    f"observation_{camera}_{time}.png"
    for camera in CAMERAS
    for time in TIMEPOINTS
)
DINO_NAMES = tuple(
    f"dino_{source}_{camera}_{time}_224"
    for source in DINO_SOURCES
    for camera in CAMERAS
    for time in TIMEPOINTS
)
DEPTH_NAMES = tuple(
    f"depth_{source}_{camera}_{time}_224"
    for source in DEPTH_SOURCES
    for camera in CAMERAS
    for time in TIMEPOINTS
)


def _split_width(
    value: torch.Tensor,
    *,
    name: str,
) -> dict[str, torch.Tensor]:
    if value.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")
    if value.shape[-1] % 2:
        raise ValueError(f"{name} width must be even, got {value.shape[-1]}")
    midpoint = value.shape[-1] // 2
    return {
        "main": value[..., :midpoint],
        "wrist": value[..., midpoint:],
    }


def _as_bhw(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    elif value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(
            f"{name} must be [B,H,W] or [B,1,H,W], got {tuple(value.shape)}"
        )
    return value


def split_rgb_cameras_224(value: Any) -> dict[str, Image.Image]:
    if isinstance(value, Image.Image):
        tensor = torch.from_numpy(np.asarray(value.convert("RGB")).copy())
    else:
        tensor = torch.as_tensor(value).detach().cpu()
    if tuple(tensor.shape) != (OUTPUT_SIZE, 2 * OUTPUT_SIZE, 3):
        raise ValueError(
            "RGB composite must be "
            f"[{OUTPUT_SIZE},{2 * OUTPUT_SIZE},3], got {tuple(tensor.shape)}"
        )
    if tensor.dtype != torch.uint8:
        maximum = float(tensor.max()) if tensor.numel() else 0.0
        tensor = tensor.float()
        if maximum <= 1.5:
            tensor = tensor * 255.0
        tensor = tensor.round().clamp(0, 255).to(torch.uint8)
    halves = _split_width(
        tensor.permute(2, 0, 1),
        name="RGB composite",
    )
    return {
        camera: Image.fromarray(half.permute(1, 2, 0).contiguous().numpy())
        for camera, half in halves.items()
    }


def project_dino_cameras_224(
    probe: DinoPCAProbe,
    features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_token_features(
        features,
        feature_dim=probe.mean.numel(),
        name="DINO features",
    )
    features = features.to(device=probe.mean.device, dtype=torch.float32)
    projected = (features - probe.mean) @ probe.basis
    projected = (
        (projected - probe.low) / (probe.high - probe.low).clamp_min(1e-6)
    ).clamp(0.0, 1.0)
    grid = projected.reshape(
        -1,
        TOKEN_GRID_SIZE,
        TOKEN_GRID_SIZE,
        3,
    ).permute(0, 3, 1, 2)
    return {
        camera: F.interpolate(
            half,
            size=(OUTPUT_SIZE, OUTPUT_SIZE),
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)
        for camera, half in _split_width(grid, name="DINO token grid").items()
    }


def resize_depth_target_cameras_224(
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    target = _as_bhw(target, name="target_depth").to(torch.float32)
    target = torch.nan_to_num(
        target,
        nan=1e-6,
        posinf=1e6,
        neginf=1e-6,
    ).clamp_min(1e-6)
    return {
        camera: F.interpolate(
            half.unsqueeze(1),
            size=(OUTPUT_SIZE, OUTPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        for camera, half in _split_width(
            target,
            name="dense Depth target",
        ).items()
    }


def decode_depth_cameras_224(
    relative: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    relative = _as_bhw(
        relative,
        name="relative_log_prediction",
    ).to(torch.float32)
    if not bool(torch.isfinite(relative).all()):
        raise ValueError("relative_log_prediction contains non-finite values")
    target_cameras = resize_depth_target_cameras_224(target)
    result = {}
    for camera, half in _split_width(
        relative,
        name="Depth token grid",
    ).items():
        prediction = F.interpolate(
            half.unsqueeze(1),
            size=(OUTPUT_SIZE, OUTPUT_SIZE),
            mode="bicubic",
            align_corners=False,
        )[:, 0]
        prediction = prediction - prediction.mean(
            dim=(-2, -1),
            keepdim=True,
        )
        truth = target_cameras[camera].to(prediction.device)
        shift = (
            (truth.log() - prediction)
            .flatten(1)
            .median(dim=1)
            .values[:, None, None]
        )
        decoded = (prediction + shift).exp()
        if not bool(torch.isfinite(decoded).all()):
            raise ValueError(f"decoded {camera} depth contains non-finite values")
        result[camera] = decoded
    return result


def save_dual_camera_sample(
    *,
    output_dir: Path,
    current_rgb: Any,
    future_rgb: Any,
    instruction: str,
    dino_maps: dict[str, torch.Tensor],
    depth_maps: dict[str, torch.Tensor],
) -> list[Path]:
    output_dir = Path(output_dir)
    if set(dino_maps) != set(DINO_NAMES):
        raise ValueError(f"DINO output names must be {DINO_NAMES}")
    if set(depth_maps) != set(DEPTH_NAMES):
        raise ValueError(f"Depth output names must be {DEPTH_NAMES}")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    observations = {
        "current": split_rgb_cameras_224(current_rgb),
        "future": split_rgb_cameras_224(future_rgb),
    }
    for camera in CAMERAS:
        for time in TIMEPOINTS:
            path = output_dir / f"observation_{camera}_{time}.png"
            observations[time][camera].save(path)
            paths.append(path)

    instruction_path = output_dir / "instruction.txt"
    instruction_path.write_text(
        instruction.strip() + "\n",
        encoding="utf-8",
    )
    paths.append(instruction_path)

    for name in DINO_NAMES:
        path = output_dir / f"{name}.png"
        _dino_image_224(dino_maps[name]).save(path)
        paths.append(path)

    ranges: dict[str, dict[str, dict[str, float]]] = {
        camera: {} for camera in CAMERAS
    }
    file_ranges: dict[str, dict[str, str | float]] = {}
    for camera in CAMERAS:
        for time in TIMEPOINTS:
            target_name = f"depth_moge_{camera}_{time}_224"
            target = torch.as_tensor(depth_maps[target_name]).detach().to(
                device="cpu",
                dtype=torch.float32,
            )
            if target.shape != (OUTPUT_SIZE, OUTPUT_SIZE):
                raise ValueError(
                    f"{target_name} must be [{OUTPUT_SIZE},{OUTPUT_SIZE}], "
                    f"got {tuple(target.shape)}"
                )
            if not bool(torch.isfinite(target).all()):
                raise ValueError(f"{target_name} contains non-finite values")
            quantiles = torch.quantile(target, torch.tensor([0.02, 0.98]))
            low, high = float(quantiles[0]), float(quantiles[1])
            ranges[camera][time] = {"low": low, "high": high}
            range_key = f"{camera}/{time}"
            for source in DEPTH_SOURCES:
                name = f"depth_{source}_{camera}_{time}_224"
                file_ranges[f"{name}.png"] = {
                    "range_key": range_key,
                    "low": low,
                    "high": high,
                }

    for name in DEPTH_NAMES:
        parts = name.split("_")
        camera = next(item for item in CAMERAS if item in parts)
        time = next(item for item in TIMEPOINTS if item in parts)
        low = ranges[camera][time]["low"]
        high = ranges[camera][time]["high"]
        path = output_dir / f"{name}.png"
        _depth_image_224(depth_maps[name], low=low, high=high).save(path)
        paths.append(path)

    ranges_path = output_dir / "depth_color_ranges.json"
    ranges_path.write_text(
        json.dumps(
            {"ranges": ranges, "files": file_ranges},
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    paths.append(ranges_path)

    expected_names = {
        *OBSERVATION_NAMES,
        *(f"{name}.png" for name in DINO_NAMES),
        *(f"{name}.png" for name in DEPTH_NAMES),
        "instruction.txt",
        "depth_color_ranges.json",
    }
    if {path.name for path in paths} != expected_names:
        raise RuntimeError("dual-camera sample output layout is incomplete")
    return paths


def load_saved_probes(
    probe_dir: Path,
    device: torch.device,
) -> tuple[DinoPCAProbe, LinearDepthProbe]:
    probe_dir = Path(probe_dir)
    dino_path = probe_dir / "dino_pca_probe.pt"
    depth_path = probe_dir / "depth_linear_probe.pt"
    for path in (dino_path, depth_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing saved probe: {path}")

    dino_checkpoint = torch.load(
        dino_path,
        map_location="cpu",
        weights_only=True,
    )
    dino_state = dino_checkpoint.get("state_dict")
    expected_dino = {"mean", "basis", "low", "high"}
    if not isinstance(dino_state, dict) or set(dino_state) != expected_dino:
        raise ValueError(
            f"DINO probe state must contain exactly {sorted(expected_dino)}"
        )
    dino_probe = DinoPCAProbe(
        dino_state["mean"],
        dino_state["basis"],
        dino_state["low"],
        dino_state["high"],
        output_size=OUTPUT_SIZE,
    ).to("cpu").eval()

    depth_checkpoint = torch.load(
        depth_path,
        map_location="cpu",
        weights_only=True,
    )
    feature_dim = int(depth_checkpoint.get("feature_dim", 0))
    grid_size = int(depth_checkpoint.get("grid_size", 0))
    if feature_dim <= 0 or grid_size != TOKEN_GRID_SIZE:
        raise ValueError(
            "Depth probe metadata must define a positive feature_dim and "
            f"grid_size={TOKEN_GRID_SIZE}, got {feature_dim} and {grid_size}"
        )
    depth_probe = LinearDepthProbe(
        feature_dim=feature_dim,
        grid_size=grid_size,
    )
    depth_probe.load_state_dict(depth_checkpoint["state_dict"], strict=True)
    return dino_probe, depth_probe.to(device).eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse saved DINO/Depth probes and render aligned main/wrist "
            "camera outputs at 224x224."
        )
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
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
    parser.add_argument("--planner-batch-size", type=int, default=8)
    parser.add_argument("--visualizations-per-suite", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
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
    paths = (
        args.checkpoint_dir,
        args.probe_dir,
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
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)


@torch.inference_mode()
def _decode_depth_feature_cameras(
    *,
    probe: LinearDepthProbe,
    features: torch.Tensor,
    target_depth: torch.Tensor,
) -> dict[str, torch.Tensor]:
    device = next(probe.parameters()).device
    relative = probe(features.to(device=device, dtype=torch.float32))
    return decode_depth_cameras_224(relative, target_depth.to(device))


def _new_metric_sums(helpers: dict[str, Any]) -> dict[str, Any]:
    dino_cases = ("planner_current", "planner_future")
    depth_cases = (
        "teacher_probe_current",
        "planner_probe_current",
        "teacher_probe_future",
        "planner_probe_future",
        "persistence_future",
    )
    return {
        camera: {
            "dino": {case: _empty_dino_sums() for case in dino_cases},
            "depth": {
                case: helpers["empty_depth_sums"]()
                for case in depth_cases
            },
        }
        for camera in CAMERAS
    }


def _finalize_metric_sums(
    sums: dict[str, Any],
    helpers: dict[str, Any],
) -> dict[str, Any]:
    return {
        camera: {
            "dino": {
                case: _finalize_dino_sums(values)
                for case, values in sums[camera]["dino"].items()
            },
            "depth": {
                case: helpers["finalize_depth_sums"](values)
                for case, values in sums[camera]["depth"].items()
            },
        }
        for camera in CAMERAS
    }


def _update_metrics(
    *,
    sums: dict[str, Any],
    dino_maps: dict[str, dict[str, torch.Tensor]],
    depth_maps: dict[str, dict[str, torch.Tensor]],
    depth_targets: dict[str, dict[str, torch.Tensor]],
    helpers: dict[str, Any],
) -> None:
    for camera in CAMERAS:
        for time in TIMEPOINTS:
            dino_case = f"planner_{time}"
            _update_dino_sums(
                sums[camera]["dino"][dino_case],
                dino_maps[f"planner_{time}"][camera],
                dino_maps[f"teacher_{time}"][camera],
            )
        for case, time in (
            ("teacher_probe_current", "current"),
            ("planner_probe_current", "current"),
            ("teacher_probe_future", "future"),
            ("planner_probe_future", "future"),
            ("persistence_future", "future"),
        ):
            helpers["update_depth_sums"](
                sums[camera]["depth"][case],
                depth_maps[case][camera],
                depth_targets[time][camera],
            )


def _sample_dino_maps(
    maps: dict[str, dict[str, torch.Tensor]],
    index: int,
) -> dict[str, torch.Tensor]:
    return {
        f"dino_{source}_{camera}_{time}_224": maps[f"{source}_{time}"][camera][index]
        for source in DINO_SOURCES
        for camera in CAMERAS
        for time in TIMEPOINTS
    }


def _sample_depth_maps(
    *,
    targets: dict[str, dict[str, torch.Tensor]],
    decoded: dict[str, dict[str, torch.Tensor]],
    index: int,
) -> dict[str, torch.Tensor]:
    result = {}
    for camera in CAMERAS:
        for time in TIMEPOINTS:
            result[f"depth_moge_{camera}_{time}_224"] = targets[time][camera][index]
            result[f"depth_teacher_probe_{camera}_{time}_224"] = decoded[
                f"teacher_probe_{time}"
            ][camera][index]
            result[f"depth_planner_probe_{camera}_{time}_224"] = decoded[
                f"planner_probe_{time}"
            ][camera][index]
    return result


def _save_summary_csv(summary: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    scopes = (("overall", summary["overall"]), *summary["suites"].items())
    for scope, camera_metrics in scopes:
        for camera, modalities in camera_metrics.items():
            for modality, cases in modalities.items():
                for case, metrics in cases.items():
                    rows.append(
                        {
                            "scope": scope,
                            "camera": camera,
                            "modality": modality,
                            "case": case,
                            **metrics,
                        }
                    )
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
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

    dino_probe, depth_probe = load_saved_probes(args.probe_dir, device)
    dino_teacher = _build_dino_teacher(args, device)
    depth_teacher = helpers["build_depth_teacher"](args, device)
    wrapper, processor, metadata, runtime_device, runtime_dtype = helpers["load_runtime"](
        args
    )

    overall_sums = _new_metric_sums(helpers)
    suite_results: dict[str, Any] = {}
    sample_dirs: list[str] = []
    total_evaluated = 0
    for suite, dataset in datasets.items():
        indices = eval_indices[suite]
        suite_sums = _new_metric_sums(helpers)
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
                f"{source}_{time}": project_dino_cameras_224(
                    dino_probe,
                    (teacher if source == "teacher" else planner)[f"{time}_dino"],
                )
                for source in DINO_SOURCES
                for time in TIMEPOINTS
            }
            dense = {
                time: teacher[f"{time}_dense_depth"]
                for time in TIMEPOINTS
            }
            depth_targets = {
                time: resize_depth_target_cameras_224(dense[time])
                for time in TIMEPOINTS
            }
            decoded_depth = {
                "teacher_probe_current": _decode_depth_feature_cameras(
                    probe=depth_probe,
                    features=teacher["current_depth"],
                    target_depth=dense["current"],
                ),
                "planner_probe_current": _decode_depth_feature_cameras(
                    probe=depth_probe,
                    features=planner["current_depth"],
                    target_depth=dense["current"],
                ),
                "teacher_probe_future": _decode_depth_feature_cameras(
                    probe=depth_probe,
                    features=teacher["future_depth"],
                    target_depth=dense["future"],
                ),
                "planner_probe_future": _decode_depth_feature_cameras(
                    probe=depth_probe,
                    features=planner["future_depth"],
                    target_depth=dense["future"],
                ),
                "persistence_future": _decode_depth_feature_cameras(
                    probe=depth_probe,
                    features=teacher["current_depth"],
                    target_depth=dense["future"],
                ),
            }
            for sums in (suite_sums, overall_sums):
                _update_metrics(
                    sums=sums,
                    dino_maps=dino_maps,
                    depth_maps=decoded_depth,
                    depth_targets=depth_targets,
                    helpers=helpers,
                )

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
                save_dual_camera_sample(
                    output_dir=sample_dir,
                    current_rgb=item["current_image"],
                    future_rgb=item["keyframe_images"][0],
                    instruction=str(item["prompt"]),
                    dino_maps=_sample_dino_maps(dino_maps, local_index),
                    depth_maps=_sample_depth_maps(
                        targets=depth_targets,
                        decoded=decoded_depth,
                        index=local_index,
                    ),
                )
                sample_dirs.append(str(sample_dir.relative_to(args.output_dir)))
                visualized += 1
            total_evaluated += len(items)
            print(
                json.dumps(
                    {
                        "phase": "dual_camera_eval",
                        "suite": suite,
                        "evaluated": start + len(items),
                        "suite_total": len(indices),
                        "total_evaluated": total_evaluated,
                    }
                ),
                flush=True,
            )
        suite_results[suite] = _finalize_metric_sums(suite_sums, helpers)

    summary = {
        "protocol": {
            "suites": list(datasets),
            "train_windows_per_suite_for_disjoint_split": args.train_windows_per_suite,
            "eval_windows_per_suite": args.eval_windows_per_suite,
            "evaluated_windows": sum(len(item) for item in eval_indices.values()),
            "future_offset_frames": 8,
            "input_camera_layout": "main|wrist horizontal 224x448",
            "token_geometry": "16x16 composite split before interpolation into two 16x8 camera grids",
            "output_size": [OUTPUT_SIZE, OUTPUT_SIZE],
            "probe_source": str(args.probe_dir),
            "probe_retrained": False,
            "depth_alignment": "per-sample per-time per-camera median log-scale",
            "depth_color_range": "per-sample per-time per-camera MoGe q02/q98 shared by all three depth maps",
        },
        "evaluation_warning": (
            "The planner and teachers were trained/evaluated on the 224x448 camera "
            "composite. Each visualization is an honest 16x8 half of that 256-token "
            "composite, not a separately encoded 16x16 camera representation."
        ),
        "overall": _finalize_metric_sums(overall_sums, helpers),
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
