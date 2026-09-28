"""E2 diagnostic: does the frozen Baton planner beat trivial feature baselines?

For LIBERO windows it compares, per camera and keyframe, the teacher-space MSE
of (a) the planner prediction, (b) copying the current frame's SigLIP2 grid,
and (c) the evaluation-set mean grid. It also measures how much the prediction
moves when the instruction is swapped for another task of the same suite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import torch

from qwen35_baton.data import BatonLiberoDataset
from qwen35_baton.provider import FrozenBatonPlanner
from qwen35_baton.teacher import FrozenSiglip2Teacher


_CAMERAS = ("main", "wrist")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qwen-path", required=True)
    parser.add_argument("--siglip2-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stat-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-episodes", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _per_frame_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[B,2,4,256,D] x2 -> summed squared error per (camera, keyframe), [2,4]."""

    return (a.float() - b.float()).square().mean(dim=(3, 4)).sum(dim=0)


def main() -> int:
    args = _parser().parse_args()
    from ge_act.data.libero_fastwam_hdf5_dataset import LiberoFastWAMHDF5Dataset

    device = torch.device("cuda")
    planner = FrozenBatonPlanner.from_checkpoint(
        args.checkpoint,
        qwen_model_path=args.qwen_path,
        qwen_tokenizer_path=args.qwen_path,
        qwen_processor_path=args.qwen_path,
        siglip2_model_path=args.siglip2_path,
        device=device,
    )
    teacher = FrozenSiglip2Teacher(args.siglip2_path, device=device)
    base = LiberoFastWAMHDF5Dataset(
        args.manifest, args.stat_file, train_dataset=False
    )
    dataset = BatonLiberoDataset(base, seed=args.seed)

    count = min(args.num_episodes, len(dataset))
    stride = max(len(dataset) // count, 1)
    indices = list(range(0, len(dataset), stride))[:count]
    by_suite: dict[str, list[str]] = {}
    for record in dataset.records:
        by_suite.setdefault(record.domain, [])
        if record.caption not in by_suite[record.domain]:
            by_suite[record.domain].append(record.caption)
    rng = random.Random(args.seed)

    sums = {
        name: torch.zeros(2, 4, dtype=torch.float64)
        for name in ("planner", "copy", "swap_shift", "planner_vs_swap_target")
    }
    feature_sum = torch.zeros(2, 4, 256, 1024, dtype=torch.float64)
    feature_sq_sum = torch.zeros(2, 4, 256, 1024, dtype=torch.float64)
    seen = 0
    swapped = 0
    started = time.time()
    for start in range(0, len(indices), args.batch_size):
        samples = [dataset[index] for index in indices[start : start + args.batch_size]]
        current = torch.stack([sample["current_images"] for sample in samples])
        future = torch.stack([sample["future_images"] for sample in samples])
        instructions = [sample["instruction"] for sample in samples]
        current_features = teacher.encode_current(current)
        future_features = teacher.encode_future(future)
        prediction = planner.predict(current, instructions).tokens

        copy = current_features.unsqueeze(2).expand_as(future_features)
        sums["planner"] += _per_frame_mse(prediction, future_features).double().cpu()
        sums["copy"] += _per_frame_mse(copy, future_features).double().cpu()
        flat = future_features.double().cpu().sum(dim=0)
        feature_sum += flat
        feature_sq_sum += future_features.double().cpu().square().sum(dim=0)

        alternatives = []
        keep = []
        for row, sample in enumerate(samples):
            options = [
                caption
                for caption in by_suite[sample["suite"]]
                if caption != sample["instruction"]
            ]
            if options:
                alternatives.append(rng.choice(options))
                keep.append(row)
        if keep:
            swapped_prediction = planner.predict(current[keep], alternatives).tokens
            sums["swap_shift"] += _per_frame_mse(
                swapped_prediction, prediction[keep]
            ).double().cpu()
            sums["planner_vs_swap_target"] += _per_frame_mse(
                swapped_prediction, future_features[keep]
            ).double().cpu()
            swapped += len(keep)
        seen += len(samples)
        print(
            json.dumps({"seen": seen, "elapsed_s": round(time.time() - started, 1)}),
            flush=True,
        )

    mean = feature_sum / seen
    variance = (feature_sq_sum / seen - mean.square()).mean(dim=(2, 3))
    result: dict[str, object] = {
        "checkpoint": args.checkpoint,
        "num_samples": seen,
        "num_swapped": swapped,
        "note": "LIBERO training used all episodes; these windows are in-distribution.",
        "per_camera": {},
    }
    for camera_index, camera in enumerate(_CAMERAS):
        rows = []
        for frame in range(4):
            rows.append(
                {
                    "keyframe": frame,
                    "planner_mse": float(sums["planner"][camera_index, frame] / seen),
                    "copy_current_mse": float(sums["copy"][camera_index, frame] / seen),
                    "dataset_mean_mse": float(variance[camera_index, frame]),
                    "swap_instruction_shift_mse": float(
                        sums["swap_shift"][camera_index, frame] / max(swapped, 1)
                    ),
                    "swapped_planner_vs_true_future_mse": float(
                        sums["planner_vs_swap_target"][camera_index, frame]
                        / max(swapped, 1)
                    ),
                }
            )
        result["per_camera"][camera] = rows
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
