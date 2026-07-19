#!/usr/bin/env python3
"""Visualize dual-camera K4 SigLIP2 + DA3 planner predictions.

WSA checkpoints are decoded with one four-layer probe.  Legacy last-layer
checkpoints retain the existing single-layer MiniDPT probe path.  One PNG is
written per sample, camera, and future keyframe.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
PLANNER_DIR = HERE.parent
REPO_ROOT = PLANNER_DIR.parent
for _path in (REPO_ROOT, PLANNER_DIR, HERE, PLANNER_DIR / "lingbot_dino_4b"):
    _path_string = str(_path)
    if _path_string not in sys.path:
        sys.path.insert(0, _path_string)


def probe_kind_for_metadata(metadata: dict[str, Any]) -> str:
    strategy = metadata.get("da3_align_strategy", "last_layer")
    if strategy == "wsa_multilayer":
        return "wsa"
    if strategy in (None, "last_layer"):
        return "last_layer"
    raise ValueError(f"unsupported DA3 alignment strategy: {strategy!r}")


def depth_features_for_probe(
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    camera_index: int,
    token_slice: slice,
    probe_kind: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one camera/keyframe and return the probe's expected layout."""
    camera_index = int(camera_index)
    if probe_kind == "wsa":
        if target.ndim != 5 or prediction.ndim != 5:
            raise ValueError(
                "WSA depth target/prediction must be [B,V,L,N,D] and "
                f"[B,V,N,L,D], got {tuple(target.shape)} and "
                f"{tuple(prediction.shape)}"
            )
        target_features = target[:, camera_index, :, token_slice, :]
        prediction_features = prediction[
            :, camera_index, token_slice, :, :
        ].transpose(1, 2).contiguous()
    elif probe_kind == "last_layer":
        if target.ndim != 4 or prediction.ndim != 4:
            raise ValueError(
                "last-layer depth target/prediction must be [B,V,N,D], got "
                f"{tuple(target.shape)} and {tuple(prediction.shape)}"
            )
        target_features = target[:, camera_index, token_slice, :]
        prediction_features = prediction[:, camera_index, token_slice, :]
    else:
        raise ValueError(f"unsupported depth probe kind: {probe_kind!r}")
    if target_features.shape != prediction_features.shape:
        raise ValueError(
            "depth target/prediction probe geometry differs: "
            f"{tuple(target_features.shape)} vs "
            f"{tuple(prediction_features.shape)}"
        )
    return target_features, prediction_features


def load_depth_probe(
    checkpoint_path: Path,
    *,
    probe_kind: str,
    device: torch.device | str,
) -> torch.nn.Module:
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if probe_kind == "wsa":
        try:
            from .wsa_depth_probe import WSAMultiLayerDPTProbe
        except ImportError:
            from wsa_depth_probe import WSAMultiLayerDPTProbe

        probe = WSAMultiLayerDPTProbe.from_config(payload["config"])
    elif probe_kind == "last_layer":
        from train_feature_probes import MiniDPTProbe

        probe = MiniDPTProbe(**payload["config"])
    else:
        raise ValueError(f"unsupported depth probe kind: {probe_kind!r}")
    probe.load_state_dict(payload["state_dict"], strict=True)
    probe.to(device).eval()
    probe.requires_grad_(False)
    return probe


@torch.no_grad()
def decode_depth_pair(
    probe: torch.nn.Module,
    target_features: torch.Tensor,
    prediction_features: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    device = next(probe.parameters()).device
    target_log_depth = probe(target_features.to(device).float())[:, 0]
    prediction_log_depth = probe(prediction_features.to(device).float())[:, 0]
    target_disparity = torch.exp(-target_log_depth).clamp_max(1e3)
    prediction_disparity = torch.exp(-prediction_log_depth).clamp_max(1e3)
    pair = torch.stack((target_disparity, prediction_disparity), dim=1)
    flattened = pair.flatten(1)
    low = flattened.min(dim=1).values.reshape(-1, 1, 1, 1)
    high = flattened.max(dim=1).values.reshape(-1, 1, 1, 1)
    pair = ((pair - low) / (high - low + 1e-6)).clamp(0, 1)

    from matplotlib import colormaps

    turbo = colormaps["turbo"]
    target_rgb = turbo(pair[0, 0].cpu().numpy())[..., :3]
    prediction_rgb = turbo(pair[0, 1].cpu().numpy())[..., :3]
    return target_rgb, prediction_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--ge-act-data-config",
        type=Path,
        default=REPO_ROOT
        / "ge_act/configs/ltx_model/libero/planner_data_libero_fastwam_ola.yaml",
    )
    parser.add_argument(
        "--siglip2-model-dir",
        type=Path,
        default=Path(os.environ.get("SIGLIP2_MODEL_DIR", "")),
    )
    parser.add_argument(
        "--da3-ckpt-dir",
        type=Path,
        default=Path(os.environ.get("DA3_CKPT_DIR", "")),
    )
    parser.add_argument(
        "--da3-code-root",
        type=Path,
        default=Path(os.environ.get("DA3_CODE_ROOT", "")),
    )
    parser.add_argument(
        "--wsa-probe",
        type=Path,
        default=Path("/data/users/junjie/probes_2b/da3_depth_wsa_probe.pt"),
    )
    parser.add_argument(
        "--legacy-probe",
        type=Path,
        default=Path("/data/users/junjie/probes_2b/da3_depth_v2_probe.pt"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _required_path(path: Path, description: str) -> Path:
    if not str(path) or not path.exists():
        raise FileNotFoundError(f"missing {description}: {path}")
    return path


def _rgb(frame: torch.Tensor) -> np.ndarray:
    return ((frame.float().cpu() + 1.0) * 0.5).clamp(0, 1).numpy()


def _heatmap(values: torch.Tensor, *, cosine: bool = False) -> np.ndarray:
    from matplotlib import colormaps

    values = values.detach().float().cpu()
    if cosine:
        values = (values + 1.0) * 0.5
    else:
        low = torch.quantile(values.flatten(), 0.02)
        high = torch.quantile(values.flatten(), 0.98)
        values = (values - low) / (high - low + 1e-6)
    return np.asarray(
        colormaps["turbo"](values.clamp(0, 1).numpy())[..., :3],
        dtype=np.float32,
    )


def _depth_feature_maps(
    target_features: torch.Tensor,
    prediction_features: torch.Tensor,
    *,
    grid_size: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    cosine = F.cosine_similarity(
        prediction_features,
        target_features,
        dim=-1,
    )
    normalized_error = (
        F.layer_norm(prediction_features.float(), (prediction_features.shape[-1],))
        - F.layer_norm(target_features.float(), (target_features.shape[-1],))
    ).square().mean(dim=-1)
    while cosine.ndim > 2:
        cosine = cosine.mean(dim=1)
        normalized_error = normalized_error.mean(dim=1)
    return (
        cosine[0].reshape(grid_size, grid_size),
        normalized_error[0].reshape(grid_size, grid_size),
        float(cosine.mean()),
        float(normalized_error.mean()),
    )


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    checkpoint_dir = _required_path(args.checkpoint_dir, "planner checkpoint")
    _required_path(args.ge_act_data_config, "GE-Act data config")
    _required_path(args.siglip2_model_dir, "SigLIP2 model")
    _required_path(args.da3_ckpt_dir, "DA3 checkpoint")
    _required_path(args.da3_code_root, "DA3 code root")
    metadata = json.loads(
        (checkpoint_dir / "planner_meta.json").read_text(encoding="utf-8")
    )
    probe_kind = probe_kind_for_metadata(metadata)
    probe_path = args.wsa_probe if probe_kind == "wsa" else args.legacy_probe
    _required_path(probe_path, f"{probe_kind} depth probe")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import train_qwen3vl4b_lingbot_dino_planner as trainer
    import visualize_qwen3vl2b_siglip2_da3_split as legacy_visualizer
    from depth_anything3_target import DepthAnything3TargetEncoder, _import_da3
    from ge_act_dual_camera import DualCameraPlannerCollator
    from qwen3vl_wrapper import (
        configure_qwen3vl_processor,
        move_qwen_inputs_to_device,
    )
    from siglip2_target import Siglip2TargetEncoder
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    processor = configure_qwen3vl_processor(
        AutoProcessor.from_pretrained(
            str(checkpoint_dir / "processor"),
            local_files_only=True,
        )
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(checkpoint_dir / "qwen3vl_lora_or_model"),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(device).eval()
    if hasattr(model.config, "text_config"):
        model.config.hidden_size = model.config.text_config.hidden_size
    model.config.use_cache = False
    wrapper = trainer.PlannerWrapper.from_exported_checkpoint(
        model=model,
        checkpoint_dir=checkpoint_dir,
        metadata=metadata,
    ).to(device).eval()
    semantic_teacher = Siglip2TargetEncoder(
        model_dir=args.siglip2_model_dir,
        input_size=int(metadata.get("siglip2_input_size") or 256),
        grid_size=int(metadata["grid_size"]),
        device=device,
    )
    depth_teacher = DepthAnything3TargetEncoder(
        ckpt_dir=args.da3_ckpt_dir,
        code_root=args.da3_code_root,
        process_res=224,
        device=device,
        align_strategy=str(metadata.get("da3_align_strategy", "last_layer")),
        teacher_layers=metadata.get("da3_teacher_layers"),
        layer_weights=metadata.get("da3_layer_weights"),
    )
    depth_probe = load_depth_probe(
        probe_path,
        probe_kind=probe_kind,
        device=device,
    )
    depth_anything = _import_da3(str(args.da3_code_root))
    full_depth = depth_anything.from_pretrained(str(args.da3_ckpt_dir)).to(device).eval()
    full_depth.requires_grad_(False)

    offsets = [int(offset) for offset in metadata["future_keyframe_offsets"]]
    camera_names = [str(name) for name in metadata["camera_names"]]
    grid_size = int(metadata["grid_size"])
    tokens_per_keyframe = int(metadata["target_tokens_per_keyframe"])
    dataset = trainer.load_ge_act_dual_camera_planner_dataset(
        args.ge_act_data_config,
        future_offsets=offsets,
    )
    collator = DualCameraPlannerCollator(
        processor=processor,
        plan_sequence=list(metadata["plan_token_strings"]),
    )
    indices = random.Random(args.seed).sample(
        range(len(dataset)),
        min(args.num_samples, len(dataset)),
    )
    manifest: list[dict[str, Any]] = []

    with torch.inference_mode():
        for sample_number, sample_index in enumerate(indices):
            item = dataset[sample_index]
            batch = collator([item])
            batch.pop("stems", None)
            current = batch.pop("current_camera_images")
            future = batch.pop("future_camera_images")
            targets = trainer.encode_dual_camera_future_targets(
                current,
                future,
                appearance_encoder=semantic_teacher,
                depth_encoder=depth_teacher,
            )
            model_inputs = move_qwen_inputs_to_device(
                batch,
                device,
                model_dtype=next(model.parameters()).dtype,
            )
            predicted_semantic, predicted_depth = (
                wrapper.predict_dino_depth_plan(**model_inputs)
            )
            target_semantic = targets["semantic_plan_labels"].to(device)
            target_depth = targets["depth_plan_labels"].to(device)

            current_full_depth = []
            for camera_index in range(len(camera_names)):
                current_frame = (
                    (current[0, camera_index].permute(2, 0, 1)[None].float() + 1.0)
                    * 0.5
                ).to(device)
                current_full_depth.append(
                    legacy_visualizer.gt_turbo(
                        full_depth.model,
                        depth_teacher,
                        current_frame,
                        device,
                    )
                )

            for camera_index, camera_name in enumerate(camera_names):
                for keyframe_index, offset in enumerate(offsets):
                    token_slice = slice(
                        keyframe_index * tokens_per_keyframe,
                        (keyframe_index + 1) * tokens_per_keyframe,
                    )
                    semantic_target = target_semantic[
                        0, camera_index, token_slice
                    ]
                    semantic_prediction = predicted_semantic[
                        0, camera_index, token_slice
                    ]
                    semantic_target_rgb, semantic_prediction_rgb = (
                        legacy_visualizer.lowres_pca(
                            semantic_target.cpu(),
                            semantic_prediction.cpu(),
                            grid_size,
                        )
                    )
                    semantic_cosine_map = F.cosine_similarity(
                        semantic_prediction,
                        semantic_target,
                        dim=-1,
                    ).reshape(grid_size, grid_size)
                    semantic_mse_map = (
                        semantic_prediction - semantic_target
                    ).square().mean(dim=-1).reshape(grid_size, grid_size)
                    semantic_mse = float(
                        F.mse_loss(semantic_prediction, semantic_target)
                    )
                    semantic_cosine = float(semantic_cosine_map.mean())

                    depth_target_features, depth_prediction_features = (
                        depth_features_for_probe(
                            target_depth,
                            predicted_depth,
                            camera_index=camera_index,
                            token_slice=token_slice,
                            probe_kind=probe_kind,
                        )
                    )
                    depth_target_rgb, depth_prediction_rgb = decode_depth_pair(
                        depth_probe,
                        depth_target_features,
                        depth_prediction_features,
                    )
                    (
                        depth_cosine_map,
                        depth_error_map,
                        depth_cosine,
                        depth_lnmse,
                    ) = _depth_feature_maps(
                        depth_target_features,
                        depth_prediction_features,
                        grid_size=grid_size,
                    )
                    future_frame = (
                        (
                            future[0, camera_index, keyframe_index]
                            .permute(2, 0, 1)[None]
                            .float()
                            + 1.0
                        )
                        * 0.5
                    ).to(device)
                    future_full_depth = legacy_visualizer.gt_turbo(
                        full_depth.model,
                        depth_teacher,
                        future_frame,
                        device,
                    )
                    blank = np.ones((224, 224, 3), dtype=np.float32)
                    panels = [
                        [
                            _rgb(current[0, camera_index]),
                            _rgb(future[0, camera_index, keyframe_index]),
                            semantic_target_rgb,
                            semantic_prediction_rgb,
                            _heatmap(semantic_cosine_map, cosine=True),
                            _heatmap(semantic_mse_map),
                        ],
                        [
                            _rgb(current[0, camera_index]),
                            _rgb(future[0, camera_index, keyframe_index]),
                            depth_target_rgb,
                            depth_prediction_rgb,
                            _heatmap(depth_cosine_map, cosine=True),
                            _heatmap(depth_error_map),
                        ],
                        [
                            blank,
                            blank,
                            current_full_depth[camera_index],
                            blank,
                            future_full_depth,
                            blank,
                        ],
                    ]
                    panels = [
                        [np.asarray(panel, dtype=np.float32) for panel in row]
                        for row in panels
                    ]
                    depth_label = (
                        "WSA 4-layer" if probe_kind == "wsa" else "L23"
                    )
                    titles = [
                        [
                            "Current RGB",
                            f"Future RGB (+{offset})",
                            "SigLIP2 TARGET",
                            "SigLIP2 PRED",
                            "SigLIP2 cosine",
                            "SigLIP2 MSE map",
                        ],
                        [
                            "Current RGB",
                            f"Future RGB (+{offset})",
                            f"Depth TARGET ({depth_label})",
                            f"Depth PRED ({depth_label})",
                            "Depth cosine",
                            "Depth LN-MSE map",
                        ],
                        [
                            "",
                            "",
                            "DA3-full current GT",
                            "",
                            "DA3-full future GT",
                            "",
                        ],
                    ]
                    filename = (
                        f"sample_{sample_number:02d}_{camera_name}_"
                        f"k{keyframe_index}_off{offset}.png"
                    )
                    path = args.output_dir / filename
                    title = (
                        f"[{camera_name} cam | future +{offset} frames] "
                        f"{str(item['prompt'])[:105]}\n"
                        f"SigLIP2 mse={semantic_mse:.4f} "
                        f"cos={semantic_cosine:.3f} | "
                        f"DA3-{depth_label} cos={depth_cosine:.3f} "
                        f"lnmse={depth_lnmse:.4f}"
                    )
                    legacy_visualizer.render(path, titles, panels, title)
                    record = {
                        "sample_number": sample_number,
                        "sample_index": sample_index,
                        "instruction": str(item["prompt"]),
                        "camera": camera_name,
                        "keyframe_index": keyframe_index,
                        "future_offset": offset,
                        "probe_kind": probe_kind,
                        "siglip2_mse": semantic_mse,
                        "siglip2_cosine": semantic_cosine,
                        "depth_cosine": depth_cosine,
                        "depth_lnmse": depth_lnmse,
                        "path": filename,
                    }
                    manifest.append(record)
                    print(json.dumps(record, sort_keys=True), flush=True)

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_dir),
                "probe": str(probe_path),
                "probe_kind": probe_kind,
                "teacher_layers": metadata.get("da3_teacher_layers"),
                "seed": args.seed,
                "records": manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "done",
                "png_count": len(manifest),
                "output_dir": str(args.output_dir),
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
