#!/usr/bin/env python3
"""Export sampled RGB, SigLIP2, and DA3 images from one LIBERO episode."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


CAMERA_VIDEO_DIRS = (
    ("main", "observation.images.image"),
    ("wrist", "observation.images.wrist_image"),
)


def sample_frame_indices(num_frames: int, stride: int) -> list[int]:
    """Return stride-spaced indices that always include the episode's last frame."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    indices = list(range(0, num_frames, stride))
    final_index = num_frames - 1
    if indices[-1] != final_index:
        indices.append(final_index)
    return indices


def artifact_paths(
    output_dir: Path,
    camera: str,
    frame_index: int,
) -> dict[str, Path]:
    """Return the three independent image paths for a camera/frame pair."""
    frame_dir = output_dir / camera / f"frame_{frame_index:06d}"
    return {
        "rgb": frame_dir / "rgb.png",
        "siglip_pca": frame_dir / "siglip_pca.png",
        "da3_depth": frame_dir / "da3_depth.png",
    }


def _robust_unit_interval(values: torch.Tensor) -> torch.Tensor:
    """Normalize the final channel dimension with shared 2–98% ranges."""
    flat = values.reshape(-1, values.shape[-1])
    low = torch.quantile(flat, 0.02, dim=0)
    high = torch.quantile(flat, 0.98, dim=0)
    return ((values - low) / (high - low + 1e-6)).clamp(0, 1)


def siglip_pca_images(
    features: torch.Tensor,
    *,
    grid_size: int,
    output_size: int,
) -> np.ndarray:
    """Project all SigLIP grids through one PCA basis and display range."""
    if (
        features.ndim != 3
        or grid_size <= 0
        or features.shape[1] != grid_size * grid_size
        or features.shape[2] < 3
    ):
        raise ValueError(
            "SigLIP features must be [frames, grid_size squared, dim>=3]"
        )
    if output_size <= 0:
        raise ValueError("output_size must be positive")

    features_cpu = features.detach().float().cpu()
    flat = features_cpu.reshape(-1, features_cpu.shape[-1])
    centered = flat - flat.mean(dim=0, keepdim=True)
    _, _, vectors = torch.linalg.svd(centered, full_matrices=False)
    projected = (centered @ vectors[:3].T).reshape(
        features_cpu.shape[0],
        grid_size,
        grid_size,
        3,
    )
    projected = _robust_unit_interval(projected)
    resized = F.interpolate(
        projected.permute(0, 3, 1, 2),
        size=(output_size, output_size),
        mode="nearest",
    ).permute(0, 2, 3, 1)
    return (resized * 255.0).round().to(torch.uint8).numpy()


def da3_depth_images(depth: torch.Tensor) -> np.ndarray:
    """Colorize DA3 depth using one disparity range for the whole episode."""
    depth_cpu = depth.detach().float().cpu()
    if depth_cpu.ndim != 3 or not bool(torch.all(depth_cpu > 0)):
        raise ValueError("DA3 depth must be positive [frames,height,width]")

    disparity = depth_cpu.reciprocal()
    low = torch.quantile(disparity, 0.02)
    high = torch.quantile(disparity, 0.98)
    normalized = ((disparity - low) / (high - low + 1e-6)).clamp(0, 1)

    from matplotlib import colormaps

    rgb = colormaps["turbo"](normalized.numpy())[..., :3]
    return np.rint(rgb * 255.0).astype(np.uint8)


def load_episode_record(
    data_root: Path,
    suite: str,
    episode_index: int,
) -> dict[str, Any]:
    """Load one exact episode record from a LeRobot episodes manifest."""
    episodes_path = data_root / suite / "meta/episodes.jsonl"
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if int(record["episode_index"]) == int(episode_index):
            return record
    raise KeyError(f"episode {episode_index} not found in {episodes_path}")


def decode_episode_frames(
    video_path: Path,
    frame_indices: list[int],
) -> np.ndarray:
    """Decode requested RGB frames exactly, without out-of-range clamping."""
    if not frame_indices:
        raise ValueError("frame_indices must not be empty")

    import av

    requested = set(frame_indices)
    decoded: dict[int, np.ndarray] = {}
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in requested:
                decoded[index] = frame.to_ndarray(format="rgb24")
            if len(decoded) == len(requested):
                break
    missing = [index for index in frame_indices if index not in decoded]
    if missing:
        raise RuntimeError(
            f"{video_path} is missing requested frames {missing}"
        )
    return np.stack([decoded[index] for index in frame_indices])


def write_export(
    output_dir: Path,
    *,
    frames: np.ndarray,
    siglip_rgb: np.ndarray,
    depth_rgb: np.ndarray,
    camera_names: tuple[str, ...],
    frame_indices: list[int],
    fps: float,
) -> list[dict[str, Any]]:
    """Write one RGB, SigLIP, and DA3 PNG per camera and sampled frame."""
    from PIL import Image

    if fps <= 0:
        raise ValueError("fps must be positive")
    if frames.ndim != 5 or frames.shape[0] != len(camera_names):
        raise ValueError("frames must be [cameras,frames,height,width,3]")
    num_frames = len(frame_indices)
    expected_flat = len(camera_names) * num_frames
    if frames.shape[1] != num_frames:
        raise ValueError("frames and frame_indices must have equal lengths")
    if siglip_rgb.shape[0] != expected_flat:
        raise ValueError("siglip_rgb must be flattened in camera-major order")
    if depth_rgb.shape[0] != expected_flat:
        raise ValueError("depth_rgb must be flattened in camera-major order")

    records: list[dict[str, Any]] = []
    for camera_index, camera_name in enumerate(camera_names):
        for sampled_index, frame_index in enumerate(frame_indices):
            flat_index = camera_index * num_frames + sampled_index
            paths = artifact_paths(output_dir, camera_name, frame_index)
            paths["rgb"].parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(frames[camera_index, sampled_index]).save(
                paths["rgb"]
            )
            Image.fromarray(siglip_rgb[flat_index]).save(paths["siglip_pca"])
            Image.fromarray(depth_rgb[flat_index]).save(paths["da3_depth"])
            records.append(
                {
                    "camera": camera_name,
                    "frame_index": frame_index,
                    "timestamp_seconds": frame_index / fps,
                    "files": {
                        modality: str(path.relative_to(output_dir))
                        for modality, path in paths.items()
                    },
                }
            )
    return records


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line contract without importing model dependencies."""
    parser = argparse.ArgumentParser(
        description=(
            "Export independent RGB, SigLIP2 PCA, and DA3 depth PNGs from "
            "one sampled LIBERO episode."
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--suite",
        default="libero_10_no_noops_lerobot",
    )
    parser.add_argument("--episode-index", type=int, default=288)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--siglip2-model-dir", type=Path, required=True)
    parser.add_argument("--da3-ckpt-dir", type=Path, required=True)
    parser.add_argument("--da3-code-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser


def _required_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"missing {description}: {path}")
    return path


@torch.inference_mode()
def _encode_siglip2(
    frames_b3hw: torch.Tensor,
    *,
    model_dir: Path,
    batch_size: int,
    device: torch.device,
    grid_size: int,
) -> torch.Tensor:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from siglip2_target import Siglip2TargetEncoder

    encoder = Siglip2TargetEncoder(
        model_dir=model_dir,
        input_size=256,
        grid_size=grid_size,
        device=device,
    )
    token_count = grid_size * grid_size
    feature_batches = []
    for start in range(0, frames_b3hw.shape[0], batch_size):
        frame_batch = frames_b3hw[start : start + batch_size]
        encoded = encoder.encode_future_keyframes(
            frame_batch[:1],
            [frame.unsqueeze(0) for frame in frame_batch],
        )
        feature_batches.append(
            encoded[0].reshape(frame_batch.shape[0], token_count, -1).cpu()
        )
    features = torch.cat(feature_batches, dim=0)
    del encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features


@torch.inference_mode()
def _encode_da3_depth(
    frames_b3hw: torch.Tensor,
    *,
    checkpoint_dir: Path,
    code_root: Path,
    batch_size: int,
    device: torch.device,
    process_res: int,
) -> torch.Tensor:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from depth_anything3_target import _import_da3

    depth_anything = _import_da3(str(code_root))
    full_model = depth_anything.from_pretrained(str(checkpoint_dir))
    full_model = full_model.to(device).eval()
    full_model.requires_grad_(False)
    model = full_model.model
    model_dtype = next(model.parameters()).dtype
    mean = torch.tensor(
        (0.485, 0.456, 0.406),
        device=device,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        (0.229, 0.224, 0.225),
        device=device,
    ).view(1, 3, 1, 1)

    depth_batches = []
    for start in range(0, frames_b3hw.shape[0], batch_size):
        frame_batch = frames_b3hw[start : start + batch_size]
        images = frame_batch.to(device).float()
        if images.max() > 1.5:
            images = images / 255.0
        images = F.interpolate(
            images,
            size=(process_res, process_res),
            mode="bilinear",
            align_corners=False,
        )
        images = ((images - mean) / std).to(model_dtype).unsqueeze(1)
        features, _ = model.backbone(
            images,
            cam_token=None,
            export_feat_layers=[],
            ref_view_strategy="saddle_balanced",
        )
        with torch.autocast(device_type=device.type, enabled=False):
            output = model._process_depth_head(
                features,
                process_res,
                process_res,
            )
        depth = output["depth"] if hasattr(output, "keys") else output.depth
        depth = depth.detach().float()
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        if depth.ndim != 3:
            raise RuntimeError(
                f"DA3 depth must be [batch,height,width], got {depth.shape}"
            )
        depth_batches.append(depth.cpu())
    return torch.cat(depth_batches, dim=0)


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    data_root = _required_path(args.data_root, "LIBERO data root")
    siglip2_model_dir = _required_path(
        args.siglip2_model_dir,
        "SigLIP2 model",
    )
    da3_checkpoint_dir = _required_path(args.da3_ckpt_dir, "DA3 checkpoint")
    da3_code_root = _required_path(args.da3_code_root, "DA3 code root")

    episode = load_episode_record(
        data_root,
        args.suite,
        args.episode_index,
    )
    num_frames = int(episode["length"])
    frame_indices = sample_frame_indices(num_frames, args.stride)
    info_path = data_root / args.suite / "meta/info.json"
    info = json.loads(
        _required_path(info_path, "LIBERO info metadata").read_text(
            encoding="utf-8"
        )
    )
    fps = float(info["fps"])
    chunks_size = int(info["chunks_size"])
    chunk_index = args.episode_index // chunks_size
    camera_names = tuple(name for name, _ in CAMERA_VIDEO_DIRS)
    video_paths = {
        name: (
            data_root
            / args.suite
            / "videos"
            / f"chunk-{chunk_index:03d}"
            / video_dir
            / f"episode_{args.episode_index:06d}.mp4"
        )
        for name, video_dir in CAMERA_VIDEO_DIRS
    }
    for camera_name, video_path in video_paths.items():
        _required_path(video_path, f"{camera_name} camera video")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "manifest.json").exists() or any(
        args.output_dir.rglob("*.png")
    ):
        raise FileExistsError(
            f"output directory already contains an export: {args.output_dir}"
        )

    print(
        json.dumps(
            {
                "status": "loading_frames",
                "suite": args.suite,
                "episode_index": args.episode_index,
                "num_frames": num_frames,
                "frame_indices": frame_indices,
                "camera_names": camera_names,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    frames = np.stack(
        [
            decode_episode_frames(video_paths[camera], frame_indices)
            for camera in camera_names
        ]
    )
    flat_frames = frames.reshape(
        len(camera_names) * len(frame_indices),
        *frames.shape[2:],
    )
    flat_frames_b3hw = torch.from_numpy(
        np.ascontiguousarray(flat_frames)
    ).permute(0, 3, 1, 2)
    device = torch.device(args.device)

    print(
        json.dumps({"status": "encoding_siglip2"}, sort_keys=True),
        flush=True,
    )
    siglip_features = _encode_siglip2(
        flat_frames_b3hw,
        model_dir=siglip2_model_dir,
        batch_size=args.batch_size,
        device=device,
        grid_size=16,
    )
    siglip_rgb = siglip_pca_images(
        siglip_features,
        grid_size=16,
        output_size=256,
    )

    print(
        json.dumps({"status": "encoding_da3"}, sort_keys=True),
        flush=True,
    )
    depth = _encode_da3_depth(
        flat_frames_b3hw,
        checkpoint_dir=da3_checkpoint_dir,
        code_root=da3_code_root,
        batch_size=args.batch_size,
        device=device,
        process_res=224,
    )
    depth_rgb = da3_depth_images(depth)

    records = write_export(
        args.output_dir,
        frames=frames,
        siglip_rgb=siglip_rgb,
        depth_rgb=depth_rgb,
        camera_names=camera_names,
        frame_indices=frame_indices,
        fps=fps,
    )
    manifest = {
        "suite": args.suite,
        "episode_index": args.episode_index,
        "instruction": str(episode["tasks"][0]),
        "fps": fps,
        "num_frames": num_frames,
        "stride": args.stride,
        "frame_indices": frame_indices,
        "camera_names": camera_names,
        "models": {
            "siglip2": str(siglip2_model_dir),
            "da3": str(da3_checkpoint_dir),
            "da3_code_root": str(da3_code_root),
        },
        "records": records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    png_count = len(list(args.output_dir.rglob("*.png")))
    expected_png_count = len(frame_indices) * len(camera_names) * 3
    if png_count != expected_png_count:
        raise RuntimeError(
            f"expected {expected_png_count} PNGs, found {png_count}"
        )
    print(
        json.dumps(
            {
                "status": "done",
                "sampled_frame_count": len(frame_indices),
                "record_count": len(records),
                "png_count": png_count,
                "manifest": str(manifest_path),
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
