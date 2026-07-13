#!/usr/bin/env python3
"""Fine-tune Qwen3-VL-4B as a semantic planner — LINGBOT-DINO variant (full lingbot-vla-v2 recipe).

Independent 4B line (2B CoVT + tasktoken untouched). Differs from the SigLIP scripts:
  * base VLM = Qwen3-VL-4B extracted from robbyant/lingbot-vla-v2-6b (--model-path → extracted dir);
  * target  = DINO-video (1024-d, 256 tokens/keyframe) from lingbot's teacher, online, NOT SigLIP;
  * head    = faithful TaskTokenResampler (LingbotDinoPlanHead), warm-startable from the 6b
              future_video_align_head (--head-warmstart-ckpt);
  * loss    = plain MSE (set via loss weights: mse=1, others=0).
Plan = num_keyframes × 256 tokens (grid_size=16 ⇒ 16²=256), e.g. 5×256 = [B, 1280, 1024].

NOTE: this plan lives in DINO-video space, so the Cosmos WM must be retrained to consume it — a
separate downstream step (not covered here). See lingbot_dino_4b/LINGBOT_DINO_SPEC.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# Per-rank torch.compile / Triton cache dirs. With N ranks sharing ONE cache dir, concurrent
# flex_attention kernel compiles race on the same artifact (FileNotFoundError on
# triton_tem_fused_*.cubin). torchrun exports LOCAL_RANK before this script runs, so scope the
# cache per rank BEFORE importing torch (inductor/triton read these env vars at compile time).
_local_rank = os.environ.get("LOCAL_RANK")
if _local_rank is not None:
    for _cache_var in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        _base = os.environ.get(_cache_var)
        if _base:
            os.environ[_cache_var] = os.path.join(_base, f"rank{_local_rank}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from qwen3vl_wrapper import load_qwen3vl_model_and_processor, move_qwen_inputs_to_device

# 4B-specific modules live in the lingbot_dino_4b/ subpackage; add it to the path (flat import,
# matching how lingbot_dino_head imports lingbot_resampler).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lingbot_dino_4b"))
from lingbot_dino_head import LingbotDinoPlanHead  # noqa: E402
from dino_video_target import DinoVideoTargetEncoder  # noqa: E402
from depth_target import DepthTargetEncoder  # noqa: E402
from distributed_runtime import (  # noqa: E402
    accumulation_context,
    build_accelerator,
    checkpoint_module,
    is_deepspeed,
    is_optimizer_update,
    validate_runtime_contract,
)


def _load_lingbot_head_state(src_6b_dir: Path) -> dict:
    """Stream all LingBot current/future DINO/depth alignment heads and queries."""
    from collections import defaultdict

    from safetensors import safe_open

    src = Path(src_6b_dir)
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    head_names = (
        "current_video_align_head",
        "future_video_align_head",
        "depth_align_head",
        "future_depth_align_head",
    )
    query_names = (
        "current_video_align_embs",
        "future_video_align_embs",
        "depth_align_embs",
        "future_depth_align_embs",
    )
    want = [
        key
        for key in index
        if any(f"{name}." in key for name in head_names)
        or key.endswith(query_names)
    ]
    by_file: dict[str, list[str]] = defaultdict(list)
    for k in want:
        by_file[index[k]].append(k)
    state: dict[str, torch.Tensor] = {}
    for fname, keys in by_file.items():
        with safe_open(src / fname, framework="pt", device="cpu") as sf:
            for k in keys:
                state[k] = sf.get_tensor(k)
    return state

PLAN_TOKEN = "<|sem_plan|>"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def configure_gradient_checkpointing(model: nn.Module, *, enabled: bool) -> None:
    if not enabled:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        return
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--fastwam-data-config", type=Path, default=None)
    parser.add_argument("--fastwam-dataset-dir", action="append", default=[])
    parser.add_argument(
        "--fastwam-text-embedding-cache-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--fastwam-pretrained-norm-stats",
        type=Path,
        default=None,
    )
    parser.add_argument("--plan-label-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    # Online labels (cosmos-style episode sampling): instead of fixed precomputed .pt windows,
    # sample a random sequence-length window per stem per epoch from frame_ranges.json and
    # encode the SigLIP2 targets on the fly — aligns the planner's window distribution with
    # the world model's episode sampling and makes the keyframe scheme an env-free switch.
    parser.add_argument("--online-plan-labels", action="store_true")
    parser.add_argument("--frame-ranges-json", type=Path, default=None)
    parser.add_argument("--siglip2-encoder-path", type=Path, default=None)
    parser.add_argument("--sequence-length", type=int, default=49)
    parser.add_argument(
        "--keyframe-scheme",
        choices=["uniform", "late", "even_future"],
        default="uniform",
    )
    parser.add_argument("--keyframe-gamma", type=float, default=0.6)
    parser.add_argument("--online-grid-size", type=int, default=0, help="<=0 keeps the native SigLIP2 grid")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-one-window-per-stem", action="store_true")
    parser.add_argument("--episode-window-sampling", choices=["random", "round_robin"], default="random")
    parser.add_argument("--episode-window-seed", type=int, default=-1)
    parser.add_argument("--num-keyframes", type=int, default=6)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--semantic-dim", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--expected-global-batch", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument(
        "--plan-head-type", choices=["mlp", "baton_crossattn", "covt", "lingbot_dino"], default="lingbot_dino"
    )
    # lingbot_dino: online DINO-video teacher target + optional head warm-start from the 6b ckpt.
    parser.add_argument("--dino-teacher-ckpt", type=Path, default=None, help="dino_video/teacher_step_*.pth")
    parser.add_argument("--dino-teacher-config", type=Path, default=None, help="dino_video/config.yaml")
    parser.add_argument("--dino-input-size", type=int, default=256)
    # Auxiliary future-DEPTH alignment (lingbot-style): MoGe-2 -> MoRGBD depth-feature teacher + a
    # second warm-started TaskTokenResampler head, smooth_L1 loss. Off unless --use-depth.
    parser.add_argument("--use-depth", action="store_true", help="add lingbot future-depth alignment (aux)")
    parser.add_argument("--depth-moge-path", type=Path, default=None, help="MoGe-2 weights (moge-2-vitb-normal/model.pt)")
    parser.add_argument("--depth-morgbd-path", type=Path, default=None, help="LingBot-Depth/MoRGBD weights (6b/depth/model.pt)")
    parser.add_argument("--depth-input-size", type=int, default=256)
    parser.add_argument("--depth-grid-size", type=int, default=16, help="depth teacher token grid side (16 -> 256)")
    parser.add_argument("--depth-dim", type=int, default=1024, help="MoRGBD feature dim")
    parser.add_argument("--depth-loss-weight", type=float, default=0.004, help="lingbot future_depth_loss_weight")
    parser.add_argument(
        "--use-current-alignment",
        action="store_true",
        help="LingBot-native current+one-future DINO/depth alignment",
    )
    parser.add_argument(
        "--independent-modality-task-tokens",
        action="store_true",
        help=(
            "use four private --num-task-tokens groups for current/future DINO/depth "
            "instead of sharing one group between modalities"
        ),
    )
    parser.add_argument("--num-task-tokens", type=_positive_int, default=8)
    parser.add_argument("--current-dino-loss-weight", type=float, default=0.004)
    parser.add_argument("--future-dino-loss-weight", type=float, default=0.004)
    parser.add_argument("--current-depth-loss-weight", type=float, default=0.004)
    parser.add_argument("--future-depth-loss-weight", type=float, default=0.004)
    parser.add_argument(
        "--head-warmstart-ckpt", type=Path, default=None,
        help="lingbot-vla-v2-6b dir: warm-start the head from future_video_align_head.*",
    )
    # CoVT-style bottleneck: the LM emits only num-latent-per-keyframe continuous latents per
    # keyframe (a compact "visual thought"), and a decoder reconstructs the dense SigLIP grid
    # from them. Decouples the LM sequence length from the target grid resolution.
    parser.add_argument("--num-latent-per-keyframe", type=int, default=4)
    parser.add_argument("--shared-latent-per-keyframe", type=_positive_int, default=32)
    parser.add_argument("--private-latent-per-keyframe", type=_positive_int, default=32)
    parser.add_argument("--plan-head-num-heads", type=int, default=16)
    parser.add_argument("--plan-head-dropout", type=float, default=0.0)
    parser.add_argument("--sem-mlp-hidden-size", type=int, default=0)
    # Loss recipe: plain per-token MSE regresses multimodal futures to their mean (norm
    # shrinkage + spatially blurred, non-discriminative plans). Direction (cosine) and
    # magnitude (relative norm) are supervised separately, InfoNCE supervises token-level
    # discriminativeness (the quantity sample_retrieval_top1 measures), and the dispersion
    # term penalizes predicting the per-sample mean at every token. The pre-fix recipe is
    # `--mse-loss-weight 1 --cosine-loss-weight 0.1` with the other weights at 0.
    parser.add_argument("--mse-loss-weight", type=float, default=1.0)
    parser.add_argument("--cosine-loss-weight", type=float, default=1.0)
    parser.add_argument("--norm-loss-weight", type=float, default=0.2)
    parser.add_argument("--variance-loss-weight", type=float, default=0.1)
    parser.add_argument("--infonce-loss-weight", type=float, default=0.1)
    parser.add_argument("--infonce-temperature", type=float, default=0.07)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--lr-schedule", choices=["cosine", "constant", "none"], default="cosine")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="enable non-reentrant gradient checkpointing (disabled by default)",
    )
    parser.add_argument("--freeze-vision", action="store_true", default=True)
    parser.add_argument("--no-freeze-vision", action="store_false", dest="freeze_vision")
    parser.add_argument("--freeze-lm-head", action="store_true", default=True)
    parser.add_argument("--no-freeze-lm-head", action="store_false", dest="freeze_lm_head")
    parser.add_argument("--train-plan-token-embedding", action="store_true", default=True)
    parser.add_argument("--no-train-plan-token-embedding", action="store_false", dest="train_plan_token_embedding")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--log-steps", type=int, default=10)
    args = parser.parse_args()
    if (args.dataset_root is None) == (args.fastwam_data_config is None):
        parser.error(
            "exactly one of --dataset-root or --fastwam-data-config is required"
        )
    if args.fastwam_dataset_dir and args.fastwam_data_config is None:
        parser.error("--fastwam-dataset-dir requires --fastwam-data-config")
    if (
        args.fastwam_text_embedding_cache_dir is not None
        and args.fastwam_data_config is None
    ):
        parser.error(
            "--fastwam-text-embedding-cache-dir requires --fastwam-data-config"
        )
    if (
        args.fastwam_pretrained_norm_stats is not None
        and args.fastwam_data_config is None
    ):
        parser.error(
            "--fastwam-pretrained-norm-stats requires --fastwam-data-config"
        )
    if args.fastwam_data_config is not None:
        if not args.online_plan_labels:
            parser.error("--fastwam-data-config requires --online-plan-labels")
        try:
            offsets = keyframe_offsets(
                args.sequence_length,
                args.num_keyframes,
                args.keyframe_scheme,
                args.keyframe_gamma,
            )
        except ValueError as error:
            parser.error(f"invalid FastWAM keyframe geometry: {error}")
        expected_keyframes = 1 if args.use_current_alignment else 4
        expected_offsets = [8] if args.use_current_alignment else [2, 4, 6, 8]
        if (
            args.sequence_length != 9
            or args.num_keyframes != expected_keyframes
            or offsets != expected_offsets
        ):
            parser.error(
                "FastWAM planner training requires sequence_length=9, "
                f"num_keyframes={expected_keyframes}, and keyframe offsets "
                f"{expected_offsets}"
            )
    loss_weights = (
        args.current_dino_loss_weight,
        args.future_dino_loss_weight,
        args.current_depth_loss_weight,
        args.future_depth_loss_weight,
    )
    if any(not math.isfinite(weight) or weight < 0 for weight in loss_weights):
        parser.error("current/future alignment loss weights must be finite and non-negative")
    if args.use_current_alignment and not (
        args.plan_head_type == "lingbot_dino" and args.use_depth
    ):
        parser.error("--use-current-alignment requires --plan-head-type lingbot_dino --use-depth")
    if args.independent_modality_task_tokens and not args.use_current_alignment:
        parser.error(
            "--independent-modality-task-tokens requires --use-current-alignment"
        )
    return args


def is_main(rank: int) -> bool:
    return rank == 0


def load_frame(video_path: Path, index: int) -> Image.Image:
    import decord

    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    index = min(max(int(index), 0), len(vr) - 1)
    return Image.fromarray(vr[index].asnumpy()).convert("RGB")


class SemanticPlanDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        label_dir: Path,
        num_keyframes: int,
        grid_size: int,
        max_samples: int = 0,
        sample_one_window_per_stem: bool = False,
        episode_window_sampling: str = "random",
        episode_window_seed: int = 0,
    ) -> None:
        self.dataset_root = dataset_root
        self.label_dir = label_dir
        self.num_keyframes = num_keyframes
        self.grid_size = grid_size
        self.sample_one_window_per_stem = sample_one_window_per_stem
        self.episode_window_sampling = episode_window_sampling
        self.episode_window_seed = int(episode_window_seed)
        self.epoch = 0
        manifest_paths = [label_dir / "manifest.jsonl"] if (label_dir / "manifest.jsonl").exists() else sorted(label_dir.glob("manifest*.jsonl"))
        entries = []
        if manifest_paths:
            for manifest_path in manifest_paths:
                for line in manifest_path.read_text().splitlines():
                    if line.strip():
                        entries.append(json.loads(line))
            entry_paths = [(entry, Path(entry["path"])) for entry in entries]
            entry_paths = [(entry, path) for entry, path in entry_paths if path.exists()]
            paths = [path for _, path in entry_paths]
        else:
            paths = sorted(label_dir.glob("*.pt"))
            entry_paths = []
        paths = [p for p in paths if p.exists()]
        if sample_one_window_per_stem:
            grouped: dict[str, list[Path]] = {}
            if entry_paths:
                for entry, path in entry_paths:
                    stem = str(entry.get("stem") or Path(entry.get("path", path)).stem.split("__r", 1)[0])
                    grouped.setdefault(stem, []).append(path)
            else:
                for path in paths:
                    stem = path.stem.split("__r", 1)[0]
                    grouped.setdefault(stem, []).append(path)
            self.groups = [sorted(grouped[k], key=lambda p: str(p)) for k in sorted(grouped)]
            if max_samples > 0:
                self.groups = self.groups[:max_samples]
            self.paths = []
        else:
            self.paths = paths
            if max_samples > 0:
                self.paths = self.paths[:max_samples]
            self.groups = []
        if sample_one_window_per_stem:
            if not self.groups:
                raise RuntimeError(f"No semantic plan label groups found under {label_dir}")
        elif not self.paths:
            raise RuntimeError(f"No semantic plan labels found under {label_dir}")

    def __len__(self) -> int:
        return len(self.groups) if self.sample_one_window_per_stem else len(self.paths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _choose_window_path(self, index: int) -> Path:
        group = self.groups[index]
        if len(group) == 1:
            return group[0]
        if self.episode_window_sampling == "round_robin":
            return group[self.epoch % len(group)]
        seed = (self.episode_window_seed + 1_000_003 * self.epoch + 9_176 * index) & 0xFFFFFFFF
        rng = random.Random(seed)
        return group[rng.randrange(len(group))]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.sample_one_window_per_stem:
            path = self._choose_window_path(index)
        else:
            path = self.paths[index]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        stem = payload["stem"]
        video_path = Path(payload.get("video_path") or self.dataset_root / "videos" / f"{stem}.mp4")
        prompt = payload.get("prompt") or "A robot manipulates the target object."
        first_frame_index = int(payload.get("first_frame_index", 0))
        image = load_frame(video_path, first_frame_index)
        plan = payload["semantic_plan"].float()
        expected = self.num_keyframes * self.grid_size * self.grid_size
        plan = plan.reshape(-1, plan.shape[-1])
        if plan.shape[0] != expected:
            raise RuntimeError(f"{path} has {plan.shape[0]} plan tokens, expected {expected}")
        return {
            "stem": stem,
            "sample_id": payload.get("sample_id", path.stem),
            "image": image,
            "prompt": prompt,
            "semantic_plan": plan,
            "feature_type": payload.get("feature_type", "unknown"),
        }


def keyframe_offsets(sequence_length: int, n: int, scheme: str, gamma: float) -> list[int]:
    """Window-relative keyframe positions; mirrors the world model's
    semantic_plan_conditioning.keyframe_indices (uniform over [1, T-1] excluding the
    conditioning frame; "late" = end-densified sqrt-warp with strictly increasing fixup)."""
    n = max(min(n, sequence_length), 1)
    if sequence_length <= 1:
        return [0] * n
    if scheme == "even_future":
        future_frames = sequence_length - 1
        if future_frames % n != 0:
            raise ValueError(
                f"even_future requires {future_frames} future frames "
                f"to be divisible by {n} keyframes"
            )
        step = future_frames // n
        return [step * (index + 1) for index in range(n)]
    if scheme == "late" and n > 1:
        u = torch.linspace(0.0, 1.0, n) ** float(gamma)
        idx = (1.0 + (sequence_length - 1 - 1) * u).round().long()
        idx = torch.clamp(idx, 1, sequence_length - 1)
        for i in range(1, n):
            if idx[i] <= idx[i - 1]:
                idx[i] = min(int(idx[i - 1]) + 1, sequence_length - 1)
        return idx.tolist()
    return torch.linspace(1, sequence_length - 1, n).round().long().tolist()


class OnlineSemanticPlanDataset(Dataset):
    """Cosmos-style episode sampling for planner training.

    Each item is a stem; every epoch a fresh stride-1 window of ``sequence_length`` frames
    is drawn from that stem's valid frame ranges (range picked ∝ its number of valid starts,
    start uniform within the range).  Returns the window's first frame (planner input) plus
    the raw keyframe frames; SigLIP2 targets are encoded on-GPU in the training loop with the
    offline builder's exact encode_images, so online targets match precomputed labels."""

    def __init__(
        self,
        dataset_root: Path,
        frame_ranges_json: Path,
        num_keyframes: int,
        sequence_length: int,
        keyframe_scheme: str,
        keyframe_gamma: float,
        max_samples: int = 0,
        seed: int = 0,
    ) -> None:
        from build_siglip2_semantic_plan_labels import load_frame_ranges

        self.dataset_root = dataset_root
        self.sequence_length = int(sequence_length)
        self.offsets = keyframe_offsets(self.sequence_length, num_keyframes, keyframe_scheme, keyframe_gamma)
        self.seed = int(seed)
        self.epoch = 0
        ranges = load_frame_ranges(frame_ranges_json)
        self.items: list[tuple[str, list[tuple[int, int, int]]]] = []
        for stem in sorted(ranges):
            fitting = []
            for range_start, range_end in ranges[stem]:
                n_starts = int(range_end) - int(range_start) - self.sequence_length + 1
                if n_starts > 0:
                    fitting.append((int(range_start), int(range_end), n_starts))
            if fitting:
                self.items.append((stem, fitting))
        if max_samples > 0:
            self.items = self.items[:max_samples]
        if not self.items:
            raise RuntimeError(f"No stems with a fitting {self.sequence_length}-frame window in {frame_ranges_json}")

    def __len__(self) -> int:
        return len(self.items)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import numpy as np

        from build_siglip2_semantic_plan_labels import load_frames, read_meta

        stem, fitting = self.items[index]
        rng_seed = (self.seed + 1_000_003 * self.epoch + 9_176 * index) & 0xFFFFFFFF
        rng = random.Random(rng_seed)
        total = sum(n for _, _, n in fitting)
        pick = rng.randrange(total)
        for range_start, _range_end, n_starts in fitting:
            if pick < n_starts:
                start = range_start + pick
                break
            pick -= n_starts
        frame_indices = [start] + [start + o for o in self.offsets]
        video_path = self.dataset_root / "videos" / f"{stem}.mp4"
        frames = load_frames(video_path, frame_indices)
        keyframes = np.stack([np.asarray(img, dtype=np.uint8) for img in frames[1:]], axis=0)
        return {
            "stem": stem,
            "sample_id": f"{stem}__s{start:06d}",
            "image": frames[0],
            "prompt": read_meta(self.dataset_root, stem),
            "keyframe_images": torch.from_numpy(keyframes),  # (K, H, W, 3) uint8
            # current frame (frames[0]) as raw uint8 too — the DINO teacher needs it as the clip's
            # current/warmup frame (the PIL `image` above is consumed by the VLM processor).
            "current_image": torch.from_numpy(np.asarray(frames[0], dtype=np.uint8)),  # (H, W, 3) uint8
        }


def _resolve_fastwam_path(path: str | os.PathLike, *, base_dir: Path) -> str:
    expanded = Path(os.fspath(path)).expanduser()
    if not expanded.is_absolute():
        expanded = base_dir / expanded
    return str(expanded.resolve())


def prepare_fastwam_data_config(
    config_path: Path,
    *,
    dataset_dirs: Sequence[str] | None = None,
    text_embedding_cache_dir: Path | None = None,
    pretrained_norm_stats: Path | None = None,
):
    """Load FastWAM data config with explicit, cwd-independent data paths."""
    from omegaconf import OmegaConf

    resolved_config_path = Path(config_path).expanduser().resolve()
    fastwam_root = resolved_config_path.parents[2]
    caller_cwd = Path.cwd()
    data_config = OmegaConf.load(resolved_config_path)
    root_config = OmegaConf.create(
        {
            "data": OmegaConf.to_container(
                data_config,
                resolve=False,
            )
        }
    )
    train_config = root_config.data.train
    if dataset_dirs:
        train_config.dataset_dirs = [
            _resolve_fastwam_path(path, base_dir=caller_cwd)
            for path in dataset_dirs
        ]
    else:
        train_config.dataset_dirs = [
            _resolve_fastwam_path(path, base_dir=fastwam_root)
            for path in train_config.dataset_dirs
        ]
    selected_text_cache = (
        text_embedding_cache_dir
        if text_embedding_cache_dir is not None
        else train_config.get("text_embedding_cache_dir")
    )
    if selected_text_cache is not None:
        train_config.text_embedding_cache_dir = _resolve_fastwam_path(
            selected_text_cache,
            base_dir=(
                caller_cwd
                if text_embedding_cache_dir is not None
                else fastwam_root
            ),
        )
    selected_norm_stats = (
        pretrained_norm_stats
        if pretrained_norm_stats is not None
        else train_config.get("pretrained_norm_stats")
    )
    if selected_norm_stats is not None:
        train_config.pretrained_norm_stats = _resolve_fastwam_path(
            selected_norm_stats,
            base_dir=(
                caller_cwd
                if pretrained_norm_stats is not None
                else fastwam_root
            ),
        )
    return root_config


def preflight_fastwam_data_config(
    config_path: Path,
    *,
    dataset_dirs: Sequence[str] | None = None,
    text_embedding_cache_dir: Path | None = None,
    pretrained_norm_stats: Path | None = None,
):
    """Validate FastWAM assets and import its Hydra target without allocation."""
    from hydra.utils import get_class

    root_config = prepare_fastwam_data_config(
        config_path,
        dataset_dirs=dataset_dirs,
        text_embedding_cache_dir=text_embedding_cache_dir,
        pretrained_norm_stats=pretrained_norm_stats,
    )
    train_config = root_config.data.train
    selected_dataset_dirs = list(train_config.get("dataset_dirs") or [])
    if not selected_dataset_dirs:
        raise ValueError("FastWAM dataset directories are not configured")
    for dataset_dir in selected_dataset_dirs:
        dataset_path = Path(os.fspath(dataset_dir))
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"FastWAM dataset directory does not exist: {dataset_path}"
            )
        if not dataset_path.is_dir():
            raise ValueError(
                f"FastWAM dataset directory is not a directory: {dataset_path}"
            )

    text_cache_dir = train_config.get("text_embedding_cache_dir")
    if text_cache_dir is None:
        raise ValueError("FastWAM text-embedding cache is not configured")
    text_cache_path = Path(os.fspath(text_cache_dir))
    if not text_cache_path.exists():
        raise FileNotFoundError(
            f"FastWAM text-embedding cache does not exist: {text_cache_path}"
        )
    if not text_cache_path.is_dir():
        raise ValueError(
            f"FastWAM text-embedding cache is not a directory: {text_cache_path}"
        )

    norm_stats = train_config.get("pretrained_norm_stats")
    if norm_stats is not None:
        norm_stats_path = Path(os.fspath(norm_stats))
        if not norm_stats_path.exists():
            raise FileNotFoundError(
                "FastWAM pretrained normalization stats do not exist: "
                f"{norm_stats_path}"
            )
        if not norm_stats_path.is_file():
            raise ValueError(
                "FastWAM pretrained normalization stats are not a regular file: "
                f"{norm_stats_path}"
            )

    target = root_config.data.train.get("_target_")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("FastWAM train data config requires a non-empty _target_")
    get_class(target)
    return root_config


class FastWAMOnlinePlannerDataset(Dataset):
    def __init__(
        self,
        dataset,
        max_samples: int = 0,
        offsets: Sequence[int] | None = None,
    ):
        self.dataset = dataset
        self.max_samples = int(max_samples)
        self.offsets = [int(offset) for offset in (offsets or [2, 4, 6, 8])]

    @classmethod
    def from_config(
        cls,
        config_path: Path,
        *,
        dataset_dirs: Sequence[str] | None = None,
        text_embedding_cache_dir: Path | None = None,
        pretrained_norm_stats: Path | None = None,
        max_samples: int = 0,
        offsets: Sequence[int] | None = None,
    ):
        from hydra.utils import instantiate

        root_config = prepare_fastwam_data_config(
            config_path,
            dataset_dirs=dataset_dirs,
            text_embedding_cache_dir=text_embedding_cache_dir,
            pretrained_norm_stats=pretrained_norm_stats,
        )
        dataset = instantiate(root_config.data.train)
        return cls(dataset, max_samples=max_samples, offsets=offsets)

    @classmethod
    def from_dataset(
        cls,
        dataset,
        max_samples: int = 0,
        offsets: Sequence[int] | None = None,
    ):
        return cls(dataset, max_samples=max_samples, offsets=offsets)

    def __len__(self):
        size = len(self.dataset)
        return min(size, self.max_samples) if self.max_samples > 0 else size

    def set_epoch(self, _epoch: int) -> None:
        return None

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        video = sample["video"]
        if video.ndim != 4 or tuple(video.shape[:2]) != (3, 9):
            raise ValueError(
                "FastWAM planner input must be [3, 9, H, W], got "
                f"{tuple(video.shape)}"
            )
        instruction = sample.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(
                "FastWAM planner sample needs a non-empty raw instruction"
            )
        selected = video[:, [0, *self.offsets]]
        selected = (
            ((selected.to(torch.float32) + 1.0) * 127.5)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 3, 0)
            .contiguous()
        )
        current = selected[0]
        return {
            "stem": f"fastwam_{index:09d}",
            "sample_id": f"fastwam_{index:09d}",
            "image": Image.fromarray(current.numpy()),
            "prompt": instruction,
            "keyframe_images": selected[1:],
            "current_image": current,
        }


PLANNER_USER_TEMPLATE = (
    "You are a robot video semantic planner. Given the first frame and instruction, "
    "predict future spatial semantic plan tokens for the manipulation video.\n"
    "Instruction: {instruction}"
)


def build_planner_inputs(
    processor: Any,
    images: list[Any],
    instructions: list[Any],
    plan_sequence: str | list[str],
) -> Any:
    if len(images) != len(instructions):
        raise ValueError(
            f"images/instructions batch mismatch: {len(images)} != {len(instructions)}"
        )
    plan_text = plan_sequence if isinstance(plan_sequence, str) else " ".join(plan_sequence)
    conversations = []
    for image, instruction in zip(images, instructions, strict=True):
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": PLANNER_USER_TEMPLATE.format(instruction=str(instruction)),
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": plan_text,
                },
            ]
        )
    texts = [
        processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        for conversation in conversations
    ]
    return processor(
        text=texts,
        images=list(images),
        padding=True,
        return_tensors="pt",
    )


@dataclass
class Collator:
    processor: Any
    plan_sequence: list[str]

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [x["image"] for x in batch]
        labels = None
        keyframes = None
        current = None
        if "semantic_plan" in batch[0]:
            labels = torch.stack([x["semantic_plan"] for x in batch], dim=0)
        else:  # online mode: raw keyframes (+ current frame for the DINO teacher), encoded in the loop
            keyframes = torch.stack([x["keyframe_images"] for x in batch], dim=0)
            if "current_image" in batch[0]:
                current = torch.stack([x["current_image"] for x in batch], dim=0)
        inputs = build_planner_inputs(
            self.processor,
            images,
            [item["prompt"] for item in batch],
            self.plan_sequence,
        )
        if labels is not None:
            inputs["semantic_plan_labels"] = labels
        if keyframes is not None:
            inputs["keyframe_images"] = keyframes
        if current is not None:
            inputs["current_image"] = current
        inputs["stems"] = [x["stem"] for x in batch]
        return inputs


class MLPPlanHead(nn.Module):
    def __init__(self, hidden_size: int, semantic_dim: int, sem_mlp_hidden_size: int) -> None:
        super().__init__()
        if sem_mlp_hidden_size > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, sem_mlp_hidden_size),
                nn.GELU(),
                nn.Linear(sem_mlp_hidden_size, semantic_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, semantic_dim),
            )

    def forward(self, plan_hidden: torch.Tensor) -> torch.Tensor:
        return self.net(plan_hidden)


class BatonCrossAttentionPlanHead(nn.Module):
    """Video-only Baton-style semantic alignment tower.

    Baton uses learnable queries that cross-attend to MLLM planning hidden
    states, then a Sem-MLP projects to the perceptual feature space.  We keep
    the same video tower idea and omit the audio cross-modal tower.
    """

    def __init__(
        self,
        *,
        plan_len: int,
        hidden_size: int,
        semantic_dim: int,
        sem_mlp_hidden_size: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.query = nn.Parameter(torch.empty(plan_len, hidden_size))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.sem_mlp = MLPPlanHead(hidden_size, semantic_dim, sem_mlp_hidden_size)

    def forward(self, plan_hidden: torch.Tensor) -> torch.Tensor:
        batch = plan_hidden.shape[0]
        query = self.query.unsqueeze(0).expand(batch, -1, -1)
        context = self.context_norm(plan_hidden)
        attn_out, _ = self.cross_attn(
            self.query_norm(query),
            context,
            context,
            need_weights=False,
        )
        query = query + attn_out
        query = query + self.ffn(self.ffn_norm(query))
        return self.sem_mlp(query)


class CoVTLatentDecoderHead(nn.Module):
    """CoVT DINO-branch-style latent bottleneck decoder.

    Faithful to the reference DINO head in CoVT (arXiv 2511.19418,
    train/src/training/covt_qwen2_5_vl.py): the VLM emits a small budget of continuous
    thinking tokens (CoVT uses 4 for DINO); each is projected to the target feature dim
    and L2-normalized, then a bank of learnable grid queries (the full target token grid)
    cross-attends to them through a single MultiheadAttention whose output is *directly*
    the reconstructed dense feature map (regressed to the frozen encoder features by MSE).
    No residual / LayerNorm / extra MLP — CoVT keeps this head minimal.

    Reference (per single image):
        dino_projection   = nn.Linear(hidden, 1024)         # LM hidden -> DINO dim
        dino_query_vectors= nn.Parameter(randn(1025, 1024)) # full DINO grid
        dino_cross_attn   = nn.MultiheadAttention(1024, 8)
        kv   = F.normalize(dino_projection(hidden[<dino> positions]))   # [B, 4, 1024]
        pred = dino_cross_attn(query=dino_query_vectors, key=kv, value=kv)  # [B, 1025, 1024]

    Here the target is the per-keyframe SigLIP grid, so the same head is applied per
    keyframe: keyframe kf is reconstructed from *its own* num_latent_per_keyframe thinking
    tokens, with a keyframe embedding added to the shared grid queries.  The LM sequence
    length (num_keyframes*num_latent_per_keyframe) is decoupled from the target grid
    resolution (num_keyframes*grid_size**2).
    """

    def __init__(
        self,
        *,
        num_keyframes: int,
        num_latent_per_keyframe: int,
        grid_size: int,
        hidden_size: int,
        semantic_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if semantic_dim % num_heads != 0:
            raise ValueError(f"semantic_dim={semantic_dim} must be divisible by num_heads={num_heads}")
        if grid_size <= 0:
            raise ValueError(f"covt head needs a positive target grid_size, got {grid_size}")
        self.num_keyframes = int(num_keyframes)
        self.num_latent_per_keyframe = int(num_latent_per_keyframe)
        self.grid_size = int(grid_size)
        self.grid_tokens = self.grid_size * self.grid_size
        # Project LM hidden -> target feature dim, then the queries/attention live in that dim.
        self.latent_projection = nn.Linear(hidden_size, semantic_dim)
        self.grid_query = nn.Parameter(torch.randn(self.grid_tokens, semantic_dim))
        self.keyframe_embed = nn.Parameter(torch.zeros(self.num_keyframes, semantic_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=semantic_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, latent_hidden: torch.Tensor) -> torch.Tensor:
        # latent_hidden: (B, num_keyframes * num_latent_per_keyframe, H)
        batch = latent_hidden.shape[0]
        hidden = latent_hidden.shape[-1]
        expected = self.num_keyframes * self.num_latent_per_keyframe
        if latent_hidden.shape[1] != expected:
            raise RuntimeError(
                f"covt head expected {expected} latent tokens, got {latent_hidden.shape[1]}"
            )
        latents = latent_hidden.reshape(batch * self.num_keyframes, self.num_latent_per_keyframe, hidden)
        # CoVT: project thinking tokens to the target dim and L2-normalize before cross-attn.
        kv = F.normalize(self.latent_projection(latents), dim=-1)
        # Per-keyframe grid queries: (num_keyframes, grid_tokens, D) -> (B*num_keyframes, grid_tokens, D)
        query = self.grid_query.unsqueeze(0) + self.keyframe_embed.unsqueeze(1)
        query = query.unsqueeze(0).expand(batch, -1, -1, -1)
        query = query.reshape(batch * self.num_keyframes, self.grid_tokens, self.grid_query.shape[-1])
        attn_out, _ = self.cross_attn(query.to(kv.dtype), kv, kv, need_weights=False)
        return attn_out.reshape(batch, self.num_keyframes * self.grid_tokens, attn_out.shape[-1])


class PlannerWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        hidden_size: int,
        semantic_dim: int,
        plan_token_ids: list[int],
        target_len: int,
        num_keyframes: int,
        grid_size: int,
        num_latent_per_keyframe: int = 4,
        plan_head_type: str = "mlp",
        plan_head_num_heads: int = 16,
        plan_head_dropout: float = 0.0,
        sem_mlp_hidden_size: int = 0,
        mse_loss_weight: float = 1.0,
        cosine_loss_weight: float = 1.0,
        norm_loss_weight: float = 0.2,
        variance_loss_weight: float = 0.1,
        infonce_loss_weight: float = 0.1,
        infonce_temperature: float = 0.07,
        use_depth: bool = False,
        depth_dim: int = 1024,
        depth_grid_size: int = 16,
        depth_loss_weight: float = 0.004,
        shared_latent_per_keyframe: int = 32,
        private_latent_per_keyframe: int = 32,
        use_current_alignment: bool = False,
        independent_modality_task_tokens: bool = False,
        num_task_tokens: int = 8,
        current_dino_loss_weight: float = 0.004,
        future_dino_loss_weight: float = 0.004,
        current_depth_loss_weight: float = 0.004,
        future_depth_loss_weight: float = 0.004,
    ) -> None:
        super().__init__()
        if sem_mlp_hidden_size < 0:
            sem_mlp_hidden_size = hidden_size
        self.mse_loss_weight = float(mse_loss_weight)
        self.cosine_loss_weight = float(cosine_loss_weight)
        self.norm_loss_weight = float(norm_loss_weight)
        self.variance_loss_weight = float(variance_loss_weight)
        self.infonce_loss_weight = float(infonce_loss_weight)
        self.infonce_temperature = float(infonce_temperature)
        self.sem_mlp_hidden_size = int(sem_mlp_hidden_size)
        self.plan_head_type = plan_head_type
        self.plan_head_num_heads = int(plan_head_num_heads)
        self.plan_head_dropout = float(plan_head_dropout)
        self.num_keyframes = int(num_keyframes)
        self.use_current_alignment = bool(use_current_alignment)
        self.independent_modality_task_tokens = bool(
            independent_modality_task_tokens
        )
        self.num_task_tokens = int(num_task_tokens)
        self.current_dino_loss_weight = float(current_dino_loss_weight)
        self.future_dino_loss_weight = float(future_dino_loss_weight)
        self.current_depth_loss_weight = float(current_depth_loss_weight)
        self.future_depth_loss_weight = float(future_depth_loss_weight)
        self.shared_latent_per_keyframe = int(shared_latent_per_keyframe)
        self.private_latent_per_keyframe = int(private_latent_per_keyframe)
        self.branch_latent_per_keyframe = (
            self.shared_latent_per_keyframe + self.private_latent_per_keyframe
        )
        self.total_unique_latent_per_keyframe = (
            self.shared_latent_per_keyframe + 2 * self.private_latent_per_keyframe
        )
        self.use_depth = bool(use_depth) and plan_head_type == "lingbot_dino"
        if self.use_current_alignment and not self.use_depth:
            raise ValueError("current alignment requires lingbot_dino with depth")
        if self.use_current_alignment and self.num_keyframes != 1:
            raise ValueError("current alignment predicts exactly one future keyframe")
        if self.independent_modality_task_tokens and not self.use_current_alignment:
            raise ValueError(
                "independent modality task tokens require current alignment"
            )
        if self.use_depth and not self.use_current_alignment and (
            self.shared_latent_per_keyframe <= 0 or self.private_latent_per_keyframe <= 0
        ):
            raise ValueError(
                "DINO+depth mode requires positive shared/private latent counts"
            )
        if getattr(self, "use_current_alignment", False):
            self.branch_latent_per_keyframe = self.num_task_tokens
            self.total_unique_latent_per_keyframe = (
                4 if self.independent_modality_task_tokens else 2
            ) * self.num_task_tokens
            self.num_latent_per_keyframe = self.num_task_tokens
        else:
            self.num_latent_per_keyframe = (
                self.branch_latent_per_keyframe
                if self.use_depth
                else int(num_latent_per_keyframe)
            )
        # target_len = tokens the loss regresses (dense SigLIP grid). latent_len = <|sem_plan|>
        # tokens the LM actually emits. They differ only for the CoVT bottleneck head.
        self.target_len = int(target_len)
        if plan_head_type == "mlp":
            self.latent_len = int(target_len)
            self.plan_head = MLPPlanHead(hidden_size, semantic_dim, sem_mlp_hidden_size)
        elif plan_head_type == "baton_crossattn":
            self.latent_len = int(target_len)
            self.plan_head = BatonCrossAttentionPlanHead(
                plan_len=self.latent_len,
                hidden_size=hidden_size,
                semantic_dim=semantic_dim,
                sem_mlp_hidden_size=sem_mlp_hidden_size,
                num_heads=plan_head_num_heads,
                dropout=plan_head_dropout,
            )
        elif plan_head_type == "covt":
            self.latent_len = int(num_keyframes) * int(num_latent_per_keyframe)
            self.plan_head = CoVTLatentDecoderHead(
                num_keyframes=num_keyframes,
                num_latent_per_keyframe=num_latent_per_keyframe,
                grid_size=grid_size,
                hidden_size=hidden_size,
                semantic_dim=semantic_dim,
                num_heads=plan_head_num_heads,
                dropout=plan_head_dropout,
            )
        elif plan_head_type == "lingbot_dino":
            # lingbot-vla-v2 rich-KV head predicting DINO-video patches: shared TaskTokenResampler
            # (warm-startable from future_video_align_head) run per keyframe. grid_size**2 = the DINO
            # patch-token count per keyframe (16**2 = 256). semantic_dim is the DINO dim (1024).
            self.latent_len = (
                (4 if self.independent_modality_task_tokens else 2)
                * self.num_task_tokens
                if self.use_current_alignment
                else (
                    self.num_keyframes * self.total_unique_latent_per_keyframe
                    if self.use_depth
                    else self.num_keyframes * self.num_latent_per_keyframe
                )
            )
            self.plan_head = LingbotDinoPlanHead(
                num_keyframes=num_keyframes,
                num_latent_per_keyframe=self.num_latent_per_keyframe,
                num_backbone_tokens=int(grid_size) * int(grid_size),
                llm_hidden=hidden_size,
                dim_out=semantic_dim,
            )
        else:
            raise ValueError(f"Unsupported plan_head_type: {plan_head_type}")
        # Auxiliary future-DEPTH alignment head (lingbot-style, only for lingbot_dino): a second shared
        # TaskTokenResampler, warm-started from future_depth_align_head, reading the SAME image+latent
        # context and regressing LingBot-Depth features (MoGe-2 -> MoRGBD) with smooth_L1.
        self.depth_loss_weight = float(depth_loss_weight)
        self.depth_head = None
        self.current_plan_head = None
        self.current_depth_head = None
        if self.use_depth:
            self.depth_head = LingbotDinoPlanHead(
                num_keyframes=num_keyframes,
                num_latent_per_keyframe=self.branch_latent_per_keyframe,
                num_backbone_tokens=int(depth_grid_size) * int(depth_grid_size),
                llm_hidden=hidden_size,
                dim_out=depth_dim,
            )
        if getattr(self, "use_current_alignment", False):
            self.current_plan_head = LingbotDinoPlanHead(
                num_keyframes=1,
                num_latent_per_keyframe=self.num_task_tokens,
                num_backbone_tokens=int(grid_size) * int(grid_size),
                llm_hidden=hidden_size,
                dim_out=semantic_dim,
            )
            self.current_depth_head = LingbotDinoPlanHead(
                num_keyframes=1,
                num_latent_per_keyframe=self.num_task_tokens,
                num_backbone_tokens=int(depth_grid_size) * int(depth_grid_size),
                llm_hidden=hidden_size,
                dim_out=depth_dim,
            )
        self.plan_token_ids = [int(x) for x in plan_token_ids]
        self.model = model
        # image-token id used by the lingbot_dino head to gather the LLM's image-token hiddens
        self.image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)

    @classmethod
    def from_exported_checkpoint(
        cls,
        *,
        model: nn.Module,
        checkpoint_dir: str | Path,
        metadata: dict[str, Any],
    ) -> "PlannerWrapper":
        """Rebuild and freeze a planner from the files written by ``save_checkpoint``."""
        checkpoint_dir = Path(checkpoint_dir)
        text_config = getattr(model.config, "text_config", model.config)
        hidden_size = int(text_config.hidden_size)
        wrapper = cls(
            model=model,
            hidden_size=hidden_size,
            semantic_dim=int(metadata["semantic_dim"]),
            plan_token_ids=[int(value) for value in metadata["plan_token_ids"]],
            target_len=int(metadata["target_tokens"]),
            num_keyframes=int(metadata["num_keyframes"]),
            grid_size=int(metadata["grid_size"]),
            num_latent_per_keyframe=int(
                metadata["branch_latent_per_keyframe"]
            ),
            shared_latent_per_keyframe=int(
                metadata["shared_latent_per_keyframe"]
            ),
            private_latent_per_keyframe=int(
                metadata["private_latent_per_keyframe"]
            ),
            plan_head_type=str(metadata["plan_head_type"]),
            plan_head_num_heads=int(metadata["plan_head_num_heads"]),
            plan_head_dropout=float(metadata["plan_head_dropout"]),
            sem_mlp_hidden_size=int(metadata["sem_mlp_hidden_size"]),
            mse_loss_weight=float(metadata["mse_loss_weight"]),
            cosine_loss_weight=float(metadata["cosine_loss_weight"]),
            norm_loss_weight=float(metadata["norm_loss_weight"]),
            variance_loss_weight=float(metadata["variance_loss_weight"]),
            infonce_loss_weight=float(metadata["infonce_loss_weight"]),
            infonce_temperature=float(metadata["infonce_temperature"]),
            use_depth=True,
            depth_dim=int(metadata["depth_feature_dim"]),
            depth_grid_size=int(metadata["depth_grid_size"]),
            depth_loss_weight=float(metadata["depth_loss_weight"]),
            use_current_alignment=bool(metadata.get("use_current_alignment", False)),
            independent_modality_task_tokens=bool(
                metadata.get("independent_modality_task_tokens", False)
            ),
            num_task_tokens=int(metadata.get("num_task_tokens", 8)),
            current_dino_loss_weight=float(metadata.get("current_dino_loss_weight", 0.004)),
            future_dino_loss_weight=float(metadata.get("future_dino_loss_weight", 0.004)),
            current_depth_loss_weight=float(metadata.get("current_depth_loss_weight", 0.004)),
            future_depth_loss_weight=float(metadata.get("future_depth_loss_weight", 0.004)),
        )
        wrapper.plan_head.load_state_dict(
            torch.load(
                checkpoint_dir / "plan_head.pt",
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        wrapper.depth_head.load_state_dict(
            torch.load(
                checkpoint_dir / "depth_head.pt",
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        if wrapper.use_current_alignment:
            wrapper.current_plan_head.load_state_dict(
                torch.load(
                    checkpoint_dir / "current_plan_head.pt",
                    map_location="cpu",
                    weights_only=True,
                ),
                strict=True,
            )
            wrapper.current_depth_head.load_state_dict(
                torch.load(
                    checkpoint_dir / "current_depth_head.pt",
                    map_location="cpu",
                    weights_only=True,
                ),
                strict=True,
            )
        plan_embedding = torch.load(
            checkpoint_dir / "plan_token_embedding.pt",
            map_location="cpu",
            weights_only=True,
        )
        embedding_weight = model.get_input_embeddings().weight
        expected_embedding_shape = (
            len(wrapper.plan_token_ids),
            embedding_weight.shape[1],
        )
        if tuple(plan_embedding.shape) != expected_embedding_shape:
            raise ValueError(
                "incompatible plan token embedding shape: "
                f"expected {expected_embedding_shape}, "
                f"got {tuple(plan_embedding.shape)}"
            )
        plan_ids = torch.tensor(
            wrapper.plan_token_ids,
            device=embedding_weight.device,
            dtype=torch.long,
        )
        with torch.no_grad():
            embedding_weight.index_copy_(
                0,
                plan_ids,
                plan_embedding.to(
                    device=embedding_weight.device,
                    dtype=embedding_weight.dtype,
                ),
            )
        wrapper.eval()
        for parameter in wrapper.parameters():
            parameter.requires_grad_(False)
        return wrapper

    def collect_plan_hidden(self, hidden: torch.Tensor, input_ids: torch.Tensor, plan_len: int) -> torch.Tensor:
        ids = torch.as_tensor(self.plan_token_ids, device=input_ids.device)
        # Distinct latent tokens each appear once, in emit order (keyframe-major); the single
        # <|sem_plan|> token (mlp/baton) appears plan_len times. Both gather in sequence order.
        plan_mask = torch.isin(input_ids, ids)
        plan_hidden = []
        for b in range(input_ids.shape[0]):
            h = hidden[b, plan_mask[b]]
            if h.shape[0] != plan_len:
                raise RuntimeError(f"Found {h.shape[0]} plan tokens, expected {plan_len}")
            plan_hidden.append(h)
        return torch.stack(plan_hidden, dim=0)

    def collect_image_hidden(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Gather the LLM's image-token hidden states (rich-KV context for the lingbot_dino head)."""
        if self.image_token_id is None:
            raise RuntimeError("lingbot_dino head needs model.config.image_token_id, which is unset")
        mask = input_ids == int(self.image_token_id)
        counts = mask.sum(dim=1)
        n_img = int(counts[0].item())
        if n_img == 0 or not bool(torch.all(counts == counts[0])):
            raise RuntimeError(
                f"lingbot_dino head needs an equal, nonzero image-token count per batch item, got {counts.tolist()}"
            )
        batch, _, hidden_dim = hidden.shape
        return hidden[mask].reshape(batch, n_img, hidden_dim)

    def _forward_hiddens(self, **inputs: Any) -> tuple[torch.Tensor | None, torch.Tensor]:
        """One VLM forward -> (image_hidden|None, plan_hidden). Image tokens are DETACHED (lingbot
        detach_image_feats=True). Shared by the video head, the depth head, and inference."""
        outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[-1]
        input_ids = inputs["input_ids"]
        plan_hidden = self.collect_plan_hidden(hidden, input_ids, self.latent_len)
        image_hidden = None
        if self.plan_head_type == "lingbot_dino":
            image_hidden = self.collect_image_hidden(hidden, input_ids).detach()
        return image_hidden, plan_hidden

    def split_lingbot_query_hidden(
        self,
        plan_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, token_count, hidden_size = plan_hidden.shape
        expected = self.num_keyframes * self.total_unique_latent_per_keyframe
        if token_count != expected:
            raise RuntimeError(
                f"expected {expected} shared/private query tokens, got {token_count}"
            )
        grouped = plan_hidden.reshape(
            batch,
            self.num_keyframes,
            self.total_unique_latent_per_keyframe,
            hidden_size,
        )
        shared_end = self.shared_latent_per_keyframe
        dino_end = shared_end + self.private_latent_per_keyframe
        depth_end = dino_end + self.private_latent_per_keyframe
        shared = grouped[:, :, :shared_end]
        dino_private = grouped[:, :, shared_end:dino_end]
        depth_private = grouped[:, :, dino_end:depth_end]
        dino_hidden = torch.cat([shared, dino_private], dim=2)
        depth_hidden = torch.cat([shared, depth_private], dim=2)
        return (
            dino_hidden.reshape(
                batch,
                self.num_keyframes * self.branch_latent_per_keyframe,
                hidden_size,
            ),
            depth_hidden.reshape(
                batch,
                self.num_keyframes * self.branch_latent_per_keyframe,
                hidden_size,
            ),
        )

    def split_current_future_task_hidden(
        self,
        plan_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = 2 * int(self.num_task_tokens)
        if plan_hidden.ndim != 3 or plan_hidden.shape[1] != expected:
            raise RuntimeError(
                f"expected {expected} current/future task tokens, got "
                f"{tuple(plan_hidden.shape)}"
            )
        return (
            plan_hidden[:, : self.num_task_tokens],
            plan_hidden[:, self.num_task_tokens :],
        )

    def split_independent_current_future_task_hidden(
        self,
        plan_hidden: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        expected = 4 * int(self.num_task_tokens)
        if plan_hidden.ndim != 3 or plan_hidden.shape[1] != expected:
            raise RuntimeError(
                f"expected {expected} independent modality task tokens, got "
                f"{tuple(plan_hidden.shape)}"
            )
        width = self.num_task_tokens
        return {
            "current_dino": plan_hidden[:, 0 * width : 1 * width],
            "future_dino": plan_hidden[:, 1 * width : 2 * width],
            "current_depth": plan_hidden[:, 2 * width : 3 * width],
            "future_depth": plan_hidden[:, 3 * width : 4 * width],
        }

    def compute_current_future_losses(
        self,
        plans: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        required = {
            "current_dino",
            "future_dino",
            "current_depth",
            "future_depth",
        }
        missing = sorted(required.difference(plans) | required.difference(targets))
        if missing:
            raise ValueError(f"missing current/future plan loss tensors: {missing}")
        current_dino = F.mse_loss(plans["current_dino"], targets["current_dino"])
        future_dino = F.mse_loss(plans["future_dino"], targets["future_dino"])
        current_depth = F.smooth_l1_loss(
            plans["current_depth"], targets["current_depth"]
        )
        future_depth = F.smooth_l1_loss(
            plans["future_depth"], targets["future_depth"]
        )
        weighted = {
            "current_dino_weighted": self.current_dino_loss_weight * current_dino,
            "future_dino_weighted": self.future_dino_loss_weight * future_dino,
            "current_depth_weighted": self.current_depth_loss_weight * current_depth,
            "future_depth_weighted": self.future_depth_loss_weight * future_depth,
        }
        return {
            "loss": sum(weighted.values()),
            "current_dino_mse": current_dino.detach(),
            "future_dino_mse": future_dino.detach(),
            "current_depth_smooth_l1": current_depth.detach(),
            "future_depth_smooth_l1": future_depth.detach(),
            **{name: value.detach() for name, value in weighted.items()},
        }

    def predict_dino_depth_plan(
        self,
        **model_inputs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.plan_head_type != "lingbot_dino":
            raise RuntimeError(
                "DINO+depth prediction requires plan_head_type=lingbot_dino"
            )
        if self.depth_head is None:
            raise RuntimeError(
                "DINO+depth prediction requires a configured depth head"
            )
        if getattr(self, "use_current_alignment", False):
            plans = self.predict_current_future_plans(**model_inputs)
            return plans["future_dino"], plans["future_depth"]
        image_hidden, plan_hidden = self._forward_hiddens(**model_inputs)
        dino_hidden, depth_hidden = self.split_lingbot_query_hidden(plan_hidden)
        dino_dtype = next(self.plan_head.parameters()).dtype
        depth_dtype = next(self.depth_head.parameters()).dtype
        dino_plan = self.plan_head(
            image_hidden.to(dtype=dino_dtype),
            dino_hidden.to(dtype=dino_dtype),
        ).float()
        depth_plan = self.depth_head(
            image_hidden.to(dtype=depth_dtype),
            depth_hidden.to(dtype=depth_dtype),
        ).float()
        if dino_plan.shape != depth_plan.shape:
            raise RuntimeError(
                "DINO/depth head output mismatch: "
                f"{tuple(dino_plan.shape)} != {tuple(depth_plan.shape)}"
            )
        return dino_plan, depth_plan

    def predict_current_future_plans(
        self,
        **model_inputs: Any,
    ) -> dict[str, torch.Tensor]:
        if not self.use_current_alignment:
            raise RuntimeError("current/future prediction requires current alignment mode")
        if self.current_plan_head is None or self.current_depth_head is None:
            raise RuntimeError("current alignment heads are not configured")
        image_hidden, task_hidden = self._forward_hiddens(**model_inputs)
        if self.independent_modality_task_tokens:
            hidden_by_branch = self.split_independent_current_future_task_hidden(
                task_hidden
            )
        else:
            current_hidden, future_hidden = self.split_current_future_task_hidden(
                task_hidden
            )
            hidden_by_branch = {
                "current_dino": current_hidden,
                "future_dino": future_hidden,
                "current_depth": current_hidden,
                "future_depth": future_hidden,
            }
        heads_and_hiddens = {
            "current_dino": (self.current_plan_head, hidden_by_branch["current_dino"]),
            "future_dino": (self.plan_head, hidden_by_branch["future_dino"]),
            "current_depth": (self.current_depth_head, hidden_by_branch["current_depth"]),
            "future_depth": (self.depth_head, hidden_by_branch["future_depth"]),
        }
        plans = {}
        for name, (head, hidden) in heads_and_hiddens.items():
            head_dtype = next(head.parameters()).dtype
            plans[name] = head(
                image_hidden.to(dtype=head_dtype),
                hidden.to(dtype=head_dtype),
            ).float()
        return plans

    def predict_semantic_plan(self, **inputs: Any) -> torch.Tensor:
        if self.plan_head_type == "lingbot_dino" and getattr(
            self, "use_current_alignment", False
        ):
            return self.predict_current_future_plans(**inputs)["future_dino"]
        image_hidden, plan_hidden = self._forward_hiddens(**inputs)
        head_dtype = next(self.plan_head.parameters()).dtype
        if self.plan_head_type == "lingbot_dino":
            if self.depth_head is not None:
                plan_hidden, _ = self.split_lingbot_query_hidden(plan_hidden)
            return self.plan_head(
                image_hidden.to(dtype=head_dtype), plan_hidden.to(dtype=head_dtype)
            ).float()
        return self.plan_head(plan_hidden.to(dtype=head_dtype)).float()

    def compute_plan_losses(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        eps = 1e-6
        mse = F.mse_loss(pred, target)
        cosine = 1.0 - F.cosine_similarity(pred.flatten(0, 1), target.flatten(0, 1), dim=-1).mean()

        pred_norm_B_L = pred.norm(dim=-1)
        target_norm_B_L = target.norm(dim=-1)
        norm_loss = ((pred_norm_B_L - target_norm_B_L) / (target_norm_B_L + eps)).pow(2).mean()

        # Per-sample dispersion of tokens around their mean: collapses to 0 when the head
        # predicts the (conditional) mean at every token.
        pred_disp_B = (pred - pred.mean(dim=1, keepdim=True)).norm(dim=-1).mean(dim=1)
        target_disp_B = (target - target.mean(dim=1, keepdim=True)).norm(dim=-1).mean(dim=1)
        variance_loss = ((pred_disp_B - target_disp_B) / (target_disp_B + eps)).pow(2).mean()

        infonce = pred.new_zeros(())
        if self.infonce_loss_weight > 0:
            pred_dir = F.normalize(pred, dim=-1)
            target_dir = F.normalize(target, dim=-1)
            logits_B_L_L = torch.bmm(pred_dir, target_dir.transpose(1, 2)) / self.infonce_temperature
            labels_L = torch.arange(pred.shape[1], device=pred.device)
            labels = labels_L.unsqueeze(0).expand(pred.shape[0], -1).reshape(-1)
            infonce = 0.5 * (
                F.cross_entropy(logits_B_L_L.flatten(0, 1), labels)
                + F.cross_entropy(logits_B_L_L.transpose(1, 2).flatten(0, 1), labels)
            )

        loss = (
            self.mse_loss_weight * mse
            + self.cosine_loss_weight * cosine
            + self.norm_loss_weight * norm_loss
            + self.variance_loss_weight * variance_loss
            + self.infonce_loss_weight * infonce
        )

        with torch.no_grad():
            pred_dir = F.normalize(pred, dim=-1)
            target_dir = F.normalize(target, dim=-1)
            sims_B_L_L = torch.bmm(pred_dir, target_dir.transpose(1, 2))
            hits = sims_B_L_L.argmax(dim=-1) == torch.arange(pred.shape[1], device=pred.device)
            token_retrieval = hits.float().mean()

        return {
            "loss": loss,
            "mse": mse.detach(),
            "cosine_loss": cosine.detach(),
            "norm_loss": norm_loss.detach(),
            "variance_loss": variance_loss.detach(),
            "infonce_loss": infonce.detach(),
            "pred_norm": pred_norm_B_L.mean().detach(),
            "target_norm": target_norm_B_L.mean().detach(),
            "norm_ratio": (pred_norm_B_L.mean() / target_norm_B_L.mean().clamp_min(eps)).detach(),
            "token_disp_ratio": (pred_disp_B.mean() / target_disp_B.mean().clamp_min(eps)).detach(),
            "token_retrieval_top1": token_retrieval,
        }

    def forward(
        self,
        semantic_plan_labels: torch.Tensor,
        depth_plan_labels: torch.Tensor | None = None,
        current_dino_labels: torch.Tensor | None = None,
        current_depth_labels: torch.Tensor | None = None,
        **inputs: Any,
    ) -> dict[str, torch.Tensor]:
        batch, target_len, _ = semantic_plan_labels.shape
        if target_len != self.target_len:
            raise RuntimeError(f"Batch has {target_len} target tokens, wrapper expects {self.target_len}")
        if self.plan_head_type == "lingbot_dino":
            if getattr(self, "use_current_alignment", False):
                if current_dino_labels is None or current_depth_labels is None:
                    raise ValueError(
                        "current alignment requires current DINO and depth labels"
                    )
                if depth_plan_labels is None:
                    raise ValueError("current alignment requires future depth labels")
                plans = self.predict_current_future_plans(**inputs)
                targets = {
                    "current_dino": current_dino_labels.to(
                        device=plans["current_dino"].device, dtype=torch.float32
                    ),
                    "future_dino": semantic_plan_labels.to(
                        device=plans["future_dino"].device, dtype=torch.float32
                    ),
                    "current_depth": current_depth_labels.to(
                        device=plans["current_depth"].device, dtype=torch.float32
                    ),
                    "future_depth": depth_plan_labels.to(
                        device=plans["future_depth"].device, dtype=torch.float32
                    ),
                }
                out = self.compute_current_future_losses(plans, targets)
                with torch.no_grad():
                    for name in (
                        "current_dino",
                        "future_dino",
                        "current_depth",
                        "future_depth",
                    ):
                        out[f"{name}_norm_ratio"] = (
                            plans[name].norm(dim=-1).mean()
                            / targets[name].norm(dim=-1).mean().clamp_min(1e-6)
                        ).detach()
                return out
            # One VLM forward feeds BOTH the video head and (optionally) the auxiliary depth head.
            image_hidden, plan_hidden = self._forward_hiddens(**inputs)
            dino_hidden = plan_hidden
            depth_hidden = None
            if self.depth_head is not None:
                dino_hidden, depth_hidden = self.split_lingbot_query_hidden(plan_hidden)
            dino_dtype = next(self.plan_head.parameters()).dtype
            pred = self.plan_head(
                image_hidden.to(dino_dtype),
                dino_hidden.to(dino_dtype),
            ).float()
            if pred.shape[0] != batch or pred.shape[1] != self.target_len:
                raise RuntimeError(f"Prediction {tuple(pred.shape[:2])} != ({batch}, {self.target_len})")
            out = self.compute_plan_losses(pred, semantic_plan_labels.to(device=pred.device, dtype=torch.float32))
            if self.depth_head is not None and depth_plan_labels is not None:
                depth_dtype = next(self.depth_head.parameters()).dtype
                depth_pred = self.depth_head(
                    image_hidden.to(depth_dtype),
                    depth_hidden.to(depth_dtype),
                ).float()
                depth_target = depth_plan_labels.to(device=pred.device, dtype=torch.float32)
                depth_l = F.smooth_l1_loss(depth_pred, depth_target)  # lingbot depth loss (_emb_loss)
                out["loss"] = out["loss"] + self.depth_loss_weight * depth_l
                out["depth_smooth_l1"] = depth_l.detach()
                out["depth_norm_ratio"] = (
                    depth_pred.norm(dim=-1).mean() / depth_target.norm(dim=-1).mean().clamp_min(1e-6)
                ).detach()
            return out
        pred = self.predict_semantic_plan(**inputs)
        if pred.shape[0] != batch:
            raise RuntimeError(f"Prediction batch {pred.shape[0]} does not match labels batch {batch}")
        if pred.shape[1] != self.target_len:
            raise RuntimeError(f"Prediction has {pred.shape[1]} tokens, expected target_len {self.target_len}")
        target = semantic_plan_labels.to(device=pred.device, dtype=torch.float32)
        return self.compute_plan_losses(pred, target)


def apply_lora(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    if args.lora_r <= 0:
        return model
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def set_trainable(model: nn.Module, *, freeze_vision: bool, freeze_lm_head: bool, full_finetune: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(full_finetune)
    if freeze_vision and hasattr(model, "visual"):
        for p in model.visual.parameters():
            p.requires_grad_(False)
    if freeze_lm_head and hasattr(model, "lm_head"):
        for p in model.lm_head.parameters():
            p.requires_grad_(False)


def register_plan_rows_hook(param: nn.Parameter, row_ids: list[int]) -> None:
    if param.ndim != 2:
        raise RuntimeError(f"Expected embedding weight with ndim=2, got {tuple(param.shape)}")
    row_mask = torch.zeros((param.shape[0], 1), dtype=torch.float32)
    for row_id in row_ids:
        if row_id < 0 or row_id >= param.shape[0]:
            raise RuntimeError(f"Plan token id {row_id} outside embedding rows {param.shape[0]}")
        row_mask[row_id] = 1.0

    def hook(grad: torch.Tensor) -> torch.Tensor:
        return grad * row_mask.to(device=grad.device, dtype=grad.dtype)

    param.requires_grad_(True)
    param.register_hook(hook)


def build_optimizer(wrapper: PlannerWrapper, args: argparse.Namespace) -> torch.optim.Optimizer:
    head_names = (
        "plan_head",
        "depth_head",
        "current_plan_head",
        "current_depth_head",
    )
    head_params = []
    for name in head_names:
        head = getattr(wrapper, name, None)
        if head is not None:
            head_params.extend(p for p in head.parameters() if p.requires_grad)
    other_params = [
        p
        for n, p in wrapper.named_parameters()
        if p.requires_grad
        and not any(n.startswith(f"{name}.") for name in head_names)
    ]
    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": args.head_lr, "weight_decay": args.weight_decay})
    return torch.optim.AdamW(groups)


def build_scheduler(
    optimizer: torch.optim.Optimizer, args: argparse.Namespace
) -> torch.optim.lr_scheduler.LambdaLR | None:
    """Linear warmup to the base LR, then cosine decay to ``min_lr_ratio`` of it.

    ``--warmup-steps`` was previously parsed but never consumed, leaving the run
    at a constant LR with no warmup. The multiplier is shared across param groups
    so the backbone (--lr) and plan head (--head-lr) decay proportionally.
    """
    if args.lr_schedule == "none":
        return None
    warmup = max(0, int(args.warmup_steps))
    total = max(1, int(args.max_steps))
    min_ratio = float(args.min_lr_ratio)

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if args.lr_schedule == "constant" or total <= warmup:
            return 1.0
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def count_trainable_parameters(module: nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in module.parameters():
        numel = param.numel()
        total += numel
        if param.requires_grad:
            trainable += numel
    return trainable, total


def validate_fastwam_export_contract(
    module: PlannerWrapper,
    args: argparse.Namespace,
) -> list[int]:
    """Reject any checkpoint geometry that the production FastWAM provider cannot load."""
    try:
        offsets = keyframe_offsets(
            sequence_length=int(args.sequence_length),
            n=int(args.num_keyframes),
            scheme=str(args.keyframe_scheme),
            gamma=float(args.keyframe_gamma),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            f"FastWAM planner export contract has invalid keyframe geometry: {error}"
        ) from error

    plan_token_ids = getattr(module, "plan_token_ids", ())
    try:
        plan_token_count = len(plan_token_ids)
        unique_plan_token_count = len({int(token_id) for token_id in plan_token_ids})
    except (TypeError, ValueError):
        plan_token_count = -1
        unique_plan_token_count = -1

    actual = {
        "use_depth": bool(getattr(args, "use_depth", False)),
        "sequence_length": getattr(args, "sequence_length", None),
        "num_keyframes": getattr(args, "num_keyframes", None),
        "wrapper_num_keyframes": getattr(module, "num_keyframes", None),
        "keyframe_scheme": str(getattr(args, "keyframe_scheme", "")),
        "keyframe_offsets": [int(offset) for offset in offsets],
        "grid_size": getattr(args, "grid_size", None),
        "semantic_dim": getattr(args, "semantic_dim", None),
        "depth_grid_size": getattr(args, "depth_grid_size", None),
        "depth_dim": getattr(args, "depth_dim", None),
        "target_len": getattr(module, "target_len", None),
        "shared_latent_per_keyframe": getattr(
            module, "shared_latent_per_keyframe", None
        ),
        "private_latent_per_keyframe": getattr(
            module, "private_latent_per_keyframe", None
        ),
        "branch_latent_per_keyframe": getattr(
            module, "branch_latent_per_keyframe", None
        ),
        "total_unique_latent_per_keyframe": getattr(
            module, "total_unique_latent_per_keyframe", None
        ),
        "num_latent_per_keyframe": getattr(
            module, "num_latent_per_keyframe", None
        ),
        "latent_len": getattr(module, "latent_len", None),
        "plan_token_count": plan_token_count,
        "unique_plan_token_count": unique_plan_token_count,
        "plan_head_type": getattr(module, "plan_head_type", None),
        "has_depth_head": getattr(module, "depth_head", None) is not None,
        "use_current_alignment": bool(
            getattr(module, "use_current_alignment", False)
        ),
        "num_task_tokens": getattr(module, "num_task_tokens", 8),
        "has_current_plan_head": getattr(module, "current_plan_head", None)
        is not None,
        "has_current_depth_head": getattr(module, "current_depth_head", None)
        is not None,
        "independent_modality_task_tokens": bool(
            getattr(module, "independent_modality_task_tokens", False)
        ),
    }
    legacy_expected = {
        "use_depth": True,
        "sequence_length": 9,
        "num_keyframes": 4,
        "wrapper_num_keyframes": 4,
        "keyframe_scheme": "even_future",
        "keyframe_offsets": [2, 4, 6, 8],
        "grid_size": 16,
        "semantic_dim": 1024,
        "depth_grid_size": 16,
        "depth_dim": 1024,
        "target_len": 1024,
        "shared_latent_per_keyframe": 32,
        "private_latent_per_keyframe": 32,
        "branch_latent_per_keyframe": 64,
        "total_unique_latent_per_keyframe": 96,
        "num_latent_per_keyframe": 64,
        "latent_len": 384,
        "plan_token_count": 384,
        "unique_plan_token_count": 384,
        "plan_head_type": "lingbot_dino",
        "has_depth_head": True,
        "use_current_alignment": False,
        "num_task_tokens": 8,
        "has_current_plan_head": False,
        "has_current_depth_head": False,
        "independent_modality_task_tokens": False,
    }
    configured_task_tokens = actual["num_task_tokens"]
    expected_task_tokens = (
        configured_task_tokens
        if type(configured_task_tokens) is int and configured_task_tokens > 0
        else 8
    )
    current_expected = {
        "use_depth": True,
        "sequence_length": 9,
        "num_keyframes": 1,
        "wrapper_num_keyframes": 1,
        "keyframe_scheme": "even_future",
        "keyframe_offsets": [8],
        "grid_size": 16,
        "semantic_dim": 1024,
        "depth_grid_size": 16,
        "depth_dim": 1024,
        "target_len": 256,
        "shared_latent_per_keyframe": 32,
        "private_latent_per_keyframe": 32,
        "branch_latent_per_keyframe": expected_task_tokens,
        "total_unique_latent_per_keyframe": 2 * expected_task_tokens,
        "num_latent_per_keyframe": expected_task_tokens,
        "latent_len": 2 * expected_task_tokens,
        "plan_token_count": 2 * expected_task_tokens,
        "unique_plan_token_count": 2 * expected_task_tokens,
        "plan_head_type": "lingbot_dino",
        "has_depth_head": True,
        "use_current_alignment": True,
        "num_task_tokens": expected_task_tokens,
        "has_current_plan_head": True,
        "has_current_depth_head": True,
        "independent_modality_task_tokens": False,
    }
    independent_current_expected = {
        **current_expected,
        "total_unique_latent_per_keyframe": 4 * expected_task_tokens,
        "latent_len": 4 * expected_task_tokens,
        "plan_token_count": 4 * expected_task_tokens,
        "unique_plan_token_count": 4 * expected_task_tokens,
        "independent_modality_task_tokens": True,
    }
    if actual["use_current_alignment"]:
        expected = (
            independent_current_expected
            if actual["independent_modality_task_tokens"]
            else current_expected
        )
    else:
        expected = legacy_expected
    mismatches = [
        f"{name}={actual[name]!r} (expected {expected_value!r})"
        for name, expected_value in expected.items()
        if actual[name] != expected_value
    ]
    if mismatches:
        raise ValueError(
            "FastWAM planner export contract mismatch: " + "; ".join(mismatches)
        )
    return offsets


def save_checkpoint(
    output_dir: Path,
    step: int,
    wrapper: PlannerWrapper,
    processor: Any,
    args: argparse.Namespace,
    rank: int,
) -> None:
    if not is_main(rank):
        return
    module = wrapper
    fastwam_data_config = getattr(args, "fastwam_data_config", None)
    fastwam_offsets = None
    if fastwam_data_config is not None:
        fastwam_offsets = validate_fastwam_export_contract(module, args)
    depth_head = getattr(module, "depth_head", None)
    current_plan_head = getattr(module, "current_plan_head", None)
    current_depth_head = getattr(module, "current_depth_head", None)
    ckpt = output_dir / f"step_{step:06d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    module.model.save_pretrained(ckpt / "qwen3vl_lora_or_model")
    processor.save_pretrained(ckpt / "processor")
    torch.save(module.plan_head.state_dict(), ckpt / "plan_head.pt")
    if depth_head is not None:
        torch.save(depth_head.state_dict(), ckpt / "depth_head.pt")
    if current_plan_head is not None:
        torch.save(current_plan_head.state_dict(), ckpt / "current_plan_head.pt")
    if current_depth_head is not None:
        torch.save(current_depth_head.state_dict(), ckpt / "current_depth_head.pt")
    plan_ids = torch.as_tensor(module.plan_token_ids)
    plan_embedding = module.model.get_input_embeddings().weight[plan_ids].detach().cpu()
    torch.save(plan_embedding, ckpt / "plan_token_embedding.pt")
    feature_type = str(getattr(args, "sample_feature_type", "unknown"))
    summary_path = args.plan_label_dir / "summary.json" if args.plan_label_dir else None
    if summary_path is not None and summary_path.exists():
        try:
            feature_type = json.loads(summary_path.read_text()).get("feature_type", feature_type)
        except Exception:
            feature_type = feature_type
    meta = {
        "step": step,
        "plan_token": PLAN_TOKEN,
        "plan_token_ids": module.plan_token_ids,
        "num_keyframes": args.num_keyframes,
        "grid_size": args.grid_size,
        "semantic_dim": args.semantic_dim,
        "model_path": str(args.model_path),
        "objective": "continuous_semantic_blueprint_regression",
        "feature_type": feature_type,
        "plan_label_dir": str(args.plan_label_dir) if args.plan_label_dir else None,
        "sample_one_window_per_stem": bool(args.sample_one_window_per_stem),
        "online_plan_labels": bool(args.online_plan_labels),
        "keyframe_scheme": str(args.keyframe_scheme),
        "keyframe_gamma": float(args.keyframe_gamma),
        "sequence_length": int(args.sequence_length),
        "online_grid_size": int(args.online_grid_size),
        "siglip2_encoder_path": str(args.siglip2_encoder_path) if args.siglip2_encoder_path else None,
        "frame_ranges_json": str(args.frame_ranges_json) if args.frame_ranges_json else None,
        "plan_head_type": module.plan_head_type,
        "plan_head_num_heads": int(module.plan_head_num_heads),
        "plan_head_dropout": float(module.plan_head_dropout),
        "num_latent_per_keyframe": int(module.num_latent_per_keyframe),
        "latent_len": int(module.latent_len),
        "target_len": int(module.target_len),
        "sem_mlp_hidden_size": int(module.sem_mlp_hidden_size),
        "mse_loss_weight": float(module.mse_loss_weight),
        "cosine_loss_weight": float(module.cosine_loss_weight),
        "norm_loss_weight": float(module.norm_loss_weight),
        "variance_loss_weight": float(module.variance_loss_weight),
        "infonce_loss_weight": float(module.infonce_loss_weight),
        "infonce_temperature": float(module.infonce_temperature),
        "train_plan_token_embedding": bool(args.train_plan_token_embedding),
        "full_finetune": bool(args.full_finetune),
        "freeze_vision": bool(args.freeze_vision),
        "freeze_lm_head": bool(args.freeze_lm_head),
    }
    offsets = fastwam_offsets or keyframe_offsets(
        sequence_length=int(args.sequence_length),
        n=int(args.num_keyframes),
        scheme=str(args.keyframe_scheme),
        gamma=float(args.keyframe_gamma),
    )
    meta.update(
        {
            "sequence_length": int(args.sequence_length),
            "num_keyframes": int(args.num_keyframes),
            "grid_size": int(args.grid_size),
            "semantic_dim": int(args.semantic_dim),
            "target_tokens": int(module.target_len),
            "keyframe_offsets": [int(offset) for offset in offsets],
            "normalized_keyframe_times": [
                float(offset) / float(args.sequence_length - 1)
                for offset in offsets
            ],
            "has_depth_head": depth_head is not None,
            "depth_feature_dim": (
                int(args.depth_dim) if depth_head is not None else None
            ),
            "depth_grid_size": int(args.depth_grid_size),
            "depth_loss_weight": float(module.depth_loss_weight),
            "use_current_alignment": bool(
                getattr(module, "use_current_alignment", False)
            ),
            "num_task_tokens": int(getattr(module, "num_task_tokens", 8)),
            "independent_modality_task_tokens": bool(
                getattr(module, "independent_modality_task_tokens", False)
            ),
            "current_dino_loss_weight": float(
                getattr(module, "current_dino_loss_weight", 0.004)
            ),
            "future_dino_loss_weight": float(
                getattr(module, "future_dino_loss_weight", 0.004)
            ),
            "current_depth_loss_weight": float(
                getattr(module, "current_depth_loss_weight", 0.004)
            ),
            "future_depth_loss_weight": float(
                getattr(module, "future_depth_loss_weight", 0.004)
            ),
            "shared_latent_per_keyframe": int(
                module.shared_latent_per_keyframe
            ),
            "private_latent_per_keyframe": int(
                module.private_latent_per_keyframe
            ),
            "branch_latent_per_keyframe": int(
                module.branch_latent_per_keyframe
            ),
            "total_unique_latent_per_keyframe": int(
                module.total_unique_latent_per_keyframe
            ),
            "query_layout": (
                (
                    f"current_dino_{module.num_task_tokens}_then_"
                    f"future_dino_{module.num_task_tokens}_then_"
                    f"current_depth_{module.num_task_tokens}_then_"
                    f"future_depth_{module.num_task_tokens}"
                )
                if getattr(module, "independent_modality_task_tokens", False)
                else (
                    f"current_{module.num_task_tokens}_then_"
                    f"future_{module.num_task_tokens}__"
                    "dino_depth_shared_within_time"
                )
                if getattr(module, "use_current_alignment", False)
                else (
                    "keyframe_major__shared_dino_private_depth_private"
                    if depth_head is not None
                    else "keyframe_major__legacy_single_branch"
                )
            ),
            "plan_token_strings": [
                f"<|sem_plan_{index}|>" for index in range(module.latent_len)
            ],
            "token_order": "keyframe_major_row_major",
            "planner_input_frame": (
                "fastwam_current_multicamera_composite"
                if fastwam_data_config is not None
                else "legacy_single_current_frame"
            ),
        }
    )
    (ckpt / "planner_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "latest_checkpoint.txt").write_text(str(ckpt), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.full_finetune and args.lora_r > 0:
        raise ValueError("--full-finetune is mutually exclusive with LoRA; set --lora-r 0.")
    if args.plan_head_type == "lingbot_dino" and args.use_depth and not args.use_current_alignment and (
        args.shared_latent_per_keyframe <= 0 or args.private_latent_per_keyframe <= 0
    ):
        raise ValueError(
            "DINO+depth mode requires positive --shared-latent-per-keyframe and "
            "--private-latent-per-keyframe"
        )
    if args.fastwam_data_config is not None:
        preflight_fastwam_data_config(
            args.fastwam_data_config,
            dataset_dirs=args.fastwam_dataset_dir,
            text_embedding_cache_dir=args.fastwam_text_embedding_cache_dir,
            pretrained_norm_stats=args.fastwam_pretrained_norm_stats,
        )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    accelerator = build_accelerator(grad_accum=args.grad_accum, dtype=args.dtype)
    runtime_contract = validate_runtime_contract(
        accelerator,
        per_device_batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        expected_global_batch=args.expected_global_batch,
    )
    rank = int(accelerator.process_index)
    world = int(accelerator.num_processes)
    device = accelerator.device
    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    "distributed_type": runtime_contract.distributed_type,
                    "world_size": runtime_contract.world_size,
                    "batch_size_per_gpu": runtime_contract.per_device_batch_size,
                    "gradient_accumulation_steps": runtime_contract.gradient_accumulation_steps,
                    "global_batch_size": runtime_contract.global_batch_size,
                    "zero_stage": runtime_contract.zero_stage,
                    "dtype": args.dtype,
                    "gradient_checkpointing": bool(args.gradient_checkpointing),
                }
            ),
            flush=True,
        )
    accelerator.wait_for_everyone()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    model, processor = load_qwen3vl_model_and_processor(
        args.model_path,
        device=None,
        dtype=args.dtype,
        attn_implementation="sdpa",
        local_files_only=True,
        eval_mode=False,
    )
    # DIAL-style distinct latent tokens (gr00t <|bridge_i|>): for the CoVT bottleneck the LM
    # emits a small, structured set of latents (num_keyframes x num_latent_per_keyframe), so
    # give each position its OWN learnable token/embedding instead of repeating one
    # <|sem_plan|> — identical inputs differentiated by position alone tend to collapse
    # (all latents become the same -> keyframes stop evolving). mlp/baton keep the single
    # repeated token (their latent_len = target_len = thousands of tokens).
    if args.plan_head_type == "lingbot_dino" and args.use_current_alignment:
        latent_len = (
            4 if args.independent_modality_task_tokens else 2
        ) * int(args.num_task_tokens)
        plan_token_strs = [f"<|sem_plan_{i}|>" for i in range(latent_len)]
        plan_sequence = list(plan_token_strs)
    elif (
        args.plan_head_type == "lingbot_dino"
        and args.use_depth
        and args.shared_latent_per_keyframe > 0
    ):
        latent_len = int(args.num_keyframes) * (
            int(args.shared_latent_per_keyframe)
            + 2 * int(args.private_latent_per_keyframe)
        )
        plan_token_strs = [f"<|sem_plan_{i}|>" for i in range(latent_len)]
        plan_sequence = list(plan_token_strs)
    elif args.plan_head_type in ("covt", "lingbot_dino"):
        latent_len = int(args.num_keyframes) * int(args.num_latent_per_keyframe)
        plan_token_strs = [f"<|sem_plan_{i}|>" for i in range(latent_len)]
        plan_sequence = list(plan_token_strs)
    else:
        plan_token_strs = [PLAN_TOKEN]
        plan_sequence = [PLAN_TOKEN] * (int(args.num_keyframes) * int(args.grid_size) * int(args.grid_size))
    new_tokens = [t for t in plan_token_strs if t not in processor.tokenizer.get_vocab()]
    if new_tokens:
        processor.tokenizer.add_tokens(new_tokens, special_tokens=True)
        model.resize_token_embeddings(len(processor.tokenizer))
    plan_token_ids = [processor.tokenizer.convert_tokens_to_ids(t) for t in plan_token_strs]
    model.config.use_cache = False
    configure_gradient_checkpointing(model, enabled=args.gradient_checkpointing)
    set_trainable(
        model,
        freeze_vision=args.freeze_vision,
        freeze_lm_head=args.freeze_lm_head,
        full_finetune=args.full_finetune,
    )
    if args.train_plan_token_embedding:
        embed_weight = model.get_input_embeddings().weight
        if not embed_weight.requires_grad:
            # Covers both LoRA (backbone frozen) and full-FT with tied embeddings
            # (Qwen3-VL-2B has tie_word_embeddings=true, so freezing lm_head also
            # froze the input embeddings). Re-enable ONLY the plan-token rows via a
            # masked grad hook; with no LM text loss the lm_head side gets no grad.
            register_plan_rows_hook(embed_weight, plan_token_ids)
    model = apply_lora(model, args)

    sig_model = sig_processor = sig_encode = None
    dino_encoder = None
    depth_encoder = None
    if args.plan_head_type == "lingbot_dino":
        if not args.online_plan_labels:
            raise ValueError("lingbot_dino requires --online-plan-labels (online DINO-video targets)")
        if args.frame_ranges_json is None and args.fastwam_data_config is None:
            args.frame_ranges_json = args.dataset_root / "frame_ranges.json"
        dino_encoder = DinoVideoTargetEncoder(
            ckpt_path=args.dino_teacher_ckpt,
            config_path=args.dino_teacher_config,
            input_size=args.dino_input_size,
            device=device,
        )
        args.sample_feature_type = "dino_video"
        if args.semantic_dim <= 0:
            args.semantic_dim = 1024
        if args.use_depth:
            if args.depth_moge_path is None or args.depth_morgbd_path is None:
                raise ValueError("--use-depth requires --depth-moge-path and --depth-morgbd-path")
            depth_encoder = DepthTargetEncoder(
                moge_path=args.depth_moge_path,
                morgbd_path=args.depth_morgbd_path,
                input_size=args.depth_input_size,
                num_tokens=args.depth_grid_size * args.depth_grid_size,
                device=device,
            )
    elif args.online_plan_labels:
        from build_siglip2_semantic_plan_labels import encode_images, load_siglip2, semantic_feature_type

        if args.siglip2_encoder_path is None:
            raise ValueError("--online-plan-labels requires --siglip2-encoder-path")
        if args.frame_ranges_json is None and args.fastwam_data_config is None:
            args.frame_ranges_json = args.dataset_root / "frame_ranges.json"
        sig_model, sig_processor = load_siglip2(args.siglip2_encoder_path, device, "bf16")
        sig_model.requires_grad_(False)
        sig_encode = encode_images
        # Probe dim + tokens-per-keyframe with a dummy frame; validates the head grid early.
        probe = sig_encode(sig_model, sig_processor, [Image.new("RGB", (64, 64))], args.online_grid_size, device, torch.float32)
        if probe.shape[1] != args.grid_size * args.grid_size:
            raise ValueError(
                f"online grid {args.online_grid_size} yields {probe.shape[1]} tokens/keyframe, "
                f"but --grid-size {args.grid_size} expects {args.grid_size * args.grid_size}"
            )
        args.sample_feature_type = semantic_feature_type(args.online_grid_size)
        if args.semantic_dim <= 0:
            args.semantic_dim = int(probe.shape[-1])
    else:
        if args.plan_label_dir is None:
            raise ValueError("--plan-label-dir is required unless --online-plan-labels is set")
        # Probe a sample label for semantic dim + feature type (the latter is recorded in
        # planner_meta.json; the old summary.json path never exists for these label dirs).
        first = next(iter(sorted(args.plan_label_dir.glob("*.pt"))), None)
        if first is None:
            raise RuntimeError(f"No .pt labels under {args.plan_label_dir}")
        payload = torch.load(first, map_location="cpu", weights_only=False)
        args.sample_feature_type = str(payload.get("feature_type", "unknown"))
        if args.semantic_dim <= 0:
            args.semantic_dim = int(payload["semantic_plan"].shape[-1])

    hidden_size = int(model.config.text_config.hidden_size)
    wrapper = PlannerWrapper(
        model=model,
        hidden_size=hidden_size,
        semantic_dim=args.semantic_dim,
        plan_token_ids=plan_token_ids,
        target_len=args.num_keyframes * args.grid_size * args.grid_size,
        num_keyframes=args.num_keyframes,
        grid_size=args.grid_size,
        num_latent_per_keyframe=args.num_latent_per_keyframe,
        shared_latent_per_keyframe=args.shared_latent_per_keyframe,
        private_latent_per_keyframe=args.private_latent_per_keyframe,
        plan_head_type=args.plan_head_type,
        plan_head_num_heads=args.plan_head_num_heads,
        plan_head_dropout=args.plan_head_dropout,
        sem_mlp_hidden_size=args.sem_mlp_hidden_size,
        mse_loss_weight=args.mse_loss_weight,
        cosine_loss_weight=args.cosine_loss_weight,
        norm_loss_weight=args.norm_loss_weight,
        variance_loss_weight=args.variance_loss_weight,
        infonce_loss_weight=args.infonce_loss_weight,
        infonce_temperature=args.infonce_temperature,
        use_depth=args.use_depth,
        depth_dim=args.depth_dim,
        depth_grid_size=args.depth_grid_size,
        depth_loss_weight=args.depth_loss_weight,
        use_current_alignment=args.use_current_alignment,
        independent_modality_task_tokens=args.independent_modality_task_tokens,
        num_task_tokens=args.num_task_tokens,
        current_dino_loss_weight=args.current_dino_loss_weight,
        future_dino_loss_weight=args.future_dino_loss_weight,
        current_depth_loss_weight=args.current_depth_loss_weight,
        future_depth_loss_weight=args.future_depth_loss_weight,
    )
    if args.plan_head_type == "lingbot_dino" and args.head_warmstart_ckpt is not None:
        head_state = _load_lingbot_head_state(args.head_warmstart_ckpt)
        report = wrapper.plan_head.load_lingbot_warmstart(head_state, head_name="future_video_align_head")
        if wrapper.depth_head is not None:
            depth_report = wrapper.depth_head.load_lingbot_warmstart(head_state, head_name="future_depth_align_head")
        current_report = None
        current_depth_report = None
        if wrapper.current_plan_head is not None:
            current_report = wrapper.current_plan_head.load_lingbot_warmstart(
                head_state, head_name="current_video_align_head"
            )
        if wrapper.current_depth_head is not None:
            current_depth_report = wrapper.current_depth_head.load_lingbot_warmstart(
                head_state, head_name="depth_align_head"
            )
        if accelerator.is_main_process:
            print(json.dumps({"head_warmstart": report}), flush=True)
            if wrapper.depth_head is not None:
                print(json.dumps({"depth_head_warmstart": depth_report}), flush=True)
            if current_report is not None:
                print(json.dumps({"current_head_warmstart": current_report}), flush=True)
            if current_depth_report is not None:
                print(json.dumps({"current_depth_head_warmstart": current_depth_report}), flush=True)
    wrapper.to(device)
    wrapper.train()
    if accelerator.is_main_process:
        trainable, total = count_trainable_parameters(wrapper)
        print(
            json.dumps(
                {
                    "full_finetune": bool(args.full_finetune),
                    "lora_r": int(args.lora_r),
                    "freeze_vision": bool(args.freeze_vision),
                    "freeze_lm_head": bool(args.freeze_lm_head),
                    "plan_head_type": args.plan_head_type,
                    "plan_head_num_heads": int(args.plan_head_num_heads),
                    "sample_one_window_per_stem": bool(args.sample_one_window_per_stem),
                    "trainable_params": trainable,
                    "total_params": total,
                    "trainable_ratio": trainable / max(total, 1),
                }
            ),
            flush=True,
        )

    if args.online_plan_labels:
        if args.fastwam_data_config is not None:
            dataset = FastWAMOnlinePlannerDataset.from_config(
                args.fastwam_data_config,
                dataset_dirs=args.fastwam_dataset_dir,
                text_embedding_cache_dir=(
                    args.fastwam_text_embedding_cache_dir
                ),
                pretrained_norm_stats=args.fastwam_pretrained_norm_stats,
                max_samples=args.max_samples,
                offsets=keyframe_offsets(
                    args.sequence_length,
                    args.num_keyframes,
                    args.keyframe_scheme,
                    args.keyframe_gamma,
                ),
            )
        else:
            dataset = OnlineSemanticPlanDataset(
                dataset_root=args.dataset_root,
                frame_ranges_json=args.frame_ranges_json,
                num_keyframes=args.num_keyframes,
                sequence_length=args.sequence_length,
                keyframe_scheme=args.keyframe_scheme,
                keyframe_gamma=args.keyframe_gamma,
                max_samples=args.max_samples,
                seed=args.seed,
            )
        if accelerator.is_main_process:
            print(
                json.dumps(
                    {
                        "online_plan_labels": True,
                        "stems": len(dataset),
                        "keyframe_scheme": args.keyframe_scheme,
                        "keyframe_offsets": dataset.offsets,
                        "sequence_length": args.sequence_length,
                        "feature_type": args.sample_feature_type,
                    }
                ),
                flush=True,
            )
    else:
        dataset = SemanticPlanDataset(
            dataset_root=args.dataset_root,
            label_dir=args.plan_label_dir,
            num_keyframes=args.num_keyframes,
            grid_size=args.grid_size,
            max_samples=args.max_samples,
            sample_one_window_per_stem=args.sample_one_window_per_stem,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=Collator(
            processor=processor,
            plan_sequence=plan_sequence,
        ),
        pin_memory=True,
    )

    optim = build_optimizer(wrapper, args)
    scheduler = build_scheduler(optim, args)
    wrapper, optim, loader = accelerator.prepare(wrapper, optim, loader)
    wandb_run = None
    if accelerator.is_main_process and os.environ.get("PLANNER_WANDB", "1") == "1":
        try:
            import wandb

            wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "qwen3vl_semantic_planner"),
                name=os.environ.get("WANDB_NAME", args.output_dir.name),
                dir=str(args.output_dir),
                mode=os.environ.get("WANDB_MODE", "offline"),
                config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            )
        except Exception as exc:  # missing wandb must not kill training; JSON stdout remains
            print(f"wandb disabled: {exc}", flush=True)
    step = 0
    micro_step = 0
    running_loss = 0.0
    deepspeed_enabled = is_deepspeed(accelerator)
    if not deepspeed_enabled:
        optim.zero_grad(set_to_none=True)
    pbar = tqdm(
        total=args.max_steps,
        disable=not accelerator.is_local_main_process,
        desc="qwen3vl planner",
    )
    while step < args.max_steps:
        # Advance the dataset epoch too, else sample_one_window_per_stem never resamples the
        # window per stem (the new epoch is picked up when the DataLoader respawns workers).
        dataset.set_epoch(step)
        for batch in loader:
            batch.pop("stems", None)
            keyframes = batch.pop("keyframe_images", None)
            current = batch.pop("current_image", None)
            module = accelerator.unwrap_model(wrapper)
            model_dtype = next(module.model.parameters()).dtype
            batch = move_qwen_inputs_to_device(batch, device, model_dtype=model_dtype)
            if keyframes is not None:
                with torch.no_grad():
                    if dino_encoder is not None:
                        # Online DINO-video targets: teacher over [current, current, keyframe_k] clips
                        # -> [B, K*256, 1024], matching the LingbotDinoPlanHead output.
                        cur = current.permute(0, 3, 1, 2).contiguous()  # (B,3,H,W)
                        kfs = [keyframes[:, j].permute(0, 3, 1, 2).contiguous() for j in range(keyframes.shape[1])]
                        if args.use_current_alignment:
                            current_dino, future_dino = dino_encoder.encode_current_and_future(
                                cur, kfs[0]
                            )
                            batch["current_dino_labels"] = current_dino.float()
                            batch["semantic_plan_labels"] = future_dino.float()
                            current_depth, future_depth = depth_encoder.encode_current_and_future(
                                cur, kfs[0]
                            )
                            batch["current_depth_labels"] = current_depth.float()
                            batch["depth_plan_labels"] = future_depth.float()
                        else:
                            batch["semantic_plan_labels"] = dino_encoder.encode_future_keyframes(cur, kfs).float()
                            if depth_encoder is not None:
                                # LingBot-Depth targets over the SAME future keyframes -> [B, K*256, 1024].
                                batch["depth_plan_labels"] = depth_encoder.encode_future_keyframes(kfs).float()
                    else:
                        # Online SigLIP2 targets (bit-consistent with the offline builder).
                        b, k = keyframes.shape[0], keyframes.shape[1]
                        imgs = [keyframes[i, j].detach().cpu().numpy() for i in range(b) for j in range(k)]
                        target = sig_encode(sig_model, sig_processor, imgs, args.online_grid_size, device, torch.float32)
                        batch["semantic_plan_labels"] = target.reshape(b, k * target.shape[1], target.shape[-1])
            with accumulation_context(accelerator, wrapper):
                out = wrapper(**batch)
                accelerator.backward(out["loss"])
                if not deepspeed_enabled:
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            (
                                parameter
                                for parameter in wrapper.parameters()
                                if parameter.requires_grad
                            ),
                            1.0,
                        )
                    optim.step()
                    optim.zero_grad(set_to_none=True)
            running_loss += float(out["loss"].detach())
            micro_step += 1
            if is_optimizer_update(accelerator, micro_step, args.grad_accum):
                if scheduler is not None:
                    scheduler.step()
                step += 1
                if accelerator.is_main_process:
                    pbar.update(1)
                    if step % args.log_steps == 0:
                        average_loss = running_loss / max(
                            args.log_steps * args.grad_accum,
                            1,
                        )
                        log_entry = {
                            "step": step,
                            "loss": average_loss,
                            "lr": (
                                scheduler.get_last_lr()[0]
                                if scheduler is not None
                                else args.lr
                            ),
                        }
                        log_entry.update(
                            {key: float(value) for key, value in out.items() if key != "loss"}
                        )
                        print(json.dumps(log_entry), flush=True)
                        if wandb_run is not None:
                            wandb_run.log(log_entry, step=step)
                if step % args.log_steps == 0:
                    running_loss = 0.0
                if step % args.save_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_checkpoint(
                            args.output_dir,
                            step,
                            checkpoint_module(accelerator, wrapper),
                            processor,
                            args,
                            rank=0,
                        )
                    accelerator.wait_for_everyone()
                if step >= args.max_steps:
                    break
        if len(loader) == 0:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(
            args.output_dir,
            step,
            checkpoint_module(accelerator, wrapper),
            processor,
            args,
            rank=0,
        )
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        pbar.close()
    if accelerator.is_main_process and wandb_run is not None:
        wandb_run.finish()
    accelerator.end_training()


if __name__ == "__main__":
    main()
