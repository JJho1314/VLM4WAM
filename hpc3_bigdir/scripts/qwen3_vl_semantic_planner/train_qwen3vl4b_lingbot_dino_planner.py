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
from typing import Any

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
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from qwen3vl_wrapper import load_qwen3vl_model_and_processor, move_qwen_inputs_to_device

# 4B-specific modules live in the lingbot_dino_4b/ subpackage; add it to the path (flat import,
# matching how lingbot_dino_head imports lingbot_resampler).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lingbot_dino_4b"))
from lingbot_dino_head import LingbotDinoPlanHead  # noqa: E402
from dino_video_target import DinoVideoTargetEncoder  # noqa: E402
from depth_target import DepthTargetEncoder  # noqa: E402


def _load_lingbot_head_state(src_6b_dir: Path) -> dict:
    """Stream the future-video AND future-depth align heads + query tensors from a lingbot-vla-v2 6b
    checkpoint (one dict warm-starts both heads; each head picks its own keys by marker)."""
    from collections import defaultdict

    from safetensors import safe_open

    src = Path(src_6b_dir)
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    # NOTE the leading dots: lingbot's current-depth head is plain "depth_align_head", which is a
    # substring of "future_depth_align_head" — the dot pins the match to the exact head name
    # (keys look like "model.depth_align_head.projector...").
    markers = (
        ".future_video_align_head.", ".future_depth_align_head.",
        ".current_video_align_head.", ".depth_align_head.",
    )
    suffixes = (
        ".future_video_align_embs", ".future_depth_align_embs",
        ".current_video_align_embs", ".depth_align_embs",
    )
    want = [k for k in index if any(m in k for m in markers) or k.endswith(suffixes)]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
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
    parser.add_argument("--keyframe-scheme", choices=["uniform", "late"], default="uniform")
    # Explicit comma-separated keyframe offsets (e.g. "48" for a single official-lingbot-style
    # future frame at the horizon end). Overrides --keyframe-scheme when set.
    parser.add_argument("--keyframe-offsets", type=str, default="")
    # Official-lingbot CURRENT alignment: extra current-time latent group + current_video /
    # current_depth heads (warm-started from current_video_align_head / depth_align_head).
    parser.add_argument("--use-current", action="store_true")
    parser.add_argument("--current-video-loss-weight", type=float, default=1.0)
    parser.add_argument("--current-depth-loss-weight", type=float, default=0.004)
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
        "--head-warmstart-ckpt", type=Path, default=None,
        help="lingbot-vla-v2-6b dir: warm-start the head from future_video_align_head.*",
    )
    # CoVT-style bottleneck: the LM emits only num-latent-per-keyframe continuous latents per
    # keyframe (a compact "visual thought"), and a decoder reconstructs the dense SigLIP grid
    # from them. Decouples the LM sequence length from the target grid resolution.
    parser.add_argument("--num-latent-per-keyframe", type=int, default=4)
    # lingbot_dino only: per-keyframe latents OWNED by each align head (video / depth), ON TOP of
    # the SHARED --num-latent-per-keyframe group. Layout per keyframe in the LM sequence:
    # [shared | video-own | depth-own(if --use-depth)]. Each head reads [shared + its own] latents.
    # 0 = fully shared (official lingbot behavior).
    parser.add_argument("--num-head-latent-per-keyframe", type=int, default=0)
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
    parser.add_argument("--freeze-vision", action="store_true", default=True)
    parser.add_argument("--no-freeze-vision", action="store_false", dest="freeze_vision")
    parser.add_argument("--freeze-lm-head", action="store_true", default=True)
    parser.add_argument("--no-freeze-lm-head", action="store_false", dest="freeze_lm_head")
    parser.add_argument("--train-plan-token-embedding", action="store_true", default=True)
    parser.add_argument("--no-train-plan-token-embedding", action="store_false", dest="train_plan_token_embedding")
    parser.add_argument("--ddp-find-unused-parameters", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--log-steps", type=int, default=10)
    return parser.parse_args()


def ddp_info() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank
    return 0, 1, 0


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
        offsets_override: list[int] | None = None,
    ) -> None:
        from build_siglip2_semantic_plan_labels import load_frame_ranges

        self.dataset_root = dataset_root
        self.sequence_length = int(sequence_length)
        self.offsets = (
            [int(o) for o in offsets_override]
            if offsets_override
            else keyframe_offsets(self.sequence_length, num_keyframes, keyframe_scheme, keyframe_gamma)
        )
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
        texts = []
        plan_text = " ".join(self.plan_sequence)
        for item in batch:
            user_text = (
                "You are a robot video semantic planner. Given the first frame and instruction, "
                "predict future spatial semantic plan tokens for the manipulation video.\n"
                f"Instruction: {item['prompt']}"
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ],
                },
                {"role": "assistant", "content": plan_text},
            ]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
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
        num_head_latent_per_keyframe: int = 0,
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
        use_current: bool = False,
        current_video_loss_weight: float = 1.0,
        current_depth_loss_weight: float = 0.004,
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
        self.num_latent_per_keyframe = int(num_latent_per_keyframe)
        self.num_head_latent_per_keyframe = int(num_head_latent_per_keyframe)
        self.num_keyframes = int(num_keyframes)
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
            # Latent layout per keyframe: [shared | video-own | depth-own], own groups only when
            # num_head_latent_per_keyframe > 0; each head reads shared + its own group.
            # With use_current (official lingbot parity) ONE extra current-time group is PREPENDED:
            # [current(shared) | future_kf1 | ... | future_kfK].
            spec = self.num_head_latent_per_keyframe
            self.use_current = bool(use_current)
            if self.use_current and spec > 0:
                raise ValueError("--use-current supports only fully-shared latents (num-head-latent-per-keyframe=0)")
            own_groups = (2 if (bool(use_depth) and spec > 0) else (1 if spec > 0 else 0))
            self.per_kf_latents = int(num_latent_per_keyframe) + spec * own_groups
            self.head_latents_per_kf = int(num_latent_per_keyframe) + spec
            self.latent_len = (int(num_keyframes) + (1 if self.use_current else 0)) * self.per_kf_latents
            self.plan_head = LingbotDinoPlanHead(
                num_keyframes=num_keyframes,
                num_latent_per_keyframe=self.head_latents_per_kf,
                num_backbone_tokens=int(grid_size) * int(grid_size),
                llm_hidden=hidden_size,
                dim_out=semantic_dim,
            )
        else:
            raise ValueError(f"Unsupported plan_head_type: {plan_head_type}")
        # Auxiliary future-DEPTH alignment head (lingbot-style, only for lingbot_dino): a second shared
        # TaskTokenResampler, warm-started from future_depth_align_head, reading the SAME image+latent
        # context and regressing LingBot-Depth features (MoGe-2 -> MoRGBD) with smooth_L1.
        self.use_depth = bool(use_depth) and plan_head_type == "lingbot_dino"
        self.depth_loss_weight = float(depth_loss_weight)
        self.depth_head = None
        if self.use_depth:
            self.depth_head = LingbotDinoPlanHead(
                num_keyframes=num_keyframes,
                num_latent_per_keyframe=self.head_latents_per_kf,
                num_backbone_tokens=int(depth_grid_size) * int(depth_grid_size),
                llm_hidden=hidden_size,
                dim_out=depth_dim,
            )
        # Official-lingbot CURRENT alignment heads (aux): same head class with num_keyframes=1,
        # reading the prepended current latent group; warm-started from current_video_align_head /
        # depth_align_head (lingbot's current-depth head is named plain "depth_align_head").
        self.current_video_loss_weight = float(current_video_loss_weight)
        self.current_depth_loss_weight = float(current_depth_loss_weight)
        self.current_plan_head = None
        self.current_depth_head = None
        if getattr(self, "use_current", False) and plan_head_type == "lingbot_dino":
            self.current_plan_head = LingbotDinoPlanHead(
                num_keyframes=1,
                num_latent_per_keyframe=self.head_latents_per_kf,
                num_backbone_tokens=int(grid_size) * int(grid_size),
                llm_hidden=hidden_size,
                dim_out=semantic_dim,
            )
            if self.use_depth:
                self.current_depth_head = LingbotDinoPlanHead(
                    num_keyframes=1,
                    num_latent_per_keyframe=self.head_latents_per_kf,
                    num_backbone_tokens=int(depth_grid_size) * int(depth_grid_size),
                    llm_hidden=hidden_size,
                    dim_out=depth_dim,
                )
        self.plan_token_ids = [int(x) for x in plan_token_ids]
        self.model = model
        # image-token id used by the lingbot_dino head to gather the LLM's image-token hiddens
        self.image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)

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

    def _split_current(self, plan_hidden: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Split off the PREPENDED current-time latent group -> (current latents | None, future latents)."""
        if not getattr(self, "use_current", False):
            return None, plan_hidden
        g = self.per_kf_latents
        return plan_hidden[:, :g], plan_hidden[:, g:]

    def _split_latents(self, plan_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Split grouped lingbot_dino latents into per-head views.

        Per-keyframe layout: [shared | video-own | depth-own(if use_depth)]. Returns
        (video latents, depth latents), each (B, K*(shared+own), H). With no own groups
        (official fully-shared config) both heads read the same tensor."""
        spec = self.num_head_latent_per_keyframe
        if self.plan_head_type != "lingbot_dino" or spec <= 0:
            return plan_hidden, (plan_hidden if self.use_depth else None)
        b, _, h = plan_hidden.shape
        s = self.num_latent_per_keyframe
        lat = plan_hidden.reshape(b, self.num_keyframes, self.per_kf_latents, h)
        video = torch.cat([lat[:, :, :s], lat[:, :, s:s + spec]], dim=2).reshape(b, -1, h)
        depth = None
        if self.use_depth:
            depth = torch.cat([lat[:, :, :s], lat[:, :, s + spec:s + 2 * spec]], dim=2).reshape(b, -1, h)
        return video, depth

    def predict_semantic_plan(self, **inputs: Any) -> torch.Tensor:
        image_hidden, plan_hidden = self._forward_hiddens(**inputs)
        head_dtype = next(self.plan_head.parameters()).dtype
        if self.plan_head_type == "lingbot_dino":
            _, future_hidden = self._split_current(plan_hidden)
            video_lat, _ = self._split_latents(future_hidden)
            return self.plan_head(
                image_hidden.to(dtype=head_dtype), video_lat.to(dtype=head_dtype)
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
        current_video_labels: torch.Tensor | None = None,
        current_depth_labels: torch.Tensor | None = None,
        **inputs: Any,
    ) -> dict[str, torch.Tensor]:
        batch, target_len, _ = semantic_plan_labels.shape
        if target_len != self.target_len:
            raise RuntimeError(f"Batch has {target_len} target tokens, wrapper expects {self.target_len}")
        if self.plan_head_type == "lingbot_dino":
            # One VLM forward feeds the future video/depth heads and (optionally) the current heads.
            image_hidden, plan_hidden = self._forward_hiddens(**inputs)
            head_dtype = next(self.plan_head.parameters()).dtype
            current_lat, future_hidden = self._split_current(plan_hidden)
            video_lat, depth_lat = self._split_latents(future_hidden)
            pred = self.plan_head(image_hidden.to(head_dtype), video_lat.to(head_dtype)).float()
            if pred.shape[0] != batch or pred.shape[1] != self.target_len:
                raise RuntimeError(f"Prediction {tuple(pred.shape[:2])} != ({batch}, {self.target_len})")
            out = self.compute_plan_losses(pred, semantic_plan_labels.to(device=pred.device, dtype=torch.float32))
            if self.depth_head is not None and depth_plan_labels is not None:
                depth_pred = self.depth_head(image_hidden.to(head_dtype), depth_lat.to(head_dtype)).float()
                depth_target = depth_plan_labels.to(device=pred.device, dtype=torch.float32)
                depth_l = F.smooth_l1_loss(depth_pred, depth_target)  # lingbot depth loss (_emb_loss)
                out["loss"] = out["loss"] + self.depth_loss_weight * depth_l
                out["depth_smooth_l1"] = depth_l.detach()
                out["depth_norm_ratio"] = (
                    depth_pred.norm(dim=-1).mean() / depth_target.norm(dim=-1).mean().clamp_min(1e-6)
                ).detach()
            if self.current_plan_head is not None and current_video_labels is not None:
                cur_pred = self.current_plan_head(image_hidden.to(head_dtype), current_lat.to(head_dtype)).float()
                cur_target = current_video_labels.to(device=pred.device, dtype=torch.float32)
                cur_l = F.mse_loss(cur_pred, cur_target)
                out["loss"] = out["loss"] + self.current_video_loss_weight * cur_l
                out["current_video_mse"] = cur_l.detach()
            if self.current_depth_head is not None and current_depth_labels is not None:
                cd_pred = self.current_depth_head(image_hidden.to(head_dtype), current_lat.to(head_dtype)).float()
                cd_target = current_depth_labels.to(device=pred.device, dtype=torch.float32)
                cd_l = F.smooth_l1_loss(cd_pred, cd_target)
                out["loss"] = out["loss"] + self.current_depth_loss_weight * cd_l
                out["current_depth_smooth_l1"] = cd_l.detach()
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
    head_params = [p for p in wrapper.plan_head.parameters() if p.requires_grad]
    for aux in ("depth_head", "current_plan_head", "current_depth_head"):
        m = getattr(wrapper, aux, None)
        if m is not None:
            # all auxiliary align heads train at head_lr alongside plan_head (fresh-ish resamplers)
            head_params += [p for p in m.parameters() if p.requires_grad]
    other_params = [
        p
        for n, p in wrapper.named_parameters()
        if p.requires_grad
        and not n.startswith(("plan_head.", "depth_head.", "current_plan_head.", "current_depth_head."))
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


def save_checkpoint(
    output_dir: Path,
    step: int,
    wrapper: PlannerWrapper | DDP,
    processor: Any,
    args: argparse.Namespace,
    rank: int,
) -> None:
    if not is_main(rank):
        return
    module = wrapper.module if isinstance(wrapper, DDP) else wrapper
    ckpt = output_dir / f"step_{step:06d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    module.model.save_pretrained(ckpt / "qwen3vl_lora_or_model")
    processor.save_pretrained(ckpt / "processor")
    torch.save(module.plan_head.state_dict(), ckpt / "plan_head.pt")
    if getattr(module, "depth_head", None) is not None:
        # aux head, but save it too: needed to visualize Depth-Pred from a checkpoint.
        torch.save(module.depth_head.state_dict(), ckpt / "depth_head.pt")
    if getattr(module, "current_plan_head", None) is not None:
        torch.save(module.current_plan_head.state_dict(), ckpt / "current_plan_head.pt")
    if getattr(module, "current_depth_head", None) is not None:
        torch.save(module.current_depth_head.state_dict(), ckpt / "current_depth_head.pt")
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
        "num_latent_per_keyframe": args.num_latent_per_keyframe,
        "num_head_latent_per_keyframe": int(getattr(args, "num_head_latent_per_keyframe", 0)),
        "use_depth": bool(getattr(args, "use_depth", False)),
        "use_current": bool(getattr(args, "use_current", False)),
        "keyframe_offsets": str(getattr(args, "keyframe_offsets", "")),
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
    (ckpt / "planner_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (output_dir / "latest_checkpoint.txt").write_text(str(ckpt), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.full_finetune and args.lora_r > 0:
        raise ValueError("--full-finetune is mutually exclusive with LoRA; set --lora-r 0.")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rank, world, local_rank = ddp_info()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    if args.plan_head_type in ("covt", "lingbot_dino"):
        _spec = int(args.num_head_latent_per_keyframe) if args.plan_head_type == "lingbot_dino" else 0
        _own_groups = (2 if (args.use_depth and _spec > 0) else (1 if _spec > 0 else 0))
        _per_kf = int(args.num_latent_per_keyframe) + _spec * _own_groups
        _cur_groups = 1 if (args.use_current and args.plan_head_type == "lingbot_dino") else 0
        _latent_len = (int(args.num_keyframes) + _cur_groups) * _per_kf
        plan_token_strs = [f"<|sem_plan_{i}|>" for i in range(_latent_len)]
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
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
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
        if args.frame_ranges_json is None:
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
        if args.frame_ranges_json is None:
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
        num_head_latent_per_keyframe=args.num_head_latent_per_keyframe,
        use_current=args.use_current,
        current_video_loss_weight=args.current_video_loss_weight,
        current_depth_loss_weight=args.current_depth_loss_weight,
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
    )
    if args.plan_head_type == "lingbot_dino" and args.head_warmstart_ckpt is not None:
        head_state = _load_lingbot_head_state(args.head_warmstart_ckpt)
        report = wrapper.plan_head.load_lingbot_warmstart(head_state, head_name="future_video_align_head")
        if wrapper.depth_head is not None:
            depth_report = wrapper.depth_head.load_lingbot_warmstart(head_state, head_name="future_depth_align_head")
        if wrapper.current_plan_head is not None:
            cur_report = wrapper.current_plan_head.load_lingbot_warmstart(head_state, head_name="current_video_align_head")
        if wrapper.current_depth_head is not None:
            # lingbot's current-depth head is named plain "depth_align_head"
            cur_depth_report = wrapper.current_depth_head.load_lingbot_warmstart(head_state, head_name="depth_align_head")
        if is_main(rank):
            print(json.dumps({"head_warmstart": report}), flush=True)
            if wrapper.depth_head is not None:
                print(json.dumps({"depth_head_warmstart": depth_report}), flush=True)
            if wrapper.current_plan_head is not None:
                print(json.dumps({"current_video_head_warmstart": cur_report}), flush=True)
            if wrapper.current_depth_head is not None:
                print(json.dumps({"current_depth_head_warmstart": cur_depth_report}), flush=True)
    wrapper.to(device)
    wrapper.train()
    if is_main(rank):
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
    if world > 1:
        wrapper = DDP(
            wrapper,
            device_ids=[local_rank],
            find_unused_parameters=args.ddp_find_unused_parameters,
            static_graph=not args.ddp_find_unused_parameters,
        )

    if args.online_plan_labels:
        _offsets_override = (
            [int(x) for x in str(args.keyframe_offsets).split(",") if str(x).strip()]
            if getattr(args, "keyframe_offsets", "") else None
        )
        dataset = OnlineSemanticPlanDataset(
            dataset_root=args.dataset_root,
            frame_ranges_json=args.frame_ranges_json,
            num_keyframes=args.num_keyframes,
            sequence_length=args.sequence_length,
            keyframe_scheme=args.keyframe_scheme,
            keyframe_gamma=args.keyframe_gamma,
            max_samples=args.max_samples,
            seed=args.seed,
            offsets_override=_offsets_override,
        )
        if is_main(rank):
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
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        collate_fn=Collator(
            processor=processor,
            plan_sequence=plan_sequence,
        ),
        pin_memory=True,
    )

    optim = build_optimizer(wrapper.module if isinstance(wrapper, DDP) else wrapper, args)
    scheduler = build_scheduler(optim, args)
    wandb_run = None
    if is_main(rank) and os.environ.get("PLANNER_WANDB", "1") == "1":
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
    accum = 0
    running_loss = 0.0
    pbar = tqdm(total=args.max_steps, disable=not is_main(rank), desc="qwen3vl planner")
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        # Advance the dataset epoch too, else sample_one_window_per_stem never resamples the
        # window per stem (the new epoch is picked up when the DataLoader respawns workers).
        dataset.set_epoch(step)
        for batch in loader:
            batch.pop("stems", None)
            keyframes = batch.pop("keyframe_images", None)
            current = batch.pop("current_image", None)
            module = wrapper.module if isinstance(wrapper, DDP) else wrapper
            model_dtype = next(module.model.parameters()).dtype
            batch = move_qwen_inputs_to_device(batch, device, model_dtype=model_dtype)
            if keyframes is not None:
                with torch.no_grad():
                    if dino_encoder is not None:
                        # Online DINO-video targets: teacher over [current, current, keyframe_k] clips
                        # -> [B, K*256, 1024], matching the LingbotDinoPlanHead output.
                        cur = current.permute(0, 3, 1, 2).contiguous()  # (B,3,H,W)
                        kfs = [keyframes[:, j].permute(0, 3, 1, 2).contiguous() for j in range(keyframes.shape[1])]
                        batch["semantic_plan_labels"] = dino_encoder.encode_future_keyframes(cur, kfs).float()
                        if depth_encoder is not None:
                            # LingBot-Depth targets over the SAME future keyframes -> [B, K*256, 1024].
                            batch["depth_plan_labels"] = depth_encoder.encode_future_keyframes(kfs).float()
                        if getattr(module, "use_current", False):
                            # Official-lingbot CURRENT targets: teacher on the current frame itself
                            # (clip [cur, cur, cur] gives the current frame in the same temporal stats).
                            batch["current_video_labels"] = dino_encoder.encode_future_keyframes(cur, [cur]).float()
                            if depth_encoder is not None:
                                batch["current_depth_labels"] = depth_encoder.encode_future_keyframes([cur]).float()
                    else:
                        # Online SigLIP2 targets (bit-consistent with the offline builder).
                        b, k = keyframes.shape[0], keyframes.shape[1]
                        imgs = [keyframes[i, j].numpy() for i in range(b) for j in range(k)]
                        target = sig_encode(sig_model, sig_processor, imgs, args.online_grid_size, device, torch.float32)
                        batch["semantic_plan_labels"] = target.reshape(b, k * target.shape[1], target.shape[-1])
            out = wrapper(**batch)
            (out["loss"] / args.grad_accum).backward()
            running_loss += float(out["loss"].detach())
            accum += 1
            if accum >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_([p for p in wrapper.parameters() if p.requires_grad], 1.0)
                optim.step()
                if scheduler is not None:
                    scheduler.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                accum = 0
                if is_main(rank):
                    pbar.update(1)
                    if step % args.log_steps == 0:
                        avg = running_loss / max(args.log_steps * args.grad_accum, 1)
                        log_entry = {"step": step, "loss": avg, "lr": optim.param_groups[0]["lr"]}
                        log_entry.update(
                            {key: float(value) for key, value in out.items() if key != "loss"}
                        )
                        print(json.dumps(log_entry), flush=True)
                        if wandb_run is not None:
                            wandb_run.log(log_entry, step=step)
                        running_loss = 0.0
                    if step % args.save_steps == 0:
                        save_checkpoint(args.output_dir, step, wrapper, processor, args, rank)
                if step >= args.max_steps:
                    break
        if len(loader) == 0:
            break

    save_checkpoint(args.output_dir, step, wrapper, processor, args, rank)
    if is_main(rank):
        pbar.close()
    if wandb_run is not None:
        wandb_run.finish()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
