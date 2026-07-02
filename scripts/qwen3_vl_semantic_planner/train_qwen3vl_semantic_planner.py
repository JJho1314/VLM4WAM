#!/usr/bin/env python3
"""Fine-tune Qwen3-VL-Instruct as a continuous semantic planner.

The model receives the first video frame and instruction, then exposes a fixed
number of <|sem_plan|> query tokens.  The hidden states at those query tokens
are projected to future semantic plan tokens precomputed by
build_qwen3vl_semantic_plan_labels.py.

This is intentionally separate from any segmentation or grounding model code.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from qwen3vl_wrapper import load_qwen3vl_model_and_processor, move_qwen_inputs_to_device

PLAN_TOKEN = "<|sem_plan|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--plan-label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--plan-head-type", choices=["mlp", "baton_crossattn"], default="mlp")
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


@dataclass
class Collator:
    processor: Any
    plan_token: str
    plan_len: int

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [x["image"] for x in batch]
        labels = torch.stack([x["semantic_plan"] for x in batch], dim=0)
        texts = []
        plan_text = " ".join([self.plan_token] * self.plan_len)
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
        inputs["semantic_plan_labels"] = labels
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


class PlannerWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        hidden_size: int,
        semantic_dim: int,
        plan_token_id: int,
        plan_len: int,
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
        self.plan_len = int(plan_len)
        if plan_head_type == "mlp":
            self.plan_head = MLPPlanHead(hidden_size, semantic_dim, sem_mlp_hidden_size)
        elif plan_head_type == "baton_crossattn":
            self.plan_head = BatonCrossAttentionPlanHead(
                plan_len=plan_len,
                hidden_size=hidden_size,
                semantic_dim=semantic_dim,
                sem_mlp_hidden_size=sem_mlp_hidden_size,
                num_heads=plan_head_num_heads,
                dropout=plan_head_dropout,
            )
        else:
            raise ValueError(f"Unsupported plan_head_type: {plan_head_type}")
        self.plan_token_id = int(plan_token_id)
        self.model = model

    def collect_plan_hidden(self, hidden: torch.Tensor, input_ids: torch.Tensor, plan_len: int) -> torch.Tensor:
        plan_mask = input_ids == self.plan_token_id
        plan_hidden = []
        for b in range(input_ids.shape[0]):
            h = hidden[b, plan_mask[b]]
            if h.shape[0] != plan_len:
                raise RuntimeError(f"Found {h.shape[0]} plan tokens, expected {plan_len}")
            plan_hidden.append(h)
        return torch.stack(plan_hidden, dim=0)

    def predict_semantic_plan(self, **inputs: Any) -> torch.Tensor:
        outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[-1]
        input_ids = inputs["input_ids"]
        plan_hidden = self.collect_plan_hidden(hidden, input_ids, self.plan_len)
        head_dtype = next(self.plan_head.parameters()).dtype
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

    def forward(self, semantic_plan_labels: torch.Tensor, **inputs: Any) -> dict[str, torch.Tensor]:
        batch, plan_len, _ = semantic_plan_labels.shape
        if plan_len != self.plan_len:
            raise RuntimeError(f"Batch has {plan_len} plan tokens, wrapper expects {self.plan_len}")
        pred = self.predict_semantic_plan(**inputs)
        if pred.shape[0] != batch:
            raise RuntimeError(f"Prediction batch {pred.shape[0]} does not match labels batch {batch}")
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


def register_single_row_hook(param: nn.Parameter, row_id: int) -> None:
    if param.ndim != 2:
        raise RuntimeError(f"Expected embedding weight with ndim=2, got {tuple(param.shape)}")
    if row_id < 0 or row_id >= param.shape[0]:
        raise RuntimeError(f"Plan token id {row_id} outside embedding rows {param.shape[0]}")
    row_mask = torch.zeros((param.shape[0], 1), dtype=torch.float32)
    row_mask[row_id] = 1.0

    def hook(grad: torch.Tensor) -> torch.Tensor:
        return grad * row_mask.to(device=grad.device, dtype=grad.dtype)

    param.requires_grad_(True)
    param.register_hook(hook)


def build_optimizer(wrapper: PlannerWrapper, args: argparse.Namespace) -> torch.optim.Optimizer:
    head_params = [p for p in wrapper.plan_head.parameters() if p.requires_grad]
    other_params = [
        p
        for n, p in wrapper.named_parameters()
        if p.requires_grad and not n.startswith("plan_head.")
    ]
    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": args.head_lr, "weight_decay": args.weight_decay})
    return torch.optim.AdamW(groups)


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
    plan_embedding = module.model.get_input_embeddings().weight[module.plan_token_id].detach().cpu()
    torch.save(plan_embedding, ckpt / "plan_token_embedding.pt")
    feature_type = "unknown"
    summary_path = args.plan_label_dir / "summary.json"
    if summary_path.exists():
        try:
            feature_type = json.loads(summary_path.read_text()).get("feature_type", feature_type)
        except Exception:
            feature_type = feature_type
    meta = {
        "step": step,
        "plan_token": PLAN_TOKEN,
        "plan_token_id": module.plan_token_id,
        "num_keyframes": args.num_keyframes,
        "grid_size": args.grid_size,
        "semantic_dim": args.semantic_dim,
        "model_path": str(args.model_path),
        "objective": "continuous_semantic_blueprint_regression",
        "feature_type": feature_type,
        "plan_label_dir": str(args.plan_label_dir),
        "sample_one_window_per_stem": bool(args.sample_one_window_per_stem),
        "plan_head_type": module.plan_head_type,
        "plan_head_num_heads": int(module.plan_head_num_heads),
        "plan_head_dropout": float(module.plan_head_dropout),
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
    added = 0 if PLAN_TOKEN in processor.tokenizer.get_vocab() else processor.tokenizer.add_tokens([PLAN_TOKEN], special_tokens=True)
    plan_token_id = processor.tokenizer.convert_tokens_to_ids(PLAN_TOKEN)
    if added:
        model.resize_token_embeddings(len(processor.tokenizer))
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
    if args.train_plan_token_embedding and not args.full_finetune:
        register_single_row_hook(model.get_input_embeddings().weight, plan_token_id)
    model = apply_lora(model, args)

    # Probe semantic dim if not supplied.
    if args.semantic_dim <= 0:
        first = next(iter(sorted(args.plan_label_dir.glob("*.pt"))), None)
        if first is None:
            raise RuntimeError(f"No .pt labels under {args.plan_label_dir}")
        payload = torch.load(first, map_location="cpu", weights_only=False)
        args.semantic_dim = int(payload["semantic_plan"].shape[-1])

    hidden_size = int(model.config.text_config.hidden_size)
    wrapper = PlannerWrapper(
        model=model,
        hidden_size=hidden_size,
        semantic_dim=args.semantic_dim,
        plan_token_id=plan_token_id,
        plan_len=args.num_keyframes * args.grid_size * args.grid_size,
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
    )
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
            plan_token=PLAN_TOKEN,
            plan_len=args.num_keyframes * args.grid_size * args.grid_size,
        ),
        pin_memory=True,
    )

    optim = build_optimizer(wrapper.module if isinstance(wrapper, DDP) else wrapper, args)
    step = 0
    accum = 0
    running_loss = 0.0
    pbar = tqdm(total=args.max_steps, disable=not is_main(rank), desc="qwen3vl planner")
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            batch.pop("stems", None)
            module = wrapper.module if isinstance(wrapper, DDP) else wrapper
            model_dtype = next(module.model.parameters()).dtype
            batch = move_qwen_inputs_to_device(batch, device, model_dtype=model_dtype)
            out = wrapper(**batch)
            (out["loss"] / args.grad_accum).backward()
            running_loss += float(out["loss"].detach())
            accum += 1
            if accum >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_([p for p in wrapper.parameters() if p.requires_grad], 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                accum = 0
                if is_main(rank):
                    pbar.update(1)
                    if step % args.log_steps == 0:
                        avg = running_loss / max(args.log_steps * args.grad_accum, 1)
                        log_entry = {"step": step, "loss": avg}
                        log_entry.update(
                            {key: float(value) for key, value in out.items() if key != "loss"}
                        )
                        print(json.dumps(log_entry), flush=True)
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
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
