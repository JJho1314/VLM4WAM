#!/usr/bin/env python3
"""Train Qwen3-VL semantic planner with Bernini-style masked flow matching.

This is a separate route from the Baton-style direct regression trainer.

The model receives the first frame, the instruction, and a sequence of
``<|sem_plan|>`` target semantic-token slots.  During training, target semantic
tokens are projected into the Qwen hidden space and used as dense target-token
inputs, but a random subset is replaced with a learned mask embedding.  The
hidden states at the masked positions condition a lightweight flow-matching
decoder that predicts the corresponding continuous semantic tokens.

This follows Bernini's key idea at the planner level:
masked dense visual/semantic tokens + MLLM context + flow matching in the
semantic embedding space.  It intentionally does not modify Cosmos/DiT code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

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
    parser.add_argument("--num-keyframes", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--semantic-dim", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=0)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--freeze-vision", action="store_true", default=True)
    parser.add_argument("--no-freeze-vision", action="store_false", dest="freeze_vision")
    parser.add_argument("--freeze-lm-head", action="store_true", default=True)
    parser.add_argument("--no-freeze-lm-head", action="store_false", dest="freeze_lm_head")
    parser.add_argument("--ddp-find-unused-parameters", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--log-steps", type=int, default=10)

    # Bernini-style masked target token modeling.
    parser.add_argument("--mask-beta-alpha", type=float, default=10.0)
    parser.add_argument("--mask-beta-beta", type=float, default=1.0)
    parser.add_argument("--min-mask-ratio", type=float, default=0.25)
    parser.add_argument("--max-mask-ratio", type=float, default=1.0)
    parser.add_argument("--decoder-hidden-size", type=int, default=2048)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--decoder-dropout", type=float, default=0.0)
    parser.add_argument("--time-embed-dim", type=int, default=256)
    parser.add_argument("--semantic-input-hidden-size", type=int, default=0)
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
    ) -> None:
        self.dataset_root = dataset_root
        self.label_dir = label_dir
        self.num_keyframes = num_keyframes
        self.grid_size = grid_size
        self.sample_one_window_per_stem = sample_one_window_per_stem

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
            self.groups = [grouped[k] for k in sorted(grouped)]
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

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.sample_one_window_per_stem:
            path = random.choice(self.groups[index])
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
        plan_text = " ".join([self.plan_token] * self.plan_len)
        texts = []
        for item in batch:
            user_text = (
                "You are a robot video semantic planner. Given the first frame and instruction, "
                "complete the masked future spatial semantic tokens for the manipulation video.\n"
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


class SemanticInputProjector(nn.Module):
    def __init__(self, semantic_dim: int, hidden_size: int, mlp_hidden_size: int = 0) -> None:
        super().__init__()
        if mlp_hidden_size <= 0:
            self.net = nn.Sequential(
                nn.LayerNorm(semantic_dim),
                nn.Linear(semantic_dim, hidden_size),
                nn.LayerNorm(hidden_size),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(semantic_dim),
                nn.Linear(semantic_dim, mlp_hidden_size),
                nn.GELU(),
                nn.Linear(mlp_hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    if t.ndim == 2:
        t = t.squeeze(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class SemanticFlowDecoder(nn.Module):
    """Small rectified-flow decoder in semantic-token space."""

    def __init__(
        self,
        *,
        semantic_dim: int,
        cond_dim: int,
        hidden_size: int,
        depth: int,
        time_embed_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.time_embed_dim = int(time_embed_dim)
        self.x_proj = nn.Linear(semantic_dim, hidden_size)
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden_size),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_size, dropout) for _ in range(depth)])
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, semantic_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        time_emb = sinusoidal_timestep_embedding(t, self.time_embed_dim).to(device=x_t.device, dtype=x_t.dtype)
        h = self.x_proj(x_t) + self.cond_proj(cond) + self.time_proj(time_emb)
        for block in self.blocks:
            h = block(h)
        return self.out(h)


class BerniniMaskedPlannerWrapper(nn.Module):
    def __init__(
        self,
        *,
        model: nn.Module,
        hidden_size: int,
        semantic_dim: int,
        plan_token_id: int,
        plan_len: int,
        mask_beta_alpha: float,
        mask_beta_beta: float,
        min_mask_ratio: float,
        max_mask_ratio: float,
        decoder_hidden_size: int,
        decoder_depth: int,
        decoder_dropout: float,
        time_embed_dim: int,
        semantic_input_hidden_size: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.plan_token_id = int(plan_token_id)
        self.plan_len = int(plan_len)
        self.semantic_dim = int(semantic_dim)
        self.mask_beta_alpha = float(mask_beta_alpha)
        self.mask_beta_beta = float(mask_beta_beta)
        self.min_mask_ratio = float(min_mask_ratio)
        self.max_mask_ratio = float(max_mask_ratio)
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.semantic_input_proj = SemanticInputProjector(
            semantic_dim=semantic_dim,
            hidden_size=hidden_size,
            mlp_hidden_size=semantic_input_hidden_size,
        )
        self.flow_decoder = SemanticFlowDecoder(
            semantic_dim=semantic_dim,
            cond_dim=hidden_size,
            hidden_size=decoder_hidden_size,
            depth=decoder_depth,
            dropout=decoder_dropout,
            time_embed_dim=time_embed_dim,
        )

    def collect_plan_hidden(self, hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        plan_mask = input_ids == self.plan_token_id
        plan_hidden = []
        for b in range(input_ids.shape[0]):
            h = hidden[b, plan_mask[b]]
            if h.shape[0] != self.plan_len:
                raise RuntimeError(f"Found {h.shape[0]} plan tokens, expected {self.plan_len}")
            plan_hidden.append(h)
        return torch.stack(plan_hidden, dim=0)

    def sample_target_mask(self, batch: int, device: torch.device) -> torch.Tensor:
        beta = torch.distributions.Beta(
            torch.tensor(self.mask_beta_alpha, device=device),
            torch.tensor(self.mask_beta_beta, device=device),
        )
        ratios = beta.sample((batch,)).clamp(self.min_mask_ratio, self.max_mask_ratio)
        mask = torch.rand(batch, self.plan_len, device=device) < ratios[:, None]
        for b in range(batch):
            if not bool(mask[b].any()):
                idx = torch.randint(0, self.plan_len, (1,), device=device)
                mask[b, idx] = True
        return mask

    @contextmanager
    def override_plan_embeddings(self, plan_inputs: torch.Tensor) -> Iterator[None]:
        embedding = self.model.get_input_embeddings()

        def hook(_module: nn.Module, hook_inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
            input_ids = hook_inputs[0]
            plan_mask = input_ids == self.plan_token_id
            out = output.clone()
            for b in range(input_ids.shape[0]):
                if int(plan_mask[b].sum().item()) != self.plan_len:
                    raise RuntimeError(
                        f"Found {int(plan_mask[b].sum().item())} plan tokens in batch {b}, expected {self.plan_len}"
                    )
                out[b, plan_mask[b]] = plan_inputs[b].to(device=out.device, dtype=out.dtype)
            return out

        handle = embedding.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def forward(self, semantic_plan_labels: torch.Tensor, **inputs: Any) -> dict[str, torch.Tensor]:
        target = semantic_plan_labels.to(device=self.mask_token.device, dtype=torch.float32)
        batch, plan_len, semantic_dim = target.shape
        if plan_len != self.plan_len:
            raise RuntimeError(f"Batch has {plan_len} plan tokens, wrapper expects {self.plan_len}")
        if semantic_dim != self.semantic_dim:
            raise RuntimeError(f"Batch semantic dim {semantic_dim}, wrapper expects {self.semantic_dim}")

        target_mask = self.sample_target_mask(batch, target.device)
        visible_inputs = self.semantic_input_proj(target.to(dtype=next(self.semantic_input_proj.parameters()).dtype))
        mask_inputs = self.mask_token.expand(batch, self.plan_len, -1)
        plan_inputs = torch.where(target_mask.unsqueeze(-1), mask_inputs, visible_inputs)

        input_ids = inputs["input_ids"]
        with self.override_plan_embeddings(plan_inputs):
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                logits_to_keep=1,
            )
        hidden = outputs.hidden_states[-1]
        plan_hidden = self.collect_plan_hidden(hidden, input_ids)

        masked_hidden = plan_hidden[target_mask].to(dtype=next(self.flow_decoder.parameters()).dtype)
        masked_target = target[target_mask].to(dtype=masked_hidden.dtype)
        noise = torch.randn_like(masked_target)
        t = torch.rand(masked_target.shape[0], 1, device=masked_target.device, dtype=masked_target.dtype)
        x_t = (1.0 - t) * noise + t * masked_target
        target_velocity = masked_target - noise
        pred_velocity = self.flow_decoder(x_t, t, masked_hidden)

        flow_loss = F.mse_loss(pred_velocity.float(), target_velocity.float())
        recon = x_t + (1.0 - t) * pred_velocity
        recon_mse = F.mse_loss(recon.float(), masked_target.float())
        cosine = F.cosine_similarity(recon.float(), masked_target.float(), dim=-1).mean()
        mask_ratio = target_mask.float().mean()
        return {
            "loss": flow_loss,
            "flow_loss": flow_loss.detach(),
            "recon_mse": recon_mse.detach(),
            "mean_cosine": cosine.detach(),
            "mask_ratio": mask_ratio.detach(),
        }


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


def build_optimizer(wrapper: BerniniMaskedPlannerWrapper, args: argparse.Namespace) -> torch.optim.Optimizer:
    head_prefixes = ("semantic_input_proj.", "flow_decoder.", "mask_token")
    head_params = [
        p for n, p in wrapper.named_parameters() if p.requires_grad and n.startswith(head_prefixes)
    ]
    other_params = [
        p
        for n, p in wrapper.named_parameters()
        if p.requires_grad and not n.startswith(head_prefixes)
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


def probe_semantic_dim(label_dir: Path) -> int:
    first = next(iter(sorted(label_dir.glob("*.pt"))), None)
    if first is None:
        manifest_paths = [label_dir / "manifest.jsonl"] if (label_dir / "manifest.jsonl").exists() else sorted(label_dir.glob("manifest*.jsonl"))
        for manifest_path in manifest_paths:
            for line in manifest_path.read_text().splitlines():
                if not line.strip():
                    continue
                path = Path(json.loads(line)["path"])
                if path.exists():
                    first = path
                    break
            if first is not None:
                break
    if first is None:
        raise RuntimeError(f"No semantic label .pt found under {label_dir}")
    payload = torch.load(first, map_location="cpu", weights_only=False)
    return int(payload["semantic_plan"].shape[-1])


def read_feature_type(label_dir: Path) -> str:
    summary_path = label_dir / "summary.json"
    if summary_path.exists():
        try:
            return str(json.loads(summary_path.read_text()).get("feature_type", "unknown"))
        except Exception:
            return "unknown"
    return "unknown"


def save_checkpoint(
    output_dir: Path,
    step: int,
    wrapper: BerniniMaskedPlannerWrapper | DDP,
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
    torch.save(module.semantic_input_proj.state_dict(), ckpt / "semantic_input_proj.pt")
    torch.save(module.flow_decoder.state_dict(), ckpt / "semantic_flow_decoder.pt")
    torch.save(module.mask_token.detach().cpu(), ckpt / "semantic_mask_token.pt")
    plan_embedding = module.model.get_input_embeddings().weight[module.plan_token_id].detach().cpu()
    torch.save(plan_embedding, ckpt / "plan_token_embedding.pt")
    meta = {
        "step": step,
        "objective": "bernini_masked_semantic_flow_matching",
        "plan_token": PLAN_TOKEN,
        "plan_token_id": module.plan_token_id,
        "num_keyframes": args.num_keyframes,
        "grid_size": args.grid_size,
        "semantic_dim": args.semantic_dim,
        "model_path": str(args.model_path),
        "feature_type": read_feature_type(args.plan_label_dir),
        "plan_label_dir": str(args.plan_label_dir),
        "sample_one_window_per_stem": bool(args.sample_one_window_per_stem),
        "mask_beta_alpha": float(args.mask_beta_alpha),
        "mask_beta_beta": float(args.mask_beta_beta),
        "min_mask_ratio": float(args.min_mask_ratio),
        "max_mask_ratio": float(args.max_mask_ratio),
        "decoder_hidden_size": int(args.decoder_hidden_size),
        "decoder_depth": int(args.decoder_depth),
        "decoder_dropout": float(args.decoder_dropout),
        "time_embed_dim": int(args.time_embed_dim),
        "semantic_input_hidden_size": int(args.semantic_input_hidden_size),
        "full_finetune": bool(args.full_finetune),
        "lora_r": int(args.lora_r),
        "freeze_vision": bool(args.freeze_vision),
        "freeze_lm_head": bool(args.freeze_lm_head),
    }
    (ckpt / "planner_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (output_dir / "latest_checkpoint.txt").write_text(str(ckpt), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.full_finetune and args.lora_r > 0:
        raise ValueError("--full-finetune is mutually exclusive with LoRA; set --lora-r 0.")
    if not (0.0 <= args.min_mask_ratio <= args.max_mask_ratio <= 1.0):
        raise ValueError("--min-mask-ratio and --max-mask-ratio must satisfy 0 <= min <= max <= 1")
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
    model = apply_lora(model, args)

    if args.semantic_dim <= 0:
        args.semantic_dim = probe_semantic_dim(args.plan_label_dir)

    hidden_size = int(model.config.text_config.hidden_size)
    wrapper = BerniniMaskedPlannerWrapper(
        model=model,
        hidden_size=hidden_size,
        semantic_dim=args.semantic_dim,
        plan_token_id=plan_token_id,
        plan_len=args.num_keyframes * args.grid_size * args.grid_size,
        mask_beta_alpha=args.mask_beta_alpha,
        mask_beta_beta=args.mask_beta_beta,
        min_mask_ratio=args.min_mask_ratio,
        max_mask_ratio=args.max_mask_ratio,
        decoder_hidden_size=args.decoder_hidden_size,
        decoder_depth=args.decoder_depth,
        decoder_dropout=args.decoder_dropout,
        time_embed_dim=args.time_embed_dim,
        semantic_input_hidden_size=args.semantic_input_hidden_size,
    )
    wrapper.to(device)
    wrapper.train()

    if is_main(rank):
        trainable, total = count_trainable_parameters(wrapper)
        print(
            json.dumps(
                {
                    "objective": "bernini_masked_semantic_flow_matching",
                    "full_finetune": bool(args.full_finetune),
                    "lora_r": int(args.lora_r),
                    "freeze_vision": bool(args.freeze_vision),
                    "freeze_lm_head": bool(args.freeze_lm_head),
                    "num_keyframes": int(args.num_keyframes),
                    "grid_size": int(args.grid_size),
                    "semantic_dim": int(args.semantic_dim),
                    "sample_one_window_per_stem": bool(args.sample_one_window_per_stem),
                    "mask_beta_alpha": float(args.mask_beta_alpha),
                    "mask_beta_beta": float(args.mask_beta_beta),
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
    pbar = tqdm(total=args.max_steps, disable=not is_main(rank), desc="qwen3vl bernini-fm planner")
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
                        print(
                            json.dumps(
                                {
                                    "step": step,
                                    "loss": avg,
                                    "flow_loss": float(out["flow_loss"]),
                                    "recon_mse": float(out["recon_mse"]),
                                    "mean_cosine": float(out["mean_cosine"]),
                                    "mask_ratio": float(out["mask_ratio"]),
                                }
                            ),
                            flush=True,
                        )
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
