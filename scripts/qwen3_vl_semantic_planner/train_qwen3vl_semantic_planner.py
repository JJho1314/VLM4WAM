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
    parser.add_argument("--num-keyframes", type=int, default=6)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--semantic-dim", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--freeze-vision", action="store_true", default=True)
    parser.add_argument("--no-freeze-vision", action="store_false", dest="freeze_vision")
    parser.add_argument("--train-plan-token-embedding", action="store_true", default=True)
    parser.add_argument("--no-train-plan-token-embedding", action="store_false", dest="train_plan_token_embedding")
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
    ) -> None:
        self.dataset_root = dataset_root
        self.label_dir = label_dir
        self.num_keyframes = num_keyframes
        self.grid_size = grid_size
        if (label_dir / "manifest.jsonl").exists():
            entries = []
            for line in (label_dir / "manifest.jsonl").read_text().splitlines():
                if line.strip():
                    entries.append(json.loads(line))
            paths = [Path(x["path"]) for x in entries]
        else:
            paths = sorted(label_dir.glob("*.pt"))
        self.paths = [p for p in paths if p.exists()]
        if max_samples > 0:
            self.paths = self.paths[:max_samples]
        if not self.paths:
            raise RuntimeError(f"No semantic plan labels found under {label_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        payload = torch.load(self.paths[index], map_location="cpu", weights_only=False)
        stem = payload["stem"]
        video_path = Path(payload.get("video_path") or self.dataset_root / "videos" / f"{stem}.mp4")
        prompt = payload.get("prompt") or "A robot manipulates the target object."
        first_frame_index = int(payload.get("first_frame_index", 0))
        image = load_frame(video_path, first_frame_index)
        plan = payload["semantic_plan"].float()
        expected = self.num_keyframes * self.grid_size * self.grid_size
        plan = plan.reshape(-1, plan.shape[-1])
        if plan.shape[0] != expected:
            raise RuntimeError(f"{self.paths[index]} has {plan.shape[0]} plan tokens, expected {expected}")
        return {
            "stem": stem,
            "image": image,
            "prompt": prompt,
            "semantic_plan": plan,
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


class PlannerWrapper(nn.Module):
    def __init__(self, model: nn.Module, hidden_size: int, semantic_dim: int, plan_token_id: int) -> None:
        super().__init__()
        self.model = model
        self.plan_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, semantic_dim),
        )
        self.plan_token_id = int(plan_token_id)

    def forward(self, semantic_plan_labels: torch.Tensor, **inputs: Any) -> dict[str, torch.Tensor]:
        outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[-1]
        input_ids = inputs["input_ids"]
        plan_mask = input_ids == self.plan_token_id
        batch, plan_len, sem_dim = semantic_plan_labels.shape
        pred = hidden.new_zeros((batch, plan_len, sem_dim), dtype=torch.float32)
        valid = []
        for b in range(batch):
            h = hidden[b, plan_mask[b]]
            if h.shape[0] != plan_len:
                raise RuntimeError(f"Found {h.shape[0]} plan tokens, expected {plan_len}")
            head_dtype = next(self.plan_head.parameters()).dtype
            pred[b] = self.plan_head(h.to(dtype=head_dtype)).float()
            valid.append(h.shape[0])
        target = semantic_plan_labels.to(device=pred.device, dtype=torch.float32)
        mse = F.mse_loss(pred, target)
        cosine = 1.0 - F.cosine_similarity(pred.flatten(0, 1), target.flatten(0, 1), dim=-1).mean()
        loss = mse + 0.1 * cosine
        return {"loss": loss, "mse": mse.detach(), "cosine_loss": cosine.detach()}


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


def set_trainable(model: nn.Module, freeze_vision: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    if freeze_vision and hasattr(model, "visual"):
        for p in model.visual.parameters():
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
    meta = {
        "step": step,
        "plan_token": PLAN_TOKEN,
        "plan_token_id": module.plan_token_id,
        "num_keyframes": args.num_keyframes,
        "grid_size": args.grid_size,
        "semantic_dim": args.semantic_dim,
        "model_path": str(args.model_path),
        "objective": "continuous_semantic_blueprint_regression",
        "feature_type": "qwen3vl_last_hidden_image_tokens_pooled",
        "train_plan_token_embedding": bool(args.train_plan_token_embedding),
    }
    (ckpt / "planner_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (output_dir / "latest_checkpoint.txt").write_text(str(ckpt), encoding="utf-8")


def main() -> None:
    args = parse_args()
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
    set_trainable(model, freeze_vision=args.freeze_vision)
    if args.train_plan_token_embedding:
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
    wrapper = PlannerWrapper(model=model, hidden_size=hidden_size, semantic_dim=args.semantic_dim, plan_token_id=plan_token_id)
    wrapper.to(device)
    wrapper.train()
    if world > 1:
        wrapper = DDP(wrapper, device_ids=[local_rank], find_unused_parameters=False, static_graph=True)

    dataset = SemanticPlanDataset(
        dataset_root=args.dataset_root,
        label_dir=args.plan_label_dir,
        num_keyframes=args.num_keyframes,
        grid_size=args.grid_size,
        max_samples=args.max_samples,
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
                        print(json.dumps({"step": step, "loss": avg, "mse": float(out["mse"]), "cosine_loss": float(out["cosine_loss"])}), flush=True)
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
