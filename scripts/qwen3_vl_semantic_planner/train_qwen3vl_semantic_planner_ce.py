#!/usr/bin/env python3
"""Stage-A discrete semantic planner training for Qwen3-VL.

This follows the Plan-X Stage-A objective more closely than continuous feature
regression: the planner receives the first DROID frame and instruction, then
teacher-forces a sequence of vector-quantized future semantic tokens with a
standard causal-LM cross entropy loss.
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
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from qwen3vl_wrapper import load_qwen3vl_model_and_processor, move_qwen_inputs_to_device


SEM_TOKEN_TEMPLATE = "<|sem_{:06d}|>"
SEM_TOKEN_SEPARATORS = {"none": "", "space": " "}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--discrete-label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-keyframes", type=int, default=6)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--codebook-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--token-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--freeze-vision", action="store_true", default=True)
    parser.add_argument("--no-freeze-vision", action="store_false", dest="freeze_vision")
    parser.add_argument("--train-semantic-token-embeddings", action="store_true", default=True)
    parser.add_argument("--no-train-semantic-token-embeddings", action="store_false", dest="train_semantic_token_embeddings")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--semantic-token-separator", choices=sorted(SEM_TOKEN_SEPARATORS), default="none")
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


def semantic_token(code_id: int) -> str:
    return SEM_TOKEN_TEMPLATE.format(int(code_id))


def semantic_tokens(codebook_size: int) -> list[str]:
    return [semantic_token(i) for i in range(codebook_size)]


def join_semantic_tokens(tokens: list[str], separator: str) -> str:
    return SEM_TOKEN_SEPARATORS[separator].join(tokens)


def planner_instruction(prompt: str, expected_tokens: int, separator: str) -> str:
    if separator == "none":
        output_format = (
            f"Output exactly {expected_tokens} consecutive semantic tokens with no spaces, "
            "newlines, punctuation, or natural language."
        )
    else:
        output_format = (
            f"Output exactly {expected_tokens} semantic tokens separated by single spaces, "
            "with no other text."
        )
    return (
        "You are a robot video semantic planner. Given the first frame and instruction, "
        "predict the future spatial semantic token sequence for the manipulation video. "
        f"{output_format}\n"
        f"Instruction: {prompt}"
    )


def load_frame(video_path: Path, index: int) -> Image.Image:
    import decord

    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    index = min(max(int(index), 0), len(vr) - 1)
    return Image.fromarray(vr[index].asnumpy()).convert("RGB")


def list_payload_paths(label_dir: Path, max_samples: int) -> list[Path]:
    manifest = label_dir / "manifest.jsonl"
    if manifest.exists():
        paths = []
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            p = Path(rec["path"])
            if p.exists():
                paths.append(p)
    else:
        paths = [p for p in sorted(label_dir.glob("*.pt")) if p.name != "semantic_codebook.pt"]
    if max_samples > 0:
        paths = paths[:max_samples]
    if not paths:
        raise RuntimeError(f"No discrete semantic labels found under {label_dir}")
    return paths


def infer_codebook_size(label_dir: Path, paths: list[Path]) -> int:
    summary = label_dir / "summary.json"
    if summary.exists():
        data = json.loads(summary.read_text())
        if int(data.get("codebook_size", 0)) > 0:
            return int(data["codebook_size"])
    payload = torch.load(paths[0], map_location="cpu", weights_only=False)
    size = int(payload.get("codebook_size", 0))
    if size <= 0:
        codes = payload["semantic_token_ids"].reshape(-1)
        size = int(codes.max().item()) + 1
    return size


class DiscreteSemanticPlanDataset(Dataset):
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
        self.paths = list_payload_paths(label_dir, max_samples)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        payload = torch.load(self.paths[index], map_location="cpu", weights_only=False)
        stem = payload["stem"]
        video_path = Path(payload.get("video_path") or self.dataset_root / "videos" / f"{stem}.mp4")
        prompt = payload.get("prompt") or "A robot manipulates the target object."
        first_frame_index = int(payload.get("first_frame_index", 0))
        image = load_frame(video_path, first_frame_index)
        codes = payload["semantic_token_ids"].long().reshape(-1)
        expected = self.num_keyframes * self.grid_size * self.grid_size
        if codes.numel() != expected:
            raise RuntimeError(f"{self.paths[index]} has {codes.numel()} semantic tokens, expected {expected}")
        return {
            "stem": stem,
            "image": image,
            "prompt": " ".join(str(prompt).split()),
            "semantic_token_ids": codes,
        }


@dataclass
class CECollator:
    processor: Any
    codebook_size: int
    semantic_token_ids: torch.Tensor
    semantic_token_separator: str = "none"

    def __post_init__(self) -> None:
        self.id_to_text = semantic_tokens(self.codebook_size)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [x["image"] for x in batch]
        texts = []
        expected_counts = []
        for item in batch:
            codes = item["semantic_token_ids"].tolist()
            expected_counts.append(len(codes))
            assistant_text = join_semantic_tokens(
                [self.id_to_text[int(c)] for c in codes],
                self.semantic_token_separator,
            )
            user_text = planner_instruction(
                item["prompt"],
                expected_tokens=len(codes),
                separator=self.semantic_token_separator,
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ],
                },
                {"role": "assistant", "content": assistant_text},
            ]
            texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))

        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        labels = torch.full_like(inputs["input_ids"], -100)
        semantic_ids = self.semantic_token_ids.to(dtype=inputs["input_ids"].dtype)
        for b in range(inputs["input_ids"].shape[0]):
            sem_mask = torch.isin(inputs["input_ids"][b], semantic_ids)
            count = int(sem_mask.sum().item())
            if count != expected_counts[b]:
                raise RuntimeError(f"Found {count} semantic token positions, expected {expected_counts[b]}")
            labels[b, sem_mask] = inputs["input_ids"][b, sem_mask]
        inputs["labels"] = labels
        inputs["stems"] = [x["stem"] for x in batch]
        return inputs


def freeze_base_model(model: nn.Module, freeze_vision: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    if freeze_vision and hasattr(model, "visual"):
        for param in model.visual.parameters():
            param.requires_grad_(False)


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


def register_semantic_row_hook(param: nn.Parameter, token_ids: list[int]) -> None:
    if param.ndim != 2:
        raise RuntimeError(f"Expected embedding/head weight with ndim=2, got {tuple(param.shape)}")
    if max(token_ids) >= param.shape[0]:
        raise RuntimeError(f"Semantic token id {max(token_ids)} exceeds vocab rows {param.shape[0]}")
    row_mask = torch.zeros((param.shape[0], 1), dtype=torch.float32)
    row_mask[token_ids] = 1.0

    def hook(grad: torch.Tensor) -> torch.Tensor:
        return grad * row_mask.to(device=grad.device, dtype=grad.dtype)

    param.requires_grad_(True)
    param.register_hook(hook)


def enable_semantic_token_training(model: nn.Module, token_ids: list[int]) -> set[int]:
    trained_param_ids: set[int] = set()
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    for emb in (input_emb, output_emb):
        if emb is None:
            continue
        param = emb.weight
        if id(param) in trained_param_ids:
            continue
        register_semantic_row_hook(param, token_ids)
        trained_param_ids.add(id(param))
    return trained_param_ids


def collect_semantic_token_weights(model: nn.Module, token_ids: list[int], tokens: list[str]) -> dict[str, Any]:
    state: dict[str, Any] = {
        "semantic_tokens": tokens,
        "semantic_token_ids": token_ids,
    }
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()
    if input_emb is not None:
        state["input_embeddings"] = input_emb.weight.detach().cpu()[token_ids].clone()
    if output_emb is not None:
        state["output_embeddings"] = output_emb.weight.detach().cpu()[token_ids].clone()
        state["output_tied_to_input"] = (
            input_emb is not None and output_emb.weight.data_ptr() == input_emb.weight.data_ptr()
        )
    return state


def build_optimizer(model: nn.Module, semantic_param_ids: set[int], args: argparse.Namespace) -> torch.optim.Optimizer:
    sem_params = []
    other_params = []
    seen = set()
    for param in model.parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if id(param) in semantic_param_ids:
            sem_params.append(param)
        else:
            other_params.append(param)
    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay})
    if sem_params:
        groups.append({"params": sem_params, "lr": args.token_lr, "weight_decay": 0.0})
    if not groups:
        raise RuntimeError("No trainable parameters found.")
    return torch.optim.AdamW(groups)


def build_scheduler(optim: torch.optim.Optimizer, warmup_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    if warmup_steps <= 0:
        return torch.optim.lr_scheduler.LambdaLR(optim, lambda _: 1.0)
    return torch.optim.lr_scheduler.LambdaLR(optim, lambda step: min(float(step + 1) / float(warmup_steps), 1.0))


@torch.no_grad()
def semantic_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels != -100
    if not bool(valid.any()):
        return 0.0
    pred = shift_logits.argmax(dim=-1)
    return float((pred[valid] == shift_labels[valid]).float().mean().item())


def trainable_param_summary(model: nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable_params": int(trainable), "total_params": int(total)}


def save_checkpoint(
    output_dir: Path,
    step: int,
    model_or_ddp: nn.Module | DDP,
    processor: Any,
    args: argparse.Namespace,
    rank: int,
    sem_tokens: list[str],
    sem_token_ids: list[int],
) -> None:
    if not is_main(rank):
        return
    model = model_or_ddp.module if isinstance(model_or_ddp, DDP) else model_or_ddp
    ckpt = output_dir / f"step_{step:06d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt / "qwen3vl_lora_or_model")
    processor.save_pretrained(ckpt / "processor")
    torch.save(collect_semantic_token_weights(model, sem_token_ids, sem_tokens), ckpt / "semantic_token_weights.pt")
    meta = {
        "step": step,
        "objective": "causal_lm_ce_over_vq_semantic_tokens",
        "semantic_token_template": SEM_TOKEN_TEMPLATE,
        "codebook_size": args.codebook_size,
        "num_keyframes": args.num_keyframes,
        "grid_size": args.grid_size,
        "model_path": str(args.model_path),
        "discrete_label_dir": str(args.discrete_label_dir),
        "train_semantic_token_embeddings": bool(args.train_semantic_token_embeddings),
        "semantic_token_separator": args.semantic_token_separator,
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

    paths = list_payload_paths(args.discrete_label_dir, args.max_samples)
    if args.codebook_size <= 0:
        args.codebook_size = infer_codebook_size(args.discrete_label_dir, paths)
    sem_tokens = semantic_tokens(args.codebook_size)

    model, processor = load_qwen3vl_model_and_processor(
        args.model_path,
        device=None,
        dtype=args.dtype,
        attn_implementation="sdpa",
        local_files_only=True,
        eval_mode=False,
    )
    existing = processor.tokenizer.get_vocab()
    missing = [tok for tok in sem_tokens if tok not in existing]
    if missing:
        processor.tokenizer.add_special_tokens({"additional_special_tokens": missing})
        model.resize_token_embeddings(len(processor.tokenizer))
    sem_token_ids = [int(processor.tokenizer.convert_tokens_to_ids(tok)) for tok in sem_tokens]
    if any(x < 0 for x in sem_token_ids):
        raise RuntimeError("Failed to register all semantic tokens.")

    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    freeze_base_model(model, freeze_vision=args.freeze_vision)
    model = apply_lora(model, args)
    semantic_param_ids: set[int] = set()
    if args.train_semantic_token_embeddings:
        semantic_param_ids = enable_semantic_token_training(model, sem_token_ids)
    model.to(device)
    model.train()

    if is_main(rank):
        summary = trainable_param_summary(model)
        summary.update(
            {
                "codebook_size": args.codebook_size,
                "num_semantic_tokens_per_sample": args.num_keyframes * args.grid_size * args.grid_size,
                "num_samples": len(paths),
            }
        )
        print(json.dumps(summary, indent=2), flush=True)

    if world > 1:
        try:
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False, static_graph=True)
        except TypeError:
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
            if hasattr(model, "_set_static_graph"):
                model._set_static_graph()

    dataset = DiscreteSemanticPlanDataset(
        dataset_root=args.dataset_root,
        label_dir=args.discrete_label_dir,
        num_keyframes=args.num_keyframes,
        grid_size=args.grid_size,
        max_samples=args.max_samples,
    )
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    sem_token_id_tensor = torch.tensor(sem_token_ids, dtype=torch.long)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
            collate_fn=CECollator(
                processor=processor,
                codebook_size=args.codebook_size,
                semantic_token_ids=sem_token_id_tensor,
                semantic_token_separator=args.semantic_token_separator,
            ),
        pin_memory=True,
    )

    module_for_optim = model.module if isinstance(model, DDP) else model
    optim = build_optimizer(module_for_optim, semantic_param_ids, args)
    scheduler = build_scheduler(optim, args.warmup_steps)
    optim.zero_grad(set_to_none=True)

    step = 0
    accum = 0
    running_loss = 0.0
    running_acc = 0.0
    pbar = tqdm(total=args.max_steps, disable=not is_main(rank), desc="qwen3vl stage-a ce")
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            batch.pop("stems", None)
            module = model.module if isinstance(model, DDP) else model
            model_dtype = next(module.parameters()).dtype
            batch = move_qwen_inputs_to_device(batch, device, model_dtype=model_dtype)
            out = model(**batch, use_cache=False)
            loss = out.loss
            (loss / args.grad_accum).backward()
            running_loss += float(loss.detach())
            running_acc += semantic_accuracy(out.logits.detach(), batch["labels"])
            accum += 1
            if accum >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if is_main(rank):
                    pbar.update(1)
                    if step % args.log_steps == 0:
                        denom = max(args.log_steps * args.grad_accum, 1)
                        log = {
                            "step": step,
                            "loss": running_loss / denom,
                            "semantic_token_acc": running_acc / denom,
                            "lr": scheduler.get_last_lr()[0],
                        }
                        print(json.dumps(log), flush=True)
                        running_loss = 0.0
                        running_acc = 0.0
                    if step % args.save_steps == 0:
                        save_checkpoint(args.output_dir, step, model, processor, args, rank, sem_tokens, sem_token_ids)
                accum = 0
                if step >= args.max_steps:
                    break
        if len(loader) == 0:
            break

    save_checkpoint(args.output_dir, step, model, processor, args, rank, sem_tokens, sem_token_ids)
    if is_main(rank):
        pbar.close()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
