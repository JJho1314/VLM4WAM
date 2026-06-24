#!/usr/bin/env python3
"""Evaluate Baton-style continuous Qwen3-VL semantic planner checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from qwen3vl_wrapper import load_qwen3vl_model_and_processor, move_qwen_inputs_to_device
from train_qwen3vl_semantic_planner import (
    PLAN_TOKEN,
    Collator,
    PlannerWrapper,
    SemanticPlanDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--plan-label-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_meta(checkpoint_dir: Path) -> dict[str, Any]:
    meta_path = checkpoint_dir / "planner_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing checkpoint metadata: {meta_path}")
    return json.loads(meta_path.read_text())


def restore_planner(args: argparse.Namespace, meta: dict[str, Any], device: torch.device) -> tuple[PlannerWrapper, Any]:
    model_path = args.model_path or Path(meta["model_path"])
    model, processor = load_qwen3vl_model_and_processor(
        model_path,
        device=None,
        dtype=args.dtype,
        attn_implementation="sdpa",
        local_files_only=True,
        eval_mode=True,
    )
    added = 0 if PLAN_TOKEN in processor.tokenizer.get_vocab() else processor.tokenizer.add_tokens([PLAN_TOKEN], special_tokens=True)
    plan_token_id = processor.tokenizer.convert_tokens_to_ids(PLAN_TOKEN)
    if added:
        model.resize_token_embeddings(len(processor.tokenizer))

    adapter_dir = args.checkpoint_dir / "qwen3vl_lora_or_model"
    if (adapter_dir / "adapter_config.json").exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    else:
        model = model.from_pretrained(adapter_dir, local_files_only=True)

    embedding_path = args.checkpoint_dir / "plan_token_embedding.pt"
    if embedding_path.exists():
        vec = torch.load(embedding_path, map_location="cpu", weights_only=False)
        model.get_input_embeddings().weight.data[plan_token_id].copy_(vec.to(dtype=model.get_input_embeddings().weight.dtype))

    hidden_size = int(model.config.text_config.hidden_size)
    wrapper = PlannerWrapper(
        model=model,
        hidden_size=hidden_size,
        semantic_dim=int(meta["semantic_dim"]),
        plan_token_id=plan_token_id,
    )
    state = torch.load(args.checkpoint_dir / "plan_head.pt", map_location="cpu", weights_only=False)
    wrapper.plan_head.load_state_dict(state)
    wrapper.to(device)
    wrapper.eval()
    return wrapper, processor


@torch.no_grad()
def predict(wrapper: PlannerWrapper, batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    target = batch.pop("semantic_plan_labels").to(device=device, dtype=torch.float32)
    batch.pop("stems", None)
    model_dtype = next(wrapper.model.parameters()).dtype
    batch = move_qwen_inputs_to_device(batch, device, model_dtype=model_dtype)
    outputs = wrapper.model(**batch, output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[-1]
    input_ids = batch["input_ids"]
    plan_mask = input_ids == wrapper.plan_token_id
    bsz, plan_len, sem_dim = target.shape
    pred = hidden.new_zeros((bsz, plan_len, sem_dim), dtype=torch.float32)
    head_dtype = next(wrapper.plan_head.parameters()).dtype
    for b in range(bsz):
        h = hidden[b, plan_mask[b]]
        if h.shape[0] != plan_len:
            raise RuntimeError(f"Found {h.shape[0]} plan tokens, expected {plan_len}")
        pred[b] = wrapper.plan_head(h.to(dtype=head_dtype)).float()
    return pred, target


def sample_retrieval_top1(pred: torch.Tensor, target: torch.Tensor) -> float:
    correct = 0
    total = 0
    for p, t in zip(pred, target, strict=True):
        p = F.normalize(p, dim=-1)
        t = F.normalize(t, dim=-1)
        nn_idx = (p @ t.t()).argmax(dim=-1)
        correct += int((nn_idx == torch.arange(t.shape[0], device=t.device)).sum().item())
        total += int(t.shape[0])
    return correct / max(total, 1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    meta = load_meta(args.checkpoint_dir)
    wrapper, processor = restore_planner(args, meta, device)

    dataset = SemanticPlanDataset(
        dataset_root=args.dataset_root,
        label_dir=args.plan_label_dir,
        num_keyframes=int(meta["num_keyframes"]),
        grid_size=int(meta["grid_size"]),
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=Collator(
            processor=processor,
            plan_token=PLAN_TOKEN,
            plan_len=int(meta["num_keyframes"]) * int(meta["grid_size"]) * int(meta["grid_size"]),
        ),
        pin_memory=True,
    )

    n_batches = 0
    n_tokens = 0
    mse_sum = 0.0
    cos_sum = 0.0
    retrieval_sum = 0.0
    pred_norm_sum = 0.0
    target_norm_sum = 0.0
    for batch in loader:
        pred, target = predict(wrapper, batch, device)
        n_batches += 1
        n_tokens += int(pred.shape[0] * pred.shape[1])
        mse_sum += float(F.mse_loss(pred, target, reduction="sum").item())
        pred_flat = pred.flatten(0, 1)
        target_flat = target.flatten(0, 1)
        cos = F.cosine_similarity(pred_flat, target_flat, dim=-1)
        cos_sum += float(cos.sum().item())
        retrieval_sum += sample_retrieval_top1(pred, target) * pred.shape[0]
        pred_norm_sum += float(pred_flat.norm(dim=-1).sum().item())
        target_norm_sum += float(target_flat.norm(dim=-1).sum().item())

    denom = max(n_tokens, 1)
    summary = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "num_samples": len(dataset),
        "num_tokens": n_tokens,
        "mse_per_value": mse_sum / max(denom * int(meta["semantic_dim"]), 1),
        "mean_cosine": cos_sum / denom,
        "sample_retrieval_top1": retrieval_sum / max(len(dataset), 1),
        "pred_norm": pred_norm_sum / denom,
        "target_norm": target_norm_sum / denom,
        "meta": meta,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
