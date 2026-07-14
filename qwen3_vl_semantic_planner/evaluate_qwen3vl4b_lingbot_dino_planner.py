#!/usr/bin/env python3
"""Standalone held-out eval for the 4B lingbot-DINO planner (independent script; trainer untouched).

At inference the planner sees ONLY the current frame + language instruction and predicts the
5-keyframe DINO-video plan [B, 1280, 1024]. We score that predicted plan against the DINO-video
teacher's TRUE future features (the same online teacher used in training) and report regression
metrics — the real test of whether the plan captures the future (and whether 8 latents/keyframe
suffice), since here the plan is genuinely PREDICTED, not fit with a live gradient.

Reuses the trainer verbatim by import (dataset / wrapper / teacher / compute_plan_losses), so the
numbers are directly comparable to the training-time diagnostics. Differences vs training: a fixed
held-out window set, no grad, and weights loaded from a saved checkpoint.

NOTE on data: there is currently no 10Hz/320x576 held-out split, so by default this evaluates on
the SAME (train) dataset root with a distinct seed -> in-distribution (seen stems), which measures
fit + inference-time plan quality but NOT generalization. Point --dataset-root at a true 10Hz
held-out set once one exists for a generalization number.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_qwen3vl4b_lingbot_dino_planner as T  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, required=True, help="a step_0XX000/ dir")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--frame-ranges-json", type=Path, default=None)
    p.add_argument("--dino-teacher-ckpt", type=Path, required=True)
    p.add_argument("--dino-teacher-config", type=Path, required=True)
    p.add_argument("--dino-input-size", type=int, default=256)
    p.add_argument("--num-keyframes", type=int, default=5)
    p.add_argument("--grid-size", type=int, default=16)
    p.add_argument("--num-latent-per-keyframe", type=int, default=8)
    p.add_argument("--semantic-dim", type=int, default=1024)
    p.add_argument("--sequence-length", type=int, default=49)
    p.add_argument("--keyframe-scheme", default="uniform")
    p.add_argument("--keyframe-gamma", type=float, default=0.6)
    p.add_argument("--keyframe-offsets", type=str, default="", help="explicit offsets; falls back to ckpt meta")
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=4321, help="distinct from the train seed -> fresh windows")
    p.add_argument("--output-json", type=Path, required=True)
    return p.parse_args()


def load_planner(ckpt: Path, args: argparse.Namespace, device: torch.device, *, use_depth: bool = False):
    """Load the fine-tuned VLM (full-FT saved via save_pretrained) + the trained video plan head.

    With use_depth=True the wrapper also gets a depth head; its weights are loaded from
    ckpt/depth_head.pt (or depth_head_refit.pt) when present, else left for the caller to init."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    meta = json.loads((ckpt / "planner_meta.json").read_text())
    # geometry from ckpt meta BEFORE wrapper construction (CLI values are only old-ckpt fallbacks):
    # K / grid / semantic_dim drive target_len and the latent-group layout.
    args.num_keyframes = int(meta.get("num_keyframes", args.num_keyframes))
    args.grid_size = int(meta.get("grid_size", args.grid_size))
    args.semantic_dim = int(meta.get("semantic_dim", args.semantic_dim))
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    # processor was saved to ckpt/processor (already carries the <|sem_plan_i|> tokens);
    # the model to ckpt/qwen3vl_lora_or_model (full-FT, vocab already resized).
    processor = AutoProcessor.from_pretrained(str(ckpt / "processor"), local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(ckpt / "qwen3vl_lora_or_model"),
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.config.use_cache = False
    model.to(device).eval()

    hidden_size = int(model.config.text_config.hidden_size)
    wrapper = T.PlannerWrapper(
        model=model,
        hidden_size=hidden_size,
        semantic_dim=args.semantic_dim,
        plan_token_ids=[int(x) for x in meta["plan_token_ids"]],
        target_len=args.num_keyframes * args.grid_size * args.grid_size,
        num_keyframes=args.num_keyframes,
        grid_size=args.grid_size,
        # latent geometry comes from the checkpoint meta when present (new grouped-latent runs
        # record it); CLI args are only the fallback for old checkpoints.
        num_latent_per_keyframe=int(meta.get("num_latent_per_keyframe", args.num_latent_per_keyframe)),
        num_head_latent_per_keyframe=int(meta.get("num_head_latent_per_keyframe", 0)),
        use_current=bool(meta.get("use_current", False)),
        bidirectional_plan_attn=bool(meta.get("bidirectional_plan_attn", False)),
        plan_head_type="lingbot_dino",
        mse_loss_weight=1.0,
        cosine_loss_weight=0.0,
        norm_loss_weight=0.0,
        variance_loss_weight=0.0,
        infonce_loss_weight=0.0,
        use_depth=use_depth,
    ).to(device)
    state = torch.load(ckpt / "plan_head.pt", map_location="cpu", weights_only=False)
    report = wrapper.plan_head.load_state_dict(state, strict=True)
    depth_report = "no_depth"
    if use_depth and wrapper.depth_head is not None:
        for name in ("depth_head.pt", "depth_head_refit.pt"):
            if (ckpt / name).exists():
                dstate = torch.load(ckpt / name, map_location="cpu", weights_only=False)
                depth_report = f"{name}: {wrapper.depth_head.load_state_dict(dstate, strict=True)}"
                break
        else:
            depth_report = "depth_head UNINITIALIZED (no depth_head*.pt in ckpt)"
    wrapper.eval()
    print(json.dumps({"loaded_ckpt": str(ckpt), "plan_head_load": str(report),
                      "depth_head_load": str(depth_report)}), flush=True)
    return wrapper, processor, meta


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.frame_ranges_json is None:
        args.frame_ranges_json = args.dataset_root / "frame_ranges.json"

    wrapper, processor, meta = load_planner(args.checkpoint_dir, args, device)
    # plan geometry comes from the checkpoint meta (single source of truth for K / grid / offsets)
    args.num_keyframes = int(meta.get("num_keyframes", args.num_keyframes))
    args.grid_size = int(meta.get("grid_size", args.grid_size))
    _offs = str(args.keyframe_offsets or meta.get("keyframe_offsets", "") or "")
    offsets_override = [int(x) for x in _offs.split(",") if x.strip()] or None

    dino_encoder = T.DinoVideoTargetEncoder(
        ckpt_path=args.dino_teacher_ckpt,
        config_path=args.dino_teacher_config,
        input_size=args.dino_input_size,
        device=device,
    )

    latent_len = wrapper.latent_len  # from ckpt meta (handles grouped shared+own latents)
    plan_sequence = [f"<|sem_plan_{i}|>" for i in range(latent_len)]
    dataset = T.OnlineSemanticPlanDataset(
        dataset_root=args.dataset_root,
        frame_ranges_json=args.frame_ranges_json,
        num_keyframes=args.num_keyframes,
        sequence_length=args.sequence_length,
        keyframe_scheme=args.keyframe_scheme,
        keyframe_gamma=args.keyframe_gamma,
        max_samples=args.max_samples,
        seed=args.seed,
        offsets_override=offsets_override,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=T.Collator(processor=processor, plan_sequence=plan_sequence),
        pin_memory=True,
    )

    K = args.num_keyframes
    agg: dict[str, float] = {}
    cos_per_kf = [0.0] * K
    mse_per_kf = [0.0] * K
    retr_per_kf = [0.0] * K  # top-1 retrieval WITHIN each keyframe's own token pool (fair across K)
    n = 0
    model_dtype = next(wrapper.model.parameters()).dtype
    with torch.no_grad():
        for batch in loader:
            batch.pop("stems", None)
            keyframes = batch.pop("keyframe_images", None)
            current = batch.pop("current_image", None)
            if keyframes is None or current is None:
                continue
            inputs = T.move_qwen_inputs_to_device(batch, device, model_dtype=model_dtype)
            pred = wrapper.predict_semantic_plan(**inputs)  # (B, K*256, 1024), float

            cur = current.permute(0, 3, 1, 2).contiguous()
            kfs = [keyframes[:, j].permute(0, 3, 1, 2).contiguous() for j in range(keyframes.shape[1])]
            target = dino_encoder.encode_future_keyframes(cur, kfs).float().to(pred.device)

            bs = target.shape[0]
            metrics = wrapper.compute_plan_losses(pred, target)
            for k, v in metrics.items():
                if k == "loss":
                    continue
                agg[k] = agg.get(k, 0.0) + float(v) * bs

            tok = pred.shape[1] // K
            pred_k = pred.reshape(bs, K, tok, pred.shape[-1])
            tgt_k = target.reshape(bs, K, tok, target.shape[-1])
            for i in range(K):
                cos_per_kf[i] += float(
                    F.cosine_similarity(pred_k[:, i].flatten(0, 1), tgt_k[:, i].flatten(0, 1), dim=-1).mean()
                ) * bs
                mse_per_kf[i] += float(F.mse_loss(pred_k[:, i], tgt_k[:, i])) * bs
                pd = F.normalize(pred_k[:, i], dim=-1)
                td = F.normalize(tgt_k[:, i], dim=-1)
                sims = torch.bmm(pd, td.transpose(1, 2))  # (B, tok, tok) within this keyframe only
                hits = sims.argmax(dim=-1) == torch.arange(pred_k.shape[2], device=pd.device)
                retr_per_kf[i] += float(hits.float().mean()) * bs
            n += bs
            print(f"[eval] {n}/{args.max_samples} samples", flush=True)

    denom = max(n, 1)
    result = {
        "checkpoint": str(args.checkpoint_dir),
        "dataset": str(args.dataset_root),
        "num_samples": n,
        "num_keyframes": K,
        "metrics": {k: v / denom for k, v in agg.items()},
        "per_keyframe_cosine_sim": [cos_per_kf[i] / denom for i in range(K)],
        "per_keyframe_mse": [mse_per_kf[i] / denom for i in range(K)],
        "per_keyframe_retrieval_top1": [retr_per_kf[i] / denom for i in range(K)],
        "note": "in-distribution (train stems, fresh seed) unless --dataset-root is a true held-out set",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
