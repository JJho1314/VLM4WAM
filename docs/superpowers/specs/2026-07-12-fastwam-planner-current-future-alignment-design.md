# FastWAM Planner Current-and-Future Alignment Design

**Date:** 2026-07-12

## Goal

Retrain the FastWAM-aligned Qwen3-VL 4B planner with LingBot-VLA 2.0-style
current-frame DINO and depth alignment plus one final future keyframe. Preserve
the FastWAM video horizon while simplifying semantic guidance: inference returns
one future DINO plan and one future depth plan, each shaped `[B, 256, 1024]`.

## Alternatives Considered

1. **Faithful LingBot current/future task-query alignment — selected.** Add one
   eight-token current task-query segment and one eight-token future segment,
   share each segment across DINO and depth, and use separate current/future
   heads. This matches LingBot's current/future separation without custom query
   expansion.
2. **Reuse the first future group and future heads — rejected.** The same hidden
   queries would be asked to reconstruct both the observed current frame and a
   future frame, weakening temporal specialization.
3. **Reconstruct current features directly from image tokens — rejected.** This
   omits LingBot's current task/query alignment mechanism and does not train the
   planner query pathway that FastWAM later depends on.

## Query Geometry

The Qwen input contains two task-query segments in this order:

1. current frame
2. future offset 8 (the last image in the FastWAM action window)

Each segment contains the original LingBot `num_task_tokens=8`. DINO and depth
share the same task-query hidden states within a temporal segment. Their output
resampler heads and 256-token output query banks remain separate.

- Current task-query tokens: `8`
- Future task-query tokens: `8`
- Total unique Qwen task-query tokens: `16`
- Current output per branch: `[B, 256, 1024]`
- Future output per branch: `[B, 256, 1024]`

The current group participates in the same single Qwen forward as all future
groups. Current heads are training-only auxiliaries. The online FastWAM provider
uses the current query group as model context but returns only future outputs.

## Heads and Warm Start

Use four independent resampler heads:

- current DINO: warm-start from `model.current_video_align_head` and
  `model.current_video_align_embs`
- future DINO: warm-start from `model.future_video_align_head` and
  `model.future_video_align_embs`
- current depth: warm-start from `model.depth_align_head` and
  `model.depth_align_embs`
- future depth: warm-start from `model.future_depth_align_head` and
  `model.future_depth_align_embs`

All four sources exist in the original LingBot-VLA 2.0 6B checkpoint. The
current-enabled training starts from the Qwen 4B base plus these head warm
starts; it does not resume the incompatible 384-query future-only checkpoint.

## Online Teacher Targets

For DINO, build one `[warmup=current, current, future@offset8]` clip. Return the
`current_index=1` patch target and the final-frame future patch target from the
same teacher forward.

For depth, encode the current frame and the one future keyframe through the same
frozen MoGe-2 to MoRGBD pipeline. Batch them together where possible. All
teacher operations remain frozen, under `torch.no_grad()`, and detached.

## Losses

Use the released LingBot-VLA 2.0 alignment weights without the custom 90/10
reweighting:

```text
total_loss =
    0.004 * future_dino_mse
  + 0.004 * current_dino_mse
  + 0.004 * future_depth_smooth_l1
  + 0.004 * current_depth_smooth_l1
```

Current and future supervision are equally weighted within each modality, as in
LingBot. The small numeric coefficient is intentional because DINO and depth
teacher features have different raw loss scales; this first run prioritizes
configuration fidelity over custom balancing. Cosine, norm, variance, and
InfoNCE weights remain zero.

Log all four raw losses, all four weighted losses, current/future norm ratios,
and the final total loss separately.

## Checkpoint and Runtime Contract

Current-enabled checkpoints save both current heads, both future heads, all 16
task-token embeddings, and metadata describing:

- current alignment enabled
- two temporal query groups and ordering
- `num_task_tokens=8` with DINO/depth sharing per temporal segment
- all four loss weights
- future offsets `[8]`

The provider validates the 16-token checkpoint and discards current predictions
at its public boundary. FastWAM receives 256 semantic tokens at normalized time
`1.0` (`offset 8 / original 8-frame future horizon`). Its video output remains
the original current-plus-eight-future, nine-frame sequence.

## Training Configuration

- 8 H100 GPUs on HPC3 under `jhe724`
- global batch 128
- preferred micro-batch 4, gradient accumulation 4
- automatic fallback to micro-batch 2, accumulation 8 after failed smoke
- 12,000 optimizer steps
- Qwen backbone LR `3e-5`
- all four alignment heads LR `3e-4`
- AdamW, weight decay `0.01`
- 1,000-step linear warmup, then cosine decay to 10% of initial LR
- bf16, gradient checkpointing, gradient clipping at 1.0
- frozen Qwen vision encoder, DINO teacher, MoGe-2, and MoRGBD teacher
- DINO `num_future_frames=1`, `use_warmup_frame=true`, and `effective_fps=1.0`

The stopped future-only run remains as a comparison checkpoint and is never
used as the initialization of this run.

## Validation and Failure Handling

Unit tests must cover query ordering and splitting, one-Qwen-forward behavior,
current/future teacher extraction, four-head warm start, weighted-loss arithmetic,
checkpoint completeness, metadata rejection, and provider future-only output.

Deployment uses a two-step full-finetune smoke with the exact production batch
configuration. Missing current head weights, target shape mismatches, malformed
16-token metadata, non-finite losses, or missing current targets fail fast.
Formal training is submitted only through a successful smoke dependency.

At steps 1,000, 2,000, and every subsequent 1,000 steps, inspect all four loss
terms, weighted contributions, and norm ratios. A healthy run has finite
decreasing losses and current/future norm ratios moving toward the `0.8–1.2`
range. Report any modality imbalance, but do not silently change the released
LingBot weights during this fidelity run.
