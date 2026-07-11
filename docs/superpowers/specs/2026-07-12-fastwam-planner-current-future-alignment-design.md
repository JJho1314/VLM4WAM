# FastWAM Planner Current-and-Future Alignment Design

**Date:** 2026-07-12

## Goal

Retrain the FastWAM-aligned Qwen3-VL 4B planner with LingBot-VLA 2.0-style
current-frame DINO and depth alignment in addition to the existing four future
keyframes. Preserve the FastWAM runtime output contract: inference still returns
only the four future DINO and depth plans, each shaped `[B, 1024, 1024]`.

## Alternatives Considered

1. **Dedicated current query group and current heads — selected.** Add one
   current group before the four future groups and use separate current-DINO and
   current-depth heads. This matches LingBot's separation of current and future
   heads and avoids conflicting targets.
2. **Reuse the first future group and future heads — rejected.** The same hidden
   queries would be asked to reconstruct both the observed current frame and a
   future frame, weakening temporal specialization.
3. **Reconstruct current features directly from image tokens — rejected.** This
   omits LingBot's current task/query alignment mechanism and does not train the
   planner query pathway that FastWAM later depends on.

## Query Geometry

The Qwen input contains five temporal groups in this order:

1. current frame
2. future offset 2
3. future offset 4
4. future offset 6
5. future offset 8

Every group contains 96 unique query tokens in
`[32 shared, 32 DINO-private, 32 depth-private]` order. Each modality head sees
64 tokens per group: the 32 shared tokens plus its 32 private tokens.

- Current query tokens: `1 * 96 = 96`
- Future query tokens: `4 * 96 = 384`
- Total unique Qwen query tokens: `480`
- Current output per branch: `[B, 256, 1024]`
- Future output per branch: `[B, 4 * 256, 1024] = [B, 1024, 1024]`

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

For DINO, build the same `[warmup=current, current, future]` clips already used
by the future teacher. Return the `current_index=1` patch target once and all
four future patch targets without an extra teacher forward.

For depth, encode the current frame and four future keyframes through the same
frozen MoGe-2 to MoRGBD pipeline. Batch them together where possible. All
teacher operations remain frozen, under `torch.no_grad()`, and detached.

## Losses

Use time-point-balanced weights while strengthening depth relative to the
future-only run:

```text
total_loss =
    0.8    * future_dino_mse
  + 0.2    * current_dino_mse
  + 0.0064 * future_depth_smooth_l1
  + 0.0016 * current_depth_smooth_l1
```

The four future frames jointly receive 80% and the single current frame receives
20%, so each of the five time points has equal aggregate weight. Total DINO
weight remains `1.0`; total depth weight becomes `0.008`. Cosine, norm,
variance, and InfoNCE weights remain zero.

Log all four raw losses, all four weighted losses, current/future norm ratios,
and the final total loss separately.

## Checkpoint and Runtime Contract

Current-enabled checkpoints save both current heads, both future heads, all 480
plan-token embeddings, and metadata describing:

- current alignment enabled
- five temporal query groups and ordering
- 32 shared plus 32 private queries per branch
- all four loss weights
- future offsets `[2, 4, 6, 8]`

The provider validates the 480-token checkpoint and discards current predictions
at its public boundary. The FastWAM semantic fusion and cross-attention shapes
remain unchanged.

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

The stopped future-only run remains as a comparison checkpoint and is never
used as the initialization of this run.

## Validation and Failure Handling

Unit tests must cover query ordering and splitting, one-Qwen-forward behavior,
current/future teacher extraction, four-head warm start, weighted-loss arithmetic,
checkpoint completeness, metadata rejection, and provider future-only output.

Deployment uses a two-step full-finetune smoke with the exact production batch
configuration. Missing current head weights, target shape mismatches, malformed
480-token metadata, non-finite losses, or missing current targets fail fast.
Formal training is submitted only through a successful smoke dependency.

At steps 1,000, 2,000, and every subsequent 1,000 steps, inspect all four loss
terms and norm ratios. A healthy run has finite decreasing losses and current/
future norm ratios moving toward the `0.8–1.2` range without either modality
dominating the weighted total.
