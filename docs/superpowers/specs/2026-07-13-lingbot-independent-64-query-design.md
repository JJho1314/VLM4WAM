# LingBot Independent 64-Query Alignment Design

## Goal

Train the LIBERO Qwen3-VL 4B planner with separate VLM task-token capacity for every aligned target. The four targets are current DINO, future DINO, current depth, and future depth.

## Token contract

- Each target owns 64 private task tokens.
- The prompt contains 256 task tokens in this order: current DINO `[0:64]`, future DINO `[64:128]`, current depth `[128:192]`, and future depth `[192:256]`.
- `latent_len` and `total_unique_latent_per_keyframe` are both 256 for this configuration.
- The four existing prediction heads stay independent and continue producing 256 teacher-feature tokens per target. Only their VLM task-token input groups change from 8 to 64.

## Compatibility

The group size is an explicit training/checkpoint parameter rather than a new global constant. Existing shared 16-token and independent 32-token checkpoints remain loadable by inferring the old contract when the new metadata field is absent. New checkpoints record the independent-mode flag, `task_tokens_per_group=64`, total length 256, and the exact four-group layout.

## Training

The HPC3 run uses one eight-GPU Slurm node, batch 16 per GPU, gradient accumulation 1, global batch 128, gradient checkpointing disabled, 12,000 steps, and the previously approved learning rates and four loss weights. A two-step smoke job runs first; the formal job has an `afterok` dependency on the smoke job.

## Verification

Tests must cover the four 64-token slices, wrapper geometry, dynamic checkpoint metadata/provider validation, legacy checkpoint compatibility, and the HPC3 launcher contract. Remote verification checks syntax/imports, successful smoke forward/backward, non-OOM memory behavior, and exported 256-token metadata before relying on the formal run.
