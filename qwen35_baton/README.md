# Qwen3.5 continuous Baton planner

This package trains Qwen3.5-2B to predict four future SigLIP2 semantic grids
from the current robot observation and instruction. The WorldArena Stage-1
route preserves the existing model topology and Equation-8 MSE objective while
using the repaired Baton input/data contract.

## WorldArena contract

- Qwen3.5-2B, including the Qwen vision encoder, is trainable.
- The frozen teacher is `SigLIP2-large-patch16-256`, penultimate layer.
- Every target has shape `[1, 4, 256, 1024]`: one head camera, four future
  frames, and a 16×16 patch grid per frame.
- Each Qwen row is a system/user/assistant conversation. The current image,
  discriminative instruction, current frame index, and four target indices are
  in the user message; the 1,024 `<PLAN_PAD>` tokens are in the assistant
  blueprint.
- The shared 169-character WorldArena generation boilerplate is removed only
  at collation time. The original instruction remains in batch and validation
  provenance.
- Training uses `all_windows_v1`: every episode contributes all 117 valid
  current frames. With 465 train episodes this is 54,405 examples, about 426
  optimizer steps per epoch at global batch 128.

The production recipe is
[`configs/worldarena_stage1.json`](configs/worldarena_stage1.json): 5,000
optimizer steps (about 11.7 exhaustive epochs), bf16, AdamW, learning rate
`1e-5`, and global batch 128 on eight GPUs.

## Checkpoints and validation

The run saves step 20 plus steps 500, 1000, 2000, 3000, 4000, and 5000. A
deterministic pass over all 44 validation episodes runs every 500 steps and
compares:

1. the correct instruction;
2. a task-distinct shuffled instruction with the same current image;
3. the current-feature persistence baseline.

Each pass writes an atomic adjacent artifact named
`step_XXXXXX.grounding_validation.json`. A checkpoint is eligible for the
follow-on GE-Act semantic stages only when the same pass has 44 finite examples,
at least 60% correct-over-shuffle wins, at least 5% MSE improvement over
shuffle, at least 25% MSE improvement over persistence, and a prediction/target
norm ratio in `[0.85, 1.15]`. Failure of these quality gates does not extend the
run past 5,000 steps.

Validation rows are materialized once, use no DataLoader workers, and close all
HDF5 files before training. This avoids the worker FD/IPC failure seen in the
older 30,000-step run. The shuffled branch discards future targets so it does
not duplicate teacher preprocessing or GPU transfer.

## Compatibility

New checkpoints use metadata format 5 and record:

- `input_template_kind=baton_assistant_time_v2`;
- `worldarena_sampling_kind=all_windows_v1`;
- `instruction_rendering_kind=strip_worldarena_boilerplate_v1`.

Format-4 checkpoints remain loadable and are interpreted as the exact legacy
`legacy_user_plan_v1` / `episode_random_v1` / `verbatim_v1` behavior. They are
not silently resumed into a new-format run because the runtime and checkpoint
behavior contracts must match.

## Launch

Fill the local model, cache, and SHA-256 fields in the tracked JSON, then run:

```bash
bash qwen35_baton/scripts/train_worldarena_semantic_planner.sh
```

The launcher defaults to eight GPUs, per-device batch 2, and gradient
accumulation 8, which gives global batch 128. Run the CPU-only preflight before
a long job when deploying to a new server.

GE-Act Stage 2/3 is intentionally not started by this Stage-1 command. First
select a planner checkpoint that passes the grounding gates, then train GE-Act
with ground-truth semantic features followed by the frozen accepted planner's
predicted features.
