# LingBot Planner ZeRO-2 Training and Launcher Cleanup Design

## Goal

Replace the current hand-written DDP runtime for the Qwen3-VL 4B LingBot
planner with the proven StarVLA-style Accelerate + DeepSpeed ZeRO-2 runtime,
while preserving the current planner objective and reducing the accumulated
experiment launchers to one generic launcher plus one POD profile and one HPC3
profile.

The target run remains:

- one current frame plus one future frame at offset 8 from a nine-frame sample;
- four independent 64-token VLM query groups, totaling 256 task tokens;
- current DINO, future DINO, current depth, and future depth supervision;
- four independent prediction heads, each producing 256 × 1024 features;
- all four loss weights equal to 0.004;
- full Qwen language-model fine-tuning with the vision tower and LM head frozen;
- BF16, gradient checkpointing disabled, 12,000 optimizer steps;
- eight GPUs, per-GPU batch 8, gradient accumulation 2, global batch 128.

## Chosen approach

Keep the existing model, dataset, online teachers, loss computation, checkpoint
contract, and FastWAM provider intact. Replace only the distributed runtime and
launcher layer.

This is preferred over merely adding `DDP.no_sync()` because ZeRO-2 also shards
optimizer and gradient state, which is the memory saving needed to test batch 8.
It is preferred over transplanting the whole StarVLA trainer because this
repository has specialized online DINO/MoGe/MoRGBD teachers and a production
FastWAM checkpoint format that StarVLA does not implement.

## Distributed runtime

`train_qwen3vl4b_lingbot_dino_planner.py` will construct `Accelerator` only
after parsing command-line arguments. The requested gradient accumulation value
will therefore be available when the runtime is created instead of becoming a
dead configuration value at module import time.

The runtime will:

1. build the planner, optimizer, and DataLoader as ordinary PyTorch objects;
2. pass them through `accelerator.prepare`;
3. let Accelerate shard the DataLoader instead of installing a second manual
   `DistributedSampler`;
4. use `accelerator.backward` for all backward calls;
5. bypass `accelerator.accumulate(model)` under DeepSpeed ZeRO-2 because its
   `no_sync` context is incompatible with ZeRO-2 gradient partitioning;
6. let the DeepSpeed engine perform loss scaling, gradient accumulation,
   clipping, optimizer updates, and gradient clearing;
7. count microsteps explicitly and only advance the optimizer-step counter,
   scheduler, progress bar, logging, and checkpoint schedule at accumulation
   boundaries;
8. use the normal `accelerator.accumulate(model)` path when the trainer is run
   without DeepSpeed, so single-GPU local smoke tests remain supported;
9. synchronize ranks around checkpoint saving and export from
   `accelerator.unwrap_model` on the main process;
10. retain the existing FastWAM checkpoint directory and metadata layout.

The runtime will log its distributed type, world size, per-GPU batch,
accumulation count, computed global batch, mixed precision, and ZeRO stage before
loading the large online teachers. It will fail before model training if the
trainer, Accelerator, and DeepSpeed accumulation values disagree or if the
computed global batch differs from the requested 128.

## DeepSpeed configuration

A small Python generator will create a matched pair of Accelerate and DeepSpeed
configuration files for each launch. The generated DeepSpeed configuration will
use:

- ZeRO stage 2;
- BF16 enabled and FP16 disabled;
- the requested numeric gradient accumulation value;
- automatic train micro/global batch values;
- reduce-scatter and all-gather partitioning;
- communication overlap and contiguous gradients;
- 500 MB reduce/all-gather buckets;
- gradient clipping at 1.0;
- no CPU optimizer or parameter offload.

The generated files will be stored below the run output directory so each run
records the exact runtime configuration. FlashAttention 2 is not part of this
change because the validated POD environment does not currently provide
`flash_attn`; the trainer keeps SDPA.

## Launcher organization

The existing
`scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh`
becomes the sole generic launcher. It owns the current training defaults and
invokes Accelerate with the generated ZeRO-2 configuration.

Two thin machine profiles remain:

- a POD profile containing `/root/nas/junjie` paths and direct eight-GPU launch
  defaults;
- an HPC3 Slurm profile containing the `jhe724` workspace paths and resource
  directives.

Both profiles call the same generic launcher and default to batch 8,
accumulation 2, and global batch 128. Hyperparameters remain overridable through
environment variables for smoke tests. A two-step smoke run is required before
using batch 8 for a formal run; batch 4 with accumulation 4 is the fallback if
the ZeRO-2 smoke still exceeds memory.

## Cleanup scope

The following experiment-specific wrappers are replaced and removed:

- `train_lingbot_current_future_fastwam_k1.sh`
- `train_lingbot_current_future_fastwam_k1_pod30274.sh`
- `train_lingbot_current_future_fastwam_k1_hpc3.sbatch`
- `train_lingbot_dino_depth_fastwam_k4.sh`
- `train_lingbot_dino_depth_fastwam_k4_hpc3.sbatch`
- `train_lingbot_independent_queries_fastwam_k1_pod30274.sh`
- `train_lingbot_independent_queries_fastwam_k1_hpc3.sbatch`

Tests that only assert literal text in those deleted wrappers will also be
removed. Tests of observable behavior remain, including teacher batching,
four-way query slicing, 64-token geometry, warm-start routing, four-term loss,
checkpoint metadata, provider validation, and FastWAM loading.

The depth probe and planner evaluation scripts and their tests remain because
they are still used to visualize and evaluate trained checkpoints. The 2B CoVT
and task-token training lines and all FastWAM runtime/evaluation scripts are
outside this cleanup scope.

## Testing and acceptance

Implementation follows test-first development. New focused tests will cover:

- generation of matched ZeRO-2 and Accelerate configs;
- batch 8 × accumulation 2 × eight GPUs = global batch 128;
- hard failures for accumulation or global-batch mismatches;
- correct optimizer-boundary detection for both DeepSpeed and non-DeepSpeed
  execution;
- use of unwrapped planner modules for checkpoint export;
- preservation of the current 4 × 64 query and checkpoint contracts;
- syntax and invocation of the generic, POD, and HPC3 launchers;
- absence of references to deleted launchers.

Local acceptance requires the focused planner/FastWAM tests, Python compilation,
shell syntax checks, and `git diff --check` to pass. Remote acceptance requires a
two-optimizer-step eight-GPU POD smoke run with batch 8 and accumulation 2,
correct global-batch/runtime logging, no OOM or distributed error, and a
loadable step-2 FastWAM planner checkpoint. The existing long-running POD job is
not stopped or modified by this implementation.

