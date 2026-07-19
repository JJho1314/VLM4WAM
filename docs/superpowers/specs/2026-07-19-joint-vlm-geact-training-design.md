# Joint Qwen3-VL and GE-Act Training Design

## Goal

Jointly fine-tune the complete Qwen3-VL planner and the GE-Act LTX video model on
LIBERO. The GE-Act video loss must backpropagate through the semantic cross-attention
conditioning into Qwen3-VL, while the planner's original SigLIP2 and four-layer DA3
alignment losses preserve its future-feature representation.

The run starts from:

- VLM planner: dual-camera K4 WSA `step_030000`.
- GE-Act: OLA `ltx_step_50000`.
- Data: verified LIBERO FastWAM predecoded RGB cache, with separate main and wrist
  cameras and `require_predecoded=true`. OLA currently has no compatible FastWAM
  HDF5 manifest, so this recipe performs no online video decode and does not claim
  an HDF5 backend.

## Architecture

Add one trainable composite module owned by a single Accelerate/DeepSpeed engine. It
contains the exported `PlannerWrapper` and the LTX transformer. This avoids two
independent distributed engines and gives DeepSpeed one complete parameter set to
partition, synchronize, clip, save, and restore.

For each sample:

1. Select the last observation frame from both cameras and build the Qwen3-VL input.
2. Run Qwen3-VL once and decode the planner heads into:
   - SigLIP2 features `[B,2,4,256,1024]`.
   - DA3 WSA features `[B,2,4*256,4,2048]`.
3. Pass the SigLIP2 tensor, without `detach`, to GE-Act semantic cross-attention in all
   28 LTX blocks at normalized K4 times for offsets `2,4,6,8`.
4. Use the same future RGB frames to compute frozen online SigLIP2 targets and frozen
   DA3 layers `11,15,19,23`.
5. Compute one combined loss and one backward pass through the composite module.

The current provider's K1-only shape assertion must become metadata-driven so K1
legacy checkpoints remain valid while the joint run requires K4.

## Losses

The total objective is:

```text
total_loss = video_loss + 0.1 * planner_alignment_loss
```

`planner_alignment_loss` reuses the checkpoint-compatible planner objective:

- future SigLIP2 MSE weight: `1.0`;
- future DA3 WSA alignment weight: `0.004`;
- DA3 per-layer weights: `1.0,1.2,1.4,1.6` for layers `11,15,19,23`.

Semantic conditioning dropout affects only the GE-Act conditioning branch. It must not
drop or mask the planner auxiliary supervision. Frozen SigLIP2 and DA3 teachers never
receive gradients or enter the optimizer.

The planner API must return semantic predictions, depth predictions, and alignment
losses from the same Qwen forward. Running a second Qwen forward for the auxiliary
loss is forbidden.

## Trainable Parameters and Optimizer

All Qwen3-VL parameters are trainable, including its visual tower and language output
head. Use four optimizer groups:

| Group | Initial learning rate |
|---|---:|
| LTX base parameters | `2e-5` |
| LTX semantic conditioning modules | `1e-4` |
| Qwen3-VL backbone, visual tower, and LM head | `1e-6` |
| Planner query embeddings, plan head, and depth head | `3e-5` |

Use AdamW, weight decay `1e-5`, betas `0.9/0.95`, maximum gradient norm `1.0`, and a
constant schedule with 1,000 warmup optimizer steps. Gradient norm clipping covers
both VLM and LTX parameters.

## Distributed and Memory Contract

The formal configuration uses 8 H100 GPUs, bf16, TF32, DeepSpeed ZeRO-2, per-device
batch size `1`, and gradient accumulation `16`, for global batch size `128`.
Gradient checkpointing is enabled for both Qwen3-VL and LTX. VAE, T5, SigLIP2, and
DA3 remain frozen.

Accelerate accumulation must operate on the composite model. Calling a planner method
outside the prepared composite forward would bypass distributed synchronization and
is an error.

## Checkpoint Contract

Each keeper step stores a self-contained joint checkpoint with:

- LTX transformer weights and config;
- an exported planner directory compatible with the existing standalone planner
  loader;
- DeepSpeed optimizer and scheduler state for exact resume;
- trainer progress metadata (global step, epoch, next prepared-dataloader batch,
  world size, accumulation, and batches per epoch), restored only after
  `accelerator.prepare` and validated before `accelerator.load_state`;
- joint metadata recording source checkpoints, loss weights, all optimizer-group
  learning rates, K4 geometry, global batch size, and trainable parameter counts.

The default keeper steps remain `20000`, `25000`, and `30000`. A final checkpoint is
also saved if training stops at another requested step. Existing frozen-planner
checkpoints and inference paths remain loadable.

## Configuration and Launching

Add a separate joint-training YAML and launcher. Do not repurpose or delete the
existing frozen VLM-planner GE-Act configuration. Preflight validates:

- planner metadata is dual-camera K4 WSA with offsets `2,4,6,8`;
- GE-Act semantic geometry is two views, four keyframes, 256 tokens, and width 1024;
- both initialization checkpoints and online teacher checkpoints exist;
- the configured world size, per-device batch, and accumulation produce global batch
  128;
- full-Qwen mode, both gradient-checkpointing flags, and the four optimizer-group LRs
  match the joint contract.

## Verification

Unit and integration tests must prove:

1. K1 and K4 provider shapes are metadata-driven and backward compatible.
2. One planner forward returns K4 semantic/depth predictions and auxiliary losses.
3. With the real zero-initialized semantic gate, the first combined step updates
   Qwen through planner alignment and opens the LTX semantic gate; on the following
   step, GE-Act video loss produces finite non-zero gradients on Qwen and LTX
   semantic cross-attention parameters.
4. Frozen SigLIP2 and DA3 teachers have no gradients.
5. Optimizer groups are disjoint, complete, and use the specified learning rates.
6. Joint save/load preserves both planner and LTX outputs and resumes optimizer,
   scheduler, global step, epoch, and dataloader position without changing the
   distributed batch geometry.
7. A one-GPU one-step smoke passes before an 8-GPU ten-step smoke; the formal launch
   is permitted only after both smoke tests have finite losses and bounded memory.

The first formal run should log total, video, planner semantic, planner depth, and
per-layer DA3 losses, all four learning rates, VLM/LTX gradient norms, throughput, and
peak GPU memory.
