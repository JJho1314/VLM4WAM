# Qwen3.5 Baton Training Throughput Design

## Goal

Accelerate the eight-GPU Qwen3.5-2B Baton WorldArena Stage-1 run while preserving its training objective, 1,024-token four-frame plan geometry, global batch size of 128, bf16 numerical policy, online frozen SigLIP2 targets, and checkpoint reproducibility contract.

## Scope and invariants

- Keep four future frames and 256 SigLIP2 tokens per frame.
- Keep the Stage-1 loss, learning rate, warmup, optimizer-step count, and global batch size unchanged.
- Keep Qwen3.5 and the Baton query tower jointly trainable.
- Keep gradient checkpointing disabled in the preferred fast configuration.
- Do not add an offline SigLIP feature cache or increase persistent dataset storage.
- Preserve deterministic sample order and resumable cursor semantics.
- Do not launch a long training run until short throughput probes select a stable configuration.

## Root cause

The stopped WorldArena run used per-device batch 2 and gradient accumulation 8. Its measured microbatch time was dominated by backward and Qwen, while data loading was about 1.3 percent. The training loop also synchronizes CUDA repeatedly for timing and validation and converts several GPU metrics to CPU scalars on every microbatch. The frozen SigLIP2 path moves already-device-resident uint8 frames back to CPU for preprocessing and then returns them to the GPU.

VLAForge's useful pattern is to maximize the physical microbatch, minimize gradient accumulation, use bf16 ZeRO-2 only when necessary to unlock memory, and overlap pinned host-to-device transfers. Its absolute batch size is not directly portable because its QwenQuery recipe uses 64 query tokens whereas this planner sends 1,024 plan tokens through Qwen.

## Design

### 1. Production-safe metrics and validation

Add a production timing mode that keeps per-step training asynchronous:

- Accumulate loss and per-frame MSE as detached GPU tensors.
- Materialize and reduce metrics only at the configured logging boundary.
- Use CUDA events for optional detailed profiling windows instead of unconditional `torch.cuda.synchronize()` calls.
- Register profiling hooks once for a profiling window, not once per microbatch.
- Keep fail-fast non-finite loss handling at optimizer-update boundaries.
- Move static plan-position contract checks to the CPU collator/preflight path. Retain a debug runtime-validation mode for tests and diagnosis.

The default production path must produce the same loss and gradients as the current path for the same batch and parameters.

### 2. Fixed-global-batch throughput sweep

Provide a short-run launcher that benchmarks these eight-GPU configurations without changing global batch 128:

- per-device 4, accumulation 4;
- per-device 8, accumulation 2;
- per-device 16, accumulation 1.

Each trial performs warmup before recording steady-state samples per second, peak allocated/reserved memory, utilization-compatible step timing, and OOM status. It writes results to a separate benchmark directory and never overwrites training checkpoints. The selected production configuration is the fastest stable candidate with memory headroom; batch 16 is not assumed to fit.

### 3. Optional ZeRO-2 memory path

Wire the existing DeepSpeed configuration into `Accelerator` as an explicit optional runtime path. Use ZeRO stage 2, bf16, overlapping communication, contiguous gradients, and no CPU/NVMe offload. Ordinary DDP remains supported.

ZeRO-2 is enabled for the production run only if the DDP sweep cannot reach the best physical microbatch because of memory. Checkpoint save/resume must remain fail-closed: a checkpoint records the distributed strategy and cannot silently resume under an incompatible strategy.

### 4. Exact online SigLIP preprocessing with overlap

Keep SigLIP targets online, but move the released SigLIP2 image processor into DataLoader workers:

- The collator produces the exact processor `pixel_values` for all future frames as one batched operation.
- The batch carries processor outputs instead of sending raw future images from GPU back to CPU.
- Enable pinned memory and a dedicated CUDA transfer stream following VLAForge's device-prefetch pattern.
- The teacher gains an `encode_pixel_values` entry point and retains `encode_future` for compatibility and tests.
- Equivalence tests compare worker-produced inputs and teacher features with the existing released processor path.

Qwen image/text processor calls are also batched across camera rows when the released processor produces outputs equivalent to the existing row-wise path.

## Error handling

- Any non-finite synchronized loss or gradient update aborts with the current fail-closed behavior.
- Batch/global-batch mismatches fail during preflight.
- Throughput trials record OOM and continue to the next smaller candidate only through the launcher, never by mutating a running job.
- DeepSpeed configuration mismatches and incompatible checkpoint strategy metadata fail before model preparation.
- Processor-equivalence failures keep the original online teacher path available and block enabling worker preprocessing.

## Testing and acceptance

Implementation follows red-green-refactor tests for each unit. Acceptance requires:

1. Existing Baton unit tests remain green.
2. Optimized and reference paths produce matching loss and gradients on deterministic tiny inputs.
3. Worker-preprocessed SigLIP pixel values and features match the current path within the bf16 contract.
4. DDP and optional ZeRO-2 global-batch validation both resolve to 128.
5. Checkpoint strategy mismatch is rejected before loading optimizer state.
6. A short eight-GPU sweep reports throughput and memory for every attempted candidate.
7. The chosen configuration improves steady-state samples per second over batch 2 / accumulation 8 without changing the training objective.

## Rollout order

1. Remove hot-path synchronization and CPU scalarization.
2. Run the DDP batch 4/8/16 sweep.
3. Add and test exact worker-side SigLIP preprocessing and asynchronous transfer.
4. Re-run the sweep.
5. Enable ZeRO-2 only if it unlocks a faster stable physical batch.
6. Resume or restart long training only after checkpoint compatibility is explicitly selected.
