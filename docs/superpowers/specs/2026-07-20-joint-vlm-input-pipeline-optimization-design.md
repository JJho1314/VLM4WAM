# Joint VLM + GE-Act Strict-Equivalent Input Pipeline Optimization

## Context

The joint dual-camera K4 training path currently uses predecoded episode-level
RGB caches, but it still performs avoidable work at two boundaries:

1. `CustomLeRobotDataset` reads 512x512 `uint8` main/wrist frames, converts them
   to float32, resizes them to 256x256, normalizes them, and also reads parquet
   action/state data even though the formal run is video-only.
2. Accelerate moves the normalized video batch to the GPU. The trainer selects
   both current camera frames there, then `FrozenDualCameraVLMPlanner` copies
   them back to CPU, converts them to PIL, runs the Qwen processor, and moves the
   processor outputs back to the GPU.

Measurements on the active 8xH100 host establish the optimization budget:

- the predecoded caches are 512x512 `uint8`;
- a dataset sample takes about 132--146 ms after warm-up;
- its two-camera resize accounts for about 45 ms/sample;
- the current batch-4 PIL conversion takes about 11.3 ms/microbatch;
- the Qwen processor takes about 13.0 ms/microbatch;
- 64 persistent DataLoader workers provide roughly 12x the sample rate needed
  by the current 39.5 samples/s run.

Therefore the worker pipeline is not currently starving the GPUs. Moving the
planner preprocessing out of the rank process can reasonably recover about
2--4% throughput, but this work must not be presented as a route to a large
GPU-power increase.

## Goals

- Remove the training-rank GPU-to-CPU image round trip.
- Move the unchanged Qwen image/text processor into persistent DataLoader
  workers so it overlaps GPU compute.
- Avoid action/state parquet work when `return_action=false`.
- Reduce host-to-device video traffic only when CPU and GPU BF16 casts are
  proven bitwise identical on real samples.
- Preserve camera order, sample order, frame selection, pixel values, Qwen
  tokens, teacher targets, model inputs, losses, gradients, and resume behavior.
- Keep the original dataset and trainer path as an explicit fallback.

## Non-goals

- Do not interrupt or mutate the active 30k formal run.
- Do not change batch size, gradient accumulation, checkpointing, loss weights,
  model architecture, teacher execution, or optimizer behavior.
- Do not replace the Hugging Face Qwen processor with a hand-written tensor
  implementation in this change.
- Do not generate a new resized image cache in this change.
- Do not cache SigLIP2 or DA3 targets offline.

## Selected Architecture

Add an opt-in joint-training collator. The generic dataset remains the source of
the sampled video and caption, while the collator owns planner-only batching and
preprocessing.

```text
CustomLeRobotDataset worker
  sample main/wrist frames using the existing deterministic sampler
  -> run the existing 512->256 resize and [-1, 1] normalization
  -> omit action/state work when explicitly disabled

JointVLMGEActCollator in the same persistent worker
  default-collate the batch
  -> select current frame n_previous - 1 in main,wrist order
  -> reproduce the current BF16 planner image values
  -> call the existing PIL conversion helper
  -> call the same saved Qwen AutoProcessor and the same chat template
  -> return video, caption, and nested planner_inputs

Accelerate
  recursively move video and planner_inputs to the rank GPU once

Trainer
  use prebuilt planner_inputs when present
  -> retain the current prepare_inputs path as fallback
  -> run online teachers, Qwen, VAE, and LTX unchanged
```

### Collator lifecycle

The collator stores only checkpoint paths, immutable plan-token strings, camera
geometry, and the requested video dtype in its serialized state. Each worker
loads its own `AutoProcessor` lazily on the first `__call__`. This avoids sharing
a possibly non-fork-safe tokenizer object. `persistent_workers=true` ensures the
processor is loaded once per worker rather than once per 13-update epoch.

The collator must call the existing `build_dual_camera_planner_inputs` function.
It must not duplicate the chat template, camera flattening, plan-token suffix,
padding policy, or image-processor logic.

### Planner input transport

The collator returns the unchanged processor tensors under `planner_inputs`:

- `input_ids`;
- `attention_mask`;
- `pixel_values`;
- `image_grid_thw`;
- any additional tensor key produced by the pinned checkpoint's processor.

Accelerate handles the nested tensors. After device placement, the trainer uses
the same model-dtype conversion rules as the current
`move_qwen_inputs_to_device` path. Unknown processor keys are forwarded rather
than silently dropped.

### Optional early BF16 video cast

The current path creates a float32 CPU video batch and casts it to BF16 on the
GPU. The optimized collator may cast the normalized float32 batch to BF16 in the
worker, halving pinned-memory and H2D bytes, only after an equivalence gate proves
that CPU and H100 casts produce identical BF16 bit patterns for every value in
the comparison corpus.

If any value differs, the collator must return float32 video and the trainer must
retain the current GPU cast. Worker-side planner preprocessing remains useful
independently of this optional optimization.

### Video-only dataset mode

Add a dataset argument with a backward-compatible default:

```yaml
load_action_state: true
```

The formal joint video-only recipe sets it to `false`. In that mode the dataset
does not read episode parquet files and does not compute or return `actions` or
`state`. The trainer and validation path may use this mode only when
`return_action=false`; preflight and runtime checks reject incompatible
configurations.

## Strict Equivalence Contract

The optimized path is accepted only if all of the following hold on 100 real,
deterministically selected batches spanning all four LIBERO domains:

1. Sample indices, frame indices, captions, main/wrist ordering, and normalized
   video shapes are identical.
2. Final BF16 video tensors are bitwise identical. If the early CPU cast fails
   this condition, it is disabled without blocking the rest of the design.
3. `input_ids`, `attention_mask`, and `image_grid_thw` are exactly equal.
4. `pixel_values`, after applying the current model-dtype movement rule, are
   bitwise identical.
5. No processor tensor key is missing, added unexpectedly, reordered, or given
   a different dtype or shape.
6. With identical model state and restored CPU/CUDA RNG state, planner outputs,
   teacher targets, per-loss tensors, and trainable gradients match the current
   path. Exact equality is required where PyTorch is deterministic; otherwise
   the test uses zero relative tolerance and the smallest justified absolute
   tolerance for the same CUDA kernel.
7. An uninterrupted run and a save/resume run consume the same sample stream and
   produce the same state at the comparison step.

## Error Handling and Fallback

- Validate `[B,2,3,H,W]`, current-frame index, finite/range constraints, camera
  order, processor batch size, and required output keys in the collator.
- A worker processor-load failure reports the checkpoint path and worker id.
- When `joint_training.preprocess_planner_in_worker=true`, a missing
  `planner_inputs` batch key is an error rather than a silent performance
  fallback.
- The config flag `joint_training.preprocess_planner_in_worker=false` selects
  the existing rank-side `prepare_inputs` implementation without code changes.
- The original dataset defaults remain unchanged for non-joint and action
  training.

## Verification

### Unit and contract tests

- Collator main/wrist and current-frame ordering.
- Lazy processor initialization and reuse per worker.
- Nested processor-key forwarding and device/dtype conversion.
- Strict old/new planner-input equality on fixture images.
- Conditional action/state omission and rejection when actions are required.
- Optional BF16 gate success and automatic fallback.
- Deterministic sampler and resume tests already protecting the joint loader.

### Real-data equivalence test

Run the 100-batch contract on the exact OLA checkpoint, processor, predecoded
cache, and H100 software stack. Save only a compact JSON report of shapes,
dtypes, mismatch counts, and maximum absolute differences; do not save duplicate
image or model artifacts.

### Throughput A/B

Use the same 8 GPUs, model checkpoint, seed, batch 4, accumulation 4, BF16,
ZeRO-2, disabled gradient checkpointing, and predecoded cache.

- Run the current and optimized paths for 30 optimizer updates each.
- Exclude initialization and the first five updates.
- Compare median and p90 update time, cumulative samples/s, GPU SM utilization,
  board power, peak allocated memory, CPU utilization, and dataloader wait time.
- Keep the worker preprocessing only if strict equivalence passes and median
  steady-state samples/s improves by at least 2% without a p90 regression over
  2%.
- Keep the early BF16 video cast as a separately reported ablation; do not bundle
  its result with the worker-processor result.

## Rollout

Development and CPU-only contract work may proceed without touching the active
formal run. An 8-GPU A/B must run on a spare node or after a durable formal
checkpoint exists. The active run is never stopped merely to test this design.

Roll out in this order:

1. timing and equivalence harness;
2. video-only action/state bypass;
3. worker-side Qwen processor with float32 video transport;
4. optional early BF16 video cast after its independent bitwise gate;
5. 8-GPU A/B and retain/revert decision.

## Expected Outcome

The likely gain is 2--4% steady-state throughput, primarily from overlapping the
roughly 24 ms/microbatch rank-side PIL/Qwen work and removing its CUDA
synchronization boundary. Action/state bypass reduces unnecessary CPU and NAS
work but is not expected to change GPU throughput materially because the current
worker pool already has substantial headroom. Larger gains require a separate
GPU-side design such as a higher microbatch, kernel profiling/fusion, or changing
online teacher execution.
