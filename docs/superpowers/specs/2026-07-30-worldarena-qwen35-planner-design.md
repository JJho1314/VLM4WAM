# WorldArena Qwen3.5 Future-Semantic Planner Design

Date: 2026-07-30

Status: approved design

## Objective

Adapt the existing strict Baton-style Qwen3.5 planner to WorldArena-style
RoboTwin2.0 videos before integrating the planner with the Boundless Wan video
model.

Given one current head-camera frame and one task instruction, Qwen3.5 predicts
four future spatial SigLIP2 feature grids. The first milestone validates the
data contract and optimization path on the 509 released training videos. The
same loader contract will then scale to filtered RoboTwin2.0 trajectories.

This phase trains only the semantic planner. It does not train the Boundless
Wan model or inject actions into Qwen3.5.

## Dataset Boundaries

### Training subset

The initial training subset is:

`DavidxWang/worldarena2026-robotwin-data`

It provides 509 generated RoboTwin2.0 episodes covering all 50 WorldArena
Track-1 tasks. Every training episode contains:

- one 640x480 head-camera `video.mp4`;
- 121 temporally aligned RGB frames;
- `first_frame.jpg`;
- a WorldArena-style instruction;
- 14-dimensional joint actions;
- 16-dimensional dual-end-effector actions;
- camera intrinsics and extrinsics; and
- episode metadata.

Only the RGB video and instruction are required by the first planner stage.
Action and camera-calibration files are validated and retained in the sample
schema for later Boundless Wan integration, but they are not planner inputs or
targets.

### Held-out benchmark data

The official `WorldArena/WorldArena_Robotwin2.0` validation and test archives
are evaluation-only. They must never appear in the training manifest, target
generation, normalization statistics, or random sampling pool.

The adapter rejects records whose resolved path is under an official
validation/test root. This prevents accidental benchmark leakage even if
multiple archives share one parent directory.

### Scale-up subset

After the 509-episode milestone passes, the same schema will be produced from
the public RoboTwin2.0 trajectory corpus:

- retain only the 50 WorldArena task families;
- retain successful trajectories with valid head-camera RGB and instruction;
- generate deterministic train/validation splits by episode identity;
- preserve the official WorldArena validation/test data as a separate,
  immutable evaluation domain.

The 509-episode run is a pipeline and representation validation run, not the
final data scale for a converged planner.

## Model Contract

The implementation reuses the strict Baton Qwen3.5 planner without modifying
the existing LIBERO path:

- dense Qwen3.5-2B vision-language backbone;
- frozen `SigLIP2-large-patch16-256` teacher;
- one learnable query per target visual token;
- one 16x16 continuous feature grid per future keyframe;
- four future keyframes;
- direct pointwise feature MSE;
- no codebook, DINO, depth, counterfactual rows, query self-attention, or
  auxiliary losses.

WorldArena uses one head-camera row per sample. The planner weights and token
geometry are unchanged, but the camera axis has size one:

```text
input:
    current RGB [B, 1, 3, H, W]
    instruction [B]

target:
    future SigLIP2 [B, 1, 4, 256, 1024]
```

The Qwen processor retains its own configured visual preprocessing. The
SigLIP2 teacher independently resizes target frames to 256x256.

## Temporal Sampling

Each source episode exposes 121 aligned frames. A training sample chooses a
current frame `c` and four strictly future target frames:

```text
f_k = c + round((k + 1) * (120 - c) / 4),  k in [0, 1, 2, 3]
```

The current index is sampled only from `[0, 116]`, guaranteeing four future
positions. Duplicate indices caused by rounding are rejected. The sample
stores the five exact source indices for reproducibility.

For validation, current-frame indices are deterministically derived from the
episode identity and seed. Training may resample the current index across
epochs.

This normalized remaining-horizon sampling supports trajectories with
different lengths when the loader is later extended to full RoboTwin2.0.

## Instruction Handling

The original WorldArena instruction is preserved. The loader removes no object
descriptions and does not generate synthetic negative instructions.

If the standard WorldArena rigid-workspace prefix is present, it remains in the
text by default because it matches downstream generation prompts. The adapter
also stores the task-specific suffix separately for analysis and future
ablation, without changing the first training run.

Samples with empty instructions or unresolved video paths fail validation
before distributed training begins.

## Data Adapter

A new WorldArena dataset module is added beside, not inside, the LIBERO HDF5
loader. Its responsibilities are:

1. read a localized JSONL/manifest;
2. resolve relative paths under a configured dataset root;
3. reject official validation/test paths;
4. inspect video metadata without decoding the full episode;
5. decode only the selected current and four future frames;
6. return the existing Baton batch contract with a single camera; and
7. expose action/calibration paths as metadata for the later Wan stage.

The first implementation supports MP4 input because the 509-episode release is
small. It also includes a deterministic predecode command that writes a
versioned HDF5 cache with source hashes. Production training uses this cache;
online MP4 decoding remains available for correctness comparison and smoke
tests.

The predecoded cache stores RGB frames, instruction text, task name, episode
identity, source frame count, and optional action/calibration metadata. It does
not store online SigLIP2 targets.

## Configuration and Entrypoints

New files are scoped under WorldArena-specific names:

- `qwen35_baton/worldarena_data.py`;
- `qwen35_baton/cli/predecode_worldarena.py`;
- `qwen35_baton/configs/worldarena_stage1.json`;
- `qwen35_baton/scripts/train_worldarena_semantic_planner.sh`.

The existing `train_semantic_planner.py` remains the common trainer and accepts
either the existing LIBERO artifact provider or the new WorldArena provider
through an explicit dataset type. Existing Qwen3-VL-2B, LIBERO, GE-Act, and
FastWAM paths remain unchanged.

## Training Milestones

The first run is deliberately staged:

1. one-episode decode and collation smoke test;
2. one-GPU finite forward/backward test;
3. eight-GPU 20-step checkpoint probe;
4. a bounded 509-episode training run with held-out generated episodes; and
5. only after metric and visualization review, the full filtered RoboTwin2.0
   run.

The 509-episode run uses the existing strict Baton optimizer contract:

- BF16;
- AdamW;
- learning rate `1e-5`;
- frozen SigLIP2 teacher;
- full Qwen3.5 and alignment tower training;
- effective global batch 128 when hardware permits; and
- early step-20 checkpoint plus periodic checkpoints.

Per-device batch and accumulation are selected by an OOM-safe probe on the
target GPU type while preserving the effective global batch.

## Validation

Tests must prove:

- the 509-record manifest localizes all paths;
- official validation/test paths are rejected;
- one record decodes exactly one current and four strictly future frames;
- sampled source indices are ordered, unique, deterministic in validation, and
  in range;
- one WorldArena sample produces one Qwen row, not LIBERO's two camera rows;
- the target shape is `[B, 1, 4, 256, 1024]`;
- instructions survive collation unchanged;
- action/calibration paths remain metadata and never enter the Qwen input;
- the HDF5 cache matches online decoding for selected frames;
- SigLIP2 remains frozen and feature MSE remains the only Stage-1 loss;
- the legacy LIBERO Baton tests remain green; and
- a tiny optimizer step is finite.

## Boundless Wan Handoff

The subsequent video-model phase will vendor the Apache-2.0 Boundless World
Model and use its Wan2.2-TI2V-5B action-conditioned architecture. That phase
will consume:

- the same current head-camera frame;
- the WorldArena instruction;
- normalized dual-arm action sequences; and
- Qwen3.5-predicted future semantic grids as an additional condition.

No Boundless Wan implementation is part of this first Qwen3.5 data-adapter
milestone.
