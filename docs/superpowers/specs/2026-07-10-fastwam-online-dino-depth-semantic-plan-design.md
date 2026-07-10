# FastWAM Online DINO+Depth Semantic Plan Design

**Date:** 2026-07-10

**Status:** Approved for implementation planning

## Goal

Condition the FastWAM Cosmos video DiT with online semantic plans produced by the
repository's 4B LingBot-aligned planner. The plan combines spatially aligned
DINO-video and LingBot-Depth features while preserving the existing Cosmos
semantic cross-attention mechanism.

FastWAM currently trains on nine RGB frames: frame 0 is the current observation
and frames 1 through 8 are future targets. The planner must therefore be trained
for the same nine-frame horizon before its checkpoint is frozen and used by
FastWAM.

## Non-Goals

- Do not backpropagate FastWAM losses into the 4B planner.
- Do not precompute or cache planner outputs for FastWAM training or inference.
- Do not concatenate DINO and depth along the token axis.
- Do not change the FastWAM video horizon from nine RGB frames.
- Do not change the action expert or GR00T action head to attend directly to
  planner tokens.
- Do not reuse the existing 49-frame planner checkpoint as if its temporal slots
  represented the nine-frame FastWAM horizon.

## Confirmed Baseline

### FastWAM temporal layout

The current LIBERO Cosmos task samples 33 LeRobot steps with
`action_video_freq_ratio=4`, producing RGB indices `[0, 4, ..., 32]` and nine
video frames. The Wan VAE maps these nine RGB frames to three latent frames. In
the default MoT coupling, latent frame 0 is a clean current-observation condition
and video loss is computed on latent frames 1 and 2, which decode to eight future
RGB frames.

### 4B planner outputs

The LingBot-aligned 4B planner uses a shared VLM forward followed by two
TaskTokenResampler heads:

- DINO-video head: 256 row-major `16 x 16` tokens per keyframe, 1024 features per
  token.
- Depth head: 256 row-major `16 x 16` tokens per keyframe, 1024 features per
  token.

The existing code returns only the DINO head from inference. The depth head is
optional, auxiliary, disabled by default, and is not saved in current planner
checkpoints. This design promotes the depth head to a required inference output.

## Decisions

1. Use five semantic keyframes over the eight future RGB frames.
2. Retrain or fine-tune the 4B planner on the same nine-frame horizon and camera
   composition used by FastWAM.
3. Run the planner online during both FastWAM training and inference.
4. Keep the planner frozen under `eval()` and `torch.no_grad()` inside FastWAM.
5. Keep DINO and depth tensors separate until a trainable FastWAM fusion module.
6. Fuse modalities at matching spatial positions, not by doubling the token
   sequence.
7. Preserve the existing `SemanticPlanContextAdapter` and per-block semantic
   cross-attention after fusion.
8. Pass the actual sampled RGB video FPS into video RoPE and semantic RoPE.

## Tensor Contract

For batch size `B`, keyframe count `K=5`, grid side `G=16`, and modality
dimension `D=1024`:

```text
dino_plan:          [B, K * G * G, D] = [B, 1280, 1024]
depth_plan:         [B, K * G * G, D] = [B, 1280, 1024]
semantic_plan_times:[B, K]            = [B, 5]
fused_plan:         [B, K * G * G, D] = [B, 1280, 1024]
vlm_query_hidden:   [B, K, 96, H]
dino_query_hidden:  [B, K, 64, H]
depth_query_hidden: [B, K, 64, H]
```

The contract is keyframe-major, then row-major spatial order within each
keyframe. DINO and depth token `[..., k, y, x, :]` must describe the same future
time and spatial cell.

## Planner Stage

### Training horizon

The 4B planner is trained or fine-tuned with:

```text
sequence_length = 9
num_keyframes = 5
grid_size = 16
semantic_dim = 1024
shared_latent_per_keyframe = 32
private_latent_per_keyframe = 32
branch_latent_per_keyframe = 64
total_unique_latent_per_keyframe = 96
keyframe_scheme = uniform
```

The shared keyframe-offset helper produces future indices `[1, 3, 4, 6, 8]` for
a nine-frame window. Both online teachers consume exactly those future frames.
The planner input is the same current multi-camera composite seen by FastWAM,
plus the task instruction.

### Shared/private VLM query layout

For every future keyframe, the VLM emits three distinct query groups in this
fixed order:

```text
[32 shared queries, 32 DINO-private queries, 32 depth-private queries]
```

The DINO head receives the 32 shared queries concatenated with the 32
DINO-private queries. The depth head receives the same 32 shared queries
concatenated with the 32 depth-private queries. Each branch therefore consumes
64 query hidden states per keyframe, while the VLM sequence contains 96 unique
query tokens per keyframe and 480 across all five keyframes.

The shared queries are the only query hidden states consumed by both heads.
Private queries never cross into the other modality head. Query ordering is
keyframe-major, then group-major in the exact shared/DINO/depth order above.
This query layout does not change either dense output: each head still decodes
256 spatial feature tokens per keyframe and returns `[B, 1280, 1024]`.

### Online targets

No semantic feature files are read during planner training:

- `DinoVideoTargetEncoder` produces `[B, 1280, 1024]` DINO-video targets.
- `DepthTargetEncoder` produces `[B, 1280, 1024]` LingBot-Depth targets.

The DINO and depth heads share one VLM forward and the 32-query shared subset
for each keyframe, while retaining separate 32-query private subsets. Their
losses remain independently measurable. The depth loss weight is configurable
and retains the LingBot-compatible default unless experiments justify changing
it.

### Checkpoint contract

Every checkpoint intended for FastWAM must contain:

```text
qwen3vl_lora_or_model/
processor/
plan_head.pt
depth_head.pt
plan_token_embedding.pt
planner_meta.json
```

`planner_meta.json` records at least the horizon, keyframe scheme and offsets,
grid, modality dimensions, shared/private/branch/total query counts, query
layout and ordering, presence of the depth head, feature types, and source
checkpoint identifiers. The provider rejects legacy 8-query, DINO-only, or
49-frame checkpoints.

### Frozen online provider

A reusable provider loads the saved processor, 4B VLM, DINO head, and depth
head. Its public operation accepts current RGB tensors and instructions and
returns both modality tensors plus keyframe times. It performs one VLM forward,
then evaluates both heads from the shared hidden states.

The provider:

- runs in `eval()` mode under `torch.no_grad()`;
- returns detached tensors;
- is held outside FastWAM's registered `nn.Module` tree so its weights are absent
  from FastWAM optimizer, FSDP state, EMA, and checkpoint payloads;
- loads once per distributed rank on that rank's CUDA device;
- is configured by a planner checkpoint path rather than copied into a FastWAM
  checkpoint.

## FastWAM Stage

### Online invocation

For each FastWAM batch, before the coupled video/action forward:

1. Read `sample["video"][:, :, 0]`, the current RGB composite before VAE
   encoding.
2. Read the corresponding task instruction.
3. Convert the current frame from FastWAM's `[-1, 1]` tensor convention to the
   planner processor's RGB convention.
4. Run the frozen provider once for the batch.
5. Validate and fuse the DINO and depth tensors.
6. Send the fused plan and keyframe times through the existing Cosmos semantic
   conditioning path.

Training and inference use this same provider and tensor contract. FastWAM does
not load semantic-plan manifests or feature files in this mode.

### Same-position dual-branch fusion

The registered, trainable `DinoDepthPlanFusion` module computes:

```text
d = ProjDINO(LN_DINO(dino_plan))
z = ProjDepth(LN_DEPTH(depth_plan))
g = sigmoid(depth_gate_logit)
fused_plan = LN_out(d + g * z)
```

Both projections map 1024 features to 1024 features. The depth gate is a learned
scalar initialized so `g` is approximately 0.1. This preserves a strong DINO
baseline at initialization while allowing gradients to reach the depth branch
from the first step. The fusion output remains `[B, 1280, 1024]`.

The fusion module belongs to the Cosmos video expert. It is optimized with
FastWAM and saved in the FastWAM checkpoint.

### Cosmos semantic adapter

The existing adapter is configured as:

```yaml
semantic_plan_context: true
semantic_plan_in_dim: 1024
semantic_plan_hidden_dim: 2048
semantic_plan_num_keyframes: 5
semantic_plan_source_num_keyframes: 5
semantic_plan_spatial_grid: 16
semantic_plan_max_tokens: 1280
semantic_plan_coord_hidden_dim: 256
semantic_plan_use_rope: true
semantic_plan_cross_attention_blocks: null
```

Because source and target keyframe counts are both five, the adapter performs no
temporal subsampling. It adds coordinate/type information, projects into the
Cosmos context width, and supplies semantic context to all 28 video blocks by
default.

Semantic tokens condition only the video DiT. MoT, cross-attention, and AGRA
action paths receive planner information indirectly through their existing video
couplings.

## FPS and Keyframe Times

The effective sampled RGB video FPS is:

```text
video_fps = lerobot_fps / (global_sample_stride * action_video_freq_ratio)
```

This is the RGB-frame FPS passed to Cosmos; it is not divided again by the VAE
temporal compression factor. For the uniform nine-frame layout, the semantic
times are:

```text
semantic_plan_times = [1, 3, 4, 6, 8] / 8
```

The dataset emits `video_fps`, and FastWAM threads it through every video path:
MoT preparation, standalone cross-attention, AGRA video loss, AGRA foresight,
and inference. Video and semantic RoPE therefore share the actual sampled
timeline instead of silently falling back to Cosmos `base_fps=16`.

## Configuration

Online DINO+Depth conditioning is opt-in and requires:

- a compatible planner checkpoint path;
- online planner enabled;
- DINO+Depth fusion enabled;
- the K5/grid16 semantic adapter settings above;
- a known LeRobot FPS.

Legacy semantic-plan file loading remains available only when the online planner
mode is disabled. The two modes are mutually exclusive to avoid accidentally
mixing stale files with online predictions.

## Error Handling

The implementation fails fast when:

- `depth_head.pt` or required planner metadata is missing;
- the checkpoint horizon is not nine frames;
- keyframe count, offsets, grid, feature dimension, or modality ordering differs
  from the configured FastWAM contract;
- DINO and depth batch/token shapes differ;
- either modality contains any non-finite value;
- the current image or instruction is absent;
- dataset FPS is missing, non-positive, or inconsistent across component
  datasets;
- both online and file-backed semantic-plan modes are enabled.

No missing depth output is silently replaced with zeros. An explicit future
DINO-only ablation can be added as a separate configuration, not as fallback
behavior.

## Testing

### Planner tests

- Nine-frame/K5 uniform offsets equal `[1, 3, 4, 6, 8]`.
- Online DINO and depth targets both have shape `[B, 1280, 1024]` and aligned
  ordering.
- One VLM forward feeds both prediction heads.
- Each keyframe contains 32 shared, 32 DINO-private, and 32 depth-private
  queries; each head receives exactly 64 and the full VLM query sequence is
  exactly 480 tokens.
- Perturbing DINO-private queries cannot change the depth-head input, and
  perturbing depth-private queries cannot change the DINO-head input.
- Checkpoint save/load round-trips both heads and required metadata.
- Legacy 8-query, DINO-only, and 49-frame checkpoints are rejected by the
  FastWAM provider.
- Provider parameters are frozen, the model is in evaluation mode, and outputs
  are detached.

### Fusion tests

- Correct inputs produce `[B, 1280, 1024]`.
- DINO and depth use independent normalization/projection parameters.
- The initial depth gate is approximately 0.1 and receives gradients.
- Shape, dimension, and non-finite mismatches raise clear errors.

### FastWAM tests

- Effective FPS is computed from LeRobot metadata and sampling strides.
- Training invokes the provider from the current frame and instruction without
  reading semantic feature files.
- Inference uses the same provider and fusion path.
- Fused plans, keyframe times, and FPS reach MoT, standalone cross-attention,
  AGRA video-loss, and AGRA foresight paths.
- Frozen planner parameters are excluded from FastWAM optimizer and state dict;
  fusion parameters are included.
- Existing behavior is unchanged when online DINO+Depth conditioning is
  disabled.

### Runtime verification

After unit tests, run a real single-GPU, single-batch planner-to-FastWAM forward
and backward smoke test. Verify:

- the planner performs one forward and holds no gradients;
- fusion and FastWAM parameters receive finite gradients;
- output video/action shapes match the current task;
- peak memory is recorded before scheduling the full distributed run.

## Rollout Order

1. Extend and validate the 4B planner's depth output and checkpoint contract.
2. Train or fine-tune a nine-frame/K5 DINO+Depth planner checkpoint.
3. Implement and unit-test the frozen online provider.
4. Implement FastWAM fusion, FPS routing, and coupling plumbing.
5. Run the single-GPU smoke test.
6. Launch distributed FastWAM training only after the smoke test passes.
