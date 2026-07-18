# OLA Dual-Camera K4 VLM Planner Design

## Goal

Train a new Qwen3-VL-2B semantic planner on OLA that receives the current main
and wrist camera images as two independent visual inputs and predicts four
future SigLIP2 feature grids for each camera. The export must be directly
consumable by the two-view, four-keyframe GE-Act semantic-conditioning path.

This is a fresh planner run from the Qwen3-VL-2B-Instruct base model. No tensor
from the existing composite-input `step_020000` planner is used to initialize
the model or prediction heads.

## Fixed Contract

- Camera order: `main=0`, `wrist=1`.
- Input layout: two independent image slots in one Qwen conversation. Camera
  images are never concatenated, cropped into a composite, or copied between
  views.
- Prediction window: nine sampled frames with future offsets `[2, 4, 6, 8]`.
- SigLIP2 teacher: `siglip2-large-patch16-256`.
- Per-camera, per-keyframe SigLIP2 target: `16 x 16 x 1024`, represented as
  `256 x 1024` tokens.
- Public future SigLIP2 output: `[B, 2, 4, 256, 1024]`.
- DA3 four-layer WSA supervision remains an auxiliary training objective. GE-Act
  consumes only the predicted SigLIP2 branch.

## Architecture

One Qwen conversation contains an explicitly labelled main-camera image slot,
an explicitly labelled wrist-camera image slot, and the language instruction.
The model performs one forward pass so planner tokens can use language and
cross-view context. The implementation nevertheless extracts exactly two
ordered image-token spans and retains them as `[B, 2, N_image, H]`.

The K4 planner keeps the current shared/private query layout. Each future
keyframe owns 32 shared task tokens, 32 SigLIP-private task tokens, and 32
depth-private task tokens. This gives 96 unique tokens per keyframe and 384
planner tokens in total. A SigLIP or depth head receives 64 tokens per
keyframe: its 32 shared tokens plus its 32 private tokens.

The prediction heads and query banks are shared across cameras. The same head
is applied once with the main image-token span and once with the wrist
image-token span. Sharing parameters does not merge the view contexts and does
not double the 384-token planner prefix.

## Data and Teacher Flow

The LIBERO loader returns video in camera-preserving form, with a fixed main
then wrist dimension. For every sample it provides two current frames and four
future frames per camera at offsets `[2, 4, 6, 8]`.

Teacher encoding may flatten batch, camera, and keyframe dimensions only while
calling the frozen teachers. Outputs are restored before loss computation:

```text
current inputs:  [B, 2, H, W, 3]
future inputs:   [B, 2, 4, H, W, 3]
future SigLIP2:  [B, 2, 4, 256, 1024]
future DA3:      [B, 2, 4, layers, tokens, width]
```

Losses are computed against the matching camera and keyframe and then averaged
over batch and both views. Swapping, duplicating, or broadcasting one camera's
teacher target to the other camera is an error.

## Initialization and Optimization

- Base model:
  `/data/users/junjie/vlm4wam_2b/weights/Qwen3-VL-2B-Instruct`.
- Do not load the existing `qwen3vl2b_siglip2_da3_libero_future_k4_wsa`
  checkpoint.
- Freeze the Qwen vision tower.
- Train the Qwen language backbone, planner-token embeddings, SigLIP2 plan
  head, and DA3 depth head.
- Keep the existing online SigLIP2 and four-layer DA3 WSA teacher definitions
  so the only intended experiment changes are camera separation and the fresh
  initialization.
- Use bf16, ZeRO-2, TF32 where supported, and no gradient checkpointing unless
  the validated batch does not fit.
- Eight OLA H100 GPUs, batch size 8 per GPU, gradient accumulation 2, global
  batch size 128.
- Train for 30,000 optimizer steps.
- Backbone learning rate `3e-5`, head learning rate `3e-4`, weight decay
  `0.01`, and 2,500 warmup steps followed by the existing decay schedule.
- Save only steps 20,000, 25,000, and 30,000.

## Export Contract

Every checkpoint must record and validate at least:

```text
planner_input_layout = "separate_camera_images"
camera_names = ["main", "wrist"]
num_camera_views = 2
camera_head_sharing = "shared_head_per_view_image_context"
semantic_output_layout = "batch_view_keyframe_token_feature"
semantic_teacher = "siglip2-large-patch16-256"
future_keyframe_offsets = [2, 4, 6, 8]
num_keyframes = 4
grid_size = 16
semantic_dim = 1024
target_tokens_per_keyframe = 256
planner_token_count = 384
```

The loader must reject the legacy composite-input K4 checkpoint and any export
with missing, reordered, or incompatible camera/keyframe metadata. Metadata is
never edited to make an incompatible checkpoint appear valid.

## GE-Act Boundary

The future GE-Act provider will freeze this planner and consume
`[B, 2, 4, 256, 1024]` using view-aligned cross-attention at temporal offsets
`[2, 4, 6, 8]`. Preparing or launching that GE-Act run is outside this VLM
training change; the VLM export and one provider preflight must pass first.

## Validation

Tests must cover:

1. two ordered image slots and no horizontal concatenation;
2. exact extraction of two non-empty image-token spans;
3. four future frames per camera at offsets `[2, 4, 6, 8]`;
4. 384 total planner tokens with the K4 shared/private layout;
5. predicted SigLIP2 shape `[B, 2, 4, 256, 1024]`;
6. independent teacher targets and losses for main and wrist cameras;
7. changing one camera changes its conditioned prediction without replacing
   the other camera stream;
8. strict export rejection of composite-input or K1 checkpoints;
9. a one-step distributed smoke run using the production OLA launcher;
10. a frozen-planner GE-Act provider preflight on the produced checkpoint.

The training launch proceeds only after unit/contract tests, data preflight,
checkpoint export preflight, and the one-step smoke run pass.

## Acceptance Criteria

- The production OLA command uses two independent camera images and K4 future
  targets.
- No existing VLM checkpoint initializes the new run.
- The vision tower is frozen and the intended Qwen/planner parameters are
  trainable.
- Global batch size is 128 across eight GPUs.
- Checkpoints are written only at 20k, 25k, and 30k.
- A checkpoint predicts finite `[B, 2, 4, 256, 1024]` SigLIP2 features and
  passes the strict GE-Act provider preflight.
