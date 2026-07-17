# GE-Act Dual-Camera VLM Planner Design

## Goal

Train a Qwen3-VL planner that receives the current main-camera image and the
current wrist-camera image as two independent visual inputs, predicts one
future SigLIP2 feature grid for each camera, and supplies those two grids to
GE-Act as view-aligned semantic cross-attention context.

The planner keeps the existing current/future SigLIP2 and depth auxiliary
objectives. The GE-Act stage consumes only the future SigLIP2 predictions.

## Scope

- Camera order is fixed as `main=0`, `wrist=1`.
- Each sample contains two independent current RGB images and two independent
  future RGB targets. No horizontally concatenated image is accepted.
- The temporal target remains the end of the current nine-frame prediction
  window: offset `8`, represented to GE-Act as one keyframe at normalized time
  `1.0`.
- SigLIP2 uses a `16 x 16` spatial grid with feature width `1024`, so every
  camera produces `256` semantic tokens.
- The supplied `step_030000` planner is an initialization checkpoint, not a
  valid final dual-camera planner. It must be fine-tuned on the independent
  two-camera contract before GE-Act uses it.

This design does not depend on FastWAM data structures or composite images.

## Architecture

### VLM input

One Qwen3-VL conversation contains two image slots followed by the instruction.
The prompt identifies the slots explicitly as `main camera` and `wrist camera`.
Both images participate in one VLM forward pass, allowing the planner tokens to
use cross-view and language context while preserving separate visual-token
spans for the two cameras.

The VLM returns:

- a shared set of planner-token hidden states;
- main-camera image-token hidden states;
- wrist-camera image-token hidden states.

The implementation must find the two contiguous image-token spans explicitly
and validate that every batch item contains exactly two non-empty spans. It
must not flatten them into one undifferentiated image context.

### Query and head sharing

The existing four independent query groups are retained:

1. current SigLIP2: 64 tokens;
2. current depth: 64 tokens;
3. future SigLIP2: 64 tokens;
4. future depth: 64 tokens.

There are still 256 planner tokens in total. The four prediction heads and
their query banks are shared between cameras. For each head, the same
task-hidden group is decoded twice: once against the main image-token span and
once against the wrist image-token span. Different per-view image context and
the explicit camera order make the two predictions distinct without doubling
the planner prefix to 512 tokens.

Current-alignment heads keep gradients through the corresponding image-token
span. Future-alignment heads retain the existing detached image-context
behavior. This preserves the current checkpoint's LingBot-style optimization
contract.

### Planner outputs

The public dual-camera prediction API returns:

```text
current_siglip: [B, 2, 256, 1024]
future_siglip:  [B, 2, 256, 1024]
current_depth:  [B, 2, 256, 2048]
future_depth:   [B, 2, 256, 2048]
```

The depth width follows the existing `step_030000` DA3-Large last-layer head.
It is auxiliary planner supervision only; GE-Act consumes the 1024-D
`future_siglip` branch.

The depth width is restored from the current DA3 checkpoint contract and is
fixed to `1024` for this run. The historical internal `dino` name may remain
temporarily for checkpoint compatibility, but the exported API and metadata
must identify this channel as SigLIP2. View index `0` is always main and view
index `1` is always wrist.

## Training Data Flow

For each sample:

1. Read the current main and wrist RGB frames independently.
2. Read the future endpoint main and wrist RGB frames independently.
3. Send the two current frames and instruction through one VLM forward pass.
4. Flatten the camera dimension only for teacher computation:
   `[B, 2, H, W, 3] -> [B * 2, H, W, 3]`.
5. Encode current and future frames independently with SigLIP2 and DA3, then
   reshape the targets back to `[B, 2, 256, D]`.
6. Decode each of the four planner branches separately for each camera.
7. Average every branch loss over batch, camera, token, and feature dimensions,
   then apply the existing branch loss weights unchanged.

No camera may use the other camera's teacher target. Tests must detect a swapped
or duplicated target stream.

## Checkpoint Initialization and Export

The new training run initializes from:

```text
/data/users/junjie/code/VLM4WAM_k1_zero2_bidir/outputs/
qwen3vl2b_siglip2_da3_libero_cur_k1/step_030000
```

Initialization restores the full Qwen3-VL model, all four alignment heads, the
pooled query-bank embeddings, and the processor. No camera-specific head is
created, so all restored tensors retain their original shapes.

Dual-camera exports add and validate these metadata fields:

```text
planner_input_layout = "separate_camera_images"
camera_names = ["main", "wrist"]
num_camera_views = 2
camera_head_sharing = "shared_head_per_view_image_context"
semantic_output_layout = "batch_view_token_feature"
semantic_teacher = "siglip2-large-patch16-256"
future_keyframe_offsets = [8]
```

Loading a composite-input checkpoint directly through the dual-camera inference
API must fail with an actionable contract error. The old checkpoint is allowed
only through the explicit initialization path used for fine-tuning.

## GE-Act Integration

During GE-Act training the dual-camera planner is frozen, kept in evaluation
mode, and executed without gradients. It receives the current main/wrist
observations and instruction and exposes only:

```text
future_siglip[:, :, None, :, :]  # [B, 2, 1, 256, 1024]
```

The semantic adapter adds or applies:

- a projection from SigLIP2 width `1024` to the LTX semantic context width;
- two-dimensional `16 x 16` patch position information;
- one future-time position at normalized time `1.0`;
- a learned main/wrist view embedding.

GE-Act routes main-view video tokens only to main semantic context and wrist-view
video tokens only to wrist semantic context. Text conditioning remains a
separate cross-attention source. There is no interpolation, spatial crop, or
heuristic split of predicted semantic features.

The initial integration freezes the planner and trains the LTX model plus its
semantic adapter. Joint VLM/LTX optimization is outside this scope because it
would substantially increase memory use and make planner regressions harder to
isolate.

## Validation and Failure Handling

The implementation rejects a batch or checkpoint when:

- the input has anything other than two camera views;
- camera order or camera names are missing or inconsistent;
- a conversation does not contain exactly two image-token spans;
- either image-token span is empty;
- a SigLIP2 output is not `256 x 1024` per view;
- GE-Act requests more than the single exported future keyframe;
- metadata describes the legacy composite-input planner.

There is no fallback that concatenates cameras or copies one camera's feature
stream into the other.

## Test Strategy

Tests are added before implementation and cover:

1. the collator emits two ordered image slots and preserves main/wrist images;
2. image hidden states are split into exactly two per-view spans;
3. the planner still uses 256 total query tokens;
4. all four predictions have shape `[B, 2, 256, D]`;
5. changing only the wrist image changes the wrist branch and does not replace
   the main branch;
6. losses average over both views and catch swapped teacher targets;
7. the old `step_030000` tensors load through initialization without shape
   expansion;
8. exported metadata rejects legacy composite inference;
9. the GE-Act provider returns `[B, 2, 1, 256, 1024]` in main/wrist order;
10. semantic routing prevents cross-view conditioning leakage;
11. a small end-to-end smoke run completes one planner training step and one
    GE-Act forward/backward step with the planner frozen.

## Acceptance Criteria

- The planner consumes two independent images and never creates a composite.
- It predicts distinct, ordered main and wrist future SigLIP2 grids.
- Query-token count remains 256 and all four auxiliary objectives remain active.
- The `step_030000` checkpoint initializes all compatible tensors exactly.
- GE-Act consumes two view-aligned semantic streams with spatial, temporal, and
  view identity information.
- Unit, contract, checkpoint-load, and smoke tests pass before a training launch.
