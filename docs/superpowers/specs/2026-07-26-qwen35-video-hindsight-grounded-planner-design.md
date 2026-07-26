# Qwen3.5 Video-Hindsight Grounded Semantic Planner Design

Date: 2026-07-26

Status: Approved

## Objective

Build a Qwen3.5 semantic planner whose predicted future features satisfy both
of the following requirements:

1. They retain a spatially indexed visual representation suitable for GE-Act
   cross-attention.
2. They are explicitly conditioned on the LIBERO instruction, so the same
   image paired with different source objects, target objects, or actions
   produces different spatial attention and semantic features.

LIBERO does not provide semantic segmentation masks. The design therefore uses
the complete training trajectory, action/gripper signals, frozen vision
teachers, and counterfactual instructions to create soft video-hindsight
grounding targets. Complete future video and action information are
training-only supervision and are never required at inference.

The planner keeps Qwen3.5-VL as the reasoning backbone and adopts the released
384-pixel TA-Tok representation:

- teacher: `google/siglip2-so400m-patch14-384`;
- input resolution: `384 x 384`;
- spatial grid: `27 x 27`, or 729 codes per frame;
- codebook: 65,536 entries of width 1,536;
- future semantic keyframes: four per camera;
- main and wrist camera streams: separate examples with shared model weights.

## Relationship to the Previous Design

This specification supersedes
`2026-07-25-qwen35-planx-ta-tok-planner-design.md` and its implementation plan
for the new Qwen3.5 Plan-X path.

In particular, the new path:

- uses the released 384-pixel TA-Tok checkpoint instead of training the
  project-specific 256-pixel tokenizer;
- does not initialize Qwen's 2,048-dimensional visual-vocabulary rows by
  projecting the 1,536-dimensional TA-Tok codebook;
- adds an explicit video-hindsight grounding objective;
- exposes a separate text-aligned semantic subspace and relevance maps to
  GE-Act.

Existing Qwen3-VL-2B, K-means/query-token, and GE-Act baseline paths remain
unchanged. Partial implementation files from the superseded Qwen3.5 tokenizer
training plan are not part of this design and must not be mixed into the new
path.

## Non-Goals

- Producing hard or pixel-accurate semantic segmentation masks.
- Using future RGB, future actions, DINO, or gripper state during inference.
- Concatenating main and wrist images into one wide VLM image.
- Modifying the released TA-Tok codebook geometry.
- Treating generic arm motion as sufficient evidence of task relevance.
- Claiming exact reproduction of the original Plan-X language backbone. The
  visual tokenizer follows the released configuration, while the planner
  remains Qwen3.5-VL.
- Replacing the existing GE-Act text condition. The grounded semantic plan is
  an additional same-camera condition.

## System Boundaries

The system is split into four independently testable components:

1. **Hindsight target builder**: consumes complete training trajectories and
   writes versioned, low-resolution soft targets.
2. **Causal Qwen3.5 planner**: consumes only current RGB and instruction, then
   predicts future TA codes, grounded hidden states, and relevance maps.
3. **Semantic plan compressor**: converts each 27x27 predicted grid into a
   compact combination of scene coverage tokens and high-relevance tokens.
4. **GE-Act provider**: injects same-camera semantic plans through a
   zero-gated cross-attention branch.

No teacher-only component is imported or instantiated by planner or GE-Act
inference.

## Instruction Representation

Each LIBERO instruction is represented in two forms:

1. The original natural-language instruction is preserved unchanged.
2. The existing target-aware parser produces explicit semantic fields:
   `action`, `source`, and `target`.

For example:

```text
<ACT>pick up and place</ACT>
<SRC>the black bowl</SRC>
<TGT>between the plate and the ramekin</TGT>
```

The structured fields are prepended to the original instruction rather than
replacing it. Parsing failures retain the full instruction, mark the missing
field, and reduce the confidence of losses that require that field. They do
not discard an otherwise valid planner sample.

Every stored or predicted three-role axis uses the canonical order
`(source, target, action)`, even though the human-readable prompt prints
`ACT`, `SRC`, and `TGT`.

After the unchanged original instruction, the planner prompt appends three
learned structural query tokens in canonical order:
`<SRC_QUERY><TGT_QUERY><ACT_QUERY>`. Their causal states have consumed the
current image, every structured field, and the full original instruction.

During target-cache construction, the frozen SigLIP2 text tower encodes the
full instruction and the three phrases into normalized 1,152-dimensional
teacher embeddings. These embeddings define the common space used for direct
text/feature cosine similarity. They supervise a Qwen phrase-anchor head at
the three role-query positions; planner inference predicts the anchors from
Qwen prompt states and never loads SigLIP2. Qwen3.5
hidden states are not assumed to be isotropic or directly cosine-comparable
with text.

## Video-Hindsight Grounding Teacher

### Inputs

For every training trajectory and camera, the target builder consumes:

- the complete predecoded RGB trajectory;
- the original and structured instruction;
- per-step actions;
- gripper open/close state when available;
- the four selected future keyframe indices;
- the released frozen TA-Tok;
- frozen SigLIP2-SO400M/Patch14/384;
- the project's configured frozen DINO checkpoint.

The exact model identifiers, preprocessing values, checkpoint hashes, camera
order, and keyframe offsets are stored in target-cache metadata.

### Phrase-Relevance Seeds

SigLIP2 produces a global image/text similarity score but does not expose a
guaranteed dense patch/text projection. Dense phrase seeds are therefore
created with gradient-weighted relevance through the frozen SigLIP2 image
tower:

1. Encode a frame and one of `action`, `source`, or `target`.
2. Backpropagate the global image/text similarity only to the final spatial
   patch activations.
3. Combine normalized activation-gradient products with the vision pooling
   attention.
4. Clamp negative relevance to zero and normalize the result over the 27x27
   grid.

This is an offline teacher operation; no gradients update SigLIP2. It avoids
assuming that raw patch hidden states already inhabit the text projection
space.

### Temporal Evidence

DINO patch correspondence supplies cycle-consistent tracks between adjacent
frames and between the initial, interaction, and final phases. It is used for:

- propagating confident source/target phrase seeds;
- estimating correspondence-corrected feature change;
- rejecting isolated relevance peaks that jump between unrelated locations;
- enforcing smooth motion of a manipulated object.

Action and gripper signals define soft phase priors:

- before and around gripper closure: emphasize the source;
- during transport: emphasize a source-consistent moving track;
- around release: emphasize the target and destination;
- after release: emphasize persistent initial-to-final scene change.

If action/gripper phase detection is unreliable, the target builder falls back
to visual/text evidence and lowers the phase confidence rather than inventing
a sharp phase boundary.

### Counterfactual Evidence

For each valid instruction, construct hard negatives from the LIBERO
instruction vocabulary by replacing exactly one semantic field:

- source object or color;
- target object or spatial relation;
- action.

Negatives are drawn preferentially from the same suite and scene vocabulary.
They are used only as contrastive negatives; the system does not fabricate a
positive spatial target for an instruction that was not executed.

This comparison is required because temporal change alone would often select
the robot arm, and global image/text similarity alone may select every object
whose category appears in the scene.

### Soft Target Construction

For phrase `p`, time `t`, and camera `c`, the teacher combines:

- SigLIP2 phrase relevance;
- DINO track support;
- correspondence-corrected feature change;
- the action-phase prior.

All signals are normalized to `[0, 1]`. Their confidence-weighted geometric
mean is normalized spatially to produce:

```text
A_source[t, c, 27, 27]
A_target[t, c, 27, 27]
A_action[t, c, 27, 27]
```

Default evidence exponents are fixed and recorded in cache metadata:

| Map | Text | Track | Change | Phase |
|---|---:|---:|---:|---:|
| Source | 0.45 | 0.30 | 0.15 | 0.10 |
| Target | 0.45 | 0.15 | 0.25 | 0.15 |
| Action | 0.35 | 0.25 | 0.20 | 0.20 |

The target remains a probability distribution, not a binary mask. Every map
also receives a scalar confidence in `[0, 1]`. Missing phrases or conflicting
teacher signals set the corresponding confidence to zero, which disables that
grounding term while preserving code prediction supervision.

The text-aligned semantic target at a spatial position is:

```text
S_target[i] = normalize(
    A_source[i] * T_source
  + A_target[i] * T_target
  + A_action[i] * T_action
)
```

Positions with negligible total relevance are not forced toward an arbitrary
phrase embedding. Their scene information remains represented by the visual
code branch.

### Cache Layout

The cache stores only the four supervised future keyframes, not dense teacher
features for every video frame:

- TA code IDs: `uint16 [K, camera, 729]`;
- three relevance maps: quantized `uint8 [K, camera, 3, 27, 27]`;
- per-map confidence and quantization scales: FP16;
- DINO keyframe-to-keyframe displacement and confidence:
  FP16 `[K-1, camera, 27, 27, 3]`, where the final dimension is `(dx, dy,
  confidence)`;
- normalized phrase embeddings: FP16 `[3, 1152]`;
- current/future frame indices, trajectory ID, suite, camera ID;
- hashes for data split, instruction parser, TA-Tok, SigLIP2, DINO, and
  preprocessing configuration.

Versioned NumPy memmaps and a JSONL index follow the existing Qwen3.5 cache
pattern and support read-only multi-worker access. Cache construction is
trajectory-sharded and atomically publishes a completed shard. Training fails
closed on partial shards or metadata/hash mismatches.

## Qwen3.5 Causal Planner

### Camera Handling

Main and wrist images remain independent planner examples in an effective
`B*2` batch. They share Qwen3.5 and all prediction heads but do not share
causal state. A camera token identifies each stream. A single-camera input is
also valid and produces only that camera's plan.

### Vocabulary

Add 65,536 contiguous visual token IDs plus plan/frame/camera delimiters to an
experiment-local Qwen tokenizer and model. Never modify the base Qwen3.5
directory.

The released TA-Tok codebook has width 1,536, while Qwen3.5 token embeddings
have width 2,048. The code IDs themselves are dimensionless, so the planner
does not add a 1,536-to-2,048 codebook projection for vocabulary
initialization. New Qwen input/output rows use the official fallback policy:
the mean of existing Qwen token rows. They are then learned by code
cross-entropy. The TA codebook remains frozen and unchanged.

### Sequence

For each camera:

```text
<PLAN_START>
  <FRAME_1> 729 visual code tokens </FRAME_1>
  <FRAME_2> 729 visual code tokens </FRAME_2>
  <FRAME_3> 729 visual code tokens </FRAME_3>
  <FRAME_4> 729 visual code tokens </FRAME_4>
<PLAN_END>
```

The four project keyframes retain offsets `[0, 3, 5, 8]` after the first future
frame. Code tokens are raster ordered. Training masks all prompt and delimiter
positions from code loss. Inference uses constrained decoding: code positions
can emit only the visual vocabulary, while structural positions can emit only
their required delimiter.

### Hidden-State Alignment

Two causal states have different roles:

- `h_pre[i]` predicts visual code `c[i]` and is supervised by code
  cross-entropy and continuous visual-code regression.
- `h_post[i]` is the hidden state after generated/teacher-forced code `c[i]`
  has been consumed. It contains the code identity, instruction, current
  image, spatial history, and previous future frames. It drives semantic and
  grounding heads.

Teacher forcing obtains all states in one packed forward pass. Constrained
decoding records `h_post` as each generated token is fed through the KV cache;
the final code state is collected by the frame-end step. No second full
sequence forward is required.

### Prediction Heads

For each future position:

1. `visual_regression(h_pre) -> R^1536` predicts the normalized released
   TA-Tok embedding of the target code.
2. `semantic_projection(h_post) -> R^1152` predicts a normalized feature in
   the frozen SigLIP2 text space.
3. `phrase_projection(h_SRC_QUERY, h_TGT_QUERY, h_ACT_QUERY) -> R^(3x1152)`
   predicts normalized source, target, and action anchors from the three
   learned role-query states after the full instruction has been consumed.
4. `grounding_head(h_post, predicted_phrase_anchors) -> R^3` predicts source,
   target, and action relevance logits without an inference-time text teacher.
5. `fusion_gate(h_post, grounding_logits)` predicts a bounded semantic
   residual strength through a sigmoid in `[0, 1]`.

No head changes the released TA-Tok codebook.

## Planner Objectives

The planner loss is:

```text
L_planner =
    1.0 * L_code
  + 0.5 * L_dense_feature
  + 0.5 * L_grounding
  + 0.2 * L_counterfactual
  + 0.1 * L_temporal
```

Where:

- `L_code` is visual-token cross-entropy.
- `L_dense_feature` is the mean of:
  - cosine regression from `visual_regression(h_pre)` to the frozen TA
    codebook vector of the target code;
  - confidence-weighted cosine regression from
    `semantic_projection(h_post)` to `S_target`;
  - field-mask-weighted cosine regression from the three predicted phrase
    anchors to the cached SigLIP2 phrase embeddings.
- `L_grounding` is confidence-weighted Jensen-Shannon divergence between the
  predicted and hindsight source/target/action distributions.
- `L_counterfactual` is an InfoNCE/ranking objective requiring correct
  phrase-pooled semantic features to score above same-scene hard-negative
  phrases.
- `L_temporal` compares attention maps after warping them with cached DINO
  correspondences between future keyframes.

A weak effective-support regularizer prevents two failure modes:

- uniform attention over the whole scene;
- collapse to one patch for every instruction.

For a spatial distribution `A`, normalized effective support is
`exp(entropy(A)) / 729`. A hinge penalty is applied only outside `[0.01, 0.40]`
and enters `L_grounding` with coefficient `0.01`. This avoids imposing a fixed
object area. The support term is disabled for low-confidence teacher maps.

## Semantic Feature Supplied to GE-Act

For code `c[i]`, post-code hidden state `h_post[i]`, and predicted relevance
`g[i]`, construct:

```text
Z[i] = LayerNorm(
    W_visual(E_TA[c[i]])
  + g[i] * W_hidden(h_post[i])
  + E_time[i] + E_xy[i] + E_camera[i]
)
```

`W_visual: 1536 -> D_condition` and
`W_hidden: 2048 -> D_condition` are downstream GE-Act adapters. They are not
used to initialize Qwen vocabulary rows and therefore do not alter the
released visual-token geometry.

The provider exposes:

- `spatial_feature`: the fused spatial plan;
- `semantic_feature`: the normalized 1,152-dimensional text-aligned output;
- `source`, `target`, and `action` relevance maps;
- time, coordinate, and camera metadata.

## Spatial Compression for LTX

Passing all 729 tokens from every future frame through every LTX block is
unnecessarily expensive relative to the 8x8 LTX latent grid. Each 27x27 future
grid is compressed into:

1. **64 coverage tokens**: relevance-weighted area pooling into an 8x8 grid.
   Fractional area overlap handles the non-divisible 27-to-8 geometry.
2. **32 exact high-relevance tokens**: the highest-scoring original positions
   across source, target, and action maps.

The coverage path preserves the whole scene; the high-relevance path preserves
small task objects that pooling could erase. Duplicate top tokens are removed,
and unused slots are masked. Coordinates of both pooled and exact tokens are
retained for positional encoding.

Each future frame therefore contributes at most 96 tokens. Four frames
contribute at most 384 tokens per camera instead of 2,916.

## GE-Act Integration

The new provider is opt-in. Existing providers and default GE-Act behavior do
not change.

Main video latents attend only to main-camera semantic tokens, and wrist
latents attend only to wrist-camera tokens. The existing LTX view interaction
remains responsible for cross-camera information flow.

The semantic branch uses the existing all-block cross-attention placement and
3D `(time, y, x)` positional treatment. Predicted relevance also supplies a
bounded logit bias:

```text
attention_logit[j, i] =
    Q[j] K[i]^T
  + gate_bias * log(epsilon + relevance[i])
```

Both the semantic residual gate and bias gate are initialized to zero. At
initialization the model is numerically equivalent to the GE-Act baseline.
The bias gate is parameterized as `2.0 * tanh(raw_bias_gate)`, bounding its
magnitude while permitting exact zero initialization. Background tokens are
down-weighted, not deleted.

## Training Schedule

### Stage 0: Target Cache

Build and validate all hindsight targets before planner optimization. The
complete video, frozen teachers, actions, and gripper signals are used only in
this stage.

### Stage 1: Planner Training

Frozen:

- released TA-Tok;
- SigLIP2 image and text towers;
- DINO teacher.

Trainable:

- Qwen3.5 vision encoder;
- Qwen3.5 language backbone;
- experiment-local visual vocabulary rows;
- visual regression, semantic, grounding, and fusion heads.

Initial optimizer groups:

| Parameter group | Learning rate |
|---|---:|
| Qwen3.5 language | `1e-5` |
| Qwen3.5 vision | `5e-6` |
| New visual vocabulary and visual prediction head | `1e-4` |
| Semantic, phrase-anchor, grounding, and fusion-gate heads | `1e-4` |

Use:

- 30,000 optimizer steps;
- effective global batch 256;
- 1,000 warmup steps;
- cosine decay;
- gradient clipping at 1.0;
- checkpoints every 5,000 steps;
- four future keyframes;
- selective activation checkpointing where required by the 2,916-token target
  sequence.

The largest stable per-GPU microbatch is measured by preflight, and gradient
accumulation is derived from it without changing the effective global batch.

### Stage 2: Joint GE-Act Training

Load the validated planner and the unchanged GE-Act baseline. Keep TA-Tok,
SigLIP2, DINO, and all but the top eight Qwen language transformer layers
frozen.

Initial optimizer groups:

| Parameter group | Learning rate |
|---|---:|
| LTX video expert | `2e-5` |
| Action expert | `1e-4` |
| Semantic/grounding heads and adapters | `5e-5` |
| Top eight Qwen language layers | `1e-6` |
| Qwen vision encoder | `5e-7` |

Run 30,000 optimizer steps and save every 5,000 steps. Keep planner auxiliary
losses active at `0.25 * L_planner`. Multiply gradients originating from
GE-Act and entering Qwen by 0.1. This permits action-aware adaptation without
allowing diffusion noise to erase language grounding.

The original GE-Act video/action loss definitions and relative weights remain
unchanged.

## Evaluation

### Offline Planner Metrics

Report:

- visual-code cross-entropy, top-k accuracy, and perplexity;
- visual-code embedding cosine;
- text-aligned semantic cosine;
- correct-instruction versus same-scene hard-negative retrieval;
- source/target/action hindsight-map Jensen-Shannon divergence;
- counterfactual attention change;
- temporal correspondence consistency;
- attention entropy, effective support, and background concentration;
- main and wrist metrics separately.

Because no ground-truth segmentation exists, hindsight-map agreement is not
reported as segmentation accuracy.

### Counterfactual Single-Image Test

For one fixed image, run the original instruction and at least three
single-field substitutions. The output visualization must show:

- current RGB;
- source, target, and action heatmaps;
- correct and counterfactual similarity scores;
- the 8x8 coverage tokens and selected top-32 locations.

The test detects a planner that always attends to the robot or produces the
same feature for every instruction.

### Future-Plan Visualization

For each camera independently, visualize all four keyframes:

- future RGB target;
- decoded TA target and prediction;
- source/target/action hindsight targets;
- predicted relevance maps;
- semantic cosine and code metrics.

### Downstream Ablations

Use identical GE-Act initialization and training data for:

1. GE-Act without a VLM semantic condition;
2. future TA codes only;
3. TA codes plus Qwen hidden states;
4. full codes, text-aligned semantic branch, grounding, and attention bias.

The primary result is LIBERO task success. Planner-only metrics are diagnostic
and cannot substitute for downstream control performance.

## Failure Handling and Reproducibility

- Cache/model/preprocessing hash mismatch is fatal.
- Cross-trajectory train/validation leakage is fatal.
- Invalid keyframe bounds are fatal during cache construction.
- Low-confidence grounding disables only the affected auxiliary term; code
  prediction remains valid.
- Missing action/gripper data lowers phase confidence and invokes the
  visual-only fallback.
- Non-finite teacher maps are discarded and recorded with trajectory/frame
  identifiers.
- Constrained decoding rejects illegal vocabulary IDs and malformed frame
  boundaries.
- Checkpoints record every teacher hash, vocabulary range, phrase parser
  version, camera order, keyframe offsets, loss weights, and optimizer group.
- All generated caches, checkpoints, logs, and visualizations use new
  directories and never overwrite legacy Qwen3-VL-2B or Qwen3.5 baselines.

## Verification

Unit coverage must include:

- 384 preprocessing and exact 27x27/729 geometry;
- Qwen vocabulary expansion without a 1,536-to-2,048 initialization
  projection;
- correct `h_pre`/`h_post` causal alignment, including the final code token;
- prompt field-position alignment and phrase-anchor prediction without
  importing SigLIP2 at inference;
- independent main/wrist causal state;
- phrase parsing and missing-field confidence behavior;
- target-cache quantization round trip and hash rejection;
- grounding loss confidence masking;
- counterfactual negative construction;
- DINO temporal warp shape and boundary behavior;
- 27x27 to 8x8 coverage pooling;
- top-32 selection, de-duplication, masks, and coordinates;
- zero-gate numerical equality with baseline GE-Act;
- inference graph independence from DINO, future RGB, actions, and gripper
  signals;
- legacy Qwen3-VL-2B and GE-Act providers remaining unchanged.

Integration verification must run:

- a tiny target-cache build from a complete trajectory;
- one planner training/validation step;
- constrained generation of all four 729-code frames;
- extraction of all post-code hidden states and relevance maps;
- one GE-Act video/action forward-backward step;
- the single-image counterfactual visualization.

## Approved Design Summary

The released 384 TA-Tok remains a generic, frozen visual tokenizer. Qwen3.5
predicts its future discrete codes while a separate post-code semantic branch
learns a text-aligned, spatially indexed representation from complete-video
hindsight. GE-Act receives a fused visual-semantic plan plus an explicit
relevance bias. This separates visual reconstructability from task grounding,
avoids an unvalidated codebook-to-Qwen vocabulary projection, and allows
downstream video/action losses to make the planner more useful for control.
