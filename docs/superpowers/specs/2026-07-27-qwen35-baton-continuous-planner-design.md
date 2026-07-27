# Qwen3.5 Baton-Style Continuous Semantic Planner Design

Date: 2026-07-27
Status: approved design, pending implementation plan

## 1. Objective

Replace the Qwen3.5 planner's TA-Tok/codebook/autoregressive discrete-code
path with a Baton-style continuous semantic planner.

Given one current camera image and a LIBERO instruction, the planner predicts
four future keyframe grids in the continuous feature space of a frozen
SigLIP2 teacher. Main and wrist cameras are processed independently with
shared weights. GE-Act consumes the complete predicted grids through
same-camera semantic cross-attention and Relative Semantic RoPE.

The existing Qwen3-VL-2B planner remains unchanged.

The planner backbone is the dense Qwen3.5-4B vision-language model, not a
mixture-of-experts variant.

## 2. Motivation

The required output is a spatially structured, instruction-conditioned future
representation, not a compact language-like code sequence. The discrete
TA-Tok route adds codebook training, vocabulary expansion, thousands of
autoregressive decoding steps, quantization error, and an additional
text-alignment problem.

Baton demonstrates a simpler interface:

1. use one learnable query slot per target perceptual token;
2. condition those slots on MLLM hidden states;
3. project them into a frozen perceptual encoder's continuous feature space;
4. supervise them directly against position-matched teacher features; and
5. teach the diffusion model first with clean teacher features and then with
   predicted features.

This interface matches the GE-Act use case more directly.

## 3. Scope and Non-Goals

### In scope

- A new continuous Qwen3.5 planner namespace.
- Two independent camera rows with shared Qwen and Query Tower weights.
- Four future keyframes at offsets `[0, 3, 5, 8]`.
- A frozen `SigLIP2-large-patch16-256` online teacher.
- Full `16 x 16` continuous grids at every keyframe.
- Block-causal spatiotemporal query reasoning.
- Direct feature, direction, temporal-change, and instruction-counterfactual
  supervision.
- Full-grid injection into GE-Act with Relative Semantic RoPE.
- A three-stage training curriculum.
- Strict checkpoint metadata and compatibility validation.

### Out of scope for the first implementation

- TA-Tok or any learned visual codebook in the Qwen3.5 main path.
- Autoregressive visual-code decoding.
- DINO, DA3, or depth targets.
- Top-token selection or `96`-token compression.
- Relevance logit bias in LTX attention.
- Main/wrist cross-attention.
- Full-Qwen3.5 fine-tuning.
- Joint Qwen3.5 and GE-Act updates in the baseline Stage 3.
- Deleting legacy TA-Tok code or old checkpoints.

## 4. Fixed Geometry

| Dimension | Value |
|---|---:|
| Cameras | 2: main, wrist |
| Future keyframes | 4 |
| Future offsets | `[0, 3, 5, 8]` |
| SigLIP2 input | `256 x 256` |
| SigLIP2 patch size | `16` |
| Spatial grid | `16 x 16` |
| Tokens per keyframe | `256` |
| Target feature width | `1024` |
| Query tokens per camera | `4 x 256 = 1024` |
| Query tokens per sample | `2 x 4 x 256 = 2048` |
| Planner output | `[B, 2, 4, 256, 1024]` |

The teacher target is the penultimate SigLIP2 vision-layer patch output. Class
or pooling tokens are not targets.

## 5. Independent Camera Input Contract

Each sample is flattened into two sample-major Qwen rows:

```text
sample 0 main
sample 0 wrist
sample 1 main
sample 1 wrist
...
```

Each row contains exactly one camera image and the shared instruction:

```text
<SYSTEM>
<CURRENT_IMAGE>
Instruction: {instruction}
<PLAN_START>
<FRAME_0> <PLAN_PAD> x 256
<FRAME_1> <PLAN_PAD> x 256
<FRAME_2> <PLAN_PAD> x 256
<FRAME_3> <PLAN_PAD> x 256
<PLAN_END>
```

The Qwen input image uses the resolution and preprocessing defined by its
persisted processor. The SigLIP2 teacher independently receives a `256 x 256`
image. These resolutions do not need to match.

Main and wrist rows do not share visual tokens or cross-attend to each other.
They share all model weights and differ through their images, camera-view
embedding, and spatial evidence.

## 6. Qwen Hidden-State Extraction

Qwen3.5 performs one causal forward over each camera row. No visual token is
sampled or decoded.

The hidden states at the four blocks of `<PLAN_PAD>` positions are gathered and
reshaped to:

```text
[B, 2, 4, 256, D_qwen]
```

Because every planning block occurs after the current image and instruction,
all plan positions can attend to the complete observation and task text.
Later plan blocks can also attend to earlier plan blocks through Qwen's causal
attention.

The text template, special-token IDs, image processor, padding side, M-RoPE
state, and flatten/unflatten order are part of the persisted checkpoint
contract.

The planner vocabulary adds `<PLAN_START>`, four `<FRAME_n>` tokens, one
`<PLAN_PAD>` token, and `<PLAN_END>`. A dedicated plan-token embedding adapter
owns and trains only these added rows. The base Qwen token-embedding matrix
remains frozen.

## 7. Spatiotemporal Query Tower

### 7.1 Queries

The tower owns:

```text
Q: [4, 256, D_query]
```

with one learnable query for every `(future_time, y, x)` target position.
Queries receive:

- a future-frame embedding;
- a `16 x 16` spatial position;
- a camera-view embedding; and
- three-dimensional `(t, y, x)` RoPE.

### 7.2 Block-causal self-attention

The `256` queries within one future frame attend bidirectionally to each
other. A query at future frame `t` may attend to:

- every query in frame `t`; and
- every query in frames earlier than `t`.

It may not attend to a later future frame. This is a block-causal mask, not a
raster-token causal mask.

### 7.3 Cross-attention

Query states cross-attend the corresponding camera row's gathered Qwen plan
hidden states. The cross-attention mask is also block-causal by future frame:
frame `t` may read gathered Qwen states from frames `0..t`, but not later
frames. Camera rows remain isolated.

The tower has four identical blocks with:

- hidden width `1024`;
- `16` attention heads;
- feed-forward width `4096`; and
- dropout probability `0.1`.

Each block contains:

```text
block-causal self-attention
cross-attention to Qwen plan hidden states
feed-forward network
```

Residual paths and normalization follow the repository's existing
pre-normalized transformer convention.

### 7.4 Continuous projection

A shared Sem-MLP maps Query Tower hidden states to the `1024`-dimensional
SigLIP2 target space:

```text
Linear(1024, 2048) -> GELU -> Linear(2048, 1024)
```

No vector quantization, code lookup, or discrete vocabulary is used.

## 8. Online Frozen SigLIP2 Teacher

The HDF5 data path supplies predecoded RGB frames. For each sample:

1. select main and wrist future frames at offsets `[0, 3, 5, 8]`;
2. resize them to `256 x 256` using the persisted teacher preprocessing;
3. run frozen `SigLIP2-large-patch16-256` in frame microbatches;
4. extract penultimate-layer patch tokens; and
5. reshape them to `[B, 2, 4, 256, 1024]`.

Teacher extraction runs under `no_grad`, in evaluation mode, and outside the
optimizer. Full feature targets are not cached to disk.

The input HDF5 manifest and frame-index contract remain content-hashed and
fail closed.

## 9. Planner Objectives

The current observation strongly predicts static scene content, so pure MSE
could learn to copy current-frame features and ignore the instruction. Stage
1 therefore uses four complementary terms.

### 9.1 Feature regression

```text
L_mse = weighted mean squared error(predicted, teacher)
```

The per-patch weight is:

```text
r = L2_norm(future_teacher - current_teacher)
weight = 1 + clamp(r / (mean(r) + 1e-6), min=0, max=2)
```

The mean is computed independently per sample, camera, and future keyframe.
Weights therefore lie in `[1, 3]`: unchanged background remains supervised
while changed regions receive more weight.

### 9.2 Directional alignment

```text
L_cos = 1 - cosine_similarity(predicted, teacher)
```

This stabilizes the semantic direction independently of feature magnitude.

### 9.3 Future-change prediction

```text
L_delta = distance(
    predicted_future - current_teacher,
    teacher_future - current_teacher,
)
```

This explicitly penalizes static copying.

### 9.4 Instruction counterfactual

For every training example, sample a different instruction from the same
LIBERO suite. The wrong instruction must have worse teacher alignment than
the correct instruction by margin `0.1`:

```text
L_instruction_cf = max(
    0,
    0.1 + cosine_distance(correct_prediction, teacher)
        - cosine_distance(wrong_prediction, teacher),
)
```

Negatives are drawn from the same suite to prevent scene/domain shortcuts.
The image remains unchanged.

### 9.5 Total Stage-1 objective

```text
L_planner =
    1.0 * L_mse
  + 0.5 * L_cos
  + 0.5 * L_delta
  + 0.2 * L_instruction_cf
```

Loss terms and their weights are checkpoint metadata and configuration
fields.

## 10. Trainable Parameters

During Stage 1:

### Trainable

- Query Tower;
- Sem-MLP;
- camera/frame/position embeddings;
- Qwen3.5 vision tower;
- top eight Qwen3.5 language layers.

### Frozen

- lower Qwen3.5 language layers;
- base Qwen token embeddings;
- final language norm and unrelated heads;
- SigLIP2 teacher;
- GE-Act.

Optimizer ownership is explicit and exhaustive. It must not depend on
substring matching alone.

Stage-1 optimizer groups are:

| Group | Learning rate |
|---|---:|
| Query Tower, Sem-MLP, plan-token adapter | `5e-5` |
| Qwen top eight language layers | `1e-6` |
| Qwen vision tower | `5e-7` |

The effective global batch is `128`. Per-GPU batch and gradient accumulation
are selected by a memory preflight without changing this effective batch or
loss normalization.

## 11. GE-Act Semantic Interface

The planner output is not compressed:

```text
[B, 2, 4, 256, 1024]
```

For each camera it is flattened to:

```text
[B * 2, 1024, 1024]
```

A Semantic Adapter projects `1024` to the LTX hidden width.

The semantic position of each token is its exact future keyframe time and
`16 x 16` patch center:

```text
(t, y, x)
```

LTX latent queries retain their own latent-grid positions. Relative Semantic
RoPE maps both token sets into one continuous coordinate frame before
semantic cross-attention.

Semantic attention is same-camera only. In every selected LTX block, text
cross-attention establishes coarse task context first and semantic
cross-attention then adds fine-grained future structure.

The existing zero-initialized semantic residual gate remains. A checkpoint
without continuous planning fields must preserve the old forward output.

Relevance bias, exact-token selection, and `96`-token compression are disabled
for this baseline.

## 12. Three-Stage Curriculum

### Stage 1: Continuous Planner Pretraining

- Train for `30,000` optimizer steps.
- Save every `5,000` steps.
- Train Qwen vision, top eight language layers, Query Tower, and Sem-MLP.
- Freeze SigLIP2 and GE-Act.
- Use the objective in Section 9.

### Stage 2: Clean-Blueprint GE-Act Adaptation

- Train for `20,000` optimizer steps.
- Save every `5,000` steps.
- Freeze the planner and teacher.
- Feed ground-truth future SigLIP2 grids directly to GE-Act.
- Train the LTX video expert, action expert, and Semantic Adapter.

This stage teaches GE-Act how to use the semantic interface without planner
prediction noise.

Stage-2 optimizer learning rates are `2e-5` for the LTX video expert, `1e-4`
for the action expert, and `5e-5` for the Semantic Adapter. The effective
global batch is `128`.

### Stage 3: Predicted-Blueprint GE-Act Adaptation

- Train for `30,000` optimizer steps.
- Save every `5,000` steps.
- Freeze the planner.
- Replace teacher grids with predicted grids.
- Continue training the LTX video expert, action expert, and Semantic Adapter.

This stage removes exposure bias between clean teacher grids and inference
conditions.

Stage 3 retains the Stage-2 optimizer groups, learning rates, and effective
global batch.

Full planner/GE-Act joint fine-tuning is not part of the baseline. It may be
evaluated later as a separate Stage-4 ablation.

## 13. Checkpoint Contract

Every continuous-planner checkpoint records and validates:

- architecture kind: `qwen35_baton_continuous`;
- Qwen config, tokenizer, processor, and input-template hashes;
- all added special tokens and exact IDs;
- camera order and sample-major flattening;
- SigLIP2 model identifier and artifact hash;
- teacher image size, patch size, feature layer, preprocessing, and dtype;
- target shape `[2, 4, 256, 1024]`;
- future offsets `[0, 3, 5, 8]`;
- Query Tower geometry and block-causal mask version;
- Qwen trainable-layer ownership;
- loss definitions and weights;
- HDF5 manifest hash;
- optimizer and scheduler topology;
- exact distributed training cursor and RNG state.

Old TA-Tok checkpoints are rejected by the continuous loader. Existing legacy
loaders remain available for their original branches.

## 14. Validation and Acceptance

### Unit and contract tests

- Exact output shape `[B, 2, 4, 256, 1024]`.
- Exact SigLIP2 preprocessing and penultimate-layer extraction.
- One query per target patch.
- Main/wrist isolation and sample-major unflattening.
- Block-causal attention: same frame and past frames allowed, future frames
  forbidden.
- Correct `(t, y, x)` positions and RS-RoPE scaling.
- No discrete vocabulary, codebook, or autoregressive decoder dependency.
- Correct gradient ownership: Query Tower, Sem-MLP, Qwen vision, and top eight
  language layers receive gradients; all frozen modules do not.
- Teacher features never receive gradients.
- Legacy Qwen3-VL-2B and old GE-Act paths remain unchanged.

### Planner metrics

- Feature MSE.
- Mean patch cosine similarity.
- Future-delta error.
- Counterfactual instruction ranking accuracy.
- Per-keyframe and per-camera metrics.
- Attention maps for instruction-to-query and query-to-Qwen inspection.

### GE-Act metrics

- Stage-2 teacher-condition validation.
- Stage-3 predicted-condition validation.
- LIBERO and LIBERO-Plus task success.
- Video prediction metrics.
- Action prediction metrics.
- Performance with semantic condition disabled as a regression baseline.

### Required smoke tests

- One local Stage-1 forward/backward/optimizer step.
- One Stage-2 GE-Act step with teacher features.
- One Stage-3 GE-Act step with predicted features.
- Two-rank distributed gradient and exact-resume smoke.
- Launcher and environment preflight that validates Qwen3.5 and SigLIP2
  artifacts before allocating either large model.

Live 4B or eight-GPU success is reported only when those executions actually
run.

## 15. Migration

The continuous planner is introduced in a new namespace and configuration.
The existing Qwen3-VL-2B implementation is not modified.

The current Qwen3.5 TA-Tok implementation becomes a legacy experimental path.
Its files are not deleted in the first migration so existing checkpoints and
comparisons remain reproducible. New default Qwen3.5 launchers and GE-Act
configs point only to the continuous planner.

No discrete-planner checkpoint is silently converted. A future distillation
or conversion experiment requires a separate design.

## 16. Reference

This design adapts the continuous semantic-alignment and three-stage training
principles from:

Shuyuan Tu et al., *Baton: Explicit Semantic Blueprints for Joint Video-Audio
Generation*, 2026, local preprint:

`/data/LFT-W02_data/junjie/workspace/VLM4WAM/Baton- Explicit Semantic Blueprints for Joint Video-Audio Generation.pdf`

The adaptation differs from Baton by conditioning on a current robot camera
image, maintaining independent main/wrist predictions, using a `256 x 256`
SigLIP2 teacher, and adding future-change plus instruction-counterfactual
supervision to prevent current-frame copying.
