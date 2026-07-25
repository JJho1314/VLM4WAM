# Qwen3.5 Plan-X-Style TA-Tok Planner Design

Date: 2026-07-25

Status: Approved

## Objective

Replace the current Qwen3.5 planner's fixed learnable-query/K-means path with a
Plan-X-style causal semantic planner adapted to this project's visual backbone:

- Qwen3.5-VL remains the multimodal backbone and consumes raw current RGB plus
  the instruction.
- A project-specific Text-Aligned Tokenizer (TA-Tok) uses
  SigLIP2-Large/Patch16/256 and produces a fixed 16x16 grid.
- Qwen3.5 autoregressively predicts four future semantic keyframes per camera
  using an expanded 65,536-entry visual vocabulary.
- Main-camera and wrist-camera plans are predicted independently with shared
  model weights.
- GE-Act consumes the generated visual-token hidden states through its existing
  same-camera semantic cross-attention and 3D RoPE path.

The design aligns the tokenizer, autoregressive prediction, semantic hidden
state injection, and staged training mechanics with Plan-X while intentionally
retaining Qwen3.5-VL and the project's SigLIP2 resolution.

## Compatibility Boundary

This work is isolated from the existing Qwen3-VL-2B implementation.

- Do not modify the existing trainers, providers, checkpoint formats, or
  launchers under `qwen3_vl_semantic_planner`.
- Put the Qwen3.5 TA-Tok, planner, data preparation, evaluation, and launch
  code in a new Qwen3.5-specific module tree.
- Never resize or save over the base Qwen3.5 model or tokenizer in place.
  Vocabulary expansion happens in memory and is saved only in the new
  experiment directory.
- Add GE-Act support through a new explicitly selected
  `planner_backend=qwen35_planx` provider. Existing providers and default
  behavior remain unchanged.
- Store all labels, checkpoints, logs, and visualizations in new directories.
- Add regression coverage proving that the legacy Qwen3-VL-2B path is
  unchanged when the new backend is not selected.

## Non-Goals

- Reproducing Plan-X's Qwen2.5-Instruct backbone.
- Using the released TA-Tok checkpoint, which targets
  SigLIP2-SO400M/Patch14/384 and a different token grid.
- Retaining the existing 1,024 learnable planner queries.
- Sharing causal state between the main and wrist camera streams.
- Overwriting or resuming from the existing K-means/query planner checkpoints.

The current batch-32 K-means/query run remains a complete baseline.

## Architectural Decisions

### No Planner Query Tokens

The current `K*P` learnable query module is removed from the new path. It is not
augmented with query self-attention. Instead, the generated TA token at every
position is a normal causal language-model token, and all previously generated
TA tokens form the causal context for subsequent positions.

This matches the Plan-X main method. The learnable-query connector described in
Plan-X is an ablation baseline, not the main semantic planner. The `Q` used in
the downstream cross-attention equations is the ordinary query projection of
video latents, not an additional set of planner query tokens.

### Camera Layout

The main camera (`observation.images.image`) and wrist camera
(`observation.images.wrist_image`) are separate examples in an effective
`B*2` planner batch.

Both streams share:

- Qwen3.5-VL weights;
- the expanded visual vocabulary;
- the TA-Tok;
- all optimizer parameters.

A camera special token identifies the stream. The two streams do not attend to
one another and may be decoded in parallel as two batch elements.

### Future Sequence

For a current frame at index `c`, each camera predicts the four existing
project keyframes:

```text
c + 1, c + 4, c + 6, c + 9
```

These correspond to future offsets `[0, 3, 5, 8]` after the first future
frame. Every frame contains a raster-ordered 16x16 grid, or 256 TA tokens.
Each camera therefore predicts 1,024 visual tokens, and a dual-camera sample
contains 2,048 supervised visual tokens.

The target layout is:

```text
<plan_start>
  <frame_start> 256 TA codes <frame_end>
  <frame_start> 256 TA codes <frame_end>
  <frame_start> 256 TA codes <frame_end>
  <frame_start> 256 TA codes <frame_end>
<plan_end>
```

Frame delimiter positions are deterministic. At inference, constrained
decoding permits only TA vocabulary IDs at code positions and the required
delimiter at boundary positions.

## Domain TA-Tok

### Data

Train the tokenizer on every available frame from both cameras across
LIBERO-Object, LIBERO-Spatial, LIBERO-Goal, and LIBERO-10. Split by trajectory
before extracting frames so adjacent frames from an episode never cross the
train/validation boundary.

The TA-Tok stage uses raw frames rather than only the 22,278 planner windows.
No text captions are required for this stage; text alignment is inherited from
the frozen Qwen3.5 embedding anchors.

### Components

Let the Qwen3.5 hidden width be `Dq`.

- A trainable SigLIP2-Large/Patch16/256 student produces 256 spatial features
  of width 1,024.
- A frozen copy of the same SigLIP2 model is the reconstruction teacher.
- Select 65,536 representative Qwen3.5 vocabulary embeddings
  `E_selected` of shape `[65536, Dq]`. The selection is deterministic and
  recorded in the tokenizer checkpoint.
- Keep `E_selected` frozen.
- A learned projection constructs the text-aligned visual codebook from the
  frozen anchors.
- A learned student projection maps SigLIP2 features into the codebook space.
- A lightweight three-block reconstruction decoder maps quantized
  representations back to the frozen teacher's 1,024-dimensional feature
  space.

The output remains a 16x16 code grid. Scale-adaptive pooling from the released
TA-Tok is not used because the chosen project encoder already produces the
required 16x16 grid.

### Objective

Train with:

- cosine reconstruction loss against the frozen SigLIP2 teacher;
- VQ commitment loss;
- codebook projection loss with stop-gradient placement matching TA-Tok.

Track code usage, codebook perplexity, dead-code ratio, reconstruction cosine,
and per-camera validation metrics. A tokenizer exhibiting code collapse cannot
be used to build planner labels.

After validation, freeze the TA-Tok and its vocabulary mapping permanently for
all subsequent stages.

## Offline Planner Dataset

Build a new trajectory-split cache containing:

- current main-camera RGB;
- current wrist-camera RGB;
- instruction text;
- main-camera future TA codes of shape `[4, 256]`;
- wrist-camera future TA codes of shape `[4, 256]`;
- trajectory ID, suite, frame indices, camera preprocessing metadata, and
  TA-Tok checkpoint hash.

Code IDs use `uint16`, whose range exactly covers IDs 0 through 65,535.
Planner preprocessing must fail fast when the tokenizer hash, vocabulary
mapping, camera order, keyframe indices, or image normalization does not match
the training configuration.

## Qwen3.5 Causal Planner

### Vocabulary Expansion

Add:

- 65,536 TA visual tokens;
- plan and frame boundary tokens;
- main-camera and wrist-camera identity tokens.

Initialize the new TA token embedding/output rows from the trained text-aligned
visual codebook. Text vocabulary rows retain their base initialization.
Expanded tokenizer/model artifacts are saved only with this experiment.

### Training Format

For each camera, Qwen3.5-VL receives:

- the raw current camera RGB through its native vision path;
- the instruction;
- the camera identity token;
- the target duration and four-keyframe output format.

The assistant response is the future TA sequence. Training uses teacher forcing
and the native fully causal attention mask. Prompt, current-image, camera, and
formatting tokens receive label `-100`; cross-entropy is computed only on the
future TA code positions and required structural delimiters.

Qwen3.5 vision, language, and newly added visual vocabulary parameters are
trainable. The old query module and K-means classifier are absent.

### Inference

Decode main and wrist streams as two batch elements with shared weights and
independent KV caches. At each semantic code position, mask logits to the TA
vocabulary. Insert or constrain deterministic frame delimiters. Validate exact
length and structure before returning a plan.

Return:

- predicted code IDs `[B, 2, 4, 256]`;
- pre-output-head Qwen hidden states `[B, 2, 4, 256, Dq]`;
- per-token probabilities or confidence summaries for diagnostics.

Malformed or incomplete generations are rejected rather than padded with
arbitrary codes.

## GE-Act Integration

Add an optional Qwen3.5 Plan-X provider. It returns Qwen hidden states rather
than K-means codebook vectors.

GE-Act keeps:

- same-camera semantic cross-attention;
- explicit camera/view identity;
- semantic condition dropout;
- time-aligned `(t, y, x)` coordinates;
- independent query/key 3D RoPE.

Its semantic adapter accepts `Dq` and projects directly to the LTX hidden width.
The four keyframe times retain the existing resolution-aware mapping to the LTX
latent grid.

Training is staged:

1. Train the semantic branch using ground-truth TA tokens.
2. Replace ground-truth conditions with planner-produced pre-head hidden states.
3. Jointly fine-tune planner and GE-Act with:

```text
total_loss = ge_act_loss + 0.1 * ta_token_cross_entropy
```

The action path remains enabled during complete GE-Act training.

## Evaluation and Acceptance

Use the finished batch-32 K-means/query experiment as the baseline on the same
trajectory-held-out split.

### TA-Tok

Report:

- feature reconstruction cosine;
- VQ loss;
- vocabulary coverage and perplexity;
- dead-code ratio;
- results split by suite and camera.

### Planner

Report:

- token cross-entropy and accuracy;
- accuracy by camera, keyframe, and spatial position;
- decoded semantic cosine against frozen SigLIP2 teacher features;
- temporal consistency between consecutive predicted keyframes;
- constrained-decoding validity rate;
- autoregressive latency and tokens per second.

Generate separate main-camera and wrist-camera visualizations of current RGB,
future RGB, target semantic maps, predicted semantic maps, and causal attention
heatmaps.

### GE-Act

Evaluate:

- no-semantic GE-Act baseline;
- current K-means/query planner;
- Qwen3.5 Plan-X TA-Tok planner;
- ground-truth TA-token upper bound.

Report LIBERO success for all four suites, average success, and per-camera or
per-keyframe semantic ablations. A planner checkpoint is not considered
superior based on token accuracy alone; downstream success is the final
selection criterion.

## Testing and Failure Handling

Required automated coverage includes:

- trajectory-level split isolation;
- exact two-camera ordering;
- Qwen3-VL-2B legacy-path regression;
- deterministic vocabulary and anchor selection;
- visual embedding/output-head initialization;
- full causal masking with no future-token leakage;
- loss masking restricted to assistant TA positions;
- constrained decoding and exact output structure;
- offline-cache metadata/hash rejection;
- planner provider output shapes;
- GE-Act backend gating;
- same-camera cross-attention and 3D RoPE preservation.

Run a one-batch overfit test before distributed training. Run a short
single-GPU autoregressive smoke test before multi-GPU launch. Any OOM, invalid
generation, code collapse, NaN, tokenizer mismatch, or legacy regression blocks
the next training stage.

## References

- Plan-X: Instruct Video Generation via Semantic Planning,
  https://arxiv.org/abs/2511.17986
- Vision as a Dialect: Unifying Visual Understanding and Generation via
  Text-Aligned Representations, https://arxiv.org/abs/2506.18898
- Official Tar/TA-Tok code, https://github.com/csuhan/Tar
