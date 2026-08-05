# WorldArena Baton Planner Repair Design

## Status and scope

This design repairs the WorldArena Qwen3.5 planner before any new long
training run. It covers the video-only Stage-1 planner and its validation
contract. GE-Act semantic adaptation and planner-noise adaptation remain a
separate follow-on stage because they require a planner checkpoint that has
already passed the grounding gates in this document.

The implementation keeps the user-selected backbones and geometry:

- Qwen3.5-2B with its current vision encoder trainable;
- SigLIP2-large-patch16-256, penultimate hidden states, frozen;
- one head-camera observation for WorldArena;
- four future keyframes and 256 continuous SigLIP2 patch tokens per keyframe;
- global batch size 128, bf16, AdamW, and learning rate `1e-5`;
- a single positive training sequence with pointwise feature regression.

It must not restore the previously removed positive/negative four-sequence
training path.

## Evidence motivating the repair

The step-15000 checkpoint predicts future features better than current-feature
persistence, but it is only weakly conditioned on language:

- correct instruction: MSE 3.219 and cosine 0.661;
- shuffled instruction: MSE 3.344 and cosine 0.654;
- persistence: MSE 4.816 and cosine 0.539;
- correct instructions beat shuffled instructions in only 23 of 44 validation
  episodes;
- the prediction norm and spatial standard deviation are respectively 70.7%
  and 73.9% of the target values.

The implementation audit found four actionable causes:

1. Plan placeholders are currently rendered inside the user message, whereas
   Baton renders the blueprint scaffold as the assistant response.
2. The four frame blocks represent a normalized remaining horizon, but neither
   the current position nor target timestamps are supplied to Qwen.
3. All 465 WorldArena training prompts share a 169-character boilerplate
   prefix, approximately 62% of the average prompt by word count.
4. The dataset exposes one stochastic window per episode per epoch. At step
   15000 this reports epoch 4285, obscuring the finite set of 54,405 possible
   `(episode, current frame)` training examples and encouraging excessive
   repetition.

## Approaches considered

### Selected: Baton-corrected single-path planner

Keep the current observation because robot prediction needs instance and
spatial state, but move the blueprint scaffold to the assistant role, expose
the exact temporal coordinates, train over an explicit finite window index,
and select checkpoints using validation grounding diagnostics. This changes
the minimum number of learning assumptions while preserving the requested
single-path training cost.

### Rejected for this repair: hard-negative training rows

Running the same image with a mismatched instruction and adding a ranking loss
would make instruction dependence more explicit, but it reintroduces the
positive/negative branch that was deliberately removed and increases Qwen
compute. It can be reconsidered only if the corrected single-path planner still
fails the grounding gate.

### Rejected for this repair: strict text-only Baton

Removing the current image would make language unavoidable and more closely
match Baton's text-to-video Stage 1, but it would discard the current robot and
object layout required for spatially grounded future prediction.

## Input and sequence contract

### Conversation layout

Every Qwen row will use a complete three-message conversation:

```text
system:
  You are a helpful assistant that predicts spatially grounded future visual
  semantic blueprints for embodied robot videos.

user:
  [current camera image]
  Predict four future semantic keyframes for this observation.
  Instruction: {discriminative instruction}
  Current frame: {c}/120, normalized time {c/120 formatted to six decimals}.
  Target frames: {f0}/120, {f1}/120, {f2}/120, {f3}/120.

assistant:
  <PLAN_START>
  <FRAME_0> <PLAN_PAD> x 256
  <FRAME_1> <PLAN_PAD> x 256
  <FRAME_2> <PLAN_PAD> x 256
  <FRAME_3> <PLAN_PAD> x 256
  <PLAN_END>
```

The processor will render the complete conversation with
`add_generation_prompt=False`. The resulting row must contain exactly one
image, four ordered frame markers, and 1,024 plan-pad positions. Hidden states
at those assistant-side plan-pad positions remain the inputs to the visual
alignment tower.

### Discriminative instruction

WorldArena cache records retain the original instruction unchanged for
provenance. At collation time, the exact shared WorldArena boilerplate is
removed only when it is present. The remaining task clause must be nonblank.
Other datasets, including LIBERO, keep their full original instruction.

The original and rendered instructions are both carried in the batch metadata
so validation artifacts remain auditable.

### Temporal coordinates

WorldArena samples already contain source indices
`(c, f0, f1, f2, f3)`. The collator validates that they are integers satisfying
`0 <= c < f0 < f1 < f2 < f3 <= 120` and inserts them into the user message.
No learned time projection or additional query-tower parameters are introduced
in this repair. This preserves checkpoint topology and isolates whether the
previous ambiguity was caused by missing conditioning rather than tower size.

LIBERO continues to use its existing fixed temporal policy. Its dataset adapter
is extended to emit the current and four future indices relative to the loaded
video window, so every new-template batch has explicit temporal coordinates.

## Dataset indexing and training duration

WorldArena training will use an explicit all-window index:

```text
record_index = index // 117
current_index = index % 117
future_indices = normalized_remaining_horizon(current_index)
```

For 465 training episodes this yields exactly 54,405 examples per epoch. The
dataset no longer changes the selected current frame through shared epoch
state in this mode; sampler epochs only control row order. The legacy
one-window-per-episode mode remains available for old checkpoints and tests.

With global batch size 128, one all-window epoch is approximately 426 optimizer
steps. The new default WorldArena run uses 5,000 optimizer steps, approximately
11.7 exhaustive epochs, rather than 30,000 steps over stochastic episode rows.

## Model and loss contract

The planner topology remains:

```text
Qwen3.5 final hidden states at 1,024 assistant plan pads
  -> one learned query per target patch
  -> one visual cross-attention layer
  -> Sem-MLP 2048 -> 2048 -> 1024
  -> [batch, camera, 4, 256, 1024]
```

The Stage-1 objective remains Baton's continuous pointwise MSE against frozen
SigLIP2 penultimate patch features. No cosine, variance, ranking, or negative
loss is added in this repair. Keeping the loss fixed ensures that improvements
can be attributed to the corrected sequence, temporal conditioning, and data
coverage.

All Qwen text layers, the Qwen vision encoder, plan-token embeddings, query
tower, and Sem-MLP remain trainable. The SigLIP2 teacher remains frozen.

## Checkpoint compatibility

New checkpoints add explicit metadata fields:

- `input_template_kind = "baton_assistant_time_v2"`;
- `worldarena_sampling_kind = "all_windows_v1"`;
- `instruction_rendering_kind = "strip_worldarena_boilerplate_v1"`.

The checkpoint format version is incremented from 4 to 5. Format-4 checkpoints that
lack these fields are interpreted as:

- `input_template_kind = "legacy_user_plan_v1"`;
- `worldarena_sampling_kind = "episode_random_v1"`;
- `instruction_rendering_kind = "verbatim_v1"`.

Inference and evaluation select their collator behavior from checkpoint
metadata. Existing step-15000 checkpoints must remain loadable and must produce
the same legacy row layout.

## Validation and checkpoint selection

A deterministic validation pass runs every 500 optimizer steps over all 44
validation episodes. It evaluates:

1. correct instruction;
2. a task-distinct shuffled instruction while keeping the current image fixed;
3. current-feature persistence.

Validation records aggregate and per-horizon MSE/cosine, correct-versus-shuffle
sample wins, prediction/target norm ratio, prediction/target spatial standard
deviation ratio, and future-delta cosine. Task-distinct shuffling must never
pair an episode with another instruction from the same task when a different
task exists.

The training run saves an early diagnostic checkpoint at step 20 and evaluated
checkpoints at steps 500, 1000, 2000, 3000, 4000, and 5000. A checkpoint is
eligible for downstream GE-Act work only if it satisfies all of these gates on
the same validation pass:

- finite outputs for all 44 episodes;
- at least 60% correct-instruction wins over task-distinct shuffled
  instructions;
- at least 5% aggregate MSE improvement over shuffled instructions;
- at least 25% aggregate MSE improvement over persistence;
- prediction-to-target norm ratio between 0.85 and 1.15.

These are diagnostic eligibility gates, not values that the implementation is
allowed to fabricate or assume. If no checkpoint passes, training stops at
5,000 steps and reports the failed gates instead of silently extending the run.

## Failure handling and observability

- Invalid or nonmonotonic source indices fail during collation before a model
  forward.
- Missing assistant-side plan positions fail with a message naming the
  template kind and observed token count.
- Checkpoint metadata and runtime configuration must agree on template,
  sampling, and instruction-rendering kinds before resume.
- Validation writes an atomic JSON artifact and retains per-sample task and
  source-index provenance.
- A failed grounding gate is a model-quality result, not an infrastructure
  exception; it must not trigger automatic step-count inflation.

## Test strategy

Implementation follows red-green-refactor cycles for each boundary:

1. Sequence tests prove plan pads occur inside the assistant message and legacy
   rendering remains byte-for-byte stable.
2. Data tests prove exact boilerplate stripping, time rendering, monotonic
   index rejection, and one-image/1,024-pad batching.
3. Dataset tests prove all 54,405 training windows are enumerated exactly once
   per epoch and legacy stochastic indexing remains available.
4. Checkpoint tests prove format-4 legacy loading and new-format metadata
   round-trips without ambiguity.
5. Validation tests prove task-distinct shuffling, metric calculations, gate
   decisions, and atomic artifact output.
6. A tiny two-rank training smoke test proves the new configuration can take an
   optimizer step and save/resume without changing the global batch contract.

## Follow-on GE-Act stages

After a planner checkpoint passes the grounding gates, a separate design and
implementation will apply the rest of Baton's curriculum:

1. train GE-Act semantic cross-attention using ground-truth future SigLIP2
   features;
2. freeze the accepted planner and continue GE-Act training using predicted
   planner features;
3. retain the original uncontrolled GE-Act inference path for baseline
   evaluation.

This ordering prevents planner prediction noise from being confused with a
broken GE-Act conditioning interface.
