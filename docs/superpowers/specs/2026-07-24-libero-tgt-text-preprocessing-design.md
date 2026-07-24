# LIBERO `[TGT]` Text Preprocessing Design

## Goal

Restore the target-aware text convention used by the Cosmos WM reference
before retraining the semantic planner and the frozen-planner GE-Act variant.
Every training and inference path must present the same marked instruction to
its text encoder.

## Scope

This change covers:

- standalone dual-camera semantic-planner training;
- standalone semantic-planner inference;
- joint VLM/GE-Act training, for both the Qwen planner prompt and the GE-Act
  T5 text condition;
- joint LIBERO inference, for both text paths;
- checkpoint metadata and compatibility validation;
- coverage tests over all four LIBERO task suites.

It does not add the Cosmos TAViD target-attention loss, target masks,
InstructSAM features, or target-token indices. It does not change model
architecture, query-token geometry, loss weights, or optimizer policy.

## Instruction Transformation

The preprocessing contract is named `libero_tgt_v1`.

Given a raw LIBERO instruction, it inserts one literal `[TGT]` marker before
the first direct object of the first recognized manipulation verb. This follows
the conservative Cosmos caption-rewrite rule.

Examples:

```text
pick up the black bowl between the plate and the ramekin
-> pick up the [TGT] black bowl between the plate and the ramekin

open the middle drawer of the cabinet
-> open the [TGT] middle drawer of the cabinet

turn on the stove and put the moka pot on it
-> turn on the [TGT] stove and put the moka pot on it
```

The transformation is idempotent: an instruction already containing `[TGT]`
is returned unchanged. Exactly one marker is permitted. An empty instruction,
multiple existing markers, or a LIBERO instruction for which no target can be
identified is an error rather than a silent fallback.

`[TGT]` remains ordinary tokenizer input text. The change does not extend or
resize Qwen or T5 vocabularies.

## Components

### Shared preprocessor

A lightweight module under `qwen3_vl_semantic_planner` owns:

- the `libero_tgt_v1` transformation;
- validation of marker count and input type;
- batch preprocessing;
- the public preprocessing-version constant.

It has no dependency on PyTorch, Transformers, the dataset implementation, or
the model. This keeps the behavior directly unit-testable and usable from both
the planner and GE-Act.

### Standalone planner

The dual-camera planner dataset continues to expose the raw task caption.
`build_dual_camera_planner_inputs` preprocesses each instruction before
formatting the shared user conversation. Consequently, the standalone
training collator and `FrozenDualCameraVLMPlanner.prepare_inputs` use the same
marked prompt without duplicating the rewrite logic.

The resulting user turn is:

```text
Main camera: <image>
Wrist camera: <image>
You are a robot video semantic planner...
Instruction: <marked LIBERO instruction>
```

The assistant turn remains the ordered semantic-plan-token sequence.

### Joint training

At the start of each joint microbatch, the trainer creates one marked-caption
list from `batch["caption"]`. That list is used for:

- `semantic_planner.prepare_inputs`, which is idempotent and validates it;
- `get_cached_text_conditions`, so the GE-Act T5 cache key and T5 input also
  contain the marker.

The raw dataset captions remain unchanged. No rewritten dataset, feature cache,
or duplicate caption store is created.

### Joint inference

The LIBERO joint evaluator applies the same batch preprocessor before invoking
the semantic planner or constructing the GE-Act text condition. Evaluation
must not contain a separate prompt rewrite implementation.

## Checkpoint Compatibility

New planner exports record:

```json
{
  "instruction_preprocessing": "libero_tgt_v1"
}
```

The new target-aware training and inference configurations require this exact
metadata value. A legacy planner checkpoint without the field, including the
existing 30k checkpoint, is rejected with a compatibility error when used by
the new configuration.

Legacy configurations remain loadable under their existing no-marker contract;
the compatibility check is selected by configuration rather than globally
invalidating old checkpoints.

Because the current 30k planner was trained without `[TGT]`, it is not used as
the frozen planner for the new GE-Act recipe. The target-aware planner is
retrained first, then frozen for GE-Act-only joint training.

## Failure Handling

- Unsupported or ambiguous LIBERO text fails before model forward.
- Marker-count violations report the original instruction.
- Training and inference verify the requested preprocessing version against
  planner metadata during initialization.
- The four-suite vocabulary audit runs before a long training launch.

## Tests

Tests must cover:

- representative `pick up`, `open`, `turn on`, `put`, and multi-stage tasks;
- the confirmed first-direct-object rule for multi-stage instructions;
- idempotence for an already marked instruction;
- rejection of zero or multiple markers after preprocessing;
- all task strings from LIBERO-10, Goal, Object, and Spatial metadata;
- identical marked prompt text in standalone training and inference builders;
- use of the same marked captions by Qwen and T5 in joint training;
- metadata save/load acceptance for `libero_tgt_v1`;
- rejection of the legacy 30k planner by the new target-aware configuration.

## Rollout

1. Implement and test the shared preprocessor and template integration.
2. Run the complete four-suite caption audit.
3. Retrain and export the dual-camera semantic planner with target-aware
   metadata.
4. Create the frozen-planner GE-Act-only joint recipe against that new export.
5. Keep the old no-marker recipe available only for legacy comparison.
