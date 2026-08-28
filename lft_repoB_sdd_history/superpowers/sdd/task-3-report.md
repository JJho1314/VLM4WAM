# Task 3 Report: Metadata-Selected Standalone Planner Inference

## Status and commit

- Status: complete
- Worktree: `/data/user/jhe724/workspace/VLM4WAM_tgt_sdd_20260724`
- Commit: `26c64f6ebe34afb1c748eb2a4f621866dee0aecb`
- Subject: `feat: validate planner prompt preprocessing metadata`
- Commit scope: exactly `ge_act/models/ltx_models/vlm_semantic_planner.py` and
  `tests/test_ge_act_vlm_semantic_planner.py`.

## TDD evidence

Tests were written before the production implementation.

RED on HPC3 after adding the target-aware/legacy contract tests:

```text
/data/user/jhe724/.venvs/vlm4wam_joint/bin/python -m pytest -q \
  tests/test_ge_act_vlm_semantic_planner.py \
  -k "metadata_selected_target or target_aware_provider or legacy_provider"

1 failed, 1 passed, 22 deselected
TypeError: validate_dual_camera_planner_metadata() got an unexpected keyword
argument 'expected_instruction_preprocessing'
```

The metadata-selected builder-routing test also failed before implementation:

```text
TypeError: FrozenDualCameraVLMPlanner.from_components() got an unexpected
keyword argument 'expected_instruction_preprocessing'
```

GREEN on HPC3 after the minimal implementation:

```text
2 passed, 22 deselected
24 passed in 9.43s
```

Fresh post-commit verification on HPC3:

```text
/data/user/jhe724/.venvs/vlm4wam_joint/bin/python -m pytest -q \
  tests/test_ge_act_vlm_semantic_planner.py

24 passed in 14.34s
```

## Implementation

- `validate_dual_camera_planner_metadata` accepts and validates an optional
  `expected_instruction_preprocessing`; unsupported actual or expected values
  fail closed, and a target-aware expected contract rejects missing legacy
  metadata.
- Both standalone planner constructors accept the expected contract.
  `from_checkpoint` validates it immediately after reading metadata and before
  checking/loading Qwen checkpoint components.
- `FrozenDualCameraVLMPlanner` retains the validated metadata-selected value.
  Legacy metadata maps to `None`; `prepare_inputs` always forwards the value
  as the builder's keyword-only `instruction_preprocessing` argument.
- Tests cover metadata-selected forwarding, target-aware rejection of legacy
  metadata, and legacy unmarked compatibility. Existing fake builders now
  assert that only the supported `None` and `libero_tgt_v1` contracts arrive.

## Files

- Modified `ge_act/models/ltx_models/vlm_semantic_planner.py`
- Modified `tests/test_ge_act_vlm_semantic_planner.py`

## Self-review

- Confirmed validation happens before `_load_checkpoint_components`, so a
  target-aware expected contract cannot allocate Qwen for incompatible
  metadata.
- Confirmed metadata is the sole source of the runtime builder contract;
  callers cannot override it through `expected_instruction_preprocessing`.
- Confirmed missing metadata remains explicitly unmarked (`None`) for legacy
  checkpoints despite the dual-camera builder's target-aware default.
- Ran `git diff --check`, the focused contract tests, and the full planner
  test file. The committed diff contains only the two requested Task 3 files.

## Concerns / coverage limits

- Verification uses fake wrappers/builders rather than allocating a real Qwen
  model. The code path is covered structurally: metadata validation precedes
  the only Qwen component loader, and the full lightweight provider suite is
  green on HPC3.
