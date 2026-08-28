# Task 2 Report: Standalone Dual-Camera Planner Template and Export Metadata

## Status

Implemented in the protected checkout and committed as `09a083c feat: version
dual-camera planner instruction contract`.

## Implementation

- `ge_act_dual_camera.py`
  - Uses Task 1's `LIBERO_TGT_PREPROCESSING` and
    `preprocess_libero_instructions` in the shared dual-camera conversation
    builder.
  - Defaults the standalone builder and `DualCameraPlannerCollator` to the
    versioned target-marker contract.
  - Preserves raw dataset captions in `GEActDualCameraPlannerDataset`; callers
    can explicitly retain the legacy unmarked prompt contract with
    `instruction_preprocessing=None`.
- `train_semantic_planner.py`
  - Adds `--instruction-preprocessing`, restricted to `libero_tgt_v1` with a
    default of `None`.
  - Passes the selected mode to the GE-Act collator and records it in
    dual-camera export metadata when selected.
  - Validates both configured and saved instruction-preprocessing contracts;
    legacy metadata lacking the field remains valid only when no expected
    contract is requested.
- `tests/test_ge_act_dual_camera_planner.py`
  - Covers target insertion in the shared user turn, explicit legacy no-marker
    behavior, K4 export recording, and rejection of a missing required export
    contract.

## TDD RED / GREEN Evidence

1. RED (HPC3):
   ```text
   pytest -q tests/test_ge_act_dual_camera_planner.py -k
   "marks_instruction or legacy_no_marker or target_text_contract"
   ```
   Result: `4 failed, 84 deselected`. Each failure was the intended missing
   builder or metadata keyword interface.
2. GREEN (HPC3), after the minimal wiring:
   ```text
   pytest -q tests/test_ge_act_dual_camera_planner.py -k
   "marks_instruction or legacy_no_marker or target_text_contract"
   ```
   Result: `4 passed, 84 deselected in 18.91s`.

## Final Verification

Fresh HPC3 command using
`/data/user/jhe724/.venvs/vlm4wam_joint/bin/python`:

```text
pytest -q tests/test_ge_act_dual_camera_planner.py -k
"dual_camera and (builder or metadata or checkpoint)"
```

Result: `32 passed, 56 deselected in 20.88s`.

`git diff --check` was clean before commit.

## Files and Commit

- `qwen3_vl_semantic_planner/ge_act_dual_camera.py`
- `qwen3_vl_semantic_planner/train_semantic_planner.py`
- `tests/test_ge_act_dual_camera_planner.py`
- Commit: `09a083c feat: version dual-camera planner instruction contract`

This report is intentionally outside the protected checkout and therefore was
not included in the Task 2 code commit.

## Self-review and Concerns

- Confirmed all imports support package and script execution paths.
- Confirmed builder/collator defaults and CLI default intentionally differ:
  standalone reuse defaults to the marker contract, while training remains
  legacy-compatible unless the explicit CLI flag is selected.
- Confirmed metadata validation accepts omitted legacy fields only with no
  expected contract and rejects their use when `libero_tgt_v1` is required.
- Existing raw/incomplete test prompts explicitly request `None`; this avoids
  altering the dataset and preserves the legacy unmarked behavior.
- No open implementation concerns. A broader full-file pytest attempt emitted
  only progress dots over SSH without a terminal summary, so it is not counted
  as verification; the required focused 32-test command above is the reported
  evidence.
