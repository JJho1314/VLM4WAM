# Task 4 Implementation Report

Status: DONE

## Scope

- Added `prepare_joint_instruction_prompts(captions, *, preprocessing)` as the
  single joint-training preprocessing boundary.
- Shared each microbatch's marked `captions` list between Qwen planner inputs
  and cached frozen-T5 conditions.
- Applied the same preprocessing contract to T5 cache prewarm.
- Passed `instruction_preprocessing` into the metadata-selected frozen planner
  loader.
- Required the configured preprocessing contract to match source planner
  metadata, emitted it in `joint_meta.json`, and verified the standalone planner
  export retains it.
- Preserved legacy `None` behavior: captions remain unchanged and metadata
  without the field still matches a missing configuration.

## Files

- `ge_act/runner/ge_trainer.py`
- `tests/test_joint_vlm_geact_training.py`

## TDD Evidence

All tests ran remotely on HPC3 from
`/data/user/jhe724/workspace/VLM4WAM_tgt_sdd_20260724` with
`/data/user/jhe724/.venvs/vlm4wam_joint/bin/python`.

### RED

```text
/data/user/jhe724/.venvs/vlm4wam_joint/bin/python -m pytest -q \
  tests/test_joint_vlm_geact_training.py \
  -k "instruction_prompts or target_aware_text_cache or instruction_preprocessing"
```

Before production changes: `3 failed, 61 deselected in 29.76s`.
All three failures were the expected missing
`prepare_joint_instruction_prompts` contract.

### Focused GREEN

The same command after the minimal implementation produced:
`3 passed, 61 deselected in 40.11s`.

### Full Joint-Training GREEN

```text
/data/user/jhe724/.venvs/vlm4wam_joint/bin/python -m pytest -q \
  tests/test_joint_vlm_geact_training.py
```

Result: `64 passed in 430.60s (0:07:10)`.

### Diff Verification

`git diff --check` and `git diff --cached --check` both exited successfully
with no output. Only the two Task 4 files were staged and committed.

## Commit

`98fd62d937377cd7af8bf14c56849addeb8a4dae`
(`feat: share target-aware prompts in joint training`)

## Self-Review

- Confirmed non-joint training still uses raw `batch["caption"]`.
- Confirmed the later raw-caption reassignment was removed.
- Confirmed planner loading receives the configured expected preprocessing.
- Confirmed cache keys are the marked, idempotently preprocessed instructions.
- Confirmed the worktree was clean immediately after commit.

## Concerns

No known functional concerns. The full joint-training file is intentionally
slow on HPC3 (about seven minutes), but it completed without failures.

## Review Fix: Validate Before Export

Status: DONE

The review found that main rank validated the source planner preprocessing
contract only after creating the step directory and exporting LTX/planner
artifacts, while non-main ranks skipped the validation entirely.

### Changes

- Moved source `planner_meta.json` loading and
  `instruction_preprocessing` validation above the
  `accelerator.is_main_process` branch.
- Reused the validated metadata in main-process checkpoint metadata/export
  handling.
- Added a rank-parameterized mismatch regression proving both main and
  non-main ranks raise before the checkpoint step directory or artifacts are
  created.
- Added focused source-level coverage proving `unique_captions()` is
  preprocessed before `prewarm_text_condition_cache()` receives the captions.
- Updated the existing non-main save-state fixture with matching legacy-`None`
  planner metadata, since every rank now evaluates the contract.

### Review-Fix RED

```text
/data/user/jhe724/.venvs/vlm4wam_joint/bin/python -m pytest -q \
  tests/test_joint_vlm_geact_training.py \
  -k "preprocessing_mismatch or prewarm_marks"
```

After correcting a positional-argument mistake in the test stub, the valid RED
result was: `2 failed, 1 passed, 64 deselected in 6.18s`.

- Main rank raised the mismatch only after `step_20000` existed.
- Non-main rank did not raise.
- The prewarm caller coverage passed because Task 4 already implemented that
  behavior; this review added the missing regression lock.

The existing non-main save-state test then produced
`1 failed in 4.88s` because its old fixture had no planner metadata, confirming
that all-rank validation now reached that path. The fixture was updated with
matching legacy-`None` metadata.

### Review-Fix Focused GREEN

```text
/data/user/jhe724/.venvs/vlm4wam_joint/bin/python -m pytest -q \
  tests/test_joint_vlm_geact_training.py \
  -k "preprocessing_mismatch or prewarm_marks or calls_save_state_on_non_main_rank"
```

Result: `4 passed, 63 deselected in 5.79s`.

### Review-Fix Full GREEN

```text
/data/user/jhe724/.venvs/vlm4wam_joint/bin/python -m pytest -q \
  tests/test_joint_vlm_geact_training.py
```

Result: `67 passed in 25.34s`.

`git diff --check` and `git diff --cached --check` both exited successfully
with no output.

### Review-Fix Commit

`a7ab183` (`fix: validate joint planner metadata before export`)

### Review-Fix Concerns

None known.
