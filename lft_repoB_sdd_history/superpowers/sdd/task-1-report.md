# Task 1 Report: Shared `libero_tgt_v1` Preprocessor

## Status

Completed and committed in the protected target checkout.

## TDD evidence

### RED

Added `tests/test_libero_target_text.py` before production code, then ran on HPC3:

```bash
cd /data/user/jhe724/workspace/VLM4WAM_tgt_sdd_20260724
/data/user/jhe724/.venvs/vlm4wam_joint/bin/python -m pytest -q tests/test_libero_target_text.py
```

Result: collection failed as expected with:

```text
ModuleNotFoundError: No module named 'qwen3_vl_semantic_planner.libero_target_text'
```

### GREEN

Implemented the dependency-free versioned target-marker contract and reran the same focused suite on HPC3.

Result:

```text
11 passed in 0.10s
```

Also verified the module compiles with the required HPC3 interpreter.

## Files

- `qwen3_vl_semantic_planner/libero_target_text.py`
- `tests/test_libero_target_text.py`

## Commit

- `9dc4c71d3f0500825a6bc174df1a2170688d419f feat: add LIBERO target text preprocessing`

## Self-review

- The public constants, exception, and three required functions match the task interface.
- The module imports only standard-library modules and no model, dataset, PyTorch, or Transformers objects.
- Target marking is fail-closed for blank, non-string, missing-target, and multiple-marker inputs.
- Legacy (`None`) preprocessing validates string/nonblank instruction values without rewriting them.
- Focused tests cover required transformations, idempotency, invalid input, and preprocessing selection.
- The commit contains only the two Task 1 files.

## Concerns

- The remote commit emitted an existing Git LFS hook warning (`git-lfs` is absent), but the commit completed successfully and the working tree was clean.

## Review-fix follow-up

### Issue fixed

`mark_libero_target` previously continued to a later recognized verb when the first manipulation verb had an unresolved/pronominal direct object. This could rewrite the later object, such as marking `bowl` in `put it on the shelf and pick up the bowl`.

The first recognized verb now fails closed: an empty or pronominal direct object raises `InstructionPreprocessingError` with `no target object` in the message. Added `test_mark_libero_target_rejects_pronominal_first_object` for the exact reported input.

### TDD evidence

RED command on HPC3:

```bash
cd /data/user/jhe724/workspace/VLM4WAM_tgt_sdd_20260724
/data/user/jhe724/.venvs/vlm4wam_joint/bin/python -m pytest -q tests/test_libero_target_text.py
```

RED output:

```text
FAILED tests/test_libero_target_text.py::test_mark_libero_target_rejects_pronominal_first_object
Failed: DID NOT RAISE InstructionPreprocessingError
1 failed, 11 passed in 0.11s
```

GREEN command (same command) output:

```text
12 passed in 0.05s
```

### Fix commit

- `506f9123692b3a079781ca73610b3923e5f714d3 fix: reject unresolved first LIBERO target`
