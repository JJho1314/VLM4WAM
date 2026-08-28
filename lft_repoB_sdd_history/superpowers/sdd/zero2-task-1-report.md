# Task 1 Report: Testable Accelerate/ZeRO-2 Runtime Contract

## Status

Implemented the prescribed runtime contract and focused boundary tests. No commit was created because the approved dirty-worktree constraint requires review through a task snapshot diff.

## Implementation

Created `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/distributed_runtime.py` with the exact interfaces required by the task brief:

- `RuntimeContract`: immutable dataclass recording the normalized distributed type, world size, per-device batch size, gradient accumulation, computed global batch size, and optional ZeRO stage.
- `build_accelerator`: lazily imports Accelerate and maps `bf16`, `fp16`, and `fp32` to the corresponding Accelerate mixed-precision mode.
- `validate_runtime_contract`: checks positive batch/accumulation values, Accelerator accumulation agreement, expected global batch size, resolved DeepSpeed configuration, DeepSpeed accumulation agreement, and ZeRO stage 2.
- `is_deepspeed`: normalizes string or enum-like distributed types before comparison.
- `accumulation_context`: bypasses Accelerate's accumulation/no-sync context for DeepSpeed and uses `accelerator.accumulate(model)` otherwise.
- `is_optimizer_update`: uses explicit microstep boundaries for DeepSpeed and `sync_gradients` for non-DeepSpeed runtimes.
- `checkpoint_module`: unwraps the model through the Accelerator before checkpoint access.

Created `tests/test_lingbot_zero2_runtime.py` with the exact fake runtime helpers and prescribed tests for the valid 8 x 8 x 2 contract, mismatch errors, global-batch rejection, DeepSpeed microstep boundaries, non-DeepSpeed accumulation behavior, and checkpoint unwrapping.

## TDD Evidence

### RED

After creating only the test file, ran exactly:

```text
pytest -q tests/test_lingbot_zero2_runtime.py
```

Result: exit code 1. Pytest reported:

```text
FFFFFFFF                                                                 [100%]
8 failed in 0.09s
```

All eight failures had the expected cause:

```text
FileNotFoundError: [Errno 2] No such file or directory: '/data/LFT-W02_data/junjie/workspace/VLM4WAM/scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/distributed_runtime.py'
```

This demonstrated that the new tests exercised the missing runtime module before production code existed.

### GREEN

After adding the minimal runtime implementation, reran exactly:

```text
pytest -q tests/test_lingbot_zero2_runtime.py
```

Result: exit code 0:

```text
........                                                                 [100%]
8 passed in 0.03s
```

## Focused Diff Verification

Ran exactly:

```text
git diff --check -- tests/test_lingbot_zero2_runtime.py scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/distributed_runtime.py
```

Result: exit code 0 with no output.

## Files Changed

- `tests/test_lingbot_zero2_runtime.py` — new focused unit tests and fake runtime helpers.
- `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/distributed_runtime.py` — new runtime contract and distributed-boundary helpers.
- `.superpowers/sdd/zero2-task-1-report.md` — this requested task report.

No other repository file was intentionally edited. Commits: none (approved dirty-worktree constraint).

## Self-Review

- Compared the created test and implementation files line by line with the task brief; the requested code and exact values are present without additional behavior.
- Confirmed the happy-path global batch calculation is `8 * 8 * 2 == 128` and the returned contract records ZeRO stage 2.
- Confirmed validation order produces the required `Accelerator`, `DeepSpeed`, `ZeRO stage`, and `global batch` error text for the prescribed cases.
- Confirmed DeepSpeed avoids `accelerator.accumulate`, while non-DeepSpeed delegates accumulation and optimizer-boundary decisions to Accelerate.
- Confirmed the Accelerate import remains inside `build_accelerator`, so the focused tests do not require importing Accelerate or initializing a distributed runtime.
- Confirmed no staging, commit, reset, checkout, or other git-state mutation was performed.

## Concerns

No blocker for Task 1. The focused suite intentionally uses fake Accelerator/DeepSpeed state and does not instantiate a real distributed process group; real-stack integration remains for the later trainer migration tasks.
