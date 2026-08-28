# Task 3 Report: Accelerate/DeepSpeed Trainer Migration

## Status

PASS. The Qwen3-VL LingBot-DINO planner trainer now consumes the Task 1 distributed runtime, prepares the wrapper/optimizer/DataLoader once through Accelerate, and uses update-boundary bookkeeping that distinguishes DeepSpeed from ordinary Accelerate execution.

Commits: none (approved dirty-worktree constraint).

## Scope and files

Implementation changes were limited to the three files named in the Task 3 brief:

- `tests/test_lingbot_zero2_runtime.py`
  - Added the trainer source-integration contract for Accelerate usage and absence of manual DDP/direct backward.
  - Added the independent-review regression requiring prepared legacy SigLIP keyframes to move to CPU before NumPy conversion.
- `tests/test_lingbot_dino_depth_contract.py`
  - Added the `--expected-global-batch` default/explicit-value parser contract.
  - Removed the obsolete `ddp_info` monkeypatch from the FastWAM preflight-order test.
- `scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py`
  - Replaced manual process-group/DDP/sampler setup with the Task 1 runtime and Accelerate APIs.
  - Added runtime-contract logging and rank-aware seeding.
  - Prepared the model wrapper, optimizer, and DataLoader exactly once.
  - Reworked backward, optimizer, scheduler, logging, and checkpoint boundaries for DeepSpeed and non-DeepSpeed execution.
  - Made legacy SigLIP keyframe conversion safe for device-resident batches produced by the prepared DataLoader.
  - Replaced manual teardown with `accelerator.end_training()`.

This required report and the regenerated review diff are the only additional artifacts written. Pre-existing user changes in the trainer and depth-contract test were preserved.

## TDD evidence

### RED

Tests were added before production changes, then the exact focused command from the brief was run:

```text
pytest -q tests/test_lingbot_zero2_runtime.py::test_trainer_uses_accelerate_runtime_without_manual_ddp tests/test_lingbot_dino_depth_contract.py::test_expected_global_batch_cli_defaults_to_unconstrained_and_accepts_128 tests/test_lingbot_dino_depth_contract.py::test_main_preflights_fastwam_before_loading_qwen
```

Observed result: exit 1, `2 failed, 1 passed in 1.54s`.

- `test_trainer_uses_accelerate_runtime_without_manual_ddp` failed because `accelerator.prepare(wrapper, optim, loader)` was absent from the manual-DDP trainer.
- `test_expected_global_batch_cli_defaults_to_unconstrained_and_accepts_128` failed because the parsed namespace had no `expected_global_batch` attribute.
- `test_main_preflights_fastwam_before_loading_qwen` continued to pass, proving FastWAM preflight still occurred before runtime/model initialization.

### GREEN

After the minimal trainer implementation, the same focused command was rerun.

Observed result: exit 0, `3 passed in 1.45s`.

## Final verification

The exact full test command from the brief was run:

```text
pytest -q tests/test_lingbot_zero2_runtime.py tests/test_lingbot_dino_depth_contract.py
```

Observed result after the independent-review fix: exit 0, `86 passed in 16.89s`.

The exact compile command from the brief was run:

```text
python -m py_compile scripts/qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/distributed_runtime.py
```

Observed result: exit 0 with no output.

Additional checks:

- `git diff --check` on the tracked modified trainer/test files exited 0 with no output.
- Source-symbol audit found no `DistributedDataParallel`, `DistributedSampler`, `torch.distributed`, `DDP`, `ddp_info`, direct `.backward()`, manual process-group initialization/teardown, sampler epoch branch, or `wrapper.module` access in the trainer.
- `is_main(rank)` remains only as the checkpoint helper's explicit-rank unit-test API.
- There is one `accelerator.prepare(wrapper, optim, loader)` call.
- Periodic scheduler, step counter, logging, and checkpoint logic are all gated by `is_optimizer_update(...)`.

## Independent review fix: prepared SigLIP keyframes

The independent review identified that Accelerate may place `keyframe_images` on the accelerator device after `accelerator.prepare(..., loader)`, while the legacy online SigLIP branch called `.numpy()` directly. CUDA tensors cannot be converted directly to NumPy.

The focused regression was added first and run with:

```text
pytest -q tests/test_lingbot_zero2_runtime.py::test_trainer_moves_prepared_siglip_keyframes_to_cpu_before_numpy
```

RED result: exit 1, `1 failed in 0.08s`; the expected `keyframes[i, j].detach().cpu().numpy()` conversion was absent.

The trainer's single conversion expression was then changed to detach each frame, transfer it to CPU, and convert it to NumPy. This preserves the legacy SigLIP array input while supporting both CPU and accelerator-resident prepared batches.

GREEN result from the same focused command: exit 0, `1 passed in 0.02s`.

The full Task 3 suite then passed all 86 tests, and the specified Python compile check exited 0 with no output.

## Requirement and invariant self-review

- FastWAM preflight remains before `build_accelerator` and Qwen loading.
- `--expected-global-batch` defaults to `0`, accepts `128`, and feeds runtime-contract validation; the obsolete DDP CLI option was removed.
- Runtime rank/world/device come from Accelerate, output creation and runtime logging are main-process-only, all processes synchronize before model loading, and seeds are offset by process rank.
- The DataLoader relies on Accelerate sharding (`shuffle=True`) with no manual distributed sampler; `dataset.set_epoch(step)` remains.
- Device-resident legacy SigLIP keyframes are detached and moved to CPU before NumPy conversion.
- DeepSpeed bypasses `accelerator.accumulate` through `accumulation_context`, while ordinary Accelerate uses it.
- All backward calls go through `accelerator.backward(out["loss"])`.
- Non-DeepSpeed optimizer step/zero-grad and synchronized gradient clipping remain inside the accumulation context; DeepSpeed engine stepping is left to Accelerate's DeepSpeed backward wrapper.
- Scheduler, logical step, progress/logging, running-loss reset, and save decisions happen only on actual optimizer-update boundaries.
- Periodic and final checkpoints synchronize all ranks and pass `checkpoint_module(accelerator, wrapper)` to the unchanged FastWAM export logic on the main process.
- W&B initialization/logging/finalization and parameter/runtime logging are main-process-only; progress display uses Accelerate's local-main-process flag as specified.
- Model construction, online DINO/depth teacher target generation, current/future four-loss behavior, independent 4 x 64 query geometry, and checkpoint metadata/export contents were not refactored.

## Concerns

- No live multi-GPU DeepSpeed launch was run as part of this focused task. DeepSpeed boundary/accumulation behavior is covered by the Task 1 fake-Accelerator unit tests and the trainer integration/source contracts, but an environment-level ZeRO-2 smoke run remains the next operational validation.
- The repository was already substantially dirty, including pre-existing edits in two scoped files. No Git state was mutated, and no unrelated changes were altered.
