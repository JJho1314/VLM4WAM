# Task 4 Report: Production recipe and regression verification

## Implementation

- Pinned the LIBERO Stage 1 production data-loader recipe to eight workers, persistent workers, and synchronized recycling every 100 complete epochs.
- Extended the existing Stage 1 recipe regression assertion to protect all three worker-lifecycle requirements.

## Changed files

- `qwen35_baton/configs/libero_stage1.json`
- `tests/test_qwen35_baton_training.py`

## TDD evidence

### RED

Command: `PYTHONPATH=. pytest -q tests/test_qwen35_baton_training.py::test_stage1_recipe_requirements_and_launchers_are_fixed`

Result: failed as expected before the recipe change, with `KeyError: 'persistent_workers'` at the new assertion. Summary: `1 failed in 1.50s`.

### GREEN

The recipe now contains `"persistent_workers": true` and `"worker_restart_interval_epochs": 100` immediately after `num_workers`.

The focused command then passed: `1 passed in 1.31s`.

## Regression verification

Command: `PYTHONPATH=. pytest -q tests/test_qwen35_baton_worker_lifecycle.py tests/test_qwen35_baton_training.py tests/test_qwen35_baton_data.py tests/test_qwen35_baton_checkpoint.py`

Result: `119 passed, 4 warnings in 19.20s`. The four warnings are Python 3.12 `multiprocessing` fork deprecation warnings from `test_persistent_worker_samples_match_a_fresh_epoch_resume`; there were no test failures. A process check after the suite found no leaked `pt_data_worker` processes (only the check shell and `rg` process itself matched the text).

## Static verification

Commands: `git diff --check`; `python -m compileall -q qwen35_baton`; `git status --short`.

Results:

- `git diff --check`: no output; no whitespace errors.
- `python -m compileall -q qwen35_baton`: exit 0; compilation succeeded.
- `git status --short` before commit listed only the two intended modified tracked files and the pre-existing untracked `runtime/` directory.

## Self-review

- The regression fails if either lifecycle field is omitted or has an incorrect value, and it also protects the required eight-worker setting.
- JSON boolean spelling is valid (`true`) and both fields use the exact required production values.
- The change is intentionally limited to recipe configuration and its existing recipe contract test; no runtime artifacts are staged.

## Concerns

- None. The existing Python 3.12 fork deprecation warnings remain, but they are unrelated to this change and did not cause worker leaks or failures.
