# Task 5 Implementation Report

Status: DONE

## Scope

- Added the import-safe
  `prepare_joint_inference_prompt(prompt: str) -> str` helper to the joint
  evaluator contract.
- Applied `libero_tgt_v1` exactly once at the start of each joint rollout
  `play` call.
- Passed the same `marked_prompt` value to both online semantic planning and
  the base GE-Act/T5 evaluator.
- Required both `joint_meta.json` and `planner/planner_meta.json` to declare
  `instruction_preprocessing: libero_tgt_v1`.
- Required the frozen planner loader to receive the same expected
  preprocessing contract.
- Kept the base `eval_libero.py` evaluator unchanged.

## Files

- `ge_act/experiments/eval_libero_joint.py`
- `ge_act/experiments/joint_libero_eval_contract.py`
- `tests/test_joint_vlm_geact_libero_eval.py`

## TDD Evidence

All tests ran remotely on HPC3 from
`/data/user/jhe724/workspace/VLM4WAM_tgt_sdd_20260724` with
`/data/user/jhe724/.venvs/vlm4wam_joint/bin/python`.

### RED

The first focused run after adding the tests failed during collection because
the new pure helper did not exist:

```text
/data/user/jhe724/.venvs/vlm4wam_joint/bin/python -m pytest -q \
  tests/test_joint_vlm_geact_libero_eval.py \
  -k "prompt or legacy_prompt_contract"
```

Result: one collection error:
`ImportError: cannot import name 'prepare_joint_inference_prompt'`.

After adding only the pure helper, the focused run exposed the intended
metadata failure:

```text
2 failed, 2 passed, 24 deselected in 2.20s
```

Both the legacy joint metadata and legacy planner metadata were accepted when
their `instruction_preprocessing` fields were removed.

### Focused GREEN

The same focused command after the minimal implementation produced:

```text
4 passed, 24 deselected in 2.45s
```

### Full Joint-Evaluator GREEN

```text
/data/user/jhe724/.venvs/vlm4wam_joint/bin/python -m pytest -q \
  tests/test_joint_vlm_geact_libero_eval.py
```

Result: `28 passed in 1.56s`.

### Relevant Regression Verification

```text
/data/user/jhe724/.venvs/vlm4wam_joint/bin/python -m pytest -q \
  tests/test_joint_vlm_geact_libero_eval.py \
  tests/test_libero_target_text.py \
  tests/test_ge_act_vlm_semantic_planner.py
```

Result: `64 passed in 2.36s`.

Both modified evaluator modules also passed `python -m py_compile`.
`git diff --check` and `git diff --cached --check` exited successfully.

## Commit

`721f04ee22a96ee806acba13c8c90179102791bf`
(`feat: use target-aware prompts in joint inference`)

Only the three Task 5 files listed above were staged and committed.

## Self-Review

- Confirmed `marked_prompt` is constructed once per `play` call.
- Confirmed the identical local variable is passed to
  `build_joint_semantic_condition` and `super().play`.
- Confirmed `joint_meta.json` missing or mismatching the preprocessing field
  fails closed through the exact equality contract.
- Confirmed planner metadata missing or mismatching the field fails closed
  through `expected_instruction_preprocessing=LIBERO_TGT_PREPROCESSING`.
- Confirmed planner loading independently enforces the same expected metadata.
- Added an explicit regression for legacy planner metadata in addition to the
  requested legacy joint metadata regression.
- Confirmed `ge_act/experiments/eval_libero.py` has no diff.
- Confirmed the target worktree was clean immediately after commit.

## Concerns

No known functional concerns. The source-level assertions intentionally lock
the single-variable prompt-sharing structure because importing the full LIBERO
rollout evaluator would require the heavyweight runtime.
