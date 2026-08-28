# Task 5 implementation report

Status: **PASS**

## Exact removals

Deleted these seven superseded wrappers with `apply_patch` delete operations:

1. `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_fastwam_k1.sh`
2. `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_fastwam_k1_pod30274.sh`
3. `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_current_future_fastwam_k1_hpc3.sbatch`
4. `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k4.sh`
5. `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_depth_fastwam_k4_hpc3.sbatch`
6. `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_independent_queries_fastwam_k1_pod30274.sh`
7. `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_independent_queries_fastwam_k1_hpc3.sbatch`

Removed exactly these four obsolete launcher-only test functions:

1. `tests/test_lingbot_k1_current_future.py::test_independent_query_pod_launcher_pins_fair_ablation_contract`
2. `tests/test_lingbot_k1_current_future.py::test_independent_query_hpc3_launcher_pins_64_tokens_per_feature`
3. `tests/test_lingbot_dino_depth_contract.py::test_fastwam_launcher_pins_nine_frame_dual_branch_contract`
4. `tests/test_lingbot_dino_depth_contract.py::test_hpc3_launcher_defaults_to_recommended_12k_budget`

The plan owner approved one minimal plan-oversight correction after the first stale-reference scan: the single negative launcher-literal assertion
`assert "train_lingbot_current_future_fastwam_k1.sh" not in pod + hpc` was removed from
`tests/test_lingbot_zero2_runtime.py`. The containing canonical-launcher test and all of its positive canonical contract assertions remain. Its exact pre-edit version was added to
`.superpowers/sdd/snapshots/zero2-task-5-before/tests/test_lingbot_zero2_runtime.py` before editing.

## README contract

Updated `scripts/qwen3_vl_semantic_planner/README.md` to document the canonical current contract:

- current frame 0 and future frame 8 from a 9-frame sample;
- four independent query groups × 64 = 256 VLM tokens;
- current/future DINO and current/future depth outputs, each 256 × 1024;
- Accelerate + DeepSpeed ZeRO-2;
- preferred global batch 128 from 8 GPUs × 8/GPU × accumulation 2;
- generic, POD, and HPC3 canonical launcher paths.

The obsolete 5-keyframe/`[B, 1280, 1024]` and self-contained-torchrun launcher descriptions were replaced.
After review, the smoke example was corrected to match the generic launcher's validated single-GPU
contract: `USE_DEEPSPEED=0 NUM_GPUS=1 BATCH_SIZE=1 GRAD_ACCUM=1 EXPECTED_GLOBAL_BATCH=1 MAX_STEPS=2 SAVE_STEPS=2 FULL_FINETUNE=0`.

## Test evidence

Baseline, before Task 5 edits:

```text
$ pytest -q tests/test_lingbot_k1_current_future.py tests/test_lingbot_dino_depth_contract.py
........................................................................ [ 84%]
.............                                                            [100%]
85 passed in 9.97s
```

After deleting the four named launcher-only functions:

```text
$ pytest -q tests/test_lingbot_k1_current_future.py tests/test_lingbot_dino_depth_contract.py
........................................................................ [ 88%]
.........                                                                [100%]
81 passed in 10.10s
```

Initial post-cleanup verification including the minimally corrected runtime test file:

```text
$ pytest -q tests/test_lingbot_k1_current_future.py tests/test_lingbot_dino_depth_contract.py tests/test_lingbot_zero2_runtime.py
........................................................................ [ 75%]
.......................                                                  [100%]
95 passed in 9.40s
```

The 85 → 81 count change in the two affected files is exactly the four explicitly removed test functions. All remaining behavioral tests in those files passed.

### Review follow-up TDD evidence

Added one focused regression test,
`tests/test_lingbot_zero2_runtime.py::test_readme_smoke_command_matches_single_gpu_runtime_contract`,
before changing the README.

RED, with the obsolete README still present:

```text
$ pytest -q tests/test_lingbot_zero2_runtime.py::test_readme_smoke_command_matches_single_gpu_runtime_contract
F                                                                        [100%]
=================================== FAILURES ===================================
________ test_readme_smoke_command_matches_single_gpu_runtime_contract _________
>       assert smoke_contract in readme
E       AssertionError: assert 'USE_DEEPSPEED=0 NUM_GPUS=1 BATCH_SIZE=1 GRAD_ACCUM=1 EXPECTED_GLOBAL_BATCH=1 MAX_STEPS=2 SAVE_STEPS=2 FULL_FINETUNE=0' in '# Qwen3-VL semantic planner\n\nTrains a Qwen3-VL model to act as a **semantic planner**: from the first video frame +...gbot_fastwam_hpc3.sbatch      # HPC3 profile\n    └── LINGBOT_DINO_SPEC.md                   # spec + swap plan\n```\n'
tests/test_lingbot_zero2_runtime.py:243: AssertionError
=========================== short test summary info ============================
FAILED tests/test_lingbot_zero2_runtime.py::test_readme_smoke_command_matches_single_gpu_runtime_contract
1 failed in 0.06s
```

GREEN, after only the README smoke command was corrected:

```text
$ pytest -q tests/test_lingbot_zero2_runtime.py::test_readme_smoke_command_matches_single_gpu_runtime_contract
.                                                                        [100%]
1 passed in 0.01s
```

Final fresh three-file verification:

```text
$ pytest -q tests/test_lingbot_k1_current_future.py tests/test_lingbot_dino_depth_contract.py tests/test_lingbot_zero2_runtime.py
........................................................................ [ 75%]
........................                                                 [100%]
96 passed in 10.10s
```

### EOF whitespace follow-up

Task 6's static gate exposed the final separator left behind when Task 5 removed the former EOF test.
The failing reproduction was:

```text
$ git diff --check
tests/test_lingbot_dino_depth_contract.py:1317: new blank line at EOF.
```

Exit status: `2`. The one-variable fix removed only that final empty line, leaving exactly one newline
after `assert result.returncode == 0, result.stderr`.

```text
$ git diff --check
```

Exit status: `0`. Standard output and standard error were empty.

Fresh three-file verification after the whitespace fix:

```text
$ pytest -q tests/test_lingbot_k1_current_future.py tests/test_lingbot_dino_depth_contract.py tests/test_lingbot_zero2_runtime.py
........................................................................ [ 75%]
........................                                                 [100%]
96 passed in 9.19s
```

## Launcher syntax evidence

```text
$ bash -n scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_pod.sh scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_hpc3.sbatch
```

Exit status: `0`. Standard output and standard error were empty.

## Stale-reference evidence and classification

The required command was run exactly:

```text
$ rg -n "train_lingbot_(current_future_fastwam_k1|dino_depth_fastwam_k4|independent_queries_fastwam_k1)" scripts tests docs -S
```

It returned only historical design/plan documentation:

- `docs/superpowers/plans/2026-07-13-lingbot-independent-64-query.md`: lines 126, 127, 163;
- `docs/superpowers/plans/2026-07-13-lingbot-zero2-training-cleanup.md`: lines 815, 1065–1071, 1217;
- `docs/superpowers/plans/2026-07-10-fastwam-online-dino-depth-semantic-plan.md`: lines 39, 1334, 1501, 1558, 1579, 1591, 2771, 2784;
- `docs/superpowers/specs/2026-07-13-lingbot-zero2-training-cleanup-design.md`: lines 112–118.

These are historical records and are allowed by the Task 5 brief. A focused live-scope proof was also run:

```text
$ rg -n "train_lingbot_(current_future_fastwam_k1|dino_depth_fastwam_k4|independent_queries_fastwam_k1)" scripts tests -S
```

Exit status: `1` (no matches). Standard output was empty. Therefore no stale match remains in executable scripts, current tests, or the current README.

## Scope and review artifact

The snapshot-based review artifact is `.superpowers/sdd/zero2-task-5-review.diff`. It contains diffs for exactly the README, the seven deleted wrappers, the two originally named test files, and the explicitly approved assertion cleanup plus README smoke-contract regression test in `tests/test_lingbot_zero2_runtime.py`.

No model, teacher, loss, export, provider, probe, evaluation, or visualization file was changed or deleted. No files were staged or committed, and git state was not inspected.

Concerns: none blocking. Historical documentation intentionally retains obsolete launcher names as design/implementation history.
