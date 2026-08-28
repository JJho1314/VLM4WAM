# Task 7 — Full Contract Verification Report

## Result: PASS, with two non-blocking test-coverage follow-ups

Verification was read-only on the protected HPC3 checkout. No source files, Git metadata, checkpoints, or dataset files were changed. The only local write was this requested report.

## Environment and checkout

- Remote host: HPC3
- Protected checkout: /data/user/jhe724/workspace/VLM4WAM_tgt_sdd_20260724
- Required interpreter: /data/user/jhe724/.venvs/vlm4wam_joint/bin/python
- Interpreter: Python 3.10.4
- HEAD: 5211cc34709fff18c9101b8a8086f7c829305bc3 (5211cc3)
- Initial and final remote git status --short: empty / clean.

Python and pytest commands used PYTHONDONTWRITEBYTECODE=1 and pytest used -p no:cacheprovider, preventing verification-created bytecode or pytest-cache artifacts in the target checkout.

## 1. Formatting and prompt-template checks

Command:

~~~bash
cd /data/user/jhe724/workspace/VLM4WAM_tgt_sdd_20260724
git diff --check HEAD~6..HEAD
grep -R -nE "Instruction: \\{instruction\\}" qwen3_vl_semantic_planner ge_act | sort
~~~

Results:

- git diff --check printed nothing and exited 0.
- Source-template occurrences:

~~~text
qwen3_vl_semantic_planner/ge_act_dual_camera.py:28:    "Instruction: {instruction}"
qwen3_vl_semantic_planner/train_qwen3vl4b_lingbot_dino_planner.py:1577:    "Instruction: {instruction}"
qwen3_vl_semantic_planner/train_semantic_planner.py:1736:    "Instruction: {instruction}"
~~~

- grep also reported pre-existing ignored __pycache__ binary matches; they are not source templates and Git status remained clean.
- Inspection confirms the dual-camera builder calls preprocess_libero_instructions before formatting, the collator forwards its selected preprocessing contract, and the joint/eval paths use the metadata contract. No inspected joint-T5 path restores batch["caption"] after marked captions are prepared.

## 2. Five focused test files

Command:

~~~bash
PYTHONDONTWRITEBYTECODE=1 \
  /data/user/jhe724/.venvs/vlm4wam_joint/bin/python -m pytest \
  -p no:cacheprovider -q \
  tests/test_libero_target_text.py \
  tests/test_ge_act_dual_camera_planner.py \
  tests/test_ge_act_vlm_semantic_planner.py \
  tests/test_joint_vlm_geact_training.py \
  tests/test_joint_vlm_geact_libero_eval.py
~~~

Result:

~~~text
221 passed in 18.63s
~~~

## 3. Broader GE-Act semantic-contract tests

Command:

~~~bash
PYTHONDONTWRITEBYTECODE=1 \
  /data/user/jhe724/.venvs/vlm4wam_joint/bin/python -m pytest \
  -p no:cacheprovider -q \
  tests/test_ge_act_semantic_training_contract.py \
  tests/test_ge_act_ltx_semantic_guidance.py \
  tests/test_ge_act_siglip2_config.py
~~~

Result:

~~~text
52 passed, 14 warnings in 17.11s
~~~

The warnings are installed-environment Matplotlib/Pyparsing deprecations; there were no test failures, geometry regressions, loss regressions, or semantic-injection errors.

## 4. Legacy-versus-target-aware metadata behavior

The supplied no-weights Python validation was run against /data/user/jhe724/junjie/vlm4wam_joint_assets/planner_step_030000. It reads planner_meta.json, calls validate_dual_camera_planner_metadata(metadata), then expects the same legacy metadata to fail when expected_instruction_preprocessing="libero_tgt_v1".

Command result:

~~~text
legacy accepted by legacy contract and rejected by target-aware contract
~~~

The command only read metadata and invoked the validator; it did not construct or load model weights.

## 5. Real four-suite dataset audit

Command:

~~~bash
PYTHONDONTWRITEBYTECODE=1 \
  /data/user/jhe724/.venvs/vlm4wam_joint/bin/python -m \
  qwen3_vl_semantic_planner.audit_libero_target_text \
  /data/user/jhe724/junjie/datasets/LIBERO-fastwam/libero_10_no_noops_lerobot/meta/tasks.jsonl \
  /data/user/jhe724/junjie/datasets/LIBERO-fastwam/libero_goal_no_noops_lerobot/meta/tasks.jsonl \
  /data/user/jhe724/junjie/datasets/LIBERO-fastwam/libero_object_no_noops_lerobot/meta/tasks.jsonl \
  /data/user/jhe724/junjie/datasets/LIBERO-fastwam/libero_spatial_no_noops_lerobot/meta/tasks.jsonl
~~~

The terminal JSON was not saved in the Git repository. Its observed fields were:

| Field | Result |
| --- | --- |
| instruction_preprocessing | libero_tgt_v1 |
| total_tasks | 40 |
| total_marked | 40 |
| libero_10 | 10 tasks / 10 marked |
| libero_goal | 10 tasks / 10 marked |
| libero_object | 10 tasks / 10 marked |
| libero_spatial | 10 tasks / 10 marked |

Each sampled JSON example contains exactly one [TGT] marker in the expected target-object position.

## 6. Recorded minor-gap assessment

### Default builder/collator test coverage

Focused tests explicitly cover target-aware builder behavior, legacy builder behavior, legacy collator behavior, and metadata validation. They do not directly call the builder or collator while relying on their default argument.

The implementation defaults both build_dual_camera_planner_inputs and DualCameraPlannerCollator to LIBERO_TGT_PREPROCESSING, and the collator forwards that value directly. A read-only temporary probe instantiated each with no preprocessing argument and produced:

~~~text
default builder/collator mark one [TGT]; audit boundaries reject no_paths, missing_file, empty_file, missing_task
~~~

Assessment: not functionally blocking. The actual default path was exercised and puts exactly one target marker in the rendered instruction. It is a small regression-coverage gap; add direct default-builder and default-collator tests before a later signature/default refactor.

### Audit invalid-input boundary coverage

The unit suite tests successful audit output and invalid raw text through mark_libero_target, but does not directly assert audit_task_files boundaries. The implementation explicitly rejects no paths, missing files, non-string tasks, and empty task files; malformed JSON is surfaced as json.JSONDecodeError and unmarkable text is surfaced from mark_libero_target.

Read-only temporary probes produced:

~~~text
default builder/collator mark one [TGT]; audit boundaries reject no_paths, missing_file, empty_file, missing_task
audit invalid boundaries reject malformed_json, non_string_task, unmarkable_task
~~~

Assessment: not functionally blocking. The real 40-task audit passes and the invalid boundaries fail rather than silently accepting invalid metadata. Add focused automated boundary tests as follow-up coverage.

## 7. Final Git review

Command:

~~~bash
git diff --check HEAD~6..HEAD
git status --short
git diff --stat HEAD~6..HEAD
git diff --name-only HEAD~6..HEAD
git log --oneline -6
git rev-parse HEAD
~~~

Results:

~~~text
git diff --check: no output
git status --short: no output
14 files changed, 747 insertions(+), 18 deletions(-)
5211cc34709fff18c9101b8a8086f7c829305bc3
~~~

The 14 changed paths are exactly:

~~~text
ge_act/experiments/eval_libero_joint.py
ge_act/experiments/joint_libero_eval_contract.py
ge_act/models/ltx_models/vlm_semantic_planner.py
ge_act/runner/ge_trainer.py
qwen3_vl_semantic_planner/README.md
qwen3_vl_semantic_planner/audit_libero_target_text.py
qwen3_vl_semantic_planner/ge_act_dual_camera.py
qwen3_vl_semantic_planner/train_semantic_planner.py
tests/fixtures/libero_task_texts.json
tests/test_ge_act_dual_camera_planner.py
tests/test_ge_act_vlm_semantic_planner.py
tests/test_joint_vlm_geact_libero_eval.py
tests/test_joint_vlm_geact_training.py
tests/test_libero_target_text.py
~~~

Six commits, newest first:

~~~text
5211cc3 test: audit LIBERO target text coverage
721f04e feat: use target-aware prompts in joint inference
a7ab183 fix: validate joint planner metadata before export
98fd62d feat: share target-aware prompts in joint training
26c64f6 feat: validate planner prompt preprocessing metadata
09a083c feat: version dual-camera planner instruction contract
~~~

## Launch gate

Task 7 verification clears retraining the dual-camera planner with --instruction-preprocessing libero_tgt_v1. Do not start the frozen-planner GE-Act-only run until the new planner export's planner_meta.json contains "instruction_preprocessing": "libero_tgt_v1".

