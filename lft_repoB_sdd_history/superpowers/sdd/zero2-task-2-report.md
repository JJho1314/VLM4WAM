# Task 2 Report: Matched Accelerate and DeepSpeed Configuration Generation

## Status

PASS after a production compatibility correction on 2026-07-13.

Implemented deterministic matched configuration generation for the LingBot planner ZeRO-2 migration, including the Python API, CLI, positive configuration assertions, and nonpositive runtime-value validation.

The production correction keeps precision owned by the external DeepSpeed JSON and removes the incompatible duplicate top-level `mixed_precision` field from the generated Accelerate YAML.

## Commits

None (approved dirty-worktree constraint).

No files were staged, committed, reset, checked out, or otherwise used to mutate Git state.

## Files

The compatibility correction changed only the requested test and generator behavior; the requested evidence artifacts were also refreshed:

- Modified `tests/test_lingbot_zero2_runtime.py`
  - Added JSON and YAML config loading.
  - Added dynamic loading for `make_zero2_config.py`.
  - Added the required matched-config test.
  - Added the required parameterized nonpositive-value test.
  - Added a regression assertion that an Accelerate YAML using
    `deepspeed_config_file` has no top-level `mixed_precision`; the existing
    assertions continue to require DeepSpeed BF16 enabled and FP16 disabled.
- Created `scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/make_zero2_config.py`
  - Added `make_zero2_configs(grad_accum, num_processes, output_dir)`.
  - Added deterministic filenames and serialized DeepSpeed/Accelerate contents.
  - Added validation requiring positive gradient accumulation and process count.
  - Added the required CLI and Accelerate YAML path output.
  - Removed only the generated YAML line `mixed_precision: bf16`.
- Updated `.superpowers/sdd/zero2-task-2-report.md` and regenerated
  `.superpowers/sdd/zero2-task-2-review.diff` from the existing Task 2 before
  snapshot.
- Added `.superpowers/sdd/zero2-task-7-config-fix-report.md` for the
  production-failure correction evidence.

The DeepSpeed JSON BF16/FP16 settings and trainer-side `Accelerator(...,
mixed_precision=...)` setting were intentionally not removed.

## Production Failure and Root Cause

The POD launch under Accelerate `1.0.0` failed before training. The generated
Accelerate YAML combined an external DeepSpeed config with a top-level
precision setting:

```yaml
deepspeed_config:
  deepspeed_config_file: ".../deepspeed_np8_ga2_zero2.json"
mixed_precision: bf16
```

Accelerate's launch path appended `mixed_precision` to the fields exported via
`ACCELERATE_CONFIG_DS_FIELDS`. `DeepSpeedPlugin._deepspeed_config_checks`
rejects those Accelerate-owned fields when `deepspeed_config_file` is present,
so every rank raised:

```text
ValueError: When using `deepspeed_config_file`, the following accelerate config
variables will be ignored: [..., 'mixed_precision'].
Please specify them appropriately in the DeepSpeed config file.
If you are using an accelerate config file, remove others config variables
mentioned in the above specified list.
```

The referenced DeepSpeed JSON already owns precision with
`bf16.enabled=true` and `fp16.enabled=false`. The StarVLA reference files
`deepspeed_zero2.yaml` and `deepspeed_zero2_fastwam.yaml` follow the compatible
pattern: they provide `deepspeed_config_file` and omit top-level
`mixed_precision`, while their referenced JSON files own BF16. Therefore the
single root-cause fix is to omit only the generated Accelerate YAML field.

## Compatibility Correction: Strict TDD Evidence

### RED

The regression assertion was added before changing the generator.

Command:

```text
pytest -q tests/test_lingbot_zero2_runtime.py::test_zero2_config_matches_batch_accumulation_and_process_count
```

Exit code: `1`

Exact failure evidence:

```text
>       assert "mixed_precision" not in accelerate
E       AssertionError: assert 'mixed_precision' not in {...}
FAILED tests/test_lingbot_zero2_runtime.py::test_zero2_config_matches_batch_accumulation_and_process_count
1 failed in 0.06s
```

All preceding assertions in the test passed, including the external
`deepspeed_config_file` path and DeepSpeed JSON BF16/FP16 ownership. This is
the intended RED, not a collection or syntax error.

### GREEN: focused regression

After removing only the YAML generator line, the identical command was rerun.

Exit code: `0`

Exact output:

```text
.                                                                        [100%]
1 passed in 0.02s
```

### GREEN: full runtime test file

Command:

```text
pytest -q tests/test_lingbot_zero2_runtime.py
```

Exit code: `0`

Exact output:

```text
...............                                                          [100%]
15 passed in 0.04s
```

### Syntax verification

Command:

```text
python -m py_compile scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/make_zero2_config.py
```

Exit code: `0`; no stdout/stderr.

## Original Task 2 Strict TDD Evidence

### RED

The prescribed tests were added before the production file existed.

Command:

```text
pytest -q tests/test_lingbot_zero2_runtime.py -k zero2_config
```

Exit code: `1`

Exact summary:

```text
FFF                                                                      [100%]
FAILED tests/test_lingbot_zero2_runtime.py::test_zero2_config_matches_batch_accumulation_and_process_count
FAILED tests/test_lingbot_zero2_runtime.py::test_zero2_config_rejects_nonpositive_runtime_values[0-8]
FAILED tests/test_lingbot_zero2_runtime.py::test_zero2_config_rejects_nonpositive_runtime_values[2-0]
3 failed, 8 deselected in 0.07s
```

All three failures had the intended cause:

```text
FileNotFoundError: [Errno 2] No such file or directory: '/data/LFT-W02_data/junjie/workspace/VLM4WAM/scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/make_zero2_config.py'
```

This was the expected RED: the requested feature module did not yet exist, rather than a test typo or unrelated failure.

### GREEN: focused config tests

After adding only the specified generator implementation, the same command was rerun.

Command:

```text
pytest -q tests/test_lingbot_zero2_runtime.py -k zero2_config
```

Exit code: `0`

Exact output:

```text
...                                                                      [100%]
3 passed, 8 deselected in 0.02s
```

### GREEN: CLI

Command:

```text
python scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/make_zero2_config.py --grad-accum 2 --num-processes 8 --output-dir /tmp/vlm4wam-zero2-config-test
```

Exit code: `0`

Exact output:

```text
/tmp/vlm4wam-zero2-config-test/accelerate_np8_ga2_zero2.yaml
```

### Final regression verification

Fresh full-file command:

```text
pytest -q tests/test_lingbot_zero2_runtime.py
```

Exit code: `0`

Exact output:

```text
...........                                                              [100%]
11 passed in 0.04s
```

A fresh CLI run after the full test also exited `0` and printed the same expected Accelerate YAML path.

## Generated Configuration Inspection

The generated DeepSpeed JSON was directly inspected and contains the required matched values:

- `gradient_accumulation_steps: 2`
- ZeRO stage `2`
- communication overlap enabled
- contiguous gradients enabled
- BF16 enabled and FP16 disabled
- automatic per-GPU micro-batch and global train batch sizes
- the prescribed bucket sizes, clipping, and print interval

The generated Accelerate YAML was directly inspected and contains:

- `distributed_type: DEEPSPEED`
- `num_processes: 8`
- an absolute `deepspeed_config_file` path pointing to the generated JSON
- no top-level `mixed_precision` key
- the exact prescribed local-machine, rendezvous, launcher, and CPU/network fields

Observed generated-file SHA-256 values for the corrected `np8`/`ga2`
invocation under `/tmp/vlm4wam-zero2-config-compat-test`:

```text
a9dd371caf6dad7c864d522752f5778b3b6c5e2c8e61a597d43457d5c4ea5164  deepspeed_np8_ga2_zero2.json
84a4f1341461f2812e07a8830e8a7e49134c8dcdb7c637ef587600d0535b14ad  accelerate_np8_ga2_zero2.yaml
```

## Self-Review

- Confirmed the public API is keyword-only and returns `(accelerate_path, deepspeed_path)` in the required order.
- Confirmed validation occurs before resolving or creating the output directory, so invalid runtime values do not produce configuration files.
- Confirmed `output_dir` is resolved and created recursively, making the YAML's DeepSpeed reference absolute and usable from a later launcher working directory.
- Confirmed filenames encode both process count and accumulation count as `np{num_processes}_ga{grad_accum}_zero2`.
- Confirmed JSON formatting and handwritten YAML field order/content are deterministic for identical inputs.
- Confirmed the CLI prints only the generated Accelerate configuration path to standard output and returns success.
- Confirmed test coverage uses the real generated files without mocks and covers both invalid inputs prescribed by the brief.
- Confirmed the generated YAML still references the external DeepSpeed JSON but
  no longer duplicates precision ownership at the top level.
- Confirmed the DeepSpeed JSON still enables BF16 and disables FP16.
- Confirmed trainer runtime precision remains in
  `distributed_runtime.build_accelerator` via
  `Accelerator(mixed_precision=mixed_precision)`.
- Confirmed no pre-existing unrelated working-tree edits were staged, committed, reverted, or incorporated into this task.

## Concerns

The compatibility contract is locally regression-tested, but the production POD
was not relaunched as part of this scoped source correction. The repository
remains intentionally dirty from pre-existing user work, so no commit was
created as required.
