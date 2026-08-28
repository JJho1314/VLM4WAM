# Task 7 Report: Accelerate/DeepSpeed External-Config Compatibility Fix

## Status

PASS for the scoped source correction and local verification.

The production POD has not been relaunched in this task, so end-to-end remote
confirmation remains outstanding.

## Production Failure

The POD used Accelerate `1.0.0`. Its generated Accelerate YAML contained both
an external DeepSpeed JSON reference and a top-level precision field:

```yaml
deepspeed_config:
  deepspeed_config_file: ".../deepspeed_np8_ga2_zero2.json"
mixed_precision: bf16
```

Accelerate's launch path appended `mixed_precision` to the fields exported in
`ACCELERATE_CONFIG_DS_FIELDS`. With `deepspeed_config_file` present,
`DeepSpeedPlugin._deepspeed_config_checks` rejected that field on every rank:

```text
ValueError: When using `deepspeed_config_file`, the following accelerate config
variables will be ignored: [..., 'mixed_precision'].
Please specify them appropriately in the DeepSpeed config file.
If you are using an accelerate config file, remove others config variables
mentioned in the above specified list.
```

This was a configuration-compatibility failure before training, not an OOM or
model-runtime failure.

## Root Cause and Reference Pattern

Precision ownership was duplicated across the Accelerate YAML and external
DeepSpeed JSON. The JSON already specifies:

```json
"fp16": {"enabled": false},
"bf16": {"enabled": true}
```

The StarVLA reference Accelerate files
`deepspeed_zero2.yaml` and `deepspeed_zero2_fastwam.yaml` both use
`deepspeed_config_file` without top-level `mixed_precision`; their referenced
DeepSpeed JSON files own the same BF16/FP16 decision. The compatible fix is
therefore to remove only `mixed_precision: bf16` from generated Accelerate
YAML.

Trainer runtime precision remains intentionally explicit in
`distributed_runtime.build_accelerator`, which still constructs
`Accelerator(..., mixed_precision=mixed_precision)`. DeepSpeed JSON BF16 was
also preserved.

## Strict TDD Evidence

### RED

The test first gained the assertion that a generated YAML with
`deepspeed_config_file` must omit top-level `mixed_precision`. Its existing
assertions continued to require `bf16.enabled is True` and
`fp16.enabled is False` in the DeepSpeed JSON.

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

The failure was at the new ownership assertion, after the JSON BF16/FP16 and
external-config-path assertions had passed.

### Minimal production change

Removed exactly one generated YAML list entry from
`scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/make_zero2_config.py`:

```text
mixed_precision: bf16
```

No DeepSpeed JSON setting, trainer `Accelerator` precision setting, launcher,
or unrelated source was changed.

### GREEN: focused

The identical focused command was rerun.

Exit code: `0`

```text
.                                                                        [100%]
1 passed in 0.02s
```

### GREEN: full test file

Command:

```text
pytest -q tests/test_lingbot_zero2_runtime.py
```

Exit code: `0`

```text
...............                                                          [100%]
15 passed in 0.04s
```

### Generator syntax

Command:

```text
python -m py_compile scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/make_zero2_config.py
```

Exit code: `0`; no stdout/stderr.

### Diff hygiene

Command:

```text
git diff --check
```

Exit code: `0`; no stdout/stderr.

## Generated-Config Inspection

A fresh `np8`/`ga2` generation under
`/tmp/vlm4wam-zero2-config-compat-test` exited `0`. Inspection confirmed:

- Accelerate YAML retains the absolute `deepspeed_config_file` and omits
  top-level `mixed_precision`.
- DeepSpeed JSON retains `bf16.enabled=true`, `fp16.enabled=false`,
  `gradient_accumulation_steps=2`, and ZeRO stage `2`.

Observed hashes for that output path:

```text
a9dd371caf6dad7c864d522752f5778b3b6c5e2c8e61a597d43457d5c4ea5164  deepspeed_np8_ga2_zero2.json
84a4f1341461f2812e07a8830e8a7e49134c8dcdb7c637ef587600d0535b14ad  accelerate_np8_ga2_zero2.yaml
```

## Review Artifact

`.superpowers/sdd/zero2-task-2-review.diff` was regenerated from the existing
Task 2 before snapshot plus the generator's `/dev/null` baseline. A fresh
comparison showed the artifact exactly matched the two current diffs: `8,146`
bytes on both sides.

## Concerns

- No production POD relaunch was authorized or performed, so the original
  Accelerate `1.0.0` launch path still needs one remote confirmation run.
- The repository was already intentionally dirty. No file was staged,
  committed, reset, checked out, or reverted.
