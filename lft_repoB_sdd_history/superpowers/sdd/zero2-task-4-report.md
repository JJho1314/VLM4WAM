# Task 4 implementation report

## Result

PASS. The generic launcher now defaults to Accelerate + DeepSpeed ZeRO-2, the
POD and HPC3 files are thin machine profiles, and the direct
`USE_DEEPSPEED=0` path remains executable by the existing fake-Python tests.
The generic launcher uses strict `set -euo pipefail`, so runtime-config or
directory creation failures stop before Accelerate can launch.

## TDD evidence

### RED

Command:

```text
pytest -q tests/test_lingbot_zero2_runtime.py::test_only_canonical_launchers_are_referenced
```

Observed before launcher implementation:

```text
exit code: 1
FAILED tests/test_lingbot_zero2_runtime.py::test_only_canonical_launchers_are_referenced
FileNotFoundError: .../train_lingbot_fastwam_pod.sh
1 failed in 0.07s
```

This was the expected failure: the canonical POD profile did not yet exist.

### GREEN

The same focused command after implementation produced:

```text
exit code: 0
.                                                                        [100%]
1 passed in 0.01s
```

Shell syntax checks:

```text
bash -n scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_dino_4b.sh
bash -n scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_pod.sh
bash -n scripts/qwen3_vl_semantic_planner/lingbot_dino_4b/train_lingbot_fastwam_hpc3.sbatch
```

All three returned exit code 0 with no output.

Additional focused regression evidence:

```text
pytest -q tests/test_lingbot_zero2_runtime.py
14 passed in 0.04s

pytest -q tests/test_lingbot_dino_depth_contract.py::test_base_launcher_emits_nonempty_fastwam_cache_and_stats_overrides tests/test_lingbot_dino_depth_contract.py::test_base_launcher_omits_empty_fastwam_cache_and_stats_overrides
2 passed in 1.43s

pytest -q tests/test_lingbot_dino_depth_contract.py::test_even_future_offsets_cover_every_second_future_frame tests/test_lingbot_k1_current_future.py::test_k1_independent_modality_query_split_supports_four_64_token_groups tests/test_lingbot_k1_current_future.py::test_k1_wrapper_supports_64_tokens_per_independent_group tests/test_lingbot_k1_current_future.py::test_lingbot_four_term_loss_matches_released_weights
4 passed in 1.49s
```

### Review-fix RED/GREEN

The strict-shell review finding was covered by adding
`assert "set -euo pipefail" in generic` before changing the launcher.

RED command:

```text
pytest -q tests/test_lingbot_zero2_runtime.py::test_only_canonical_launchers_are_referenced
```

Observed against `set -uo pipefail`:

```text
exit code: 1
AssertionError: assert 'set -euo pipefail' in generic
1 failed in 0.05s
```

After the one-line production change to `set -euo pipefail`, fresh verification
produced:

```text
focused canonical test: 1 passed in 0.01s
full runtime test file: 14 passed in 0.04s
direct USE_DEEPSPEED=0 capture tests: 2 passed in 1.43s
three bash -n checks: exit code 0, no output
```

## Invariant review

- The generic launcher defaults to eight GPUs, batch size 8, accumulation 2,
  expected global batch 128, 12,000 steps, LR `3e-5`, head LR `3e-4`, and
  1,000 warmup steps. The batch identity is `8 * 8 * 2 = 128`.
- The current four independent query groups remain enabled by default:
  current DINO, future DINO, current depth, and future depth, each with 64 task
  tokens. Depth alignment and current alignment both remain enabled.
- `SEQUENCE_LENGTH=9`, `NUM_KEYFRAMES=1`, and
  `KEYFRAME_SCHEME=even_future` preserve K=1 at future offset 8. Grid size 16,
  semantic/depth dimensions 1024, and all four alignment loss weights 0.004
  remain unchanged.
- Full fine-tuning, frozen vision, plan-token embedding training, BF16, and all
  pre-existing trainer arguments remain present. The only removed trainer flag
  is the obsolete `--ddp-find-unused-parameters`; the new
  `--expected-global-batch` argument is supplied.
- Both profiles use `RUN_KIND=formal` by default. Their smoke branch defaults
  `MAX_STEPS=2` and `SAVE_STEPS=2`; their formal branch defaults
  `MAX_STEPS=12000` and `SAVE_STEPS=1000`. Explicit caller overrides are
  preserved by `${VAR:-default}` in either branch.
- The ZeRO-2 path generates a matched runtime config and launches through
  `accelerate.commands.launch`. Strict shell error handling ensures failed
  config generation or directory creation cannot fall through to an
  Accelerate launch with an empty config. `USE_DEEPSPEED=0` directly execs the
  trainer; the fake launcher environment pins batch/accumulation/expected
  global batch to 1 and passed both existing capture tests.
- The POD and HPC3 profiles validate machine paths, set their offline/cache
  environment, and invoke only the generic `train_lingbot_dino_4b.sh`.

## Concerns and limits

- Verification was intentionally CPU/static and used fake Python for the
  direct path. No real eight-GPU DeepSpeed job or site-specific filesystem was
  launched as part of this scoped task.
