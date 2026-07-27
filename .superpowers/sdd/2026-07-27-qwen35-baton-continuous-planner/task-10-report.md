# Task 10 Report — Baton Teacher and Predicted GE-Act Curricula

## Result

- Base: `4fa0ccec8c9c3aa075848e55190504ce60f7541b`
- Test interpreter:
  `/data/LFT-W02_data/.conda/envs/ge-act/bin/python`
- The environment and installed packages were not changed.

Implemented strict Stage-2 online-teacher and Stage-3 frozen-planner source
ownership, exact full-grid conditioning, detached fixed positions/times with no
relevance or normal-training mask, exact three-way GE-Act optimizer ownership,
source-labelled validation including a zero-mask `semantic_disabled` mode,
predicted Baton inference for generic/LIBERO/LIBERO-Plus paths, two recipes,
local-only provenance checks, and single-/multi-node launchers.

`FrozenSiglip2Teacher` now passes `local_files_only=True` to both Hugging Face
loaders. This is an intentional Task 10 boundary hardening: the Stage-2 online
teacher must be pinned to the Task 7 local artifact and may not silently resolve
remote state.

## TDD Evidence

Initial RED:

```text
ImportError: cannot import name 'BatonConditioningComponents'
1 error in 7.44s
```

Final required gate:

```text
tests/test_ge_act_baton_training_contract.py
tests/test_ge_act_baton_semantic_guidance.py
tests/test_ge_act_semantic_training_contract.py
tests/test_ge_act_siglip2_config.py

81 passed, 4 warnings in 7.23s
```

Before the final protected-file correction, the new Task 10 suite alone passed:

```text
28 passed, 4 warnings in 6.11s
```

Static checks passed:

```text
git diff --check
compileall on changed Python paths
bash -n on all four launchers
```

No GPU, live SigLIP2/Qwen/LTX weights, or distributed training was run or
claimed.

## Regression Evidence and Limits

The pre-edit baseline source/semantic gate passed:

```text
60 passed, 4 warnings in 5.63s
```

A combined Task 10/legacy/Task 3/8/9 run reached 73% with two failures before it
was interrupted on parent direction. A short attribution run found one failure
to be the legacy immutable-SHA assertion for
`ge_act/scripts/preflight_ltx_siglip2.py`. Task 10 explicitly extends that file,
so the old hash assertion is incompatible with the requested file modification;
the legacy code paths themselves were kept as the default branches. The
protected HDF5 preflight file was restored unchanged after this attribution.

Known deployment limitations found by self/independent review and intentionally
left for the parent formal review:

- Baton training currently saves diffusion weights only and does not restore
  optimizer/scheduler/RNG/data cursor state from `resume_from_checkpoint`; the
  recipes therefore are not yet truly resumable.
- Stage 3 defaults to the base GE-Act checkpoint rather than requiring a Stage-2
  final checkpoint.
- Checked-in provenance values are fail-closed placeholders, so a real launch
  requires materializing pinned hashes.
- The immutable legacy HDF5 preflight validates an older video-only schedule and
  rejects the new action+video curricula even though each launcher invokes it;
  a Baton-specific compatibility boundary is still needed.
- Paired semantic-enabled/disabled validation currently fetches separate random
  batches.

The LIBERO evaluator was corrected during review to reuse the generic inference
source validator, so the teacher-only training source now fails explicitly
instead of silently running unconditioned deployment inference.

## Self-Review

- Stage 2 constructs only `FrozenSiglip2Teacher`; Stage 3 constructs only
  `FrozenDualCameraBatonPlanner`, and both are frozen/eval and excluded from
  optimizer ownership.
- Teacher inputs use both cameras and future offsets `(0,3,5,8)`; planner inputs
  use the last current observation.
- Both paths validate and retain `[B,2,4,256,1024]`, exact row-major patch
  centers, and full-clip semantic times.
- Optimizer groups are nonempty, disjoint by parameter identity, exhaustive over
  trainable GE-Act parameters, and use rates `2e-5`, `1e-4`, and `5e-5`.
- Stage-3 checkpoint/topology/local-artifact checks run before LTX allocation.
- No hindsight cache, planner auxiliary loss, Qwen training group, relevance,
  or ordinary-training semantic mask is accepted.
