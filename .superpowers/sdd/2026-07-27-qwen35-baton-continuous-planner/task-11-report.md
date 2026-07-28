# Task 11 Report — End-to-End, Two-Rank Resume, and Legacy Gates

## Result

- Base: `78f66f78b7611379fe653f43b1ead61fdb8a41af`
- Integrated interpreter:
  `/data/LFT-W02_data/.conda/envs/ge-act/bin/python`
- No packages, environments, templates, weights, or datasets were changed or
  downloaded.
- No GPU, live Qwen3.5-2B, live SigLIP2, live LTX, real HDF5 training, or
  launcher submission was run or claimed.

The new smoke executes one real optimizer step for each curriculum stage:

- Stage 1 calls `BatonQwen35Planner`, production ownership and optimizer-group
  construction, the online frozen teacher, and the approved Stage-1 loss. Its
  output is exactly `[1,2,4,256,1024]`.
- Stage 2 calls the production teacher conditioning, semantic coordinates and
  times, GE-Act forward boundary, and three-owner GE-Act optimizer grouping.
- Stage 2 publishes a Task-10 Baton checkpoint containing model, optimizer,
  scheduler, cursor, RNG, source, provenance, snapshot topology, and file
  manifests.
- Stage 3 constructs a fresh GE-Act module, strictly loads the exact Stage-2
  safetensor bytes through `strict_load_baton_stage3_diffusion_model`, then
  takes one optimizer step through production predicted-source conditioning.
- Frozen source hashes remain unchanged; mutable parameter hashes change after
  every intended optimizer step.

The exact wrapper keeps the documented `torchrun --standalone
--nproc_per_node=2 ...` command while dispatching it through the required
GE-Act Python because the checked-in environment's `torchrun` entrypoint has a
stale StarVLA shebang. Every invocation has a UUID output directory and a
post-command result validator, so a stale result, initialization failure, or
partial-rank run exits nonzero.

## TDD Evidence

The end-to-end and source-entrypoint tests were written first. Initial RED:

```text
ModuleNotFoundError: No module named 'qwen35_baton.cli.smoke_pipeline'
1 collection error
```

The first GREEN attempt exposed and fixed a scalar tensor hashing defect. The
first real unsandboxed two-rank attempt then exposed a Task-10 integration
race: rank 1 could observe rank 0's `.step_000001.incomplete` directory between
the shared pre-check and creation. `save_baton_training_checkpoint` now makes
the shared-filesystem existence check and creation rank-zero-only before the
existing barrier. This is the only production expansion outside the Task-11
entrypoints.

## Verification

Focused end-to-end:

```text
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q \
  tests/test_qwen35_baton_end_to_end.py

4 passed, 4 warnings in 5.21s
```

Source completeness:

```text
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q \
  tests/test_ge_act_source_completeness.py

8 passed, 4 warnings in 4.71s
```

Fresh focused integration after the distributed fix:

```text
6 passed, 81 deselected, 4 warnings in 5.04s
```

Final Task-10 checkpoint plus Task-11 focused/source gate:

```text
87 passed, 4 warnings in 7.03s
```

Real two-rank CPU/Gloo wrapper, rerun outside the restricted sandbox:

```text
bash qwen35_baton/scripts/smoke_two_rank.sh

exit 0 in 17.10s
rank_agreement=true
executed_ranks=[0,1]
stage1/stage2/stage3 optimizer_steps=1/1/1
plan_shape=[1,2,4,256,1024]
envelope_loaded=true
strict_stage3_loaded=true
exact_resume=true
fresh_process_restore=true
```

The distributed result also reported:

```text
Stage-2 artifact:
441e0dbab6b3c43b19fb5b397e7fc9dcb027ce4e3f10016c48a95642c1b4ecb7
optimizer:
ab183a4b4584fa4a57b4b40026dc11a1846622d1fe8ee176a374dbaf93be6a7c
scheduler:
8898d39492c78825d9b810c1080bde10873d175d41f49595db919cc289589690
RNG:
12a99027430436c0b2d409722c53c53fd77b6240386fac7f11ac1951c75bfa50
```

Required source/legacy gate:

```text
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q \
  tests/test_qwen3vl2b_legacy_unchanged.py \
  tests/test_qwen35_grounded_config.py \
  tests/test_qwen35_sequence.py \
  tests/test_qwen35_grounded_planner.py \
  tests/test_qwen35_provider.py \
  tests/test_ge_act_source_completeness.py \
  tests/test_ge_act_ltx_semantic_guidance.py \
  tests/test_ge_act_qwen35_grounded.py \
  tests/test_ge_act_vlm_semantic_planner.py \
  tests/test_libero_fastwam_hdf5.py

448 passed, 4 warnings in 144.62s
```

Static verification returned zero:

```text
git diff --check
python -m compileall -q <all changed Python>
bash -n qwen35_baton/scripts/smoke_two_rank.sh
```

## Environment and Preflight Limits

The complete repository suite is not collectable in the required interpreter:

```text
pytest -q
PluginValidationError: unknown hook 'pytest_xdist_auto_num_workers'
in cosmos-predict2.5/conftest.py
5 warnings, 41 collection errors, exit 3
```

No xdist/plugin installation was attempted.

The checked-in fail-closed preflights were executed without changing their
templates:

- Stage 1 reports missing `/path/to/Qwen3.5-2B`.
- Stage 2 reports placeholder SigLIP2 hashes plus missing local LTX, diffusion,
  SigLIP2, HDF5, and writable output artifacts.
- Stage 3 reports its placeholder Stage-2 topology/checkpoint plus missing
  Stage-1 planner/topology, Qwen, SigLIP2, HDF5, LTX, and output artifacts.

These are artifact-readiness failures, not smoke failures, and no live-model
readiness is claimed. The GE-Act environment also emits existing xFormers
binary/version warnings; the CPU/Gloo smoke does not use those extensions.

## Self-Review

- Both ranks execute all three stages; missing rank results fail closed.
- DDP synchronizes real gradients, and hashes are compared after the
  synchronized update.
- Exact resume uses Task-10 `save_baton_training_checkpoint` and
  `load_baton_training_checkpoint`, restores a different process, and compares
  final model, optimizer, scheduler, cursor, RNG probes/state, and deterministic
  sample sequence against uninterrupted two-update execution.
- Stage 3 loads the Stage-2 snapshot through the production same-byte strict
  loader before predicted conditioning.
- Source modules remain frozen, eval-only, and outside optimizer state.
- Semantic tokens retain all 1,024 patches per camera, exact production times
  and patch centers, and no mask or relevance fields.
- The wrapper cannot turn a failed rendezvous or missing result into success.

## Formal Review Fix Round 1

Base reviewed commit: `3560e4e3646d5c5f3096410ce137fda2594cebdd`.

The checkpoint publisher now turns every rank-zero filesystem phase into a
serialized status broadcast before any later collective. This covers output
directory creation, destination/staging checks, staging creation,
`save_pretrained`, metadata and manifest generation, atomic replacement, and
cleanup. Accelerator save failures are also gathered across all ranks before
collective cleanup. Two real two-rank regressions prove that a pre-existing
destination and an injected rank-zero snapshot exception terminate promptly
with the same exception class and message on both ranks.

The synthetic `_CheckpointStateAdapter` was removed. Stage 2 now uses
`Accelerator(cpu=True)` with a prepared model, optimizer, scheduler, and
deterministically sharded dataloader. The production Task-10 save/load
functions persist and restore the real Accelerate files. Resume calls
`set_dataloader_epoch`, `skip_first_batches`, and
`advance_training_cursor`. Only outer rank zero launches the required fresh
nested command:

```text
sys.executable -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m qwen35_baton.cli.smoke_pipeline --internal-resume-worker ...
```

Both fresh child ranks restore their own saved RNG state and exact cursor,
consume their rank-specific next deterministic sample, and match the
corresponding uninterrupted rank for model, optimizer, scheduler, cursor, RNG
probes/state, and sample. The parent also requires child rank coverage
`[0,1]`, world size two, fresh PIDs, and synchronized final mutable hashes.

Stage 1, Stage 2, and Stage 3 now deliberately use different local rank
inputs. Their normal synchronized parameter hashes agree. A two-rank negative
control bypasses the Stage-1 DDP wrapper and must fail with rank-divergent
parameter hashes, proving the agreement check is sensitive to a missing
gradient collective. Invocation IDs now require the exact regex
`invocation-[0-9a-f]{32}`; wrong alphabet, case, prefix, and lengths are
rejected.

Fresh fix-round verification:

```text
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q \
  tests/test_qwen35_baton_end_to_end.py

10 passed, 4 warnings in 21.48s

/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m pytest -q \
  tests/test_ge_act_baton_training_contract.py

77 passed, 4 warnings in 25.00s

bash qwen35_baton/scripts/smoke_two_rank.sh
exit 0 in 24.00s
rank_agreement=true
executed_ranks=[0,1]
stage1/stage2/stage3 optimizer_steps=1/1/1
envelope_loaded=true
strict_stage3_loaded=true
exact_resume=true
fresh_process_restore=true
```

The end-to-end file includes the two-rank missing-gradient-collective negative
control, and the full training-contract file includes both two-rank
rank-zero-failure regressions. Static verification also returned zero:

```text
git diff --check
/data/LFT-W02_data/.conda/envs/ge-act/bin/python -m compileall -q \
  ge_act/runner/ge_trainer.py \
  qwen35_baton/cli/smoke_pipeline.py \
  tests/test_ge_act_baton_training_contract.py \
  tests/test_qwen35_baton_end_to_end.py \
  tests/helpers/baton_checkpoint_failure_worker.py
bash -n qwen35_baton/scripts/smoke_two_rank.sh
```

The previously recorded 448-test legacy gate remains the bounded legacy
evidence; it was not rerun in this fix round because the formal-review changes
are covered by the focused production checkpoint, smoke, and negative-control
gates above.
