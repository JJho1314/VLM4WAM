# Task 10 Report — Baton Teacher and Predicted GE-Act Curricula

## Result

- Formal-fix base: `78405451b010165eec6ad1d3c6653f30c45b434d`
- Test interpreter:
  `/data/LFT-W02_data/.conda/envs/ge-act/bin/python`
- The environment and installed packages were not changed.

The formal-review findings are implemented:

- Baton Stage 2 and Stage 3 now use a source-dispatched HDF5 preflight with
  exact schedules, objective, camera order, action geometry, batch size, and
  provenance checks; legacy and grounded branches remain separate.
- Both launchers fail closed on placeholder/zero hashes, materialize a private
  temporary YAML from explicit environment values, run both preflights against
  that resolved YAML, and delete it on exit.
- Stage 3 requires a valid final Stage-2 GE-Act checkpoint envelope. Its
  diffusion path is derived from that checkpoint and is distinct from the
  frozen Stage-1 planner checkpoint. Source, step, topology, model files, and
  HDF5/artifact provenance are checked before LTX allocation.
- Baton checkpoints atomically bind the source, runtime and serialized
  diffusion topology, exact training cursor, immutable provenance, diffusion
  files, and every Accelerate state file. Resume validates the complete
  envelope before restoring model, optimizer, scheduler, sampler/data cursor,
  and Python/NumPy/Torch/per-rank RNG state.
- Paired validation computes the source conditioning once and reuses the same
  batch, prompt inputs, action initialization, diffusion timesteps, and video
  noise for enabled and disabled modes. The only changed input is the semantic
  gate. Runtime trace equality makes this contract executable.
- Both Baton sources have execution-level forward tests for the exact
  `[B,2,4,256,1024]` condition, positions, full gate, and absent normal mask
  and relevance. Frozen teacher/planner parameters remain outside the prepared
  mutable training boundary.
- LIBERO predicted inference explicitly rejects the teacher-only training
  source.

## TDD Evidence

The formal tests were added before implementation. Representative RED results
were:

```text
HDF5 contract: 5 failed, 1 passed
ImportError: cannot import name 'load_baton_training_checkpoint'
resume_cursor was None
AttributeError: Trainer has no attribute 'validate_baton_modes'
ImportError: cannot import name 'forward_baton_ge_act'
```

Final Task 10 suite:

```text
tests/test_ge_act_baton_training_contract.py
56 passed, 4 warnings in 7.21s
```

Final required combined gate:

```text
tests/test_ge_act_baton_training_contract.py
tests/test_ge_act_baton_semantic_guidance.py
tests/test_ge_act_semantic_training_contract.py
tests/test_ge_act_siglip2_config.py

109 passed, 4 warnings in 7.84s
```

Protected and functional HDF5/preflight gate:

```text
tests/test_libero_fastwam_hdf5.py -k 'preflight or protected'
53 passed, 255 deselected in 4.60s
```

The protected assertion was changed only for the intentionally extended
`ge_act/scripts/preflight_ltx_siglip2.py`; the other three protected hashes are
unchanged.

Static checks passed:

```text
git diff --check
compileall on all changed Python paths
bash -n on all four Baton launchers
```

## Regression Evidence and Limits

The broader source/semantic/grounded/planner/teacher/provider gate completed:

```text
99 passed, 3 skipped, 1 failed, 4 warnings in 40.00s
```

The sole failure was the pre-existing two-rank Gloo execution test
`test_two_rank_ddp_joint_forward_synchronizes_provider_and_diffusion`. Both
spawned workers failed during `dist.init_process_group`, before application
code, because this restricted sandbox cannot resolve `127.0.0.1` to a local
interface:

```text
RuntimeError: Cannot resolve 127.0.0.1 to a (local) address
```

No GPU, live SigLIP2/Qwen/LTX weights, real HDF5 training, multi-process
distributed execution, or launcher submission was run or claimed. Exact resume,
paired validation, provenance rejection, source isolation, and optimizer
ownership are covered by deterministic CPU test doubles and real small
PyTorch optimizer/scheduler/RNG state.

## Self-Review

- Stage 2 constructs only `FrozenSiglip2Teacher`; Stage 3 constructs only
  `FrozenDualCameraBatonPlanner`. Both stay frozen/eval and outside optimizer,
  Accelerator/DeepSpeed preparation, and Baton checkpoint state.
- Teacher inputs use both cameras and future offsets `(0,3,5,8)`; planner inputs
  use the last current observation.
- Optimizer groups are nonempty, disjoint by identity, exhaustive over mutable
  GE-Act parameters, and preserve rates `2e-5`, `1e-4`, and `5e-5`.
- Checked-in provenance remains intentionally fail-closed; launchers require
  explicit concrete local artifacts and hashes.
- No hindsight cache, planner auxiliary loss, Qwen training group, relevance,
  ordinary-training semantic mask, or silent unconditioned deployment path is
  accepted.

## Round 2 Formal Fixes

The second formal review is fully addressed:

- Baton HDF5 window selection is now an opt-in stateless function of the
  immutable algorithm/version/seed, epoch, dataset index, and episode identity.
  It uses local Python and NumPy RNG streams and a shared epoch scalar, so worker
  assignment and prefetch timing cannot change selected windows. Wrapped and
  subset datasets receive the same epoch, and Baton training fails closed when
  a leaf dataset lacks the exact sampling contract.
- Resume requires the caller's complete immutable source-specific provenance.
  The checkpoint and runtime mappings must have exactly the same fields and
  values before `accelerator.load_state` is called. The mapping includes the
  HDF5 manifest, semantic source artifacts, planner/Qwen/template artifacts
  where applicable, and the stateless sampling contract.
- Stage 3 validates the complete trusted Stage-1 planner checkpoint and the
  final Stage-2 teacher checkpoint as one artifact chain. The HDF5 manifest and
  SigLIP2 configuration, artifact, and preprocessing hashes must match exactly
  before LTX construction.
- Stage-3 LTX is instantiated without weights, checked for exact key/shape/dtype
  topology parity with the Stage-2 snapshot, and loaded with strict state-dict
  semantics. Missing, extra, and runtime-incompatible tensors fail without
  partially mutating the runtime module.

Round-2 RED evidence was observed before each production change:

```text
stateless sampling contract: 3 failed (unknown constructor fields/missing API)
exact resume provenance: 4 failed (missing required expected-provenance API)
Stage-1/Stage-2 binding: 5 failed (missing artifact-chain validator)
strict Stage-3 loading: 4 failed (missing strict loader)
```

Final round-2 Task 10 suite:

```text
tests/test_ge_act_baton_training_contract.py
73 passed, 4 warnings
```

Final required combined gate:

```text
tests/test_ge_act_baton_training_contract.py
tests/test_ge_act_baton_semantic_guidance.py
tests/test_ge_act_semantic_training_contract.py
tests/test_ge_act_siglip2_config.py

126 passed, 4 warnings
```

Protected and functional HDF5 gate:

```text
tests/test_libero_fastwam_hdf5.py -k 'hdf5_dataset or preflight or protected'
80 passed, 228 deselected
```

The only protected hash update is again the intentional
`ge_act/scripts/preflight_ltx_siglip2.py` digest. `git diff --check`,
`compileall` on all changed Python paths, and `bash -n` on all four Baton
launchers pass.

The real resume regression uses a real PyTorch `DataLoader`, HDF5 dataset
window implementation, multiprocessing workers, prefetching, an interrupted
prefix, Accelerate batch skipping, and a different worker count after resume.
It was run outside the restricted filesystem sandbox because PyTorch worker
tensor IPC opens a local Unix-domain socket; no network service or package
change was involved.

No GPU, live SigLIP2/Qwen/LTX weights, real training run, distributed training,
or launcher submission was run or claimed in round 2.
