# Baton Stage-1 DataLoader Worker Recycling Design

## Context

The Qwen3.5 Baton Stage-1 trainer runs one PyTorch `DataLoader` per DDP rank.
The throughput-oriented HPC3 configuration uses eight ranks and eight
persistent workers per rank, for 64 workers in total. During the failed run,
the Slurm memory cgroup reached its 1.2 TiB hard limit. The 64 workers held
about 1.13 TiB of RSS, and one worker held about 113 GiB of anonymous RSS.
The kernel killed that worker, one DDP rank failed, and the remaining ranks
waited in distributed synchronization until teardown produced secondary
CUDA/NVLink errors.

Disabling persistent workers releases memory at every epoch, but the measured
throughput fell from about 33 to 16 samples/s because an epoch contains only
214 microbatches. Reducing the worker count to two per rank produced the same
data-loading bottleneck. The repair therefore needs to retain eight persistent
workers per rank while bounding their lifetime.

## Goals

- Keep the steady-state eight-worker-per-rank throughput.
- Bound worker RSS by terminating workers at deterministic, infrequent epoch
  boundaries.
- Preserve sampler order, cursor semantics, checkpoint/resume behavior,
  optimizer state, scheduler state, and global batch size.
- Make all ranks recycle workers at the same logical boundary.
- Fail clearly if the installed PyTorch/Accelerate versions do not expose the
  lifecycle needed to guarantee recycling.
- Record recycling events without changing the existing
  `training_metrics.jsonl` schema.

## Non-goals

- Changing HDF5 contents, preprocessing, planner architecture, loss, optimizer,
  or batch composition.
- Dynamically tuning the interval from RSS or cgroup telemetry.
- Restarting the whole training process.
- Masking a dead worker or retrying failed batches.

## Selected approach

Add a production configuration field:

```text
worker_restart_interval_epochs = 100
```

The field is either `None` or a positive integer. It is active only when
`num_workers > 0` and `persistent_workers` is true. `None` disables scheduled
recycling. Production HPC3 configurations use 100.

The interval is based on the absolute training cursor epoch, not the number of
epochs elapsed since process launch. A run resumed at epoch 1495 therefore
recycles after epoch 1499, before epoch 1500 starts. This keeps the policy
deterministic across interruption and resume.

At the end of every complete epoch:

1. Determine from the completed absolute epoch whether recycling is due.
2. Synchronize all ranks with `accelerator.wait_for_everyone()`.
3. Locate the underlying PyTorch `DataLoader` through the prepared
   Accelerate loader.
4. Shut down its current persistent worker iterator and clear the iterator
   reference, forcing a clean spawn on the next `iter(loader)`.
5. Collect each rank's success flag before any rank raises, then require all
   ranks to report the same successful result.
6. Rank zero appends a lifecycle event containing the completed epoch,
   restart count, duration, and configured interval.

No recycling occurs when the loop exits partway through an epoch because the
target step has been reached.

## Worker lifecycle adapter

PyTorch does not expose a public `DataLoader.close()` method. The lifecycle
operation will therefore be isolated in one small adapter instead of spreading
private attribute access through the training loop.

The adapter:

- unwraps known loader wrappers through `base_dataloader`;
- identifies the owned PyTorch `DataLoader`;
- checks whether a persistent iterator currently exists;
- invokes the iterator's worker shutdown operation;
- clears the loader iterator only after shutdown completes;
- returns whether active workers were actually recycled;
- raises a descriptive `RuntimeError` if recycling is configured but the
  runtime loader topology cannot provide the required lifecycle operation.

Calling the adapter with no active iterator is a valid no-op. Calling it for a
loader without persistent workers is also a no-op because no long-lived worker
state exists.

Terminating the worker processes releases their Qwen processor allocations,
HDF5 handles, HDF5 caches, and Python allocator arenas. The next epoch lazily
spawns fresh workers using the existing `spawn` multiprocessing policy, so
they do not inherit the CUDA-initialized parent.

## Reproducibility

The sampler permutation is reconstructed from `sampler_seed + epoch`.
`BatonLiberoDataset` derives sample randomness from the absolute epoch and
sample identity. Recycling therefore does not alter the selected records,
frames, camera rows, or instructions.

The training cursor remains the single source of truth:

- `cursor.epoch` selects the sampler epoch;
- `cursor.consumed_microbatches` restores a partial epoch;
- recycling only occurs after `consumed_microbatches` returns to zero at a
  complete epoch boundary.

The optimizer, scheduler, scaler, model, and RNG checkpoint payloads are not
recreated by worker recycling.

## Event logging

Worker lifecycle events go to:

```text
<output_dir>/worker_lifecycle.jsonl
```

Each rank-zero record contains:

- schema version;
- event name `dataloader_workers_restarted`;
- completed absolute epoch;
- next epoch;
- cumulative restart count;
- configured interval;
- elapsed seconds.

The existing checksummed `training_metrics.jsonl` records remain unchanged.
Lifecycle records are appended only after the initial barrier, the distributed
status collective, and a successful restart operation on every rank.

## Error handling

- Invalid intervals fail during `Stage1TrainingConfig` validation.
- Each rank catches its local lifecycle error long enough to participate in
  the distributed status collective. If any rank fails or reports a different
  recycled state, every rank raises before training continues; no rank waits
  forever at a barrier after another rank has already raised.
- An unsupported loader wrapper or missing shutdown capability fails closed
  with the loader type in the error message.
- Worker shutdown exceptions are reported after the status collective; the
  trainer does not continue with a partially recycled distributed job.
- Rank-zero lifecycle logging happens after distributed success and uses the
  same append discipline as other local JSONL artifacts.

## Testing

Tests exercise real DataLoader worker processes where lifecycle behavior is
the contract.

1. Configuration accepts `None` and positive intervals and rejects booleans,
   zero, and negative values.
2. The interval predicate fires only after the configured absolute epochs.
3. A real spawn DataLoader with persistent workers:
   - creates worker PIDs;
   - keeps the same PIDs on a non-restart epoch;
   - terminates those PIDs when recycling is due;
   - creates different PIDs on the next iterator;
   - produces the same deterministic sample order.
4. An Accelerate-prepared loader is unwrapped and recycled through the same
   adapter.
5. A fake unsupported wrapper is rejected with a descriptive error; no source
   text or mock-call assertion is used.
6. Tiny training across multiple epochs preserves expected optimizer steps and
   cursor state while producing the expected lifecycle event count.
7. Existing Baton training, checkpoint, sampler, and resume tests remain
   green.

## Operational default

For the measured 214-microbatch epochs, an interval of 100 corresponds to
roughly 1,337 optimizer steps and about 1.4 hours at the observed speed.
One worker respawn pause per interval should keep overhead near 1–2%, compared
with the roughly 50% throughput loss from restarting every epoch. The interval
is explicit in runtime configuration so it can be tightened if later memory
telemetry shows faster growth on another node.
