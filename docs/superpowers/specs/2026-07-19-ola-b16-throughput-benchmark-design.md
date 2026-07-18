# OLA B16 Planner Throughput Benchmark Design

## Objective

Determine whether changing the eight-H100 dual-camera K4 planner from
per-device batch 8 with accumulation 2 to per-device batch 16 with
accumulation 1 improves throughput while preserving global batch 128 and all
training semantics.

## Compared Runs

- Baseline: 8 GPUs, batch 8/GPU, accumulation 2, global batch 128. Its measured
  stable throughput is 1.08--1.10 seconds per optimizer step and its memory use
  is 38.6--40.0 GiB/GPU.
- Candidate: 8 GPUs, batch 16/GPU, accumulation 1, global batch 128. All model,
  data, optimizer, learning-rate, precision, checkpointing, and loss settings
  remain unchanged.

## Procedure and Decision Rule

Run the candidate fresh for 60 optimizer steps in a separate benchmark output
directory, with saving disabled by setting the save threshold beyond the run.
The candidate passes only if it has no OOM/non-finite loss, uses less than 75
GiB on every GPU, and its steady-state seconds per step improve by at least 10%
relative to the 1.09-second baseline. On pass, start a fresh 30k formal run with
batch 16 and accumulation 1. On failure, restore the proven batch 8 and
accumulation 2 run.

The formal launcher must continue to require all 3,424 predecoded RGB caches,
and its runtime log must report `model_gradient_checkpointing: false`.

