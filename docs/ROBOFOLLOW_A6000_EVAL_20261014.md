# A6000 RoboFollow evaluation

This isolated evaluation deployment uses A6000-1 (`LFT_A6000_1_LFT-W02`), physical GPU 1. GPU 0 and A6000-2 were occupied. HPC3 jobs 711560 and 714381 were left untouched. Training remains on HPC3 jhe724.

## Reproduction

Repository: `/data/LFT-W02_data/junfeng/workspace/VLM4WAM_robofollow_a6000`.
Payload: `/data/LFT-W02_data/junfeng/robofollow_eval`.

The 53,000-step checkpoint, LTX weights, stats, manifest, RoboTwin assets and installed environments were copied directly from HPC3. Only paths were rewritten in `eval.yaml`. Evaluation retains action chunk 32, ten denoising steps, sixteen policy calls, a 500-action budget, disabled planner and the existing scoring implementation. All remote Python runs through `autoresearch/runtime/a6000/pixi.toml`.

The copied curobo extensions only supported sm90. Rebuild for sm86 using the Pixi `build-sim` task, CUDA 12.1 and system GCC 9; inherited Miniforge compiler flags are incompatible. The migrated Torch 2.7.0+cu128 environment passed an actual CUDA bf16 operation and full-budget model-driven episodes. No system driver or other users' environment was changed.

From the repository, on an idle GPU:

```sh
CUDA_VISIBLE_DEVICES=1 bash scripts/a6000_bench_eval.sh \
  /data/LFT-W02_data/junfeng/robofollow_eval/eval.yaml \
  /data/LFT-W02_data/junfeng/robofollow_eval/weights/diffusion_pytorch_model.safetensors \
  /data/LFT-W02_data/junfeng/robofollow_eval/NEW_OUTPUT_DIRECTORY
```

The launcher refuses an occupied GPU or an existing output directory. It checks cached/uncached action parity, then runs the same two scene1 L0 tasks, one round each, with one and two workers on the selected GPU. This is a throughput smoke benchmark, not a full RoboFollow score. Each worker has its own policy server and simulator.

## NVIDIA 535 shutdown workaround

The installed 535.261.03 driver segfaulted after evaluation completed and wrote results. A one-step HoldPolicy reproduction also failed. GDB located SIGSEGV in the `[vkrt] Analysis` thread while Python's main thread was exiting with status 0. Explicit simulator cache clearing and driver library preloads did not resolve it.

Only the A6000 launcher sets `ROBOFOLLOW_SIM_MODULE=rf_a6000_sim`. This entrypoint registers an early Python atexit handler, runs the original evaluator, and, only after a normal return and later-registered Python cleanup handlers, flushes output and exits without running the problematic native destructors. Evaluation exceptions remain failures. The benchmark still requires exit status zero and complete, valid result files. Remove this workaround after a driver upgrade is independently validated; it is not a driver repair.

Validation: two subprocess tests cover successful Python cleanup/output and propagation of evaluation errors. Together with existing benchmark completeness tests, seven tests passed on the remote environment. A real one-step HoldPolicy simulation exited zero using the entrypoint. Original failed runs and diagnostic logs were preserved separately from the new benchmark.

## Results

Benchmark output: `/data/LFT-W02_data/junfeng/robofollow_eval/bench-safe-20261014`.
Supervisor log: repository `.agent/remote-jobs/bench-safe.log`.
Both configurations completed without retries, OOM, incomplete results or failures. The supervisor exited 0.

| Workers on GPU 1 | Left task (s) | Right task (s) | Total launcher wall time (s) | Observed GPU memory |
| --- | ---: | ---: | ---: | ---: |
| 1 | 118.799 | 126.739 | 311 | 19,936 MiB |
| 2 | 160.836 | 166.823 | 258 | 39,856 MiB |

The two-worker run reduced measured total wall time by 17.0%, giving 1.21x throughput. Total time includes independent policy startup, task discovery and the runner's 60-second progress polling interval; do not interpret it as steady-state throughput or a precise large-suite speedup. Simulator subprocess times include rendering, physics, inference, IPC and startup. The two parallel tasks overlapped; summing their durations would misrepresent elapsed time.

Both runs used seeds 539 and 605. Both produced Intent 0.8, Execution 0.8 and legacy Completion 1.0 on each task. These are two selected one-round cases, not benchmark-wide metrics or success rates. Cached/uncached synthetic action parity also passed for all six predictions.

Historical HPC3 records for these two tasks took approximately 165.6 and 176.4 seconds per round (ten-round runs, duration estimated from start metadata and result modification time). The A6000 single-worker samples look promising, but the different round counts and cluster contention prevent a controlled hardware speedup claim. One A6000 with two workers is not a replacement for aggregate throughput across eight H100s.

Use two workers on the currently free A6000 GPU as the tested configuration. Three full policy replicas would likely exceed 48GB, based on the measured memory footprint; no three-worker test was attempted. A6000 is ready for additional evaluation, but this turn only ran the matched smoke benchmark, not a second full suite. HPC3's running/queued jobs remain unchanged.

Do not use the earlier `bench-20261013` runs for speed comparisons: native shutdown failures triggered retries and diagnostic simulations overlapped some attempts. The benchmark's `commit.txt` records base 69033e4; `source.sha256` records the exact tested uncommitted source files, which match the files committed with this note.
