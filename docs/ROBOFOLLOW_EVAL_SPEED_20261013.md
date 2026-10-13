# RoboFollow evaluation speed changes

The policy now caches the most recent actual instruction embedding using a bounded, per-policy LRU cache. It uses the pipeline's existing embedding arguments and preserves the original guidance setting, tokenizer length, weights, observations, action budget, and scoring. Instruction changes replace the cached entry, including shuffled or empty text. `make_policy(cache_text=False)` retains the uncached path for comparison.

`ROBOFOLLOW_PROFILE=1` logs synchronized end-to-end policy inference time (the CPU action transfer waits for CUDA) and cache hit/miss counts. The benchmark logs elapsed wall time for each simulator subprocess. That wall time includes inference, simulator setup, rendering, physics, IPC and process startup; it must not be labelled pure physics time.

`rf_bench_eval.py --sequential-modes --modes correct,shuffle` allocates all workers to correct instructions first and then completes the shuffled control. It writes a partial summary between phases. Existing completion checks, ten rounds, random seeds, and resume contract remain unchanged. `hpc3_auto_eval.sh` enables this ordering and accepts `EVAL_SLOTS_PER_GPU` (default 1), `EVAL_CONFIG`, and the existing `EVAL_RESUME_OUT`.

## Validation

- 13 tests passed on HPC3: policy contracts, cache eviction/instruction separation, sequential scheduling, readiness and resume behavior.
- `scripts/rf_profile_policy.py` compared uncached/cached actions for two instructions, three calls each, using the 53,000-step checkpoint and deterministic synthetic camera observations. All six action arrays were exactly equal.
- Initial GPU timing shared a card with the baseline and showed variable contention; it does not establish an end-to-end speedup or benchmark-wide numerical equivalence.
- Shell syntax, Python compilation, and whitespace checks passed.

## Deployment

Prepared job: `.agent/.superpowers/remote-jobs/fast-resume.sbatch`, HPC3 jhe724/acd_u, one node, eight GPUs, 96 CPUs, 1920 GB, seven days. The old evaluation job 711560 continues while this replacement queues and validates itself. Together these allocations use at most two nodes / sixteen GPUs; no new training is submitted by this change.

On allocation, the replacement checks whether the baseline has already finished, reruns GPU cache parity, and measures the same two full-budget scene1 tasks with one and two workers on one otherwise idle GPU. It chooses two workers only if elapsed time improves by more than 10%; this short test includes startup overhead and is not a full-node throughput guarantee. Any validation failure leaves the old evaluation untouched.

Only after validation succeeds does the job verify ownership/name of job 711560, cancel it, wait for its allocation to disappear, then resume the original output with the original config path. Complete tasks are reused; interrupted tasks are archived and retried. Scientific protocol and both instruction modes are preserved. Partial in-flight work may be repeated during handover.

The generated launcher and GPU timing artifacts remain outside Git. The tracked model and benchmark changes are in this isolated experiment branch, not applied to the running baseline's files.
