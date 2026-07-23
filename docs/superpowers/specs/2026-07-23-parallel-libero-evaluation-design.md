# Parallel LIBERO Evaluation Design

## Goal

Reduce the remaining wall-clock time of the step-40k joint VLM + GE-Act
LIBERO evaluation without changing model inference, rollout behavior, or metric
semantics.

## Current State

- The existing `joint_geact_eval40k` tmux session is evaluating
  `libero_spatial` on physical GPU 1.
- At design time, `libero_spatial` had completed 295 of 500 episodes.
- One evaluator uses approximately 20.5 GiB of the 49.1 GiB A6000.
- Physical GPU 0 is already running two unrelated data-collection jobs.
- The original full launcher evaluates suites in this order:
  `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`.

## Selected Design

Run one additional evaluator in a separate tmux session on physical GPU 1.
The additional evaluator starts `libero_10`, the last suite in the original
queue, while the original launcher continues `libero_spatial`,
`libero_object`, and `libero_goal`.

This suite-level split does not require changes to the evaluator or model
code. The second process uses the same configuration, checkpoint, execution
step, threshold, and 50 trials per task as the original run.

## Isolation and Outputs

- Bind the new process to physical GPU 1 with `CUDA_VISIBLE_DEVICES=1`.
- Use a dedicated tmux session and a dedicated log file.
- Keep the common results root because each suite already writes below its
  own suite directory.
- Do not run a second copy of any active or completed suite.
- Preserve the existing worktree until all evaluator processes exit because
  both processes import code from that worktree.

## Safety Checks

After launch:

1. Verify both evaluator processes are alive.
2. Verify total GPU memory remains below the device limit and no OOM appears.
3. Wait for model initialization and at least one completed `libero_10`
   episode.
4. Compare aggregate episode throughput with the single-process baseline.
5. Stop only the new worker if it causes OOM, repeated rollout errors, or a
   clear aggregate throughput regression.

Physical GPU 0 is intentionally excluded to avoid interfering with the two
existing data-collection jobs.

## Completion Behavior

The additional worker evaluates only `libero_10` and then exits. The original
launcher is expected to reach `libero_10` later. A lightweight guard process
in a separate tmux session watches only for an evaluator whose parent is the
original launcher and whose arguments select `libero_10`. If that duplicate
appears, the guard terminates that child process; the original launcher then
finishes its loop normally. The guard does not signal the parallel worker or
any earlier suite.

The parallel worker writes a completion marker only after exiting with status
zero. Results are summarized per suite from the individual inference logs,
using the timestamped directory produced by the parallel worker.
