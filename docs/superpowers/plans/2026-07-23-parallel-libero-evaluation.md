# Parallel LIBERO Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start and validate a second step-40k joint VLM + GE-Act evaluator so `libero_10` runs concurrently with the existing full evaluation.

**Architecture:** Keep the current full evaluator unchanged and launch a suite-specific worker in its own tmux session on physical GPU 1. A second lightweight tmux guard prevents the original sequential launcher from later duplicating `libero_10`.

**Tech Stack:** Bash, tmux, NVIDIA CUDA, Python, LIBERO, GE-Act

## Global Constraints

- Do not change model inference, rollout behavior, or metric semantics.
- Use the same config, checkpoint, execution step, threshold, and 50 trials per task.
- Do not use physical GPU 0 because two unrelated data-collection jobs are active there.
- Preserve the evaluation worktree until every evaluator exits.
- Write the worker log and completion marker beneath the existing results root.

---

### Task 1: Launch and validate the parallel evaluator

**Files:**
- Read: `ge_act/experiments/eval_libero_joint.py`
- Read: `ge_act/configs/ltx_model/libero/action_model_libero_joint_step40000_eval.yaml`
- Create at runtime: `/data/LFT-W02_data/junjie/eval_results/joint_vlm_geact_action_k4_step40000/libero_10_parallel.log`
- Create on successful completion: `/data/LFT-W02_data/junjie/eval_results/joint_vlm_geact_action_k4_step40000/libero_10_parallel.complete`

**Interfaces:**
- Consumes: the project-local step-40k checkpoint, joint evaluator, and original launcher PID `2743988`
- Produces: one isolated `libero_10` evaluation process plus a duplicate-prevention guard

- [ ] **Step 1: Recheck the active process and GPU memory**

Run:

```bash
tmux list-panes -t joint_geact_eval40k \
  -F 'pane_pid=#{pane_pid} pane_dead=#{pane_dead} command=#{pane_current_command}'
pgrep -af 'eval_libero_joint.py'
nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader
```

Expected: original pane PID `2743988` is alive, one evaluator is selecting
`libero_spatial`, and physical GPU 1 has at least 24 GiB free.

- [ ] **Step 2: Start the duplicate-prevention guard**

Run a detached tmux session named `joint_geact_eval40k_guard`. It polls only
children of original launcher PID `2743988`; when such a child selects
`libero_10`, it sends `TERM` to that duplicate and exits.

Run:

```bash
tmux new-session -d -s joint_geact_eval40k_guard "bash -lc '
ORIGINAL_PARENT=2743988
GUARD_LOG=/data/LFT-W02_data/junjie/eval_results/joint_vlm_geact_action_k4_step40000/libero_10_duplicate_guard.log
while kill -0 \"\$ORIGINAL_PARENT\" 2>/dev/null; do
  while read -r pid; do
    [[ -n \"\$pid\" ]] || continue
    cmd=\$(tr \"\\0\" \" \" <\"/proc/\$pid/cmdline\")
    if [[ \"\$cmd\" == *\"--task_suite_name libero_10\"* ]]; then
      kill -TERM \"\$pid\"
      date \"+%F %T terminated original duplicate pid=\$pid\" >\"\$GUARD_LOG\"
      exit 0
    fi
  done < <(pgrep -P \"\$ORIGINAL_PARENT\" -f \"eval_libero_joint.py\" || true)
  sleep 2
done
date \"+%F %T original launcher exited before duplicate\" >\"\$GUARD_LOG\"
'"
```

Expected: `tmux has-session -t joint_geact_eval40k_guard` exits zero.

- [ ] **Step 3: Start the parallel `libero_10` worker**

Use:

```bash
CUDA_VISIBLE_DEVICES=1
PYTHONPATH=/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/.worktrees/joint-vlm-geact-libero-eval:/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/.worktrees/joint-vlm-geact-libero-eval/ge_act:/data/LFT-W02_data/junjie/VLA_RL/docker_libero/LIBERO
MUJOCO_GL=egl
PYTHONUNBUFFERED=1
```

Run `/data/LFT-W02_data/.conda/envs/ge-act/bin/python` with:

```text
ge_act/experiments/eval_libero_joint.py
--config_file ge_act/configs/ltx_model/libero/action_model_libero_joint_step40000_eval.yaml
--joint_ckpt_dir /data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/checkpoints/joint_vlm_geact_action_k4_50k/step_40000
--output_dir /data/LFT-W02_data/junjie/eval_results/joint_vlm_geact_action_k4_step40000
--device 0
--exec_step 8
--threshold 20
--task_suite_name libero_10
--num_trails_per_task 50
```

Pipe combined output through `tee` to `libero_10_parallel.log`. Record the
Python exit status using `PIPESTATUS[0]`; touch `libero_10_parallel.complete`
only when that status is zero.

Run:

```bash
tmux new-session -d -s joint_geact_eval40k_libero10 "bash -lc '
set -o pipefail
ROOT=/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/.worktrees/joint-vlm-geact-libero-eval
PY=/data/LFT-W02_data/.conda/envs/ge-act/bin/python
LIBERO_ROOT=/data/LFT-W02_data/junjie/VLA_RL/docker_libero/LIBERO
OUTPUT=/data/LFT-W02_data/junjie/eval_results/joint_vlm_geact_action_k4_step40000
LOG=\$OUTPUT/libero_10_parallel.log
MARKER=\$OUTPUT/libero_10_parallel.complete
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=\"\$ROOT:\$ROOT/ge_act:\$LIBERO_ROOT\${PYTHONPATH:+:\$PYTHONPATH}\"
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
rm -f \"\$MARKER\"
cd \"\$ROOT/ge_act\"
\"\$PY\" \"\$ROOT/ge_act/experiments/eval_libero_joint.py\" \
  --config_file \"\$ROOT/ge_act/configs/ltx_model/libero/action_model_libero_joint_step40000_eval.yaml\" \
  --joint_ckpt_dir /data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/checkpoints/joint_vlm_geact_action_k4_50k/step_40000 \
  --output_dir \"\$OUTPUT\" \
  --device 0 \
  --exec_step 8 \
  --threshold 20 \
  --task_suite_name libero_10 \
  --num_trails_per_task 50 \
  2>&1 | tee \"\$LOG\"
status=\${PIPESTATUS[0]}
if [[ \"\$status\" -eq 0 ]]; then
  touch \"\$MARKER\"
fi
exit \"\$status\"
'"
```

Expected: tmux session `joint_geact_eval40k_libero10` exists and its child
process command selects `libero_10`.

- [ ] **Step 4: Validate initialization and resource safety**

Run:

```bash
tmux list-sessions
pgrep -af 'eval_libero_joint.py'
nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader
tail -80 /data/LFT-W02_data/junjie/eval_results/joint_vlm_geact_action_k4_step40000/libero_10_parallel.log
```

Expected: both evaluators remain alive, physical GPU 1 memory stays below
49.1 GiB, and the new log contains no CUDA OOM or traceback.

- [ ] **Step 5: Verify the first completed parallel episode**

Run:

```bash
rg -n 'Success:|episodes completed so far|Traceback|CUDA out of memory' \
  /data/LFT-W02_data/junjie/eval_results/joint_vlm_geact_action_k4_step40000/libero_10_parallel.log |
  tail -20
```

Expected: at least one `Success: True` or `Success: False` line and
`# episodes completed so far: 1`, with no traceback or CUDA OOM.

- [ ] **Step 6: Commit the operational plan**

Run:

```bash
git add \
  docs/superpowers/specs/2026-07-23-parallel-libero-evaluation-design.md \
  docs/superpowers/plans/2026-07-23-parallel-libero-evaluation.md
git commit -m "docs(eval): plan parallel LIBERO execution"
```

Expected: one commit containing only the corrected design and implementation
plan; runtime logs and completion markers remain untracked outside the repo.
