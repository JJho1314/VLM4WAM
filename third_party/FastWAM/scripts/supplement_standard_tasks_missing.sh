#!/usr/bin/env bash
set -u

REPO="${REPO:-/data/users/junjie/FastWAM_cosmos}"
VENV="${VENV:-/data/users/junjie/cosmos-predict2.5-fw/.venv}"
PY="${PY:-$VENV/bin/python}"
TRAIN_ENV="${TRAIN_ENV:-/data/users/junjie/current_train_joint_denoise.env}"
OUT="${OUT:?OUT is required}"
STEP="${STEP:?STEP is required}"
NIS="${NIS:-10}"
NPROC="${NPROC:-10}"
GPU_LIST="${GPU_LIST:-3,4,5,6,7}"
NUM_TRIALS="${NUM_TRIALS:-50}"
TASKS_PER_JOB="${TASKS_PER_JOB:-1}"
TAG_PREFIX="${TAG_PREFIX:-standard_tasks_$(date +%Y%m%d_%H%M%S)}"
SET_MUJOCO_EGL_DEVICE_ID="${SET_MUJOCO_EGL_DEVICE_ID:-0}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-1}"
DRY_RUN="${DRY_RUN:-0}"

ACTION_HIDDEN_DIM="${ACTION_HIDDEN_DIM:-1024}"
ACTION_FFN_DIM="${ACTION_FFN_DIM:-4096}"
ACTION_ATTENTION_HEAD_DIM="${ACTION_ATTENTION_HEAD_DIM:-128}"

source "$TRAIN_ENV"
cd "$REPO" || exit 2

for d in "$VENV"/lib/python3.10/site-packages/nvidia/*/lib; do
  [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done
export MAGICK_HOME="${MAGICK_HOME:-/data/users/junjie/im_env}"
export LD_LIBRARY_PATH="/data/users/junjie/im_env/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
unset LIBERO_CONFIG_PATH
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
NGPU="${#GPUS[@]}"
if [ "$NGPU" -le 0 ]; then
  echo "[standard-tasks] ERROR: empty GPU_LIST=$GPU_LIST" >&2
  exit 2
fi

mkdir -p "$OUT"
JOB_DIR="$OUT/jobs_${TAG_PREFIX}"
rm -rf "$JOB_DIR"
mkdir -p "$JOB_DIR"

"$PY" - "$OUT" "$NUM_TRIALS" "$NPROC" "$TASKS_PER_JOB" "$JOB_DIR" <<'PYEOF'
import glob
import json
import os
import sys
from collections import defaultdict

out, num_trials, nproc, tasks_per_job, job_dir = (
    sys.argv[1],
    int(sys.argv[2]),
    int(sys.argv[3]),
    max(1, int(sys.argv[4])),
    sys.argv[5],
)
suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
tasks = [(suite, task_id) for suite in suites for task_id in range(10)]
expected = {(suite, task_id, trial) for suite, task_id in tasks for trial in range(num_trials)}
seen = set()
bad = []

for path in sorted(glob.glob(os.path.join(out, "results_partial_*.json")) + glob.glob(os.path.join(out, "results_*.json"))):
    if os.path.basename(path).startswith("summary_"):
        continue
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception as exc:
        bad.append((os.path.basename(path), str(exc)))
        continue
    for row in data.get("by_task", []):
        suite = row.get("suite")
        task_id = row.get("task_id")
        if suite is None or task_id is None:
            continue
        trial_start = int(row.get("trial_start", 0) or 0)
        trials = int(row.get("trials", 0) or 0)
        for trial in range(trial_start, trial_start + trials):
            key = (suite, int(task_id), trial)
            if key in expected:
                seen.add(key)

complete_tasks = defaultdict(int)
for suite, task_id, _trial in seen:
    complete_tasks[(suite, task_id)] += 1

missing_tasks = [
    (suite, task_id)
    for suite, task_id in tasks
    if complete_tasks[(suite, task_id)] < num_trials
]

jobs = []
for start in range(0, len(missing_tasks), tasks_per_job):
    jobs.append(missing_tasks[start : start + tasks_per_job])
jobs = jobs[: max(0, min(nproc, len(jobs)))]

for idx, pairs in enumerate(jobs):
    with open(os.path.join(job_dir, f"job_{idx}.txt"), "w", encoding="utf-8") as f:
        f.write(",".join(f"{suite}:{task_id}" for suite, task_id in pairs) + "\n")

with open(os.path.join(out, "missing_standard_tasks.live.txt"), "w", encoding="utf-8") as f:
    for suite, task_id in missing_tasks:
        f.write(f"{suite}:{task_id}\n")

print(
    f"[standard-tasks] expected_units={len(expected)} seen_units={len(seen)} "
    f"missing_tasks={len(missing_tasks)} jobs={len(jobs)} "
    f"tasks_per_job={tasks_per_job} bad_files={len(bad)}"
)
PYEOF

JOB_COUNT="$(find "$JOB_DIR" -maxdepth 1 -name 'job_*.txt' | wc -l | tr -d ' ')"
if [ "$JOB_COUNT" -le 0 ]; then
  echo "[standard-tasks] nothing missing"
  exit 0
fi
if [ "$DRY_RUN" = "1" ]; then
  echo "[standard-tasks] dry run only: job_dir=$JOB_DIR"
  exit 0
fi

echo "[standard-tasks] launching $JOB_COUNT procs on gpus=$GPU_LIST out=$OUT tag=$TAG_PREFIX"
pids=()
failed=0
for idx in $(seq 0 $((JOB_COUNT - 1))); do
  pairs="$(sed -n '1p' "$JOB_DIR/job_${idx}.txt")"
  [ -z "$pairs" ] && continue
  gpu="${GPUS[$((idx % NGPU))]}"
  tag="${TAG_PREFIX}_job${idx}"
  log="$OUT/proc_${tag}_gpu${gpu}.log"
  echo "[standard-tasks] launch idx=$idx gpu=$gpu tasks=$(awk -F, '{print NF}' <<< "$pairs") log=$log"
  (
    unset LIBERO_CONFIG_PATH
    export CUDA_VISIBLE_DEVICES="$gpu"
    if [ "$SET_MUJOCO_EGL_DEVICE_ID" = "1" ]; then
      export MUJOCO_EGL_DEVICE_ID="$gpu"
    else
      unset MUJOCO_EGL_DEVICE_ID
    fi
    "$PY" experiments/libero/cosmos_eval_libero.py \
      --pairs "$pairs" --tag "$tag" --num_trials "$NUM_TRIALS" --trial_start 0 \
      --num_inference_steps "$NIS" --coupling mot --run_dir "$RUN_DIR" --step "$STEP" \
      --out_dir "$OUT" --action_hidden_dim "$ACTION_HIDDEN_DIM" \
      --action_ffn_dim "$ACTION_FFN_DIM" --action_attention_head_dim "$ACTION_ATTENTION_HEAD_DIM" \
      --no-save_videos
  ) > "$log" 2>&1 &
  pids+=("$!")
  sleep "$LAUNCH_STAGGER_SECONDS"
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=$((failed + 1))
  fi
done
echo "[standard-tasks] done failed_procs=$failed"
exit 0
