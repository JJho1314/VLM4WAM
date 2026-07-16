#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/data/users/junjie/FastWAM_cosmos}"
TRAIN_ENV="${TRAIN_ENV:-/data/users/junjie/current_train_joint_denoise.env}"
START_STEP="${START_STEP:-14600}"
MIN_STEP="${MIN_STEP:-11200}"
STEP_INTERVAL="${STEP_INTERVAL:-200}"
STEPS="${STEPS:-}"
NIS="${NIS:-10}"
STANDARD_NPROC="${STANDARD_NPROC:-20}"
STANDARD_GPU_LIST="${STANDARD_GPU_LIST:-3,4,5,6,7}"
STANDARD_SET_MUJOCO_EGL_DEVICE_ID="${STANDARD_SET_MUJOCO_EGL_DEVICE_ID:-0}"

cd "$REPO"
source "$TRAIN_ENV"

AUTO_LOG_DIR="$REPO/evaluate_results/auto_eval_logs"
mkdir -p "$AUTO_LOG_DIR"

make_steps() {
  if [ -n "$STEPS" ]; then
    echo "$STEPS"
    return
  fi
  python3 - "$RUN_DIR/checkpoints/weights" "$START_STEP" "$MIN_STEP" "$STEP_INTERVAL" <<'PYEOF'
import os
import re
import sys

weight_dir, start, min_step, interval = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
steps = []
for name in os.listdir(weight_dir):
    m = re.fullmatch(r"step_(\d{6})\.pt", name)
    if not m:
        continue
    step = int(m.group(1))
    if min_step <= step <= start and (start - step) % interval == 0:
        steps.append(step)
print(" ".join(str(step) for step in sorted(steps, reverse=True)))
PYEOF
}

steps="$(make_steps)"
echo "[standard-backfill] run=$RUN_ID"
echo "[standard-backfill] steps=$steps"
echo "[standard-backfill] nproc=$STANDARD_NPROC gpus=$STANDARD_GPU_LIST nis=$NIS egl_device=$STANDARD_SET_MUJOCO_EGL_DEVICE_ID"

for step in $steps; do
  ts="$(date +%Y%m%d_%H%M%S)"
  log="$AUTO_LOG_DIR/standard_backfill_step${step}_${ts}.log"
  echo "[standard-backfill] starting step=$step log=$log"
  STEP="$step" NIS="$NIS" STANDARD_NPROC="$STANDARD_NPROC" STANDARD_GPU_LIST="$STANDARD_GPU_LIST" \
    STANDARD_SET_MUJOCO_EGL_DEVICE_ID="$STANDARD_SET_MUJOCO_EGL_DEVICE_ID" \
    SKIP_PLUS=1 SKIP_STANDARD_IF_EXISTS=1 \
    ACTION_HIDDEN_DIM="${ACTION_HIDDEN_DIM:-1024}" \
    ACTION_FFN_DIM="${ACTION_FFN_DIM:-4096}" \
    ACTION_ATTENTION_HEAD_DIM="${ACTION_ATTENTION_HEAD_DIM:-128}" \
    bash scripts/auto_eval_joint_denoise_after_train.sh > "$log" 2>&1
  echo "[standard-backfill] finished step=$step"
done

echo "[standard-backfill] all standard evals complete"
