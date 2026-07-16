#!/usr/bin/env bash
set -u

REPO="${REPO:-/data/users/junjie/FastWAM_cosmos}"
OUT="${OUT:-$(cat /data/users/junjie/current_eval_plus_out.txt 2>/dev/null || true)}"
EXPECTED="${EXPECTED:-10030}"
INTERVAL="${INTERVAL:-300}"
MAX_CHECKS="${MAX_CHECKS:-0}"
DRY_RUN="${DRY_RUN:-0}"

OLD_SCREEN="${OLD_SCREEN:-eval_back14800_gpu012_20260701_173727}"
SUPP_SCREEN="${SUPP_SCREEN:-plus_supp_gpu012_n6_20260701_201225}"
WATCH_SCREEN="${WATCH_SCREEN:-plus_watch_step14800_20260701_191515}"

START_STEP_AFTER="${START_STEP_AFTER:-14600}"
MIN_STEP="${MIN_STEP:-11200}"
STEP_INTERVAL="${STEP_INTERVAL:-200}"
STANDARD_NPROC="${STANDARD_NPROC:-40}"
PLUS_NPROC="${PLUS_NPROC:-64}"
PLUS_LIBERO10_GPU_LIST="${PLUS_LIBERO10_GPU_LIST:-}"
PLUS_LIBERO10_NPROC="${PLUS_LIBERO10_NPROC:-24}"
STD_RETRIES="${STD_RETRIES:-6}"
PLUS_RETRIES="${PLUS_RETRIES:-6}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-1}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
STANDARD_GPU_LIST="${STANDARD_GPU_LIST:-0,1,2,3,4,5,6,7}"
PLUS_GPU_LIST="${PLUS_GPU_LIST:-0,1,2,3,4,5,6,7}"
STANDARD_SET_MUJOCO_EGL_DEVICE_ID="${STANDARD_SET_MUJOCO_EGL_DEVICE_ID:-0}"
PLUS_SET_MUJOCO_EGL_DEVICE_ID="${PLUS_SET_MUJOCO_EGL_DEVICE_ID:-0}"
PLUS_SET_LIBERO_CONFIG_PATH="${PLUS_SET_LIBERO_CONFIG_PATH:-0}"

if [ -z "$OUT" ]; then
  echo "[continue-backfill] ERROR: OUT is empty" >&2
  exit 2
fi

check_ready() {
  python3 - "$OUT" "$EXPECTED" <<'PYEOF'
import json
import os
import sys

out, expected = sys.argv[1], int(sys.argv[2])
path = os.path.join(out, "summary_aggregate.json")
if not os.path.exists(path):
    print("0 no-summary 0 0 0")
    raise SystemExit(0)
try:
    summary = json.load(open(path, encoding="utf-8"))
except Exception:
    print("0 bad-summary 0 0 0")
    raise SystemExit(0)
tasks = int(summary.get("tasks", 0))
expected_tasks = int(summary.get("expected_tasks", expected))
overall = summary.get("overall") or {}
successes = int(overall.get("successes", 0))
trials = int(overall.get("trials", 0))
complete = bool(summary.get("complete")) and tasks >= expected and expected_tasks >= expected
print(("1" if complete else "0"), "summary", tasks, successes, trials)
PYEOF
}

stop_screen() {
  local name="$1"
  [ -z "$name" ] && return 0
  if [ "$DRY_RUN" = "1" ]; then
    echo "[continue-backfill] dry-run would stop screen: $name"
    return 0
  fi
  screen -S "$name" -X quit >/dev/null 2>&1 || true
}

launch_next() {
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local screen_name="${NEXT_SCREEN:-eval_back${START_STEP_AFTER}_mixed_${ts}}"
  local log_dir="$REPO/evaluate_results/auto_eval_logs"
  local log="$log_dir/${screen_name}.log"

  if screen -ls | grep -q "[.]${screen_name}[[:space:]]"; then
    echo "[continue-backfill] next screen already exists: $screen_name"
    return 0
  fi

  local cmd
  cmd="cd $REPO && START_STEP=$START_STEP_AFTER MIN_STEP=$MIN_STEP STEP_INTERVAL=$STEP_INTERVAL STANDARD_NPROC=$STANDARD_NPROC PLUS_NPROC=$PLUS_NPROC PLUS_LIBERO10_GPU_LIST=$PLUS_LIBERO10_GPU_LIST PLUS_LIBERO10_NPROC=$PLUS_LIBERO10_NPROC STD_RETRIES=$STD_RETRIES PLUS_RETRIES=$PLUS_RETRIES LAUNCH_STAGGER_SECONDS=$LAUNCH_STAGGER_SECONDS GPU_LIST=$GPU_LIST STANDARD_GPU_LIST=$STANDARD_GPU_LIST PLUS_GPU_LIST=$PLUS_GPU_LIST STANDARD_SET_MUJOCO_EGL_DEVICE_ID=$STANDARD_SET_MUJOCO_EGL_DEVICE_ID PLUS_SET_MUJOCO_EGL_DEVICE_ID=$PLUS_SET_MUJOCO_EGL_DEVICE_ID PLUS_SET_LIBERO_CONFIG_PATH=$PLUS_SET_LIBERO_CONFIG_PATH scripts/auto_eval_joint_denoise_backfill_robust.sh > $log 2>&1"

  echo "[continue-backfill] launch screen=$screen_name log=$log"
  echo "[continue-backfill] command: $cmd"
  if [ "$DRY_RUN" = "1" ]; then
    echo "[continue-backfill] dry-run would launch next screen"
    return 0
  fi
  mkdir -p "$log_dir"
  screen -dmS "$screen_name" bash -lc "$cmd"
}

checks=0
while true; do
  read -r ready source tasks successes trials < <(check_ready)
  echo "[continue-backfill] $(date -u '+%Y-%m-%dT%H:%M:%SZ') ready=$ready source=$source tasks=$tasks successes=$successes trials=$trials out=$OUT"
  if [ "$ready" = "1" ]; then
    stop_screen "$OLD_SCREEN"
    stop_screen "$SUPP_SCREEN"
    stop_screen "$WATCH_SCREEN"
    launch_next
    echo "[continue-backfill] done"
    exit 0
  fi

  checks=$((checks + 1))
  if [ "$MAX_CHECKS" -gt 0 ] && [ "$checks" -ge "$MAX_CHECKS" ]; then
    echo "[continue-backfill] max checks reached"
    exit 0
  fi
  sleep "$INTERVAL"
done
