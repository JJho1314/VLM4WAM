#!/usr/bin/env bash
# Run TAViD-style target-mask Cosmos 2B training on 480x864 DROID data.

set -uo pipefail
cd /data/user/jhe724/workspace/cosmos-predict2.5

mkdir -p outputs/manual_runs
STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="outputs/manual_runs/droid480_tavid_sweep_train_${STAMP}.summary"

echo "START $(date)" | tee -a "$SUMMARY"
hostname | tee -a "$SUMMARY"
nvidia-smi -L | tee -a "$SUMMARY"

check_count() {
  local path=$1
  local expected=$2
  local got
  got=$(find "$path/videos" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)
  echo "$path count=$got expected=$expected" | tee -a "$SUMMARY"
  [ "$got" -eq "$expected" ]
}

check_count /data/user/jhe724/workspace/datasets/droid_success_left_train_480x864 52231 || exit 20
check_count /data/user/jhe724/workspace/datasets/droid_success_left_test_480x864 1000 || exit 21
check_count /data/user/jhe724/workspace/datasets/droid_failure_left_all_clean_480x864 13180 || exit 22

BEST_BS=
BEST_ACCUM=
for spec in 4:2 2:4 1:8; do
  BS=${spec%:*}
  ACCUM=${spec#*:}
  TEST_JOB_NAME="2b_droid_success_failure_tavid_480_bs64_sweep_bs${BS}_accum${ACCUM}_${STAMP}"
  LOG="outputs/manual_runs/droid480_tavid_batch_sweep_bs${BS}_${STAMP}.log"
  echo "TEST_BS=$BS GRAD_ACCUM=$ACCUM EFFECTIVE_GLOBAL_BS=64 LOG=$LOG $(date)" | tee -a "$SUMMARY"
  JOB_NAME="$TEST_JOB_NAME" BATCH_SIZE="$BS" GRAD_ACCUM_ITER="$ACCUM" MAX_ITER=10 \
    bash scripts/sbatch_train_droid_success_failure_tavid_480.sh > "$LOG" 2>&1
  STATUS=$?
  echo "TEST_BS=$BS GRAD_ACCUM=$ACCUM STATUS=$STATUS $(date)" | tee -a "$SUMMARY"
  if [ "$STATUS" -eq 0 ]; then
    BEST_BS=$BS
    BEST_ACCUM=$ACCUM
    break
  fi
done

if [ -z "$BEST_BS" ]; then
  echo "NO_BATCH_SIZE_PASSED" | tee -a "$SUMMARY"
  exit 30
fi

TRAIN_LOG="outputs/manual_runs/droid480_tavid_train_bs${BEST_BS}_10k_${STAMP}.log"
echo "TRAIN_BS=$BEST_BS GRAD_ACCUM=$BEST_ACCUM EFFECTIVE_GLOBAL_BS=64 LOG=$TRAIN_LOG $(date)" | tee -a "$SUMMARY"
TRAIN_JOB_NAME="2b_droid_success_failure_tavid_480_bs64_bs${BEST_BS}_accum${BEST_ACCUM}_10k_${STAMP}"
JOB_NAME="$TRAIN_JOB_NAME" BATCH_SIZE="$BEST_BS" GRAD_ACCUM_ITER="$BEST_ACCUM" MAX_ITER=10000 \
  bash scripts/sbatch_train_droid_success_failure_tavid_480.sh > "$TRAIN_LOG" 2>&1
STATUS=$?
echo "TRAIN_STATUS=$STATUS $(date)" | tee -a "$SUMMARY"
exit "$STATUS"
