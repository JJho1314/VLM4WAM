#!/usr/bin/env bash
# Local monitor for the remote DROID 480p training job.

set -uo pipefail
cd /data/LFT-W02_data/junjie/cosmos-predict2.5

JOB_ID=${JOB_ID:-300329}
REMOTE_REPO=/data/user/jhe724/workspace/cosmos-predict2.5
RUN_DIR="$REMOTE_REPO/outputs/droid_success_failure_base_30k_clean_val1000_scratch_480/cosmos_predict_v2p5/video2world/2b_droid_success_failure_480_bs64_bs4_accum2_10k_20260514_170558"
TRAIN_LOG="$REMOTE_REPO/outputs/manual_runs/droid480_train_bs4_10k_20260514_170558.log"
SUMMARY="$REMOTE_REPO/outputs/manual_runs/droid480_sweep_train_20260514_170558.summary"
INTERVAL_SEC=${INTERVAL_SEC:-600}

while true; do
  echo "===== $(date) ====="
  ssh -o BatchMode=yes HPC3_jhe724 "
    set +e
    echo '--- squeue ---'
    squeue -j $JOB_ID -o '%.18i %.24j %.8T %.10M %.12l %.20R'
    echo '--- sacct ---'
    sacct -j $JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed -P 2>/dev/null | tail -5
    echo '--- summary ---'
    tail -20 '$SUMMARY' 2>/dev/null
    echo '--- latest iter ---'
    grep -E 'Iteration [0-9]+:' '$RUN_DIR/console.log' 2>/dev/null | tail -8
    echo '--- recent failures ---'
    grep -E 'OutOfMemory|Traceback|FAILED|train_exit=|TRAIN_STATUS=' '$TRAIN_LOG' '$SUMMARY' 2>/dev/null | tail -20
    echo '--- checkpoints ---'
    find '$RUN_DIR/checkpoints' -maxdepth 1 -type d -name 'iter_*' 2>/dev/null | sort | tail -5
  "
  sleep "$INTERVAL_SEC"
done
