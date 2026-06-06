#!/usr/bin/env bash
set -euo pipefail

cd /data/LFT-W02_data/junjie/cosmos-predict2.5

BASE_OUT=${BASE_OUT:-outputs/tavid_generation_runs/v21_baseline_14k_val_parallel_scenes_012_013_49f_35step}
DATASET=${DATASET:-/data/LFT-W02_data/junjie/datasets/droid_success_v21_target_aware_left_right_480x864_val}
mkdir -p "$BASE_OUT/logs"

run_one() {
  local gpu="$1"
  local sample_idx="$2"
  local out_dir="$BASE_OUT/sample_${sample_idx}_gpu${gpu}"

  CUDA_VISIBLE_DEVICES="$gpu" \
  NUM_SAMPLES=1 \
  SKIP_SAMPLES="$sample_idx" \
  SAMPLE_INDEX_OFFSET="$sample_idx" \
  MAX_BATCHES=160 \
  STANDALONE_ONLY=1 \
  OUT="$out_dir" \
  DROID_SUCCESS_V21_TAVID_DIR="$DATASET" \
  DROID_SUCCESS_V21_TAVID_VAL_DIR="$DATASET" \
  bash scripts/run_local_cosmos_baseline_eval.sh
}

run_one 0 12 >"$BASE_OUT/logs/sample_012_gpu0.log" 2>&1 &
pid0=$!

run_one 1 13 >"$BASE_OUT/logs/sample_013_gpu1.log" 2>&1 &
pid1=$!

wait "$pid0"
wait "$pid1"

echo "done: $BASE_OUT"
