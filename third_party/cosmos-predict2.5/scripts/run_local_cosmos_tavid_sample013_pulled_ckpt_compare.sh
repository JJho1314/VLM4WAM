#!/usr/bin/env bash
set -euo pipefail

cd /data/LFT-W02_data/junjie/cosmos-predict2.5

COMMON_DATASET=/data/LFT-W02_data/junjie/datasets/droid_success_v21_target_aware_left_right_480x864_val
COMMON_ARGS=(
  NUM_SAMPLES=1
  SKIP_SAMPLES=13
  SAMPLE_INDEX_OFFSET=13
  STANDALONE_ONLY=1
  FPS=8
  NUM_STEPS=35
  GUIDANCE=3.0
  DROID_SUCCESS_V21_TAVID_DIR="$COMMON_DATASET"
  DROID_SUCCESS_V21_TAVID_VAL_DIR="$COMMON_DATASET"
)

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
CKPT=outputs/pulled_checkpoints/droid_success_v21_tavid_mask_tgtfix_iter000005000 \
OUT=outputs/tavid_generation_runs/v21_tavid_mask_tgtfix_iter005k_val_sample013_standalone_49f_35step \
SEED=2026052905 \
env "${COMMON_ARGS[@]}" bash scripts/run_local_cosmos_tavid_eval.sh

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
CKPT=outputs/pulled_checkpoints/droid_success_v21_tavid_mask_tgtfix_iter000018000 \
OUT=outputs/tavid_generation_runs/v21_tavid_mask_tgtfix_iter018k_val_sample013_standalone_49f_35step \
SEED=2026052918 \
env "${COMMON_ARGS[@]}" bash scripts/run_local_cosmos_tavid_eval.sh
