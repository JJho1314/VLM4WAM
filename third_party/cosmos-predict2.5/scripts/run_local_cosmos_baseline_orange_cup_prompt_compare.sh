#!/usr/bin/env bash
set -euo pipefail

cd /data/LFT-W02_data/junjie/cosmos-predict2.5

COMMON_ENV=(
  NUM_SAMPLES=1
  MAX_BATCHES=4
  NUM_STEPS=35
  GUIDANCE=3.0
  SEED=20260528
  FPS=8
  DROID_SUCCESS_V21_TAVID_NUM_FRAMES=49
  DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=1
  DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=range_start
)

env "${COMMON_ENV[@]}" \
  DROID_SUCCESS_V21_TAVID_DIR=outputs/tavid_generation_runs/orange_cup_baseline_prompt_blue_dataset \
  DROID_SUCCESS_V21_TAVID_VAL_DIR=outputs/tavid_generation_runs/orange_cup_baseline_prompt_blue_dataset \
  OUT=outputs/tavid_generation_runs/orange_cup_baseline_14k_prompt_blue_49f_35step \
  bash scripts/run_local_cosmos_baseline_eval.sh

env "${COMMON_ENV[@]}" \
  DROID_SUCCESS_V21_TAVID_DIR=outputs/tavid_generation_runs/orange_cup_baseline_prompt_orange_dataset \
  DROID_SUCCESS_V21_TAVID_VAL_DIR=outputs/tavid_generation_runs/orange_cup_baseline_prompt_orange_dataset \
  OUT=outputs/tavid_generation_runs/orange_cup_baseline_14k_prompt_orange_49f_35step \
  bash scripts/run_local_cosmos_baseline_eval.sh
