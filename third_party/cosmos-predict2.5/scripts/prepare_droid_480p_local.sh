#!/usr/bin/env bash
# Build 480x864 Cosmos VideoDataset directories from local LeRobot DROID v3 data.

set -uo pipefail
cd /data/LFT-W02_data/junjie/cosmos-predict2.5

PY=${PY:-python}
WORKERS=${WORKERS:-96}
THREADS=${THREADS:-1}

RAW_SUCCESS=/data/LFT-W02_data/junjie/data/droid_success
RAW_FAILURE=/data/LFT-W02_data/junjie/data/droid_failure
OUT_ROOT=/data/LFT-W02_data/junjie/datasets

SUCCESS_ALL=$OUT_ROOT/droid_success_left_480x864
SUCCESS_TRAIN=$OUT_ROOT/droid_success_left_train_480x864
SUCCESS_TEST=$OUT_ROOT/droid_success_left_test_480x864
FAILURE_ALL=$OUT_ROOT/droid_failure_left_all_clean_480x864

mkdir -p "$OUT_ROOT" outputs/manual_runs

"$PY" scripts/convert_droid_v3_to_video_dataset.py \
  --src "$RAW_SUCCESS" \
  --out "$SUCCESS_ALL" \
  --view observation.images.left_external \
  --min-frames 33 \
  --workers "$WORKERS" \
  --height 480 \
  --width 864 \
  --fps 15 \
  --preset veryfast \
  --crf 18 \
  --threads "$THREADS" \
  --timeout-sec 600

"$PY" scripts/split_video_dataset.py \
  --src "$SUCCESS_ALL" \
  --train-out "$SUCCESS_TRAIN" \
  --test-out "$SUCCESS_TEST" \
  --test-count 1000 \
  --seed 20260511

"$PY" scripts/convert_droid_v3_to_video_dataset.py \
  --src "$RAW_FAILURE" \
  --out "$FAILURE_ALL" \
  --view observation.images.left_external \
  --min-frames 33 \
  --workers "$WORKERS" \
  --height 480 \
  --width 864 \
  --fps 15 \
  --preset veryfast \
  --crf 18 \
  --threads "$THREADS" \
  --timeout-sec 600
