#!/usr/bin/env bash
# Validate local 480x864 DROID VideoDataset outputs and retry failed episodes.

set -euo pipefail
cd /data/LFT-W02_data/junjie/cosmos-predict2.5

PY=${PY:-python}
WORKERS=${WORKERS:-96}
THREADS=${THREADS:-1}
VALIDATE_WORKERS=${VALIDATE_WORKERS:-64}

RAW_SUCCESS=/data/LFT-W02_data/junjie/data/droid_success
RAW_FAILURE=/data/LFT-W02_data/junjie/data/droid_failure
OUT_ROOT=/data/LFT-W02_data/junjie/datasets

SUCCESS_ALL=$OUT_ROOT/droid_success_left_480x864
SUCCESS_TRAIN=$OUT_ROOT/droid_success_left_train_480x864
SUCCESS_TEST=$OUT_ROOT/droid_success_left_test_480x864
FAILURE_ALL=$OUT_ROOT/droid_failure_left_all_clean_480x864

mkdir -p outputs/manual_runs

repair_one() {
  local raw_dir=$1
  local out_dir=$2
  local label=$3

  echo "=== remove temporary files: $label ==="
  find "$out_dir/videos" -maxdepth 1 -name '*.tmp.mp4' -delete 2>/dev/null || true

  echo "=== validate/delete bad: $label ==="
  "$PY" scripts/validate_video_dataset.py \
    --dataset "$out_dir" \
    --workers "$VALIDATE_WORKERS" \
    --bad-list "outputs/manual_runs/${label}_bad_before.tsv" \
    --delete-bad

  echo "=== retry conversion: $label ==="
  "$PY" scripts/convert_droid_v3_to_video_dataset.py \
    --src "$raw_dir" \
    --out "$out_dir" \
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

  echo "=== final validate: $label ==="
  "$PY" scripts/validate_video_dataset.py \
    --dataset "$out_dir" \
    --workers "$VALIDATE_WORKERS" \
    --bad-list "outputs/manual_runs/${label}_bad_after.tsv"
}

repair_one "$RAW_SUCCESS" "$SUCCESS_ALL" droid_success_left_480x864

"$PY" scripts/split_video_dataset.py \
  --src "$SUCCESS_ALL" \
  --train-out "$SUCCESS_TRAIN" \
  --test-out "$SUCCESS_TEST" \
  --test-count 1000 \
  --seed 20260511

repair_one "$RAW_FAILURE" "$FAILURE_ALL" droid_failure_left_all_clean_480x864
