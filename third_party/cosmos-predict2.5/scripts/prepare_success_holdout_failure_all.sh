#!/usr/bin/env bash
set -euo pipefail

cd /data/user/jhe724/workspace/cosmos-predict2.5

VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
export VIRTUAL_ENV=$VENV
export PATH=$VENV/bin:$PATH

SUCCESS_SRC=/data/user/jhe724/workspace/datasets/droid_success_left
SUCCESS_TRAIN=/data/user/jhe724/workspace/datasets/droid_success_left_train
SUCCESS_TEST=/data/user/jhe724/workspace/datasets/droid_success_left_test
FAILURE_ALL=/data/user/jhe724/workspace/datasets/droid_failure_left_all
FAILURE_TRAIN=/data/user/jhe724/workspace/datasets/droid_failure_left_train
FAILURE_TEST=/data/user/jhe724/workspace/datasets/droid_failure_left_test

python scripts/split_video_dataset.py \
  --src "$SUCCESS_SRC" \
  --train-out "$SUCCESS_TRAIN" \
  --test-out "$SUCCESS_TEST" \
  --test-ratio 0.05 \
  --seed 20260511

python scripts/link_video_datasets.py \
  --out "$FAILURE_ALL" \
  "$FAILURE_TRAIN" \
  "$FAILURE_TEST"

echo "Prepared success holdout + all failure datasets:"
for d in "$SUCCESS_TRAIN" "$SUCCESS_TEST" "$FAILURE_ALL"; do
  printf "%s videos " "$d"
  find "$d/videos" -maxdepth 1 -name '*.mp4' | wc -l
  printf "%s metas " "$d"
  find "$d/metas" -maxdepth 1 -name '*.txt' | wc -l
done
