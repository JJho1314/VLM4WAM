#!/usr/bin/env bash
set -euo pipefail

cd /data/user/jhe724/workspace/cosmos-predict2.5

SRC=/data/user/jhe724/workspace/data/droid_failure
SPLIT=/data/user/jhe724/workspace/datasets/droid_failure_split
TRAIN_OUT=/data/user/jhe724/workspace/datasets/droid_failure_left_train
TEST_OUT=/data/user/jhe724/workspace/datasets/droid_failure_left_test

VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
export VIRTUAL_ENV=$VENV
export PATH=$VENV/bin:$PATH

python scripts/prepare_droid_failure_split.py \
  --src "$SRC" \
  --out "$SPLIT" \
  --test-ratio 0.05 \
  --seed 20260511 \
  --min-frames 33

python scripts/convert_droid_v3_to_video_dataset.py \
  --src "$SRC" \
  --out "$TRAIN_OUT" \
  --episode-list "$SPLIT/train_episodes.txt" \
  --min-frames 33 \
  --workers 32

python scripts/convert_droid_v3_to_video_dataset.py \
  --src "$SRC" \
  --out "$TEST_OUT" \
  --episode-list "$SPLIT/test_episodes.txt" \
  --min-frames 33 \
  --workers 16

echo "Prepared droid_failure train/test datasets:"
find "$SPLIT" -maxdepth 1 -type f -print
find "$TRAIN_OUT/videos" -maxdepth 1 -name '*.mp4' | wc -l
find "$TEST_OUT/videos" -maxdepth 1 -name '*.mp4' | wc -l
