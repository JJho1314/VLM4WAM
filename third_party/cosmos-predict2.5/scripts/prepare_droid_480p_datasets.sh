#!/usr/bin/env bash
# Pre-resize DROID VideoDataset directories to 480x864 so training does not
# spend every epoch resizing 720p frames online.

set -uo pipefail
cd /data/user/jhe724/workspace/cosmos-predict2.5

export FFMPEG=${FFMPEG:-/data/apps/ffmpeg/7.0.2/ffmpeg}
PY=${PY:-.venv/bin/python}

COMMON_ARGS=(
  --workers "${WORKERS:-24}"
  --timeout-sec "${TIMEOUT_SEC:-600}"
  --fps 15
  --height 480
  --width 864
)

"$PY" scripts/reencode_video_dataset.py \
  --src /data/user/jhe724/workspace/datasets/droid_success_left_train \
  --out /data/user/jhe724/workspace/datasets/droid_success_left_train_480x864 \
  --status-csv /data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_success_train_480x864_status.csv \
  "${COMMON_ARGS[@]}"

"$PY" scripts/reencode_video_dataset.py \
  --src /data/user/jhe724/workspace/datasets/droid_success_left_test \
  --out /data/user/jhe724/workspace/datasets/droid_success_left_test_480x864 \
  --status-csv /data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_success_test_480x864_status.csv \
  "${COMMON_ARGS[@]}"

"$PY" scripts/reencode_video_dataset.py \
  --src /data/user/jhe724/workspace/datasets/droid_failure_left_all_clean \
  --out /data/user/jhe724/workspace/datasets/droid_failure_left_all_clean_480x864 \
  --status-csv /data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_failure_clean_480x864_status.csv \
  "${COMMON_ARGS[@]}"
