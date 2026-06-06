#!/usr/bin/env bash
# Prepare DROID success v2.1 left+right external target-aware data on HPC3.

set -uo pipefail
cd /data/user/jhe724/workspace/cosmos-predict2.5

export FFMPEG=${FFMPEG:-/data/apps/ffmpeg/7.0.2/ffmpeg}
PY=${PY:-.venv/bin/python}

SRC=${SRC:-/data/user/jhe724/workspace/data/droid_success_v21_lr}
RAW=${RAW:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_raw}
OUT=${OUT:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864}
VIZ_OUT=${VIZ_OUT:-/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_success_v21_target_aware_viz}
WORKERS=${WORKERS:-64}
TIMEOUT_SEC=${TIMEOUT_SEC:-600}

if [[ "${CLEAN:-0}" == "1" ]]; then
  rm -rf "$RAW" "$OUT" "$VIZ_OUT"
fi

mkdir -p "$(dirname "$RAW")" "$(dirname "$OUT")" "$VIZ_OUT" outputs/manual_runs

echo "date=$(date)"
echo "host=$(hostname)"
echo "src=$SRC"
echo "raw=$RAW"
echo "out=$OUT"
echo "workers=$WORKERS"

"$PY" scripts/convert_droid_v21_target_aware_to_video_dataset.py \
  --src "$SRC" \
  --out "$RAW" \
  --min-frames 33 \
  --overwrite-metadata \
  --video-keys observation.images.left_external observation.images.right_external

"$PY" scripts/reencode_video_dataset.py \
  --src "$RAW" \
  --out "$OUT" \
  --workers "$WORKERS" \
  --timeout-sec "$TIMEOUT_SEC" \
  --status-csv outputs/droid_success_v21_target_aware_480x864_status.csv \
  --fps 15 \
  --height 480 \
  --width 864

"$PY" scripts/visualize_target_masks.py \
  --dataset-dir "$OUT" \
  --out-dir "$VIZ_OUT" \
  --num-samples 24 \
  --frames-per-sample 4 \
  --seed 20260520

echo "counts:"
find "$OUT/videos" -maxdepth 1 -name '*.mp4' | wc -l
find "$OUT/masks" -maxdepth 1 -name '*.npz' | wc -l
find "$OUT/metas" -maxdepth 1 -name '*.txt' | wc -l
echo "viz=$VIZ_OUT"
