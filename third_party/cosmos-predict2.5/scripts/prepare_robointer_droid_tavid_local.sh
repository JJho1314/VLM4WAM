#!/usr/bin/env bash
# Build the full RoboInter/LeRobot DROID TAViD-mask Cosmos VideoDataset locally
# on LFT-W02 (under /data/LFT-W02_data/junjie). Result is meant to be rsynced
# to the remote SLURM cluster at
#   jhe724:/data/user/jhe724/workspace/datasets/robointer_droid_tavid_primary
# which is the path the
# `predict2_video2world_training_2b_robointer_droid_tavid_mask` experiment
# already points at.
#
# Pipeline:
#   1) convert_lerobot_droid_to_video_dataset.py: hardlink primary-camera mp4s
#      into videos/, write metas/episode_NNNNNN.txt prompts.
#   2) link_robointer_droid_masks.py: read parquet `episode_name`, then symlink
#      OXE_DROID SAM .npz into target_masks/episode_NNNNNN.npz.

set -uo pipefail
cd /data/LFT-W02_data/junjie/cosmos-predict2.5

PY=${PY:-python}
WORKERS=${WORKERS:-32}

SRC_LEROBOT=/data/LFT-W02_data/junjie/data/InternRobotics/RoboInter-Data/Annotation_with_action_lerobotv21/lerobot_droid_anno
SEG_ROOT=/data/LFT-W02_data/junjie/data/InternRobotics/RoboInter-Data/segmentation_npz/OXE_DROID/data/ann_human
OUT=${OUT:-/data/LFT-W02_data/junjie/datasets/robointer_droid_tavid_primary}

MAX_EPISODES=${MAX_EPISODES:-}        # leave empty for full set
MIN_FRAMES=${MIN_FRAMES:-33}
VIDEO_KEY=${VIDEO_KEY:-observation.images.primary}

mkdir -p "$OUT"

echo "[1/2] convert lerobot -> Cosmos VideoDataset format at $OUT"
CONVERT_ARGS=(
  --src "$SRC_LEROBOT"
  --out "$OUT"
  --min-frames "$MIN_FRAMES"
  --video-key "$VIDEO_KEY"
)
if [[ -n "$MAX_EPISODES" ]]; then
  CONVERT_ARGS+=( --max-episodes "$MAX_EPISODES" )
fi
"$PY" scripts/convert_lerobot_droid_to_video_dataset.py "${CONVERT_ARGS[@]}"
status=$?
if [[ $status -ne 0 ]]; then
  echo "convert step failed ($status)"; exit "$status"
fi

echo "[2/2] link RoboInter SAM masks into $OUT/target_masks"
"$PY" scripts/link_robointer_droid_masks.py \
  --annotation-root "$SRC_LEROBOT/data" \
  --seg-root "$SEG_ROOT" \
  --workers "$WORKERS" \
  "$OUT"
status=$?
if [[ $status -ne 0 ]]; then
  echo "link step failed ($status)"; exit "$status"
fi

echo "=== summary ==="
echo "videos:       $(ls -1 "$OUT/videos" 2>/dev/null | wc -l)"
echo "metas:        $(ls -1 "$OUT/metas" 2>/dev/null | wc -l)"
echo "target_masks: $(ls -1 "$OUT/target_masks" 2>/dev/null | wc -l)"
echo "summary json: $OUT/target_masks_link_summary.json"
[[ -f "$OUT/target_masks_link_summary.json" ]] && cat "$OUT/target_masks_link_summary.json"
