#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)

export USE_DEPTH=1
export USE_CURRENT_ALIGNMENT=1
export NUM_TASK_TOKENS=${NUM_TASK_TOKENS:-8}
export SEQUENCE_LENGTH=9
export NUM_KEYFRAMES=1
export GRID_SIZE=16
export SEMANTIC_DIM=1024
export KEYFRAME_SCHEME=even_future
export CURRENT_DINO_LOSS_WEIGHT=${CURRENT_DINO_LOSS_WEIGHT:-0.004}
export FUTURE_DINO_LOSS_WEIGHT=${FUTURE_DINO_LOSS_WEIGHT:-0.004}
export CURRENT_DEPTH_LOSS_WEIGHT=${CURRENT_DEPTH_LOSS_WEIGHT:-0.004}
export FUTURE_DEPTH_LOSS_WEIGHT=${FUTURE_DEPTH_LOSS_WEIGHT:-0.004}
export FASTWAM_DATA_CONFIG=${FASTWAM_DATA_CONFIG:-third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen3vl4b_lingbot_current_future_fastwam_k1}

exec "$SCRIPT_DIR/train_lingbot_dino_4b.sh" "$@"
