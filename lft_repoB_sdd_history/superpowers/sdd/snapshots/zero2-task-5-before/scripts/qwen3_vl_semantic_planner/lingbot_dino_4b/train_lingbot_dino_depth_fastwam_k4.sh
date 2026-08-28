#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)

export USE_DEPTH=1
export SEQUENCE_LENGTH=9
export NUM_KEYFRAMES=4
export GRID_SIZE=16
export SEMANTIC_DIM=1024
export KEYFRAME_SCHEME=even_future
export SHARED_LATENT_PER_KEYFRAME=32
export PRIVATE_LATENT_PER_KEYFRAME=32
export FASTWAM_DATA_CONFIG=${FASTWAM_DATA_CONFIG:-third_party/FastWAM/configs/data/libero_2cam_cosmos.yaml}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen3vl4b_lingbot_dino_depth_fastwam_k4}

exec "$SCRIPT_DIR/train_lingbot_dino_4b.sh" "$@"
