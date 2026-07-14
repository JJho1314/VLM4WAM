#!/usr/bin/env bash
# Precompute Cosmos-aligned SigLIP2 semantic plan labels without spatial pooling.
#
# This keeps native SigLIP2 spatial patch tokens (for siglip2-so400m-patch14-384:
# normally 27x27 tokens per keyframe) instead of pooling to 9x9.

#SBATCH --job-name=siglip2-plan-native
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=36:00:00
#SBATCH --array=0-31
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-siglip2-plan-native-%A_%a.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-siglip2-plan-native-%A_%a.err

set -euo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
export VLM4WAM_ROOT
export GRID_SIZE=${GRID_SIZE:-0}
export NUM_KEYFRAMES=${NUM_KEYFRAMES:-16}
export COSMOS_NUM_FRAMES=${COSMOS_NUM_FRAMES:-93}
export FRAME_STRIDES=${FRAME_STRIDES:-1,2,3}
export WINDOW_STRIDE=${WINDOW_STRIDE:-24}
export NUM_SHARDS=${NUM_SHARDS:-32}
export OVERWRITE=${OVERWRITE:-1}

STRIDE_TAG=${FRAME_STRIDES//,/}
DATASET_ROOT=${DATASET_ROOT:-/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half}
export DATASET_ROOT
export OUTPUT_DIR=${OUTPUT_DIR:-$DATASET_ROOT/siglip2_semantic_plan_k${NUM_KEYFRAMES}_gnative_cosmos_t${COSMOS_NUM_FRAMES}_s${STRIDE_TAG}_step${WINDOW_STRIDE}_full}

exec bash "$VLM4WAM_ROOT/qwen3_vl_semantic_planner/sbatch_precompute_siglip2_semantic_plan_labels_window_array.sh"
