#!/usr/bin/env bash
# Evaluate the match-ground-v3 ControlNet/TAVID-style side-branch checkpoint on
# the yellow-carrot-with-green-leaves example. Inference removes target masks and
# runs keep/zero/drop feature ablations.

#SBATCH --job-name=yc-mgv3ctrl
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-eval-mgv3ctrl-yc-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-eval-mgv3ctrl-yc-%j.err

set -uo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
REPO_ROOT=${REPO_ROOT:-$VLM4WAM_ROOT/third_party/cosmos-predict2.5}
BASE_SCRIPT=${BASE_SCRIPT:-$REPO_ROOT/scripts/run_match_ground_v3_spatial_iter400_yellow_carrot_local.sh}

RUN_DIR=${RUN_DIR:-/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_v21_match_ground_v3_control_side_branch_49f_s123_vlm4wam/cosmos_predict_v2p5/video2world/2b_mgv3_control_side_branch_iou50_49f_s123_bs2accum8_gbs128_1600}

export REPO_ROOT
export PROJECT_ROOT=$VLM4WAM_ROOT
export COSMOS_VENV=${COSMOS_VENV:-/data/user/jhe724/workspace/cosmos-predict2.5/.venv}
export COSMOS_CHECKPOINTS_DIR=${COSMOS_CHECKPOINTS_DIR:-/data/user/jhe724/workspace/weights}
export DATASET_DIR=${DATASET_DIR:-$REPO_ROOT/outputs/tavid_generation_runs/robointer_74616_yellow_carrot_prompt_targetaware_dataset}
export RAW_FEATURE_DIR=${RAW_FEATURE_DIR:-$DATASET_DIR/target_features_rawseg_ft}
export DENSE_FEATURE_DIR=${DENSE_FEATURE_DIR:-$DATASET_DIR/target_features_instructsam_decoder_dense_stage2_lora_green_leaf_prompt}
export CHECKPOINT=${CHECKPOINT:-$RUN_DIR/checkpoints/iter_000001600}
export EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_match_ground_v3_control_side_branch}
export RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/outputs/eval_mgv3_control_side_branch_iter1600_yellow_carrot_green_leaf_$(date +%Y%m%d_%H%M%S)}
export OUTPUT_PREFIX=${OUTPUT_PREFIX:-mgv3_control_side_branch_iter1600_yellow_carrot_green_leaf}
export SEED=${SEED:-20260613}
export NUM_STEPS=${NUM_STEPS:-35}
export GUIDANCE=${GUIDANCE:-3.0}
export FPS=${FPS:-8}
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES=${DROID_SUCCESS_V21_TAVID_NUM_FRAMES:-49}
export DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=${DROID_SUCCESS_V21_TAVID_FRAME_STRIDES:-1}
export DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=${DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY:-range_start}

mkdir -p "$VLM4WAM_ROOT/logs"

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true
export PATH=/data/apps/gcc/11.5/bin:${PATH}
export CC=/data/apps/gcc/11.5/bin/gcc
export CXX=/data/apps/gcc/11.5/bin/g++

if [ ! -f "$BASE_SCRIPT" ]; then
  echo "Missing BASE_SCRIPT: $BASE_SCRIPT" >&2
  exit 2
fi
if [ ! -d "$CHECKPOINT" ]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 3
fi
if [ ! -d "$DATASET_DIR" ]; then
  echo "Missing dataset: $DATASET_DIR" >&2
  exit 4
fi

exec bash "$BASE_SCRIPT" "$@"
