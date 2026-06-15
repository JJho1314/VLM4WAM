#!/usr/bin/env bash
# Train TAVID-style mask-free target feature input-channel conditioning.
# Cosmos receives InstructSAM decoder_dense target_feature as extra learned input
# channels; no explicit target mask is fed to the video model.

#SBATCH --job-name=cosmos-feat-chan
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=72:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4VLA/slurm-feature-input-channel-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4VLA/slurm-feature-input-channel-%j.err

set -uo pipefail

VLM4VLA_ROOT=${VLM4VLA_ROOT:-/data/user/jhe724/workspace/VLM4VLA}
REPO_ROOT=${REPO_ROOT:-$VLM4VLA_ROOT/third_party/cosmos-predict2.5}
BASE_SCRIPT=${BASE_SCRIPT:-$REPO_ROOT/scripts/sbatch_train_droid_success_v21_instructsam_strict_holdout_v3_vlm4vla.sh}

export VLM4VLA_ROOT
export REPO_ROOT
export DROID_SUCCESS_V21_TAVID_DIR=${DROID_SUCCESS_V21_TAVID_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3_scene_cap200_tasktarget}
export DROID_SUCCESS_V21_TAVID_VAL_DIR=${DROID_SUCCESS_V21_TAVID_VAL_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_val_strict_holdout_v3}
export TARGET_FEATURE_DIR_NAME=${TARGET_FEATURE_DIR_NAME:-target_features_instructsam_decoder_dense_stage2_lora}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-$VLM4VLA_ROOT/outputs/droid_success_v21_feature_input_channel_stage2_lora_scene_cap200_tasktarget}
export EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_feature_input_channel_target}

# 17,517 active train videos / global batch 128 = ceil(136.85) = 137 steps/epoch.
# Train 2,000 optimizer steps at global batch 128.
export BATCH_SIZE=${BATCH_SIZE:-2}
export GRAD_ACCUM_ITER=${GRAD_ACCUM_ITER:-8}
export MAX_ITER=${MAX_ITER:-2000}
export SAVE_ITER=${SAVE_ITER:-400}
export VALIDATION_ITER=${VALIDATION_ITER:-400}
export SAMPLE_ITER=${SAMPLE_ITER:-400}
export MAX_VAL_ITER=${MAX_VAL_ITER:-64}
export RUN_VALIDATION=${RUN_VALIDATION:-True}
export JOB_NAME=${JOB_NAME:-2b_droid_success_v21_feature_input_channel_stage2_lora_scene_cap200_tasktarget_49f_s234_bs2accum8_gbs128_2000step_val400_from_base}

if [ ! -f "$BASE_SCRIPT" ]; then
  echo "Invalid BASE_SCRIPT=${BASE_SCRIPT}; base training script not found." >&2
  exit 2
fi

for dir in "$DROID_SUCCESS_V21_TAVID_DIR/$TARGET_FEATURE_DIR_NAME" "$DROID_SUCCESS_V21_TAVID_VAL_DIR/$TARGET_FEATURE_DIR_NAME"; do
  if [ ! -d "$dir" ]; then
    echo "Missing target feature directory: $dir" >&2
    exit 3
  fi
done

exec bash "$BASE_SCRIPT" "$@"
