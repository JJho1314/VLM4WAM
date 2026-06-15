#!/usr/bin/env bash
# Train dense feature-map target injection on the DROID success v21 scene-cap200 task-target split.

#SBATCH --job-name=cosmos-dense-fmap
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=72:00:00
#SBATCH --output=/data/user/jli545/workspace/VLM4VLA/slurm-dense-feature-map-target-%j.out
#SBATCH --error=/data/user/jli545/workspace/VLM4VLA/slurm-dense-feature-map-target-%j.err

set -uo pipefail

VLM4VLA_ROOT=${VLM4VLA_ROOT:-/data/user/jli545/workspace/VLM4VLA}
REPO_ROOT=${REPO_ROOT:-$VLM4VLA_ROOT/third_party/cosmos-predict2.5}
BASE_SCRIPT=${BASE_SCRIPT:-$REPO_ROOT/scripts/sbatch_train_droid_success_v21_instructsam_strict_holdout_v3_vlm4vla.sh}

export VLM4VLA_ROOT
export REPO_ROOT
export DROID_SUCCESS_V21_TAVID_DIR=${DROID_SUCCESS_V21_TAVID_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3_scene_cap200_tasktarget}
export DROID_SUCCESS_V21_TAVID_VAL_DIR=${DROID_SUCCESS_V21_TAVID_VAL_DIR:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_val_strict_holdout_v3}
export TARGET_FEATURE_DIR_NAME=${TARGET_FEATURE_DIR_NAME:-target_features_gt_mask_spatial64}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-$VLM4VLA_ROOT/outputs/droid_success_v21_dense_feature_map_target_scene_cap200_tasktarget}
export EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_dense_feature_map_target}
# 17,517 active train videos / global batch 128 = ceil(136.85) = 137 steps/epoch.
# Train 10 epochs = 1,370 optimizer steps.
export BATCH_SIZE=${BATCH_SIZE:-2}
export GRAD_ACCUM_ITER=${GRAD_ACCUM_ITER:-8}
export MAX_ITER=${MAX_ITER:-1370}
export SAVE_ITER=${SAVE_ITER:-274}
export VALIDATION_ITER=${VALIDATION_ITER:-274}
export SAMPLE_ITER=${SAMPLE_ITER:-274}
export JOB_NAME=${JOB_NAME:-2b_droid_success_v21_dense_feature_map_target_scene_cap200_tasktarget_49f_s234_bs2accum8_gbs128_10ep1370_val274_from_base}

if [ ! -f "$BASE_SCRIPT" ]; then
  echo "Invalid BASE_SCRIPT=${BASE_SCRIPT}; dense feature-map target wrapper could not find base training script." >&2
  exit 2
fi

exec bash "$BASE_SCRIPT" "$@"
