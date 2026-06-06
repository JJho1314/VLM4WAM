#!/usr/bin/env bash
# Train target branch with oracle GT-mask-derived target features.

#SBATCH --job-name=cosmos-gtfeat-branch
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=72:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4VLA/slurm-gtmask-feature-target-branch-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4VLA/slurm-gtmask-feature-target-branch-%j.err

set -uo pipefail

VLM4VLA_ROOT=${VLM4VLA_ROOT:-/data/user/jhe724/workspace/VLM4VLA}
REPO_ROOT=${REPO_ROOT:-$VLM4VLA_ROOT/third_party/cosmos-predict2.5}
TARGET_BRANCH_SCRIPT=${TARGET_BRANCH_SCRIPT:-$REPO_ROOT/scripts/sbatch_train_droid_success_v21_instructsam_target_branch_vlm4vla.sh}

export VLM4VLA_ROOT
export REPO_ROOT
export TARGET_FEATURE_DIR_NAME=${TARGET_FEATURE_DIR_NAME:-target_features_gt_mask}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-$VLM4VLA_ROOT/outputs/droid_success_v21_gt_mask_feature_target_branch_strict_holdout_v3}
export EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_instructsam_feature_target_branch}
export JOB_NAME=${JOB_NAME:-2b_droid_success_v21_gt_mask_feature_target_branch_strict_v3_49f_s234_bs2accum4_14k_val1000_from_base}

if [ ! -f "$TARGET_BRANCH_SCRIPT" ]; then
  echo "Invalid TARGET_BRANCH_SCRIPT=${TARGET_BRANCH_SCRIPT}; target branch training script not found." >&2
  exit 2
fi

exec bash "$TARGET_BRANCH_SCRIPT" "$@"
