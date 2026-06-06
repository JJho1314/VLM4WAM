#!/usr/bin/env bash
# Train the isolated target-feature cross-attention branch variant.

#SBATCH --job-name=cosmos-isam-branch
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=72:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4VLA/slurm-instructsam-target-branch-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4VLA/slurm-instructsam-target-branch-%j.err

set -uo pipefail

REPO_ROOT=${REPO_ROOT:-/data/user/jhe724/workspace/VLM4VLA/third_party/cosmos-predict2.5}
VLM4VLA_ROOT=${VLM4VLA_ROOT:-/data/user/jhe724/workspace/VLM4VLA}
BASE_SCRIPT=${BASE_SCRIPT:-$REPO_ROOT/scripts/sbatch_train_droid_success_v21_instructsam_strict_holdout_v3_vlm4vla.sh}

export REPO_ROOT
export VLM4VLA_ROOT
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-$VLM4VLA_ROOT/outputs/droid_success_v21_instructsam_feature_target_branch_strict_holdout_v3}
export EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_instructsam_feature_target_branch}
export JOB_NAME=${JOB_NAME:-2b_droid_success_v21_instructsam_feature_target_branch_strict_v3_49f_s234_bs2accum4_14k_val1000_from_base}

if [ ! -f "$BASE_SCRIPT" ]; then
  echo "Invalid BASE_SCRIPT=${BASE_SCRIPT}; target-branch wrapper could not find base training script." >&2
  exit 2
fi

exec bash "$BASE_SCRIPT" "$@"
