#!/usr/bin/env bash
# Train dense feature-map target injection using fine-tuned InstructSAM stage2 LoRA spatial features.

#SBATCH --job-name=cosmos-dense-fmap-ft
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=72:00:00
#SBATCH --output=/data/user/jli545/workspace/VLM4VLA/slurm-dense-feature-map-stage2-lora-%j.out
#SBATCH --error=/data/user/jli545/workspace/VLM4VLA/slurm-dense-feature-map-stage2-lora-%j.err

set -uo pipefail

VLM4VLA_ROOT=${VLM4VLA_ROOT:-/data/user/jli545/workspace/VLM4VLA}
REPO_ROOT=${REPO_ROOT:-$VLM4VLA_ROOT/third_party/cosmos-predict2.5}
BASE_SCRIPT=${BASE_SCRIPT:-$REPO_ROOT/scripts/sbatch_train_droid_success_v21_dense_feature_map_target_scene_cap200_tasktarget_vlm4vla.sh}

export VLM4VLA_ROOT
export REPO_ROOT
export TARGET_FEATURE_DIR_NAME=${TARGET_FEATURE_DIR_NAME:-target_features_gt_mask_spatial64_instructsam_stage2_lora}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-$VLM4VLA_ROOT/outputs/droid_success_v21_dense_feature_map_stage2_lora_scene_cap200_tasktarget}
export JOB_NAME=${JOB_NAME:-2b_droid_success_v21_dense_feature_map_stage2_lora_scene_cap200_tasktarget_49f_s234_bs2accum8_gbs128_10ep1370_val274_from_base}

if [ ! -f "$BASE_SCRIPT" ]; then
  echo "Invalid BASE_SCRIPT=${BASE_SCRIPT}; dense feature-map stage2-lora wrapper could not find base script." >&2
  exit 2
fi

exec bash "$BASE_SCRIPT" "$@"
