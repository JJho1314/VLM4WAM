#!/usr/bin/env bash
# Evaluate the one-channel feature-control-map Cosmos model on the yellow carrot
# with green leaves example. Cosmos inference receives target_feature only; target
# mask is removed before denoising.

#SBATCH --job-name=cosmos-yc-fmap
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4VLA/slurm-yc-feature-control-map-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4VLA/slurm-yc-feature-control-map-%j.err

set -uo pipefail

VLM4VLA_ROOT=${VLM4VLA_ROOT:-/data/user/jhe724/workspace/VLM4VLA}
REPO_ROOT=${REPO_ROOT:-$VLM4VLA_ROOT/third_party/cosmos-predict2.5}
BASE_SCRIPT=${BASE_SCRIPT:-$REPO_ROOT/scripts/sbatch_eval_feature_input_channel_yellow_carrot_green_leaf_vlm4vla.sh}

export VLM4VLA_ROOT
export REPO_ROOT
export RUN_ROOT=${RUN_ROOT:-$VLM4VLA_ROOT/outputs/eval_feature_control_map_channel_iter2000_yellow_carrot_green_leaf_$(date +%Y%m%d_%H%M%S)}
export CHECKPOINT=${CHECKPOINT:-$VLM4VLA_ROOT/outputs/droid_success_v21_feature_control_map_channel_stage2_lora_scene_cap200_tasktarget/cosmos_predict_v2p5/video2world/2b_droid_success_v21_feature_control_map_channel_stage2_lora_scene_cap200_tasktarget_49f_s234_bs2accum8_gbs128_2000step_val400_from_base/checkpoints/iter_000002000}
export EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_feature_control_map_channel_target}
export OUTPUT_PREFIX=${OUTPUT_PREFIX:-feature_control_map_channel_iter2000_yellow_carrot_green_leaf}

if [ ! -f "$BASE_SCRIPT" ]; then
  echo "Invalid BASE_SCRIPT=${BASE_SCRIPT}; base eval script not found." >&2
  exit 2
fi

exec bash "$BASE_SCRIPT" "$@"
