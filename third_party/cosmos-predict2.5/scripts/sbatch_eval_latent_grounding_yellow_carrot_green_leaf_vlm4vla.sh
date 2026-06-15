#!/usr/bin/env bash
# Evaluate the mask-free latent-grounding Cosmos model on the yellow carrot with
# green leaves example. Cosmos receives only InstructSAM decoder-dense feature.

#SBATCH --job-name=cosmos-yc-latent
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4VLA/slurm-yc-latent-ground-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4VLA/slurm-yc-latent-ground-%j.err

set -uo pipefail

VLM4VLA_ROOT=${VLM4VLA_ROOT:-/data/user/jhe724/workspace/VLM4VLA}
REPO_ROOT=${REPO_ROOT:-$VLM4VLA_ROOT/third_party/cosmos-predict2.5}
BASE_SCRIPT=${BASE_SCRIPT:-$REPO_ROOT/scripts/sbatch_eval_feature_input_channel_yellow_carrot_green_leaf_vlm4vla.sh}
RUN_DIR=${RUN_DIR:-$VLM4VLA_ROOT/outputs/droid_success_v21_latent_grounding_decoder_dense_stage2_lora_scene_cap200_tasktarget/cosmos_predict_v2p5/video2world/2b_droid_success_v21_latent_grounding_decoder_dense_stage2_lora_scene_cap200_tasktarget_49f_s234_bs2accum8_gbs128_2000step_val400_from_base}

export VLM4VLA_ROOT
export REPO_ROOT
export CHECKPOINT=${CHECKPOINT:-$RUN_DIR/checkpoints/iter_000000400}
export EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_latent_grounding_decoder_dense_target}
export RUN_ROOT=${RUN_ROOT:-$VLM4VLA_ROOT/outputs/eval_latent_grounding_iter400_yellow_carrot_green_leaf_$(date +%Y%m%d_%H%M%S)}
export OUTPUT_PREFIX=${OUTPUT_PREFIX:-latent_grounding_iter400_yellow_carrot_green_leaf}
export NUM_STEPS=${NUM_STEPS:-35}
export GUIDANCE=${GUIDANCE:-3.0}
export SEED=${SEED:-20260613}

if [ ! -f "$BASE_SCRIPT" ]; then
  echo "Invalid BASE_SCRIPT=${BASE_SCRIPT}; base eval script not found." >&2
  exit 2
fi
if [ ! -d "$CHECKPOINT" ]; then
  echo "Missing checkpoint directory: $CHECKPOINT" >&2
  exit 3
fi

exec bash "$BASE_SCRIPT" "$@"
