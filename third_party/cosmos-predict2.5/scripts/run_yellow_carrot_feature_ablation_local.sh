#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/cosmos-predict2.5}
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
VENV=${VENV:-/data/LFT-W02_data/junjie/cosmos-predict2.5/.venv}
TORCHRUN=${TORCHRUN:-$VENV/bin/torchrun}
export COSMOS_CHECKPOINTS_DIR=${COSMOS_CHECKPOINTS_DIR:-/data/LFT-W02_data/junjie/weights}
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/packages/cosmos-cuda:$REPO_ROOT/packages/cosmos-oss:${PYTHONPATH:-}"
export DROID_SUCCESS_V21_TAVID_DIR=${DROID_SUCCESS_V21_TAVID_DIR:-$REPO_ROOT/outputs/tavid_generation_runs/robointer_74616_yellow_carrot_prompt_targetaware_dataset}
export DROID_SUCCESS_V21_TAVID_VAL_DIR=${DROID_SUCCESS_V21_TAVID_VAL_DIR:-$DROID_SUCCESS_V21_TAVID_DIR}
export DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=${DROID_SUCCESS_V21_TAVID_FRAME_STRIDES:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export WANDB_MODE=${WANDB_MODE:-disabled}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export COSMOS_SKIP_CUDA_VERSION_CHECK=${COSMOS_SKIP_CUDA_VERSION_CHECK:-1}

CHECKPOINT=${CHECKPOINT:-$REPO_ROOT/outputs/pulled_checkpoints/instructsam_iter000014000_dcp}
EXPERIMENT=${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_instructsam_feature_context}
RUN_ROOT=${RUN_ROOT:-$REPO_ROOT/outputs/tavid_generation_runs/robointer_74616_iter14000_feature_ablation_yellow_carrot_480p_49f_35step}
SEED=${SEED:-20260526}
NUM_STEPS=${NUM_STEPS:-35}
GUIDANCE=${GUIDANCE:-3.0}
FPS=${FPS:-8}
PRECISE_FEATURE=${PRECISE_FEATURE:-$DROID_SUCCESS_V21_TAVID_DIR/target_features_precise_prompt/74616_exterior_image_1_left.pt}
WRONG_FEATURE=${WRONG_FEATURE:-$REPO_ROOT/outputs/eval_iter14000_instructsam_follow_target_8scenes/dataset/target_features/episode_014122_left_external.pt}

mkdir -p "$RUN_ROOT/logs"
{
  date
  hostname
  echo "repo=$REPO_ROOT"
  echo "dataset=$DROID_SUCCESS_V21_TAVID_DIR"
  echo "checkpoint=$CHECKPOINT"
  echo "run_root=$RUN_ROOT"
  echo "torchrun=$TORCHRUN"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "precise_feature=$PRECISE_FEATURE"
  echo "wrong_feature=$WRONG_FEATURE"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
} | tee "$RUN_ROOT/logs/00_run_info.log"

run_one() {
  local name=$1
  local mode=$2
  local feature_path=${3:-}
  local out="$RUN_ROOT/$name"
  local log="$RUN_ROOT/logs/${name}.log"
  mkdir -p "$out"
  local extra=()
  if [[ "$mode" == "path" ]]; then
    extra=(--target-feature-path "$feature_path")
  fi
  "$TORCHRUN" --standalone --nproc_per_node=1 scripts/generate_tavid_mask_samples.py \
    --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$out" \
    --num-samples 1 \
    --num-steps "$NUM_STEPS" \
    --guidance "$GUIDANCE" \
    --seed "$SEED" \
    --fps "$FPS" \
    --max-batches 1 \
    --standalone-only \
    --reuse-encoded-latent \
    --offload-denoiser-during-vae \
    --offload-denoiser-before-decode \
    --target-feature-mode "$mode" \
    "${extra[@]}" \
    -- experiment="$EXPERIMENT" \
    dataloader_train.batch_size=1 \
    dataloader_train.num_workers=2 \
    dataloader_train.drop_last=False \
    dataloader_train.dataset.target_mask_dropout_prob=0.0 \
    dataloader_train.dataset.target_mask_default_to_zero=False \
    dataloader_train.dataset.target_feature_default_to_zero=False \
    trainer.grad_accum_iter=1 \
    trainer.run_validation=False \
    2>&1 | tee "$log"
}

run_one keep keep
run_one zero zero
run_one drop drop
run_one wrong_black_mug path "$WRONG_FEATURE"
run_one precise_prompt path "$PRECISE_FEATURE"
