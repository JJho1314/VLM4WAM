#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/cosmos-predict2.5"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${REPO_ROOT}/outputs/eval_latent_grounding_iter2000_yellow_carrot_online_feature_path_${TIMESTAMP}"
CHECKPOINT="${REPO_ROOT}/outputs/pulled_checkpoints/latent_grounding_iter_000002000"
DATASET_DIR="${REPO_ROOT}/outputs/tavid_generation_runs/robointer_74616_yellow_carrot_prompt_targetaware_dataset"
FEATURE_DIR="${DATASET_DIR}/target_features_instructsam_decoder_dense_stage2_lora_green_leaf_prompt"
ONLINE_FEATURE="${REPO_ROOT}/outputs/online_maskfree_instructsam_cosmos_yellow_carrot_iter2000_matched_20260615_232332/generated/_online_instructsam_features/yellow_carrot_online_maskfree_iter2000_matched.pt"
EXPERIMENT="predict2_video2world_training_2b_droid_success_v21_latent_grounding_decoder_dense_target"
COSMOS_VENV="/data/LFT-W02_data/junjie/cosmos-predict2.5/.venv"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/path"

{
  date
  hostname
  echo "run_root=${RUN_ROOT}"
  echo "checkpoint=${CHECKPOINT}"
  echo "dataset=${DATASET_DIR}"
  echo "reference_feature_dir=${FEATURE_DIR}"
  echo "online_feature=${ONLINE_FEATURE}"
  echo "seed=20260613"
  echo "num_steps=35"
  echo "guidance=3.0"
  echo "fps=8"
  echo "frames=49"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
} | tee "${RUN_ROOT}/logs/00_run_info.log"

cd "${REPO_ROOT}"
export VIRTUAL_ENV="${COSMOS_VENV}"
export PATH="${COSMOS_VENV}/bin:${PATH}"
unset PYTHONHOME

NV_LIB="${COSMOS_VENV}/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="${NV_LIB}/cudnn/lib:${NV_LIB}/cuda_runtime/lib:${NV_LIB}/cuda_nvrtc/lib:${NV_LIB}/cublas/lib:${NV_LIB}/cusparse/lib:${NV_LIB}/cusolver/lib:${NV_LIB}/cufft/lib:${NV_LIB}/curand/lib:${NV_LIB}/nccl/lib:${NV_LIB}/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/packages/cosmos-cuda:${REPO_ROOT}/packages/cosmos-oss:${PYTHONPATH:-}"
export COSMOS_CHECKPOINTS_DIR="${COSMOS_CHECKPOINTS_DIR:-/data/LFT-W02_data/junjie/weights}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export COSMOS_SKIP_CUDA_VERSION_CHECK="${COSMOS_SKIP_CUDA_VERSION_CHECK:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export DROID_SUCCESS_V21_TAVID_DIR="${DATASET_DIR}"
export DROID_SUCCESS_V21_TAVID_VAL_DIR="${DATASET_DIR}"
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES=49
export DROID_SUCCESS_V21_TAVID_FRAME_STRIDES=1
export DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY=range_start

set +e
torchrun --standalone --nproc_per_node=1 scripts/generate_tavid_mask_samples.py \
  --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${RUN_ROOT}/path" \
  --num-samples 1 \
  --num-steps 35 \
  --guidance 3.0 \
  --seed 20260613 \
  --fps 8 \
  --max-batches 1 \
  --standalone-only \
  --reuse-encoded-latent \
  --offload-denoiser-during-vae \
  --offload-denoiser-before-decode \
  --allow-empty-target-mask \
  --remove-target-mask \
  --target-feature-mode path \
  --target-feature-path "${ONLINE_FEATURE}" \
  -- experiment="${EXPERIMENT}" \
  dataloader_train.batch_size=1 \
  dataloader_train.num_workers=2 \
  dataloader_train.drop_last=False \
  dataloader_train.dataset.target_mask_dir=none \
  dataloader_train.dataset.target_mask_default_to_zero=True \
  dataloader_train.dataset.target_feature_dir="${FEATURE_DIR}" \
  dataloader_train.dataset.target_feature_default_to_zero=False \
  trainer.grad_accum_iter=1 \
  trainer.run_validation=False \
  2>&1 | tee "${RUN_ROOT}/logs/path.log"
status="${PIPESTATUS[0]}"
set -e

echo "${status}" > "${RUN_ROOT}/logs/path.exit"
echo "${RUN_ROOT}"
exit "${status}"
