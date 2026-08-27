#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/cosmos-predict2.5}"
UNIFIED_ENV="${UNIFIED_ENV:-/data/LFT-W02_data/.conda/envs/cosmos-instructsam}"
PYTHON="${PYTHON:-${UNIFIED_ENV}/bin/python}"
TORCHRUN="${TORCHRUN:-${UNIFIED_ENV}/bin/torchrun}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/eval_latent_grounding_iter2000_yellow_carrot_online_unified_env_${TIMESTAMP}}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/outputs/pulled_checkpoints/latent_grounding_iter_000002000}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/outputs/tavid_generation_runs/robointer_74616_yellow_carrot_prompt_targetaware_dataset}"
FEATURE_DIR="${FEATURE_DIR:-${DATASET_DIR}/target_features_instructsam_decoder_dense_stage2_lora_green_leaf_prompt}"
INPUT_VIDEO="${INPUT_VIDEO:-${DATASET_DIR}/videos/74616_exterior_image_1_left.mp4}"
TARGET_QUERY="${TARGET_QUERY:-Please segment the yellow carrot with green leaves in the image.}"
INSTRUCTSAM_SOURCE_ROOT="${INSTRUCTSAM_SOURCE_ROOT:-/data/LFT-W02_data/junjie/InstructSAM}"
INSTRUCTSAM_MODEL_PATH="${INSTRUCTSAM_MODEL_PATH:-/data/LFT-W02_data/junjie/weights/jhe724/instructsam_stage2_complete_lora_69644}"
EXPERIMENT="${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_latent_grounding_decoder_dense_target}"

NUM_STEPS="${NUM_STEPS:-35}"
GUIDANCE="${GUIDANCE:-3.0}"
SEED="${SEED:-20260613}"
FPS="${FPS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

ONLINE_FEATURE="${RUN_ROOT}/online_features/yellow_carrot_instructsam_decoder_dense.pt"
LOG_DIR="${RUN_ROOT}/logs"
OUT_DIR="${RUN_ROOT}/path"

mkdir -p "${LOG_DIR}" "${OUT_DIR}" "$(dirname "${ONLINE_FEATURE}")"
cd "${REPO_ROOT}"

export VIRTUAL_ENV="${UNIFIED_ENV}"
export PATH="${UNIFIED_ENV}/bin:${PATH}"
unset PYTHONHOME

NV_LIB="${UNIFIED_ENV}/lib/python3.10/site-packages/nvidia"
if [[ -d "${NV_LIB}" ]]; then
  export LD_LIBRARY_PATH="${NV_LIB}/cudnn/lib:${NV_LIB}/cuda_runtime/lib:${NV_LIB}/cuda_nvrtc/lib:${NV_LIB}/cublas/lib:${NV_LIB}/cusparse/lib:${NV_LIB}/cusolver/lib:${NV_LIB}/cufft/lib:${NV_LIB}/curand/lib:${NV_LIB}/nccl/lib:${NV_LIB}/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/packages/cosmos-cuda:${REPO_ROOT}/packages/cosmos-oss:${INSTRUCTSAM_SOURCE_ROOT}:${PYTHONPATH:-}"
export COSMOS_CHECKPOINTS_DIR="${COSMOS_CHECKPOINTS_DIR:-/data/LFT-W02_data/junjie/weights}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export COSMOS_SKIP_CUDA_VERSION_CHECK="${COSMOS_SKIP_CUDA_VERSION_CHECK:-1}"
export CUDA_VISIBLE_DEVICES
export DROID_SUCCESS_V21_TAVID_DIR="${DATASET_DIR}"
export DROID_SUCCESS_V21_TAVID_VAL_DIR="${DATASET_DIR}"
export DROID_SUCCESS_V21_TAVID_NUM_FRAMES="${DROID_SUCCESS_V21_TAVID_NUM_FRAMES:-49}"
export DROID_SUCCESS_V21_TAVID_FRAME_STRIDES="${DROID_SUCCESS_V21_TAVID_FRAME_STRIDES:-1}"
export DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY="${DROID_SUCCESS_V21_TAVID_FRAME_START_POLICY:-range_start}"

{
  date
  hostname
  echo "unified_env=${UNIFIED_ENV}"
  "${PYTHON}" - <<'PY'
import torch, transformers, peft
print(f"torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
print(f"transformers={transformers.__version__}")
print(f"peft={peft.__version__}")
PY
  echo "run_root=${RUN_ROOT}"
  echo "checkpoint=${CHECKPOINT}"
  echo "dataset=${DATASET_DIR}"
  echo "input_video=${INPUT_VIDEO}"
  echo "target_query=${TARGET_QUERY}"
  echo "online_feature=${ONLINE_FEATURE}"
  echo "seed=${SEED}"
  echo "num_steps=${NUM_STEPS}"
  echo "guidance=${GUIDANCE}"
  echo "fps=${FPS}"
  echo "frames=${DROID_SUCCESS_V21_TAVID_NUM_FRAMES}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
} | tee "${LOG_DIR}/00_run_info.log"

"${PYTHON}" scripts/extract_instructsam_feature_once.py \
  --input-path "${INPUT_VIDEO}" \
  --target-query "${TARGET_QUERY}" \
  --model-path "${INSTRUCTSAM_MODEL_PATH}" \
  --source-root "${INSTRUCTSAM_SOURCE_ROOT}" \
  --output-path "${ONLINE_FEATURE}" \
  --feature-mode decoder_dense \
  2>&1 | tee "${LOG_DIR}/01_extract_instructsam_feature.log"

set +e
"${TORCHRUN}" --standalone --nproc_per_node=1 scripts/generate_tavid_mask_samples.py \
  --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUT_DIR}" \
  --num-samples 1 \
  --num-steps "${NUM_STEPS}" \
  --guidance "${GUIDANCE}" \
  --seed "${SEED}" \
  --fps "${FPS}" \
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
  2>&1 | tee "${LOG_DIR}/02_generate_cosmos.log"
status="${PIPESTATUS[0]}"
set -e

echo "${status}" > "${LOG_DIR}/02_generate_cosmos.exit"
"${PYTHON}" - <<PY
import json
from pathlib import Path
summary = {
    "run_root": "${RUN_ROOT}",
    "online_feature": "${ONLINE_FEATURE}",
    "generated": "${OUT_DIR}/sample_000_generated.mp4",
    "gt": "${OUT_DIR}/sample_000_gt.mp4",
    "caption": "${OUT_DIR}/sample_000_caption.txt",
    "exit_status": ${status},
}
Path("${RUN_ROOT}/unified_env_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

exit "${status}"
