#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM/third_party/cosmos-predict2.5}"
PROJECT_ROOT="$(cd "${REPO_ROOT}/../.." && pwd)"
UNIFIED_ENV="${UNIFIED_ENV:-/data/LFT-W02_data/.conda/envs/cosmos-instructsam}"
TORCHRUN="${TORCHRUN:-${UNIFIED_ENV}/bin/torchrun}"
PYTHON="${PYTHON:-${UNIFIED_ENV}/bin/python}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/local_strict_feature_switch_iter2000_yc_vs_banana_${TIMESTAMP}}"
DATASET_DIR="${RUN_ROOT}/dataset_same_prompt"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/outputs/pulled_checkpoints/latent_grounding_iter_000002000}"
EXPERIMENT="${EXPERIMENT:-predict2_video2world_training_2b_droid_success_v21_latent_grounding_decoder_dense_target}"

SOURCE_DATASET="${PROJECT_ROOT}/eval_prev_iter2000_full/input_datasets/robointer_74616_yellow_carrot_prompt_targetaware_dataset"
SOURCE_VIDEO="${SOURCE_DATASET}/videos/74616_exterior_image_1_left.mp4"
SOURCE_MASK="${SOURCE_DATASET}/target_masks/74616_exterior_image_1_left.npz"
CARROT_FEATURE="${SOURCE_DATASET}/target_features_instructsam_decoder_dense_stage2_lora_green_leaf_prompt_s20260613/74616_exterior_image_1_left.pt"
BANANA_FEATURE="${PROJECT_ROOT}/eval_prev_iter2000_full/input_datasets/robointer_74616_banana_prompt_targetaware_dataset/target_features_instructsam_decoder_dense_stage2_lora_banana_prompt_s20260613/74616_exterior_image_1_left.pt"

SEED="${SEED:-20260613}"
NUM_STEPS="${NUM_STEPS:-35}"
GUIDANCE="${GUIDANCE:-3.0}"
FPS="${FPS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PROMPT="${PROMPT:-A Franka robotic arm with a parallel-jaw gripper carefully grasp only the [TGT] target object and place it into the black pot.}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-strict_feature_switch_iter2000_yc_vs_banana}"

for path in "${TORCHRUN}" "${PYTHON}" "${SOURCE_VIDEO}" "${CARROT_FEATURE}" "${BANANA_FEATURE}" "${CHECKPOINT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 2
  fi
done

mkdir -p "${RUN_ROOT}/logs" "${DATASET_DIR}/videos" "${DATASET_DIR}/metas"
cp -f "${SOURCE_VIDEO}" "${DATASET_DIR}/videos/74616_exterior_image_1_left.mp4"
printf '%s\n' "${PROMPT}" > "${DATASET_DIR}/metas/74616_exterior_image_1_left.txt"
if [[ -f "${SOURCE_MASK}" ]]; then
  mkdir -p "${DATASET_DIR}/target_masks"
  cp -f "${SOURCE_MASK}" "${DATASET_DIR}/target_masks/74616_exterior_image_1_left.npz"
fi

cd "${REPO_ROOT}"
export VIRTUAL_ENV="${UNIFIED_ENV}"
export PATH="${UNIFIED_ENV}/bin:${PATH}"
unset PYTHONHOME

NV_LIB="${UNIFIED_ENV}/lib/python3.10/site-packages/nvidia"
if [[ -d "${NV_LIB}" ]]; then
  export LD_LIBRARY_PATH="${NV_LIB}/cudnn/lib:${NV_LIB}/cuda_runtime/lib:${NV_LIB}/cuda_nvrtc/lib:${NV_LIB}/cublas/lib:${NV_LIB}/cusparse/lib:${NV_LIB}/cusolver/lib:${NV_LIB}/cufft/lib:${NV_LIB}/curand/lib:${NV_LIB}/nccl/lib:${NV_LIB}/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/packages/cosmos-cuda:${REPO_ROOT}/packages/cosmos-oss:${PYTHONPATH:-}"
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
  echo "run_root=${RUN_ROOT}"
  echo "dataset=${DATASET_DIR}"
  echo "prompt=${PROMPT}"
  echo "checkpoint=${CHECKPOINT}"
  echo "experiment=${EXPERIMENT}"
  echo "carrot_feature=${CARROT_FEATURE}"
  echo "banana_feature=${BANANA_FEATURE}"
  echo "seed=${SEED}"
  echo "num_steps=${NUM_STEPS}"
  echo "guidance=${GUIDANCE}"
  echo "fps=${FPS}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  sha256sum "${CARROT_FEATURE}" "${BANANA_FEATURE}" || true
} | tee "${RUN_ROOT}/logs/00_run_info.log"

"${PYTHON}" - <<PY | tee "${RUN_ROOT}/logs/01_feature_check.log"
import json
from pathlib import Path
import torch

paths = {
    "carrot_feature": Path("${CARROT_FEATURE}"),
    "banana_feature": Path("${BANANA_FEATURE}"),
}
features = {}
for name, path in paths.items():
    obj = torch.load(path, map_location="cpu", weights_only=False)
    feat = obj["target_feature"] if isinstance(obj, dict) else obj
    features[name] = feat.float()
    print(json.dumps({
        "name": name,
        "path": str(path),
        "shape": list(feat.shape),
        "query": obj.get("query") if isinstance(obj, dict) else None,
        "feature_mode": obj.get("feature_mode") if isinstance(obj, dict) else None,
        "mean": float(feat.float().mean()),
        "std": float(feat.float().std()),
    }, indent=2))

a = features["carrot_feature"].reshape(-1)
b = features["banana_feature"].reshape(-1)
cos = torch.nn.functional.cosine_similarity(a, b, dim=0)
l2 = torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a).clamp_min(1e-6)
print(json.dumps({"carrot_vs_banana_cosine": float(cos), "relative_l2": float(l2)}, indent=2))
PY

run_variant() {
  local name="$1"
  local mode="$2"
  local feature_path="${3:-}"
  local out="${RUN_ROOT}/${name}"
  local log="${RUN_ROOT}/logs/${name}.log"
  mkdir -p "${out}"
  local extra=()
  if [[ "${mode}" == "path" ]]; then
    extra=(--target-feature-path "${feature_path}")
  fi

  set +e
  "${TORCHRUN}" --standalone --nproc_per_node=1 scripts/generate_tavid_mask_samples.py \
    --config cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${out}" \
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
    --target-feature-mode "${mode}" \
    "${extra[@]}" \
    -- experiment="${EXPERIMENT}" \
    dataloader_train.batch_size=1 \
    dataloader_train.num_workers=2 \
    dataloader_train.drop_last=False \
    dataloader_train.dataset.target_mask_dir=none \
    dataloader_train.dataset.target_mask_default_to_zero=True \
    dataloader_train.dataset.target_feature_dir="$(dirname "${CARROT_FEATURE}")" \
    dataloader_train.dataset.target_feature_default_to_zero=False \
    trainer.grad_accum_iter=1 \
    trainer.run_validation=False \
    2>&1 | tee "${log}"
  local status="${PIPESTATUS[0]}"
  set -e
  echo "${status}" > "${RUN_ROOT}/logs/${name}.exit"
  return "${status}"
}

overall=0
run_variant carrot_feature path "${CARROT_FEATURE}" || overall=1
run_variant banana_feature path "${BANANA_FEATURE}" || overall=1
run_variant zero_feature zero || overall=1

if [[ "${overall}" -eq 0 ]]; then
  analyze_args=(
    --run-root "${RUN_ROOT}"
    --variants carrot_feature banana_feature zero_feature
    --output-prefix "${OUTPUT_PREFIX}"
  )
  if [[ -f "${DATASET_DIR}/target_masks/74616_exterior_image_1_left.npz" ]]; then
    analyze_args+=(--mask-npz "${DATASET_DIR}/target_masks/74616_exterior_image_1_left.npz")
  fi
  "${PYTHON}" scripts/analyze_target_feature_ablation.py "${analyze_args[@]}" \
    2>&1 | tee "${RUN_ROOT}/logs/06_analyze.log" || overall=1
fi

"${PYTHON}" - <<PY
import json
from pathlib import Path
root = Path("${RUN_ROOT}")
summary = {
    "run_root": str(root),
    "dataset": "${DATASET_DIR}",
    "prompt": "${PROMPT}",
    "checkpoint": "${CHECKPOINT}",
    "experiment": "${EXPERIMENT}",
    "seed": int("${SEED}"),
    "num_steps": int("${NUM_STEPS}"),
    "guidance": float("${GUIDANCE}"),
    "fps": int("${FPS}"),
    "variants": ["carrot_feature", "banana_feature", "zero_feature"],
    "carrot_feature": "${CARROT_FEATURE}",
    "banana_feature": "${BANANA_FEATURE}",
    "explicit_mask_used_for_inference": False,
    "cosmos_target_mask_removed": True,
    "overall_exit": int("${overall}"),
    "contact_sheet": str(root / ("${OUTPUT_PREFIX}" + "_contact_sheet.jpg")),
    "diff_sheet": str(root / ("${OUTPUT_PREFIX}" + "_diff_vs_keep.jpg")),
    "metrics": str(root / ("${OUTPUT_PREFIX}" + "_metrics.json")),
}
(root / "strict_feature_switch_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "${RUN_ROOT}"
exit "${overall}"
