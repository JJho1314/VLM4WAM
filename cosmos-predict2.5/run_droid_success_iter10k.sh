#!/usr/bin/env bash
# Inference with the droid_success post-trained checkpoint at iter_10000.
# Uses the same 5 v2 samples we ran earlier with the post-trained base, so
# outputs land in a sibling dir for easy side-by-side comparison.
set -euo pipefail

export COSMOS_CHECKPOINTS_DIR=/data/LFT-W02_data/junjie/weights
export HF_HUB_OFFLINE=1

INPUT_DIR=/data/LFT-W02_data/junjie/data/robointer_test_inputs
OUTPUT_DIR=/data/LFT-W02_data/junjie/cosmos-predict2.5/outputs/robointer_finetuned_iter10k
CHECKPOINT_PATH=/data/LFT-W02_data/junjie/weights/Cosmos-Predict2.5-2B/finetuned/droid_success_iter10000/model_ema_bf16.pt
GPU_ID=${GPU_ID:-1}

# v2 manifest names (5 samples we already have outputs for from base post-trained)
SAMPLES=(
  1762_exterior_image_1_left
  22830_exterior_image_1_left
  67448_exterior_image_1_left
  74616_exterior_image_1_left
  RH20T_cfg5_task_0001_user_0010_scene_0002_cfg_0005_104422070044
)

mkdir -p "${OUTPUT_DIR}"

for name in "${SAMPLES[@]}"; do
  out_sub="${OUTPUT_DIR}/${name}"
  if [[ -f "${out_sub}/${name}.mp4" ]]; then
    echo "skip ${name} (already done)"
    continue
  fi
  echo "[$(date +%H:%M:%S)] === ${name} ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} uv run --extra=cu128 python /data/LFT-W02_data/junjie/cosmos-predict2.5/examples/inference.py \
    -i "${INPUT_DIR}/${name}.json" \
    -o "${out_sub}" \
    --inference-type=image2world \
    --disable-guardrails \
    --checkpoint-path "${CHECKPOINT_PATH}"
done

echo "All done. Outputs in: ${OUTPUT_DIR}"
echo "Compare to base post-trained outputs: /data/LFT-W02_data/junjie/cosmos-predict2.5/outputs/robointer_zero_shot/<name>/<name>.mp4"
