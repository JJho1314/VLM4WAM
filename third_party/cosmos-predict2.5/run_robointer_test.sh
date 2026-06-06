#!/usr/bin/env bash
# Wait for the Wan2.2 zero-shot run to finish on GPU 1, then run Cosmos
# zero-shot on the same input set so the two outputs are directly comparable.
set -euo pipefail

export COSMOS_CHECKPOINTS_DIR=/data/LFT-W02_data/junjie/weights
export HF_HUB_OFFLINE=1   # belt-and-suspenders: combined with checkpoint_db patch

INPUT_DIR=/data/LFT-W02_data/junjie/data/robointer_test_inputs
OUTPUT_DIR=/data/LFT-W02_data/junjie/cosmos-predict2.5/outputs/robointer_zero_shot
CHECKPOINT_PATH=/data/LFT-W02_data/junjie/weights/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt
GPU_ID=${GPU_ID:-1}

mkdir -p "${OUTPUT_DIR}"

echo "[$(date +%H:%M:%S)] Starting Cosmos on GPU ${GPU_ID}."

# 2) Run Cosmos image2world over each sample.
for input_json in "${INPUT_DIR}"/*.json; do
  base_name="$(basename "${input_json}" .json)"
  out_sub="${OUTPUT_DIR}/${base_name}"
  if [[ -d "${out_sub}" ]] && find "${out_sub}" -name "*.mp4" -print -quit | grep -q .; then
    echo "[$(date +%H:%M:%S)] skip ${base_name} (already done)"
    continue
  fi
  # Clean stale empty dir from a prior failed run so the inference doesn't
  # think it has results to keep.
  rm -rf "${out_sub}"
  echo "[$(date +%H:%M:%S)] === ${base_name} ==="
  CUDA_VISIBLE_DEVICES=${GPU_ID} uv run --extra=cu128 python /data/LFT-W02_data/junjie/cosmos-predict2.5/examples/inference.py \
    -i "${input_json}" \
    -o "${out_sub}" \
    --inference-type=image2world \
    --disable-guardrails \
    --checkpoint-path "${CHECKPOINT_PATH}"
done

echo "[$(date +%H:%M:%S)] All done. Outputs in: ${OUTPUT_DIR}"
