#!/usr/bin/env bash
# Run cosmos image2world on a list of sample names (one per line).
# All JSONs go to a single CLI call so the model loads once per GPU.
set -euo pipefail

export COSMOS_CHECKPOINTS_DIR=/data/LFT-W02_data/junjie/weights
export HF_HUB_OFFLINE=1

INPUT_DIR=/data/LFT-W02_data/junjie/data/robointer_test_inputs
OUTPUT_DIR=/data/LFT-W02_data/junjie/cosmos-predict2.5/outputs/robointer_zero_shot
CHECKPOINT_PATH=/data/LFT-W02_data/junjie/weights/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt

LIST_FILE=${1:?usage: $0 <sample_list_file>}
GPU_ID=${GPU_ID:-1}

mkdir -p "${OUTPUT_DIR}"

# Per-sample loop. Multi-input via repeated `-i` doesn't work (tyro keeps last
# value only), so we re-launch inference.py per sample and pay the model-load
# overhead each time. Each sample lands in OUTPUT_DIR/<name>/<name>.mp4.
TOTAL=0
DONE=0
while IFS= read -r name; do
  [[ -z "$name" ]] && continue
  TOTAL=$((TOTAL+1))
  out_sub="${OUTPUT_DIR}/${name}"
  if [[ -f "${out_sub}/${name}.mp4" ]]; then
    DONE=$((DONE+1))
    continue
  fi
  echo "[$(date +%H:%M:%S)] GPU ${GPU_ID}: ${name}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} uv run --extra=cu128 python /data/LFT-W02_data/junjie/cosmos-predict2.5/examples/inference.py \
    -i "${INPUT_DIR}/${name}.json" \
    -o "${out_sub}" \
    --inference-type=image2world \
    --disable-guardrails \
    --checkpoint-path "${CHECKPOINT_PATH}" || echo "[$(date +%H:%M:%S)] WARN: ${name} failed, continuing"
done < "${LIST_FILE}"

echo "[$(date +%H:%M:%S)] GPU ${GPU_ID}: list done (${TOTAL} total, ${DONE} pre-existing)."
