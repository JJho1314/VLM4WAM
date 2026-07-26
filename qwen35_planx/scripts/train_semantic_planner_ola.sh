#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-8}"
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
MAX_STEPS="${MAX_STEPS:-30000}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
QWEN_LANGUAGE_LR="${QWEN_LANGUAGE_LR:-1e-5}"
QWEN_VISION_LR="${QWEN_VISION_LR:-5e-6}"
HEAD_LR="${HEAD_LR:-1e-4}"
ADAPTER_LR="${ADAPTER_LR:-1e-4}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
CONFIG="${CONFIG:-${1:-}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

if [[ -z "${CONFIG}" ]]; then
  echo "CONFIG must point to a planner-training JSON config" >&2
  exit 2
fi
denominator=$((NUM_GPUS * PER_DEVICE_BATCH))
if (( denominator <= 0 || GLOBAL_BATCH % denominator != 0 )); then
  echo "GLOBAL_BATCH must be divisible by NUM_GPUS * PER_DEVICE_BATCH" >&2
  exit 2
fi
GRAD_ACCUM=$((GLOBAL_BATCH / denominator))
if (( GLOBAL_BATCH != 256 )); then
  echo "planner stage requires GLOBAL_BATCH=256" >&2
  exit 2
fi

export ADAPTER_LR
"${PYTHON_BIN}" -m qwen35_planx.cli.preflight planner-training \
  --config "${CONFIG}" \
  --num-processes "${NUM_GPUS}"
"${TORCHRUN_BIN}" --standalone --nproc_per_node="${NUM_GPUS}" \
  -m qwen35_planx.cli.train_semantic_planner \
  --config "${CONFIG}" \
  --max-steps "${MAX_STEPS}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --save-every "${SAVE_EVERY}" \
  --validate-every "${SAVE_EVERY}" \
  --per-device-batch "${PER_DEVICE_BATCH}" \
  --gradient-accumulation-steps "${GRAD_ACCUM}" \
  --qwen-language-lr "${QWEN_LANGUAGE_LR}" \
  --qwen-vision-lr "${QWEN_VISION_LR}" \
  --head-lr "${HEAD_LR}"
