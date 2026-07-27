#!/usr/bin/env bash
set -euo pipefail

GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
NUM_GPUS="${NUM_GPUS:-8}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
CONFIG="${CONFIG:-${1:-qwen35_baton/configs/libero_stage1.json}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

denominator=$((NUM_GPUS * PER_DEVICE_BATCH))
if (( GLOBAL_BATCH != 128 )); then
  echo "Baton Stage 1 requires GLOBAL_BATCH=128" >&2
  exit 2
fi
if (( denominator <= 0 || GLOBAL_BATCH % denominator != 0 )); then
  echo "GLOBAL_BATCH must be divisible by NUM_GPUS * PER_DEVICE_BATCH" >&2
  exit 2
fi
GRAD_ACCUM=$((GLOBAL_BATCH / denominator))

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

"${PYTHON_BIN}" -m qwen35_baton.cli.preflight \
  --config "${CONFIG}" \
  --world-size "${NUM_GPUS}" \
  --per-device-batch "${PER_DEVICE_BATCH}" \
  --gradient-accumulation-steps "${GRAD_ACCUM}"
"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NUM_GPUS}" \
  -m qwen35_baton.cli.train_semantic_planner \
  --config "${CONFIG}" \
  --per-device-batch "${PER_DEVICE_BATCH}" \
  --gradient-accumulation-steps "${GRAD_ACCUM}"
