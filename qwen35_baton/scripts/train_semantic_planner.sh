#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-8}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
CONFIG="${CONFIG:-${1:-qwen35_baton/configs/libero_stage1.json}}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-qwen35_baton/configs/deepspeed_zero2.json}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if (( NUM_GPUS <= 0 || PER_DEVICE_BATCH <= 0 )); then
  echo "NUM_GPUS and PER_DEVICE_BATCH must be positive" >&2
  exit 2
fi
if (( GRAD_ACCUM != 1 )); then
  echo "Baton Stage 1 requires GRAD_ACCUM=1" >&2
  exit 2
fi
if [[ "${GRADIENT_CHECKPOINTING}" != "0" && "${GRADIENT_CHECKPOINTING}" != "1" ]]; then
  echo "GRADIENT_CHECKPOINTING must be 0 or 1" >&2
  exit 2
fi

checkpoint_flag="--no-gradient-checkpointing"
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  checkpoint_flag="--gradient-checkpointing"
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

"${PYTHON_BIN}" -m qwen35_baton.cli.preflight \
  --config "${CONFIG}" \
  --world-size "${NUM_GPUS}" \
  --per-device-batch "${PER_DEVICE_BATCH}" \
  --gradient-accumulation-steps 1 \
  --deepspeed-config-path "${DEEPSPEED_CONFIG}"
"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NUM_GPUS}" \
  -m qwen35_baton.cli.train_semantic_planner \
  --config "${CONFIG}" \
  --per-device-batch "${PER_DEVICE_BATCH}" \
  --gradient-accumulation-steps 1 \
  --deepspeed-config-path "${DEEPSPEED_CONFIG}" \
  "${checkpoint_flag}"
