#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-8}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
CONFIG="${CONFIG:-${1:-qwen35_baton/configs/libero_stage1.json}}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if (( NUM_GPUS <= 0 || PER_DEVICE_BATCH <= 0 )); then
  echo "NUM_GPUS and PER_DEVICE_BATCH must be positive" >&2
  exit 2
fi
if (( GRAD_ACCUM <= 0 )); then
  echo "GRAD_ACCUM must be positive" >&2
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
stop_args=()
if [[ -n "${STOP_AT_STEP:-}" ]]; then
  stop_args=(--stop-at-step "${STOP_AT_STEP}")
fi

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
  --gradient-accumulation-steps "${GRAD_ACCUM}" \
  "${stop_args[@]}" \
  "${checkpoint_flag}"
