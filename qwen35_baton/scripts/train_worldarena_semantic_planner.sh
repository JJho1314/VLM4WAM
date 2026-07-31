#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export NUM_GPUS="${NUM_GPUS:-8}"
export PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export CONFIG="${CONFIG:-qwen35_baton/configs/worldarena_stage1.json}"

exec "${SCRIPT_DIR}/train_semantic_planner.sh" "${CONFIG}"
