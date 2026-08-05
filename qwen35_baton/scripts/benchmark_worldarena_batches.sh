#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-qwen35_baton/configs/worldarena_stage1.json}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/worldarena_stage1_batches}"
NUM_GPUS="${NUM_GPUS:-8}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
MEASURED_STEPS="${MEASURED_STEPS:-20}"

exec "${PYTHON_BIN}" -m qwen35_baton.cli.benchmark_stage1_throughput \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --world-size "${NUM_GPUS}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --measured-steps "${MEASURED_STEPS}"
