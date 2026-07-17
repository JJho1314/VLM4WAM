#!/usr/bin/env bash
set -euo pipefail

GE_ACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${1:-$GE_ACT_ROOT/configs/ltx_model/libero/video_model_libero_fastwam_siglip2_hdf5.yaml}}"
NUM_PROCESSES="${NUM_PROCESSES:-${NPROC_PER_NODE:-8}}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
WORLD_SIZE="$((NNODES * NUM_PROCESSES))"

TORCHRUN_ARGS=(
  "--nnodes=$NNODES"
  "--nproc_per_node=$NUM_PROCESSES"
)
if [[ "$NNODES" == "1" && "$NODE_RANK" == "0" && -z "${MASTER_ADDR:-}" && -z "${MASTER_PORT:-}" ]]; then
  TORCHRUN_ARGS=(--standalone "${TORCHRUN_ARGS[@]}")
else
  TORCHRUN_ARGS+=(
    "--node_rank=$NODE_RANK"
    "--master_addr=${MASTER_ADDR:-127.0.0.1}"
    "--master_port=${MASTER_PORT:-29500}"
  )
fi

cd "$GE_ACT_ROOT"
python scripts/preflight_libero_fastwam_hdf5.py \
  --config "$CONFIG" \
  --world-size "$WORLD_SIZE"

torchrun \
  "${TORCHRUN_ARGS[@]}" \
  main.py \
  --config_file "$CONFIG"
