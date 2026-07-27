#!/usr/bin/env bash
set -euo pipefail

GE_ACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALLER_PWD="$PWD"
CONFIG="${CONFIG:-${1:-$GE_ACT_ROOT/configs/ltx_model/libero/action_model_libero_baton_stage3_hdf5.yaml}}"
if [[ "$CONFIG" != /* ]]; then
  CONFIG="$CALLER_PWD/$CONFIG"
fi
CONFIG="$(realpath -m -- "$CONFIG")"
REPOSITORY_ROOT="$(cd "$GE_ACT_ROOT/.." && pwd)"
export PYTHONPATH="$REPOSITORY_ROOT:$GE_ACT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_PROCESSES="${NUM_PROCESSES:-${NPROC_PER_NODE:-8}}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
WORLD_SIZE="$((NNODES * NUM_PROCESSES))"
GLOBAL_BATCH="${GLOBAL_BATCH:-128}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
DENOMINATOR="$((WORLD_SIZE * PER_DEVICE_BATCH))"
if ((DENOMINATOR <= 0 || GLOBAL_BATCH % DENOMINATOR != 0)); then
  echo "GLOBAL_BATCH must be divisible by WORLD_SIZE * PER_DEVICE_BATCH" >&2
  exit 2
fi
GRADIENT_ACCUMULATION_STEPS="$((GLOBAL_BATCH / DENOMINATOR))"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
export BATON_PER_DEVICE_BATCH="$PER_DEVICE_BATCH"
export BATON_GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS"

if ((NNODES > 1)) && [[ -z "${MASTER_ADDR:-}" ]]; then
  echo "MASTER_ADDR must be set when NNODES > 1" >&2
  exit 2
fi
TORCHRUN_ARGS=("--nnodes=$NNODES" "--nproc_per_node=$NUM_PROCESSES")
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
RESOLVED_CONFIG="$(mktemp "${TMPDIR:-/tmp}/ge-act-baton-stage3.XXXXXX.yaml")"
cleanup() {
  rm -f -- "$RESOLVED_CONFIG"
}
trap cleanup EXIT
"$PYTHON_BIN" scripts/preflight_ltx_siglip2.py \
  --config "$CONFIG" \
  --materialize-output "$RESOLVED_CONFIG"
CONFIG="$RESOLVED_CONFIG"
"$PYTHON_BIN" scripts/preflight_libero_fastwam_hdf5.py \
  --config "$CONFIG" \
  --world-size "$WORLD_SIZE"
"$PYTHON_BIN" scripts/preflight_ltx_siglip2.py \
  --config "$CONFIG" \
  --world-size "$WORLD_SIZE" \
  --per-device-batch "$PER_DEVICE_BATCH" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"

MAIN_ARGS=(--config_file "$CONFIG")
if [[ -n "$MAX_TRAIN_STEPS" ]]; then
  MAIN_ARGS+=(--max_train_steps "$MAX_TRAIN_STEPS")
fi
"$PYTHON_BIN" -m torch.distributed.run \
  "${TORCHRUN_ARGS[@]}" \
  main.py \
  "${MAIN_ARGS[@]}"
