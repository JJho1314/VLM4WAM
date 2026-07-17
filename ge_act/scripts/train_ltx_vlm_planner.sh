#!/usr/bin/env bash
set -euo pipefail

GE_ACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(dirname "$GE_ACT_ROOT")"
CONFIG="${1:-$GE_ACT_ROOT/configs/ltx_model/libero/video_model_libero_vlm_planner_hdf5.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"

export PYTHONPATH="$REPO_ROOT:$GE_ACT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

cd "$GE_ACT_ROOT"
python scripts/preflight_ltx_siglip2.py \
  --config "$CONFIG" \
  --world-size "$NPROC_PER_NODE"

MAIN_ARGS=(--config_file "$CONFIG")
if [[ -n "$MAX_TRAIN_STEPS" ]]; then
  MAIN_ARGS+=(--max_train_steps "$MAX_TRAIN_STEPS")
fi

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  main.py \
  "${MAIN_ARGS[@]}"
