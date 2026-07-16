#!/usr/bin/env bash
set -euo pipefail

GE_ACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$GE_ACT_ROOT/configs/ltx_model/libero/video_model_libero_fastwam_siglip2.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

cd "$GE_ACT_ROOT"
python scripts/preflight_ltx_siglip2.py --config "$CONFIG" --world-size "$NPROC_PER_NODE"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC_PER_NODE" \
  main.py \
  --config_file "$CONFIG"
