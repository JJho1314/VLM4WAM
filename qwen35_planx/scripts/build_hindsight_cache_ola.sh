#!/usr/bin/env bash
set -euo pipefail

: "${HDF5_MANIFEST:?set HDF5_MANIFEST to the immutable GE-Act manifest}"
: "${WINDOW_MANIFEST:?set WINDOW_MANIFEST to the hindsight window manifest}"
: "${TA_TOK_CHECKPOINT:?set TA_TOK_CHECKPOINT to the released checkpoint}"
: "${SIGLIP2_MODEL_DIR:?set SIGLIP2_MODEL_DIR to a complete local model}"
: "${DINOV3_MODEL_DIR:?set DINOV3_MODEL_DIR to a complete local model}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a new finalized cache path}"
: "${NUM_GPUS:?set NUM_GPUS to the local worker count}"

if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: NUM_GPUS must be a positive integer" >&2
  exit 2
fi

export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SHARD_ROOT="${OUTPUT_DIR}.shards"

python -m qwen35_planx.cli.preflight hindsight-cache \
  --hdf5-manifest "$HDF5_MANIFEST" \
  --window-manifest "$WINDOW_MANIFEST" \
  --ta-checkpoint "$TA_TOK_CHECKPOINT" \
  --siglip-model "$SIGLIP2_MODEL_DIR" \
  --dinov3-model "$DINOV3_MODEL_DIR" \
  --output-dir "$OUTPUT_DIR"

torchrun --standalone --nproc_per_node="$NUM_GPUS" \
  -m qwen35_planx.cli.build_hindsight_cache build \
  --hdf5-manifest "$HDF5_MANIFEST" \
  --window-manifest "$WINDOW_MANIFEST" \
  --ta-checkpoint "$TA_TOK_CHECKPOINT" \
  --siglip-model "$SIGLIP2_MODEL_DIR" \
  --dinov3-model "$DINOV3_MODEL_DIR" \
  --output "$SHARD_ROOT" \
  --num-shards "$NUM_GPUS"

python -m qwen35_planx.cli.build_hindsight_cache finalize \
  --window-manifest "$WINDOW_MANIFEST" \
  --shard-root "$SHARD_ROOT" \
  --output "$OUTPUT_DIR"

python -m qwen35_planx.cli.build_hindsight_cache audit \
  --cache "$OUTPUT_DIR" \
  --samples 128 \
  --output "$OUTPUT_DIR/metrics.json"
