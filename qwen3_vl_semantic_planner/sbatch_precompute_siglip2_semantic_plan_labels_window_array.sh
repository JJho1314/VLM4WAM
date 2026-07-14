#!/usr/bin/env bash
# Precompute Cosmos-aligned window-level SigLIP2 semantic plan labels.
#
# Each output sample corresponds to one exact Cosmos training clip:
#   start + frame_stride * arange(COSMOS_NUM_FRAMES)
# The Cosmos dataloader reads manifest*.jsonl, randomly selects one precomputed
# window per episode, and loads the matching semantic_plan by sample_id.

#SBATCH --job-name=siglip2-plan-win
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-siglip2-plan-win-%A_%a.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-siglip2-plan-win-%A_%a.err

set -uo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
cd "$VLM4WAM_ROOT" || exit 2
mkdir -p logs

module load gcc/11.5 cuda/12.8 nccl/2.25 2>/dev/null || true
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
CONDA_ENV=${CONDA_ENV:-starVLA}
conda activate "$CONDA_ENV" 2>/dev/null || true
PY=${PY:-/data/user/jhe724/.conda/envs/starVLA/bin/python}

MODEL_PATH=${MODEL_PATH:-/data/user/jhe724/workspace/weights/siglip2-so400m-patch14-384}
DATASET_ROOT=${DATASET_ROOT:-/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half}
NUM_KEYFRAMES=${NUM_KEYFRAMES:-16}
GRID_SIZE=${GRID_SIZE:-9}
COSMOS_NUM_FRAMES=${COSMOS_NUM_FRAMES:-93}
FRAME_STRIDES=${FRAME_STRIDES:-1,2,3}
WINDOW_STRIDE=${WINDOW_STRIDE:-24}
WINDOWS_PER_STEM=${WINDOWS_PER_STEM:-0}
WINDOW_SEED=${WINDOW_SEED:-20260629}
STRIDE_TAG=${FRAME_STRIDES//,/}
TOTAL_SAMPLES=${TOTAL_SAMPLES:-}
NUM_SHARDS=${NUM_SHARDS:-8}
DTYPE=${DTYPE:-bf16}
OVERWRITE=${OVERWRITE:-1}
if [[ "$GRID_SIZE" -le 0 ]]; then
  GRID_TAG="gnative"
else
  GRID_TAG="g${GRID_SIZE}"
fi
OUTPUT_DIR=${OUTPUT_DIR:-$DATASET_ROOT/siglip2_semantic_plan_k${NUM_KEYFRAMES}_${GRID_TAG}_cosmos_t${COSMOS_NUM_FRAMES}_s${STRIDE_TAG}_step${WINDOW_STRIDE}_full}

if [[ ! -x "$PY" ]]; then
  echo "ERROR: python executable not found: $PY" >&2
  exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: SigLIP2 model path not found: $MODEL_PATH" >&2
  exit 2
fi
if [[ ! -f "$DATASET_ROOT/frame_ranges.json" ]]; then
  echo "ERROR: frame_ranges.json not found under $DATASET_ROOT" >&2
  exit 2
fi

if [[ -z "$TOTAL_SAMPLES" ]]; then
  TOTAL_SAMPLES=$("$PY" qwen3_vl_semantic_planner/build_siglip2_semantic_plan_labels.py \
    --dataset-root "$DATASET_ROOT" \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --num-keyframes "$NUM_KEYFRAMES" \
    --grid-size "$GRID_SIZE" \
    --expand-ranges \
    --window-stride "$WINDOW_STRIDE" \
    --cosmos-stride-windows \
    --sequence-length "$COSMOS_NUM_FRAMES" \
    --frame-strides "$FRAME_STRIDES" \
    --windows-per-stem "$WINDOWS_PER_STEM" \
    --window-seed "$WINDOW_SEED" \
    --count-only)
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SHARD_SIZE=$(( (TOTAL_SAMPLES + NUM_SHARDS - 1) / NUM_SHARDS ))
START_INDEX=$(( TASK_ID * SHARD_SIZE ))
MAX_SAMPLES=$SHARD_SIZE

mkdir -p "$OUTPUT_DIR"
echo "dataset=$DATASET_ROOT"
echo "output=$OUTPUT_DIR"
echo "task=$TASK_ID/$NUM_SHARDS start_index=$START_INDEX max_samples=$MAX_SAMPLES total_samples=$TOTAL_SAMPLES"
echo "keyframes=$NUM_KEYFRAMES grid=$GRID_SIZE cosmos_num_frames=$COSMOS_NUM_FRAMES frame_strides=$FRAME_STRIDES window_stride=$WINDOW_STRIDE windows_per_stem=$WINDOWS_PER_STEM window_seed=$WINDOW_SEED dtype=$DTYPE overwrite=$OVERWRITE"

ARGS=(
  --dataset-root "$DATASET_ROOT"
  --model-path "$MODEL_PATH"
  --output-dir "$OUTPUT_DIR"
  --max-samples "$MAX_SAMPLES"
  --start-index "$START_INDEX"
  --num-keyframes "$NUM_KEYFRAMES"
  --grid-size "$GRID_SIZE"
  --expand-ranges
  --window-stride "$WINDOW_STRIDE"
  --cosmos-stride-windows
  --sequence-length "$COSMOS_NUM_FRAMES"
  --frame-strides "$FRAME_STRIDES"
  --windows-per-stem "$WINDOWS_PER_STEM"
  --window-seed "$WINDOW_SEED"
  --dtype "$DTYPE"
  --manifest-name "manifest_${TASK_ID}.jsonl"
  --progress-name "progress_${TASK_ID}.jsonl"
  --summary-name "summary_${TASK_ID}.json"
)

if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

"$PY" qwen3_vl_semantic_planner/build_siglip2_semantic_plan_labels.py "${ARGS[@]}"
