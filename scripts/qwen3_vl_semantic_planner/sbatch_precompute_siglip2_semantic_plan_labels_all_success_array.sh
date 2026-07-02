#!/usr/bin/env bash
# Precompute SigLIP2 semantic plan labels for the full DROID success training split.

#SBATCH --job-name=siglip2-plan-all
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-siglip2-plan-all-%A_%a.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-siglip2-plan-all-%A_%a.err

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
DATASET_ROOT=${DATASET_ROOT:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3}
OUTPUT_DIR=${OUTPUT_DIR:-$DATASET_ROOT/siglip2_semantic_plan_k6_g9_full}
TOTAL_SAMPLES=${TOTAL_SAMPLES:-72355}
NUM_SHARDS=${NUM_SHARDS:-8}
NUM_KEYFRAMES=${NUM_KEYFRAMES:-6}
GRID_SIZE=${GRID_SIZE:-9}
DTYPE=${DTYPE:-bf16}
OVERWRITE=${OVERWRITE:-0}

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

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SHARD_SIZE=$(( (TOTAL_SAMPLES + NUM_SHARDS - 1) / NUM_SHARDS ))
START_INDEX=$(( TASK_ID * SHARD_SIZE ))
MAX_SAMPLES=$SHARD_SIZE

mkdir -p "$OUTPUT_DIR"
echo "dataset=$DATASET_ROOT"
echo "output=$OUTPUT_DIR"
echo "task=$TASK_ID/$NUM_SHARDS start_index=$START_INDEX max_samples=$MAX_SAMPLES total_samples=$TOTAL_SAMPLES"
echo "keyframes=$NUM_KEYFRAMES grid=$GRID_SIZE dtype=$DTYPE overwrite=$OVERWRITE"

ARGS=(
  --dataset-root "$DATASET_ROOT"
  --model-path "$MODEL_PATH"
  --output-dir "$OUTPUT_DIR"
  --max-samples "$MAX_SAMPLES"
  --start-index "$START_INDEX"
  --num-keyframes "$NUM_KEYFRAMES"
  --grid-size "$GRID_SIZE"
  --dtype "$DTYPE"
  --manifest-name "manifest_${TASK_ID}.jsonl"
  --progress-name "progress_${TASK_ID}.jsonl"
  --summary-name "summary_${TASK_ID}.json"
)

if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

"$PY" scripts/qwen3_vl_semantic_planner/build_siglip2_semantic_plan_labels.py "${ARGS[@]}"
