#!/usr/bin/env bash
# Precompute Baton Stage-1 SigLIP2 continuous semantic blueprints for DROID videos.

#SBATCH --job-name=siglip2-plan-label
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-siglip2-plan-label-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-siglip2-plan-label-%j.err

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
OUTPUT_DIR=${OUTPUT_DIR:-$DATASET_ROOT/siglip2_semantic_plan_k6_g9_full}
MAX_SAMPLES=${MAX_SAMPLES:-0}
START_INDEX=${START_INDEX:-0}
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

echo "model=$MODEL_PATH"
echo "dataset=$DATASET_ROOT"
echo "output=$OUTPUT_DIR"
echo "max_samples=$MAX_SAMPLES start_index=$START_INDEX keyframes=$NUM_KEYFRAMES grid=$GRID_SIZE"
nvidia-smi -L || true

ARGS=(
  --dataset-root "$DATASET_ROOT"
  --model-path "$MODEL_PATH"
  --output-dir "$OUTPUT_DIR"
  --max-samples "$MAX_SAMPLES"
  --start-index "$START_INDEX"
  --num-keyframes "$NUM_KEYFRAMES"
  --grid-size "$GRID_SIZE"
  --dtype "$DTYPE"
)

if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

"$PY" scripts/qwen3_vl_semantic_planner/build_siglip2_semantic_plan_labels.py "${ARGS[@]}"
