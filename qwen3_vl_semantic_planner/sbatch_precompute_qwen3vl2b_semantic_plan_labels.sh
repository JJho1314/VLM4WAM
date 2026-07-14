#!/usr/bin/env bash
# Precompute Qwen3-VL-2B continuous semantic plan labels for DROID videos.
# This uses a plain Qwen3-VL-2B-Instruct checkpoint.

#SBATCH --job-name=q3vl2b-plan-label
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl2b-plan-label-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl2b-plan-label-%j.err

set -uo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
cd "$VLM4WAM_ROOT" || exit 2
mkdir -p logs

module load gcc/11.5 cuda/12.8 nccl/2.25 2>/dev/null || true
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
CONDA_ENV=${CONDA_ENV:-starVLA}
conda activate "$CONDA_ENV" 2>/dev/null || true
PY=${PY:-/data/user/jhe724/.conda/envs/starVLA/bin/python}

MODEL_PATH=${MODEL_PATH:-/data/user/jhe724/workspace/weights/Qwen3-VL-2B-Instruct}
DATASET_ROOT=${DATASET_ROOT:-/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half}
OUTPUT_DIR=${OUTPUT_DIR:-$DATASET_ROOT/qwen3vl2b_semantic_plan_k6_g9}
MAX_SAMPLES=${MAX_SAMPLES:-512}
START_INDEX=${START_INDEX:-0}
NUM_KEYFRAMES=${NUM_KEYFRAMES:-6}
GRID_SIZE=${GRID_SIZE:-9}
DTYPE=${DTYPE:-bf16}

if [[ ! -x "$PY" ]]; then
  echo "ERROR: python executable not found: $PY" >&2
  exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: Qwen3-VL-2B model path not found: $MODEL_PATH" >&2
  exit 2
fi

echo "model=$MODEL_PATH"
echo "dataset=$DATASET_ROOT"
echo "output=$OUTPUT_DIR"
echo "max_samples=$MAX_SAMPLES start_index=$START_INDEX keyframes=$NUM_KEYFRAMES grid=$GRID_SIZE"
nvidia-smi -L || true

"$PY" qwen3_vl_semantic_planner/build_qwen3vl_semantic_plan_labels.py \
  --dataset-root "$DATASET_ROOT" \
  --model-path "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --max-samples "$MAX_SAMPLES" \
  --start-index "$START_INDEX" \
  --num-keyframes "$NUM_KEYFRAMES" \
  --grid-size "$GRID_SIZE" \
  --dtype "$DTYPE"
