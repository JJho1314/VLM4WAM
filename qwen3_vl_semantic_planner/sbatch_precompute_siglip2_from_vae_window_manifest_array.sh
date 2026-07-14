#!/usr/bin/env bash
# Precompute SigLIP2 semantic tokens from VAE cache window manifests.

#SBATCH --job-name=siglip2-from-win
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --array=0-15%8
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-siglip2-from-win-%A_%a.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-siglip2-from-win-%A_%a.err

set -euo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
cd "$VLM4WAM_ROOT"
mkdir -p logs

module load gcc/11.5 cuda/12.8 nccl/2.25 2>/dev/null || true
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
CONDA_ENV=${CONDA_ENV:-starVLA}
conda activate "$CONDA_ENV" 2>/dev/null || true
PY=${PY:-/data/user/jhe724/.conda/envs/starVLA/bin/python}

DATASET_ROOT=${DATASET_ROOT:-/data/user/jhe724/workspace/datasets/droid_success_v21_target_aware_left_right_480x864_train_strict_holdout_v3_10hz_320x576}
NUM_FRAMES=${NUM_FRAMES:-93}
FRAME_STRIDES=${FRAME_STRIDES:-1,2,3}
CACHES_PER_RANGE=${CACHES_PER_RANGE:-4}
VAE_WINDOW_MANIFEST_DIR=${VAE_WINDOW_MANIFEST_DIR:-$DATASET_ROOT/vae_range_latents_wan2pt1_t${NUM_FRAMES}_s${FRAME_STRIDES//,/}_c${NUM_FRAMES}_r${CACHES_PER_RANGE}_full}
MODEL_PATH=${MODEL_PATH:-/data/user/jhe724/workspace/weights/siglip2-so400m-patch14-384}
NUM_KEYFRAMES=${NUM_KEYFRAMES:-16}
GRID_SIZE=${GRID_SIZE:-9}
DTYPE=${DTYPE:-bf16}
OVERWRITE=${OVERWRITE:-1}
if [[ "$GRID_SIZE" -le 0 ]]; then
  GRID_TAG="gnative"
else
  GRID_TAG="g${GRID_SIZE}"
fi
OUTPUT_DIR=${OUTPUT_DIR:-$DATASET_ROOT/siglip2_semantic_plan_k${NUM_KEYFRAMES}_${GRID_TAG}_from_vae_window_t${NUM_FRAMES}_s${FRAME_STRIDES//,/}_r${CACHES_PER_RANGE}_full}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
WINDOW_MANIFEST=${WINDOW_MANIFEST:-$VAE_WINDOW_MANIFEST_DIR/window_manifest_${TASK_ID}.jsonl}

if [[ ! -x "$PY" ]]; then
  echo "ERROR: python executable not found: $PY" >&2
  exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: SigLIP2 model path not found: $MODEL_PATH" >&2
  exit 2
fi
if [[ ! -f "$WINDOW_MANIFEST" ]]; then
  echo "ERROR: window manifest not found: $WINDOW_MANIFEST" >&2
  exit 2
fi

echo "mode=SigLIP2 semantic token from VAE window manifest"
echo "task=$TASK_ID"
echo "dataset=$DATASET_ROOT"
echo "window_manifest=$WINDOW_MANIFEST"
echo "output=$OUTPUT_DIR"
echo "model=$MODEL_PATH keyframes=$NUM_KEYFRAMES grid=$GRID_SIZE dtype=$DTYPE overwrite=$OVERWRITE"

ARGS=(
  --dataset-root "$DATASET_ROOT"
  --model-path "$MODEL_PATH"
  --output-dir "$OUTPUT_DIR"
  --window-manifest "$WINDOW_MANIFEST"
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

exec "$PY" qwen3_vl_semantic_planner/build_siglip2_semantic_plan_labels.py "${ARGS[@]}"
