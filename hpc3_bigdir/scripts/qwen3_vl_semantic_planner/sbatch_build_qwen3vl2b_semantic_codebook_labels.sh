#!/usr/bin/env bash
# Build VQ semantic-token labels from Qwen3-VL-2B continuous future-frame features.

#SBATCH --job-name=q3vl2b-vq-label
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl2b-vq-label-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl2b-vq-label-%j.err

set -uo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
cd "$VLM4WAM_ROOT" || exit 2
mkdir -p logs

module load gcc/11.5 cuda/12.8 nccl/2.25 2>/dev/null || true
source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
CONDA_ENV=${CONDA_ENV:-starVLA}
conda activate "$CONDA_ENV" 2>/dev/null || true
PY=${PY:-/data/user/jhe724/.conda/envs/starVLA/bin/python}

DATASET_ROOT=${DATASET_ROOT:-/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half}
CONTINUOUS_LABEL_DIR=${CONTINUOUS_LABEL_DIR:-$DATASET_ROOT/qwen3vl2b_semantic_plan_k6_g9_stagea4096}
OUTPUT_DIR=${OUTPUT_DIR:-$DATASET_ROOT/qwen3vl2b_semantic_codes_k6_g9_c1024_stagea4096}
MAX_SAMPLES=${MAX_SAMPLES:-4096}
CODEBOOK_SIZE=${CODEBOOK_SIZE:-1024}
MAX_TRAIN_TOKENS=${MAX_TRAIN_TOKENS:-262144}
KMEANS_ITERS=${KMEANS_ITERS:-30}
CHUNK_SIZE=${CHUNK_SIZE:-65536}

if [[ ! -x "$PY" ]]; then
  echo "ERROR: python executable not found: $PY" >&2
  exit 2
fi
if [[ ! -d "$CONTINUOUS_LABEL_DIR" ]]; then
  echo "ERROR: continuous label dir not found: $CONTINUOUS_LABEL_DIR" >&2
  exit 2
fi

echo "continuous_labels=$CONTINUOUS_LABEL_DIR"
echo "output=$OUTPUT_DIR"
echo "max_samples=$MAX_SAMPLES codebook_size=$CODEBOOK_SIZE max_train_tokens=$MAX_TRAIN_TOKENS"
nvidia-smi -L || true

"$PY" scripts/qwen3_vl_semantic_planner/build_semantic_codebook_labels.py \
  --continuous-label-dir "$CONTINUOUS_LABEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --max-samples "$MAX_SAMPLES" \
  --codebook-size "$CODEBOOK_SIZE" \
  --max-train-tokens "$MAX_TRAIN_TOKENS" \
  --kmeans-iters "$KMEANS_ITERS" \
  --chunk-size "$CHUNK_SIZE" \
  --overwrite
