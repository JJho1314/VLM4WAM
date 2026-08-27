#!/usr/bin/env bash
# Train a lightweight dense-feature -> target-mask probe.

#SBATCH --job-name=dense-probe
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-dense-probe-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-dense-probe-%j.err

set -euo pipefail

VLM4WAM_ROOT=${VLM4WAM_ROOT:-/data/user/jhe724/workspace/VLM4WAM}
REPO_ROOT=${REPO_ROOT:-$VLM4WAM_ROOT/third_party/cosmos-predict2.5}
DATASET_DIR=${DATASET_DIR:-/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half}
OUTPUT_DIR=${OUTPUT_DIR:-$VLM4WAM_ROOT/feature_guidance_analysis/dense_feature_mask_probe_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$VLM4WAM_ROOT/logs" "$OUTPUT_DIR"
cd "$REPO_ROOT"

module load gcc/11.5 cuda/12.6 2>/dev/null || true

VENV=${COSMOS_VENV:-/data/user/jhe724/workspace/cosmos-predict2.5/.venv}
export VIRTUAL_ENV=$VENV
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/packages/cosmos-oss:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

{
  date
  hostname
  echo "repo=$REPO_ROOT"
  echo "dataset=$DATASET_DIR"
  echo "output=$OUTPUT_DIR"
  nvidia-smi -L || true
} | tee "$OUTPUT_DIR/run_info.log"

python scripts/train_dense_feature_mask_probe.py \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --max-train "${MAX_TRAIN:-800}" \
  --max-val "${MAX_VAL:-200}" \
  --epochs "${EPOCHS:-30}" \
  --batch-samples "${BATCH_SAMPLES:-16}" \
  --lr "${LR:-0.003}" \
  --weight-decay "${WEIGHT_DECAY:-0.0001}" \
  --hidden-dim "${HIDDEN_DIM:-0}" \
  --normalize-features \
  --num-visuals "${NUM_VISUALS:-8}" \
  2>&1 | tee "$OUTPUT_DIR/train.log"

echo "output=$OUTPUT_DIR"
