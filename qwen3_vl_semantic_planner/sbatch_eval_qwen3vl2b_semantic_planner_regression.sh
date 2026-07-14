#!/usr/bin/env bash
# Evaluate Baton-style continuous semantic planner checkpoint.

#SBATCH --job-name=q3vl2b-baton-eval
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl2b-baton-eval-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl2b-baton-eval-%j.err

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
PLAN_LABEL_DIR=${PLAN_LABEL_DIR:-$DATASET_ROOT/qwen3vl2b_semantic_plan_k6_g9_full}
RUN_DIR=${RUN_DIR:-/data/user/jhe724/workspace/VLM4WAM/outputs/qwen3vl_semantic_planner/qwen3vl2b_baton_continuous_k6_g9_full_8gpu_b4_acc2_gbs64_3000step}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$RUN_DIR/step_003000}
OUTPUT_JSON=${OUTPUT_JSON:-$RUN_DIR/eval_step003000_regression_n128.json}
MAX_SAMPLES=${MAX_SAMPLES:-128}
BATCH_SIZE=${BATCH_SIZE:-1}
DTYPE=${DTYPE:-bf16}

if [[ ! -x "$PY" ]]; then
  echo "ERROR: python executable not found: $PY" >&2
  exit 2
fi
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "ERROR: checkpoint dir not found: $CHECKPOINT_DIR" >&2
  exit 2
fi

echo "checkpoint=$CHECKPOINT_DIR"
echo "labels=$PLAN_LABEL_DIR"
echo "output_json=$OUTPUT_JSON"
echo "max_samples=$MAX_SAMPLES batch_size=$BATCH_SIZE"
nvidia-smi -L || true

"$PY" qwen3_vl_semantic_planner/evaluate_qwen3vl_semantic_planner_regression.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --dataset-root "$DATASET_ROOT" \
  --plan-label-dir "$PLAN_LABEL_DIR" \
  --output-json "$OUTPUT_JSON" \
  --max-samples "$MAX_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --dtype "$DTYPE"
