#!/usr/bin/env bash
# Evaluate a saved Qwen3-VL Stage-A CE semantic planner checkpoint.

#SBATCH --job-name=q3vl2b-stagea-eval
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl2b-stagea-eval-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl2b-stagea-eval-%j.err

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
DISCRETE_LABEL_DIR=${DISCRETE_LABEL_DIR:-$DATASET_ROOT/qwen3vl2b_semantic_codes_k6_g9_c1024_full}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/data/user/jhe724/workspace/VLM4WAM/outputs/qwen3vl_semantic_planner/qwen3vl2b_stagea_ce_c1024_full_8gpu_b4_acc2_gbs64_3000step/step_003000}
OUTPUT_JSON=${OUTPUT_JSON:-/data/user/jhe724/workspace/VLM4WAM/outputs/qwen3vl_semantic_planner/qwen3vl2b_stagea_ce_c1024_full_8gpu_b4_acc2_gbs64_3000step/eval_step003000_n32.json}
NUM_SAMPLES=${NUM_SAMPLES:-32}
NUM_KEYFRAMES=${NUM_KEYFRAMES:-6}
GRID_SIZE=${GRID_SIZE:-9}
DTYPE=${DTYPE:-bf16}
TEACHER_FORCED=${TEACHER_FORCED:-1}
CONSTRAIN_SEMANTIC_VOCAB=${CONSTRAIN_SEMANTIC_VOCAB:-0}
SEMANTIC_TOKEN_SEPARATOR=${SEMANTIC_TOKEN_SEPARATOR:-auto}

if [[ ! -x "$PY" ]]; then
  echo "ERROR: python executable not found: $PY" >&2
  exit 2
fi
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "ERROR: checkpoint dir not found: $CHECKPOINT_DIR" >&2
  exit 2
fi
if [[ ! -d "$DISCRETE_LABEL_DIR" ]]; then
  echo "ERROR: discrete label dir not found: $DISCRETE_LABEL_DIR" >&2
  exit 2
fi

echo "model=$MODEL_PATH"
echo "checkpoint=$CHECKPOINT_DIR"
echo "labels=$DISCRETE_LABEL_DIR"
echo "output=$OUTPUT_JSON"
echo "num_samples=$NUM_SAMPLES teacher_forced=$TEACHER_FORCED constrain_semantic_vocab=$CONSTRAIN_SEMANTIC_VOCAB semantic_token_separator=$SEMANTIC_TOKEN_SEPARATOR"
nvidia-smi -L || true

ARGS=(
  --model-path "$MODEL_PATH"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --dataset-root "$DATASET_ROOT"
  --discrete-label-dir "$DISCRETE_LABEL_DIR"
  --output-json "$OUTPUT_JSON"
  --num-samples "$NUM_SAMPLES"
  --num-keyframes "$NUM_KEYFRAMES"
  --grid-size "$GRID_SIZE"
  --dtype "$DTYPE"
  --semantic-token-separator "$SEMANTIC_TOKEN_SEPARATOR"
)

if [[ "$TEACHER_FORCED" == "1" ]]; then
  ARGS+=(--teacher-forced)
fi
if [[ "$CONSTRAIN_SEMANTIC_VOCAB" == "1" ]]; then
  ARGS+=(--constrain-semantic-vocab)
fi

"$PY" scripts/qwen3_vl_semantic_planner/evaluate_qwen3vl_semantic_planner_ce.py "${ARGS[@]}"
