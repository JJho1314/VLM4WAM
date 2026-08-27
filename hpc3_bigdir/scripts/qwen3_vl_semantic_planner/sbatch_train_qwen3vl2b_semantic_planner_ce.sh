#!/usr/bin/env bash
# Stage-A LoRA fine-tune Qwen3-VL-2B-Instruct with CE over VQ semantic tokens.

#SBATCH --job-name=q3vl2b-stagea-ce
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl2b-stagea-ce-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-q3vl2b-stagea-ce-%j.err

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
DISCRETE_LABEL_DIR=${DISCRETE_LABEL_DIR:-$DATASET_ROOT/qwen3vl2b_semantic_codes_k6_g9_c1024_stagea4096}
OUTPUT_DIR=${OUTPUT_DIR:-/data/user/jhe724/workspace/VLM4WAM/outputs/qwen3vl_semantic_planner/qwen3vl2b_stagea_ce_c1024_4096_3000step}
MAX_SAMPLES=${MAX_SAMPLES:-4096}
MAX_STEPS=${MAX_STEPS:-3000}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
NUM_GPUS=${NUM_GPUS:-1}
NUM_KEYFRAMES=${NUM_KEYFRAMES:-6}
GRID_SIZE=${GRID_SIZE:-9}
CODEBOOK_SIZE=${CODEBOOK_SIZE:-1024}
LR=${LR:-2e-5}
TOKEN_LR=${TOKEN_LR:-5e-5}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
DTYPE=${DTYPE:-bf16}
SAVE_STEPS=${SAVE_STEPS:-500}
SEMANTIC_TOKEN_SEPARATOR=${SEMANTIC_TOKEN_SEPARATOR:-none}

if [[ ! -x "$PY" ]]; then
  echo "ERROR: python executable not found: $PY" >&2
  exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: Qwen3-VL-2B model path not found: $MODEL_PATH" >&2
  exit 2
fi
if [[ ! -d "$DISCRETE_LABEL_DIR" ]]; then
  echo "ERROR: discrete semantic label dir not found: $DISCRETE_LABEL_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
echo "model=$MODEL_PATH"
echo "dataset=$DATASET_ROOT"
echo "labels=$DISCRETE_LABEL_DIR"
echo "output=$OUTPUT_DIR"
echo "max_samples=$MAX_SAMPLES max_steps=$MAX_STEPS batch_per_gpu=$BATCH_SIZE accum=$GRAD_ACCUM num_gpus=$NUM_GPUS"
echo "codebook_size=$CODEBOOK_SIZE keyframes=$NUM_KEYFRAMES grid=$GRID_SIZE semantic_token_separator=$SEMANTIC_TOKEN_SEPARATOR"
nvidia-smi -L || true

TRAIN_ARGS=(
  --model-path "$MODEL_PATH" \
  --dataset-root "$DATASET_ROOT" \
  --discrete-label-dir "$DISCRETE_LABEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --max-samples "$MAX_SAMPLES" \
  --max-steps "$MAX_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --num-keyframes "$NUM_KEYFRAMES" \
  --grid-size "$GRID_SIZE" \
  --codebook-size "$CODEBOOK_SIZE" \
  --lr "$LR" \
  --token-lr "$TOKEN_LR" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --dtype "$DTYPE" \
  --save-steps "$SAVE_STEPS" \
  --semantic-token-separator "$SEMANTIC_TOKEN_SEPARATOR"
)

if [[ "$NUM_GPUS" -gt 1 ]]; then
  "$PY" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NUM_GPUS" \
    scripts/qwen3_vl_semantic_planner/train_qwen3vl_semantic_planner_ce.py \
    "${TRAIN_ARGS[@]}"
else
  "$PY" scripts/qwen3_vl_semantic_planner/train_qwen3vl_semantic_planner_ce.py \
    "${TRAIN_ARGS[@]}"
fi
