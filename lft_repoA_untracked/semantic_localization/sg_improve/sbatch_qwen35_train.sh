#!/usr/bin/env bash
#SBATCH --job-name=q35-discrete-plan
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/q35-train-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/q35-train-%j.err
set -uo pipefail
ROOT=/data/user/jhe724/workspace/VLM4WAM
cd "$ROOT"; mkdir -p logs
module load gcc/11.5 cuda/12.8 nccl/2.25 2>/dev/null || true
source /share/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
PY=/data/user/jhe724/.conda/envs/qwen35/bin/python   # transformers 5.x for Qwen3_5

export QWEN_DIR=/data/user/jhe724/workspace/weights/Qwen3.5-2B
export DATA_DIR=${DATA_DIR:-$ROOT/data/qwen35_train}   # BUGFIX: was hard-coded, silently ignored --export DATA_DIR
export OUT_DIR=${OUT_DIR:-$ROOT/outputs/qwen35_discrete_plan}
export NUM_CODES=${NUM_CODES:-2048}
export MAX_STEPS=${MAX_STEPS:-6000}
export BATCH_SIZE=${BATCH_SIZE:-8}
export LR=${LR:-3e-4}
export SAVE_STEPS=${SAVE_STEPS:-1000}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
echo "torchrun nproc=$NPROC steps=$MAX_STEPS bs=$BATCH_SIZE codes=$NUM_CODES -> $OUT_DIR"
exec "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29517}" \
  semantic_localization/sg_improve/sg_qwen35_train.py
