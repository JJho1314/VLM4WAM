#!/usr/bin/env bash
#SBATCH --job-name=q35-precompute
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/q35-precompute-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/q35-precompute-%j.err
set -uo pipefail
ROOT=/data/user/jhe724/workspace/VLM4WAM
cd "$ROOT"; mkdir -p logs data
module load gcc/11.5 cuda/12.8 nccl/2.25 2>/dev/null || true
source /share/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
PY=/data/user/jhe724/.conda/envs/starVLA/bin/python   # SigLIP2 label builder needs only transformers 4.57
export LIBERO_ROOT=/data/user/jhe724/workspace/datasets/LIBERO-fastwam
export SIGLIP2_DIR=/data/user/jhe724/workspace/weights/siglip2-large-patch16-256
export OUT_DIR=${OUT_DIR:-$ROOT/data/qwen35_train_mw}   # multi-window set (don't clobber the single-window dir)
export SUITES=${SUITES:-object,spatial,goal,10}
export PER_SUITE=${PER_SUITE:-0}
export WINDOW_STRIDE=${WINDOW_STRIDE:-10}
export MAX_WINDOWS_PER_EP=${MAX_WINDOWS_PER_EP:-16}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "precompute -> $OUT_DIR"
exec "$PY" semantic_localization/sg_improve/sg_qwen35_precompute.py
