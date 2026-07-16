#!/usr/bin/env bash
#SBATCH --job-name=geact_ltx_sig2
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/slurm-geact-ltx-siglip2-%j.out

set -euo pipefail

GE_ACT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-/data/user/jhe724/.conda/envs/genie_envisioner}"
CONFIG="${CONFIG:-$GE_ACT_ROOT/configs/ltx_model/libero/video_model_libero_fastwam_siglip2.yaml}"

export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NPROC_PER_NODE=8

mkdir -p "$GE_ACT_ROOT/logs"
cd "$GE_ACT_ROOT"
bash scripts/train_ltx_siglip2.sh "$CONFIG"
