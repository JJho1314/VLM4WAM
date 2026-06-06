#!/usr/bin/env bash
#SBATCH --job-name=cosmos-convert
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=00:30:00
#SBATCH --output=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-%j.out
#SBATCH --error=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-%j.err

# Convert DCP checkpoint at $CKPT_DIR (sub-dir `model/`) to consolidated .pt:
#   model.pt, model_ema_fp32.pt, model_ema_bf16.pt
# Pass CKPT_DIR via --export.
# Usage: sbatch --export=CKPT_DIR=/path/to/iter_000010000 scripts/sbatch_convert_ckpt.sh

set -uo pipefail
cd /data/user/jhe724/workspace/cosmos-predict2.5

module load gcc/11.5 cuda/12.6 2>/dev/null || true

VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
NV_LIB=$VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

echo "Converting: $CKPT_DIR"
$VENV/bin/python scripts/convert_distcp_to_pt.py "$CKPT_DIR/model" "$CKPT_DIR"
echo "exit=$?"
ls -lh "$CKPT_DIR"/*.pt 2>&1
