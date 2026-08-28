#!/usr/bin/env bash
#SBATCH --job-name=probe-c
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-probe-c-%j.out
#SBATCH --error=/data/user/jhe724/workspace/VLM4WAM/logs/slurm-probe-c-%j.err
set -uo pipefail
REPO_ROOT=/data/user/jhe724/workspace/VLM4WAM/third_party/cosmos-predict2.5
cd "$REPO_ROOT"
module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true
VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
export VIRTUAL_ENV=$VENV
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
NV_LIB=$VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export COSMOS_CHECKPOINTS_DIR=/data/user/jhe724/workspace/weights
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export COSMOS_SKIP_CUDA_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DROID_SUCCESS_V21_TAVID_DIR=/data/user/jhe724/workspace/datasets/droid_v21_iou50_taskdiverse_half
export DROID_SUCCESS_V21_TAVID_VAL_DIR=$DROID_SUCCESS_V21_TAVID_DIR
export PROBE_N=${PROBE_N:-24}
python /data/user/jhe724/workspace/VLM4WAM/mg_eval/probe_c_depth.py
echo "probe_c_exit=$?"
