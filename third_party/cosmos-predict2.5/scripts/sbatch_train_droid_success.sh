#!/usr/bin/env bash
#SBATCH --job-name=cosmos-droid
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=48:00:00
#SBATCH --output=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-%j.out
#SBATCH --error=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-%j.err

# Cosmos 2B post-training on the droid_success dataset (lerobot v3.0 release,
# 1280x720 @ 15 fps). Trains across 8 H100s on a single node with FSDP via
# torchrun.

set -uo pipefail
cd /data/user/jhe724/workspace/cosmos-predict2.5

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
export VIRTUAL_ENV=$VENV
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME

# Triton JIT compiles need real gcc (user's ~/bin/gcc is broken).
export CC=/data/apps/gcc/11.5/bin/gcc
export CXX=/data/apps/gcc/11.5/bin/g++

# transformer_engine.so dlopens libcudnn_adv.so.9 etc. from the nvidia-*-cu12
# wheels installed in the venv.
NV_LIB=$VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

export COSMOS_CHECKPOINTS_DIR=/data/user/jhe724/workspace/weights
export HF_HUB_OFFLINE=1

# Internal wandb instance hosted on the cluster (no external internet).
export WANDB_MODE=online
export WANDB_BASE_URL="http://10.12.1.245:8080"
export WANDB_API_KEY="local-37151658708fac20809135dce9e234842db32f97"
export IMAGINAIRE_OUTPUT_ROOT=/data/user/jhe724/workspace/cosmos-predict2.5/outputs/droid_success
export TOKENIZERS_PARALLELISM=false

# NCCL: keep default settings. If multi-node later, add NCCL_SOCKET_IFNAME etc.
export NCCL_DEBUG=WARN

mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"

nvidia-smi -L
echo "HOST_GLIBC: $(ldd --version 2>&1 | head -1)"
echo "VENV_PY: $(which python)"
python -c "import torch; print('cuda count:', torch.cuda.device_count())"

echo "=== TRAIN (8 GPUs, FSDP) ==="
torchrun --standalone --nproc_per_node=8 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
  -- experiment=predict2_video2world_training_2b_droid_success
echo "train_exit=$?"
