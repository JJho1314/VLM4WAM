#!/usr/bin/env bash
#SBATCH --job-name=cosmos-sanity
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-%j.out
#SBATCH --error=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-%j.err

# Cosmos 2B sanity: load model + run a few mock training steps to confirm the
# environment, checkpoint mirror, and data pipeline are wired up correctly.

set -uo pipefail

cd /data/user/jhe724/workspace/cosmos-predict2.5

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

# Triton JIT-compiles launcher stubs; default `gcc` resolves to user's broken
# /data/user/jhe724/bin/gcc which is missing cc1plus. Force the gcc/11.5 toolchain.
export CC=/data/apps/gcc/11.5/bin/gcc
export CXX=/data/apps/gcc/11.5/bin/g++

# `source .venv/bin/activate` silently fails inside slurm's batch shell on this
# cluster (leaves PATH pointing at /share/anaconda3). Set the env vars directly.
VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
export VIRTUAL_ENV=$VENV
# Put venv first AND gcc/11.5 ahead of user's broken /data/user/jhe724/bin
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME

# transformer_engine.so dlopens libcudnn_adv.so.9 / libnccl.so.2 etc. from the
# nvidia-*-cu12 wheels installed in the venv. Without these on LD_LIBRARY_PATH
# the dlopen fails with "cannot open shared object file".
NV_LIB=$VENV/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

export COSMOS_CHECKPOINTS_DIR=/data/user/jhe724/workspace/weights
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_INIT_TIMEOUT=10
export IMAGINAIRE_OUTPUT_ROOT=/data/user/jhe724/workspace/cosmos-predict2.5/outputs/sanity
export TOKENIZERS_PARALLELISM=false
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"

nvidia-smi || true
echo "HOST_GLIBC: $(ldd --version 2>&1 | head -1)"
echo "VENV_PY: $(which python)"
python --version
python -c "import torch, transformer_engine, flash_attn, natten; print('torch:', torch.__version__, 'te:', transformer_engine.__version__, 'fa:', flash_attn.__version__, 'na:', natten.__version__)"
python -c "import torch; print('cuda available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

echo "=== TRAIN ==="
torchrun --nproc_per_node=1 --master_port=12345 -m scripts.train \
  --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
  -- experiment=predict2_video2world_training_2b_robointer_droid_sanity 2>&1
echo "train_exit=$?"
