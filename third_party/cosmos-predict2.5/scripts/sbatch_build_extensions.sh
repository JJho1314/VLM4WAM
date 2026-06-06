#!/usr/bin/env bash
#SBATCH --job-name=cosmos-build-te
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-build-%j.out
#SBATCH --error=/data/user/jhe724/workspace/cosmos-predict2.5/slurm-build-%j.err

# Build the transformer-engine PyTorch extension. The bare `transformer-engine`
# wheel only ships the standalone CUDA library (transformer_engine.common); the
# torch bindings (transformer_engine_torch) need to be compiled against this
# venv's torch, gcc 11.5 and CUDA 12.6.

set -uo pipefail
cd /data/user/jhe724/workspace/cosmos-predict2.5

module load gcc/11.5 cuda/12.6 nccl/2.25 2>/dev/null || true

VENV=/data/user/jhe724/workspace/cosmos-predict2.5/.venv
export VIRTUAL_ENV=$VENV
export PATH=/data/apps/gcc/11.5/bin:$VENV/bin:$PATH
unset PYTHONHOME

export CC=/data/apps/gcc/11.5/bin/gcc
export CXX=/data/apps/gcc/11.5/bin/g++
export CUDA_HOME=/data/apps/cuda/12.6
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
export MAX_JOBS=12
export NVCC_THREADS=2

# transformer-engine-torch needs cudnn.h; the cudnn pip wheel ships only libs
# unless we point the compiler at its include dir.
NV=$VENV/lib/python3.10/site-packages/nvidia
export CPATH=$NV/cudnn/include:${CPATH:-}
export CPLUS_INCLUDE_PATH=$NV/cudnn/include:${CPLUS_INCLUDE_PATH:-}
export LIBRARY_PATH=$NV/cudnn/lib:${LIBRARY_PATH:-}
export CUDNN_HOME=$NV/cudnn
export CUDNN_PATH=$NV/cudnn

export PIP_INDEX_URL=http://harbor.internal.com:8081/repository/pypi-hkust/simple
export PIP_TRUSTED_HOST=harbor.internal.com

echo "=== ENV ==="
which python gcc nvcc
gcc --version | head -1
nvcc --version | tail -2

echo "=== TE[pytorch] ==="
T0=$(date +%s)
python -m pip install --no-build-isolation --no-cache-dir "transformer-engine[pytorch]==2.2.0" 2>&1 | tail -25
echo "exit=$?  elapsed=$(( $(date +%s) - T0 ))s"

echo "=== VERIFY ==="
NV_LIB=$VENV/lib/python3.10/site-packages/nvidia
LD_LIBRARY_PATH="$NV_LIB/cudnn/lib:$NV_LIB/cuda_runtime/lib:$NV_LIB/cuda_nvrtc/lib:$NV_LIB/cublas/lib:$NV_LIB/cusparse/lib:$NV_LIB/cusolver/lib:$NV_LIB/cufft/lib:$NV_LIB/curand/lib:$NV_LIB/nccl/lib:$NV_LIB/nvjitlink/lib:${LD_LIBRARY_PATH:-}" \
  python -c "import transformer_engine.pytorch as tep; print('te.pytorch ok')"
echo "DONE"
